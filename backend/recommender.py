"""CBF + RL inference engine + nearest-neighbor TSP scheduler.

Loads the pickled artifacts produced by the training notebook and serves
itineraries that match the response shape in
RecSys-Output/data/processed/sample_api_response.json.

Defensive design: if any pickle is missing or has an unexpected schema
(scikit-learn version drift, training format change), we fall back to a
rating-based score so the API stays available.
"""
from __future__ import annotations

import ast
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Repository layout: backend/recommender.py — models live one level up at /models/
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
DATA_PATH = BACKEND_DIR / "data" / "destinations.csv"
LAST_UPDATED_PATH = BACKEND_DIR / "data" / "last_updated.txt"

CITY_TRAVEL_SPEED_KMH = 28.0  # Bandung urban average for back-of-envelope ETA


# ── Helpers ──────────────────────────────────────────────────────────────
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lng) points in km."""
    R = 6371.0
    lat1, lng1 = a
    lat2, lng2 = b
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    s1 = math.sin(dlat / 2) ** 2
    s2 = math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return 2 * R * math.asin(math.sqrt(s1 + s2))


def _safe_load_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        log.warning("Pickle not found: %s", path)
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load %s (%s) — falling back.", path.name, exc)
        return None


def _parse_tags(raw: Any) -> List[str]:
    """destinations.csv stores tags as a stringified Python list."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(t) for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


# ── Recommender ──────────────────────────────────────────────────────────
class Recommender:
    """Bundles destinations + ML artifacts + scheduling logic.

    Public surface used by main.py:
        .last_updated      → str
        .n_destinations    → int
        .model_loaded      → bool
        .build_itinerary() → dict matching the API contract
    """

    def __init__(self) -> None:
        self.df = self._load_destinations()
        self.n_destinations = len(self.df)
        self.last_updated = self._load_last_updated()

        self.cbf_model = _safe_load_pickle(MODELS_DIR / "cbf_model.pkl")
        self.rl_agent = _safe_load_pickle(MODELS_DIR / "rl_agent.pkl")
        self.label_encoders = _safe_load_pickle(MODELS_DIR / "label_encoders.pkl")
        self.scaler = _safe_load_pickle(MODELS_DIR / "scaler.pkl")

        self.q_table = self._extract_q_table(self.rl_agent)
        self.model_loaded = any(
            x is not None for x in (self.cbf_model, self.rl_agent, self.label_encoders, self.scaler)
        )
        log.info(
            "Recommender ready — %d destinations, model_loaded=%s, q_table_size=%d",
            self.n_destinations,
            self.model_loaded,
            len(self.q_table),
        )

    # ── Loading ──────────────────────────────────────────────────────────
    def _load_destinations(self) -> pd.DataFrame:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"destinations.csv not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        required = {"id", "name", "category", "lat", "lng", "ticket", "duration", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"destinations.csv missing columns: {missing}")
        # Normalize: ensure no NaN coords or essential numerics
        df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
        df["ticket"] = df["ticket"].fillna(0).astype(int)
        df["duration"] = df["duration"].fillna(60).astype(int)
        df["rating"] = df["rating"].fillna(4.0).astype(float)
        df["category"] = df["category"].fillna("Wisata").astype(str)
        df["desc"] = df.get("desc", "").fillna("").astype(str)
        if "tags" not in df.columns:
            df["tags"] = ""
        if "gmaps_url" not in df.columns:
            df["gmaps_url"] = df["name"].apply(
                lambda n: f"https://www.google.com/maps/search/?api=1&query={n.replace(' ', '%20')}%2C%20Bandung"
            )
        if "stay_detail" not in df.columns:
            df["stay_detail"] = ""
        return df

    def _load_last_updated(self) -> str:
        if LAST_UPDATED_PATH.exists():
            return LAST_UPDATED_PATH.read_text().strip()
        return "unknown"

    @staticmethod
    def _extract_q_table(rl_agent: Any) -> Dict[str, float]:
        """Flatten the trained Q-table to a {dest_id: score} lookup.

        Training stores the agent as a dict with key 'q_table' which itself maps
        (state, action) → q_value. We collapse to per-action max so a single
        scalar represents "how good is it to visit this destination overall".
        """
        if not isinstance(rl_agent, dict):
            return {}
        q = rl_agent.get("q_table", {})
        if not isinstance(q, dict):
            return {}

        scores: Dict[str, float] = {}
        for key, val in q.items():
            # action could be the destination id directly, or part of (state, action) tuple
            if isinstance(key, tuple) and len(key) >= 2:
                action = key[-1]
            else:
                action = key
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            action_id = str(action)
            if action_id not in scores or fval > scores[action_id]:
                scores[action_id] = fval
        return scores

    # ── Scoring ──────────────────────────────────────────────────────────
    def _score_candidates(
        self,
        pool: pd.DataFrame,
        categories: List[str],
        home: Tuple[float, float],
    ) -> pd.DataFrame:
        """Combine CBF-style category match, RL Q-value, rating, and proximity."""
        df = pool.copy()
        # Distance from home (closer is generally better; capped)
        df["_dist_home"] = df.apply(lambda r: haversine_km(home, (r["lat"], r["lng"])), axis=1)

        # Category match score (CBF stand-in: exact category match)
        if categories:
            df["_cat_score"] = df["category"].isin(categories).astype(float)
        else:
            df["_cat_score"] = 1.0

        # Normalize rating to [0, 1] roughly
        df["_rating_score"] = (df["rating"] - 3.5).clip(lower=0) / 2.0

        # RL bonus from learned Q-values (zero if not in q_table)
        if self.q_table:
            qvals = df["id"].astype(str).map(self.q_table).fillna(0.0)
            qmax = qvals.max() or 1.0
            df["_rl_score"] = qvals / qmax
        else:
            df["_rl_score"] = 0.0

        # Distance penalty (further from home → small malus)
        df["_dist_penalty"] = (df["_dist_home"] / 50.0).clip(upper=1.0)

        # Final score: tuned weights — category dominates intent, RL refines
        df["_score"] = (
            0.45 * df["_cat_score"]
            + 0.25 * df["_rl_score"]
            + 0.20 * df["_rating_score"]
            - 0.10 * df["_dist_penalty"]
            + np.random.RandomState(42).random(len(df)) * 0.02  # tiebreak jitter
        )
        return df

    # ── Itinerary builder ────────────────────────────────────────────────
    def build_itinerary(
        self,
        home: Tuple[float, float],
        count: int,
        max_km: Optional[float],
        start_min: int,
        end_min: int,
        budget: Optional[int],
        categories: List[str],
    ) -> Dict[str, Any]:
        # 1. Filter by category preference (soft — fallback to all if too narrow)
        if categories:
            pool = self.df[self.df["category"].isin(categories)].copy()
            if len(pool) < count:
                log.info("Category filter too narrow (%d), expanding to all.", len(pool))
                pool = self.df.copy()
        else:
            pool = self.df.copy()

        # 2. Score & take top candidates (overshoot for NN-TSP slack)
        scored = self._score_candidates(pool, categories, home)
        top_n = max(count * 4, 16)
        candidates = scored.nlargest(top_n, "_score").reset_index(drop=True)

        # 3. Nearest-neighbor walk from home, respecting budget + max_km
        chain: List[pd.Series] = []
        cursor: Tuple[float, float] = home
        used: set[str] = set()
        spent = 0

        for _ in range(count):
            best_idx, best_dist = -1, math.inf
            for i, row in candidates.iterrows():
                if row["id"] in used:
                    continue
                d = haversine_km(cursor, (row["lat"], row["lng"]))
                if max_km is not None and d > max_km:
                    continue
                if budget is not None and spent + int(row["ticket"]) > budget:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx == -1:
                # Relax constraints — pick the closest remaining
                for i, row in candidates.iterrows():
                    if row["id"] in used:
                        continue
                    d = haversine_km(cursor, (row["lat"], row["lng"]))
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

            if best_idx == -1:
                break

            picked = candidates.iloc[best_idx]
            used.add(picked["id"])
            chain.append((picked, best_dist))
            spent += int(picked["ticket"])
            cursor = (picked["lat"], picked["lng"])

        # 4. Schedule (arrival/departure times) and totals
        steps: List[Dict[str, Any]] = []
        t = start_min
        total_cost = 0
        total_km = 0.0
        cursor = home

        for idx, (row, dist_from_prev) in enumerate(chain, start=1):
            travel_min = round((dist_from_prev / CITY_TRAVEL_SPEED_KMH) * 60)
            arrive = t + travel_min
            depart = arrive + int(row["duration"])
            steps.append(
                {
                    "idx": idx,
                    "dest": {
                        "id": str(row["id"]),
                        "name": str(row["name"]),
                        "category": str(row["category"]),
                        "desc": str(row["desc"]) or f"Destinasi wisata {row['category']} di Bandung",
                        "ticket": int(row["ticket"]),
                        "duration": int(row["duration"]),
                        "lat": float(row["lat"]),
                        "lng": float(row["lng"]),
                        "rating": float(row["rating"]),
                        "tags": _parse_tags(row.get("tags")),
                        "gmaps_url": str(row.get("gmaps_url") or ""),
                        "stay_detail": str(row.get("stay_detail") or ""),
                    },
                    "travelMin": travel_min,
                    "travelKm": round(dist_from_prev, 2),
                    "arriveAt": arrive,
                    "departAt": depart,
                }
            )
            total_cost += int(row["ticket"])
            total_km += dist_from_prev
            t = depart
            cursor = (row["lat"], row["lng"])

        # 5. Return-home leg
        return_km = haversine_km(cursor, home) if chain else 0.0
        return_min = round((return_km / CITY_TRAVEL_SPEED_KMH) * 60)
        total_km += return_km
        arrive_home = t + return_min
        total_time = arrive_home - start_min

        return {
            "steps": steps,
            "totalCost": total_cost,
            "totalKm": round(total_km, 2),
            "totalTime": total_time,
            "returnKm": round(return_km, 2),
            "returnMin": return_min,
            "arriveHome": arrive_home,
            "overBudget": arrive_home > end_min,
            "spareMin": end_min - arrive_home,
        }
