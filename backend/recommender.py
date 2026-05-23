"""CBF + RL inference engine + nearest-neighbor TSP scheduler.

Loads the pickled artifacts produced by the training notebook and serves
itineraries that match the response shape in docs/api/sample_response.json.

CBF scoring uses the precomputed `sim_matrix` from cbf_model.pkl directly —
that matrix already fuses all 30 features (5 one-hot category, 5 numeric,
20 TF-IDF tag) into a single (N, N) cosine similarity. Re-computing TF-IDF
from `label_encoders["tfidf"]` would be both wasteful and inferior because
it'd drop the categorical and numeric channels.

Schema expected from cbf_model.pkl (dict):
    sim_matrix     : np.ndarray (N, N) — cosine similarity between destinations
    id_to_sim_idx  : Dict[str, int]    — destination id → row/col in sim_matrix
    df_index       : List[Dict]        — destination metadata indexed by row
                                         (must contain "id" key per entry)

If any of those keys are absent the scorer logs a warning and falls back to
rating + RL alone — the API stays up.
"""
from __future__ import annotations

import ast
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Repository layout: backend/recommender.py — models live one level up at /models/
import os
BACKEND_DIR = Path(__file__).resolve().parent
_repo_candidates = [
    BACKEND_DIR.parent,
    Path("/app"),
    Path("/workspace"),
]
REPO_ROOT = next(
    (p for p in _repo_candidates if (p / "models" / "cbf_model.pkl").exists()),
    BACKEND_DIR.parent,
)
MODELS_DIR = Path(os.environ.get("MODEL_DIR", str(REPO_ROOT / "models")))
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

        self.sim_matrix, self.id_to_sim_idx = self._extract_cbf(self.cbf_model)
        self.q_table = self._extract_q_table(self.rl_agent)

        self.cbf_loaded = self.sim_matrix is not None and bool(self.id_to_sim_idx)
        self.model_loaded = any(
            x is not None for x in (self.cbf_model, self.rl_agent, self.label_encoders, self.scaler)
        )

        sim_shape = self.sim_matrix.shape if self.sim_matrix is not None else None
        log.info(
            "Recommender ready — %d destinations, model_loaded=%s, "
            "cbf_loaded=%s sim_matrix=%s, q_table_size=%d",
            self.n_destinations,
            self.model_loaded,
            self.cbf_loaded,
            sim_shape,
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
    def _extract_cbf(
        cbf_model: Any,
    ) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
        """Pull `sim_matrix` and `id_to_sim_idx` out of cbf_model.pkl.

        Trained by notebooks/rec-engine.ipynb. The sim_matrix already fuses
        category + numeric + TF-IDF features, so callers should use it as-is
        and never re-vectorize the tags separately.
        """
        if not isinstance(cbf_model, dict):
            log.warning("cbf_model.pkl is not a dict — CBF scoring disabled.")
            return None, {}
        
        # SESUDAH — support kedua format key:
        sim = cbf_model.get("sim_matrix") or cbf_model.get("similarity_matrix")
        
        # id_to_sim_idx bisa berupa dict atau list of {id, name, category}
        idx_map = cbf_model.get("id_to_sim_idx") or {}
        if not idx_map:
            df_index = cbf_model.get("df_index") or []
            idx_map = {item["id"]: i for i, item in enumerate(df_index)}

        if sim is None or not isinstance(idx_map, dict) or not idx_map:
            log.warning(
                "cbf_model.pkl missing sim_matrix/id_to_sim_idx — CBF scoring disabled. "
                "Got keys: %s",
                list(cbf_model.keys()),
            )
            return None, {}

        # Coerce to ndarray once so callers can fancy-index without surprises.
        sim_arr = np.asarray(sim)
        if sim_arr.ndim != 2 or sim_arr.shape[0] != sim_arr.shape[1]:
            log.warning("sim_matrix has unexpected shape %s — CBF disabled.", sim_arr.shape)
            return None, {}

        # Normalize dict keys to str so lookups by destination id are robust.
        idx_map_norm = {str(k): int(v) for k, v in idx_map.items()}
        return sim_arr, idx_map_norm

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
    def _cbf_scores(self, pool: pd.DataFrame) -> pd.Series:
        """Mean cosine similarity of each pool destination to the rest of the pool.

        Uses the precomputed `sim_matrix` from cbf_model.pkl directly. This is
        the *correct* way to score CBF candidates: the matrix already encodes
        the full 30-dim feature space (category one-hot + numeric + TF-IDF tags).

            pool_idxs = [id_to_sim_idx[d] for d in pool.id if d in id_to_sim_idx]
            scores    = sim_matrix[pool_idxs, :][:, pool_idxs].mean(axis=0)

        Higher score → destination is more "central" within the user's preference
        cluster (similar to many other plausible picks).

        Returns a Series aligned with `pool.index` and filled with 0.0 for
        destinations that aren't in id_to_sim_idx (or when CBF is disabled).
        """
        if not self.cbf_loaded or self.sim_matrix is None:
            return pd.Series(0.0, index=pool.index)

        ids = pool["id"].astype(str).tolist()
        # Indices into sim_matrix for the destinations that are in the pool AND
        # known to the trained model. We need both the pool DataFrame index and
        # the matching matrix index to write scores back correctly.
        pool_pairs: List[Tuple[int, int]] = [
            (df_idx, self.id_to_sim_idx[did])
            for df_idx, did in zip(pool.index, ids)
            if did in self.id_to_sim_idx
        ]
        if len(pool_pairs) < 2:
            # Need at least 2 rows for a meaningful "similarity to others" score.
            return pd.Series(0.0, index=pool.index)

        df_idxs, mat_idxs = zip(*pool_pairs)
        sub = self.sim_matrix[np.ix_(mat_idxs, mat_idxs)]
        # mean across columns → average similarity from each row to all peers in pool
        mean_sim = sub.mean(axis=0)

        scores = pd.Series(0.0, index=pool.index)
        scores.loc[list(df_idxs)] = mean_sim
        return scores

    def _score_candidates(
        self,
        pool: pd.DataFrame,
        categories: List[str],
        home: Tuple[float, float],
    ) -> pd.DataFrame:
        """Combine sim_matrix CBF score, RL Q-value, rating, and proximity."""
        df = pool.copy()

        # Distance from home (further → small malus)
        df["_dist_home"] = df.apply(lambda r: haversine_km(home, (r["lat"], r["lng"])), axis=1)
        df["_dist_penalty"] = (df["_dist_home"] / 50.0).clip(upper=1.0)

        # Primary CBF signal: mean cosine similarity within the (category-filtered) pool
        cbf = self._cbf_scores(df)
        # Min-max normalize so the [0,1] weight in the final formula is meaningful.
        cbf_min, cbf_max = float(cbf.min()), float(cbf.max())
        if cbf_max > cbf_min:
            df["_cbf_score"] = (cbf - cbf_min) / (cbf_max - cbf_min)
        else:
            df["_cbf_score"] = 0.0

        # Normalize rating to [0, 1] roughly (3.5 → 0, 5.5 → 1)
        df["_rating_score"] = (df["rating"] - 3.5).clip(lower=0) / 2.0

        # RL bonus from learned Q-values (zero if not in q_table)
        if self.q_table:
            qvals = df["id"].astype(str).map(self.q_table).fillna(0.0)
            qmax = qvals.max() or 1.0
            df["_rl_score"] = qvals / qmax
        else:
            df["_rl_score"] = 0.0

        # Final score: CBF dominates content match, RL & rating refine,
        # distance is a soft penalty. Tiny jitter for deterministic tiebreak.
        df["_score"] = (
            0.50 * df["_cbf_score"]
            + 0.20 * df["_rl_score"]
            + 0.20 * df["_rating_score"]
            - 0.10 * df["_dist_penalty"]
            + np.random.RandomState(42).random(len(df)) * 0.02
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
