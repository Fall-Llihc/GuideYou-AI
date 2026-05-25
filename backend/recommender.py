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

# Repository layout: backend/recommender.py — models now inside backend/models/
BACKEND_DIR       = Path(__file__).resolve().parent
MODELS_DIR        = BACKEND_DIR / "models"
DATA_PATH         = BACKEND_DIR / "data" / "destinations.csv"
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
        """Build an itinerary that strictly respects user constraints.

        Constraint hierarchy (notebook v3 spec):
          - max_km : HARD constraint, distance from HOME (radius), never relaxed.
                     If no destinations within radius → return empty steps + notice.
          - budget : HARD per-step constraint; no relaxation.
          - categories : preference. If user selects ≥2 categories AND count ≥ N
                     categories, every category MUST appear at least once
                     (post-pick swap enforces this).

        The response includes a `notices` array of human-readable strings the
        frontend can surface to explain why constraints affected output.
        """
        notices: List[str] = []

        # 1. Filter by category preference. Hard if user explicitly picked
        # categories — we don't expand to all categories silently anymore
        # because that subverts user intent. If too narrow, we keep the
        # narrow pool and let max_km/count be the limiter.
        if categories:
            pool = self.df[self.df["category"].isin(categories)].copy()
            if pool.empty:
                # User picked categories but none exist in dataset (shouldn't happen
                # post-validation but guard anyway).
                pool = self.df.copy()
                notices.append(
                    "Kategori yang kamu pilih tidak ada di database — menggunakan semua kategori."
                )
        else:
            pool = self.df.copy()

        # 2. HARD max_km filter from HOME (radius semantic, not per-hop).
        # This matches notebook v3 §11 (env.get_valid_actions hard-gate).
        if max_km is not None:
            pool["_dist_home_pre"] = pool.apply(
                lambda r: haversine_km(home, (r["lat"], r["lng"])), axis=1
            )
            n_before = len(pool)
            pool = pool[pool["_dist_home_pre"] <= float(max_km)].copy()
            n_after = len(pool)

            if pool.empty:
                # No destination satisfies the radius — explain why.
                msg = (
                    f"Tidak ada destinasi wisata dalam radius {max_km:.0f} km dari "
                    f"titik mulai. Coba perbesar batasan jarak (mis. 15-25 km), atau "
                    f"pilih titik mulai yang lebih dekat ke pusat kota."
                )
                notices.append(msg)
                log.info("max_km=%s filtered out all %d candidates.", max_km, n_before)
                return self._empty_itinerary(home, start_min, end_min, notices)

            log.info(
                "max_km=%.1f kept %d/%d destinations in radius from home.",
                max_km, n_after, n_before
            )

        # 3. HARD budget filter (no destination above remaining budget can be picked).
        # Pre-filter at the pool level: any single-destination > budget is a no-op.
        if budget is not None and budget > 0:
            n_before = len(pool)
            pool = pool[pool["ticket"].astype(int) <= int(budget)].copy()
            if len(pool) < n_before:
                log.info(
                    "Budget %s removed %d destinations above budget.",
                    budget, n_before - len(pool)
                )
            if pool.empty:
                notices.append(
                    f"Tidak ada destinasi dengan tiket di bawah budget Rp {budget:,}. "
                    "Coba naikkan budget atau biarkan kosong."
                )
                return self._empty_itinerary(home, start_min, end_min, notices)

        # 4. Score & take top candidates (overshoot for NN-TSP slack)
        scored = self._score_candidates(pool, categories, home)
        top_n = max(count * 4, 16)
        candidates = scored.nlargest(top_n, "_score").reset_index(drop=True)

        # 5. Nearest-neighbor walk from home, respecting all constraints.
        # max_km is enforced as distance-from-HOME (radius), NOT per-hop.
        chain: List[pd.Series] = []
        cursor: Tuple[float, float] = home
        used: set[str] = set()
        spent = 0

        for _ in range(count):
            best_idx, best_dist_from_cursor = -1, math.inf
            for i, row in candidates.iterrows():
                if row["id"] in used:
                    continue
                # Hard radius check (distance from home)
                d_home = haversine_km(home, (row["lat"], row["lng"]))
                if max_km is not None and d_home > max_km:
                    continue
                # Hard budget check (cumulative)
                if budget is not None and spent + int(row["ticket"]) > budget:
                    continue
                # Among allowed, prefer closest to current cursor (NN-TSP)
                d_cursor = haversine_km(cursor, (row["lat"], row["lng"]))
                if d_cursor < best_dist_from_cursor:
                    best_dist_from_cursor = d_cursor
                    best_idx = i

            if best_idx == -1:
                # No more valid candidates within constraints. Stop early —
                # never relax max_km/budget per the user's HARD requirement.
                break

            picked = candidates.iloc[best_idx]
            used.add(picked["id"])
            chain.append((picked, best_dist_from_cursor))
            spent += int(picked["ticket"])
            cursor = (picked["lat"], picked["lng"])

        # 6. Category fairness post-pick (notebook v3 §15 enforce_category_guarantee)
        # Only enforce when count > 1, len(categories) >= 2, and we actually have room.
        # Single-destination trips bypass guarantee per user's spec.
        if (
            count > 1
            and len(categories) >= 2
            and len(chain) >= min(count, len(categories))
        ):
            chain, swap_notices = self._enforce_category_fairness(
                chain=chain,
                candidates=candidates,
                categories=categories,
                home=home,
                max_km=max_km,
                budget=budget,
            )
            notices.extend(swap_notices)

            # Recompute travel distances for the (possibly reordered) chain.
            chain = self._reflow_chain_distances(chain, home)

        # 7. Notice if we couldn't fill all slots
        if len(chain) < count:
            notices.append(
                f"Hanya {len(chain)} destinasi yang memenuhi semua batasan "
                f"(diminta {count}). Coba longgarkan jarak/budget atau "
                f"perpanjang jam perjalanan."
            )

        # 8. Schedule (arrival/departure times) and totals
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

        # 9. Return-home leg
        return_km = haversine_km(cursor, home) if chain else 0.0
        return_min = round((return_km / CITY_TRAVEL_SPEED_KMH) * 60)
        total_km += return_km
        arrive_home = t + return_min
        total_time = arrive_home - start_min

        # Defensive: budget overrun should not happen with HARD filter, but log if it does.
        if budget is not None and total_cost > budget:
            log.warning(
                "Budget overrun despite hard filter: total=%d budget=%d", total_cost, budget
            )

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
            "notices": notices,
        }

    def _empty_itinerary(
        self,
        home: Tuple[float, float],
        start_min: int,
        end_min: int,
        notices: List[str],
    ) -> Dict[str, Any]:
        """Short-circuit response when no destinations satisfy hard constraints."""
        return {
            "steps": [],
            "totalCost": 0,
            "totalKm": 0.0,
            "totalTime": 0,
            "returnKm": 0.0,
            "returnMin": 0,
            "arriveHome": start_min,
            "overBudget": False,
            "spareMin": end_min - start_min,
            "notices": notices,
        }

    def _enforce_category_fairness(
        self,
        chain: List[Tuple[pd.Series, float]],
        candidates: pd.DataFrame,
        categories: List[str],
        home: Tuple[float, float],
        max_km: Optional[float],
        budget: Optional[int],
    ) -> Tuple[List[Tuple[pd.Series, float]], List[str]]:
        """Ensure every user-selected category appears at least once.

        Mirrors notebook v3 §15 enforce_category_guarantee. Strategy:
          1. Identify missing categories from the chain.
          2. For each missing category, find the highest-scored candidate
             (in pool, satisfying all HARD constraints) and swap with the
             most over-represented destination that has the lowest rating.
          3. Travel distances are recomputed by `_reflow_chain_distances` after.
        """
        notices: List[str] = []
        if not chain:
            return chain, notices

        chosen = list(chain)  # copies of (row, dist) tuples
        chosen_ids = {row["id"] for row, _ in chosen}
        chosen_cats = [row["category"] for row, _ in chosen]
        from collections import Counter
        cat_counts = Counter(chosen_cats)
        missing = [c for c in categories if c not in cat_counts]
        if not missing:
            return chosen, notices

        budget_left = (
            int(budget) - sum(int(r["ticket"]) for r, _ in chosen)
            if budget is not None
            else None
        )

        for missing_cat in missing:
            # Find best-scored candidate of this missing category that fits constraints.
            cand_pool = candidates[
                (candidates["category"] == missing_cat)
                & (~candidates["id"].isin(chosen_ids))
            ].sort_values("_score", ascending=False)

            picked_replacement = None
            for _, row in cand_pool.iterrows():
                # Hard radius check (from HOME)
                d_home = haversine_km(home, (row["lat"], row["lng"]))
                if max_km is not None and d_home > max_km:
                    continue
                # Hard budget — must fit the swap delta.
                # We compare against the *swap target's* ticket so we know the net.
                picked_replacement = row
                break

            if picked_replacement is None:
                continue

            # Find swap target: kategori paling over-represented dengan rating terendah.
            over_cats = [
                c for c in cat_counts
                if c != missing_cat and cat_counts[c] > 1
            ]
            if over_cats:
                # Most over-represented first
                target_cat = max(over_cats, key=lambda c: cat_counts[c])
                # Lowest rated entry of that category in chosen
                swap_target_idx = -1
                lowest_rating = math.inf
                for i, (r, _) in enumerate(chosen):
                    if r["category"] == target_cat and float(r["rating"]) < lowest_rating:
                        lowest_rating = float(r["rating"])
                        swap_target_idx = i
            else:
                # No over-represented category — swap the lowest-rated NON-missing entry
                swap_target_idx = -1
                lowest_rating = math.inf
                for i, (r, _) in enumerate(chosen):
                    if r["category"] != missing_cat and float(r["rating"]) < lowest_rating:
                        lowest_rating = float(r["rating"])
                        swap_target_idx = i

            if swap_target_idx == -1:
                continue

            swap_target_row, _ = chosen[swap_target_idx]
            # Verify budget after swap
            if budget_left is not None:
                delta = int(picked_replacement["ticket"]) - int(swap_target_row["ticket"])
                if budget_left - delta < 0:
                    continue
                budget_left -= delta

            # Apply swap (distance will be recomputed)
            chosen[swap_target_idx] = (picked_replacement, 0.0)
            chosen_ids.discard(swap_target_row["id"])
            chosen_ids.add(picked_replacement["id"])
            cat_counts[swap_target_row["category"]] -= 1
            cat_counts[missing_cat] = cat_counts.get(missing_cat, 0) + 1
            log.info(
                "Category swap: '%s' (%s) → '%s' (%s)",
                swap_target_row["name"], swap_target_row["category"],
                picked_replacement["name"], missing_cat,
            )

        # Final notice if some categories still missing
        final_cats = {r["category"] for r, _ in chosen}
        still_missing = [c for c in categories if c not in final_cats]
        if still_missing:
            notices.append(
                f"Kategori {', '.join(still_missing)} tidak terwakili karena "
                f"tidak ada destinasi yang memenuhi semua batasan."
            )

        return chosen, notices

    @staticmethod
    def _reflow_chain_distances(
        chain: List[Tuple[pd.Series, float]],
        home: Tuple[float, float],
    ) -> List[Tuple[pd.Series, float]]:
        """Recompute travel-from-previous distances after a category swap.

        Order is preserved (we don't re-run TSP) but distances need to reflect
        the new sequence. Returns new list of (row, dist_from_prev_or_home).
        """
        out: List[Tuple[pd.Series, float]] = []
        prev: Tuple[float, float] = home
        for row, _ in chain:
            d = haversine_km(prev, (row["lat"], row["lng"]))
            out.append((row, d))
            prev = (row["lat"], row["lng"])
        return out
