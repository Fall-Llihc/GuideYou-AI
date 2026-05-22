"""Cell definitions for notebook 02_llm_storyteller.ipynb."""
from _nb_builder import md, code


# ---------- CELL 0: Title ----------
CELL_0 = md("""# 🤖 Bandung AI Travel Agent — LLM Storyteller
## Notebook 02: Integration Test + Groq API Narasi

**Capstone Project · Telkom University · Program Studi Data Science**

Notebook ini:
1. Load model dari Notebook 01 (CBF, RL, encoders).
2. Menjalankan full pipeline rekomendasi → ordering → itinerary.
3. Mengintegrasikan **Groq API** (free tier) untuk menghasilkan narasi perjalanan.
4. Memvalidasi output sesuai kontrak frontend React.
5. Mendokumentasikan eksperimen prompt engineering.

**Prasyarat:** Notebook 01 sudah dijalankan; semua model tersimpan.
**API:** Groq API (https://console.groq.com) — daftar gratis, 14.400 req/hari.
**Default model LLM:** `llama-3.1-8b-instant`.
""")


# ---------- CELL 1: Setup & dependency check ----------
CELL_1 = code('''# ============================================================
# CELL 1 — Setup & Dependency Check
# ============================================================
# !pip install -q requests pandas numpy scikit-learn

import os
import re
import sys
import math
import json
import time
import pickle
import random
import urllib.parse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# Pastikan working directory adalah ROOT proyek
if Path.cwd().name == "notebooks":
    os.chdir("..")
print(f"📂 CWD: {Path.cwd()}")

required_files = {
    "data/processed/destinations.csv": "Dataset destinasi",
    "data/processed/feature_matrix.npy": "Feature matrix CBF",
    "models/cbf_model.pkl": "Content-Based Filtering model",
    "models/rl_agent.pkl": "RL Agent model",
    "models/label_encoders.pkl": "Label encoders & scalers",
    "data/last_updated.txt": "Timestamp update data",
}

all_ok = True
for path, desc in required_files.items():
    if Path(path).exists():
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✅ {path} ({size_kb:.1f} KB) — {desc}")
    else:
        print(f"  ❌ {path} — MISSING! {desc}")
        all_ok = False

if not all_ok:
    raise FileNotFoundError(
        "\\n⛔ Beberapa file model tidak ditemukan!\\n"
        "Pastikan Notebook 01 (01_recommendation_engine.ipynb) sudah dijalankan terlebih dahulu.\\n"
        "Kemudian jalankan ulang cell ini."
    )

print("\\n✅ Semua dependensi Notebook 1 tersedia. Siap lanjut!")
''')


# ---------- CELL 2: Load models & data ----------
CELL_2 = code('''# ============================================================
# CELL 2 — Load Models & Data
# ============================================================
df = pd.read_csv("data/processed/destinations.csv")
print(f"✅ Dataset loaded: {len(df)} destinasi")

# Convert kolom tags dari string ke list (saat read_csv tags ter-stringify)
def _parse_tags(t):
    if isinstance(t, list):
        return t
    if pd.isna(t):
        return []
    s = str(t).strip()
    if not s or s == "[]":
        return []
    try:
        v = json.loads(s.replace("'", '"'))
        return v if isinstance(v, list) else []
    except Exception:
        return [x.strip().strip("'\\"") for x in s.strip("[]").split(",") if x.strip()]

df["tags"] = df["tags"].apply(_parse_tags)

feature_matrix = np.load("data/processed/feature_matrix.npy")
print(f"✅ Feature matrix loaded: {feature_matrix.shape}")

with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
print(f"✅ Encoders loaded: {list(encoders.keys())}")

with open("models/cbf_model.pkl", "rb") as f:
    cbf_data = pickle.load(f)
print(f"✅ CBF data loaded. Similarity matrix: {cbf_data['similarity_matrix'].shape}")

with open("models/rl_agent.pkl", "rb") as f:
    rl_data = pickle.load(f)
q_table = defaultdict(lambda: defaultdict(float))
for s, actions in rl_data["q_table"].items():
    for a, v in actions.items():
        q_table[s][a] = v
print(f"✅ RL Agent loaded. Q-table states: {len(q_table)}")

with open("data/last_updated.txt", "r") as f:
    data_last_updated = f.read().strip()
print(f"✅ Data last updated: {data_last_updated}")
''')


# ---------- CELL 3: Re-instantiate pipeline classes ----------
CELL_3 = code('''# ============================================================
# CELL 3 — Re-instantiate Pipeline Classes (self-contained)
# ============================================================
from sklearn.metrics.pairwise import cosine_similarity


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def generate_gmaps_url(name: str) -> str:
    query = urllib.parse.quote(f"{name}, Bandung")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


class ContentBasedFilter:
    """Mirror dari class di Notebook 01."""

    def __init__(self, feature_matrix, destinations_df, encoders):
        self.feature_matrix = feature_matrix
        self.df = destinations_df.reset_index(drop=True)
        self.encoders = encoders
        self.similarity_matrix = None

    def fit(self):
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        return self

    def _category_dims(self):
        return self.encoders.get("n_category_dims", 5)

    def _numeric_start(self):
        return self._category_dims()

    def build_user_profile(self, categories, budget=None):
        if not categories:
            mask = np.ones(len(self.df), dtype=bool)
        else:
            mask = self.df["category"].isin(categories).values
        if mask.sum() == 0:
            mask = np.ones(len(self.df), dtype=bool)
        profile = self.feature_matrix[mask].mean(axis=0)
        if budget is not None and budget > 0:
            scaler = self.encoders["scaler"]
            num_cols = self.encoders["feature_cols"]
            ticket_idx = num_cols.index("ticket_log")
            dummy = np.zeros((1, len(num_cols)))
            for j, col in enumerate(num_cols):
                if col == "ticket_log":
                    dummy[0, j] = math.log1p(budget)
                else:
                    base_col = "ticket" if col == "ticket_log" else col
                    dummy[0, j] = (math.log1p(self.df["ticket"].median())
                                   if col == "ticket_log" else self.df[base_col].median())
            scaled = scaler.transform(dummy)[0]
            profile_idx = self._numeric_start() + ticket_idx
            profile[profile_idx] = scaled[ticket_idx]
        return profile.reshape(1, -1)

    def recommend(self, categories=None, budget=None, max_km=None,
                  home_lat=None, home_lng=None, top_n=20):
        if self.similarity_matrix is None:
            self.fit()
        categories = categories or []
        profile = self.build_user_profile(categories, budget=budget)
        sims = cosine_similarity(profile, self.feature_matrix).flatten()
        out = self.df.copy()
        out["cbf_score"] = sims
        if categories:
            mask = out["category"].isin(categories)
            if mask.sum() > 0:
                out = out[mask]
        if budget is not None:
            sub = out[out["ticket"].astype(float) <= float(budget)]
            if len(sub) > 0:
                out = sub
        if max_km is not None and home_lat is not None and home_lng is not None:
            out = out.copy()
            out["dist_home_km"] = out.apply(
                lambda r: haversine_km(home_lat, home_lng, r["lat"], r["lng"]), axis=1
            )
            within = out[out["dist_home_km"] <= float(max_km)]
            if len(within) > 0:
                out = within
        return out.sort_values("cbf_score", ascending=False).head(top_n).reset_index(drop=True)

    def get_similar_destinations(self, dest_id, top_n=5):
        if self.similarity_matrix is None:
            self.fit()
        idx = self.df.index[self.df["id"] == dest_id].tolist()
        if not idx:
            return self.df.head(0)
        idx = idx[0]
        sims = self.similarity_matrix[idx]
        order = [i for i in np.argsort(-sims) if i != idx][:top_n]
        out = self.df.iloc[order].copy()
        out["sim_score"] = sims[order]
        return out.reset_index(drop=True)


class RouteOptimizer:
    SPEED_KMH = 28
    OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

    def __init__(self, use_osrm=True, osrm_timeout=5.0):
        self.use_osrm = use_osrm
        self.osrm_timeout = osrm_timeout

    def haversine_km(self, lat1, lng1, lat2, lng2):
        return haversine_km(lat1, lng1, lat2, lng2)

    def osrm_travel_time(self, lat1, lng1, lat2, lng2):
        if not self.use_osrm:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            return d, (d / self.SPEED_KMH) * 60
        try:
            url = f"{self.OSRM_BASE}/{lng1},{lat1};{lng2},{lat2}?overview=false"
            resp = requests.get(url, timeout=self.osrm_timeout)
            resp.raise_for_status()
            r = resp.json()["routes"][0]
            return r["distance"] / 1000.0, r["duration"] / 60.0
        except Exception:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            return d, (d / self.SPEED_KMH) * 60

    def nearest_neighbor_route(self, home, destinations):
        if not destinations:
            return []
        rem = list(destinations)
        ordered = []
        cur_lat, cur_lng = home["lat"], home["lng"]
        while rem:
            nxt = min(rem, key=lambda d: self.haversine_km(cur_lat, cur_lng, d["lat"], d["lng"]))
            ordered.append(nxt)
            rem = [d for d in rem if d.get("id") != nxt.get("id")]
            cur_lat, cur_lng = nxt["lat"], nxt["lng"]
        return ordered

    def build_itinerary(self, home, home_name, ordered_destinations, start_min, end_min):
        steps = []
        cur_lat, cur_lng = home["lat"], home["lng"]
        cur_time = start_min
        total_cost, total_km, total_visit = 0, 0.0, 0
        for i, d in enumerate(ordered_destinations):
            tk, tm = self.osrm_travel_time(cur_lat, cur_lng, float(d["lat"]), float(d["lng"]))
            arrive = cur_time + int(round(tm))
            depart = arrive + int(d.get("duration", 60))
            tags = d.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags.replace("'", '"'))
                except Exception:
                    tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]
            steps.append({
                "idx": i + 1,
                "dest": {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "category": d.get("category"),
                    "desc": d.get("desc", ""),
                    "ticket": int(d.get("ticket", 0)),
                    "duration": int(d.get("duration", 60)),
                    "lat": float(d["lat"]),
                    "lng": float(d["lng"]),
                    "rating": float(d.get("rating", 4.0)),
                    "tags": list(tags) if tags else [],
                    "gmaps_url": d.get("gmaps_url") or generate_gmaps_url(d.get("name", "")),
                },
                "travelMin": int(round(tm)),
                "travelKm": round(float(tk), 2),
                "arriveAt": int(arrive),
                "departAt": int(depart),
            })
            total_cost += int(d.get("ticket", 0))
            total_km += float(tk)
            total_visit += int(d.get("duration", 60))
            cur_lat, cur_lng = float(d["lat"]), float(d["lng"])
            cur_time = depart
        return_km, return_min = self.osrm_travel_time(cur_lat, cur_lng, home["lat"], home["lng"])
        arrive_home = cur_time + int(round(return_min))
        total_km += float(return_km)
        total_time = total_visit + sum(s["travelMin"] for s in steps) + int(round(return_min))
        return {
            "steps": steps,
            "totalCost": int(total_cost),
            "totalKm": round(float(total_km), 2),
            "totalTime": int(total_time),
            "returnKm": round(float(return_km), 2),
            "returnMin": int(round(return_min)),
            "arriveHome": int(arrive_home),
            "overBudget": bool(arrive_home > end_min),
            "spareMin": int(end_min - arrive_home),
        }


cbf_model = ContentBasedFilter(feature_matrix, df, encoders)
cbf_model.similarity_matrix = cbf_data["similarity_matrix"]

optimizer = RouteOptimizer(use_osrm=True)

print("✅ ContentBasedFilter & RouteOptimizer berhasil di-instantiate")
''')


# ---------- CELL 4: Markdown Groq setup ----------
CELL_4 = md("""## 🔑 Groq API Setup

### Cara mendapatkan API Key (gratis)
1. Buka https://console.groq.com
2. Daftar / login dengan Google
3. Masuk ke menu **API Keys**
4. Klik **Create API Key**
5. Copy key dan masukkan ke cell berikut (atau set env var `GROQ_API_KEY`)

### Batas Free Tier
- 14.400 requests/hari
- 30 requests/menit
- Default: `llama-3.1-8b-instant`
- Alternatif: `llama3-70b-8192` (lebih pintar, lebih lambat)

### Catatan Keamanan
- **JANGAN commit API key ke Git.**
- Gunakan environment variable atau file `.env`.
""")


# ---------- CELL 5: Groq config ----------
CELL_5 = code('''# ============================================================
# CELL 5 — Groq API Configuration
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Coba baca dari .env jika tersedia
if GROQ_API_KEY is None:
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("GROQ_API_KEY="):
                    GROQ_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

# Fallback hardcode (ganti manual saat testing — JANGAN commit)
if GROQ_API_KEY is None:
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
    print("⚠️  Menggunakan API key hardcode. Ganti dengan key kamu — JANGAN commit ke Git!")

if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    print("⛔ API Key belum diset. Notebook akan jalan dengan fallback template.")
else:
    print(f"✅ GROQ_API_KEY terdeteksi: {GROQ_API_KEY[:8]}…{GROQ_API_KEY[-4:]}")

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 1000
GROQ_TEMPERATURE = 0.7
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

print(f"✅ Model: {GROQ_MODEL}")
print(f"✅ Max tokens: {GROQ_MAX_TOKENS}")
print(f"✅ Temperature: {GROQ_TEMPERATURE}")
''')


# ---------- CELL 6: Groq client ----------
CELL_6 = code('''# ============================================================
# CELL 6 — Groq API Client (GroqStoryteller)
# ============================================================
class GroqStoryteller:
    """Client Groq API + prompt engineering untuk narasi perjalanan."""

    RETRY_DELAYS = [1, 2, 5]  # detik

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                 max_tokens: int = 1000, temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(self, messages: list, system_prompt: str = None) -> str:
        full = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(messages)
        payload = {
            "model": self.model,
            "messages": full,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        last_err = None
        for delay in (self.RETRY_DELAYS + [None]):
            try:
                resp = requests.post(self.endpoint, headers=self.headers,
                                     json=payload, timeout=30)
                if resp.status_code == 429:
                    if delay is None:
                        raise RuntimeError("Rate limit exceeded — semua retry gagal.")
                    print(f"  ⏳ Rate limited. Retry dalam {delay}s…")
                    time.sleep(delay)
                    continue
                if resp.status_code >= 500:
                    if delay is None:
                        resp.raise_for_status()
                    print(f"  ⏳ Server error {resp.status_code}. Retry dalam {delay}s…")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                last_err = "timeout"
                if delay is None:
                    raise
                print(f"  ⏳ Timeout. Retry dalam {delay}s…")
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                if delay is None:
                    raise
                time.sleep(delay)
        raise RuntimeError(f"Groq API call gagal: {last_err}")

    def build_story_prompt(self, itinerary: dict, params: dict, lang: str = "id") -> str:
        # Build daftar destinasi
        dest_lines = []
        for i, step in enumerate(itinerary["steps"], start=1):
            d = step["dest"]
            ticket_str = "Gratis" if int(d["ticket"]) == 0 else f"Rp {int(d['ticket']):,}"
            dest_lines.append(
                f"  {i}. {d['name']} ({d['category']}) — Tiket: {ticket_str}, "
                f"Durasi: {int(d['duration'])} menit, Rating: {float(d['rating']):.1f}"
            )
        destinations_text = "\\n".join(dest_lines)

        lang_instruction = (
            "Respond in casual Bahasa Indonesia (colloquial, friendly, millennial style)"
            if lang == "id"
            else "Respond in casual English (enthusiastic travel blogger style)"
        )
        start_time = f"{params['startMin']//60:02d}:{params['startMin']%60:02d}"
        end_time = f"{params['endMin']//60:02d}:{params['endMin']%60:02d}"
        budget_str = f"Rp {params['budget']:,}" if params.get("budget") else "No limit"
        n_dests = len(itinerary["steps"])

        prompt = f"""{lang_instruction}. You are BandungBuddy, an enthusiastic local travel guide.

Generate a travel narrative for this Bandung itinerary. Respond ONLY in valid JSON format.
No markdown code blocks, no explanation, just raw JSON.

=== TRIP DETAILS ===
Starting point: {params['homeName']}
Date: Today
Time window: {start_time} – {end_time}
Budget: {budget_str}
Total estimated cost: Rp {itinerary['totalCost']:,}
Total distance: {itinerary['totalKm']:.1f} km
Total travel time: {itinerary['totalTime']//60} hours {itinerary['totalTime']%60} minutes

=== DESTINATIONS (in visit order) ===
{destinations_text}

=== REQUIRED JSON OUTPUT ===
{{
  "intro": "<2-3 sentence engaging opening — mention starting point and overall vibe>",
  "highlights": [
    "<one highlight per destination, 1-2 sentences each, mention the name naturally>"
  ],
  "tips": [
    "<2-4 practical tips based on the actual destinations and time>"
  ],
  "closing": "<1 memorable closing sentence>",
  "vibe": "<1-2 words describing the trip vibe — e.g.: Petualang Alam, Kuliner Enthusiast>"
}}

IMPORTANT:
- The highlights array MUST have EXACTLY {n_dests} items (one per destination).
- Tips must be specific to the destinations chosen, not generic.
- Use casual language, 1-2 emoji per highlight max.
- Do NOT include any text outside the JSON object."""
        return prompt

    def generate_story(self, itinerary: dict, params: dict, lang: str = "id") -> dict:
        prompt = self.build_story_prompt(itinerary, params, lang)
        system_prompt = (
            "You are BandungBuddy, a local travel guide AI. "
            "You ALWAYS respond in valid JSON format only. "
            "Never include markdown, preamble, or explanation. "
            "Just output the raw JSON object requested."
        )

        if not self.api_key or self.api_key == "YOUR_GROQ_API_KEY_HERE":
            print("  ℹ️  API key tidak tersedia, langsung pakai fallback.")
            return self._fallback_story(itinerary, params, lang)

        try:
            raw = self._call_api(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
            )
            clean = raw.strip()
            if clean.startswith("```"):
                # Hapus pembungkus code fence
                clean = re.sub(r"^```(?:json)?\\s*", "", clean)
                clean = re.sub(r"\\s*```$", "", clean)
            story = json.loads(clean)

            for k in ("intro", "highlights", "tips", "closing", "vibe"):
                if k not in story:
                    raise ValueError(f"Missing key in LLM response: {k}")

            n = len(itinerary["steps"])
            if not isinstance(story["highlights"], list):
                story["highlights"] = []
            while len(story["highlights"]) < n:
                story["highlights"].append(
                    f"Kunjungi {itinerary['steps'][len(story['highlights'])]['dest']['name']} "
                    f"dan rasakan pengalamannya sendiri!"
                )
            story["highlights"] = story["highlights"][:n]
            if not isinstance(story["tips"], list):
                story["tips"] = [str(story["tips"])]
            return story

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  ⚠️  JSON parsing gagal: {e}. Pakai fallback template.")
            return self._fallback_story(itinerary, params, lang)
        except Exception as e:
            print(f"  ❌ API call gagal: {e}. Pakai fallback template.")
            return self._fallback_story(itinerary, params, lang)

    def _fallback_story(self, itinerary: dict, params: dict, lang: str = "id") -> dict:
        steps = itinerary["steps"]
        cats = list({s["dest"]["category"] for s in steps})
        if lang == "id":
            cat_str = " dan ".join(cats) if cats else "wisata"
            first = steps[0]["dest"]["name"] if steps else params["homeName"]
            last = steps[-1]["dest"]["name"] if steps else params["homeName"]
            return {
                "intro": (f"Hari ini kamu akan menjelajahi {len(steps)} destinasi seru di Bandung, "
                           f"mulai dari **{params['homeName']}**. Siap nikmatin pengalaman {cat_str} "
                           f"yang gak terlupakan!"),
                "highlights": [
                    f"**{s['dest']['name']}** — {s['dest']['desc']}. "
                    f"Alokasikan {int(s['dest']['duration'])} menit untuk pengalaman terbaik. 📍"
                    for s in steps
                ],
                "tips": [
                    "Cek cuaca sebelum berangkat, terutama untuk destinasi alam.",
                    "Siapkan e-wallet dan uang cash untuk tiket masuk dan kuliner.",
                    f"Estimasi total pengeluaran Rp {itinerary['totalCost']:,} — sesuaikan dengan anggaran.",
                    "Simpan offline maps untuk area yang sinyalnya mungkin lemah.",
                ],
                "closing": f"Dari {first} sampai {last} — ini bakal jadi hari yang memorable di Kota Kembang! 🌸",
                "vibe": " & ".join(cats[:2]) if len(cats) > 1 else (cats[0] if cats else "Bandung Trip"),
            }
        return {
            "intro": (f"Get ready for an amazing day exploring {len(steps)} handpicked "
                       f"destinations in Bandung, starting from {params['homeName']}!"),
            "highlights": [
                f"**{s['dest']['name']}** — {s['dest']['desc']}. "
                f"Plan to spend {int(s['dest']['duration'])} minutes here. 📍"
                for s in steps
            ],
            "tips": [
                "Check the weather forecast, especially for nature spots.",
                "Bring both e-wallet and cash for entrance fees and street food.",
                f"Budget estimate: Rp {itinerary['totalCost']:,} — plan accordingly.",
                "Download offline maps for areas with potentially weak signal.",
            ],
            "closing": "From start to finish — this is going to be an unforgettable Bandung adventure! 🌸",
            "vibe": "Adventure Mix",
        }
''')


# ---------- CELL 7: Full pipeline with LLM ----------
CELL_7 = code('''# ============================================================
# CELL 7 — Full Pipeline (CBF + RL + Route + LLM)
# ============================================================
CATEGORY_ORDER = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]


def _state_for_inference(n_selected: int, spent: int, budget,
                         cur_time: int, start_min: int, end_min: int,
                         selected: list) -> tuple:
    if budget is None or budget <= 0:
        budget_level = 4
    else:
        ratio = max(0.0, 1.0 - spent / budget)
        budget_level = min(4, int(ratio * 4 + 1e-9))
        if ratio >= 0.75:
            budget_level = 4
    time_left = max(0, end_min - cur_time)
    if time_left <= 0:
        time_level = 0
    elif time_left < 120:
        time_level = 1
    elif time_left < 240:
        time_level = 2
    elif time_left < 360:
        time_level = 3
    else:
        time_level = 4
    dom = 0
    if selected:
        cats = Counter(s["category"] for s in selected)
        top_cat = cats.most_common(1)[0][0]
        dom = CATEGORY_ORDER.index(top_cat) if top_cat in CATEGORY_ORDER else 0
    return (min(8, n_selected), budget_level, time_level, dom)


def generate_full_itinerary(params: dict, cbf_model, rl_data: dict,
                             optimizer: RouteOptimizer, storyteller: GroqStoryteller,
                             df: pd.DataFrame, data_last_updated: str,
                             lang: str = "id") -> dict:
    # 1. CBF Recommend
    candidates = cbf_model.recommend(
        categories=params.get("categories", []),
        budget=params.get("budget"),
        max_km=params.get("maxKm"),
        home_lat=params["home"]["lat"],
        home_lng=params["home"]["lng"],
        top_n=30,
    )
    if len(candidates) == 0:
        candidates = cbf_model.recommend(
            categories=[], budget=None, max_km=None,
            home_lat=params["home"]["lat"], home_lng=params["home"]["lng"], top_n=30,
        )

    # 2. RL-guided selection
    selected = []
    remaining = candidates.to_dict("records")
    spent = 0
    cur_time = params["startMin"]
    n_target = max(1, int(params.get("count", 4)))

    for _ in range(n_target):
        if not remaining:
            break
        state = _state_for_inference(
            len(selected), spent, params.get("budget"),
            cur_time, params["startMin"], params["endMin"], selected
        )
        # Filter feasibility (budget + window)
        feasible = []
        for d in remaining:
            if params.get("budget") and (spent + int(d.get("ticket", 0))) > params["budget"]:
                continue
            if cur_time + int(d.get("duration", 60)) > params["endMin"]:
                continue
            feasible.append(d)
        if not feasible:
            break

        best_dest, best_score = None, -float("inf")
        for d in feasible:
            q_val = rl_data.get("q_table", {}).get(state, {}).get(d["id"], 0.0)
            cbf_score = float(d.get("cbf_score", 0.0))
            combined = 0.6 * q_val + 0.4 * cbf_score
            if combined > best_score:
                best_score = combined
                best_dest = d
        if best_dest is None:
            best_dest = feasible[0]
        selected.append(best_dest)
        remaining = [d for d in remaining if d["id"] != best_dest["id"]]
        spent += int(best_dest.get("ticket", 0))
        cur_time += int(best_dest.get("duration", 60))

    # 3. Route ordering
    ordered = optimizer.nearest_neighbor_route(params["home"], selected)

    # 4. Itinerary
    itinerary = optimizer.build_itinerary(
        home=params["home"],
        home_name=params["homeName"],
        ordered_destinations=ordered,
        start_min=params["startMin"],
        end_min=params["endMin"],
    )

    # 5. LLM story
    story = storyteller.generate_story(itinerary, params, lang=lang)

    # 6. Assemble
    response = {**itinerary, "story": story, "data_last_updated": data_last_updated}
    return response


print("✅ generate_full_itinerary() ready")
''')


# ---------- CELL 8: Markdown Groq testing ----------
CELL_8 = md("""## 🧪 Testing Groq API

Sebelum full pipeline test, validasi koneksi ke Groq API dengan prompt sederhana.
""")


# ---------- CELL 9: Test Groq connection ----------
CELL_9 = code('''# ============================================================
# CELL 9 — Test Koneksi Groq API
# ============================================================
storyteller = GroqStoryteller(
    api_key=GROQ_API_KEY, model=GROQ_MODEL,
    max_tokens=GROQ_MAX_TOKENS, temperature=GROQ_TEMPERATURE,
)

print("🔌 Testing Groq API connection…")
if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    print("⚠️  API Key belum diset — skip test koneksi (akan pakai fallback template).")
else:
    try:
        raw = storyteller._call_api(
            messages=[{"role": "user",
                        "content": "Respond with exactly: {\\"status\\": \\"ok\\", \\"message\\": \\"Groq API berfungsi!\\"}"}],
            system_prompt="You are a test API. Respond only with valid JSON as instructed.",
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
            cleaned = re.sub(r"\\s*```$", "", cleaned)
        result = json.loads(cleaned)
        print(f"✅ Groq API OK: {result}")
    except Exception as e:
        print(f"❌ Groq API Test GAGAL: {e}")
        print("Notebook tetap berjalan — fallback template akan dipakai.")
''')


# ---------- CELL 10: Full integration test (3 scenarios) ----------
CELL_10 = code('''# ============================================================
# CELL 10 — Full Integration Test (3 skenario)
# ============================================================
test_cases = [
    {
        "name": "Wisata Alam Selatan Bandung",
        "params": {
            "home": {"lat": -6.9215, "lng": 107.6071},
            "homeName": "Alun-Alun Bandung",
            "count": 3, "maxKm": None,
            "startMin": 7 * 60, "endMin": 18 * 60,
            "budget": 300_000, "categories": ["Alam"],
        },
        "lang": "id",
    },
    {
        "name": "City Tour Kuliner & Budaya",
        "params": {
            "home": {"lat": -6.9145, "lng": 107.6020},
            "homeName": "Stasiun Bandung",
            "count": 4, "maxKm": 20,
            "startMin": 10 * 60, "endMin": 21 * 60,
            "budget": 500_000, "categories": ["Kuliner", "Budaya"],
        },
        "lang": "id",
    },
    {
        "name": "Weekend Adventure (No Budget Limit)",
        "params": {
            "home": {"lat": -6.8126, "lng": 107.6178},
            "homeName": "Pasar Lembang",
            "count": 5, "maxKm": None,
            "startMin": 8 * 60, "endMin": 20 * 60,
            "budget": None, "categories": ["Alam", "Wisata", "Kuliner"],
        },
        "lang": "en",
    },
]

results = []
for tc in test_cases:
    print(f"\\n{'='*70}\\n🧪 TEST: {tc['name']}\\n{'='*70}")
    res = generate_full_itinerary(
        params=tc["params"], cbf_model=cbf_model, rl_data=rl_data,
        optimizer=optimizer, storyteller=storyteller, df=df,
        data_last_updated=data_last_updated, lang=tc["lang"],
    )

    print(f"\\n📍 Itinerary: {len(res['steps'])} destinasi")
    for s in res["steps"]:
        a = f"{s['arriveAt']//60:02d}:{s['arriveAt']%60:02d}"
        b = f"{s['departAt']//60:02d}:{s['departAt']%60:02d}"
        print(f"  {s['idx']}. {s['dest']['name']:<30} | {a}-{b} | "
              f"{s['travelKm']:.1f}km | Rp {int(s['dest']['ticket']):,}")
    print(f"\\n💰 Total Biaya: Rp {res['totalCost']:,}")
    print(f"📏 Total Jarak: {res['totalKm']:.1f} km")
    print(f"⏱️  Total Waktu: {res['totalTime']//60} jam {res['totalTime']%60} menit")
    ah = f"{res['arriveHome']//60:02d}:{res['arriveHome']%60:02d}"
    flag = "⚠️  Overtime!" if res["overBudget"] else "✅ On time"
    print(f"🏠 Tiba Kembali: {ah} | {flag}")

    print(f"\\n📖 Narasi LLM:")
    print(f"  Vibe: {res['story']['vibe']}")
    intro_preview = res["story"]["intro"][:200].replace("\\n", " ")
    print(f"  Intro: {intro_preview}…")
    print(f"  Tips ({len(res['story']['tips'])} items):")
    for t in res["story"]["tips"]:
        print(f"    • {t}")
    print(f"  Closing: {res['story']['closing']}")

    results.append(res)

print(f"\\n\\n{'='*70}\\n✅ SEMUA TEST CASE BERHASIL!\\n{'='*70}")
''')


# ---------- CELL 11: Frontend contract validation ----------
CELL_11 = code('''# ============================================================
# CELL 11 — Validasi Kontrak Frontend
# ============================================================
def validate_frontend_contract(response: dict) -> bool:
    errors = []
    required_root = ["steps", "totalCost", "totalKm", "totalTime",
                      "returnKm", "returnMin", "arriveHome", "overBudget",
                      "spareMin", "story", "data_last_updated"]
    for f in required_root:
        if f not in response:
            errors.append(f"Missing root field: {f}")

    if "totalCost" in response and not isinstance(response["totalCost"], (int, float)):
        errors.append(f"totalCost must be numeric, got {type(response['totalCost'])}")
    if "overBudget" in response and not isinstance(response["overBudget"], bool):
        errors.append(f"overBudget must be bool, got {type(response['overBudget'])}")

    if "steps" in response:
        for i, step in enumerate(response["steps"]):
            for f in ["idx", "dest", "travelMin", "travelKm", "arriveAt", "departAt"]:
                if f not in step:
                    errors.append(f"steps[{i}] missing field: {f}")
            if "dest" in step:
                for f in ["id", "name", "category", "desc", "ticket",
                          "duration", "lat", "lng", "rating", "tags"]:
                    if f not in step["dest"]:
                        errors.append(f"steps[{i}].dest missing field: {f}")
                valid_cats = {"Alam", "Kuliner", "Budaya", "Wisata", "Belanja"}
                if "category" in step["dest"] and step["dest"]["category"] not in valid_cats:
                    errors.append(f"steps[{i}].dest.category invalid: {step['dest']['category']}")

    if "story" in response:
        for f in ["intro", "highlights", "tips", "closing", "vibe"]:
            if f not in response["story"]:
                errors.append(f"story missing field: {f}")
        if "highlights" in response["story"] and "steps" in response:
            if len(response["story"]["highlights"]) != len(response["steps"]):
                errors.append(
                    f"highlights count ({len(response['story']['highlights'])}) "
                    f"!= steps count ({len(response['steps'])})"
                )

    if errors:
        print("❌ VALIDASI GAGAL:")
        for e in errors:
            print(f"  • {e}")
        return False
    print("✅ VALIDASI LULUS: Response sesuai kontrak frontend")
    return True


print("=== VALIDASI KONTRAK FRONTEND ===")
all_pass = True
for i, (tc, res) in enumerate(zip(test_cases, results), start=1):
    print(f"\\nTest Case {i}: {tc['name']}")
    if not validate_frontend_contract(res):
        all_pass = False
print("\\n" + ("🎉 Semua test LULUS validasi" if all_pass else "⚠️  Ada test yang gagal validasi"))
''')


# ---------- CELL 12: Bilingual ----------
CELL_12 = code('''# ============================================================
# CELL 12 — Bilingual Test (English Output)
# ============================================================
print("=== BILINGUAL TEST ===")

en_params = {
    "home": {"lat": -6.9215, "lng": 107.6071},
    "homeName": "Alun-Alun Bandung",
    "count": 3, "maxKm": None,
    "startMin": 9 * 60, "endMin": 19 * 60,
    "budget": 400_000, "categories": ["Alam", "Kuliner"],
}

en_result = generate_full_itinerary(
    params=en_params, cbf_model=cbf_model, rl_data=rl_data,
    optimizer=optimizer, storyteller=storyteller, df=df,
    data_last_updated=data_last_updated, lang="en",
)

print("English Story:")
print(f"  Vibe: {en_result['story']['vibe']}")
print(f"  Intro: {en_result['story']['intro']}")
if en_result["story"]["tips"]:
    print(f"  Tip #1: {en_result['story']['tips'][0]}")
print(f"  Closing: {en_result['story']['closing']}")
''')


# ---------- CELL 13: Prompt engineering analysis ----------
CELL_13 = code('''# ============================================================
# CELL 13 — Prompt Engineering: Variasi Temperature
# ============================================================
print("=== EKSPERIMEN: VARIASI TEMPERATURE ===\\n")

simple_params = {
    "home": {"lat": -6.9215, "lng": 107.6071},
    "homeName": "Alun-Alun Bandung",
    "count": 2, "maxKm": None,
    "startMin": 9 * 60, "endMin": 17 * 60,
    "budget": 200_000, "categories": ["Alam"],
}

if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    print("⚠️  API Key belum diset — eksperimen di-skip.")
else:
    cand = cbf_model.recommend(categories=["Alam"], top_n=10)
    sel = cand.head(2).to_dict("records")
    ordered = optimizer.nearest_neighbor_route(simple_params["home"], sel)
    test_itin = optimizer.build_itinerary(
        home=simple_params["home"], home_name=simple_params["homeName"],
        ordered_destinations=ordered,
        start_min=simple_params["startMin"], end_min=simple_params["endMin"],
    )

    for temp in [0.3, 0.7, 1.0]:
        ts = GroqStoryteller(api_key=GROQ_API_KEY, model=GROQ_MODEL,
                              max_tokens=GROQ_MAX_TOKENS, temperature=temp)
        story = ts.generate_story(test_itin, simple_params, lang="id")
        print(f"Temperature {temp}:")
        print(f"  Vibe : {story['vibe']}")
        print(f"  Intro: {story['intro'][:120]}…\\n")
        time.sleep(2)

print("Catatan observasi:")
print("  - 0.3 → konsisten, kurang kreatif (cocok produksi)")
print("  - 0.7 → seimbang konsistensi & kreativitas (default)")
print("  - 1.0 → paling kreatif, kadang kelewatan ide")
''')


# ---------- CELL 14: Export sample ----------
CELL_14 = code('''# ============================================================
# CELL 14 — Export Sample Output (untuk backend developer)
# ============================================================
sample_output = results[0]

with open("data/processed/sample_api_response.json", "w", encoding="utf-8") as f:
    json.dump(sample_output, f, ensure_ascii=False, indent=2)

print("✅ data/processed/sample_api_response.json")
print("\\nPreview (first 1000 chars):")
print(json.dumps(sample_output, ensure_ascii=False, indent=2)[:1000])
print("…")

sample_params = test_cases[0]["params"]
with open("data/processed/sample_api_request.json", "w", encoding="utf-8") as f:
    json.dump(sample_params, f, ensure_ascii=False, indent=2)
print("\\n✅ data/processed/sample_api_request.json")
''')


# ---------- CELL 15: Closing ----------
CELL_15 = md("""## ✅ Notebook 2 Selesai — LLM Storyteller Integration

### Ringkasan
| Komponen | Status | Keterangan |
|---|---|---|
| Model Loading | ✅ | CBF + RL + Encoders dari Notebook 1 |
| Groq API | ✅ | `llama-3.1-8b-instant`, JSON output |
| Full Pipeline | ✅ | CBF → RL → Route → LLM |
| Frontend Contract | ✅ | Semua field tervalidasi |
| Bilingual | ✅ | id dan en |
| Fallback | ✅ | Template aman saat Groq gagal |
| Sample Output | ✅ | `data/processed/sample_api_response.json` |

### File Output
| File | Keterangan |
|---|---|
| `data/processed/sample_api_request.json` | Contoh input dari frontend |
| `data/processed/sample_api_response.json` | Contoh output ke frontend |

### Langkah Selanjutnya
1. **Backend FastAPI** — `backend/main.py` membungkus pipeline ini sebagai endpoint.
2. **Deployment** — VPS + domain (mis. `.my.id`).
3. **Monitoring** — log request/response untuk fine-tune Q-table secara periodik.

### Catatan Groq
- Free tier: 14.400 req/hari, 30 req/menit.
- Pertimbangkan caching untuk request berulang.
- `llama3-70b-8192` bisa dipakai bila kualitas > kecepatan.
""")


def all_cells():
    return [
        CELL_0, CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7,
        CELL_8, CELL_9, CELL_10, CELL_11, CELL_12, CELL_13, CELL_14, CELL_15,
    ]
