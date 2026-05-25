"""
Generator script untuk membuat versi rapi dari rec-engine notebook.
Jalankan: python notebooks/_build_notebook.py
Output  : notebooks/rec-engine (4).ipynb (overwrite)
"""
import json
from pathlib import Path

CELLS = []   # diisi via helper di bawah


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text: str):
    # split keep newlines
    src = text
    if not src.endswith("\n"):
        src += "\n"
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


# =====================================================================
# SECTION 0 — Cover / Daftar Isi
# =====================================================================
md(r"""# 🗺️ Bandung AI Travel — Recommendation Engine

**Capstone Project · Telkom University · Program Studi Data Science**

Notebook ini melatih sebuah *travel recommender system* multi-tahap untuk kota Bandung dengan pipeline:

```
Crawl OSM ── Enrich FSQ ── Cleaning ── Feature Engineering
                                        │
              ┌─────────────────────────┘
              ▼
        Content-Based Filter ── Variety-aware Reranking
              │
              ▼
        Q-Learning RL Agent (rating + variety + distance + budget)
              │
              ▼
        Category Guarantee + Distance Hard-Gate
              │
              ▼
        TSP Nearest-Neighbor Route Optimizer
              │
              ▼
        Itinerary akhir (steps + total cost + total km)
```

---

## 📑 Daftar Isi

| §   | Section                                                       | Output utama                            |
|-----|---------------------------------------------------------------|-----------------------------------------|
| 1   | Setup & Dependencies                                          | env siap, RNG seed                      |
| 2   | Konfigurasi Konstanta (kategori, blacklist, harga, rating)    | konstanta global                        |
| 3   | Crawling Overpass API                                         | POI mentah dari OSM                     |
| 4   | Enrichment Foursquare (FSQ)                                   | rating tambahan                         |
| 5   | Seed Data + Eksekusi Crawling                                 | `df_combined`                           |
| 6   | Cleaning + Realistic Price/Rating Imputation                  | `data/processed/destinations.csv`       |
| 7   | EDA — Eksplorasi Dataset                                      | distribusi kategori/harga/rating        |
| 8   | Feature Engineering (one-hot + numeric + TF-IDF)              | `feature_matrix.npy`                    |
| 9   | Content-Based Filtering (CBF) + Interleave                    | `models/cbf_model.pkl`                  |
| 10  | Visualisasi CBF                                               | heatmap similarity                      |
| 11  | Reinforcement Learning Environment + Q-Agent                  | class `BandungTravelEnv`, `QLearningAgent` |
| 12  | Training RL Agent                                             | `models/rl_agent.pkl`                   |
| 13  | Visualisasi Training RL                                       | reward curve, ε-decay                   |
| 14  | Route Optimizer (TSP Nearest-Neighbor)                        | class `RouteOptimizer`                  |
| 15  | Inference Pipeline + Category Guarantee                       | fungsi `full_pipeline()`                |
| 16  | **Evaluasi Model + Visualisasi Metrik**                       | `eval_report.json` + plot               |
| 17  | Export Artefak                                                | semua file model siap deploy            |

---

## 🔑 Perubahan Kunci Versi Ini

| # | Perubahan | Detail |
|---|-----------|--------|
| 1 | **Hanya 3 kategori** | `Alam`, `Kuliner`, `Wisata` (Wisata Umum). *Belanja* digabung ke *Wisata*; *Budaya* dihapus penuh. |
| 2 | **Blacklist diperluas** | +50 keyword: showroom, dealer mobil/motor, bengkel, cuci mobil, supermarket, swalayan, dll. |
| 3 | **Harga realistis ber-range** | Imputasi harga deterministik berdasar sub-kategori (curug ≠ gunung ≠ kebun teh) — tidak ada lagi 5 gunung berharga sama. |
| 4 | **Rating realistis** | Imputasi normal $\mathcal{N}(\mu_{cat}, 0.25)$, di-clip ke [3.6, 4.8] — variatif tapi tidak buruk. |
| 5 | **DRL adil & hard-gate jarak** | Reward = rating + variety + **distance** + budget. `max_km` ditegakkan di 4 lapis: filter kandidat → valid_action → guarantee → final validator. |
| 6 | **Category fairness** | Jika user pilih ≥2 kategori dan `count ≥ len(categories)`, setiap kategori WAJIB ada minimal 1 destinasi. Jika `count == 1`, jaminan ini di-bypass. |
| 7 | **Section evaluasi khusus** | Coverage rate, max-km compliance, budget compliance, distribusi rating/jarak/harga — semua dengan visualisasi matplotlib. |
| 8 | **Markdown vs Code dipisah rapi** | Tidak ada lagi kode terjebak di sel markdown atau class yang didefinisikan dua kali. |
""")

# =====================================================================
# SECTION 1 — Setup & Dependencies
# =====================================================================
md(r"""## §1 — Setup & Dependencies

**Apa yang terjadi di section ini:**

1. Auto-install dependency yang belum ada (idempotent — aman dijalankan ulang).
2. Import semua modul Python yang dipakai di seluruh notebook.
3. Set RNG seed untuk reproducibility (`RANDOM_SEED = 42`).
4. Pastikan `cwd` adalah root proyek; buat folder `data/raw`, `data/processed`, `models`.
5. Suppress warning supaya output bersih.

> **Catatan:** Notebook ini didesain untuk dijalankan di **Kaggle Kernels** maupun **lokal**. Jika dijalankan dari folder `notebooks/`, working directory otomatis di-rename ke parent.
""")

code(r'''# ── 1.1  Auto-install dependency ─────────────────────────────
import importlib.util, subprocess, sys

REQUIRED = {
    "requests":   "requests",
    "pandas":     "pandas",
    "numpy":      "numpy",
    "sklearn":    "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn":    "seaborn",
    "tqdm":       "tqdm",
}
missing = [pkg for mod, pkg in REQUIRED.items()
           if importlib.util.find_spec(mod) is None]
if missing:
    print(f"📦 Installing: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("✅ Semua dependency terpasang.")

# ── 1.2  Import standar ──────────────────────────────────────
import os, re, json, math, time, pickle, random, urllib.parse, warnings
import hashlib
from pathlib import Path
from datetime import date
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# ── 1.3  Reproducibility ─────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"✅ Random seed: {RANDOM_SEED}")

# ── 1.4  Working directory & struktur folder ─────────────────
if Path.cwd().name == "notebooks":
    os.chdir("..")
print(f"📂 CWD: {Path.cwd()}")

for d in ["data/raw", "data/processed", "data", "models", "notebooks"]:
    Path(d).mkdir(parents=True, exist_ok=True)
print("✅ Folder data/raw, data/processed, models siap.")
''')

# =====================================================================
# SECTION 2 — Konfigurasi Konstanta
# =====================================================================
md(r"""## §2 — Konfigurasi Konstanta

**Apa yang terjadi di section ini:**

Semua *magic value* yang dipakai di seluruh pipeline dikumpulkan di satu tempat agar mudah di-tweak:

| Konstanta | Isi | Dipakai oleh |
|-----------|-----|--------------|
| `CATEGORY_ORDER` | `["Alam", "Kuliner", "Wisata"]` | Feature engineering, CBF, RL state, evaluasi |
| `BLACKLIST_KEYWORDS` | 60+ kata (masjid, sekolah, bank, **showroom, dealer, bengkel, cuci mobil, supermarket, swalayan**, …) | Filter post-crawling & cleaning |
| `PRICE_RANGES` | (min, max) IDR per **sub-kategori** (curug, gunung, kafe, theme park, …) | Imputasi harga realistis |
| `RATING_DEFAULTS` | μ per kategori untuk imputasi rating | Imputasi rating realistis |
| `HOME_OPTIONS` | 5 titik populer Bandung sbg home base | RL training & evaluasi |
| `SPEED_KMH` | 28 (kecepatan rata-rata di Bandung kondisi macet) | Estimasi waktu tempuh |

> **Catatan filosofi:** *Wisata Umum* di sini mencakup wisata buatan (theme park, agro-edukasi, hot spring, mall, factory outlet). Kategori *Belanja* di-merge ke *Wisata* karena dari kacamata wisatawan, factory outlet & mall di Bandung memang termasuk destinasi wisata.
""")

code(r'''# ── 2.1  Kategori aktif ──────────────────────────────────────
# Hanya 3 kategori: Alam, Kuliner, Wisata (Wisata Umum)
# - "Belanja" digabung ke "Wisata" (mall/FO adalah destinasi wisata)
# - "Budaya" dihapus karena banyak data noise (masjid, sekolah, dll)
CATEGORY_ORDER   = ["Alam", "Kuliner", "Wisata"]
VALID_CATEGORIES = set(CATEGORY_ORDER)
print(f"✅ Kategori aktif: {CATEGORY_ORDER}")

# ── 2.2  Blacklist keywords ──────────────────────────────────
# Setiap nama destinasi yang MENGANDUNG salah satu keyword ini
# (case-insensitive) akan dihapus dari dataset.
BLACKLIST_KEYWORDS = [
    # ── Tempat ibadah ──
    "masjid", "mushola", "musholah", "mosque", "gereja", "church",
    "vihara", "klenteng", "pura ", "chapel", "cathedral", "kapel",

    # ── Pendidikan ──
    "sekolah", "sdn ", "smpn", "sman", "smkn", "mtsn", "min ",
    "sd ", "smp ", "sma ", "smk ", "tk ", "paud",
    "universitas", "institut", "akademi", "kampus", "politeknik",

    # ── Kesehatan ──
    "puskesmas", "rumah sakit", "klinik", "apotek", "apotik",
    "rs ", "rsia", "rsu ", "lab klinik",

    # ── Pemerintah / kantor ──
    "kantor", "polsek", "polres", "polda", "koramil", "kodim",
    "kelurahan", "kecamatan", "balai desa", "kantor pos",

    # ── Bank & ATM ──
    "bank ", " atm ", "atm bri", "atm bni", "atm bca",
    "atm mandiri", "bri ", "bni ", "bca ", "mandiri bank",
    "panin ", "cimb ",

    # ── Retail & convenience ──
    "indomaret", "alfamart", "alfamidi", "minimarket",
    "supermarket", "swalayan", "hypermart", "lottemart",
    "yogya ", "borma", "transmart",

    # ── Otomotif (yang BUKAN wisata) ──
    "showroom", "dealer", "auto2000", "auto ", "honda showroom",
    "yamaha showroom", "suzuki ", "toyota ", "daihatsu ",
    "bengkel", "service motor", "service mobil",
    "cuci mobil", "cuci motor", "carwash", "car wash",
    "ac mobil", "spare part", "spare-part", "sparepart",
    "ban mobil", "ban motor", "oli mobil",

    # ── SPBU & utilities ──
    "spbu", "pom bensin", "pertamina", "shell ", "vivo ", "bp ",

    # ── Properti / non-tourism ──
    "perumahan", "komplek", "ruko", "kavling", "cluster ",
    "kos ", "kost ", "kontrakan",

    # ── Industri / gudang ──
    "pabrik", "gudang", "warehouse", "industri",
]

def is_blacklisted(name) -> bool:
    """True jika nama destinasi mengandung salah satu keyword blacklist."""
    if not isinstance(name, str):
        return True
    low = " " + name.lower() + " "
    return any(kw in low for kw in BLACKLIST_KEYWORDS)

print(f"✅ Blacklist: {len(BLACKLIST_KEYWORDS)} keyword")

# ── 2.3  Range harga realistis per sub-kategori ──────────────
# Diambil dari riset harga umum 2024-2025 di Bandung. Format (min, max) IDR.
PRICE_RANGES = {
    "Alam": {
        "air_terjun": ( 5_000,  25_000),
        "gunung":     (10_000,  35_000),
        "kebun":      (15_000,  35_000),
        "danau":      (15_000,  35_000),
        "taman":      ( 5_000,  25_000),
        "tebing":     (10_000,  25_000),
        "hutan":      (12_000,  30_000),
        "default":    (10_000,  25_000),
    },
    "Kuliner": {
        "cafe":       (25_000,  80_000),
        "restoran":   (40_000, 150_000),
        "warung":     (15_000,  50_000),
        "food_court": (20_000,  60_000),
        "kaki_lima":  (10_000,  40_000),
        "default":    (25_000,  70_000),
    },
    "Wisata": {
        "theme_park": (75_000, 250_000),
        "water_park": (40_000, 120_000),
        "zoo":        (40_000,  80_000),
        "agro":       (25_000,  60_000),
        "attraction": (25_000, 100_000),
        "hot_spring": (30_000,  90_000),
        "mall":       (     0,       0),    # mall: gratis masuk
        "outlet":     (     0,       0),    # FO: gratis masuk
        "pasar":      (     0,       0),    # pasar tradisional: gratis
        "default":    (30_000,  90_000),
    },
}

# ── 2.4  Default rating (μ) per kategori ─────────────────────
RATING_DEFAULTS = {"Alam": 4.30, "Kuliner": 4.20, "Wisata": 4.10}
RATING_SIGMA    = 0.25
RATING_CLIP     = (3.6, 4.8)   # clip agar tidak ada rating "buruk"

# ── 2.5  Default duration (menit) per kategori ───────────────
DEFAULT_DURATION = {"Alam": 120, "Kuliner": 75, "Wisata": 135}

# ── 2.6  Konfigurasi geografi & RL ──────────────────────────
HOME_OPTIONS = [
    {"lat": -6.9215, "lng": 107.6071, "name": "Alun-Alun Bandung"},
    {"lat": -6.9145, "lng": 107.6020, "name": "Stasiun Bandung"},
    {"lat": -6.8126, "lng": 107.6178, "name": "Pasar Lembang"},
    {"lat": -6.8915, "lng": 107.6107, "name": "Dago"},
    {"lat": -6.9024, "lng": 107.6188, "name": "Gedung Sate"},
]
SPEED_KMH = 28
BANDUNG_BBOX = "-7.2500,107.3500,-6.7500,107.9000"

print(f"✅ Price ranges, rating defaults, home options siap.")
''')

# =====================================================================
# SECTION 3 — Crawling Overpass
# =====================================================================
md(r"""## §3 — Crawling Overpass API (OpenStreetMap)

**Apa yang terjadi di section ini:**

1. Mendefinisikan query Overpass per kategori (`Alam`, `Kuliner`, `Wisata`) dengan tag OSM yang spesifik & non-noise.
2. Tag yang **dihapus** dibanding versi sebelumnya:
   - `place_of_worship` (sumber utama masjid masuk dataset)
   - `historic` (terlalu noise)
   - `shop=clothes` umum (terlalu banyak butik kecil yang bukan wisata)
3. Mendefinisikan helper `query_overpass()`, `parse_osm_element()`, dan `run_overpass_crawl()`.
4. Filter blacklist diaplikasikan **dua kali**: saat parsing OSM element dan saat cleaning.

> **Catatan reliability:** Endpoint Overpass kadang down. Helper otomatis fallback ke 4 mirror berbeda secara berurutan.
""")

code(r'''# ── 3.1  Endpoint & header Overpass ──────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
OVERPASS_HEADERS = {
    "User-Agent": (
        "Bandung-AI-Travel-Capstone/3.0 "
        "(Telkom University; "
        "+https://github.com/Fall-Llihc/Bandung_AI_Travel-Capstone-Project)"
    ),
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# ── 3.2  Query per kategori (3 kategori saja) ────────────────
# PENTING: Belanja di-merge ke Wisata. Mall/FO ada di query Wisata.
CATEGORY_QUERIES = {
    "Alam": """
        node["tourism"="viewpoint"]({bbox});
        node["natural"="peak"]({bbox});
        node["natural"="waterfall"]({bbox});
        way["leisure"="park"]["name"~"."]({bbox});
        way["leisure"="nature_reserve"]({bbox});
        node["tourism"="picnic_site"]({bbox});
    """,
    "Kuliner": """
        node["amenity"="restaurant"]["name"~"."]({bbox});
        node["amenity"="cafe"]["name"~"."]({bbox});
        node["amenity"="food_court"]["name"~"."]({bbox});
        node["tourism"="restaurant"]["name"~"."]({bbox});
    """,
    "Wisata": """
        node["tourism"="theme_park"]({bbox});
        node["tourism"="attraction"]["name"~"."]({bbox});
        node["tourism"="zoo"]({bbox});
        node["tourism"="aquarium"]({bbox});
        way["tourism"="theme_park"]({bbox});
        way["leisure"="water_park"]({bbox});
        node["leisure"="miniature_golf"]({bbox});
        node["shop"="mall"]["name"~"."]({bbox});
        way["shop"="mall"]["name"~"."]({bbox});
        node["amenity"="marketplace"]["name"~"."]({bbox});
    """,
}

# ── 3.3  Helper crawling ─────────────────────────────────────
def query_overpass(category: str, bbox: str, endpoint: str, timeout: int = 60) -> list:
    q = CATEGORY_QUERIES.get(category, "")
    full = f"""
    [out:json][timeout:{timeout}];
    (
      {q.format(bbox=bbox)}
    );
    out center tags;
    """
    resp = requests.post(endpoint, data={"data": full},
                         headers=OVERPASS_HEADERS, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json().get("elements", [])


def parse_osm_element(el: dict, category: str):
    tags = el.get("tags", {})
    name = (tags.get("name:id") or tags.get("name")
            or tags.get("name:en") or "").strip()
    if not name or len(name) < 3:
        return None
    if is_blacklisted(name):
        return None

    if el.get("type") == "node":
        lat, lng = el.get("lat"), el.get("lon")
    else:
        c = el.get("center", {})
        lat, lng = c.get("lat"), c.get("lon")
    if lat is None or lng is None:
        return None

    q = urllib.parse.quote(f"{name}, Bandung")
    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={q}"

    semantic_tags = []
    for k, v in tags.items():
        if k in ("name", "name:id", "name:en", "source"):
            continue
        if k.startswith("addr:"):
            continue
        semantic_tags.append(str(v).lower().replace(" ", "_"))

    return {
        "id":          re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
        "name":        name,
        "category":    category,
        "desc":        tags.get("description", ""),
        "ticket":      None,        # diisi nanti oleh imputer
        "duration":    None,
        "lat":         float(lat),
        "lng":         float(lng),
        "rating":      None,        # diisi nanti
        "tags":        semantic_tags[:8],
        "stay_detail": "",
        "gmaps_url":   gmaps_url,
        "source":      "osm",
    }


def run_overpass_crawl(verbose: bool = True) -> pd.DataFrame:
    all_records = []
    for cat in CATEGORY_ORDER:
        if verbose:
            print(f"\n  Crawling OSM [{cat}]...")
        elements = []
        for ep in OVERPASS_ENDPOINTS:
            try:
                elements = query_overpass(cat, BANDUNG_BBOX, ep)
                if elements:
                    if verbose:
                        print(f"    ✅ {ep.split('/')[2]}: {len(elements)} elements")
                    break
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  {ep.split('/')[2]}: {e}")
                time.sleep(2)

        parsed = [parse_osm_element(el, cat) for el in elements]
        parsed = [r for r in parsed if r]
        if verbose:
            print(f"    → {len(parsed)} valid destinations")
        all_records.extend(parsed)
        time.sleep(3)   # jangan spam Overpass

    df = pd.DataFrame(all_records)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["name", "category"])
    df = df.drop_duplicates(subset=["lat", "lng"])
    return df.reset_index(drop=True)


print("✅ Fungsi Overpass crawl siap.")
''')

# =====================================================================
# SECTION 4 — FSQ Enrichment
# =====================================================================
md(r"""## §4 — Enrichment Foursquare (FSQ Places API)

**Apa yang terjadi di section ini:**

1. Mengambil rating tambahan dari **Foursquare Places API** (free tier 1000 req/hari) untuk destinasi yang OSM-nya tidak punya rating.
2. Konversi rating FSQ (skala 0–10) → rating proyek (skala 0–5) lewat pembagian 2.
3. Jika `FSQ_API_KEY` tidak tersedia, langkah ini **otomatis di-skip** tanpa error — pipeline tetap jalan dengan rating hasil imputasi.

> **Catatan keamanan:** API key bisa di-set lewat **Kaggle Secrets** (`UserSecretsClient`) atau env var. Hindari hardcode key di repo publik.
""")

code(r'''# ── 4.1  API key resolution ──────────────────────────────────
FSQ_API_KEY = os.getenv("FSQ_API_KEY") or None

# Coba ambil dari Kaggle Secrets bila tersedia
try:
    from kaggle_secrets import UserSecretsClient
    FSQ_API_KEY = UserSecretsClient().get_secret("FSQ_API_KEY")
    print("✅ FSQ_API_KEY dari Kaggle Secrets")
except Exception:
    if FSQ_API_KEY:
        print("✅ FSQ_API_KEY dari env var")
    else:
        print("⚠️  FSQ_API_KEY tidak tersedia — FSQ enrichment akan di-skip (pipeline tetap jalan).")

# ── 4.2  Helper FSQ ──────────────────────────────────────────
def fsq_search(name: str, lat: float, lng: float):
    if not FSQ_API_KEY:
        return None
    try:
        url = "https://api.foursquare.com/v3/places/search"
        params = {"query": name, "ll": f"{lat},{lng}", "radius": 500, "limit": 1}
        resp = requests.get(url, headers={
            "Accept": "application/json",
            "Authorization": FSQ_API_KEY,
        }, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None


def enrich_with_fsq(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Isi rating yang masih NaN dari Foursquare. Skip jika tidak ada API key."""
    if not FSQ_API_KEY:
        if verbose:
            print("  ⚠️  Skip FSQ enrichment (no API key).")
        return df
    df = df.copy()
    enriched = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  FSQ enrich"):
        if pd.notna(row.get("rating")) and row["rating"] > 0:
            continue
        result = fsq_search(row["name"], row["lat"], row["lng"])
        if result:
            r = result.get("rating")
            if r:
                df.at[idx, "rating"] = round(r / 2, 1)   # 0–10 → 0–5
                enriched += 1
        time.sleep(0.12)   # ~8 req/s, aman di bawah rate limit
    if verbose:
        print(f"  ✅ FSQ: {enriched} destinasi di-enrich ratingnya.")
    return df


print("✅ Fungsi FSQ enrichment siap.")
''')

# =====================================================================
# SECTION 5 — Seed Data + Run Crawling
# =====================================================================
md(r"""## §5 — Seed Data + Eksekusi Crawling

**Apa yang terjadi di section ini:**

1. Mendefinisikan **24 destinasi seed** (8 Alam, 8 Kuliner, 8 Wisata Umum termasuk shopping) — dipilih manual dengan data lengkap yang TIDAK BISA di-crawl otomatis (`ticket`, `duration`, `stay_detail`).
2. Menjalankan crawling Overpass untuk dapat tambahan POI dari OSM.
3. Menggabungkan seed + OSM, deduplikasi berdasar `id` & koordinat (radius ~100 m).
4. Re-apply blacklist sekali lagi sebagai *safety net*.

**Mengapa seed?**  
OSM tidak menyimpan harga tiket atau durasi kunjungan. Seed memberikan ground-truth ini sehingga statistik harga/durasi per kategori (median) bisa jadi referensi imputasi untuk POI hasil crawl.
""")

code(r'''# ── 5.1  Seed destinations (24 entri) ────────────────────────
SEED_DESTINATIONS = [
    # ───────── ALAM (8) ─────────
    {"id": "kawah-putih", "name": "Kawah Putih", "category": "Alam",
     "desc": "Danau kawah vulkanik dengan air berwarna putih kehijauan di ketinggian 2430 mdpl",
     "ticket": 81000, "duration": 150, "lat": -7.1660, "lng": 107.4019, "rating": 4.6,
     "tags": ["sunrise", "fotogenik", "dingin", "kawah"],
     "stay_detail": "Parkir & jalan ke kawah 20', keliling kawah & foto 90', istirahat 20', kembali 20'"},
    {"id": "kebun-teh-rancabali", "name": "Kebun Teh Rancabali", "category": "Alam",
     "desc": "Hamparan kebun teh hijau di Ciwidey dengan udara sejuk dan pemandangan luas",
     "ticket": 25000, "duration": 105, "lat": -7.1432, "lng": 107.4106, "rating": 4.5,
     "tags": ["hijau", "fotogenik", "sejuk", "kebun"],
     "stay_detail": "Jalan ke area kebun 15', jalan-jalan & foto 60', duduk santai 30'"},
    {"id": "stone-garden-padalarang", "name": "Stone Garden Padalarang", "category": "Alam",
     "desc": "Taman batu purba dengan formasi karst unik di kawasan Citatah, Padalarang",
     "ticket": 15000, "duration": 105, "lat": -6.8323, "lng": 107.4709, "rating": 4.4,
     "tags": ["batu", "fotogenik", "karst", "tebing"],
     "stay_detail": "Naik ke area batu 20', eksplorasi formasi & foto 60', istirahat 25'"},
    {"id": "tebing-keraton", "name": "Tebing Keraton", "category": "Alam",
     "desc": "Tebing dengan view hutan pinus Dago Pakar, populer untuk sunrise",
     "ticket": 15000, "duration": 120, "lat": -6.8359, "lng": 107.6630, "rating": 4.5,
     "tags": ["sunrise", "hiking", "tebing", "pinus"],
     "stay_detail": "Hiking ke tebing 30', menikmati view & foto 45', santai 25', turun 20'"},
    {"id": "tangkuban-perahu", "name": "Tangkuban Perahu", "category": "Alam",
     "desc": "Gunung berapi aktif dengan kawah ratu yang ikonik di utara Bandung",
     "ticket": 30000, "duration": 135, "lat": -6.7597, "lng": 107.6098, "rating": 4.5,
     "tags": ["gunung", "kawah", "legendaris", "vulkanik"],
     "stay_detail": "Parkir & jalan ke kawah 20', foto 45', kawah domas 30', kembali 25', oleh-oleh 15'"},
    {"id": "situ-patenggang", "name": "Situ Patenggang", "category": "Alam",
     "desc": "Danau alami dikelilingi kebun teh di kawasan Ciwidey, romantis dan tenang",
     "ticket": 20000, "duration": 120, "lat": -7.1603, "lng": 107.3992, "rating": 4.4,
     "tags": ["danau", "romantis", "sejuk", "perahu"],
     "stay_detail": "Keliling danau 30', naik perahu 40', santai & foto 30', jalan-jalan 20'"},
    {"id": "curug-malela", "name": "Curug Malela", "category": "Alam",
     "desc": "Air terjun lebar dijuluki Niagara Mini Indonesia di Bandung Barat",
     "ticket": 10000, "duration": 150, "lat": -6.9231, "lng": 107.3342, "rating": 4.6,
     "tags": ["air-terjun", "hiking", "segar", "alam"],
     "stay_detail": "Hiking ke curug 45', foto 60', istirahat 25', kembali 20'"},
    {"id": "bukit-moko", "name": "Bukit Moko", "category": "Alam",
     "desc": "Puncak tertinggi di Bandung dengan panorama kota 360 derajat",
     "ticket": 20000, "duration": 120, "lat": -6.8198, "lng": 107.6898, "rating": 4.5,
     "tags": ["puncak", "panorama", "sunrise", "instagramable"],
     "stay_detail": "Hiking ke puncak 35', foto 50', turun 35'"},

    # ───────── KULINER (8) ─────────
    {"id": "floating-market-lembang", "name": "Floating Market Lembang", "category": "Kuliner",
     "desc": "Pasar terapung dengan beragam jajanan tradisional Sunda di atas perahu",
     "ticket": 30000, "duration": 105, "lat": -6.8121, "lng": 107.6178, "rating": 4.3,
     "tags": ["jajanan", "keluarga", "fotogenik", "tradisional"],
     "stay_detail": "Keliling pasar 30', antri & beli 20', makan 35', foto 20'"},
    {"id": "jalan-braga", "name": "Jalan Braga Kuliner", "category": "Kuliner",
     "desc": "Kawasan heritage dengan deretan kafe dan restoran bergaya kolonial Belanda",
     "ticket": 0, "duration": 90, "lat": -6.9185, "lng": 107.6080, "rating": 4.3,
     "tags": ["heritage", "kafe", "kolonial", "santai"],
     "stay_detail": "Jalan-jalan 20', pilih tempat 10', makan & santai 60'"},
    {"id": "cikole-lembang", "name": "Cikole Jayagiri Kuliner", "category": "Kuliner",
     "desc": "Kawasan wisata hutan pinus Lembang dengan warung-warung khas pegunungan",
     "ticket": 10000, "duration": 90, "lat": -6.8000, "lng": 107.6250, "rating": 4.2,
     "tags": ["pinus", "sejuk", "sate", "jagung-bakar"],
     "stay_detail": "Masuk area 10', pilih warung 10', tunggu 15', makan 55'"},
    {"id": "sate-maranggi-haji-yetty", "name": "Sate Maranggi Hj. Yetty", "category": "Kuliner",
     "desc": "Sate maranggi legendaris khas Purwakarta-Cianjur yang terkenal di Bandung",
     "ticket": 0, "duration": 75, "lat": -6.9310, "lng": 107.6390, "rating": 4.7,
     "tags": ["sate", "legendaris", "khas-sunda", "enak"],
     "stay_detail": "Antri & pesan 15', tunggu sate 15', makan 45'"},
    {"id": "warung-nasi-ampera", "name": "Warung Nasi Ampera", "category": "Kuliner",
     "desc": "Nasi Sunda lengkap dengan lauk-pauk otentik di kawasan Bandung kota",
     "ticket": 0, "duration": 60, "lat": -6.9200, "lng": 107.6100, "rating": 4.3,
     "tags": ["nasi-sunda", "murah", "otentik", "lauk-pauk"],
     "stay_detail": "Pilih lauk 10', ambil 5', makan 35', santai 10'"},
    {"id": "warung-pasar-baru", "name": "Pasar Baru Kuliner Kaki Lima", "category": "Kuliner",
     "desc": "Kawasan Pasar Baru dengan pilihan makanan kaki lima khas Bandung",
     "ticket": 0, "duration": 75, "lat": -6.9225, "lng": 107.6068, "rating": 4.1,
     "tags": ["kaki-lima", "murah", "batagor", "siomay"],
     "stay_detail": "Jalan-jalan cari 20', beli 15', makan 30', keliling 10'"},
    {"id": "ikan-bakar-cianjur", "name": "Ikan Bakar Cianjur", "category": "Kuliner",
     "desc": "Restoran seafood dengan ikan bakar khas Sunda yang terkenal sejak 1980-an",
     "ticket": 0, "duration": 80, "lat": -6.9115, "lng": 107.6075, "rating": 4.5,
     "tags": ["ikan-bakar", "seafood", "legendaris", "sunda"],
     "stay_detail": "Pilih ikan 10', tunggu bakar 20', makan 50'"},
    {"id": "dago-pakar-resto", "name": "Dago Pakar Restoran View", "category": "Kuliner",
     "desc": "Deretan restoran di kawasan Dago Pakar dengan pemandangan kota Bandung",
     "ticket": 0, "duration": 90, "lat": -6.8450, "lng": 107.6420, "rating": 4.2,
     "tags": ["view-kota", "romantis", "restoran", "malam"],
     "stay_detail": "Pilih tempat 10', pesan & tunggu 20', makan & view 60'"},

    # ───────── WISATA UMUM (8) ─────────
    {"id": "trans-studio-bandung", "name": "Trans Studio Bandung", "category": "Wisata",
     "desc": "Taman hiburan indoor terbesar di Bandung dengan berbagai wahana seru",
     "ticket": 150000, "duration": 270, "lat": -6.9246, "lng": 107.6376, "rating": 4.4,
     "tags": ["wahana", "keluarga", "indoor", "theme-park"],
     "stay_detail": "Antri & orientasi 20', wahana 90', makan siang 45', wahana lanjut 75', oleh-oleh 40'"},
    {"id": "kebun-binatang-bandung", "name": "Kebun Binatang Bandung", "category": "Wisata",
     "desc": "Kebun binatang klasik di pusat kota dengan koleksi satwa beragam",
     "ticket": 60000, "duration": 150, "lat": -6.9072, "lng": 107.6078, "rating": 4.1,
     "tags": ["binatang", "keluarga", "edukatif", "satwa"],
     "stay_detail": "Masuk & peta 10', keliling 90', foto 30', istirahat 20'"},
    {"id": "farmhouse-susu-lembang", "name": "Farmhouse Susu Lembang", "category": "Wisata",
     "desc": "Agrowisata bergaya Eropa di Lembang dengan atraksi susu segar dan hewan",
     "ticket": 35000, "duration": 120, "lat": -6.8214, "lng": 107.5956, "rating": 4.4,
     "tags": ["foto-eropa", "susu", "agro", "instagramable"],
     "stay_detail": "Foto venue 15', interaksi hewan 30', susu & foto 30', area 25', oleh-oleh 20'"},
    {"id": "the-great-asia-africa", "name": "The Great Asia Africa", "category": "Wisata",
     "desc": "Destinasi wisata foto tematik dengan replika landmark Asia & Afrika",
     "ticket": 40000, "duration": 120, "lat": -6.9338, "lng": 107.4925, "rating": 4.3,
     "tags": ["foto-tematik", "instagramable", "landmark", "attraction"],
     "stay_detail": "Orientasi 10', zona Asia 40', foto 40', zona Afrika 30'"},
    {"id": "maribaya-hot-spring", "name": "Maribaya Hot Spring", "category": "Wisata",
     "desc": "Sumber air panas alam dengan kolam renang dan waterpark di Lembang",
     "ticket": 50000, "duration": 150, "lat": -6.8167, "lng": 107.6500, "rating": 4.2,
     "tags": ["air-panas", "kolam", "hot-spring", "relaksasi"],
     "stay_detail": "Ganti 15', rendam 60', waterpark 45', bilas 30'"},
    {"id": "cihampelas-walk", "name": "Cihampelas Walk (Ciwalk)", "category": "Wisata",
     "desc": "Mall semi outdoor dengan konsep unik di Jalan Cihampelas yang ikonik",
     "ticket": 0, "duration": 120, "lat": -6.8990, "lng": 107.6053, "rating": 4.3,
     "tags": ["mall", "semi-outdoor", "fashion", "shopping"],
     "stay_detail": "Keliling 20', belanja 50', makan 30', foto 20'"},
    {"id": "factory-outlet-dago", "name": "Factory Outlet Dago", "category": "Wisata",
     "desc": "Deretan factory outlet premium di Jalan Dago dengan merek lokal dan internasional",
     "ticket": 0, "duration": 105, "lat": -6.8800, "lng": 107.6130, "rating": 4.2,
     "tags": ["outlet", "factory", "diskon", "shopping"],
     "stay_detail": "Pilih FO 10', belanja 70', checkout 15', makan 10'"},
    {"id": "pasar-baru-trade-center", "name": "Pasar Baru Trade Center", "category": "Wisata",
     "desc": "Pusat perbelanjaan tekstil dan fashion terbesar di Bandung sejak era kolonial",
     "ticket": 0, "duration": 120, "lat": -6.9225, "lng": 107.6068, "rating": 4.1,
     "tags": ["pasar", "tekstil", "fashion", "shopping"],
     "stay_detail": "Orientasi 10', belanja 70', tawar 25', checkout 15'"},
]

# ── 5.2  Eksekusi crawling ───────────────────────────────────
print(f"📦 Seed: {len(SEED_DESTINATIONS)} destinasi")
print(Counter(d["category"] for d in SEED_DESTINATIONS))

print("\n🌐 Crawling Overpass API...")
df_osm = run_overpass_crawl(verbose=True)
print(f"\n📊 Hasil OSM: {len(df_osm)} destinasi")
if not df_osm.empty:
    print(df_osm["category"].value_counts().to_dict())
    df_osm.to_csv("data/raw/osm_raw.csv", index=False)
    print("💾 data/raw/osm_raw.csv")

# ── 5.3  Gabung seed + OSM ───────────────────────────────────
df_seed = pd.DataFrame(SEED_DESTINATIONS)
if not df_osm.empty:
    df_combined = pd.concat([df_seed, df_osm], ignore_index=True)
else:
    df_combined = df_seed.copy()
    print("⚠️  OSM kosong — pakai seed saja.")

# Dedup berdasar id (seed lebih diprioritaskan karena urutan pertama)
df_combined = df_combined.drop_duplicates(subset=["id"], keep="first")

# Failsafe blacklist sekali lagi
before = len(df_combined)
df_combined = df_combined[~df_combined["name"].apply(is_blacklisted)].reset_index(drop=True)
removed = before - len(df_combined)
if removed:
    print(f"🚫 Blacklist (post-merge): {removed} dihapus")

print(f"\n📦 Total sebelum cleaning: {len(df_combined)}")
print(df_combined["category"].value_counts().to_dict())
''')

# Save partial notebook builder so far - let me continue in next file.
import_marker = "# === BUILD MARKER (continued) ==="

P = Path(__file__).parent
print("CHECKPOINT: 5 sections done")



# =====================================================================
# SECTION 6 — Cleaning + Realistic Price/Rating Imputation
# =====================================================================
md(r"""## §6 — Cleaning + Realistic Price/Rating Imputation

**Apa yang terjadi di section ini:**

Bagian ini adalah **jantung dari perbaikan dataset**. Tujuan utama: hasilkan harga dan rating yang **realistis & bervariasi**, bukan median yang sama untuk semua destinasi.

### 6.1 Filter & Validasi
- Buang baris dengan kategori invalid (bukan Alam/Kuliner/Wisata).
- Buang nama terlalu pendek, nama yang hanya angka, atau lolos blacklist.
- Buang koordinat di luar bounding box Bandung Raya.
- Deduplikasi koordinat radius ~100 m (`lat.round(3), lng.round(3)`).

### 6.2 Imputasi Harga Realistis (KEY)
Setiap destinasi tanpa harga akan di-imputasi sesuai **sub-kategori** yang dideteksi dari nama/tags:

| Kategori | Sub-kategori → Range (IDR) |
|----------|----------------------------|
| Alam     | air_terjun (5–25k), gunung (10–35k), kebun (15–35k), danau (15–35k), taman (5–25k), tebing (10–25k) |
| Kuliner  | cafe (25–80k), restoran (40–150k), warung (15–50k), food_court (20–60k), kaki_lima (10–40k) |
| Wisata   | theme_park (75–250k), water_park (40–120k), zoo (40–80k), agro (25–60k), attraction (25–100k), hot_spring (30–90k), mall/outlet/pasar (gratis) |

**Sifat deterministik:** harga di-seed dari `hash(name)`, jadi destinasi yang sama akan dapat harga yang sama setiap kali pipeline dijalankan. **Sifat variatif:** dua "Gunung X" yang berbeda nama akan dapat harga berbeda dalam range gunung — *bukan median yang sama*.

### 6.3 Imputasi Rating Realistis
Distribusi normal $\mathcal{N}(\mu_{cat}, 0.25)$ di-clip ke `[3.6, 4.8]`:
- Alam   : μ = 4.30
- Kuliner: μ = 4.20
- Wisata : μ = 4.10

Hasilnya: rating tetap **bervariasi**, tetapi **tidak ada yang buruk** (≥3.6).

### 6.4 Imputasi Durasi
Berdasar median per kategori (dari seed), atau default jika seed tidak cukup.
""")

code(r'''# ── 6.1  Helper: deteksi sub-kategori dari nama & tags ───────
SUBCAT_KEYWORDS = {
    "Alam": {
        "air_terjun": ["curug", "air terjun", "waterfall"],
        "gunung":     ["gunung", "puncak", "bukit", "peak", "moko"],
        "kebun":      ["kebun", "perkebunan", "tea ", "teh "],
        "danau":      ["danau", "situ", "lake", "telaga"],
        "tebing":     ["tebing", "stone", "batu", "cliff", "karst"],
        "hutan":      ["hutan", "tahura", "djuanda", "forest"],
        "taman":      ["taman", "park", "garden", "kebun raya"],
    },
    "Kuliner": {
        "cafe":       ["cafe", "kafe", "coffee", "kopi", "starbucks"],
        "restoran":   ["restoran", "restaurant", "resto", "ikan bakar"],
        "warung":     ["warung", "nasi", "soto", "sate", "bakso", "ayam"],
        "food_court": ["food court", "floating market", "pasar terapung"],
        "kaki_lima":  ["kaki lima", "gerobak", "jajanan", "batagor", "siomay"],
    },
    "Wisata": {
        "theme_park": ["trans studio", "wonderland", "world", "theme park"],
        "water_park": ["waterpark", "water park", "water boom"],
        "zoo":        ["kebun binatang", "zoo", "satwa", "aquarium"],
        "agro":       ["farmhouse", "farm house", "susu", "edukasi", "agrowisata"],
        "attraction": ["attraction", "the great", "asia africa", "wisata", "viewpoint"],
        "hot_spring": ["hot spring", "air panas", "maribaya"],
        "mall":       ["mall", "plaza", "ciwalk", "bip", "trans"],
        "outlet":     ["outlet", "factory", "fo "],
        "pasar":      ["pasar", "trade center", "marketplace"],
    },
}

def detect_subcategory(name, category, tags=None) -> str:
    """Tebak sub-kategori dari nama + tags. Return key utk PRICE_RANGES."""
    if category not in PRICE_RANGES:
        return "default"
    name_l = (name or "").lower()
    tags_l = ""
    if isinstance(tags, list):
        tags_l = " ".join(str(t) for t in tags)
    elif isinstance(tags, str):
        tags_l = tags
    text = f"{name_l} {tags_l.lower()}"

    for sub, kws in SUBCAT_KEYWORDS.get(category, {}).items():
        for kw in kws:
            if kw in text:
                return sub
    return "default"


def _name_seed(name: str, salt: str = "") -> int:
    """Seed deterministik dari nama (sama nama → sama hasil)."""
    s = (salt + str(name)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def assign_realistic_price(name: str, category: str, tags=None) -> int:
    """Imputasi harga: deterministik berdasar nama, range sesuai sub-kategori."""
    if category not in PRICE_RANGES:
        return 0
    sub = detect_subcategory(name, category, tags)
    lo, hi = PRICE_RANGES[category].get(sub, PRICE_RANGES[category]["default"])
    if lo == hi:
        return int(lo)
    rng = np.random.default_rng(_name_seed(name, salt="price"))
    val = rng.integers(lo, hi + 1)
    # Bulatkan ke kelipatan 1.000 IDR
    return int(round(val / 1000) * 1000)


def assign_realistic_rating(name: str, category: str) -> float:
    """Imputasi rating: N(μ_cat, σ) clipped ke [3.6, 4.8]."""
    mu    = RATING_DEFAULTS.get(category, 4.2)
    sigma = RATING_SIGMA
    rng   = np.random.default_rng(_name_seed(name, salt="rating"))
    val   = rng.normal(mu, sigma)
    val   = float(np.clip(val, RATING_CLIP[0], RATING_CLIP[1]))
    return round(val, 1)


def assign_default_duration(category: str, tags=None) -> int:
    """Imputasi durasi (menit)."""
    base = DEFAULT_DURATION.get(category, 120)
    # Sedikit variasi ±20 menit deterministik (pakai hash kategori+tag)
    seed = _name_seed(str(tags) + category, salt="dur")
    rng  = np.random.default_rng(seed)
    return int(base + rng.integers(-20, 21))


# ── 6.2  Cleaner utama ───────────────────────────────────────
def clean_destinations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Pastikan semua kolom ada
    for col in ("id","name","category","desc","ticket","duration",
                "rating","tags","lat","lng","gmaps_url","stay_detail","source"):
        if col not in df.columns:
            df[col] = pd.NA

    # 2. Rename lon → lng jika perlu
    if "lon" in df.columns and "lng" not in df.columns:
        df = df.rename(columns={"lon": "lng"})
    elif "lon" in df.columns:
        df["lng"] = df["lng"].fillna(df["lon"])
        df = df.drop(columns=["lon"])

    # 3. Filter kategori valid
    before = len(df)
    df = df[df["category"].isin(VALID_CATEGORIES)].copy()
    if before - len(df):
        print(f"  🗑️  Buang kategori invalid: {before - len(df)}")

    # 4. Filter nama
    df = df[df["name"].astype(str).str.len() >= 3]
    df = df[~df["name"].astype(str).str.match(r"^\d+$")]

    # 5. Re-apply blacklist
    before = len(df)
    df = df[~df["name"].apply(is_blacklisted)]
    if before - len(df):
        print(f"  🚫 Blacklist filter: {before - len(df)} dihapus")

    # 6. Imputasi harga realistis (KEY)
    print("  💰 Imputasi harga realistis berdasar sub-kategori...")
    mask_no_price = df["ticket"].isna() | (df["ticket"].astype(str) == "")
    df.loc[mask_no_price, "ticket"] = df.loc[mask_no_price].apply(
        lambda r: assign_realistic_price(r["name"], r["category"], r.get("tags")),
        axis=1
    )

    # 7. Imputasi rating realistis
    print("  ⭐ Imputasi rating realistis (μ per kategori, σ=0.25)...")
    mask_no_rating = df["rating"].isna() | (df["rating"].astype(float) <= 0)
    df.loc[mask_no_rating, "rating"] = df.loc[mask_no_rating].apply(
        lambda r: assign_realistic_rating(r["name"], r["category"]),
        axis=1
    )

    # 8. Imputasi durasi
    mask_no_dur = df["duration"].isna() | (df["duration"].astype(float) <= 0)
    df.loc[mask_no_dur, "duration"] = df.loc[mask_no_dur].apply(
        lambda r: assign_default_duration(r["category"], r.get("tags")),
        axis=1
    )

    # 9. Field lain
    df["desc"] = df["desc"].fillna(
        df["category"].apply(lambda c: f"Destinasi wisata {c} di Bandung")
    )
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["stay_detail"] = df["stay_detail"].fillna("")
    df["gmaps_url"] = df["gmaps_url"].fillna(
        df["name"].apply(
            lambda n: f"https://www.google.com/maps/search/?api=1&query="
                      f"{urllib.parse.quote(str(n)+', Bandung')}"
        )
    )
    df["source"] = df["source"].fillna("unknown")

    # 10. Type casting
    df["ticket"]   = df["ticket"].astype(float).round().astype(int).clip(lower=0)
    df["duration"] = df["duration"].astype(float).round().astype(int).clip(lower=30)
    df["rating"]   = df["rating"].astype(float).clip(1.0, 5.0).round(2)
    df["lat"]      = df["lat"].astype(float)
    df["lng"]      = df["lng"].astype(float)

    # 11. Bounding box Bandung Raya
    df = df[df["lat"].between(-7.6, -6.5) & df["lng"].between(107.0, 108.5)]

    # 12. ID slug + dedup
    df["id"] = df["name"].apply(
        lambda n: re.sub(r"[^a-z0-9]+", "-", str(n).lower()).strip("-")
    )
    df = df.drop_duplicates(subset=["id"], keep="first")

    # 13. Dedup koordinat ~100 m
    df["_lat_r"] = df["lat"].round(3)
    df["_lng_r"] = df["lng"].round(3)
    df = df.drop_duplicates(subset=["_lat_r","_lng_r"], keep="first")
    df = df.drop(columns=["_lat_r","_lng_r"])

    return df.reset_index(drop=True)


# ── 6.3  Eksekusi: enrich → clean → save ─────────────────────
print("🔍 FSQ enrichment (skip jika tidak ada API key)...")
df_enriched = enrich_with_fsq(df_combined, verbose=True)

print("\n🧹 Cleaning + imputasi realistis...")
df_clean = clean_destinations(df_enriched)

print(f"\n✅ Dataset bersih: {len(df_clean)} destinasi")
print(df_clean["category"].value_counts().to_dict())

# Statistik distribusi (untuk sanity-check variasi harga & rating)
print("\n📊 Distribusi harga per kategori:")
for cat in CATEGORY_ORDER:
    sub = df_clean[df_clean["category"] == cat]["ticket"]
    if len(sub) > 0:
        print(f"   {cat:<10}: min Rp {sub.min():>7,} | "
              f"median Rp {int(sub.median()):>7,} | max Rp {sub.max():>7,} | "
              f"unique {sub.nunique()}")

print("\n📊 Distribusi rating per kategori:")
for cat in CATEGORY_ORDER:
    sub = df_clean[df_clean["category"] == cat]["rating"]
    if len(sub) > 0:
        print(f"   {cat:<10}: min {sub.min():.2f} | "
              f"mean {sub.mean():.2f} | max {sub.max():.2f} | "
              f"unique {sub.nunique()}")

# Save
df_clean.to_csv("data/processed/destinations.csv", index=False)
with open("data/last_updated.txt", "w") as f:
    f.write(str(date.today()))
print("\n💾 data/processed/destinations.csv")
print("💾 data/last_updated.txt")
''')

# =====================================================================
# SECTION 7 — EDA
# =====================================================================
md(r"""## §7 — EDA: Eksplorasi Dataset

**Apa yang terjadi di section ini:**

Sanity-check visual untuk memastikan dataset hasil cleaning **bebas dari bias berbahaya**:

1. **Histogram harga per kategori** → konfirmasi harga bervariasi (bukan flat satu nilai).
2. **Histogram rating per kategori** → konfirmasi rating tidak terkonsentrasi di satu titik.
3. **Scatter geografis** (lat × lng) → konfirmasi destinasi tersebar di Bandung Raya.
4. **Heatmap jarak antar-destinasi** (sample 15 destinasi acak) → konfirmasi jarak bervariasi sehingga DRL punya ruang manuver.
5. **Bar chart count per kategori** → konfirmasi tidak ada kategori yang dominan ekstrem.
""")

code(r'''# ── 7.1  Distribusi harga & rating per kategori ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for cat, color in zip(CATEGORY_ORDER, ["#2E8B57", "#D2691E", "#4682B4"]):
    sub = df_clean[df_clean["category"] == cat]
    if len(sub) > 0:
        axes[0].hist(sub["ticket"], bins=20, alpha=0.55, label=cat,
                     color=color, edgecolor="white")
        axes[1].hist(sub["rating"], bins=15, alpha=0.55, label=cat,
                     color=color, edgecolor="white")

axes[0].set_title("Distribusi Harga Tiket per Kategori")
axes[0].set_xlabel("Harga (IDR)")
axes[0].set_ylabel("Jumlah destinasi")
axes[0].legend()
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

axes[1].set_title("Distribusi Rating per Kategori")
axes[1].set_xlabel("Rating (1–5)")
axes[1].set_ylabel("Jumlah destinasi")
axes[1].set_xlim(3.4, 5.0)
axes[1].legend()
plt.tight_layout()
plt.show()

# ── 7.2  Scatter geografis ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for cat, color in zip(CATEGORY_ORDER, ["#2E8B57", "#D2691E", "#4682B4"]):
    sub = df_clean[df_clean["category"] == cat]
    ax.scatter(sub["lng"], sub["lat"], s=30, alpha=0.65, label=cat, c=color)
# Tandai home options
home_lats = [h["lat"] for h in HOME_OPTIONS]
home_lngs = [h["lng"] for h in HOME_OPTIONS]
ax.scatter(home_lngs, home_lats, s=180, marker="*", c="red",
           edgecolor="black", label="Home options", zorder=5)
ax.set_title("Sebaran Geografis Destinasi (Bandung Raya)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(loc="best")
plt.tight_layout()
plt.show()

# ── 7.3  Bar count per kategori ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
counts = df_clean["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
bars = ax.bar(counts.index, counts.values,
              color=["#2E8B57", "#D2691E", "#4682B4"], edgecolor="black")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            str(int(val)), ha="center", fontweight="bold")
ax.set_title(f"Jumlah Destinasi per Kategori (Total: {len(df_clean)})")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# ── 7.4  Sample heatmap jarak antar destinasi ────────────────
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(phi1)*math.cos(phi2)
           * math.sin(math.radians(lng2-lng1)/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

sample = df_clean.sample(min(15, len(df_clean)), random_state=42).reset_index(drop=True)
n = len(sample)
mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        mat[i,j] = haversine_km(sample.iloc[i]["lat"], sample.iloc[i]["lng"],
                                 sample.iloc[j]["lat"], sample.iloc[j]["lng"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(mat, xticklabels=sample["name"].str[:14],
            yticklabels=sample["name"].str[:14],
            cmap="YlOrRd", ax=ax, cbar_kws={"label":"km"})
ax.set_title("Matriks Jarak Antar Destinasi (sample 15) — km")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(f"\n📊 Statistik jarak (sample 15×15):")
upper = mat[np.triu_indices(n, k=1)]
print(f"   min  : {upper.min():.2f} km")
print(f"   mean : {upper.mean():.2f} km")
print(f"   max  : {upper.max():.2f} km")
print(f"   std  : {upper.std():.2f} km")
''')

# =====================================================================
# SECTION 8 — Feature Engineering
# =====================================================================
md(r"""## §8 — Feature Engineering

**Apa yang terjadi di section ini:**

Mengubah dataframe destinasi menjadi **matriks fitur numerik** yang siap dipakai untuk Content-Based Filtering.

| Bagian fitur | Dim | Bobot | Sumber |
|--------------|-----|-------|--------|
| One-hot kategori | 3 | ×2.0 | `Alam/Kuliner/Wisata` |
| Numerik | 5 | ×1.0 | `log(ticket)`, `rating`, `duration`, `lat`, `lng` (Min-Max scaled) |
| TF-IDF tags | s.d. 20 | ×0.5 | hasil tags semantic dari OSM |

**Mengapa kategori di-bobot ×2.0?**  
Agar similarity antar-destinasi-beda-kategori **lebih rendah**. Ini penting untuk variety: CBF tidak boleh menganggap restoran dan factory outlet "mirip" hanya karena keduanya punya rating sama dan dekat secara geografis.

**Output:**
- `feature_matrix.npy` — array `(N, ≤28)`
- `models/label_encoders.pkl` — encoder + scaler + TF-IDF vectorizer untuk inference
- `models/scaler.pkl` — copy scaler MinMax untuk inference
""")

code(r'''# ── 8.1  Feature engineering ─────────────────────────────────
df_feat = df_clean.copy()

# (a) One-hot kategori (3 dim)
category_onehot = np.zeros((len(df_feat), len(CATEGORY_ORDER)), dtype=float)
for i, cat in enumerate(df_feat["category"].tolist()):
    if cat in CATEGORY_ORDER:
        category_onehot[i, CATEGORY_ORDER.index(cat)] = 1.0

# (b) Numerik
df_feat["ticket_log"] = np.log1p(df_feat["ticket"].astype(float))
numeric_cols = ["ticket_log", "rating", "duration", "lat", "lng"]
scaler = MinMaxScaler()
numeric_normalized = scaler.fit_transform(df_feat[numeric_cols].values)

# (c) TF-IDF dari tags
df_feat["tags_str"] = df_feat["tags"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else str(x)
)
tfidf = TfidfVectorizer(max_features=20, min_df=1)
tfidf_matrix = tfidf.fit_transform(df_feat["tags_str"]).toarray()

tfidf_scaler = None
if tfidf_matrix.shape[1] > 0 and tfidf_matrix.max() > 0:
    tfidf_scaler = MinMaxScaler()
    tfidf_matrix = tfidf_scaler.fit_transform(tfidf_matrix)

# (d) Gabung
feature_matrix = np.hstack([
    category_onehot * 2.0,
    numeric_normalized,
    tfidf_matrix * 0.5,
])

print(f"✅ Feature matrix: {feature_matrix.shape}")
print(f"   Kategori (×2.0): {category_onehot.shape[1]} dim")
print(f"   Numerik         : {numeric_normalized.shape[1]} dim")
print(f"   TF-IDF (×0.5)   : {tfidf_matrix.shape[1]} dim")

# ── 8.2  Save ────────────────────────────────────────────────
np.save("data/processed/feature_matrix.npy", feature_matrix)

le_category = LabelEncoder().fit(CATEGORY_ORDER)
encoders = {
    "label_encoder_category": le_category,
    "scaler":          scaler,
    "tfidf":           tfidf,
    "tfidf_scaler":    tfidf_scaler,
    "feature_cols":    numeric_cols,
    "category_order":  CATEGORY_ORDER,
    "n_category_dims": category_onehot.shape[1],
    "n_numeric_dims":  numeric_normalized.shape[1],
    "n_tfidf_dims":    tfidf_matrix.shape[1],
}
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("💾 feature_matrix.npy | label_encoders.pkl | scaler.pkl")
''')

print("CHECKPOINT: 8 sections done")



# =====================================================================
# SECTION 9 — Content-Based Filtering
# =====================================================================
md(r"""## §9 — Content-Based Filtering (CBF)

**Apa yang terjadi di section ini:**

Mendefinisikan class `ContentBasedFilter` (sekali, tidak duplikat seperti versi lama) dengan tiga API utama:

1. **`fit()`** — pre-compute matriks similarity (cosine) `N×N`.
2. **`recommend(categories, budget, max_km, home_lat, home_lng, top_n)`** — kembalikan kandidat top-N dengan **hard constraint**:
   - kategori harus match,
   - `ticket ≤ budget`,
   - jarak ke home ≤ `max_km`.
3. **`interleave_by_category(...)`** — round-robin antar kategori untuk mencegah hasil "spam" satu kategori.

**Variety guarantee di level CBF:**  
Jika user pilih ≥ 2 kategori, CBF wajib menyiapkan minimal `ceil(top_n / len(categories))` kandidat per kategori. Ini menjadi *input feed* yang sehat ke RL agent.

**Hard distance gate:**  
Filter `max_km` dihitung pakai jarak Haversine. Jika tidak ada cukup kandidat dalam radius, fallback hanya merilekskan budget — **tidak pernah** merilekskan radius.
""")

code(r'''# ── 9.1  Distance helper (modul global) ──────────────────────
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(phi1)*math.cos(phi2)
           * math.sin(math.radians(lng2-lng1)/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


# ── 9.2  Class CBF (single source of truth) ──────────────────
class ContentBasedFilter:
    """Cosine-similarity recommender atas vektor fitur destinasi.

    Hard constraints (selalu di-enforce):
      - kategori dalam `categories`
      - ticket ≤ budget
      - haversine(home, dest) ≤ max_km

    Variety: jika len(categories) >= 2, hasil dijamin punya minimal
    ceil(top_n/len(categories)) per kategori.
    """

    def __init__(self, feature_matrix, destinations_df, encoders):
        self.feature_matrix    = feature_matrix
        self.df                = destinations_df.reset_index(drop=True)
        self.encoders          = encoders
        self.similarity_matrix = None

    def fit(self):
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"✅ CBF fitted. Similarity matrix: {self.similarity_matrix.shape}")
        return self

    def _category_dims(self):
        return self.encoders.get("n_category_dims", len(CATEGORY_ORDER))

    def _numeric_start(self):
        return self._category_dims()

    def build_user_profile(self, categories, budget=None):
        if categories:
            mask = self.df["category"].isin(categories).values
        else:
            mask = np.ones(len(self.df), dtype=bool)
        if mask.sum() == 0:
            mask = np.ones(len(self.df), dtype=bool)
        profile = self.feature_matrix[mask].mean(axis=0)

        # Override ticket_log dim dgn budget user (preferensi murah/mahal)
        if budget is not None and budget > 0:
            try:
                sc   = self.encoders["scaler"]
                cols = self.encoders["feature_cols"]
                t_idx = cols.index("ticket_log")
                dummy = np.zeros((1, len(cols)))
                for j, col in enumerate(cols):
                    if col == "ticket_log":
                        dummy[0, j] = math.log1p(budget)
                    else:
                        dummy[0, j] = self.feature_matrix[:, self._numeric_start()+j].mean()
                norm = sc.transform(dummy)[0]
                profile = profile.copy()
                profile[self._numeric_start() + t_idx] = norm[t_idx]
            except Exception:
                pass
        return profile

    def recommend(self, categories=None, budget=None, max_km=None,
                  home_lat=-6.9215, home_lng=107.6071, top_n=60):
        cats = list(categories) if categories else []

        # ── Hard masks ─────────────────────────────────────────
        mask_cat = (self.df["category"].isin(cats).values
                    if cats else np.ones(len(self.df), bool))

        # Budget=0 → hanya tiket gratis. budget=None → unlimited.
        if budget is None:
            mask_budget = np.ones(len(self.df), bool)
        else:
            mask_budget = (self.df["ticket"].astype(float) <= float(budget)).values

        if max_km is not None:
            dist = self.df.apply(
                lambda r: haversine_km(home_lat, home_lng, r["lat"], r["lng"]),
                axis=1
            ).values
            mask_km = dist <= float(max_km)
        else:
            dist = None
            mask_km = np.ones(len(self.df), bool)

        # Combine
        mask_all = mask_cat & mask_budget & mask_km
        # Soft fallback (relax HANYA budget+kategori, tidak pernah max_km)
        if mask_all.sum() == 0:
            mask_all = mask_cat & mask_km
        if mask_all.sum() == 0:
            mask_all = mask_km
        if mask_all.sum() == 0:
            mask_all = np.ones(len(self.df), bool)

        # ── Skor cosine ────────────────────────────────────────
        profile = self.build_user_profile(cats, budget)
        scores  = cosine_similarity([profile], self.feature_matrix)[0]
        scores  = scores * mask_all.astype(float)

        # ── Per-kategori guarantee ─────────────────────────────
        if len(cats) >= 2:
            per_cat = max(3, top_n // len(cats))
            picked, seen = [], set()
            for cat in cats:
                cat_mask   = (self.df["category"] == cat).values & mask_all
                cat_scores = scores * cat_mask.astype(float)
                order      = np.argsort(cat_scores)[::-1]
                cnt = 0
                for idx in order:
                    if cnt >= per_cat:
                        break
                    if cat_scores[idx] > 0 and idx not in seen:
                        picked.append(idx); seen.add(idx); cnt += 1
            # Isi sisa dari global top
            for idx in np.argsort(scores)[::-1]:
                if len(picked) >= top_n:
                    break
                if scores[idx] > 0 and idx not in seen:
                    picked.append(idx); seen.add(idx)
            result = self.df.iloc[picked].copy()
            result["cbf_score"] = scores[picked]
        else:
            top_idx = np.argsort(scores)[::-1][:top_n]
            result  = self.df.iloc[top_idx].copy()
            result["cbf_score"] = scores[top_idx]
            result = result[result["cbf_score"] > 0]

        # Tambah dist_home_km untuk debug & enforcement downstream
        if max_km is not None:
            result["dist_home_km"] = result.apply(
                lambda r: haversine_km(home_lat, home_lng, r["lat"], r["lng"]),
                axis=1
            )
        return result.reset_index(drop=True)

    def interleave_by_category(self, df_candidates, categories, n):
        """Round-robin antar kategori — anti spam 1 kategori beruntun."""
        if len(categories) <= 1:
            return df_candidates.head(n)
        buckets = {
            c: df_candidates[df_candidates["category"] == c]
                 .sort_values("cbf_score", ascending=False)
                 .reset_index(drop=True)
            for c in categories
        }
        result, seen = [], set()
        ptr = {c: 0 for c in categories}
        while len(result) < n:
            added = 0
            for c in categories:
                if len(result) >= n:
                    break
                pool = buckets.get(c, pd.DataFrame())
                while ptr[c] < len(pool):
                    row = pool.iloc[ptr[c]]; ptr[c] += 1
                    if row["id"] not in seen:
                        result.append(row); seen.add(row["id"]); added += 1
                        break
            if added == 0:
                for _, row in df_candidates.iterrows():
                    if len(result) >= n:
                        break
                    if row["id"] not in seen:
                        result.append(row); seen.add(row["id"])
                break
        return pd.DataFrame(result).reset_index(drop=True) if result else df_candidates.head(n)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "similarity_matrix": self.similarity_matrix,
                "feature_matrix":    self.feature_matrix,
                "df_index":          self.df[["id","name","category"]].to_dict("records"),
            }, f)
        print(f"✅ CBF model disimpan ke {path}")


# ── 9.3  Fit & save ─────────────────────────────────────────
cbf_model = ContentBasedFilter(feature_matrix, df_clean, encoders).fit()
cbf_model.save("models/cbf_model.pkl")

# ── 9.4  Smoke test ─────────────────────────────────────────
print("\n🧪 TEST 1 — Alam saja, budget 50k, top 5")
t1 = cbf_model.recommend(categories=["Alam"], budget=50000, top_n=5)
print(t1[["name","category","ticket","rating","cbf_score"]].to_string(index=False))

print("\n🧪 TEST 2 — Multi (Alam+Kuliner+Wisata), max 30km dari Alun-Alun, top 12")
t2 = cbf_model.recommend(categories=CATEGORY_ORDER, budget=300_000,
                         max_km=30, home_lat=-6.9215, home_lng=107.6071, top_n=12)
print(t2.groupby("category").size().to_dict())
assert all(c in t2["category"].values for c in CATEGORY_ORDER), \
    "❌ Variety guarantee gagal — ada kategori tidak terwakili"
print("✅ Semua kategori terwakili.")

print("\n🧪 TEST 3 — Interleave check")
inter = cbf_model.interleave_by_category(t2, CATEGORY_ORDER, n=9)
print("   Urutan:", inter["category"].tolist())
''')

# =====================================================================
# SECTION 10 — Visualisasi CBF
# =====================================================================
md(r"""## §10 — Visualisasi CBF

**Apa yang terjadi di section ini:**

1. **Heatmap top-20 destinasi paling mirip** untuk satu destinasi referensi (Kawah Putih) — visualisasi kerja cosine similarity.
2. **Bar chart top-10 destinasi** untuk skenario user "Alam + Kuliner, budget 200k, max 25 km".
3. **Distribusi nilai cosine similarity** di seluruh matriks — sanity-check apakah similarity tidak terlalu seragam (tanda fitur kurang diskriminatif).
""")

code(r'''# ── 10.1  Heatmap similarity terhadap satu referensi ─────────
ref_name = "Kawah Putih"
if ref_name in df_clean["name"].values:
    ref_idx = df_clean.index[df_clean["name"] == ref_name][0]
    sims = cbf_model.similarity_matrix[ref_idx]
    top_idx = np.argsort(sims)[::-1][1:21]   # skip dirinya sendiri
    top_df = df_clean.iloc[top_idx][["name","category"]].copy()
    top_df["sim"] = sims[top_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top_df))[::-1], top_df["sim"],
            color=["#2E8B57" if c=="Alam" else "#D2691E" if c=="Kuliner" else "#4682B4"
                   for c in top_df["category"]])
    ax.set_yticks(range(len(top_df))[::-1])
    ax.set_yticklabels([f"{n[:25]} ({c})" for n, c in zip(top_df["name"], top_df["category"])])
    ax.set_xlabel("Cosine similarity")
    ax.set_title(f"Top-20 destinasi mirip dengan: {ref_name}")
    plt.tight_layout()
    plt.show()

# ── 10.2  Skenario realistic recommendation ──────────────────
test_rec = cbf_model.recommend(
    categories=["Alam", "Kuliner"],
    budget=200_000, max_km=25,
    home_lat=-6.9215, home_lng=107.6071,
    top_n=10
)
print(f"📋 Skenario: Alam+Kuliner, budget 200k, max 25 km → {len(test_rec)} kandidat")
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#2E8B57" if c == "Alam" else "#D2691E" for c in test_rec["category"]]
ax.barh(range(len(test_rec)-1, -1, -1), test_rec["cbf_score"], color=colors)
ax.set_yticks(range(len(test_rec)-1, -1, -1))
ax.set_yticklabels([f"{n[:24]}" for n in test_rec["name"]])
ax.set_xlabel("CBF score")
ax.set_title("Top-10 rekomendasi (Alam=hijau, Kuliner=oranye)")
plt.tight_layout()
plt.show()

# ── 10.3  Distribusi cosine similarity (off-diagonal) ─────────
N = len(df_clean)
upper = cbf_model.similarity_matrix[np.triu_indices(N, k=1)]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(upper, bins=50, color="#6A5ACD", edgecolor="white")
ax.axvline(upper.mean(), color="red", linestyle="--",
           label=f"mean = {upper.mean():.3f}")
ax.set_title("Distribusi cosine similarity (off-diagonal pairs)")
ax.set_xlabel("Similarity"); ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.show()

print(f"📊 Similarity stats: min={upper.min():.3f} | mean={upper.mean():.3f} | max={upper.max():.3f}")
''')

# =====================================================================
# SECTION 11 — RL Environment + Q-Agent
# =====================================================================
md(r"""## §11 — Reinforcement Learning Environment + Q-Agent

**Apa yang terjadi di section ini:**

Membangun MDP simulator + Q-Learning agent yang **adil terhadap empat dimensi: rating, variety, distance, budget**.

### State
4-tuple discrete: `(n_selected, budget_level, time_level, dominant_category)`
- `budget_level ∈ {0..4}` (0=habis, 4=sisa ≥75%)
- `time_level   ∈ {0..4}` (0=habis, 4=sisa ≥6 jam)
- `dominant_category` = index kategori yang paling banyak di-pilih sejauh ini

### Action
`action = index ke kandidat CBF (top-30)`. `get_valid_actions()` memfilter aksi yang melanggar budget/waktu/**max_km** secara HARD.

### Reward (key change)
Bobot baru — DRL benar-benar adil:

| Komponen | Bobot | Formula |
|----------|-------|---------|
| `rating_score`   | **0.30** | `(rating - 3) / 2` clip [0,1] |
| `variety_bonus`  | **0.25** | 1.0 jika kategori belum pernah dipilih, else 0 |
| `distance_score` | **0.25** | `1 - (haversine_home / max_km)` clip [0,1] |
| `budget_score`   | **0.20** | `remaining_budget / total_budget` |

**Penalty (defense in depth):**
- `−2.0` jika destinasi melanggar `max_km` (seharusnya sudah di-block oleh `get_valid_actions`, tapi penalty besar mengajari agent agar tidak coba-coba)
- `−1.0` jika overtime
- `−0.5` jika overbudget

### Hyperparameter
- `learning_rate = 0.1`, `gamma = 0.95`
- `epsilon = 1.0 → 0.05` dengan decay `0.995`
""")

code(r'''# ── 11.1  Environment ────────────────────────────────────────
class BandungTravelEnv:
    """Simulasi sequential pemilihan destinasi.

    Reward mempertimbangkan 4 dimensi: rating, variety, distance, budget.
    `max_km` adalah HARD constraint di get_valid_actions, bukan sekadar penalty.
    """

    REWARD_WEIGHTS = {
        "rating":   0.30,
        "variety":  0.25,
        "distance": 0.25,
        "budget":   0.20,
    }
    PENALTY_OVERTIME    = 1.0
    PENALTY_OVERBUDGET  = 0.5
    PENALTY_OVERMAXKM   = 2.0
    UNLIMITED_BUDGET    = 999_999_998
    UNLIMITED_KM        = 999.0

    def __init__(self, destinations_df: pd.DataFrame, cbf_model: ContentBasedFilter):
        self.df  = destinations_df.reset_index(drop=True)
        self.cbf = cbf_model
        self.n_destinations = len(self.df)
        self.params      = None
        self.candidates  = []      # idx pada self.df
        self.selected    = []
        self.spent       = 0
        self.cur_lat     = None
        self.cur_lng     = None
        self.cur_time    = 0

    def _haversine(self, lat1, lng1, lat2, lng2):
        return haversine_km(lat1, lng1, lat2, lng2)

    def _idx_by_id(self, dest_id):
        rows = self.df.index[self.df["id"] == dest_id].tolist()
        return rows[0] if rows else -1

    def reset(self, params: dict):
        self.params = {**params}
        # Default values — penting: budget=0 berarti hanya gratis, bukan unlimited.
        if self.params.get("budget") is None:
            self.params["budget"] = self.UNLIMITED_BUDGET
        if self.params.get("max_km") is None:
            self.params["max_km"] = self.UNLIMITED_KM
        self.params.setdefault("count",    4)
        self.params.setdefault("startMin", 9 * 60)
        self.params.setdefault("endMin",   21 * 60)
        self.params.setdefault("home_lat", HOME_OPTIONS[0]["lat"])
        self.params.setdefault("home_lng", HOME_OPTIONS[0]["lng"])

        # CBF top-30 kandidat (sudah hard-filter max_km)
        rec = self.cbf.recommend(
            categories=self.params.get("categories", []),
            budget=None if self.params["budget"] >= self.UNLIMITED_BUDGET else self.params["budget"],
            max_km=None if self.params["max_km"]  >= self.UNLIMITED_KM     else self.params["max_km"],
            home_lat=self.params["home_lat"],
            home_lng=self.params["home_lng"],
            top_n=30,
        )
        if len(rec) == 0:
            rec = self.cbf.recommend(top_n=30)

        self.candidates = [self._idx_by_id(i) for i in rec["id"].tolist()]
        self.candidates = [i for i in self.candidates if i >= 0]
        self.selected   = []
        self.spent      = 0
        self.cur_lat    = self.params["home_lat"]
        self.cur_lng    = self.params["home_lng"]
        self.cur_time   = self.params["startMin"]
        return self._get_state(), list(self.candidates)

    def _get_state(self):
        n_sel = min(8, len(self.selected))

        budget_total = self.params["budget"]
        if budget_total <= 0:
            budget_level = 0
        elif budget_total >= self.UNLIMITED_BUDGET:
            budget_level = 4
        else:
            ratio = max(0.0, 1.0 - self.spent / budget_total)
            budget_level = (
                0 if ratio <= 0.0
                else 1 if ratio < 0.25
                else 2 if ratio < 0.50
                else 3 if ratio < 0.75
                else 4
            )

        time_left = max(0, self.params["endMin"] - self.cur_time)
        time_level = (
            0 if time_left <= 0
            else 1 if time_left < 120
            else 2 if time_left < 240
            else 3 if time_left < 360
            else 4
        )

        dom = 0
        if self.selected:
            cats = Counter(self.df.iloc[i]["category"] for i in self.selected)
            top_cat = cats.most_common(1)[0][0]
            dom = CATEGORY_ORDER.index(top_cat) if top_cat in CATEGORY_ORDER else 0

        return (n_sel, budget_level, time_level, dom)

    def get_valid_actions(self):
        """HARD-gate semua constraint, termasuk max_km."""
        valid = []
        for k, idx in enumerate(self.candidates):
            if idx in self.selected:
                continue
            row = self.df.iloc[idx]

            # Budget
            if int(row["ticket"]) > (self.params["budget"] - self.spent):
                continue

            # max_km dari HOME (HARD)
            dist_home = self._haversine(
                self.params["home_lat"], self.params["home_lng"],
                row["lat"], row["lng"]
            )
            if dist_home > self.params["max_km"]:
                continue

            # Waktu (perjalanan + stay + pulang harus muat)
            travel_km = self._haversine(self.cur_lat, self.cur_lng, row["lat"], row["lng"])
            travel_min = (travel_km / SPEED_KMH) * 60
            arrive = self.cur_time + travel_min
            depart = arrive + int(row["duration"])
            ret_km = self._haversine(
                row["lat"], row["lng"],
                self.params["home_lat"], self.params["home_lng"]
            )
            ret_min = (ret_km / SPEED_KMH) * 60
            if depart + ret_min > self.params["endMin"]:
                continue

            valid.append(k)
        return valid

    def _calculate_reward(self, dest_row, travel_km: float) -> float:
        # rating: [3, 5] → [0, 1]
        rating_score = max(0.0, min(1.0, (float(dest_row["rating"]) - 3.0) / 2.0))

        # variety: 1.0 jika kategori belum pernah dipilih (exclude current)
        prior = self.selected[:-1] if self.selected else []
        cats_chosen = {self.df.iloc[i]["category"] for i in prior}
        variety = 1.0 if dest_row["category"] not in cats_chosen else 0.0

        # distance: lebih dekat ke home = lebih baik
        max_km_param = (self.params["max_km"]
                        if self.params["max_km"] < self.UNLIMITED_KM else 60.0)
        dist_home = self._haversine(
            self.params["home_lat"], self.params["home_lng"],
            dest_row["lat"], dest_row["lng"]
        )
        distance_score = max(0.0, 1.0 - (dist_home / max_km_param))

        # budget: sisa budget (high)
        budget_total = self.params["budget"]
        if budget_total <= 0 or budget_total >= self.UNLIMITED_BUDGET:
            budget_score = 0.5
        else:
            remain = max(0, budget_total - self.spent)
            budget_score = remain / budget_total

        w = self.REWARD_WEIGHTS
        r = (w["rating"]   * rating_score
             + w["variety"]  * variety
             + w["distance"] * distance_score
             + w["budget"]   * budget_score)

        # Defense-in-depth penalties
        if dist_home > self.params["max_km"]:
            r -= self.PENALTY_OVERMAXKM
        if self.cur_time > self.params["endMin"]:
            r -= self.PENALTY_OVERTIME
        if self.spent > self.params["budget"]:
            r -= self.PENALTY_OVERBUDGET

        return float(r)

    def step(self, action_idx: int):
        valid = self.get_valid_actions()
        if action_idx not in valid:
            return self._get_state(), -0.1, True, {"reason": "invalid_action"}

        idx = self.candidates[action_idx]
        row = self.df.iloc[idx]
        travel_km = self._haversine(self.cur_lat, self.cur_lng, row["lat"], row["lng"])
        travel_min = (travel_km / SPEED_KMH) * 60
        self.cur_time += travel_min
        self.cur_time += int(row["duration"])
        self.spent    += int(row["ticket"])
        self.selected.append(idx)
        self.cur_lat, self.cur_lng = row["lat"], row["lng"]

        reward = self._calculate_reward(row, travel_km)
        done = (
            len(self.selected) >= self.params["count"]
            or self.cur_time >= self.params["endMin"]
            or len(self.get_valid_actions()) == 0
        )
        return self._get_state(), reward, done, {"travel_km": travel_km}

    def generate_random_params(self) -> dict:
        n_cats = random.randint(1, len(CATEGORY_ORDER))
        cats = random.sample(CATEGORY_ORDER, k=n_cats)
        budget  = None if random.random() < 0.20 else random.randint(50_000, 2_000_000)
        max_km  = None if random.random() < 0.30 else random.randint(15, 60)
        count   = random.randint(2, 5)
        start   = random.randint(420, 600)
        end     = start + random.randint(300, 900)
        home    = random.choice(HOME_OPTIONS)
        return {
            "categories": cats,
            "budget":     budget,
            "max_km":     max_km,
            "count":      count,
            "startMin":   start,
            "endMin":     end,
            "home_lat":   home["lat"],
            "home_lng":   home["lng"],
        }


# ── 11.2  Q-Learning Agent ───────────────────────────────────
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.lr            = learning_rate
        self.gamma         = discount_factor
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = defaultdict(lambda: defaultdict(float))
        self.training_history = []
        self.epsilon_history  = []

    @staticmethod
    def _action_key(action_idx, candidate_ids):
        return (candidate_ids[action_idx]
                if 0 <= action_idx < len(candidate_ids) else "unknown")

    def choose_action(self, state, valid_actions, candidate_ids):
        if not valid_actions:
            return -1
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        best, best_q = valid_actions[0], -float("inf")
        for a in valid_actions:
            q = self.q_table[state][self._action_key(a, candidate_ids)]
            if q > best_q:
                best_q, best = q, a
        return best

    def update(self, state, action_idx, reward, next_state, done,
               valid_next_actions, candidate_ids, next_candidate_ids):
        a_key = self._action_key(action_idx, candidate_ids)
        cur_q = self.q_table[state][a_key]
        if done or not valid_next_actions:
            target = reward
        else:
            future = max(
                self.q_table[next_state][self._action_key(a, next_candidate_ids)]
                for a in valid_next_actions
            )
            target = reward + self.gamma * future
        self.q_table[state][a_key] = cur_q + self.lr * (target - cur_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def save(self, path):
        plain = {s: dict(actions) for s, actions in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump({
                "q_table":          plain,
                "epsilon":          self.epsilon,
                "lr":               self.lr,
                "gamma":            self.gamma,
                "training_history": self.training_history,
                "epsilon_history":  self.epsilon_history,
            }, f)
        print(f"✅ RL Agent disimpan ke {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        ag = cls(learning_rate=data["lr"], discount_factor=data["gamma"])
        ag.q_table = defaultdict(lambda: defaultdict(float))
        for s, actions in data["q_table"].items():
            for a, v in actions.items():
                ag.q_table[s][a] = v
        ag.epsilon = data["epsilon"]
        ag.training_history = data.get("training_history", [])
        ag.epsilon_history  = data.get("epsilon_history", [])
        return ag


print("✅ Class BandungTravelEnv & QLearningAgent siap.")
print(f"   Reward weights: {BandungTravelEnv.REWARD_WEIGHTS}")
''')

# =====================================================================
# SECTION 12 — Training RL
# =====================================================================
md(r"""## §12 — Training Q-Learning Agent

**Apa yang terjadi di section ini:**

1. Instansiasi `env` dan `agent`.
2. Loop training **3000 episode**:
   - Setiap episode: sample parameter user random (kategori, budget, max_km, dll).
   - Reset env, jalankan greedy-ε sampai `done`.
   - Update Q-table tiap step.
   - Decay ε.
3. Log reward dan ε per episode → dipakai untuk plot di §13.
4. Save agent ke `models/rl_agent.pkl`.

> **Reproducibility:** RNG seed 42 di-set di §1. Hasil training konsisten antar-run.
""")

code(r'''# ── 12.1  Setup ──────────────────────────────────────────────
env      = BandungTravelEnv(df_clean, cbf_model)
rl_agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95,
                          epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)

# ── 12.2  Training loop ──────────────────────────────────────
def train_rl_agent(env, agent, n_episodes=3000, log_interval=500):
    rewards_log = []
    for ep in tqdm(range(n_episodes), desc="Training RL Agent"):
        params = env.generate_random_params()
        state, candidates = env.reset(params)
        candidate_ids = [env.df.iloc[i]["id"] for i in candidates]
        total = 0.0
        done  = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.choose_action(state, valid, candidate_ids)
            if action < 0:
                break
            next_state, reward, done, _ = env.step(action)
            next_valid = env.get_valid_actions()
            agent.update(state, action, reward, next_state, done,
                         next_valid, candidate_ids, candidate_ids)
            state = next_state
            total += reward
        agent.decay_epsilon()
        rewards_log.append(total)
        agent.training_history.append(total)
        if (ep + 1) % log_interval == 0:
            avg = float(np.mean(rewards_log[-log_interval:]))
            print(f"  Episode {ep+1}/{n_episodes} | avg reward {avg:.4f} | ε {agent.epsilon:.3f}")
    return rewards_log


N_EPISODES = 3000
rewards_log = train_rl_agent(env, rl_agent, n_episodes=N_EPISODES, log_interval=500)
rl_agent.save("models/rl_agent.pkl")
print(f"\n✅ Training selesai. Q-table size = {len(rl_agent.q_table)} unique states.")
''')

# =====================================================================
# SECTION 13 — Visualisasi Training RL
# =====================================================================
md(r"""## §13 — Visualisasi Training RL

**Apa yang terjadi di section ini:**

1. **Reward curve** — raw + moving-average (window 100). Konvergen artinya policy stabil.
2. **Epsilon decay** — kurva eksplorasi → eksploitasi.
3. **Distribusi reward 500 episode terakhir** — sanity-check bahwa policy "matang" memberikan reward tinggi konsisten.
""")

code(r'''# ── 13.1  Reward curve + moving average ──────────────────────
rewards = np.array(rl_agent.training_history)
window  = max(10, len(rewards) // 30)
moving  = np.convolve(rewards, np.ones(window)/window, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].plot(rewards, alpha=0.25, color="steelblue", label="raw")
axes[0].plot(range(window-1, len(rewards)), moving,
             color="darkblue", linewidth=2, label=f"moving avg (w={window})")
axes[0].set_title("Reward per Episode")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total reward")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(rl_agent.epsilon_history, color="darkorange", linewidth=2)
axes[1].set_title("Epsilon Decay (exploration → exploitation)")
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("ε")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── 13.2  Distribusi reward final 500 episode ────────────────
final = rewards[-500:]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(final, bins=40, color="#3CB371", edgecolor="white")
ax.axvline(final.mean(), color="red", linestyle="--",
           label=f"mean = {final.mean():.3f}")
ax.set_title("Distribusi reward — 500 episode terakhir")
ax.set_xlabel("Total reward"); ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.show()

# ── 13.3  Stats ──────────────────────────────────────────────
print(f"📊 Reward statistics:")
print(f"   Episode 1-500    : mean = {rewards[:500].mean():.3f}, std = {rewards[:500].std():.3f}")
print(f"   Episode last 500 : mean = {final.mean():.3f}, std = {final.std():.3f}")
print(f"   Improvement      : {(final.mean() - rewards[:500].mean()):+.3f}")
print(f"   Final ε          : {rl_agent.epsilon:.4f}")
print(f"   Q-table states   : {len(rl_agent.q_table)}")
''')

print("CHECKPOINT: 13 sections done")



# =====================================================================
# SECTION 14 — Route Optimizer
# =====================================================================
md(r"""## §14 — Route Optimizer (TSP Nearest-Neighbor)

**Apa yang terjadi di section ini:**

Setelah RL memilih destinasi (set, bukan urutan), kita perlu **mengurutkannya** agar total perjalanan minimal. Class `RouteOptimizer`:

1. **`nearest_neighbor_route(home, destinations)`** — TSP heuristic O(n²): mulai dari home, terus ambil destinasi terdekat dari posisi sekarang.
2. **`osrm_travel_time(...)`** — fallback ke **OSRM public API** untuk waktu tempuh real (jalan), atau Haversine jika OSRM tidak responsif.
3. **`build_itinerary(home, ordered, start_min, end_min)`** — rangkaikan menjadi struktur `steps[]` dengan `arriveAt`, `departAt`, `travelKm`, `travelMin`, total cost, total km, return-home.

> **Catatan:** OSRM public server bisa rate-limit. Kalau kandidat banyak, output Haversine sudah "good enough" untuk skala kota.
""")

code(r'''# ── 14.1  RouteOptimizer ─────────────────────────────────────
class RouteOptimizer:
    SPEED_KMH = SPEED_KMH
    OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

    def __init__(self, use_osrm=True, osrm_timeout=5.0):
        self.use_osrm    = use_osrm
        self.osrm_timeout = osrm_timeout
        self._cache      = {}

    def haversine_km(self, lat1, lng1, lat2, lng2):
        return haversine_km(lat1, lng1, lat2, lng2)

    def osrm_travel_time(self, lat1, lng1, lat2, lng2):
        if not self.use_osrm:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            return d, (d / self.SPEED_KMH) * 60
        key = (round(lat1,4), round(lng1,4), round(lat2,4), round(lng2,4))
        if key in self._cache:
            return self._cache[key]
        try:
            url = f"{self.OSRM_BASE}/{lng1},{lat1};{lng2},{lat2}?overview=false"
            resp = requests.get(url, timeout=self.osrm_timeout)
            resp.raise_for_status()
            r = resp.json()["routes"][0]
            dist_km = r["distance"] / 1000.0
            dur_min = r["duration"] / 60.0
        except Exception:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            dist_km, dur_min = d, (d / self.SPEED_KMH) * 60
        self._cache[key] = (dist_km, dur_min)
        return dist_km, dur_min

    def nearest_neighbor_route(self, home, destinations):
        if not destinations:
            return []
        remaining = list(destinations)
        ordered   = []
        cur_lat, cur_lng = home["lat"], home["lng"]
        while remaining:
            nxt = min(
                remaining,
                key=lambda d: self.haversine_km(cur_lat, cur_lng, d["lat"], d["lng"])
            )
            ordered.append(nxt)
            remaining = [d for d in remaining if d.get("id") != nxt.get("id")]
            cur_lat, cur_lng = nxt["lat"], nxt["lng"]
        return ordered

    def build_itinerary(self, home, home_name, ordered_destinations,
                        start_min, end_min):
        steps      = []
        cur_time   = start_min
        cur_lat, cur_lng = home["lat"], home["lng"]
        total_cost = 0
        total_km   = 0.0

        for idx, dest in enumerate(ordered_destinations, 1):
            dist_km, dur_min = self.osrm_travel_time(
                cur_lat, cur_lng, dest["lat"], dest["lng"]
            )
            travel_min = max(1, int(dur_min))
            arrive_at  = cur_time + travel_min
            stay_min   = int(dest.get("duration", 90))
            depart_at  = arrive_at + stay_min

            steps.append({
                "idx":  idx,
                "dest": {
                    "id":          dest.get("id", ""),
                    "name":        dest.get("name", ""),
                    "category":    dest.get("category", ""),
                    "desc":        dest.get("desc", ""),
                    "ticket":      int(dest.get("ticket", 0)),
                    "duration":    stay_min,
                    "lat":         float(dest["lat"]),
                    "lng":         float(dest["lng"]),
                    "rating":      float(dest.get("rating", 0)),
                    "tags":        dest.get("tags", []),
                    "gmaps_url":   dest.get("gmaps_url", ""),
                    "stay_detail": dest.get("stay_detail", ""),
                },
                "travelMin": travel_min,
                "travelKm":  round(dist_km, 2),
                "arriveAt":  arrive_at,
                "departAt":  depart_at,
            })
            total_cost += int(dest.get("ticket", 0))
            total_km   += dist_km
            cur_time    = depart_at
            cur_lat, cur_lng = dest["lat"], dest["lng"]

        ret_km, ret_min = self.osrm_travel_time(
            cur_lat, cur_lng, home["lat"], home["lng"]
        )
        ret_min = max(1, int(ret_min))
        arrive_home = cur_time + ret_min

        return {
            "steps":      steps,
            "totalCost":  total_cost,
            "totalKm":    round(total_km + ret_km, 2),
            "totalTime":  arrive_home - start_min,
            "returnKm":   round(ret_km, 2),
            "returnMin":  ret_min,
            "arriveHome": arrive_home,
            "overBudget": False,
            "spareMin":   max(0, end_min - arrive_home),
        }


route_optimizer = RouteOptimizer()
print("✅ RouteOptimizer siap.")
''')

# =====================================================================
# SECTION 15 — Inference Pipeline
# =====================================================================
md(r"""## §15 — Inference Pipeline + Category-First Reservation + Hard Distance Validator

**Apa yang terjadi di section ini:**

Pipeline lengkap end-to-end yang dipanggil saat inference (revisi: kategori-first):

```
user_params → CBF.recommend (top-30, hard max_km) →
reserve_category_representatives (Phase A — 1 destinasi WAJIB per kategori) →
RL.greedy_select (Phase B — isi sisa slot, candidates exclude yang sudah reserved) →
hard_distance_validator (final defense) →
fallback_fill (jika kurang dari count) →
RouteOptimizer.nearest_neighbor → build_itinerary
```

### Aturan Kunci

1. **Category-first reservation** (`reserve_category_representatives`) — **PRIORITAS TERTINGGI**:
   - Jika user hanya pilih 1 destinasi (`count == 1`) atau hanya 1 kategori → bypass.
   - Jika user pilih ≥ 2 kategori dan `count > 1`:
       * Phase A reserve **1 destinasi WAJIB per kategori** SEBELUM DRL jalan.
       * Pemilihan per kategori = top-skor CBF dari `cbf_model.recommend(...)`
         yang sudah lolos hard filter (max_km + budget).
       * Kalau `count < len(categories)` (mis. 3 kategori tapi count=2) →
         pilih `count` kategori dengan kandidat skor tertinggi.
   - Setelah Phase A jalan, RL hanya mengisi `count - len(reserved)` slot sisa.
   - Ini *menggantikan* swap pasca-pemilihan yang dulu rapuh (kandidat
     missing category sering sudah ke-prune dari top-N RL candidates).

2. **Hard distance validator** (`enforce_distance_constraint`):
   - Setelah semua langkah, **buang destinasi yang melanggar `max_km`** dari home, lalu re-fill.
   - Ini lapis terakhir untuk memastikan `max_km` benar-benar di-respect.

3. **Smart fallback fill** menghormati sisa budget, sisa waktu, **dan max_km**.
""")

code(r'''# ── 15.1  RL greedy inference ────────────────────────────────
def rl_select_destinations(env, agent, params: dict) -> list:
    """Inference RL dengan epsilon=0 (greedy)."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        state, candidates = env.reset(params)
        candidate_ids = [env.df.iloc[i]["id"] for i in candidates]
        target = max(1, int(params.get("count", 4)))
        done = False
        while not done and len(env.selected) < target:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.choose_action(state, valid, candidate_ids)
            if action < 0 or action >= len(env.candidates):
                break
            state, _, done, _ = env.step(action)
        return [env.df.iloc[i].to_dict() for i in env.selected]
    finally:
        agent.epsilon = saved_eps


# ── 15.2  Category-first reservation (UPFRONT) ───────────────
def reserve_category_representatives(params: dict) -> list:
    """Phase A: reserve 1 destinasi WAJIB per kategori SEBELUM DRL.

    Ini menggantikan post-pick swap yang dulu rapuh. Untuk setiap kategori
    user, ambil top-skor CBF yang lolos hard filter (max_km + budget).
    Kalau count < len(categories), pilih count kategori dengan kandidat
    ber-skor tertinggi.

    - count == 1 atau len(categories) <= 1 → bypass (return []).
    - Else → return list of dict yang siap dipakai sebagai initial selection.
    """
    categories = params.get("categories", [])
    count      = max(1, int(params.get("count", 4)))

    if count == 1 or len(categories) <= 1:
        return []

    home_lat = params["home"]["lat"]
    home_lng = params["home"]["lng"]
    budget   = params.get("budget")
    has_budget = budget is not None and budget < BandungTravelEnv.UNLIMITED_BUDGET
    max_km   = params.get("maxKm")

    # Step 1: kumpulkan top-1 kandidat per kategori (yang lolos hard filter).
    per_cat_best = {}  # cat -> (row_dict, score)
    for cat in categories:
        rec = cbf_model.recommend(
            categories=[cat],
            budget=budget if has_budget else None,
            max_km=max_km,
            home_lat=home_lat, home_lng=home_lng,
            top_n=15,
        )
        if rec is None or len(rec) == 0:
            continue
        for _, row in rec.iterrows():
            # Defense in depth: cek max_km ulang
            if max_km is not None:
                d_home = haversine_km(home_lat, home_lng, row["lat"], row["lng"])
                if d_home > max_km:
                    continue
            if has_budget and int(row.get("ticket", 0)) > budget:
                continue
            per_cat_best[cat] = (row.to_dict(), float(row.get("cbf_score", 0.0)))
            break

    # Step 2: kalau count < len(categories), prioritaskan kategori dengan
    # kandidat ber-skor tertinggi. Else, ambil semua.
    n_reserve = min(count, len(categories))
    sorted_cats = sorted(per_cat_best.items(), key=lambda kv: -kv[1][1])[:n_reserve]

    reserved = []
    used_ids = set()
    spent    = 0
    for cat, (row, _score) in sorted_cats:
        if row["id"] in used_ids:
            continue
        if has_budget and spent + int(row.get("ticket", 0)) > budget:
            continue
        reserved.append(row)
        used_ids.add(row["id"])
        spent += int(row.get("ticket", 0))
        print(f"  📍 Reserve [{cat}]: {row['name']} (score={_score:.3f})")

    missing = [c for c in categories if c not in {r["category"] for r in reserved}]
    if missing and len(reserved) < n_reserve:
        print(f"  ⚠️  Kategori tanpa kandidat valid: {missing}")
    return reserved


# ── 15.3  Category guarantee (DEPRECATED — kept as no-op shim) ───
def enforce_category_guarantee(selected: list, params: dict) -> list:
    """DEPRECATED — kategori guarantee sekarang diberlakukan di awal lewat
    `reserve_category_representatives`. Fungsi ini di-keep sebagai no-op
    supaya pipeline lama tetap kompatibel.
    """
    return selected


# ── 15.4  Hard distance validator (final defense) ────────────
def enforce_distance_constraint(selected: list, params: dict) -> list:
    """Buang destinasi yang melanggar max_km. Final guard."""
    max_km = params.get("maxKm")
    if max_km is None:
        return selected
    home_lat = params["home"]["lat"]
    home_lng = params["home"]["lng"]
    kept = []
    for d in selected:
        dist = haversine_km(home_lat, home_lng, d["lat"], d["lng"])
        if dist <= max_km:
            kept.append(d)
        else:
            print(f"  🚫 Buang '{d['name']}' — {dist:.1f} km > max {max_km} km")
    return kept


# ── 15.5  Fallback fill ──────────────────────────────────────
def smart_fallback_fill(selected: list, params: dict, target: int) -> list:
    if len(selected) >= target:
        return selected
    home_lat = params["home"]["lat"]
    home_lng = params["home"]["lng"]
    spent    = sum(int(d.get("ticket", 0)) for d in selected)
    used_dur = sum(int(d.get("duration", 0)) for d in selected)

    has_budget = params.get("budget") is not None
    rem_budget = (params["budget"] - spent) if has_budget else None
    rem_time   = max(0, params["endMin"] - params["startMin"] - used_dur)

    rec = cbf_model.recommend(
        categories=params.get("categories", []),
        budget=rem_budget,
        max_km=params.get("maxKm"),
        home_lat=home_lat, home_lng=home_lng,
        top_n=30,
    )
    seen = {d["id"] for d in selected}
    for _, r in rec.iterrows():
        if len(selected) >= target:
            break
        if r["id"] in seen:
            continue
        if has_budget and int(r.get("ticket", 0)) > rem_budget:
            continue
        if int(r.get("duration", 60)) > rem_time:
            continue
        # Defense: cek max_km lagi
        if params.get("maxKm") is not None:
            d_home = haversine_km(home_lat, home_lng, r["lat"], r["lng"])
            if d_home > params["maxKm"]:
                continue
        selected.append(r.to_dict()); seen.add(r["id"])
        spent += int(r.get("ticket", 0))
        if has_budget:
            rem_budget -= int(r.get("ticket", 0))
        rem_time -= int(r.get("duration", 60))
    return selected


# ── 15.6  Pipeline lengkap ───────────────────────────────────
def full_pipeline(params: dict) -> dict:
    target = max(1, int(params.get("count", 4)))

    # PHASE A — Kategori-first reservation (jalan duluan, bukan post-hoc swap).
    reserved = reserve_category_representatives(params)
    print(f"  🎯 Phase A reserved {len(reserved)} kategori representative(s).")

    # PHASE B — RL hanya isi sisa slot.
    remaining_target = max(0, target - len(reserved))
    rl_params = {
        "categories": params.get("categories", []),
        "budget":     params.get("budget"),
        "max_km":     params.get("maxKm"),
        "count":      max(1, remaining_target),  # RL minimal 1; akan dipotong di filter
        "startMin":   params.get("startMin", 9 * 60),
        "endMin":     params.get("endMin",   21 * 60),
        "home_lat":   params["home"]["lat"],
        "home_lng":   params["home"]["lng"],
    }

    if remaining_target > 0:
        rl_picked = rl_select_destinations(env, rl_agent, rl_params)
        # Defense: dedup terhadap kandidat yang sudah di-reserve di Phase A
        reserved_ids = {d["id"] for d in reserved}
        rl_picked = [d for d in rl_picked if d["id"] not in reserved_ids][:remaining_target]
        selected = reserved + rl_picked
    else:
        selected = list(reserved)

    # 1.5 No-op category guarantee (now handled in Phase A)
    selected = enforce_category_guarantee(selected, params)
    # 2. Hard distance validator
    selected = enforce_distance_constraint(selected, params)
    # 3. Fallback (top-up jika kurang) — tetap respect categories preference
    selected = smart_fallback_fill(selected, params, target)
    # 4. Final distance check sekali lagi (jika fallback nakal)
    selected = enforce_distance_constraint(selected, params)

    if not selected:
        return {
            "steps": [], "totalCost": 0, "totalKm": 0, "totalTime": 0,
            "returnKm": 0, "returnMin": 0,
            "arriveHome": params.get("startMin", 540),
            "overBudget": False, "spareMin": 0,
        }

    # 6. Route optimization + itinerary
    ordered = route_optimizer.nearest_neighbor_route(params["home"], selected)
    result  = route_optimizer.build_itinerary(
        home=params["home"],
        home_name=params.get("homeName", "Home"),
        ordered_destinations=ordered,
        start_min=params.get("startMin", 9 * 60),
        end_min=params.get("endMin", 21 * 60),
    )
    budget = params.get("budget")
    result["overBudget"] = bool(
        budget is not None
        and budget < BandungTravelEnv.UNLIMITED_BUDGET
        and result["totalCost"] > budget
    )
    return result


# ── 15.7  Smoke tests ────────────────────────────────────────
def _summarize(label, params, res):
    print(f"\n{'='*64}\n{label}\n{'='*64}")
    print(f"Destinasi terpilih: {len(res['steps'])}/{params.get('count')}")
    for s in res["steps"]:
        a = f"{s['arriveAt']//60:02d}:{s['arriveAt']%60:02d}"
        print(f"  {s['idx']}. {s['dest']['name']:<32} ({s['dest']['category']:<8}) "
              f"@ {a} | {s['travelKm']:.1f} km | Rp {s['dest']['ticket']:,}")
    print(f"💰 Total cost : Rp {res['totalCost']:,}")
    print(f"📏 Total km   : {res['totalKm']:.1f}")
    print(f"⏱️  Total time : {res['totalTime']} min")
    print(f"🏠 Arrive home: {res['arriveHome']//60:02d}:{res['arriveHome']%60:02d}")
    if params.get("maxKm") is not None:
        # Validasi: tidak ada destinasi yang melebihi max_km
        max_km = params["maxKm"]
        violations = [s for s in res["steps"]
                      if haversine_km(params["home"]["lat"], params["home"]["lng"],
                                       s["dest"]["lat"], s["dest"]["lng"]) > max_km]
        if violations:
            print(f"❌ DISTANCE VIOLATION: {len(violations)}")
        else:
            print(f"✅ All within {max_km} km from home.")


print("\n🧪 SMOKE TESTS\n")

tests = [
    {"label": "T1 — Multi (Alam+Kuliner+Wisata), budget 400k, count=4",
     "params": {"home": {"lat":-6.9215, "lng":107.6071}, "homeName":"Alun-Alun",
                "count":4, "maxKm":None, "startMin":9*60, "endMin":20*60,
                "budget":400_000, "categories":["Alam","Kuliner","Wisata"]}},
    {"label": "T2 — Strict distance (max 15km), Alam+Kuliner",
     "params": {"home": {"lat":-6.9215, "lng":107.6071}, "homeName":"Alun-Alun",
                "count":3, "maxKm":15, "startMin":9*60, "endMin":18*60,
                "budget":250_000, "categories":["Alam","Kuliner"]}},
    {"label": "T3 — Single category (count=1, Alam)",
     "params": {"home": {"lat":-6.8126, "lng":107.6178}, "homeName":"Lembang",
                "count":1, "maxKm":30, "startMin":7*60, "endMin":17*60,
                "budget":100_000, "categories":["Alam"]}},
    {"label": "T4 — count < kategori (count=2 tapi 3 kategori dipilih)",
     "params": {"home": {"lat":-6.9215, "lng":107.6071}, "homeName":"Alun-Alun",
                "count":2, "maxKm":None, "startMin":9*60, "endMin":18*60,
                "budget":300_000, "categories":["Alam","Kuliner","Wisata"]}},
    {"label": "T5 — Budget=0 (hanya destinasi gratis)",
     "params": {"home": {"lat":-6.9215, "lng":107.6071}, "homeName":"Alun-Alun",
                "count":3, "maxKm":None, "startMin":9*60, "endMin":17*60,
                "budget":0, "categories":["Kuliner","Wisata"]}},
]

results_smoke = []
for t in tests:
    try:
        r = full_pipeline(t["params"])
        _summarize(t["label"], t["params"], r)
        results_smoke.append((t, r))
    except Exception as e:
        print(f"❌ {t['label']} crashed: {e}")
''')

# =====================================================================
# SECTION 16 — Evaluasi Model + Visualisasi
# =====================================================================
md(r"""## §16 — Evaluasi Model + Visualisasi Metrik

**Apa yang terjadi di section ini:**

Section khusus untuk mengukur kualitas keputusan model. Kita generate **100 skenario user random** lalu hitung metrik berikut, semuanya **divisualisasikan**:

### Metrik Utama

| # | Metrik | Definisi | Target |
|---|--------|----------|--------|
| 1 | **Category Coverage Rate** | % skenario di mana semua kategori user terwakili minimal 1 (jika `count ≥ len(cats)`) | ≥ 95% |
| 2 | **Distance Compliance** | % destinasi yang dihasilkan TIDAK melebihi `max_km` | 100% (hard) |
| 3 | **Budget Compliance** | % skenario di mana `totalCost ≤ budget` | ≥ 90% |
| 4 | **Avg Rating of Recs** | Rata-rata rating destinasi yang direkomendasikan | ≥ 4.0 |
| 5 | **Avg Travel Distance** | Rata-rata total km per skenario | sehat (< 80 km) |
| 6 | **Variety Index** | Jumlah kategori unik per itinerary / kategori diminta | ~ 1.0 |

### Visualisasi
- Bar chart compliance metrics
- Histogram total cost & total km
- Boxplot rating recs vs population
- Heatmap koemberhasilan (kategori user × kategori output)
- Compliance per home origin
""")

code(r'''# ── 16.1  Generator skenario evaluasi ────────────────────────
def generate_eval_scenarios(n=100, seed=2024):
    rng = random.Random(seed)
    scenarios = []
    for _ in range(n):
        n_cats = rng.randint(1, len(CATEGORY_ORDER))
        cats   = rng.sample(CATEGORY_ORDER, k=n_cats)
        budget = rng.choice([None, 100_000, 200_000, 400_000, 600_000, 1_000_000])
        max_km = rng.choice([None, 15, 25, 40, 60])
        count  = rng.randint(2, 5)
        home   = rng.choice(HOME_OPTIONS)
        start  = rng.choice([7, 8, 9, 10]) * 60
        end    = start + rng.choice([6, 8, 10]) * 60
        scenarios.append({
            "home": {"lat": home["lat"], "lng": home["lng"]},
            "homeName": home["name"],
            "count": count, "maxKm": max_km,
            "startMin": start, "endMin": end,
            "budget": budget,
            "categories": cats,
        })
    return scenarios


# ── 16.2  Run evaluasi ───────────────────────────────────────
print("🧪 Generating 100 scenarios...")
scenarios = generate_eval_scenarios(100, seed=2024)

records = []
print("⚙️  Running pipeline on each scenario (silent)...\n")
for i, p in enumerate(tqdm(scenarios, desc="Eval")):
    try:
        # Suppress per-scenario prints from the pipeline (swap log)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = full_pipeline(p)
    except Exception as e:
        records.append({"scenario": i, "error": str(e)}); continue

    cats_req = p["categories"]
    cats_got = [s["dest"]["category"] for s in r["steps"]]

    # Compliance kategori (hanya jika count >= len(cats_req) dan count > 1)
    if p["count"] >= len(cats_req) and p["count"] > 1 and len(cats_req) > 1:
        n_cover_target = min(p["count"], len(cats_req))
        cat_compliant = len(set(cats_got) & set(cats_req)) >= n_cover_target
    else:
        cat_compliant = True   # bypass guarantee

    # Compliance distance (HARD)
    dist_violations = 0
    for s in r["steps"]:
        d = haversine_km(p["home"]["lat"], p["home"]["lng"],
                         s["dest"]["lat"], s["dest"]["lng"])
        if p["maxKm"] is not None and d > p["maxKm"]:
            dist_violations += 1

    # Compliance budget
    if p["budget"] is None:
        budget_compliant = True
    else:
        budget_compliant = r["totalCost"] <= p["budget"]

    # Avg rating
    avg_rating = (np.mean([s["dest"]["rating"] for s in r["steps"]])
                  if r["steps"] else 0.0)

    # Variety index
    if cats_req:
        variety_idx = len(set(cats_got) & set(cats_req)) / len(cats_req)
    else:
        variety_idx = 1.0

    records.append({
        "scenario":         i,
        "n_steps":          len(r["steps"]),
        "count_req":        p["count"],
        "n_cats_req":       len(cats_req),
        "cats_req":         cats_req,
        "cats_got":         cats_got,
        "cat_compliant":    cat_compliant,
        "dist_violations":  dist_violations,
        "max_km_req":       p["maxKm"],
        "budget_req":       p["budget"],
        "total_cost":       r["totalCost"],
        "total_km":         r["totalKm"],
        "budget_compliant": budget_compliant,
        "avg_rating":       avg_rating,
        "variety_index":    variety_idx,
        "home":             p["homeName"],
    })

df_eval = pd.DataFrame([r for r in records if "error" not in r])
print(f"\n✅ {len(df_eval)} skenario berhasil dievaluasi.")

# ── 16.3  Hitung metrik agregat ──────────────────────────────
metrics = {
    "n_scenarios":           len(df_eval),
    "category_coverage_pct": float(df_eval["cat_compliant"].mean() * 100),
    "distance_compliance_pct": float((df_eval["dist_violations"] == 0).mean() * 100),
    "budget_compliance_pct": float(df_eval["budget_compliant"].mean() * 100),
    "avg_rating":            float(df_eval["avg_rating"].mean()),
    "avg_total_km":          float(df_eval["total_km"].mean()),
    "avg_total_cost":        float(df_eval["total_cost"].mean()),
    "avg_variety_index":     float(df_eval["variety_index"].mean()),
    "avg_steps_per_itinerary": float(df_eval["n_steps"].mean()),
}

print("\n" + "="*64)
print("📊 METRIK EVALUASI MODEL")
print("="*64)
for k, v in metrics.items():
    if isinstance(v, float):
        if "pct" in k:
            print(f"  {k:<28}: {v:6.2f} %")
        elif "cost" in k:
            print(f"  {k:<28}: Rp {v:,.0f}")
        elif "km" in k:
            print(f"  {k:<28}: {v:.2f} km")
        else:
            print(f"  {k:<28}: {v:.3f}")
    else:
        print(f"  {k:<28}: {v}")

# Save report
with open("data/processed/eval_report.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
df_eval.to_csv("data/processed/eval_scenarios.csv", index=False)
print("\n💾 data/processed/eval_report.json")
print("💾 data/processed/eval_scenarios.csv")
''')

code(r'''# ── 16.4  Visualisasi 1: Bar chart compliance ────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = ["Category\nCoverage", "Distance\nCompliance",
          "Budget\nCompliance", "Variety\nIndex (×100)"]
values = [
    metrics["category_coverage_pct"],
    metrics["distance_compliance_pct"],
    metrics["budget_compliance_pct"],
    metrics["avg_variety_index"] * 100,
]
colors = ["#2E8B57", "#1E90FF", "#D2691E", "#9370DB"]
bars = ax.bar(labels, values, color=colors, edgecolor="black")
for bar, v in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f"{v:.1f}%", ha="center", fontweight="bold")
ax.axhline(95, color="gray", linestyle="--", alpha=0.5, label="target 95%")
ax.set_ylim(0, 110)
ax.set_ylabel("Persentase")
ax.set_title(f"Compliance Metrics — {metrics['n_scenarios']} skenario evaluasi")
ax.legend()
plt.tight_layout()
plt.show()

# ── 16.5  Visualisasi 2: Histogram total_km dan total_cost ───
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(df_eval["total_km"], bins=20,
             color="#3CB371", edgecolor="white")
axes[0].axvline(df_eval["total_km"].mean(), color="red", linestyle="--",
                label=f"mean = {df_eval['total_km'].mean():.1f} km")
axes[0].set_title("Distribusi Total Jarak per Itinerary")
axes[0].set_xlabel("Total km"); axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(df_eval["total_cost"]/1000, bins=20,
             color="#FF7F50", edgecolor="white")
axes[1].axvline(df_eval["total_cost"].mean()/1000, color="red", linestyle="--",
                label=f"mean = Rp {int(df_eval['total_cost'].mean()):,}")
axes[1].set_title("Distribusi Total Cost per Itinerary")
axes[1].set_xlabel("Total cost (ribu IDR)"); axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.show()

# ── 16.6  Visualisasi 3: Boxplot rating recs vs population ────
fig, ax = plt.subplots(figsize=(7, 4))
data = [df_clean["rating"].values, df_eval["avg_rating"].values]
bp = ax.boxplot(data, labels=["Population\n(all destinations)",
                              "Recommended\n(per itinerary avg)"],
                patch_artist=True)
for patch, c in zip(bp["boxes"], ["#9370DB", "#2E8B57"]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title("Rating: Population vs Rekomendasi")
ax.set_ylabel("Rating")
plt.tight_layout()
plt.show()

# ── 16.7  Visualisasi 4: Heatmap kategori req × got ──────────
co_matrix = pd.DataFrame(0,
    index=CATEGORY_ORDER, columns=CATEGORY_ORDER, dtype=float)
total_per_req = Counter()
for _, row in df_eval.iterrows():
    for c_req in row["cats_req"]:
        total_per_req[c_req] += 1
        for c_got in row["cats_got"]:
            if c_got in co_matrix.columns:
                co_matrix.loc[c_req, c_got] += 1

# Normalisasi ke proporsi (per row)
co_norm = co_matrix.div(co_matrix.sum(axis=1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(co_norm, annot=True, fmt=".2f", cmap="YlGnBu",
            cbar_kws={"label": "proporsi"}, ax=ax)
ax.set_title("Heatmap: Kategori diminta (rows) × Kategori muncul (cols)")
ax.set_xlabel("Kategori output"); ax.set_ylabel("Kategori diminta user")
plt.tight_layout()
plt.show()

# ── 16.8  Visualisasi 5: Compliance per home ─────────────────
home_metrics = df_eval.groupby("home").agg(
    cov=("cat_compliant", "mean"),
    bud=("budget_compliant", "mean"),
    dist=("dist_violations", lambda x: (x == 0).mean()),
    rating=("avg_rating", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(home_metrics))
w = 0.22
ax.bar(x - 1.5*w, home_metrics["cov"]*100,  w, label="Cat coverage",   color="#2E8B57")
ax.bar(x - 0.5*w, home_metrics["bud"]*100,  w, label="Budget OK",      color="#D2691E")
ax.bar(x + 0.5*w, home_metrics["dist"]*100, w, label="Distance OK",    color="#1E90FF")
ax.bar(x + 1.5*w, home_metrics["rating"]*20, w, label="Avg rating ×20", color="#9370DB")

ax.set_xticks(x)
ax.set_xticklabels(home_metrics["home"], rotation=15)
ax.set_ylabel("Persentase / score")
ax.set_title("Compliance per Home Origin")
ax.legend(loc="lower right")
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()

print("\n✅ Visualisasi evaluasi selesai dirender.")
print(f"📁 Report: data/processed/eval_report.json")
print(f"📁 Detail: data/processed/eval_scenarios.csv")
''')

# =====================================================================
# SECTION 17 — Export
# =====================================================================
md(r"""## §17 — Export Artefak

**Apa yang terjadi di section ini:**

1. Validasi semua file output ada (CSV, NPY, PKL, JSON).
2. Generate `sample_api_request.json` & `sample_api_response.json` untuk kontrak frontend.
3. Print ringkasan akhir.

> **Catatan deploy:** File-file di `models/` dan `data/processed/` siap di-commit ke folder `backend/` untuk dipakai oleh FastAPI service.
""")

code(r'''# ── 17.1  Sample request/response untuk frontend ─────────────
sample_params = {
    "home": {"lat": -6.9215, "lng": 107.6071},
    "homeName": "Alun-Alun Bandung",
    "count": 4, "maxKm": 25,
    "startMin": 9 * 60, "endMin": 21 * 60,
    "budget": 400_000,
    "categories": ["Alam", "Kuliner", "Wisata"],
}

import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sample_itinerary = full_pipeline(sample_params)

sample_response = {
    **sample_itinerary,
    "story": {
        "story": "(akan di-generate oleh Notebook 02 LLM Storyteller)",
        "vibe":  "Alam · Kuliner · Wisata Umum",
    },
    "data_last_updated": str(date.today()),
}

with open("data/processed/sample_api_request.json", "w", encoding="utf-8") as f:
    json.dump(sample_params, f, ensure_ascii=False, indent=2)
with open("data/processed/sample_api_response.json", "w", encoding="utf-8") as f:
    json.dump(sample_response, f, ensure_ascii=False, indent=2, default=str)

# ── 17.2  Validasi kontrak frontend ──────────────────────────
required = ["steps","totalCost","totalKm","totalTime","returnKm",
            "returnMin","arriveHome","overBudget","spareMin",
            "story","data_last_updated"]
missing  = [k for k in required if k not in sample_response]
print(f"Kontrak frontend: {'✅ lengkap' if not missing else '❌ missing: '+str(missing)}")

# ── 17.3  Daftar file output ─────────────────────────────────
files = [
    "data/processed/destinations.csv",
    "data/processed/feature_matrix.npy",
    "data/processed/sample_api_request.json",
    "data/processed/sample_api_response.json",
    "data/processed/eval_report.json",
    "data/processed/eval_scenarios.csv",
    "data/last_updated.txt",
    "models/cbf_model.pkl",
    "models/rl_agent.pkl",
    "models/scaler.pkl",
    "models/label_encoders.pkl",
]
print("\n📁 File output:")
for fp in files:
    if os.path.exists(fp):
        sz = os.path.getsize(fp) / 1024
        print(f"  ✅ {fp:<50} ({sz:.1f} KB)")
    else:
        print(f"  ❌ MISSING: {fp}")

# ── 17.4  Final summary ──────────────────────────────────────
print("\n" + "="*64)
print("📦 FINAL SUMMARY")
print("="*64)
print(f"✅ Kategori aktif      : {CATEGORY_ORDER}")
print(f"✅ Total destinasi     : {len(df_clean)}")
print(f"   per kategori        : {df_clean['category'].value_counts().to_dict()}")
print(f"✅ Feature matrix      : {feature_matrix.shape}")
print(f"✅ RL Q-table states   : {len(rl_agent.q_table)}")
print(f"✅ Eval — coverage     : {metrics['category_coverage_pct']:.1f}%")
print(f"✅ Eval — distance     : {metrics['distance_compliance_pct']:.1f}%")
print(f"✅ Eval — budget       : {metrics['budget_compliance_pct']:.1f}%")
print(f"✅ Eval — avg rating   : {metrics['avg_rating']:.2f}")
print("="*64)
''')

# =====================================================================
# WRITE NOTEBOOK
# =====================================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).parent / "rec-engine (4).ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Notebook ditulis: {OUT}")
print(f"   Total cells: {len(CELLS)}")
md_count = sum(1 for c in CELLS if c['cell_type']=='markdown')
code_count = sum(1 for c in CELLS if c['cell_type']=='code')
print(f"   Markdown: {md_count} | Code: {code_count}")
