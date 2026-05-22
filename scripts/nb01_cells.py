"""Cell definitions for notebook 01_recommendation_engine.ipynb."""
from _nb_builder import md, code


# ---------- CELL 0: Title ----------
CELL_0 = md("""# 🗺️ Bandung AI Travel Agent — Recommendation Engine
## Notebook 01: Data Pipeline + Model Training

**Capstone Project · Telkom University · Program Studi Data Science**

Notebook ini membangun pipeline lengkap dari:
- Crawling data destinasi wisata Bandung via Overpass API (OpenStreetMap)
- Feature engineering & preprocessing
- Training Content-Based Filtering (CBF) menggunakan cosine similarity
- Training Multi-Agent Reinforcement Learning (RL) dengan simulated environment
- Route optimization menggunakan TSP nearest-neighbor heuristic
- Evaluasi dan export model

**Output:** `models/cbf_model.pkl`, `models/rl_agent.pkl`, `models/scaler.pkl`, `data/processed/destinations.csv`

**Estimasi waktu runtime:** 5–15 menit (tergantung kecepatan internet untuk crawling)
""")


# ---------- CELL 1: Setup ----------
CELL_1 = code("""# ============================================================
# CELL 1 — Setup & Instalasi Dependencies
# ============================================================
# Jalankan baris ini sekali jika package belum terpasang.
# !pip install -q requests pandas numpy scikit-learn matplotlib seaborn tqdm

import os
import re
import sys
import json
import math
import time
import pickle
import random
import urllib.parse
import warnings
from pathlib import Path
from datetime import date
from collections import defaultdict, deque, Counter

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

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"✅ Random seed set: {RANDOM_SEED}")

# Pastikan working directory adalah ROOT proyek (bukan folder notebooks/)
if Path.cwd().name == "notebooks":
    os.chdir("..")
print(f"📂 CWD: {Path.cwd()}")

# Buat semua direktori yang dibutuhkan
for d in ["data/raw", "data/processed", "models", "notebooks"]:
    Path(d).mkdir(parents=True, exist_ok=True)

print("✅ Struktur folder berhasil dibuat")
print("📁 data/raw/        → hasil crawling mentah")
print("📁 data/processed/  → dataset bersih .csv")
print("📁 models/          → model .pkl tersimpan")
""")


# ---------- CELL 2: Markdown Phase 1 ----------
CELL_2 = md("""## 📡 Fase 1: Data Collection — Overpass API (OpenStreetMap)

Menggunakan Overpass API untuk mengambil POI (Point of Interest) wisata Bandung.
Bounding box Bandung Raya: `-7.2500, 107.3500, -6.7500, 107.9000`

| OSM Tag | Kategori Proyek |
|---|---|
| `tourism=viewpoint`, `natural=*`, `leisure=park` | Alam |
| `amenity=restaurant`, `amenity=cafe`, `amenity=food_court` | Kuliner |
| `tourism=museum`, `historic=*`, `amenity=place_of_worship` | Budaya |
| `tourism=theme_park`, `tourism=attraction`, `leisure=*` | Wisata |
| `shop=mall`, `amenity=marketplace` | Belanja |
""")


# ---------- CELL 3: Overpass crawling ----------
CELL_3 = code('''# ============================================================
# CELL 3 — Crawling Overpass API
# ============================================================
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BANDUNG_BBOX = "-7.2500,107.3500,-6.7500,107.9000"  # south,west,north,east

OSM_CATEGORY_MAP = {
    # Alam
    "viewpoint": "Alam", "natural": "Alam", "park": "Alam",
    "waterfall": "Alam", "forest": "Alam",
    # Kuliner
    "restaurant": "Kuliner", "cafe": "Kuliner", "food_court": "Kuliner",
    "fast_food": "Kuliner",
    # Budaya
    "museum": "Budaya", "memorial": "Budaya", "artwork": "Budaya",
    "place_of_worship": "Budaya", "historic": "Budaya",
    # Wisata
    "theme_park": "Wisata", "attraction": "Wisata", "zoo": "Wisata",
    "aquarium": "Wisata", "amusement_ride": "Wisata",
    # Belanja
    "mall": "Belanja", "marketplace": "Belanja", "department_store": "Belanja",
}

# Query template per kategori (kombinasi nodes & ways yang relevan)
CATEGORY_QUERIES = {
    "Alam": """
        node["tourism"="viewpoint"]({bbox});
        node["natural"="peak"]({bbox});
        node["natural"="waterfall"]({bbox});
        way["leisure"="park"]({bbox});
        way["natural"="wood"]({bbox});
    """,
    "Kuliner": """
        node["amenity"="restaurant"]({bbox});
        node["amenity"="cafe"]({bbox});
        node["amenity"="food_court"]({bbox});
        node["amenity"="fast_food"]({bbox});
    """,
    "Budaya": """
        node["tourism"="museum"]({bbox});
        node["historic"~"."]({bbox});
        node["amenity"="place_of_worship"]({bbox});
        node["tourism"="memorial"]({bbox});
        way["tourism"="museum"]({bbox});
    """,
    "Wisata": """
        node["tourism"="theme_park"]({bbox});
        node["tourism"="attraction"]({bbox});
        node["tourism"="zoo"]({bbox});
        way["tourism"="theme_park"]({bbox});
        way["leisure"="garden"]({bbox});
    """,
    "Belanja": """
        node["shop"="mall"]({bbox});
        node["amenity"="marketplace"]({bbox});
        way["shop"="mall"]({bbox});
        way["shop"="department_store"]({bbox});
    """,
}


def query_overpass(query_str: str) -> dict:
    """Kirim Overpass QL query, return parsed JSON (dict kosong jika error)."""
    try:
        full_query = f"[out:json][timeout:30];({query_str});out center 60;"
        resp = requests.post(OVERPASS_URL, data={"data": full_query}, timeout=30)
        resp.raise_for_status()
        time.sleep(2)  # rate limiting
        return resp.json()
    except Exception as e:
        print(f"  ⚠️  Overpass error: {e}")
        time.sleep(2)
        return {}


def build_overpass_query(category: str, bbox: str) -> str:
    template = CATEGORY_QUERIES.get(category, "")
    return template.format(bbox=bbox)


def generate_gmaps_url(name: str) -> str:
    query = urllib.parse.quote(f"{name}, Bandung")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


def fetch_all_categories() -> pd.DataFrame:
    """Loop seluruh kategori dan gabungkan hasil crawling jadi satu DataFrame."""
    rows = []
    for cat in ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]:
        q = build_overpass_query(cat, BANDUNG_BBOX)
        data = query_overpass(q)
        elements = data.get("elements", [])
        n_before = len(rows)
        for el in elements:
            tags = el.get("tags", {}) or {}
            name = tags.get("name") or tags.get("name:id")
            if not name or len(str(name).strip()) < 3:
                continue
            # koordinat: nodes punya lat/lon langsung; ways pakai center
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat is None or lon is None:
                continue
            rows.append({
                "id": f"osm-{el.get('type','n')}-{el.get('id')}",
                "name": str(name).strip(),
                "category": cat,
                "lat": float(lat),
                "lng": float(lon),
                "source": "overpass",
            })
        print(f"  [{cat}] {len(rows) - n_before} record dari Overpass")
    df = pd.DataFrame(rows)
    return df


print("🌍 Mulai crawling Overpass API…")
try:
    df_osm = fetch_all_categories()
except Exception as e:
    print(f"  ⚠️  Crawling gagal total ({e}). Lanjut dengan dataset kosong.")
    df_osm = pd.DataFrame(columns=["id", "name", "category", "lat", "lng", "source"])

if len(df_osm):
    df_osm.to_csv("data/raw/osm_raw.csv", index=False)
    print(f"\\n✅ {len(df_osm)} record tersimpan di data/raw/osm_raw.csv")
    print(df_osm.groupby("category").size())
    print("\\n5 baris pertama:")
    print(df_osm.head())
else:
    print("ℹ️  Tidak ada data Overpass — pipeline akan tetap jalan dari seed + manual.")
''')


# ---------- CELL 4: Seed + manual additions ----------
CELL_4 = code('''# ============================================================
# CELL 4 — Seed data (16 dari frontend) + tambahan manual
# ============================================================
SEED_DESTINATIONS = [
    # ===== ALAM (5) =====
    {"id": "kawah-putih", "name": "Kawah Putih", "category": "Alam",
     "desc": "Danau kawah vulkanik dengan air berwarna putih kehijauan di ketinggian 2430 mdpl",
     "ticket": 81000, "duration": 120, "lat": -7.1660, "lng": 107.4019,
     "rating": 4.6, "tags": ["sunrise", "fotogenik", "dingin"]},
    {"id": "kebun-teh-rancabali", "name": "Kebun Teh Rancabali", "category": "Alam",
     "desc": "Hamparan kebun teh hijau di Ciwidey dengan udara sejuk dan pemandangan luas",
     "ticket": 25000, "duration": 90, "lat": -7.1432, "lng": 107.4106,
     "rating": 4.5, "tags": ["hijau", "fotogenik", "sejuk"]},
    {"id": "stone-garden-padalarang", "name": "Stone Garden Padalarang", "category": "Alam",
     "desc": "Taman batu purba dengan formasi karst unik di kawasan Citatah, Padalarang",
     "ticket": 15000, "duration": 90, "lat": -6.8323, "lng": 107.4709,
     "rating": 4.4, "tags": ["batu", "fotogenik", "karst"]},
    {"id": "tebing-keraton", "name": "Tebing Keraton", "category": "Alam",
     "desc": "Tebing dengan view hutan pinus Dago Pakar, populer untuk sunrise",
     "ticket": 15000, "duration": 90, "lat": -6.8359, "lng": 107.6630,
     "rating": 4.5, "tags": ["sunrise", "hiking", "tebing"]},
    {"id": "tangkuban-perahu", "name": "Tangkuban Perahu", "category": "Alam",
     "desc": "Gunung berapi aktif dengan kawah ratu yang ikonik di utara Bandung",
     "ticket": 30000, "duration": 120, "lat": -6.7597, "lng": 107.6098,
     "rating": 4.5, "tags": ["gunung", "kawah", "legendaris"]},

    # ===== KULINER (3) =====
    {"id": "floating-market-lembang", "name": "Floating Market Lembang", "category": "Kuliner",
     "desc": "Pasar terapung dengan beragam jajanan tradisional Sunda di atas perahu",
     "ticket": 30000, "duration": 90, "lat": -6.8121, "lng": 107.6178,
     "rating": 4.3, "tags": ["jajanan", "keluarga", "fotogenik"]},
    {"id": "jalan-braga", "name": "Jalan Braga", "category": "Kuliner",
     "desc": "Kawasan bersejarah dengan kafe, resto, dan bangunan kolonial khas Bandung tempo dulu",
     "ticket": 0, "duration": 90, "lat": -6.9160, "lng": 107.6094,
     "rating": 4.4, "tags": ["heritage", "kafe", "malam"]},
    {"id": "punclut-bukit-bintang", "name": "Punclut Bukit Bintang", "category": "Kuliner",
     "desc": "Spot makan dengan view kota Bandung dari ketinggian, populer untuk dinner romantis",
     "ticket": 0, "duration": 120, "lat": -6.8528, "lng": 107.6235,
     "rating": 4.3, "tags": ["view", "romantis", "malam"]},

    # ===== BUDAYA (3) =====
    {"id": "saung-angklung-udjo", "name": "Saung Angklung Udjo", "category": "Budaya",
     "desc": "Pusat pertunjukan dan pelatihan angklung — warisan budaya Sunda dunia",
     "ticket": 75000, "duration": 90, "lat": -6.8924, "lng": 107.6537,
     "rating": 4.7, "tags": ["pertunjukan", "edukasi", "tradisi"]},
    {"id": "gedung-sate", "name": "Gedung Sate", "category": "Budaya",
     "desc": "Gedung pemerintahan ikonik dengan arsitektur Indo-Eropa dan ornamen tusuk sate",
     "ticket": 5000, "duration": 60, "lat": -6.9024, "lng": 107.6188,
     "rating": 4.5, "tags": ["heritage", "ikon", "arsitektur"]},
    {"id": "museum-geologi", "name": "Museum Geologi", "category": "Budaya",
     "desc": "Museum koleksi fosil, mineral, dan sejarah geologi nusantara",
     "ticket": 3000, "duration": 90, "lat": -6.9006, "lng": 107.6225,
     "rating": 4.5, "tags": ["edukasi", "fosil", "sains"]},

    # ===== WISATA (3) =====
    {"id": "farmhouse-lembang", "name": "Farmhouse Lembang", "category": "Wisata",
     "desc": "Wisata bertema Eropa dengan rumah hobbit, peternakan mini, dan susu segar",
     "ticket": 35000, "duration": 120, "lat": -6.8324, "lng": 107.6122,
     "rating": 4.3, "tags": ["keluarga", "fotogenik", "eropa"]},
    {"id": "dusun-bambu", "name": "Dusun Bambu", "category": "Wisata",
     "desc": "Resort & taman wisata keluarga bertema alam pegunungan dengan playground",
     "ticket": 25000, "duration": 150, "lat": -6.8005, "lng": 107.5701,
     "rating": 4.4, "tags": ["keluarga", "alam", "kuliner"]},
    {"id": "trans-studio-bandung", "name": "Trans Studio Bandung", "category": "Wisata",
     "desc": "Theme park indoor terbesar di Indonesia dengan wahana berstandar internasional",
     "ticket": 280000, "duration": 240, "lat": -6.9273, "lng": 107.6364,
     "rating": 4.5, "tags": ["theme-park", "keluarga", "indoor"]},

    # ===== BELANJA (2) =====
    {"id": "cihampelas-walk", "name": "Cihampelas Walk", "category": "Belanja",
     "desc": "Mall outdoor populer di Cihampelas, perpaduan factory outlet dan gaya hidup",
     "ticket": 0, "duration": 120, "lat": -6.8946, "lng": 107.6029,
     "rating": 4.3, "tags": ["mall", "kuliner", "fashion"]},
    {"id": "pasar-baru-trade-center", "name": "Pasar Baru Trade Center", "category": "Belanja",
     "desc": "Pusat grosir tekstil dan fashion legendaris di pusat kota Bandung",
     "ticket": 0, "duration": 120, "lat": -6.9166, "lng": 107.6075,
     "rating": 4.2, "tags": ["grosir", "fashion", "tradisional"]},
]

# ---------- Tambahan manual (≥34 destinasi) untuk total ≥50 ----------
EXTRA_DESTINATIONS = [
    # ALAM (8 tambahan)
    {"id": "situ-patenggang", "name": "Situ Patenggang", "category": "Alam",
     "desc": "Danau alami di Ciwidey dengan legenda Batu Cinta dan perahu wisata",
     "ticket": 30000, "duration": 120, "lat": -7.1769, "lng": 107.3656,
     "rating": 4.5, "tags": ["danau", "legenda", "perahu"]},
    {"id": "curug-dago", "name": "Curug Dago", "category": "Alam",
     "desc": "Air terjun di kawasan Taman Hutan Raya Djuanda dengan jejak prasasti raja Thailand",
     "ticket": 17000, "duration": 90, "lat": -6.8540, "lng": 107.6299,
     "rating": 4.2, "tags": ["air-terjun", "hiking", "sejarah"]},
    {"id": "curug-cimahi", "name": "Curug Cimahi", "category": "Alam",
     "desc": "Air terjun pelangi setinggi 87 meter di kawasan Cisarua",
     "ticket": 15000, "duration": 90, "lat": -6.8033, "lng": 107.5503,
     "rating": 4.4, "tags": ["air-terjun", "pelangi", "fotogenik"]},
    {"id": "tahura-djuanda", "name": "Taman Hutan Raya Ir. H. Djuanda", "category": "Alam",
     "desc": "Hutan kota dengan goa Jepang, goa Belanda, dan trek alam di Dago Pakar",
     "ticket": 17000, "duration": 150, "lat": -6.8579, "lng": 107.6299,
     "rating": 4.5, "tags": ["hutan", "goa", "hiking"]},
    {"id": "gunung-batu-lembang", "name": "Gunung Batu Lembang", "category": "Alam",
     "desc": "Bukit batu vulkanik kecil dengan view 360 derajat Lembang dan sekitarnya",
     "ticket": 10000, "duration": 90, "lat": -6.8208, "lng": 107.6238,
     "rating": 4.3, "tags": ["sunrise", "hiking", "view"]},
    {"id": "situ-lembang", "name": "Situ Lembang", "category": "Alam",
     "desc": "Danau yang dikelilingi hutan pinus, tenang dan asri di kawasan Lembang",
     "ticket": 10000, "duration": 90, "lat": -6.7880, "lng": 107.6063,
     "rating": 4.3, "tags": ["danau", "tenang", "alam"]},
    {"id": "curug-malela", "name": "Curug Malela", "category": "Alam",
     "desc": "Air terjun bertingkat luas yang dijuluki Niagara mini-nya Bandung Barat",
     "ticket": 15000, "duration": 180, "lat": -6.9181, "lng": 107.3119,
     "rating": 4.5, "tags": ["air-terjun", "petualangan", "alam-liar"]},
    {"id": "ranca-upas", "name": "Ranca Upas", "category": "Alam",
     "desc": "Bumi perkemahan dengan penangkaran rusa di tengah hutan pinus Ciwidey",
     "ticket": 25000, "duration": 150, "lat": -7.1432, "lng": 107.3779,
     "rating": 4.4, "tags": ["camping", "rusa", "hutan"]},

    # KULINER (7 tambahan)
    {"id": "warung-nasi-ampera", "name": "Warung Nasi Ampera", "category": "Kuliner",
     "desc": "Rumah makan Sunda legendaris dengan lalapan dan sambal khas",
     "ticket": 50000, "duration": 60, "lat": -6.9182, "lng": 107.6101,
     "rating": 4.3, "tags": ["sunda", "legendaris", "halal"]},
    {"id": "mie-kocok-mang-dadeng", "name": "Mie Kocok Mang Dadeng", "category": "Kuliner",
     "desc": "Mie kocok khas Bandung dengan kuah kaldu sapi gurih sejak 1960-an",
     "ticket": 35000, "duration": 45, "lat": -6.9075, "lng": 107.6094,
     "rating": 4.5, "tags": ["mie", "khas-bandung", "legendaris"]},
    {"id": "kupat-tahu-gempol", "name": "Kupat Tahu Gempol", "category": "Kuliner",
     "desc": "Kupat tahu dengan bumbu kacang khas pasar Gempol, sarapan favorit warga",
     "ticket": 20000, "duration": 45, "lat": -6.9075, "lng": 107.6135,
     "rating": 4.4, "tags": ["sarapan", "tradisional", "murah"]},
    {"id": "batagor-riri", "name": "Batagor Riri", "category": "Kuliner",
     "desc": "Batagor renyah dengan saus kacang manis-pedas khas Burangrang",
     "ticket": 25000, "duration": 45, "lat": -6.9223, "lng": 107.6116,
     "rating": 4.4, "tags": ["batagor", "khas-bandung", "jajanan"]},
    {"id": "warung-koffie-batavia", "name": "Warung Koffie Batavia", "category": "Kuliner",
     "desc": "Kafe bernuansa kolonial dengan aneka kopi nusantara di Jalan Braga",
     "ticket": 60000, "duration": 90, "lat": -6.9161, "lng": 107.6094,
     "rating": 4.4, "tags": ["kopi", "heritage", "kafe"]},
    {"id": "sindang-reret", "name": "Restoran Sindang Reret", "category": "Kuliner",
     "desc": "Restoran Sunda dengan saung lesehan dan menu nasi liwet di Cikole",
     "ticket": 75000, "duration": 90, "lat": -6.7778, "lng": 107.6499,
     "rating": 4.3, "tags": ["sunda", "saung", "keluarga"]},
    {"id": "the-valley-bistro", "name": "The Valley Bistro Cafe", "category": "Kuliner",
     "desc": "Bistro berlokasi di lembah Dago dengan view dramatis kota Bandung",
     "ticket": 100000, "duration": 90, "lat": -6.8580, "lng": 107.6290,
     "rating": 4.4, "tags": ["bistro", "view", "romantis"]},

    # BUDAYA (7 tambahan)
    {"id": "museum-konferensi-asia-afrika", "name": "Museum Konferensi Asia Afrika", "category": "Budaya",
     "desc": "Museum sejarah KAA 1955 di Gedung Merdeka, Jalan Asia Afrika",
     "ticket": 0, "duration": 90, "lat": -6.9211, "lng": 107.6098,
     "rating": 4.6, "tags": ["sejarah", "diplomasi", "gratis"]},
    {"id": "museum-sri-baduga", "name": "Museum Sri Baduga", "category": "Budaya",
     "desc": "Museum koleksi etnografi Sunda di kawasan Tegallega",
     "ticket": 5000, "duration": 90, "lat": -6.9367, "lng": 107.5996,
     "rating": 4.3, "tags": ["sunda", "edukasi", "etnografi"]},
    {"id": "museum-pos-indonesia", "name": "Museum Pos Indonesia", "category": "Budaya",
     "desc": "Koleksi prangko, alat pos antik, dan sejarah perposan nasional",
     "ticket": 0, "duration": 60, "lat": -6.9018, "lng": 107.6193,
     "rating": 4.2, "tags": ["sejarah", "prangko", "gratis"]},
    {"id": "monumen-perjuangan-jabar", "name": "Monumen Perjuangan Rakyat Jawa Barat", "category": "Budaya",
     "desc": "Monumen perjuangan dengan diorama sejarah di seberang Gasibu",
     "ticket": 5000, "duration": 60, "lat": -6.8924, "lng": 107.6207,
     "rating": 4.3, "tags": ["sejarah", "monumen", "edukasi"]},
    {"id": "taman-budaya-jawa-barat", "name": "Taman Budaya Jawa Barat", "category": "Budaya",
     "desc": "Pusat pertunjukan seni Sunda dengan amphitheater dan galeri",
     "ticket": 10000, "duration": 90, "lat": -6.8556, "lng": 107.6202,
     "rating": 4.2, "tags": ["pertunjukan", "seni", "sunda"]},
    {"id": "vihara-vipassana-graha", "name": "Vihara Vipassana Graha", "category": "Budaya",
     "desc": "Vihara Buddhis dengan arsitektur khas dan taman meditasi di Lembang",
     "ticket": 0, "duration": 60, "lat": -6.7949, "lng": 107.6242,
     "rating": 4.5, "tags": ["religi", "tenang", "arsitektur"]},
    {"id": "masjid-raya-bandung", "name": "Masjid Raya Bandung", "category": "Budaya",
     "desc": "Masjid agung kota Bandung dengan dua menara ikonik di Alun-Alun",
     "ticket": 0, "duration": 60, "lat": -6.9215, "lng": 107.6071,
     "rating": 4.7, "tags": ["religi", "ikon", "alun-alun"]},

    # WISATA (7 tambahan)
    {"id": "the-lodge-maribaya", "name": "The Lodge Maribaya", "category": "Wisata",
     "desc": "Spot foto ikonik dengan ayunan langit dan sky bridge di hutan pinus",
     "ticket": 50000, "duration": 120, "lat": -6.8154, "lng": 107.6593,
     "rating": 4.4, "tags": ["fotogenik", "ayunan", "hutan"]},
    {"id": "kampung-daun", "name": "Kampung Daun Culture Gallery", "category": "Wisata",
     "desc": "Resto-galeri bertema desa Sunda dengan saung di tengah hutan Cihideung",
     "ticket": 30000, "duration": 120, "lat": -6.8403, "lng": 107.5786,
     "rating": 4.5, "tags": ["sunda", "saung", "romantis"]},
    {"id": "observatorium-bosscha", "name": "Observatorium Bosscha", "category": "Wisata",
     "desc": "Observatorium tertua di Indonesia dengan teleskop bersejarah di Lembang",
     "ticket": 20000, "duration": 90, "lat": -6.8244, "lng": 107.6166,
     "rating": 4.5, "tags": ["edukasi", "astronomi", "sejarah"]},
    {"id": "kebun-binatang-bandung", "name": "Kebun Binatang Bandung", "category": "Wisata",
     "desc": "Kebun binatang klasik kota dengan koleksi satwa nusantara",
     "ticket": 50000, "duration": 150, "lat": -6.8919, "lng": 107.6067,
     "rating": 4.0, "tags": ["keluarga", "satwa", "edukasi"]},
    {"id": "taman-film-bandung", "name": "Taman Film Bandung", "category": "Wisata",
     "desc": "Taman tematik di bawah jalan layang Pasupati dengan layar film outdoor",
     "ticket": 0, "duration": 60, "lat": -6.8959, "lng": 107.6125,
     "rating": 4.2, "tags": ["taman", "gratis", "kreatif"]},
    {"id": "sky-lantern-lembang", "name": "Lembang Park & Zoo", "category": "Wisata",
     "desc": "Taman wisata dengan kebun binatang mini dan wahana keluarga di Lembang",
     "ticket": 50000, "duration": 150, "lat": -6.8137, "lng": 107.6172,
     "rating": 4.3, "tags": ["keluarga", "satwa", "wahana"]},
    {"id": "orchid-forest-cikole", "name": "Orchid Forest Cikole", "category": "Wisata",
     "desc": "Taman hutan pinus dengan koleksi anggrek dan wood bridge fotogenik",
     "ticket": 50000, "duration": 120, "lat": -6.7796, "lng": 107.6531,
     "rating": 4.5, "tags": ["fotogenik", "anggrek", "hutan-pinus"]},

    # BELANJA (7 tambahan)
    {"id": "btc-fashion-mall", "name": "BTC Fashion Mall", "category": "Belanja",
     "desc": "Mall fashion grosir di kawasan Pasteur dengan pilihan harga ekonomis",
     "ticket": 0, "duration": 120, "lat": -6.8917, "lng": 107.5745,
     "rating": 4.0, "tags": ["fashion", "grosir", "mall"]},
    {"id": "rumah-mode", "name": "Outlet Rumah Mode", "category": "Belanja",
     "desc": "Factory outlet veteran di Setiabudhi dengan koleksi branded berdiskon",
     "ticket": 0, "duration": 120, "lat": -6.8601, "lng": 107.5926,
     "rating": 4.3, "tags": ["factory-outlet", "branded", "diskon"]},
    {"id": "23-paskal", "name": "23 Paskal Shopping Center", "category": "Belanja",
     "desc": "Mall modern dekat stasiun dengan tenant fashion, kuliner, dan bioskop",
     "ticket": 0, "duration": 120, "lat": -6.9135, "lng": 107.5969,
     "rating": 4.4, "tags": ["mall", "modern", "kuliner"]},
    {"id": "pasar-baru-lembang", "name": "Pasar Baru Lembang", "category": "Belanja",
     "desc": "Pasar tradisional Lembang dengan sayur, buah, dan oleh-oleh khas",
     "ticket": 0, "duration": 90, "lat": -6.8126, "lng": 107.6178,
     "rating": 4.2, "tags": ["pasar", "tradisional", "oleh-oleh"]},
    {"id": "cibaduyut", "name": "Cibaduyut Shoes Center", "category": "Belanja",
     "desc": "Sentra produksi sepatu kulit handmade tertua di Indonesia",
     "ticket": 0, "duration": 120, "lat": -6.9624, "lng": 107.6021,
     "rating": 4.2, "tags": ["sepatu", "kulit", "handmade"]},
    {"id": "paris-van-java", "name": "Paris Van Java Mall", "category": "Belanja",
     "desc": "Mall lifestyle outdoor di Sukajadi dengan suasana al-fresco",
     "ticket": 0, "duration": 120, "lat": -6.8908, "lng": 107.5957,
     "rating": 4.5, "tags": ["mall", "outdoor", "lifestyle"]},
    {"id": "trans-studio-mall", "name": "Trans Studio Mall Bandung", "category": "Belanja",
     "desc": "Mall premium yang terhubung dengan Trans Studio theme park",
     "ticket": 0, "duration": 120, "lat": -6.9265, "lng": 107.6364,
     "rating": 4.5, "tags": ["mall", "premium", "kuliner"]},
]

manual_df = pd.DataFrame(SEED_DESTINATIONS + EXTRA_DESTINATIONS)
manual_df["source"] = "manual"
print(f"📝 Seed: {len(SEED_DESTINATIONS)} | Extra: {len(EXTRA_DESTINATIONS)} | Total manual: {len(manual_df)}")

# ---------- Merge dengan hasil Overpass ----------
def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def fuzzy_match(a: str, b: str) -> float:
    """Skor kesamaan berbasis token jaccard (0..1) — sederhana tapi cukup."""
    ta = set(re.findall(r"[a-z0-9]+", str(a).lower()))
    tb = set(re.findall(r"[a-z0-9]+", str(b).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def merge_with_dedup(manual: pd.DataFrame, osm: pd.DataFrame) -> pd.DataFrame:
    """Merge manual (priority) + osm, drop duplikat fuzzy >0.8."""
    if osm is None or len(osm) == 0:
        return manual.copy()
    manual_names = manual["name"].tolist()
    keep_rows = []
    for _, r in osm.iterrows():
        nm = r["name"]
        if any(fuzzy_match(nm, m) > 0.8 for m in manual_names):
            continue
        keep_rows.append(r)
    osm_unique = pd.DataFrame(keep_rows)
    return pd.concat([manual, osm_unique], ignore_index=True, sort=False)


df_enriched = merge_with_dedup(manual_df, df_osm if "df_osm" in dir() else pd.DataFrame())

# ---------- Isi field default untuk record OSM ----------
df_enriched["id"] = df_enriched.apply(
    lambda r: r["id"] if pd.notna(r.get("id")) and str(r.get("id")).strip() else slugify(r["name"]),
    axis=1,
)
df_enriched["desc"] = df_enriched.get("desc")
df_enriched["desc"] = df_enriched["desc"].fillna(
    df_enriched["category"].apply(lambda c: f"Destinasi wisata {c} di Bandung"))
df_enriched["ticket"] = df_enriched.get("ticket")
df_enriched["duration"] = df_enriched.get("duration")
df_enriched["rating"] = df_enriched.get("rating")
df_enriched["tags"] = df_enriched.get("tags")
df_enriched["tags"] = df_enriched["tags"].apply(lambda x: x if isinstance(x, list) else [])
df_enriched["gmaps_url"] = df_enriched["name"].apply(generate_gmaps_url)

# Drop duplikat id (jaga seed dulu)
df_enriched = df_enriched.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

df_enriched.to_csv("data/raw/destinations_enriched.csv", index=False)
print(f"✅ {len(df_enriched)} destinasi tersimpan ke data/raw/destinations_enriched.csv")
print(df_enriched.groupby("category").size())
''')


# ---------- CELL 5: Cleaning ----------
CELL_5 = code('''# ============================================================
# CELL 5 — Data Cleaning & Validation
# ============================================================
VALID_CATEGORIES = {"Alam", "Kuliner", "Budaya", "Wisata", "Belanja"}

# Default fallback jika semua nilai kategori NaN
DEFAULT_TICKET = {"Alam": 20000, "Kuliner": 35000, "Budaya": 10000, "Wisata": 50000, "Belanja": 0}
DEFAULT_DURATION = {"Alam": 120, "Kuliner": 60, "Budaya": 90, "Wisata": 120, "Belanja": 120}
DEFAULT_RATING = 4.2


def clean_destinations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Rename lon → lng (jika masih ada)
    if "lon" in df.columns and "lng" not in df.columns:
        df = df.rename(columns={"lon": "lng"})
    elif "lon" in df.columns and "lng" in df.columns:
        df["lng"] = df["lng"].fillna(df["lon"])
        df = df.drop(columns=["lon"])

    # 2. Filter kategori valid
    df = df[df["category"].isin(VALID_CATEGORIES)].copy()

    # 3. Fill missing — ticket, rating, duration per kategori
    for cat in VALID_CATEGORIES:
        mask = df["category"] == cat
        med_ticket = df.loc[mask, "ticket"].median()
        if pd.isna(med_ticket):
            med_ticket = DEFAULT_TICKET[cat]
        df.loc[mask, "ticket"] = df.loc[mask, "ticket"].fillna(med_ticket)

        mean_rating = df.loc[mask, "rating"].mean()
        if pd.isna(mean_rating):
            mean_rating = DEFAULT_RATING
        df.loc[mask, "rating"] = df.loc[mask, "rating"].fillna(mean_rating)

        med_dur = df.loc[mask, "duration"].median()
        if pd.isna(med_dur):
            med_dur = DEFAULT_DURATION[cat]
        df.loc[mask, "duration"] = df.loc[mask, "duration"].fillna(med_dur)

    df["desc"] = df["desc"].fillna(df["category"].apply(lambda c: f"Destinasi wisata {c} di Bandung"))
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    # 4. Type casting
    df["ticket"] = df["ticket"].astype(float).round().astype(int)
    df["duration"] = df["duration"].astype(float).round().astype(int)
    df["rating"] = df["rating"].astype(float).clip(1.0, 5.0)
    df["lat"] = df["lat"].astype(float)
    df["lng"] = df["lng"].astype(float)

    # 5. Validasi geografis & nama
    df = df[df["lat"].between(-7.4, -6.6)]
    df = df[df["lng"].between(107.2, 108.2)]
    df = df[df["name"].notna() & (df["name"].astype(str).str.len() >= 3)]

    # 6. Generate slug id jika kosong
    df["id"] = df.apply(
        lambda r: r["id"] if pd.notna(r["id"]) and str(r["id"]).strip() else slugify(r["name"]),
        axis=1,
    )

    # 7. Sort by category + rating desc
    df = df.sort_values(["category", "rating"], ascending=[True, False]).reset_index(drop=True)

    # 8. Pastikan gmaps_url ada
    if "gmaps_url" not in df.columns or df["gmaps_url"].isna().any():
        df["gmaps_url"] = df["gmaps_url"].fillna(df["name"].apply(generate_gmaps_url))

    return df


df_clean = clean_destinations(df_enriched)

# Assert minimal 50 destinasi & semua kategori
assert len(df_clean) >= 50, f"Dataset minimal 50 destinasi, dapat {len(df_clean)}"
assert set(df_clean["category"].unique()) == VALID_CATEGORIES, \\
    f"Kategori tidak lengkap: {df_clean['category'].unique()}"

# Pastikan 16 seed destinations tetap ada
seed_ids = {d["id"] for d in SEED_DESTINATIONS}
present = seed_ids & set(df_clean["id"].tolist())
assert len(present) == len(seed_ids), f"Seed hilang: {seed_ids - present}"

print(f"✅ Dataset valid: {len(df_clean)} destinasi")
print("\\nDistribusi per kategori:")
print(df_clean.groupby("category").size())

# Simpan
df_clean.to_csv("data/processed/destinations.csv", index=False)
with open("data/last_updated.txt", "w") as f:
    f.write(str(date.today()))

print("\\n💾 data/processed/destinations.csv tersimpan")
print(f"💾 data/last_updated.txt = {date.today()}")
print("\\nSample 5 destinasi teratas (rating tertinggi per kategori):")
print(df_clean.head().to_string(index=False))
''')


# ---------- CELL 6: Markdown Phase 2 ----------
CELL_6 = md("""## ⚙️ Fase 2: Feature Engineering

Mengubah atribut destinasi menjadi vektor numerik untuk:
1. **CBF Model** — menghitung cosine similarity antar destinasi
2. **RL Agent** — representasi state/action sebagai vektor

| Fitur | Tipe | Keterangan |
|---|---|---|
| `category_*` | One-hot (5 kolom) | Alam, Kuliner, Budaya, Wisata, Belanja |
| `rating_norm` | float [0,1] | Rating dinormalisasi |
| `ticket_norm` | float [0,1] | Harga tiket dinormalisasi (log-scale) |
| `duration_norm` | float [0,1] | Durasi kunjungan dinormalisasi |
| `lat_norm`, `lng_norm` | float [0,1] | Koordinat dinormalisasi |
| `tags_tfidf_*` | float [0,1] (n kolom) | TF-IDF dari kolom tags |

Total dimensi: 5 (category) + 5 (numerik) + n_tfidf (tags) ≈ 20–30 dimensi.
""")


# ---------- CELL 7: Feature engineering ----------
CELL_7 = code('''# ============================================================
# CELL 7 — Feature Engineering
# ============================================================
df_feat = df_clean.copy()

# (a) One-hot encoding kategori (urutan tetap)
CATEGORY_ORDER = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]
le_category = LabelEncoder()
le_category.fit(CATEGORY_ORDER)

category_onehot = np.zeros((len(df_feat), len(CATEGORY_ORDER)), dtype=float)
for i, cat in enumerate(df_feat["category"].tolist()):
    category_onehot[i, CATEGORY_ORDER.index(cat)] = 1.0

# (b) Normalisasi numerik (ticket → log-scale dulu)
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

# (d) Gabungkan
feature_matrix = np.hstack([category_onehot, numeric_normalized, tfidf_matrix])
print(f"✅ Feature matrix shape: {feature_matrix.shape}")
print(f"   - {category_onehot.shape[1]} kolom kategori")
print(f"   - {numeric_normalized.shape[1]} kolom numerik")
print(f"   - {tfidf_matrix.shape[1]} kolom TF-IDF")

# Simpan feature matrix
np.save("data/processed/feature_matrix.npy", feature_matrix)

# Simpan scaler & encoders
encoders = {
    "label_encoder_category": le_category,
    "scaler": scaler,
    "tfidf": tfidf,
    "feature_cols": numeric_cols,
    "category_order": CATEGORY_ORDER,
    "n_category_dims": category_onehot.shape[1],
    "n_numeric_dims": numeric_normalized.shape[1],
    "n_tfidf_dims": tfidf_matrix.shape[1],
}
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\\n💾 data/processed/feature_matrix.npy")
print("💾 models/label_encoders.pkl")
print("💾 models/scaler.pkl")
''')


# Lanjut ke cells 8-19 di file berikutnya
NB01_CELLS_PART_A = [CELL_0, CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7]



# ---------- CELL 8: Markdown CBF ----------
CELL_8 = md("""## 🔍 Fase 3A: Content-Based Filtering (CBF)

CBF bekerja dengan menghitung **cosine similarity** antar vektor fitur destinasi.
Saat user memberikan preferensi, sistem:
1. Membangun **user profile vector** dari preferensi user
2. Menghitung cosine similarity antara user profile dan setiap destinasi
3. Mengembalikan destinasi dengan similarity tertinggi (setelah hard-constraint filter)

Formula:
```
similarity(user, dest) = (user_vec · dest_vec) / (||user_vec|| × ||dest_vec||)
```
""")


# ---------- CELL 9: CBF training ----------
CELL_9 = code('''# ============================================================
# CELL 9 — Training Content-Based Filtering
# ============================================================
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Jarak Haversine (km) — modul global agar dipakai class lain."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class ContentBasedFilter:
    """Cosine-similarity recommender atas vektor fitur destinasi."""

    def __init__(self, feature_matrix: np.ndarray, destinations_df: pd.DataFrame, encoders: dict):
        self.feature_matrix = feature_matrix
        self.df = destinations_df.reset_index(drop=True)
        self.encoders = encoders
        self.similarity_matrix = None

    def fit(self):
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"✅ CBF fitted. Similarity matrix shape: {self.similarity_matrix.shape}")
        return self

    # ---- helpers ----
    def _category_dims(self) -> int:
        return self.encoders.get("n_category_dims", 5)

    def _numeric_start(self) -> int:
        return self._category_dims()

    def _numeric_end(self) -> int:
        return self._numeric_start() + self.encoders.get("n_numeric_dims", 5)

    def build_user_profile(self, categories: list, budget: float = None) -> np.ndarray:
        """Rata-rata fitur destinasi pada kategori user; budget di-override ke ticket dim."""
        if not categories:
            mask = np.ones(len(self.df), dtype=bool)
        else:
            mask = self.df["category"].isin(categories).values
        if mask.sum() == 0:
            mask = np.ones(len(self.df), dtype=bool)
        profile = self.feature_matrix[mask].mean(axis=0)

        # Override ticket_norm bila budget given.
        if budget is not None and budget > 0:
            scaler = self.encoders["scaler"]
            num_cols = self.encoders["feature_cols"]
            ticket_idx_in_numeric = num_cols.index("ticket_log")
            # Bangun row dummy dengan median per fitur supaya scaler stabil
            dummy = np.zeros((1, len(num_cols)))
            for j, col in enumerate(num_cols):
                if col == "ticket_log":
                    dummy[0, j] = math.log1p(budget)
                else:
                    dummy[0, j] = self.df[col if col != "ticket_log" else "ticket"].median() \\
                        if col != "ticket_log" else math.log1p(self.df["ticket"].median())
            scaled = scaler.transform(dummy)[0]
            profile_idx = self._numeric_start() + ticket_idx_in_numeric
            profile[profile_idx] = scaled[ticket_idx_in_numeric]
        return profile.reshape(1, -1)

    def recommend(
        self,
        categories: list = None,
        budget: float = None,
        max_km: float = None,
        home_lat: float = None,
        home_lng: float = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        if self.similarity_matrix is None:
            self.fit()
        categories = categories or []
        user_profile = self.build_user_profile(categories, budget=budget)
        sims = cosine_similarity(user_profile, self.feature_matrix).flatten()

        out = self.df.copy()
        out["cbf_score"] = sims

        # Hard constraint: kategori
        if categories:
            mask = out["category"].isin(categories)
            if mask.sum() > 0:
                out = out[mask]
        # Budget: ticket <= budget
        if budget is not None:
            out_b = out[out["ticket"].astype(float) <= float(budget)]
            if len(out_b) > 0:
                out = out_b
        # Distance: haversine dari home <= max_km
        if max_km is not None and home_lat is not None and home_lng is not None:
            out = out.copy()
            out["dist_home_km"] = out.apply(
                lambda r: haversine_km(home_lat, home_lng, r["lat"], r["lng"]),
                axis=1,
            )
            within = out[out["dist_home_km"] <= float(max_km)]
            if len(within) > 0:
                out = within
        return out.sort_values("cbf_score", ascending=False).head(top_n).reset_index(drop=True)

    def get_similar_destinations(self, dest_id: str, top_n: int = 5) -> pd.DataFrame:
        if self.similarity_matrix is None:
            self.fit()
        idx_list = self.df.index[self.df["id"] == dest_id].tolist()
        if not idx_list:
            return self.df.head(0)
        idx = idx_list[0]
        sims = self.similarity_matrix[idx]
        order = np.argsort(-sims)
        order = [i for i in order if i != idx][:top_n]
        out = self.df.iloc[order].copy()
        out["sim_score"] = sims[order]
        return out.reset_index(drop=True)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "similarity_matrix": self.similarity_matrix,
                "feature_matrix": self.feature_matrix,
                "df_index": self.df[["id", "name", "category"]].to_dict("records"),
            }, f)
        print(f"✅ CBF model disimpan ke {path}")

    @classmethod
    def load(cls, path: str, feature_matrix, destinations_df, encoders):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(feature_matrix, destinations_df, encoders)
        obj.similarity_matrix = data["similarity_matrix"]
        return obj


cbf_model = ContentBasedFilter(feature_matrix, df_clean, encoders).fit()

# ---- 3 quick tests ----
print("\\n🧪 TEST 1 — Alam, budget 200k, top 5")
t1 = cbf_model.recommend(categories=["Alam"], budget=200000, top_n=5)
print(t1[["name", "category", "ticket", "rating", "cbf_score"]])

print("\\n🧪 TEST 2 — Kuliner+Budaya, no budget, top 10")
t2 = cbf_model.recommend(categories=["Kuliner", "Budaya"], top_n=10)
print(t2[["name", "category", "ticket", "rating", "cbf_score"]])

print("\\n🧪 TEST 3 — Wisata+Belanja, budget 500k, max 30km dari Alun-Alun, top 8")
t3 = cbf_model.recommend(
    categories=["Wisata", "Belanja"], budget=500000, max_km=30,
    home_lat=-6.9215, home_lng=107.6071, top_n=8,
)
cols = ["name", "category", "ticket", "rating", "cbf_score"]
if "dist_home_km" in t3.columns:
    cols.append("dist_home_km")
print(t3[cols])

# Simpan model
cbf_model.save("models/cbf_model.pkl")
''')


# ---------- CELL 10: CBF visualizations ----------
CELL_10 = code('''# ============================================================
# CELL 10 — Visualisasi CBF
# ============================================================
sns.set_theme(style="whitegrid")

# (a) Heatmap similarity matrix — subsample 20 destinasi
sample_n = min(20, len(df_clean))
sub_idx = np.linspace(0, len(df_clean) - 1, sample_n, dtype=int)
sub_sim = cbf_model.similarity_matrix[np.ix_(sub_idx, sub_idx)]
sub_names = df_clean.iloc[sub_idx]["name"].tolist()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(sub_sim, xticklabels=sub_names, yticklabels=sub_names,
            cmap="viridis", ax=ax, cbar_kws={"label": "Cosine Similarity"})
ax.set_title("Cosine Similarity Matrix — Top 20 Destinations", fontsize=13)
plt.xticks(rotation=70, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("data/processed/cbf_similarity_heatmap.png", dpi=120)
plt.show()

# (b) Bar chart: rata-rata cbf_score per kategori
print("\\nMenghitung average CBF score per kategori (sample 100 random user profiles)…")
score_per_cat = defaultdict(list)
for _ in range(100):
    cats = random.sample(["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"],
                        k=random.randint(1, 3))
    rec = cbf_model.recommend(categories=cats, top_n=10)
    if len(rec):
        for cat, sub in rec.groupby("category"):
            score_per_cat[cat].extend(sub["cbf_score"].tolist())

avg_scores = {k: float(np.mean(v)) for k, v in score_per_cat.items() if v}
fig, ax = plt.subplots(figsize=(8, 5))
cats_sorted = sorted(avg_scores, key=avg_scores.get)
ax.barh(cats_sorted, [avg_scores[c] for c in cats_sorted], color="steelblue")
ax.set_xlabel("Average CBF Score (cosine similarity)")
ax.set_title("Average CBF Score per Kategori (100 random user profiles)")
plt.tight_layout()
plt.savefig("data/processed/cbf_category_scores.png", dpi=120)
plt.show()

print("✅ data/processed/cbf_similarity_heatmap.png")
print("✅ data/processed/cbf_category_scores.png")
''')


# ---------- CELL 11: Markdown RL ----------
CELL_11 = md("""## 🤖 Fase 3B: Multi-Agent Reinforcement Learning

### Arsitektur

**Agent 1 — Filter Agent (rule-based, deterministik):** menghasilkan binary mask kandidat
yang lolos hard constraint (kategori, budget, jarak, waktu).

**Agent 2 — Selector Agent (Q-Learning tabular):** memilih destinasi berikutnya secara
sekuensial berdasarkan state diskret.

### Reward Function
```
R(s, a, s') = 0.5 * rating_score(a)        # rating / 5.0
            + 0.2 * variety_bonus(s, a)    # +0.2 jika kategori baru
            + 0.3 * budget_efficiency(s,a) # (budget - ticket) / budget
            - 1.0 * overtime_penalty
            - 0.5 * overbudget_penalty
```

### State Discretization (Q-table)
- `n_selected`: 0–8
- `budget_level`: 0 (habis) – 4 (>75% sisa)
- `time_level`: 0 – 4
- `dominant_cat`: indeks kategori dominan dari yang sudah dipilih

State space ≈ 9 × 5 × 5 × 5 = 1125 states.
""")


# ---------- CELL 12: RL training ----------
CELL_12 = code('''# ============================================================
# CELL 12 — Training Q-Learning RL Agent
# ============================================================
HOME_OPTIONS = [
    {"lat": -6.9215, "lng": 107.6071, "name": "Alun-Alun Bandung"},
    {"lat": -6.9145, "lng": 107.6020, "name": "Stasiun Bandung"},
    {"lat": -6.8126, "lng": 107.6178, "name": "Pasar Lembang"},
    {"lat": -6.8915, "lng": 107.6107, "name": "Dago"},
    {"lat": -6.9024, "lng": 107.6188, "name": "Gedung Sate"},
]
SPEED_KMH = 28


class BandungTravelEnv:
    """Simulated env: user memilih destinasi satu per satu (sequential)."""

    REWARD_WEIGHTS = {"rating": 0.5, "variety": 0.2, "budget_eff": 0.3}
    PENALTY_OVERTIME = 1.0
    PENALTY_OVERBUDGET = 0.5

    def __init__(self, destinations_df: pd.DataFrame, cbf_model: ContentBasedFilter):
        self.df = destinations_df.reset_index(drop=True)
        self.cbf = cbf_model
        self.n_destinations = len(self.df)
        self.params = None
        self.candidates = []  # indeks pada self.df
        self.selected = []    # indeks pada self.df
        self.spent = 0
        self.cur_lat = None
        self.cur_lng = None
        self.cur_time = 0

    # ---- helpers ----
    def _haversine(self, lat1, lng1, lat2, lng2):
        return haversine_km(lat1, lng1, lat2, lng2)

    def _idx_by_id(self, dest_id: str) -> int:
        rows = self.df.index[self.df["id"] == dest_id].tolist()
        return rows[0] if rows else -1

    # ---- API ----
    def reset(self, params: dict):
        self.params = {**params}
        self.params.setdefault("budget", None)
        self.params.setdefault("max_km", None)
        if self.params["budget"] is None:
            self.params["budget"] = 999_999_999
        if self.params["max_km"] is None:
            self.params["max_km"] = 999
        self.params.setdefault("count", 4)
        self.params.setdefault("startMin", 9 * 60)
        self.params.setdefault("endMin", 21 * 60)
        self.params.setdefault("home_lat", HOME_OPTIONS[0]["lat"])
        self.params.setdefault("home_lng", HOME_OPTIONS[0]["lng"])

        # CBF top-30 jadi kandidat awal
        rec = self.cbf.recommend(
            categories=self.params.get("categories", []),
            budget=None if self.params["budget"] >= 999_999_998 else self.params["budget"],
            max_km=None if self.params["max_km"] >= 999 else self.params["max_km"],
            home_lat=self.params["home_lat"],
            home_lng=self.params["home_lng"],
            top_n=30,
        )
        if len(rec) == 0:
            rec = self.cbf.recommend(top_n=30)
        self.candidates = [self._idx_by_id(i) for i in rec["id"].tolist()]
        self.candidates = [i for i in self.candidates if i >= 0]

        self.selected = []
        self.spent = 0
        self.cur_lat = self.params["home_lat"]
        self.cur_lng = self.params["home_lng"]
        self.cur_time = self.params["startMin"]

        return self._get_state(), list(self.candidates)

    def _get_state(self):
        n_sel = min(8, len(self.selected))
        budget_total = self.params["budget"]
        if budget_total <= 0:
            budget_level = 0
        else:
            ratio = max(0.0, 1.0 - self.spent / budget_total)
            budget_level = min(4, int(ratio * 4 + 1e-9))
            if ratio >= 0.75:
                budget_level = 4
        time_total = max(1, self.params["endMin"] - self.params["startMin"])
        time_left = max(0, self.params["endMin"] - self.cur_time)
        time_ratio = time_left / time_total
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
        if self.selected:
            cats = Counter(self.df.iloc[i]["category"] for i in self.selected)
            top_cat = cats.most_common(1)[0][0]
            order = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]
            dom = order.index(top_cat) if top_cat in order else 0
        return (n_sel, budget_level, time_level, dom)

    def get_valid_actions(self):
        valid = []
        for k, idx in enumerate(self.candidates):
            if idx in self.selected:
                continue
            row = self.df.iloc[idx]
            if int(row["ticket"]) > (self.params["budget"] - self.spent):
                continue
            dist_home = self._haversine(
                self.params["home_lat"], self.params["home_lng"], row["lat"], row["lng"]
            )
            if dist_home > self.params["max_km"]:
                continue
            travel_km = self._haversine(self.cur_lat, self.cur_lng, row["lat"], row["lng"])
            travel_min = (travel_km / SPEED_KMH) * 60
            arrive = self.cur_time + travel_min
            depart = arrive + int(row["duration"])
            return_min = (self._haversine(
                row["lat"], row["lng"],
                self.params["home_lat"], self.params["home_lng"]
            ) / SPEED_KMH) * 60
            if depart + return_min > self.params["endMin"]:
                continue
            valid.append(k)
        return valid

    def _calculate_reward(self, dest_row, travel_km: float) -> float:
        rating_score = float(dest_row["rating"]) / 5.0
        cats_chosen = {self.df.iloc[i]["category"] for i in self.selected}
        variety = 0.2 if dest_row["category"] not in cats_chosen else 0.0
        budget_total = self.params["budget"]
        if budget_total <= 0 or budget_total >= 999_999_998:
            budget_eff = 0.5
        else:
            remain = max(0, budget_total - self.spent - int(dest_row["ticket"]))
            budget_eff = remain / budget_total

        w = self.REWARD_WEIGHTS
        r = w["rating"] * rating_score + w["variety"] * variety + w["budget_eff"] * budget_eff

        # Penalti
        if self.cur_time > self.params["endMin"]:
            r -= self.PENALTY_OVERTIME
        if self.spent > self.params["budget"]:
            r -= self.PENALTY_OVERBUDGET
        return float(r)

    def step(self, action_idx: int):
        valid = self.get_valid_actions()
        if action_idx not in valid:
            # Tidak valid → episode done dengan penalti kecil
            return self._get_state(), -0.1, True, {"reason": "invalid_action"}

        idx = self.candidates[action_idx]
        row = self.df.iloc[idx]
        travel_km = self._haversine(self.cur_lat, self.cur_lng, row["lat"], row["lng"])
        travel_min = (travel_km / SPEED_KMH) * 60
        self.cur_time += travel_min
        self.cur_time += int(row["duration"])
        self.spent += int(row["ticket"])
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
        cats_all = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]
        n_cats = random.randint(1, len(cats_all))
        cats = random.sample(cats_all, k=n_cats)
        budget = None if random.random() < 0.20 else random.randint(50_000, 2_000_000)
        max_km = None if random.random() < 0.40 else random.randint(20, 80)
        count = random.randint(2, 6)
        start = random.randint(420, 600)
        end = start + random.randint(300, 900)
        home = random.choice(HOME_OPTIONS)
        return {
            "categories": cats,
            "budget": budget,
            "max_km": max_km,
            "count": count,
            "startMin": start,
            "endMin": end,
            "home_lat": home["lat"],
            "home_lng": home["lng"],
        }


class QLearningAgent:
    """Tabular Q-Learning dengan epsilon-greedy."""

    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.training_history = []

    @staticmethod
    def _action_key(action_idx: int, candidate_ids: list) -> str:
        return candidate_ids[action_idx] if 0 <= action_idx < len(candidate_ids) else "unknown"

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

    def save(self, path: str):
        plain = {s: dict(actions) for s, actions in self.q_table.items()}
        data = {
            "q_table": plain,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "training_history": self.training_history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ RL Agent tersimpan ke {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        ag = cls(learning_rate=data["lr"], discount_factor=data["gamma"])
        ag.q_table = defaultdict(lambda: defaultdict(float))
        for s, actions in data["q_table"].items():
            for a, v in actions.items():
                ag.q_table[s][a] = v
        ag.epsilon = data["epsilon"]
        ag.training_history = data.get("training_history", [])
        return ag


def train_rl_agent(env: BandungTravelEnv, agent: QLearningAgent,
                   n_episodes: int = 3000, log_interval: int = 500) -> list:
    rewards_log = []
    for ep in tqdm(range(n_episodes), desc="Training RL Agent"):
        params = env.generate_random_params()
        state, candidates = env.reset(params)
        candidate_ids = [env.df.iloc[i]["id"] for i in candidates]
        total = 0.0
        done = False
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


env = BandungTravelEnv(df_clean, cbf_model)
rl_agent = QLearningAgent()
N_EPISODES = 3000
rewards_log = train_rl_agent(env, rl_agent, n_episodes=N_EPISODES, log_interval=500)
rl_agent.save("models/rl_agent.pkl")
''')


# ---------- CELL 13: RL viz ----------
CELL_13 = code('''# ============================================================
# CELL 13 — Visualisasi Training RL
# ============================================================
rewards_arr = np.array(rewards_log)
window = 100
rolling = pd.Series(rewards_arr).rolling(window=window, min_periods=1).mean().values

# (a) Learning curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rewards_arr, alpha=0.25, label="raw reward")
ax.plot(rolling, color="C1", linewidth=2, label=f"rolling mean (w={window})")
ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
ax.set_title(f"RL Agent Learning Curve — {len(rewards_arr)} Episodes")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/rl_learning_curve.png", dpi=120)
plt.show()

# (b) Epsilon decay (rekonstruksi)
ep_curve = []
eps = 1.0
for _ in range(len(rewards_arr)):
    ep_curve.append(eps)
    eps = max(rl_agent.epsilon_min, eps * rl_agent.epsilon_decay)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ep_curve, color="purple")
ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
ax.set_title("Epsilon Decay Schedule")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/rl_epsilon_decay.png", dpi=120)
plt.show()

# (c) Reward distribution awal vs akhir
first_chunk = rewards_arr[:500] if len(rewards_arr) >= 500 else rewards_arr[:len(rewards_arr)//2]
last_chunk = rewards_arr[-500:] if len(rewards_arr) >= 500 else rewards_arr[len(rewards_arr)//2:]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(first_chunk, bins=30, alpha=0.6, label="first 500 episodes", color="tomato")
ax.hist(last_chunk, bins=30, alpha=0.6, label="last 500 episodes", color="seagreen")
ax.set_xlabel("Total Reward"); ax.set_ylabel("Frequency")
ax.set_title("Reward Distribution: First vs Last 500 Episodes")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/rl_reward_distribution.png", dpi=120)
plt.show()

print("✅ Visualisasi tersimpan: rl_learning_curve.png, rl_epsilon_decay.png, rl_reward_distribution.png")
''')


# ---------- CELL 14: Markdown Route Optimizer ----------
CELL_14 = md("""## 🗺️ Fase 3C: Route Optimizer — TSP Nearest-Neighbor

Setelah RL Agent memilih N destinasi, urutan kunjungan dioptimalkan:
1. **Haversine Distance** untuk fallback offline
2. **OSRM API** (`router.project-osrm.org`) untuk waktu tempuh nyata bila tersedia
3. **Nearest-Neighbor Heuristic** untuk route ordering

OSRM endpoint:
```
GET http://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}?overview=false
```
Response → `routes[0].distance` (meter), `routes[0].duration` (detik).
""")


# ---------- CELL 15: Route Optimizer ----------
CELL_15 = code('''# ============================================================
# CELL 15 — Route Optimizer (TSP Nearest-Neighbor + OSRM)
# ============================================================
class RouteOptimizer:
    SPEED_KMH = 28
    OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

    def __init__(self, use_osrm: bool = True, osrm_timeout: float = 5.0):
        self.use_osrm = use_osrm
        self.osrm_timeout = osrm_timeout

    def haversine_km(self, lat1, lng1, lat2, lng2) -> float:
        return haversine_km(lat1, lng1, lat2, lng2)

    def osrm_travel_time(self, lat1, lng1, lat2, lng2):
        """Return (distance_km, duration_min). Fallback ke Haversine bila OSRM gagal."""
        if not self.use_osrm:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            return d, (d / self.SPEED_KMH) * 60
        try:
            url = f"{self.OSRM_BASE}/{lng1},{lat1};{lng2},{lat2}?overview=false"
            resp = requests.get(url, timeout=self.osrm_timeout)
            resp.raise_for_status()
            route = resp.json()["routes"][0]
            return route["distance"] / 1000.0, route["duration"] / 60.0
        except Exception as e:
            d = self.haversine_km(lat1, lng1, lat2, lng2)
            return d, (d / self.SPEED_KMH) * 60

    def nearest_neighbor_route(self, home: dict, destinations: list) -> list:
        if not destinations:
            return []
        remaining = list(destinations)
        ordered = []
        cur_lat, cur_lng = home["lat"], home["lng"]
        while remaining:
            nxt = min(remaining,
                      key=lambda d: self.haversine_km(cur_lat, cur_lng, d["lat"], d["lng"]))
            ordered.append(nxt)
            remaining = [d for d in remaining if d.get("id") != nxt.get("id")]
            cur_lat, cur_lng = nxt["lat"], nxt["lng"]
        return ordered

    def build_itinerary(self, home: dict, home_name: str, ordered_destinations: list,
                        start_min: int, end_min: int) -> dict:
        steps = []
        cur_lat, cur_lng = home["lat"], home["lng"]
        cur_time = start_min
        total_cost = 0
        total_km = 0.0
        total_visit_time = 0

        for i, dest in enumerate(ordered_destinations):
            travel_km, travel_min = self.osrm_travel_time(
                cur_lat, cur_lng, float(dest["lat"]), float(dest["lng"])
            )
            arrive = cur_time + int(round(travel_min))
            depart = arrive + int(dest.get("duration", 60))

            tags = dest.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags.replace("'", '"'))
                except Exception:
                    tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]

            steps.append({
                "idx": i + 1,
                "dest": {
                    "id": dest.get("id"),
                    "name": dest.get("name"),
                    "category": dest.get("category"),
                    "desc": dest.get("desc", ""),
                    "ticket": int(dest.get("ticket", 0)),
                    "duration": int(dest.get("duration", 60)),
                    "lat": float(dest["lat"]),
                    "lng": float(dest["lng"]),
                    "rating": float(dest.get("rating", 4.0)),
                    "tags": list(tags) if tags else [],
                    "gmaps_url": dest.get("gmaps_url") or generate_gmaps_url(dest.get("name", "")),
                },
                "travelMin": int(round(travel_min)),
                "travelKm": round(float(travel_km), 2),
                "arriveAt": int(arrive),
                "departAt": int(depart),
            })

            total_cost += int(dest.get("ticket", 0))
            total_km += float(travel_km)
            total_visit_time += int(dest.get("duration", 60))
            cur_lat, cur_lng = float(dest["lat"]), float(dest["lng"])
            cur_time = depart

        # return trip
        return_km, return_min = self.osrm_travel_time(cur_lat, cur_lng, home["lat"], home["lng"])
        arrive_home = cur_time + int(round(return_min))
        total_km += float(return_km)
        total_time = total_visit_time + sum(s["travelMin"] for s in steps) + int(round(return_min))

        spare = end_min - arrive_home
        return {
            "steps": steps,
            "totalCost": int(total_cost),
            "totalKm": round(float(total_km), 2),
            "totalTime": int(total_time),
            "returnKm": round(float(return_km), 2),
            "returnMin": int(round(return_min)),
            "arriveHome": int(arrive_home),
            "overBudget": bool(arrive_home > end_min),  # alias "overTime" sesuai kontrak
            "spareMin": int(spare),
        }


optimizer = RouteOptimizer(use_osrm=True)

# Test cepat dengan 4 destinasi
sample = df_clean[df_clean["id"].isin([
    "kawah-putih", "saung-angklung-udjo", "gedung-sate", "cihampelas-walk"
])].to_dict("records")
home = {"lat": -6.9215, "lng": 107.6071}
ordered = optimizer.nearest_neighbor_route(home, sample)
itin = optimizer.build_itinerary(home, "Alun-Alun Bandung", ordered, 9 * 60, 21 * 60)

print("📍 Sample itinerary (4 destinasi):")
for s in itin["steps"]:
    a = f"{s['arriveAt']//60:02d}:{s['arriveAt']%60:02d}"
    d = f"{s['departAt']//60:02d}:{s['departAt']%60:02d}"
    print(f"  {s['idx']}. {s['dest']['name']:<28} | {a}-{d} | {s['travelKm']:.1f} km")
print(f"\\n💰 Total: Rp {itin['totalCost']:,} | 📏 {itin['totalKm']:.1f} km | 🏠 tiba {itin['arriveHome']//60:02d}:{itin['arriveHome']%60:02d}")
''')

NB01_CELLS_PART_B = [CELL_8, CELL_9, CELL_10, CELL_11, CELL_12, CELL_13, CELL_14, CELL_15]



# ---------- CELL 16: Integration test ----------
CELL_16 = code('''# ============================================================
# CELL 16 — Integration Test (Full Pipeline)
# ============================================================
def rl_select_destinations(env: BandungTravelEnv, agent: QLearningAgent,
                            params: dict) -> list:
    """Inference RL: gunakan greedy policy (epsilon=0) untuk memilih destinasi."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        state, candidates = env.reset(params)
        candidate_ids = [env.df.iloc[i]["id"] for i in candidates]
        selected_rows = []
        done = False
        while not done and len(selected_rows) < params.get("count", 4):
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.choose_action(state, valid, candidate_ids)
            if action < 0 or action >= len(env.candidates):
                break
            idx = env.candidates[action]
            selected_rows.append(env.df.iloc[idx].to_dict())
            state, _, done, _ = env.step(action)
        return selected_rows
    finally:
        agent.epsilon = saved_eps


def full_pipeline_test(params: dict) -> dict:
    # 1+2: RL-guided selection (CBF candidates dipakai di env.reset)
    selected = rl_select_destinations(env, rl_agent, {
        "categories": params.get("categories", []),
        "budget": params.get("budget"),
        "max_km": params.get("maxKm"),
        "count": params.get("count", 4),
        "startMin": params["startMin"],
        "endMin": params["endMin"],
        "home_lat": params["home"]["lat"],
        "home_lng": params["home"]["lng"],
    })

    # Fallback bila RL gagal mengisi: lengkapi dari CBF
    if len(selected) < params.get("count", 4):
        rec = cbf_model.recommend(
            categories=params.get("categories", []),
            budget=params.get("budget"),
            max_km=params.get("maxKm"),
            home_lat=params["home"]["lat"],
            home_lng=params["home"]["lng"],
            top_n=20,
        )
        seen = {d["id"] for d in selected}
        for _, r in rec.iterrows():
            if len(selected) >= params.get("count", 4):
                break
            if r["id"] not in seen:
                selected.append(r.to_dict())
                seen.add(r["id"])

    # 3: Route ordering
    ordered = optimizer.nearest_neighbor_route(params["home"], selected)
    # 4: Itinerary
    itin = optimizer.build_itinerary(
        home=params["home"],
        home_name=params["homeName"],
        ordered_destinations=ordered,
        start_min=params["startMin"],
        end_min=params["endMin"],
    )
    return itin


test_cases = [
    {"home": {"lat": -6.9215, "lng": 107.6071}, "homeName": "Alun-Alun Bandung",
     "count": 4, "maxKm": None, "startMin": 9 * 60, "endMin": 21 * 60,
     "budget": 500_000, "categories": ["Alam", "Kuliner"]},
    {"home": {"lat": -6.8126, "lng": 107.6178}, "homeName": "Pasar Lembang",
     "count": 3, "maxKm": 25, "startMin": 8 * 60, "endMin": 18 * 60,
     "budget": 200_000, "categories": ["Alam"]},
    {"home": {"lat": -6.9145, "lng": 107.6020}, "homeName": "Stasiun Bandung",
     "count": 5, "maxKm": None, "startMin": 10 * 60, "endMin": 20 * 60,
     "budget": None, "categories": ["Kuliner", "Budaya", "Belanja"]},
]

for i, p in enumerate(test_cases, 1):
    print(f"\\n{'='*60}\\nTEST CASE {i}: {p['homeName']}\\n{'='*60}")
    res = full_pipeline_test(p)
    print(f"Destinasi terpilih: {len(res['steps'])}")
    for s in res["steps"]:
        a = f"{s['arriveAt']//60:02d}:{s['arriveAt']%60:02d}"
        print(f"  {s['idx']}. {s['dest']['name']:<28} ({s['dest']['category']}) — tiba {a}")
    print(f"💰 Total biaya: Rp {res['totalCost']:,}")
    print(f"📏 Total jarak: {res['totalKm']:.1f} km")
    print(f"🏠 Tiba kembali: {res['arriveHome']//60:02d}:{res['arriveHome']%60:02d}")
    print(f"⏰ Over time: {res['overBudget']}")
''')


# ---------- CELL 17: Evaluation ----------
CELL_17 = code('''# ============================================================
# CELL 17 — Evaluasi Model
# ============================================================
print("=== CBF Evaluation ===")
# (1) Intra-list diversity
diversities = []
for _ in range(50):
    cats = random.sample(["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"],
                        k=random.randint(1, 3))
    rec = cbf_model.recommend(categories=cats, top_n=10)
    if len(rec) < 2:
        continue
    idxs = [df_clean.index[df_clean["id"] == i].tolist()[0] for i in rec["id"]
            if (df_clean["id"] == i).any()]
    if len(idxs) < 2:
        continue
    sims = cbf_model.similarity_matrix[np.ix_(idxs, idxs)]
    iu = np.triu_indices_from(sims, k=1)
    diversities.append(1.0 - float(sims[iu].mean()))
diversity = float(np.mean(diversities)) if diversities else 0.0
print(f"  Intra-list Diversity: {diversity:.4f}")

# (2) Catalog coverage (100 user profiles)
recommended_ids = set()
for _ in range(100):
    cats = random.sample(["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"],
                        k=random.randint(1, 3))
    rec = cbf_model.recommend(categories=cats, top_n=10)
    recommended_ids.update(rec["id"].tolist())
coverage = len(recommended_ids) / max(1, len(df_clean))
print(f"  Catalog Coverage: {coverage:.4f} ({len(recommended_ids)}/{len(df_clean)})")

# (3) Category balance
print("  Distribusi kategori (dataset):")
print(df_clean["category"].value_counts(normalize=True).round(3).to_dict())

print("\\n=== RL Agent Evaluation ===")
chunk = min(100, len(rewards_log) // 2)
first_avg = float(np.mean(rewards_log[:chunk]))
last_avg = float(np.mean(rewards_log[-chunk:]))
denom = abs(first_avg) if abs(first_avg) > 1e-6 else 1.0
improvement = ((last_avg - first_avg) / denom) * 100
print(f"  Avg reward (first {chunk}): {first_avg:.4f}")
print(f"  Avg reward (last {chunk}):  {last_avg:.4f}")
print(f"  Improvement: {improvement:.1f}%")

# Success rate
success = 0
total_runs = 50
selected_counts = []
for _ in range(total_runs):
    p = env.generate_random_params()
    state, _ = env.reset(p)
    done = False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = rl_agent.choose_action(state, valid,
                                        [env.df.iloc[i]["id"] for i in env.candidates])
        if action < 0:
            break
        state, _, done, _ = env.step(action)
    selected_counts.append(len(env.selected))
    no_overtime = env.cur_time <= p["endMin"]
    no_overbudget = env.spent <= (p["budget"] if p["budget"] else 999_999_999)
    if len(env.selected) >= max(1, p["count"] - 1) and no_overtime and no_overbudget:
        success += 1
print(f"  Success Rate: {success/total_runs:.2%}")
print(f"  Avg destinations selected: {np.mean(selected_counts):.2f}")

print("\\n=== Route Optimizer Evaluation ===")
nn_kms, rand_kms = [], []
for _ in range(30):
    sample_df = df_clean.sample(n=4)
    home = {"lat": -6.9215, "lng": 107.6071}
    sample_list = sample_df.to_dict("records")
    ordered = optimizer.nearest_neighbor_route(home, sample_list)
    # NN total
    cur = home
    nn_total = 0.0
    for d in ordered:
        nn_total += haversine_km(cur["lat"], cur["lng"], d["lat"], d["lng"])
        cur = {"lat": d["lat"], "lng": d["lng"]}
    nn_total += haversine_km(cur["lat"], cur["lng"], home["lat"], home["lng"])
    nn_kms.append(nn_total)
    # random total
    shuffled = list(sample_list)
    random.shuffle(shuffled)
    cur = home
    rd_total = 0.0
    for d in shuffled:
        rd_total += haversine_km(cur["lat"], cur["lng"], d["lat"], d["lng"])
        cur = {"lat": d["lat"], "lng": d["lng"]}
    rd_total += haversine_km(cur["lat"], cur["lng"], home["lat"], home["lng"])
    rand_kms.append(rd_total)
nn_avg, rd_avg = float(np.mean(nn_kms)), float(np.mean(rand_kms))
imp = (rd_avg - nn_avg) / max(rd_avg, 1e-6) * 100
print(f"  Avg distance (NN heuristic): {nn_avg:.2f} km")
print(f"  Avg distance (random order): {rd_avg:.2f} km")
print(f"  Route Optimization Improvement: {imp:.1f}%")
''')


# ---------- CELL 18: Export & summary ----------
CELL_18 = code('''# ============================================================
# CELL 18 — Export & Summary
# ============================================================
print("\\n" + "=" * 60)
print("📦 EXPORT SUMMARY — Bandung AI Travel Agent")
print("=" * 60)

files_to_check = [
    "data/processed/destinations.csv",
    "data/processed/feature_matrix.npy",
    "data/last_updated.txt",
    "models/cbf_model.pkl",
    "models/rl_agent.pkl",
    "models/scaler.pkl",
    "models/label_encoders.pkl",
]
for f in files_to_check:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✅ {f} ({size:.1f} KB)")
    else:
        print(f"  ❌ {f} — MISSING!")

df_summary = pd.read_csv("data/processed/destinations.csv")
print(f"\\n📊 Dataset Summary:")
print(f"  Total destinasi: {len(df_summary)}")
print(df_summary.groupby("category").agg({
    "name": "count", "rating": "mean", "ticket": "median"
}).round(2))

print(f"\\n⏰ Last Updated: {date.today()}")
print("\\n✅ Semua model dan data siap digunakan oleh Notebook 2 dan Backend!")
''')


# ---------- CELL 19: Closing ----------
CELL_19 = md("""## ✅ Notebook 1 Selesai

### File yang dihasilkan
| File | Keterangan |
|---|---|
| `data/processed/destinations.csv` | Dataset bersih (≥50 destinasi) |
| `data/processed/feature_matrix.npy` | Matriks fitur untuk CBF & RL |
| `data/last_updated.txt` | Timestamp update data |
| `models/cbf_model.pkl` | Model Content-Based Filtering |
| `models/rl_agent.pkl` | Q-Learning RL Agent |
| `models/scaler.pkl` | MinMaxScaler untuk normalisasi |
| `models/label_encoders.pkl` | LabelEncoder + TF-IDF vectorizer |

### Langkah Selanjutnya
1. Buka **Notebook 2** (`02_llm_storyteller.ipynb`) untuk integrasi LLM.
2. Deploy ke backend FastAPI menggunakan file model di atas.
3. Re-run notebook ini secara berkala (≈1 bulan sekali) untuk refresh data.

### Catatan
- RL agent dilatih dengan simulated environment (bukan log user nyata).
- OSRM public API memiliki rate limit — jangan dipakai untuk >100 req/menit.
- Saat live, kumpulkan log interaksi user untuk fine-tune agent berkala.
""")


NB01_CELLS_PART_C = [CELL_16, CELL_17, CELL_18, CELL_19]


def all_cells():
    return NB01_CELLS_PART_A + NB01_CELLS_PART_B + NB01_CELLS_PART_C
