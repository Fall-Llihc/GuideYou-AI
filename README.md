# TASK PROMPT — Bandung AI Travel Agent: Jupyter Notebooks
## Capstone Project · Telkom University · Data Science

> **Untuk AI Coding Model:** Dokumen ini adalah spesifikasi **lengkap dan final** untuk membuat dua file Jupyter Notebook (`.ipynb`) yang menjadi backbone dari sistem Bandung AI Travel Agent. Baca seluruh dokumen sebelum menulis satu baris kode pun. Setiap detail adalah **wajib diimplementasi**, tidak ada yang boleh dilewat atau disingkat.

---

## KONTEKS PROYEK

Bandung AI Travel Agent adalah sistem rekomendasi itinerary wisata berbasis AI yang sudah memiliki **frontend React** lengkap. Frontend tersebut memiliki alur: **Welcome → Form → Loading → Results**.

### Input yang diterima Frontend (dari User):
| Parameter | Tipe | Keterangan |
|---|---|---|
| `home` | `{lat: float, lng: float}` | Koordinat titik keberangkatan |
| `homeName` | `string` | Nama lokasi awal (misal: "Alun-Alun Bandung") |
| `count` | `int` (1–8) | Jumlah destinasi yang diinginkan |
| `maxKm` | `float \| null` | Batas jarak maks antar destinasi (km), nullable |
| `startMin` | `int` | Jam mulai dalam menit sejak tengah malam (misal: 540 = 09:00) |
| `endMin` | `int` | Jam selesai dalam menit sejak tengah malam (misal: 1260 = 21:00) |
| `budget` | `int \| null` | Budget total dalam Rupiah, nullable |
| `categories` | `list[string]` | Subset dari: ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"] |

### Output yang diharapkan Backend (dikonsumsi Frontend):
```json
{
  "steps": [
    {
      "idx": 1,
      "dest": {
        "id": "kawah-putih",
        "name": "Kawah Putih",
        "category": "Alam",
        "desc": "Danau kawah vulkanik...",
        "ticket": 81000,
        "duration": 120,
        "lat": -7.166,
        "lng": 107.4019,
        "rating": 4.6,
        "tags": ["sunrise", "fotogenik", "dingin"],
        "gmaps_url": "https://www.google.com/maps/search/?api=1&query=Kawah+Putih%2C+Bandung"
      },
      "travelMin": 45,
      "travelKm": 21.3,
      "arriveAt": 585,
      "departAt": 705
    }
  ],
  "totalCost": 196000,
  "totalKm": 87.4,
  "totalTime": 480,
  "returnKm": 32.1,
  "returnMin": 68,
  "arriveHome": 1220,
  "overBudget": false,
  "spareMin": 40,
  "story": {
    "intro": "Hari ini perjalananmu dimulai dari...",
    "highlights": ["**Kawah Putih** *(Alam)* — tiba sekitar pukul 09:45..."],
    "tips": ["Datang lebih pagi biar view masih clear..."],
    "closing": "Udah, gausah mikir panjang — **save itinerary ini**...",
    "vibe": "Alam"
  },
  "data_last_updated": "2025-05-20"
}
```

### Dataset Destinasi (struktur per record):
```python
{
  "id": "kawah-putih",           # slug unik
  "name": "Kawah Putih",         # nama tampil
  "category": "Alam",            # Alam | Kuliner | Budaya | Wisata | Belanja
  "desc": "...",                  # deskripsi singkat
  "ticket": 81000,               # harga tiket masuk (int, 0 = gratis)
  "duration": 120,               # estimasi waktu kunjungan (menit)
  "lat": -7.1660,                # latitude
  "lng": 107.4019,               # longitude
  "rating": 4.6,                 # rating 1.0–5.0
  "tags": ["sunrise", "dingin"], # tag-tag karakteristik
  "gmaps_url": "https://..."     # link Google Maps
}
```

### Destinasi yang sudah ada di Frontend (data.js — WAJIB ada di dataset):
```
Alam:     Kawah Putih, Kebun Teh Rancabali, Stone Garden Padalarang,
          Tebing Keraton, Tangkuban Perahu
Kuliner:  Floating Market Lembang, Jalan Braga, Punclut Bukit Bintang
Budaya:   Saung Angklung Udjo, Gedung Sate, Museum Geologi
Wisata:   Farmhouse Lembang, Dusun Bambu, Trans Studio Bandung
Belanja:  Cihampelas Walk, Pasar Baru Trade Center
```
*Notebook wajib menghasilkan minimal 50 destinasi total (termasuk 16 di atas).*

---

## KEPUTUSAN ARSITEKTUR: 2 NOTEBOOK TERPISAH

### Alasan Pemisahan (PENTING, pahami ini):
1. **IPYNB 1 (Recommendation Engine)** adalah **ML training pipeline** yang berat — dijalankan sekali atau berkala (saat data diperbarui). Output-nya adalah file model `.pkl`.
2. **IPYNB 2 (LLM Storyteller)** adalah **integration test + prompt engineering** yang ringan — mengambil output itinerary dari model, lalu memanggil Groq API. Bisa ditest ulang kapanpun tanpa harus re-train.
3. Menggabungkan keduanya menciptakan notebook monolitik yang sulit di-debug, lambat, dan tidak modular.
4. Pemisahan mengikuti prinsip **Single Responsibility Principle** — satu notebook, satu tanggung jawab.

### Alur Data Antar Notebook:
```
IPYNB 1 Output:
  models/cbf_model.pkl
  models/rl_agent.pkl
  models/scaler.pkl
  models/label_encoders.pkl
  data/processed/destinations.csv
  data/processed/feature_matrix.npy
        ↓
IPYNB 2 Input:
  Membaca destinations.csv + model .pkl
  Mensimulasikan request dari frontend (params dict)
  Menghasilkan itinerary JSON
  Mengirim ke Groq API → story JSON
  Validasi output sesuai kontrak frontend
```

---

---

# ═══════════════════════════════════════════════
# NOTEBOOK 1: RECOMMENDATION ENGINE
# File: notebooks/01_recommendation_engine.ipynb
# ═══════════════════════════════════════════════

## TUJUAN NOTEBOOK 1
Membangun dan melatih sistem rekomendasi destinasi wisata Bandung secara end-to-end:
1. Crawling & pengumpulan data destinasi dari OpenStreetMap (Overpass API)
2. Data cleaning & enrichment
3. Feature engineering
4. Training Content-Based Filtering (CBF)
5. Training Multi-Agent Reinforcement Learning (RL)
6. Route optimization dengan TSP nearest-neighbor
7. Evaluasi model
8. Export model dan dataset ke file

---

## STRUKTUR CELL NOTEBOOK 1

### CELL 0: Judul & Deskripsi (Markdown)
```markdown
# 🗺️ Bandung AI Travel Agent — Recommendation Engine
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
```

---

### CELL 1: Setup & Instalasi Dependencies (Code)
```python
# Wajib install semua package yang dibutuhkan
# Gunakan !pip install ... untuk setiap package
```

**Package yang WAJIB diinstall dan diimport:**
- `requests` — HTTP requests ke Overpass API & OSRM
- `pandas` — manipulasi data
- `numpy` — operasi numerik
- `scikit-learn` — CBF (TfidfVectorizer, cosine_similarity, MinMaxScaler, LabelEncoder)
- `pickle` — save/load model
- `json` — parse API response
- `time` — rate limiting saat crawling
- `os`, `pathlib` — manajemen file path
- `urllib.parse` — generate gmaps_url
- `math` — kalkulasi haversine distance
- `random` — untuk RL simulation
- `collections` (defaultdict, deque) — untuk RL replay buffer
- `matplotlib.pyplot` — visualisasi evaluasi
- `seaborn` — heatmap similarity matrix
- `tqdm` — progress bar training RL
- `warnings` — suppress warning

**Setelah install, buat struktur folder:**
```python
import os
from pathlib import Path

# Buat semua direktori yang dibutuhkan
dirs = ["data/raw", "data/processed", "models", "notebooks"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    
print("✅ Struktur folder berhasil dibuat")
print("📁 data/raw/        → hasil crawling mentah")
print("📁 data/processed/  → dataset bersih .csv")
print("📁 models/          → model .pkl tersimpan")
```

---

### CELL 2: Markdown — Fase 1: Data Collection

```markdown
## 📡 Fase 1: Data Collection — Overpass API (OpenStreetMap)

Menggunakan Overpass API untuk mengambil POI (Point of Interest) wisata Bandung.
Bounding box Bandung Raya: -7.2500, 107.3500, -6.7500, 107.9000

Kategori OSM yang diambil:
| OSM Tag | Kategori Proyek |
|---|---|
| tourism=viewpoint, natural=*, park | Alam |
| amenity=restaurant, cafe, food_court | Kuliner |
| tourism=museum, historic=*, amenity=place_of_worship | Budaya |
| tourism=theme_park, tourism=attraction, leisure=* | Wisata |
| shop=mall, amenity=marketplace | Belanja |
```

---

### CELL 3: Crawling Overpass API (Code)
**Implementasikan fungsi berikut SECARA LENGKAP:**

```python
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BANDUNG_BBOX = "-7.2500,107.3500,-6.7500,107.9000"  # south,west,north,east

# Mapping OSM tag → kategori proyek
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
```

**Fungsi yang wajib diimplementasikan:**

**a) `query_overpass(query_str: str) -> dict`**
- Kirim POST request ke Overpass API
- Handle timeout (timeout=30s)
- Handle HTTP error dengan try-except
- Tambahkan `time.sleep(2)` setelah setiap request (rate limiting)
- Return parsed JSON atau empty dict jika error

**b) `build_overpass_query(category: str, bbox: str) -> str`**
- Build Overpass QL query untuk mengambil nodes dan ways
- Contoh untuk kategori Alam: query dengan `tourism=viewpoint`, `natural=peak`, `leisure=park` dalam bbox
- Query harus bisa mengambil nama dan koordinat

**c) `fetch_all_categories() -> pd.DataFrame`**
- Loop semua 5 kategori, call `query_overpass` untuk setiap kategori
- Parse response: ambil `id`, `name` (dari tags.name), `lat`, `lon`
- Skip node yang tidak punya nama (`name` null/empty)
- Assign kategori berdasarkan tag
- Gabungkan semua hasil menjadi satu DataFrame
- Tambahkan kolom `source = "overpass"`
- Print progress setiap kategori

**d) `generate_gmaps_url(name: str) -> str`**
```python
import urllib.parse
def generate_gmaps_url(name: str) -> str:
    query = urllib.parse.quote(name + ", Bandung")
    return f"https://www.google.com/maps/search/?api=1&query={query}"
```

**Setelah crawling:**
- Simpan raw result ke `data/raw/osm_raw.csv`
- Print jumlah record per kategori
- Tampilkan 5 baris pertama DataFrame

---

### CELL 4: Data Enrichment & Seed Data (Code)

**Langkah 1: Seed dataset dari Frontend**
Data 16 destinasi dari `data.js` frontend WAJIB ada. Buat sebagai Python dict list:
```python
SEED_DESTINATIONS = [
    {
        "id": "kawah-putih",
        "name": "Kawah Putih",
        "category": "Alam",
        "desc": "Danau kawah vulkanik dengan air berwarna putih kehijauan di ketinggian 2430 mdpl",
        "ticket": 81000,
        "duration": 120,
        "lat": -7.1660, "lng": 107.4019,
        "rating": 4.6,
        "tags": ["sunrise", "fotogenik", "dingin"]
    },
    # ... (masukkan SEMUA 16 destinasi dari frontend data.js LENGKAP)
    # Kawah Putih, Kebun Teh Rancabali, Stone Garden, Tebing Keraton, Tangkuban Perahu
    # Floating Market, Jalan Braga, Punclut Bukit Bintang
    # Saung Angklung Udjo, Gedung Sate, Museum Geologi
    # Farmhouse Lembang, Dusun Bambu, Trans Studio Bandung
    # Cihampelas Walk, Pasar Baru Trade Center
]
```

**Langkah 2: Tambahan destinasi (minimal 34 destinasi baru) untuk total ≥50**
Buat tambahan destinasi wisata Bandung yang nyata dan faktual. Setiap destinasi wajib memiliki:
- `id` (slug lowercase-dash)
- `name` (nama asli tempat)
- `category` (salah satu dari 5 kategori)
- `desc` (1 kalimat deskripsi akurat)
- `ticket` (harga tiket dalam Rupiah, 0 jika gratis — gunakan estimasi nyata)
- `duration` (estimasi menit kunjungan: 30–240)
- `lat`, `lng` (koordinat nyata — verifikasi akurasi!)
- `rating` (1.0–5.0, berbasis estimasi realitis)
- `tags` (list 2–4 string tag relevan)

Contoh tambahan yang disarankan (isi dengan data faktual):
- Alam: Situ Patenggang, Curug Dago, Curug Cimahi, Gunung Batu Lembang, Kebun Raya Cibodas, Situ Lembang, Curug Malela
- Kuliner: Warung Nasi Ampera, Mie Kocok Mang Dadeng, Kupat Tahu Gempol, Batagor Riri, Restoran Sindang Reret, Kafe Warung Koffie Batavia
- Budaya: Keraton Sunda Wiwitan, Museum Pos Indonesia, Museum Sri Baduga, Museum KAA, Taman Budaya Jawa Barat, Monumen Perjuangan Rakyat Jabar
- Wisata: The Lodge Maribaya, Kampung Daun, Observatorium Bosscha, Taman Hutan Raya Ir. H. Djuanda, Kebun Binatang Bandung, Taman Film, Sky Lantern Lembang
- Belanja: BTC Fashion Mall, Outlet Rumah Mode, ITC Kebon Kelapa, Pasar Baru Lembang, 23 Paskal Shopping Center, Cibaduyut (pusat sepatu)

**Langkah 3: Merge & Deduplication**
- Gabungkan seed + OSM crawling result + tambahan manual
- Remove duplicates berdasarkan nama (fuzzy: jika nama mirip >80%, pertahankan seed)
- Pastikan semua 16 seed destinations SELALU ada (jangan terhapus)
- Tambahkan `gmaps_url` untuk semua record
- Simpan ke `data/raw/destinations_enriched.csv`

---

### CELL 5: Data Cleaning & Validation (Code)

**Implementasikan fungsi `clean_destinations(df: pd.DataFrame) -> pd.DataFrame`:**

Langkah cleaning yang WAJIB:
1. **Rename kolom:** `lon` → `lng` (konsistensi dengan frontend)
2. **Fill missing values:**
   - `ticket`: fill dengan median per kategori (bukan 0)
   - `rating`: fill dengan mean per kategori
   - `duration`: fill dengan median per kategori
   - `desc`: fill dengan template `"Destinasi wisata {category} di Bandung"`
   - `tags`: fill dengan `[]`
3. **Type casting:**
   - `ticket` → `int`
   - `duration` → `int`
   - `rating` → `float`, clip ke range [1.0, 5.0]
   - `lat`, `lng` → `float`
4. **Validation:**
   - Filter baris di mana lat/lng berada di luar bbox Bandung Raya: lat ∈ [-7.4, -6.6], lng ∈ [107.2, 108.2]
   - Filter baris dengan `name` null atau panjang < 3 karakter
   - Assert tidak ada NaN di kolom: `id`, `name`, `category`, `lat`, `lng`
5. **Normalisasi kategori:** pastikan setiap kategori adalah salah satu dari `{"Alam", "Kuliner", "Budaya", "Wisata", "Belanja"}`
6. **Generate slug id** untuk record yang tidak punya id:
   ```python
   import re
   def slugify(name): 
       return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
   ```
7. **Sort** berdasarkan category, lalu rating descending

**Output validation:**
```python
assert len(df) >= 50, f"Dataset minimal 50 destinasi, dapat {len(df)}"
assert set(df['category'].unique()) == {"Alam", "Kuliner", "Budaya", "Wisata", "Belanja"}
assert df['lat'].between(-7.4, -6.6).all()
assert df['lng'].between(107.2, 108.2).all()
print(f"✅ Dataset valid: {len(df)} destinasi")
print(df.groupby('category').size())
```

**Simpan ke `data/processed/destinations.csv`**

Simpan juga timestamp ke `data/last_updated.txt`:
```python
from datetime import date
with open("data/last_updated.txt", "w") as f:
    f.write(str(date.today()))
```

---

### CELL 6: Markdown — Fase 2: Feature Engineering

```markdown
## ⚙️ Fase 2: Feature Engineering

Mengubah atribut destinasi menjadi vektor numerik untuk:
1. **CBF Model:** menghitung cosine similarity antar destinasi
2. **RL Agent:** representasi state/action sebagai vektor

Fitur yang digunakan:
| Fitur | Tipe | Keterangan |
|---|---|---|
| category_* | One-hot (5 kolom) | Alam, Kuliner, Budaya, Wisata, Belanja |
| rating_norm | float [0,1] | Rating dinormalisasi |
| ticket_norm | float [0,1] | Harga tiket dinormalisasi (log-scale) |
| duration_norm | float [0,1] | Durasi kunjungan dinormalisasi |
| lat_norm | float [0,1] | Latitude dinormalisasi |
| lng_norm | float [0,1] | Longitude dinormalisasi |
| tags_tfidf_* | float [0,1] (n kolom) | TF-IDF dari kolom tags |

Total dimensi: 5 (category) + 6 (numerik) + n_tfidf (tags) ≈ 20–35 dimensi
```

---

### CELL 7: Feature Engineering (Code)

**Implementasikan semua langkah berikut:**

**a) One-Hot Encoding Kategori:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Buat label encoder untuk kategori
le_category = LabelEncoder()
le_category.fit(["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"])

# One-hot encode
# Hasilkan 5 kolom: category_Alam, category_Kuliner, dst
```

**b) Normalisasi Fitur Numerik (MinMaxScaler):**
```python
from sklearn.preprocessing import MinMaxScaler

# PENTING: ticket menggunakan log-scale sebelum normalisasi
# karena distribusinya sangat skewed (0 hingga 280000)
df['ticket_log'] = np.log1p(df['ticket'])

# Fit scaler pada kolom: ticket_log, rating, duration, lat, lng
scaler = MinMaxScaler()
numeric_cols = ['ticket_log', 'rating', 'duration', 'lat', 'lng']
# Fit dan transform
```

**c) TF-IDF dari Tags:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert tags list menjadi string
df['tags_str'] = df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Fit TF-IDF
tfidf = TfidfVectorizer(max_features=20, min_df=1)
tfidf_matrix = tfidf.fit_transform(df['tags_str']).toarray()
```

**d) Gabungkan semua fitur menjadi feature matrix:**
```python
import numpy as np

# Gabungkan: [onehot_categories, normalized_numerics, tfidf_tags]
feature_matrix = np.hstack([category_onehot, numeric_normalized, tfidf_matrix])

# Simpan
np.save("data/processed/feature_matrix.npy", feature_matrix)
print(f"Feature matrix shape: {feature_matrix.shape}")
# Expected: (n_destinations, ~30)
```

**e) Simpan semua encoder/scaler:**
```python
import pickle

encoders = {
    'label_encoder_category': le_category,
    'scaler': scaler,
    'tfidf': tfidf,
    'feature_cols': numeric_cols,
    'category_order': ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]
}
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("✅ Encoders tersimpan")
```

---

### CELL 8: Markdown — Fase 3A: Content-Based Filtering

```markdown
## 🔍 Fase 3A: Content-Based Filtering (CBF)

### Konsep
CBF bekerja dengan menghitung **cosine similarity** antar vektor fitur destinasi.
Saat user memberikan preferensi (kategori, budget, dll), sistem:
1. Membangun **user profile vector** dari preferensi user
2. Menghitung cosine similarity antara user profile dan setiap destinasi
3. Mengembalikan destinasi dengan similarity tertinggi

### Formula
```
similarity(user, dest) = (user_vec · dest_vec) / (||user_vec|| × ||dest_vec||)
```

User profile vector dibangun dengan cara:
- Rata-rata fitur dari semua destinasi dalam kategori yang dipilih user
- Budget user → normalisasi dan masukkan ke dimensi ticket
```

---

### CELL 9: Training CBF (Code)

**Implementasikan class `ContentBasedFilter` secara LENGKAP:**

```python
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

class ContentBasedFilter:
    def __init__(self, feature_matrix: np.ndarray, destinations_df, encoders: dict):
        """
        Parameters:
        - feature_matrix: (n_dest, n_features) numpy array
        - destinations_df: DataFrame dengan kolom id, name, category, dll
        - encoders: dict berisi scaler, tfidf, label_encoder_category, category_order
        """
        self.feature_matrix = feature_matrix
        self.df = destinations_df.reset_index(drop=True)
        self.encoders = encoders
        self.similarity_matrix = None
        
    def fit(self):
        """Hitung similarity matrix antar semua destinasi."""
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"✅ CBF fitted. Similarity matrix shape: {self.similarity_matrix.shape}")
        return self
    
    def build_user_profile(self, categories: list, budget: float = None) -> np.ndarray:
        """
        Buat user profile vector dari preferensi user.
        
        Logika:
        1. Filter destinasi dalam kategori yang dipilih user
        2. Ambil rata-rata fitur dari destinasi tersebut
        3. Jika budget diberikan: override dimensi ticket_norm dengan nilai budget (dinormalisasi)
        
        Returns:
        - user_profile: (1, n_features) numpy array
        """
        # Implementasikan secara lengkap
        pass
    
    def recommend(
        self, 
        categories: list, 
        budget: float = None, 
        max_km: float = None, 
        home_lat: float = None, 
        home_lng: float = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Rekomendasikan top-N destinasi berdasarkan preferensi.
        
        Langkah:
        1. Build user profile vector
        2. Hitung cosine similarity user profile vs semua destinasi
        3. Filter berdasarkan kategori yang dipilih (hard constraint)
        4. Filter berdasarkan budget: ticket <= budget (jika budget given)
        5. Filter berdasarkan jarak dari home (jika max_km dan home_lat/lng given)
        6. Sort berdasarkan similarity score (descending)
        7. Return top_n destinasi sebagai DataFrame
        
        Returns:
        - DataFrame dengan kolom dari destinations_df + kolom 'cbf_score'
        """
        # Implementasikan secara lengkap
        pass
    
    def get_similar_destinations(self, dest_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Ambil N destinasi paling mirip dengan destinasi tertentu.
        Berguna untuk "You might also like" feature.
        """
        # Implementasikan
        pass
    
    def save(self, path: str):
        """Simpan model ke file pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'feature_matrix': self.feature_matrix,
                'df_index': self.df[['id', 'name', 'category']].to_dict('records')
            }, f)
        print(f"✅ CBF model disimpan ke {path}")
    
    @classmethod
    def load(cls, path: str, feature_matrix, destinations_df, encoders):
        """Load model dari pickle, restore similarity matrix."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(feature_matrix, destinations_df, encoders)
        obj.similarity_matrix = data['similarity_matrix']
        return obj
```

**Setelah implementasi class:**
1. Buat instance CBF
2. Fit model
3. Test recommend dengan berbagai kombinasi parameter:
   - Test 1: categories=["Alam"], budget=200000, top_n=5
   - Test 2: categories=["Kuliner", "Budaya"], budget=None, top_n=10
   - Test 3: categories=["Wisata", "Belanja"], budget=500000, max_km=30, home_lat=-6.9215, home_lng=107.6071
4. Print hasil setiap test
5. Simpan ke `models/cbf_model.pkl`

---

### CELL 10: Visualisasi CBF (Code)

Buat minimal 2 visualisasi:

**a) Heatmap Similarity Matrix (subsample 20 destinasi):**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Ambil 20 destinasi pertama untuk visualisasi (agar readable)
# Buat heatmap dengan nama destinasi sebagai label
# Title: "Cosine Similarity Matrix — Top 20 Destinations"
# Simpan sebagai data/processed/cbf_similarity_heatmap.png
```

**b) Bar chart: Average CBF Score per Kategori:**
```python
# Untuk setiap kategori, hitung rata-rata cbf_score dari hasil rekomendasi
# Visualisasikan sebagai horizontal bar chart
# Simpan sebagai data/processed/cbf_category_scores.png
```

---

### CELL 11: Markdown — Fase 3B: Multi-Agent Reinforcement Learning

```markdown
## 🤖 Fase 3B: Multi-Agent Reinforcement Learning

### Arsitektur
Dua agent bekerja secara sequential:

**Agent 1 — Filter Agent:**
- Input: state = [kategori user, budget, max_km, jam tersisa]
- Output: binary mask untuk setiap destinasi (1=lolos filter, 0=tidak)
- Policy: rule-based (deterministic) — bukan neural net
- Tujuan: eliminasi destinasi yang melanggar hard constraint

**Agent 2 — Selector Agent:**
- Input: state = [destinasi yang sudah dipilih, sisa budget, sisa waktu, kandidat tersisa]
- Output: index destinasi berikutnya yang dipilih
- Policy: Q-table (tabular RL dengan state diskretisasi)
- Tujuan: memaksimalkan reward (rating, variety, budget_efficiency)

### Reward Function
```
R(s, a, s') = w1 * rating_score(a)        # rating / 5.0
             + w2 * variety_bonus(s, a)    # +0.2 jika kategori baru
             + w3 * budget_efficiency(s,a) # (budget-ticket) / budget
             - penalty_overtime            # -1.0 jika waktu habis
             - penalty_overbudget          # -0.5 jika budget habis

Weights: w1=0.5, w2=0.2, w3=0.3
```

### State Representation (untuk Q-table)
State didiskretisasi menjadi tuple integer:
- `n_selected`: jumlah destinasi yang sudah dipilih (0–8)
- `budget_level`: sisa budget dalam bucket (0=habis, 1=<25%, 2=<50%, 3=<75%, 4=>75%)
- `time_level`: sisa waktu dalam bucket (0=habis, 1=<2jam, 2=<4jam, 3=<6jam, 4=>6jam)
- `dominant_cat`: kategori terbanyak yang sudah dipilih (int 0–4)

State space size: 9 × 5 × 5 × 5 = 1125 states
```

---

### CELL 12: Training RL Agent (Code)

**Implementasikan SEMUA komponen berikut secara lengkap:**

**a) Simulated Environment:**
```python
import numpy as np
import random
from collections import defaultdict

class BandungTravelEnv:
    """
    Simulated environment untuk training RL agent.
    Mensimulasikan user memilih destinasi satu per satu.
    """
    
    REWARD_WEIGHTS = {'rating': 0.5, 'variety': 0.2, 'budget_eff': 0.3}
    SPEED_KMH = 28  # kecepatan rata-rata di Bandung
    
    def __init__(self, destinations_df: pd.DataFrame, cbf_model: ContentBasedFilter):
        self.df = destinations_df.reset_index(drop=True)
        self.cbf = cbf_model
        self.n_destinations = len(self.df)
    
    def reset(self, params: dict) -> tuple:
        """
        Reset environment dengan parameter episode baru.
        
        Parameters:
        - params: dict dengan keys:
          - categories: list
          - budget: int (bisa None → set ke 999999999)
          - max_km: float (bisa None → set ke 999)
          - count: int (jumlah destinasi target)
          - startMin: int
          - endMin: int
          - home_lat: float
          - home_lng: float
        
        Returns:
        - (state_tuple, candidate_indices): state awal dan indeks kandidat yang valid
        """
        # Simpan params ke self
        # Gunakan CBF untuk mendapatkan kandidat awal (top-30 rekomendasi)
        # Reset: selected=[], spent=0, current_pos=home, current_time=startMin
        # Hitung dan return initial state
        pass
    
    def _get_state(self) -> tuple:
        """
        Diskretisasi state menjadi tuple untuk Q-table.
        Returns: (n_selected, budget_level, time_level, dominant_cat)
        """
        # Implementasikan diskretisasi
        pass
    
    def _haversine(self, lat1, lng1, lat2, lng2) -> float:
        """Hitung jarak Haversine dalam km."""
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    def step(self, action_idx: int) -> tuple:
        """
        Pilih destinasi berdasarkan action (indeks dalam kandidat).
        
        Returns:
        - (next_state, reward, done, info)
        
        Logika:
        1. Ambil destinasi target berdasarkan action_idx
        2. Hitung travel_time dan travel_km ke destinasi tersebut
        3. Update state: tambah ke selected, kurangi budget, update waktu
        4. Hitung reward berdasarkan reward function
        5. Cek done: len(selected) == count, atau kandidat habis, atau overtime
        6. Return next_state, reward, done, info
        """
        # Implementasikan secara lengkap
        pass
    
    def _calculate_reward(self, dest_row, travel_km: float) -> float:
        """
        Hitung reward untuk memilih destinasi tertentu.
        Gunakan REWARD_WEIGHTS: rating, variety, budget_eff
        Tambahkan penalty jika overtime atau over budget.
        """
        # Implementasikan lengkap sesuai formula di atas
        pass
    
    def get_valid_actions(self) -> list:
        """
        Return list index destinasi yang masih valid untuk dipilih:
        - Belum dipilih
        - Dalam budget (ticket <= sisa budget)
        - Tidak melebihi max_km (jika aktif)
        - Masih ada waktu tersisa untuk mengunjungi
        """
        # Implementasikan
        pass
    
    def generate_random_params(self) -> dict:
        """
        Generate parameter random untuk satu episode training.
        Digunakan agar agent dilatih pada berbagai skenario.
        
        Randomize:
        - categories: subset random dari 5 kategori (minimal 1)
        - budget: random antara 50000–2000000 (atau None 20% kemungkinan)
        - count: random 2–6
        - startMin: random antara 420–600 (07:00–10:00)
        - endMin: startMin + random antara 300–900 menit
        - max_km: random 20–80 km (atau None 40% kemungkinan)
        - home_lat/lng: random dari HOME_OPTIONS
        """
        # Implementasikan
        pass
```

**b) Q-Learning Agent:**
```python
class QLearningAgent:
    """
    Tabular Q-Learning agent untuk memilih destinasi secara sekuensial.
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,          # awal: full exploration
        epsilon_min: float = 0.05,     # minimum epsilon setelah decay
        epsilon_decay: float = 0.995,  # dikurangi setiap episode
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))  # state → {action_hash → value}
        self.training_history = []  # simpan reward per episode untuk plotting
    
    def _get_action_key(self, action_idx: int, candidate_ids: list) -> str:
        """
        Convert action ke hashable key.
        Gunakan destination id sebagai key agar portable antar episode.
        """
        return candidate_ids[action_idx] if action_idx < len(candidate_ids) else "unknown"
    
    def choose_action(self, state: tuple, valid_actions: list, candidate_ids: list) -> int:
        """
        Epsilon-greedy action selection.
        
        - Dengan probabilitas epsilon: pilih random (exploration)
        - Dengan probabilitas 1-epsilon: pilih action dengan Q-value tertinggi (exploitation)
        
        Parameters:
        - state: tuple (n_selected, budget_level, time_level, dominant_cat)
        - valid_actions: list of valid action indices
        - candidate_ids: list of destination ids (untuk action key)
        
        Returns:
        - action_idx: int
        """
        # Implementasikan epsilon-greedy
        pass
    
    def update(
        self, 
        state: tuple, 
        action_idx: int, 
        reward: float, 
        next_state: tuple, 
        done: bool,
        valid_next_actions: list,
        candidate_ids: list,
        next_candidate_ids: list
    ):
        """
        Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α × [r + γ × max_a'(Q(s',a')) - Q(s,a)]
        
        Jika done: tidak ada next state, gunakan hanya reward.
        """
        # Implementasikan Q-update
        pass
    
    def decay_epsilon(self):
        """Kurangi epsilon setelah setiap episode (tidak boleh di bawah epsilon_min)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Simpan Q-table dan hyperparameter ke pickle."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ RL Agent tersimpan ke {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load agent dari pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(learning_rate=data['lr'], discount_factor=data['gamma'])
        agent.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        agent.epsilon = data['epsilon']
        return agent
```

**c) Training Loop:**
```python
from tqdm import tqdm

def train_rl_agent(
    env: BandungTravelEnv, 
    agent: QLearningAgent, 
    n_episodes: int = 3000,
    log_interval: int = 500
) -> list:
    """
    Main training loop.
    
    Setiap episode:
    1. Generate random params
    2. Reset environment
    3. Loop sampai done:
       a. Pilih action (epsilon-greedy)
       b. Jalankan step di environment
       c. Update Q-table
       d. Update state
    4. Decay epsilon
    5. Log reward
    
    Returns:
    - episode_rewards: list of total reward per episode
    """
    episode_rewards = []
    
    for ep in tqdm(range(n_episodes), desc="Training RL Agent"):
        params = env.generate_random_params()
        state, candidates = env.reset(params)
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action = agent.choose_action(state, valid_actions, [env.df.iloc[i]['id'] for i in candidates])
            next_state, reward, done, info = env.step(action)
            
            next_valid = env.get_valid_actions()
            agent.update(state, action, reward, next_state, done, 
                        next_valid, [env.df.iloc[i]['id'] for i in candidates],
                        [env.df.iloc[i]['id'] for i in env.candidates] if hasattr(env, 'candidates') else [])
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        agent.training_history.append(total_reward)
        
        if (ep + 1) % log_interval == 0:
            recent_avg = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {ep+1}/{n_episodes} | Avg Reward (last {log_interval}): {recent_avg:.4f} | ε: {agent.epsilon:.3f}")
    
    return episode_rewards

# Jalankan training
env = BandungTravelEnv(df_clean, cbf_model)
rl_agent = QLearningAgent()
rewards = train_rl_agent(env, rl_agent, n_episodes=3000)

# Simpan agent
rl_agent.save("models/rl_agent.pkl")
```

---

### CELL 13: Visualisasi Training RL (Code)

Buat visualisasi berikut:

**a) Learning Curve:**
```python
# Plot reward per episode dengan rolling average (window=100)
# X-axis: Episode
# Y-axis: Total Reward
# Dua line: raw reward (transparan) + rolling average (solid)
# Title: "RL Agent Learning Curve — 3000 Episodes"
# Simpan sebagai data/processed/rl_learning_curve.png
```

**b) Epsilon Decay:**
```python
# Plot epsilon vs episode
# Tampilkan bagaimana exploration menurun seiring training
```

**c) Reward Distribution (akhir training vs awal):**
```python
# Histogram perbandingan reward: 500 episode pertama vs 500 episode terakhir
# Tampilkan bahwa agen sudah "belajar" dengan distribusi reward yang lebih tinggi
```

---

### CELL 14: Markdown — Fase 3C: Route Optimizer

```markdown
## 🗺️ Fase 3C: Route Optimizer — TSP Nearest-Neighbor

### Konsep
Setelah RL Agent memilih N destinasi terbaik, urutan kunjungan dioptimalkan menggunakan:
1. **Haversine Distance** untuk kalkulasi jarak antar titik (fallback)
2. **OSRM API** untuk waktu tempuh nyata (primary — jika available)
3. **Nearest-Neighbor Heuristic** untuk route ordering

### OSRM Public API
```
GET http://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}
Response: { routes: [{ duration: float, distance: float }] }
```
Note: `distance` dalam meter, `duration` dalam detik.
```

---

### CELL 15: Route Optimizer (Code)

**Implementasikan class `RouteOptimizer` secara LENGKAP:**

```python
import requests
import math
import time

class RouteOptimizer:
    SPEED_KMH = 28  # kecepatan default kota (fallback)
    OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"
    
    def __init__(self, use_osrm: bool = True, osrm_timeout: float = 5.0):
        self.use_osrm = use_osrm
        self.osrm_timeout = osrm_timeout
    
    def haversine_km(self, lat1, lng1, lat2, lng2) -> float:
        """Kalkulasi jarak Haversine dalam km."""
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    def osrm_travel_time(self, lat1, lng1, lat2, lng2) -> tuple:
        """
        Ambil waktu tempuh dari OSRM API.
        
        Returns:
        - (distance_km: float, duration_min: float)
        - Jika OSRM gagal/timeout: return haversine_km dan estimasi dari speed
        """
        try:
            url = f"{self.OSRM_BASE}/{lng1},{lat1};{lng2},{lat2}?overview=false"
            resp = requests.get(url, timeout=self.osrm_timeout)
            resp.raise_for_status()
            route = resp.json()['routes'][0]
            distance_km = route['distance'] / 1000
            duration_min = route['duration'] / 60
            return distance_km, duration_min
        except Exception:
            # Fallback ke Haversine + speed estimation
            dist_km = self.haversine_km(lat1, lng1, lat2, lng2)
            dur_min = (dist_km / self.SPEED_KMH) * 60
            return dist_km, dur_min
    
    def nearest_neighbor_route(
        self, 
        home: dict,          # {"lat": float, "lng": float}
        destinations: list,  # list of dict dengan lat, lng, id, name, ...
    ) -> list:
        """
        Urutkan destinasi menggunakan nearest-neighbor heuristic.
        
        Algoritma:
        1. Mulai dari home
        2. Setiap iterasi: pilih destinasi terdekat yang belum dikunjungi
        3. Lanjutkan sampai semua destinasi dikunjungi
        4. Return list destinasi dalam urutan optimal
        
        Returns:
        - ordered_destinations: list of dict (urutan optimal)
        """
        # Implementasikan
        pass
    
    def build_itinerary(
        self, 
        home: dict,
        home_name: str,
        ordered_destinations: list,
        start_min: int,
        end_min: int
    ) -> dict:
        """
        Buat itinerary lengkap dengan jadwal waktu.
        
        Untuk setiap destinasi dalam urutan:
        1. Hitung travel_km dan travel_min dari titik sebelumnya
        2. Hitung arrive_at = current_time + travel_min
        3. Hitung depart_at = arrive_at + destination.duration
        4. Update current_time = depart_at
        
        Tambahkan return trip ke home di akhir.
        
        Returns:
        - itinerary dict sesuai kontrak frontend:
          {
            "steps": [...],
            "totalCost": int,
            "totalKm": float,
            "totalTime": int,
            "returnKm": float,
            "returnMin": int,
            "arriveHome": int,
            "overBudget": bool,  # ini sebenarnya "overTime" — cek apakah arriveHome > endMin
            "spareMin": int      # endMin - arriveHome
          }
        """
        # Implementasikan sesuai spesifikasi output frontend
        pass
```

**Test RouteOptimizer:**
```python
# Test dengan 4 destinasi dari dataset
# Tampilkan itinerary dalam format readable
optimizer = RouteOptimizer(use_osrm=True)
# ... test
```

---

### CELL 16: Integration Test — Full Pipeline (Code)

**Test end-to-end pipeline dengan menyimulasikan request dari frontend:**

```python
def full_pipeline_test(params: dict) -> dict:
    """
    Simulasi lengkap dari params user → itinerary output.
    
    Langkah:
    1. CBF Recommend: dapatkan top-20 kandidat
    2. RL Agent: pilih N destinasi terbaik dari kandidat
    3. Route Optimizer: urutkan rute nearest-neighbor
    4. Build Itinerary: hitung jadwal waktu
    5. Return itinerary dict
    """
    # 1. CBF
    candidates = cbf_model.recommend(
        categories=params['categories'],
        budget=params['budget'],
        max_km=params['maxKm'],
        home_lat=params['home']['lat'],
        home_lng=params['home']['lng'],
        top_n=20
    )
    
    # 2. RL Agent — pilih dari kandidat
    # ... simulasikan pemilihan RL agent
    
    # 3. Route optimization
    ordered = optimizer.nearest_neighbor_route(params['home'], selected_dests)
    
    # 4. Build itinerary
    itinerary = optimizer.build_itinerary(
        home=params['home'],
        home_name=params['homeName'],
        ordered_destinations=ordered,
        start_min=params['startMin'],
        end_min=params['endMin']
    )
    
    return itinerary

# Jalankan 3 test case berbeda
test_cases = [
    {
        "home": {"lat": -6.9215, "lng": 107.6071},
        "homeName": "Alun-Alun Bandung",
        "count": 4,
        "maxKm": None,
        "startMin": 9 * 60,  # 09:00
        "endMin": 21 * 60,   # 21:00
        "budget": 500000,
        "categories": ["Alam", "Kuliner"]
    },
    {
        "home": {"lat": -6.8126, "lng": 107.6178},
        "homeName": "Pasar Lembang",
        "count": 3,
        "maxKm": 25,
        "startMin": 8 * 60,
        "endMin": 18 * 60,
        "budget": 200000,
        "categories": ["Alam"]
    },
    {
        "home": {"lat": -6.9145, "lng": 107.6020},
        "homeName": "Stasiun Bandung",
        "count": 5,
        "maxKm": None,
        "startMin": 10 * 60,
        "endMin": 20 * 60,
        "budget": None,
        "categories": ["Kuliner", "Budaya", "Belanja"]
    }
]

for i, params in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"TEST CASE {i+1}: {params['homeName']}")
    result = full_pipeline_test(params)
    print(f"Destinasi terpilih: {len(result['steps'])}")
    for step in result['steps']:
        print(f"  {step['idx']}. {step['dest']['name']} ({step['dest']['category']}) — tiba {step['arriveAt']//60:02d}:{step['arriveAt']%60:02d}")
    print(f"Total biaya: Rp {result['totalCost']:,}")
    print(f"Total jarak: {result['totalKm']:.1f} km")
    print(f"Tiba kembali: {result['arriveHome']//60:02d}:{result['arriveHome']%60:02d}")
    print(f"Over time: {result['overBudget']}")
```

---

### CELL 17: Evaluasi Model (Code)

**Implementasikan semua metrik evaluasi berikut:**

**a) Evaluasi CBF:**
```python
# Metrik 1: Intra-list Diversity
# Ukur rata-rata diversity dalam setiap hasil rekomendasi
# Diversity = 1 - cosine_similarity antar item dalam list
# Makin tinggi = makin beragam

# Metrik 2: Coverage
# Berapa % destinasi dalam dataset yang pernah direkomendasikan
# Jalankan rekomendasi untuk 100 user profile random
# Coverage = len(unique_recommended) / total_destinations

# Metrik 3: Category Balance
# Distribusi kategori dalam hasil rekomendasi vs distribusi asli di dataset

print("=== CBF Evaluation ===")
print(f"Intra-list Diversity: {diversity:.4f}")
print(f"Catalog Coverage: {coverage:.4f}")
print("Category Distribution:")
# ... print comparison
```

**b) Evaluasi RL:**
```python
# Metrik 1: Average Episode Reward — compare epoch pertama vs terakhir
# Metrik 2: Success Rate — % episode yang selesai tanpa overtime/overbudget
# Metrik 3: Average destinations selected — rata-rata berapa destinasi yang berhasil dipilih

print("\n=== RL Agent Evaluation ===")
print(f"Avg Reward (first 100 episodes): {first_100_avg:.4f}")
print(f"Avg Reward (last 100 episodes): {last_100_avg:.4f}")
print(f"Improvement: {((last_100_avg - first_100_avg) / abs(first_100_avg)) * 100:.1f}%")
print(f"Success Rate: {success_rate:.2%}")
print(f"Avg Destinations Selected: {avg_selected:.2f}")
```

**c) Evaluasi Route Optimizer:**
```python
# Metrik: Rata-rata total jarak yang dihasilkan vs baseline (random order)
# Jalankan 100 test case random
# Bandingkan TSP nearest-neighbor vs random shuffle

# Hitung berapa % improvement vs random
print("\n=== Route Optimizer Evaluation ===")
print(f"Avg distance (NN heuristic): {nn_avg_km:.2f} km")
print(f"Avg distance (random order): {random_avg_km:.2f} km")
print(f"Route Optimization Improvement: {improvement:.1f}%")
```

---

### CELL 18: Export & Summary (Code)

```python
print("\n" + "="*60)
print("📦 EXPORT SUMMARY — Bandung AI Travel Agent")
print("="*60)

# List semua file yang dihasilkan
import os
from datetime import date

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

# Summary statistik dataset
df = pd.read_csv("data/processed/destinations.csv")
print(f"\n📊 Dataset Summary:")
print(f"  Total destinasi: {len(df)}")
print(df.groupby('category').agg({'name': 'count', 'rating': 'mean', 'ticket': 'median'}).round(2))

print(f"\n⏰ Last Updated: {date.today()}")
print("\n✅ Semua model dan data siap digunakan oleh Notebook 2 dan Backend!")
```

---

### CELL 19: Markdown — Penutup Notebook 1

```markdown
## ✅ Notebook 1 Selesai

### File yang dihasilkan:
| File | Keterangan |
|---|---|
| `data/processed/destinations.csv` | Dataset bersih (≥50 destinasi) |
| `data/processed/feature_matrix.npy` | Matriks fitur untuk CBF & RL |
| `data/last_updated.txt` | Timestamp update data |
| `models/cbf_model.pkl` | Model Content-Based Filtering |
| `models/rl_agent.pkl` | Q-Learning RL Agent |
| `models/scaler.pkl` | MinMaxScaler untuk normalisasi |
| `models/label_encoders.pkl` | LabelEncoder + TF-IDF vectorizer |

### Langkah Selanjutnya:
1. Buka **Notebook 2** (`02_llm_storyteller.ipynb`) untuk testing LLM integration
2. Deploy ke backend FastAPI menggunakan file model di atas
3. Jalankan ulang notebook ini secara berkala (1 bulan sekali) untuk refresh data

### Catatan Penting:
- RL agent dilatih dengan simulated environment (bukan data user nyata)
- Saat sistem sudah live, simpan log interaksi user → fine-tune agent secara periodik
- OSRM public API memiliki rate limit — jangan gunakan untuk >100 request/menit
```

---

---

# ═══════════════════════════════════════════════
# NOTEBOOK 2: LLM STORYTELLER
# File: notebooks/02_llm_storyteller.ipynb
# ═══════════════════════════════════════════════

## TUJUAN NOTEBOOK 2
1. Load model hasil Notebook 1
2. Jalankan full pipeline rekomendasi (CBF + RL + Route Optimizer)
3. Kirim output itinerary ke Groq API untuk menghasilkan narasi
4. Validasi bahwa output sesuai kontrak frontend
5. Dokumentasikan prompt engineering dan tuning

---

## DEPENDENCY NOTEBOOK 2 TERHADAP NOTEBOOK 1

**WAJIB dijalankan terlebih dahulu Notebook 1** sebelum Notebook 2.

File yang dibutuhkan:
- `data/processed/destinations.csv` 
- `data/processed/feature_matrix.npy`
- `models/cbf_model.pkl`
- `models/rl_agent.pkl`
- `models/label_encoders.pkl`
- `data/last_updated.txt`

Notebook 2 harus **memvalidasi keberadaan file-file ini** di cell pertama dan raise error yang informatif jika ada yang missing.

---

## STRUKTUR CELL NOTEBOOK 2

### CELL 0: Judul & Deskripsi (Markdown)
```markdown
# 🤖 Bandung AI Travel Agent — LLM Storyteller
## Notebook 02: Integration Test + Groq API Narasi

**Capstone Project · Telkom University · Program Studi Data Science**

Notebook ini:
1. Load model dari Notebook 01
2. Menjalankan full pipeline rekomendasi
3. Mengintegrasikan Groq API (free tier) untuk menghasilkan narasi perjalanan
4. Memvalidasi output sesuai kontrak frontend React
5. Mendokumentasikan prompt engineering

**Prasyarat:** Notebook 01 sudah dijalankan dan semua model tersimpan.
**API:** Groq API (https://console.groq.com) — daftar gratis, 14.400 req/hari
**Model LLM:** llama-3.1-8b-instant
```

---

### CELL 1: Setup & Dependency Check (Code)

```python
# Install package yang dibutuhkan
# !pip install groq pandas numpy scikit-learn requests

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# VALIDASI: pastikan semua output Notebook 1 sudah ada
# ============================================================
required_files = {
    "data/processed/destinations.csv": "Dataset destinasi",
    "data/processed/feature_matrix.npy": "Feature matrix CBF",
    "models/cbf_model.pkl": "Content-Based Filtering model",
    "models/rl_agent.pkl": "RL Agent model",
    "models/label_encoders.pkl": "Label encoders & scalers",
    "data/last_updated.txt": "Timestamp update data"
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
        "\n⛔ Beberapa file model tidak ditemukan!\n"
        "Pastikan Notebook 01 (01_recommendation_engine.ipynb) sudah dijalankan terlebih dahulu.\n"
        "Kemudian jalankan ulang cell ini."
    )

print("\n✅ Semua dependensi Notebook 1 tersedia. Siap lanjut!")
```

---

### CELL 2: Load Models & Data (Code)

```python
# Load semua komponen dari Notebook 1

# 1. Load destinations dataset
df = pd.read_csv("data/processed/destinations.csv")
print(f"✅ Dataset loaded: {len(df)} destinasi")

# 2. Load feature matrix
feature_matrix = np.load("data/processed/feature_matrix.npy")
print(f"✅ Feature matrix loaded: {feature_matrix.shape}")

# 3. Load label encoders
with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
print(f"✅ Encoders loaded: {list(encoders.keys())}")

# 4. Load CBF model
# (re-instantiate class CBF dari notebook 1 — paste class definition di sini)
# Load similarity matrix dari pickle
with open("models/cbf_model.pkl", "rb") as f:
    cbf_data = pickle.load(f)
print(f"✅ CBF model loaded. Similarity matrix: {cbf_data['similarity_matrix'].shape}")

# 5. Load RL agent
with open("models/rl_agent.pkl", "rb") as f:
    rl_data = pickle.load(f)
from collections import defaultdict
q_table = defaultdict(lambda: defaultdict(float), rl_data['q_table'])
print(f"✅ RL Agent loaded. Q-table states: {len(q_table)}")

# 6. Load last_updated
with open("data/last_updated.txt", "r") as f:
    data_last_updated = f.read().strip()
print(f"✅ Data last updated: {data_last_updated}")
```

---

### CELL 3: Re-instantiate Pipeline Classes (Code)

**WAJIB:** Paste ulang (copy-paste) definisi class-class berikut dari Notebook 1 (tidak boleh import dari file external — notebook harus self-contained):
- Class `ContentBasedFilter` (dengan method `recommend` dan `get_similar_destinations`)
- Class `RouteOptimizer` (dengan method `nearest_neighbor_route` dan `build_itinerary`)
- Fungsi `haversine_km`

Kemudian instantiate:
```python
# Rebuild CBF model dari loaded data
cbf_model = ContentBasedFilter(feature_matrix, df, encoders)
cbf_model.similarity_matrix = cbf_data['similarity_matrix']

# Rebuild Route Optimizer
optimizer = RouteOptimizer(use_osrm=True)

print("✅ Semua pipeline class berhasil di-instantiate")
```

---

### CELL 4: Markdown — Groq API Setup

```markdown
## 🔑 Groq API Setup

### Cara mendapatkan API Key (gratis):
1. Buka https://console.groq.com
2. Daftar / login dengan Google
3. Masuk ke menu "API Keys"
4. Klik "Create API Key"
5. Copy key dan masukkan ke cell berikut

### Batas Free Tier:
- 14.400 requests/hari
- 30 requests/menit
- Model: llama-3.1-8b-instant (recommended untuk kecepatan)
- Alternatif: llama3-70b-8192 (lebih pintar, lebih lambat)

### Catatan Keamanan:
- **JANGAN commit API key ke Git**
- Gunakan environment variable atau .env file
- Di notebook ini: masukkan manual atau baca dari file .env
```

---

### CELL 5: Groq API Configuration (Code)

```python
import os

# ============================================================
# KONFIGURASI API KEY
# Opsi 1: Set langsung (untuk testing — jangan commit ke Git!)
# Opsi 2: Dari environment variable (recommended untuk production)
# Opsi 3: Dari file .env (recommended untuk development)
# ============================================================

# Opsi 2 (recommended):
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Opsi 3: Baca dari .env (jika ada)
if GROQ_API_KEY is None:
    env_path = Path(".env")
    if env_path.exists():
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    GROQ_API_KEY = line.split("=", 1)[1].strip()
                    break

# Opsi 1 (fallback — ganti dengan key kamu):
if GROQ_API_KEY is None:
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # ← GANTI INI
    print("⚠️  Menggunakan API key yang di-hardcode. Jangan commit ke Git!")

if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    print("⛔ API Key belum diset! Isi GROQ_API_KEY sebelum melanjutkan.")
else:
    print(f"✅ GROQ_API_KEY terdeteksi: {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")

# Konfigurasi model
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 1000
GROQ_TEMPERATURE = 0.7
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

print(f"✅ Model: {GROQ_MODEL}")
print(f"✅ Max tokens: {GROQ_MAX_TOKENS}")
print(f"✅ Temperature: {GROQ_TEMPERATURE}")
```

---

### CELL 6: Groq API Client (Code)

**Implementasikan class `GroqStoryteller` secara LENGKAP:**

```python
import requests
import time
import json

class GroqStoryteller:
    """
    Client untuk Groq API — menghasilkan narasi perjalanan.
    Compatible dengan free tier (rate limit handling built-in).
    """
    
    RETRY_DELAYS = [1, 2, 5]  # detik delay untuk retry
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_api(self, messages: list, system_prompt: str = None) -> str:
        """
        Panggil Groq API dengan retry logic.
        
        Parameters:
        - messages: list of {"role": "user"/"assistant", "content": str}
        - system_prompt: opsional system message
        
        Returns:
        - content: str (respons dari LLM)
        
        Error handling:
        - Rate limit (429): retry dengan exponential backoff
        - Server error (5xx): retry
        - Timeout: retry
        - Lainnya: raise exception
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        for attempt, delay in enumerate(self.RETRY_DELAYS + [None]):
            try:
                resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=30)
                
                if resp.status_code == 429:  # Rate limit
                    if delay is None:
                        raise Exception("Rate limit exceeded, semua retry gagal")
                    print(f"  ⏳ Rate limited. Retry dalam {delay}s...")
                    time.sleep(delay)
                    continue
                
                resp.raise_for_status()
                return resp.json()['choices'][0]['message']['content']
            
            except requests.exceptions.Timeout:
                if delay is None:
                    raise
                print(f"  ⏳ Timeout. Retry dalam {delay}s...")
                time.sleep(delay)
            
            except requests.exceptions.RequestException as e:
                if delay is None or resp.status_code < 500:
                    raise
                time.sleep(delay)
    
    def build_story_prompt(self, itinerary: dict, params: dict, lang: str = "id") -> str:
        """
        Build prompt dinamis untuk menghasilkan narasi itinerary.
        
        WAJIB memuat semua informasi berikut dalam prompt:
        - Bahasa instruksi (id/en)
        - Titik keberangkatan
        - Total destinasi dan kategori
        - Setiap destinasi: nama, kategori, harga tiket, durasi, rating
        - Total biaya estimasi
        - Total jarak dan waktu
        - Jam mulai dan selesai
        - Budget user (jika ada)
        
        Instruksi output yang harus ada dalam prompt:
        - "Respond ONLY in valid JSON format"
        - "No markdown, no preamble, no explanation outside JSON"
        - JSON structure yang diharapkan (lihat schema di bawah)
        
        Output JSON schema yang diminta dari LLM:
        {
          "intro": "string — 2-3 kalimat pembuka yang engaging",
          "highlights": ["string", ...] — 1 item per destinasi, max 2 kalimat,
          "tips": ["string", ...] — 2-4 tips praktis,
          "closing": "string — 1 kalimat penutup yang memorable",
          "vibe": "string — 1-2 kata deskripsi vibe perjalanan"
        }
        
        Tone instructions:
        - Bahasa Indonesia: casual, friendly, seperti teman yang merekomendasikan tempat
        - English: enthusiastic travel blogger style
        - Gunakan emoji secukupnya (1-2 per item)
        - Sebut setiap nama destinasi secara natural
        - Jangan gunakan bullet points dalam string output
        """
        # Implementasikan prompt builder secara lengkap
        
        destinations_text = "\n".join([
            f"  {i+1}. {step['dest']['name']} ({step['dest']['category']})"
            f" — Tiket: {'Gratis' if step['dest']['ticket'] == 0 else f'Rp {step[\"dest\"][\"ticket\"]:,}'}"
            f", Durasi: {step['dest']['duration']} menit"
            f", Rating: ⭐{step['dest']['rating']}"
            for i, step in enumerate(itinerary['steps'])
        ])
        
        lang_instruction = (
            "Respond in casual Bahasa Indonesia (colloquial, friendly, millennial style)"
            if lang == "id"
            else "Respond in casual English (enthusiastic travel blogger style)"
        )
        
        start_time = f"{params['startMin']//60:02d}:{params['startMin']%60:02d}"
        end_time = f"{params['endMin']//60:02d}:{params['endMin']%60:02d}"
        
        # Bangun prompt lengkap (gunakan f-string, masukkan semua variabel)
        prompt = f"""
{lang_instruction}. You are BandungBuddy, an enthusiastic local travel guide.

Generate a travel narrative for this Bandung itinerary. Respond ONLY in valid JSON format.
No markdown code blocks, no explanation, just raw JSON.

=== TRIP DETAILS ===
Starting point: {params['homeName']}
Date: Today
Time window: {start_time} – {end_time}
Budget: {'Rp ' + f"{params['budget']:,}" if params.get('budget') else 'No limit'}
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
    "<2-4 practical tips based on the actual destinations and time>",
    "<tip 2>",
    "<tip 3>"
  ],
  "closing": "<1 memorable closing sentence>",
  "vibe": "<1-2 words describing the trip vibe — e.g.: Petualang Alam, Kuliner Enthusiast>"
}}

IMPORTANT:
- Each highlights array must have EXACTLY {len(itinerary['steps'])} items (one per destination)
- Tips must be specific to the destinations chosen, not generic
- Use casual language, 1-2 emoji per highlight
- Do NOT include any text outside the JSON object
"""
        return prompt
    
    def generate_story(self, itinerary: dict, params: dict, lang: str = "id") -> dict:
        """
        Generate narasi itinerary menggunakan Groq API.
        
        Langkah:
        1. Build prompt
        2. Call API
        3. Parse JSON response
        4. Validasi struktur response
        5. Jika parsing gagal: return fallback template
        
        Returns:
        - story dict: {"intro", "highlights", "tips", "closing", "vibe"}
        """
        prompt = self.build_story_prompt(itinerary, params, lang)
        
        system_prompt = (
            "You are BandungBuddy, a local travel guide AI. "
            "You ALWAYS respond in valid JSON format only. "
            "Never include markdown, preamble, or explanation. "
            "Just output the raw JSON object requested."
        )
        
        try:
            raw_response = self._call_api(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt
            )
            
            # Clean response: hapus markdown code blocks jika ada
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            
            story = json.loads(clean)
            
            # Validasi struktur
            required_keys = ["intro", "highlights", "tips", "closing", "vibe"]
            for key in required_keys:
                if key not in story:
                    raise ValueError(f"Missing key in LLM response: {key}")
            
            # Pastikan highlights count match
            n_dests = len(itinerary['steps'])
            if len(story['highlights']) != n_dests:
                # Pad atau trim
                while len(story['highlights']) < n_dests:
                    story['highlights'].append(f"Kunjungi {itinerary['steps'][len(story['highlights'])]['dest']['name']} dan rasakan pengalamannya sendiri!")
                story['highlights'] = story['highlights'][:n_dests]
            
            return story
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  ⚠️ JSON parsing gagal: {e}. Menggunakan fallback template.")
            return self._fallback_story(itinerary, params, lang)
        
        except Exception as e:
            print(f"  ❌ API call gagal: {e}. Menggunakan fallback template.")
            return self._fallback_story(itinerary, params, lang)
    
    def _fallback_story(self, itinerary: dict, params: dict, lang: str = "id") -> dict:
        """
        Template fallback jika Groq API gagal atau parsing error.
        Pastikan output tetap valid dan bisa ditampilkan di frontend.
        """
        if lang == "id":
            cats = list(set(s['dest']['category'] for s in itinerary['steps']))
            cat_str = " dan ".join(cats)
            first = itinerary['steps'][0]['dest']['name']
            last = itinerary['steps'][-1]['dest']['name']
            
            return {
                "intro": f"Hari ini kamu akan menjelajahi {len(itinerary['steps'])} destinasi seru di Bandung, mulai dari **{params['homeName']}**. Siap-siap nikmatin pengalaman {cat_str} yang nggak terlupakan!",
                "highlights": [
                    f"**{s['dest']['name']}** — {s['dest']['desc']}. Alokasikan {s['dest']['duration']} menit untuk pengalaman terbaik. 📍"
                    for s in itinerary['steps']
                ],
                "tips": [
                    "Cek cuaca sebelum berangkat, terutama untuk destinasi alam.",
                    "Siapkan e-wallet dan uang cash untuk tiket masuk dan kuliner.",
                    f"Estimasi total pengeluaran Rp {itinerary['totalCost']:,} — sesuaikan dengan anggaran.",
                    "Simpan offline maps untuk area yang sinyalnya mungkin lemah."
                ],
                "closing": f"Dari {first} sampai {last} — ini bakal jadi hari yang memorable di Kota Kembang! 🌸",
                "vibe": " & ".join(cats[:2]) if len(cats) > 1 else cats[0]
            }
        else:
            # English fallback
            return {
                "intro": f"Get ready for an amazing day exploring {len(itinerary['steps'])} handpicked destinations in Bandung, starting from {params['homeName']}!",
                "highlights": [
                    f"**{s['dest']['name']}** — {s['dest']['desc']}. Plan to spend {s['dest']['duration']} minutes here. 📍"
                    for s in itinerary['steps']
                ],
                "tips": [
                    "Check the weather forecast, especially for nature spots.",
                    "Bring both e-wallet and cash for entrance fees and street food.",
                    f"Budget estimate: Rp {itinerary['totalCost']:,} — plan accordingly.",
                    "Download offline maps for areas with potentially weak signal."
                ],
                "closing": f"From start to finish — this is going to be an unforgettable Bandung adventure! 🌸",
                "vibe": "Adventure Mix"
            }
```

---

### CELL 7: Full Pipeline dengan LLM (Code)

**Implementasikan fungsi `generate_full_itinerary` yang menggabungkan semua komponen:**

```python
def generate_full_itinerary(
    params: dict,
    cbf_model,
    rl_data: dict,
    optimizer: RouteOptimizer,
    storyteller: GroqStoryteller,
    df: pd.DataFrame,
    data_last_updated: str,
    lang: str = "id"
) -> dict:
    """
    Full pipeline dari params user → response JSON yang siap dikirim ke frontend.
    
    Parameters:
    - params: dict sesuai kontrak input dari frontend (home, count, budget, dll)
    - cbf_model: instance ContentBasedFilter
    - rl_data: dict berisi q_table dari RL agent
    - optimizer: instance RouteOptimizer
    - storyteller: instance GroqStoryteller
    - df: DataFrame destinations
    - data_last_updated: string tanggal
    - lang: "id" atau "en"
    
    Returns:
    - response: dict sesuai kontrak output frontend (steps, totalCost, story, dll)
    
    Langkah:
    1. CBF Recommendation → kandidat top-30
    2. RL-Guided Selection → pilih N terbaik dari kandidat
       - Gunakan Q-table untuk guided selection
       - Jika state tidak ada di Q-table: fallback ke CBF score
    3. Route Optimization → nearest-neighbor ordering
    4. Build Itinerary → schedule dengan waktu
    5. LLM Storytelling → generate narasi
    6. Assemble final response
    """
    
    # Step 1: CBF Recommendation
    candidates = cbf_model.recommend(
        categories=params.get('categories', []),
        budget=params.get('budget'),
        max_km=params.get('maxKm'),
        home_lat=params['home']['lat'],
        home_lng=params['home']['lng'],
        top_n=30
    )
    
    if len(candidates) == 0:
        # Relaxed fallback: no category filter
        candidates = cbf_model.recommend(
            categories=[],
            budget=None,
            max_km=None,
            home_lat=params['home']['lat'],
            home_lng=params['home']['lng'],
            top_n=30
        )
    
    # Step 2: RL-Guided Selection
    # Gunakan Q-table untuk memilih N destinasi dari kandidat
    # Implementasi logika RL inference (bukan training)
    selected_destinations = []
    remaining = candidates.to_dict('records')
    spent = 0
    current_lat, current_lng = params['home']['lat'], params['home']['lng']
    current_time = params['startMin']
    
    for i in range(params.get('count', 4)):
        if not remaining:
            break
        
        # Build current state
        budget_spent_ratio = spent / params['budget'] if params.get('budget') else 0
        time_spent_ratio = (current_time - params['startMin']) / (params['endMin'] - params['startMin'])
        n_selected = len(selected_destinations)
        
        # Diskretisasi state untuk Q-table lookup
        budget_level = min(4, int((1 - budget_spent_ratio) * 4))
        time_level = min(4, int((1 - time_spent_ratio) * 4))
        dominant_cat_id = 0  # default
        if selected_destinations:
            from collections import Counter
            cat_counts = Counter([d['category'] for d in selected_destinations])
            dominant = cat_counts.most_common(1)[0][0]
            cat_order = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"]
            dominant_cat_id = cat_order.index(dominant) if dominant in cat_order else 0
        
        state = (n_selected, budget_level, time_level, dominant_cat_id)
        
        # Check Q-table untuk best action
        best_dest = None
        best_q = -float('inf')
        
        for dest in remaining:
            action_key = dest['id']
            q_val = rl_data.get('q_table', {}).get(state, {}).get(action_key, 0)
            # Combine Q-value dengan CBF score (weighted)
            cbf_score = dest.get('cbf_score', 0)
            combined = 0.6 * q_val + 0.4 * cbf_score
            if combined > best_q:
                best_q = combined
                best_dest = dest
        
        if best_dest is None:
            best_dest = remaining[0]
        
        selected_destinations.append(best_dest)
        remaining = [d for d in remaining if d['id'] != best_dest['id']]
        spent += best_dest.get('ticket', 0)
        current_time += best_dest.get('duration', 60)
    
    # Step 3: Route Optimization
    ordered = optimizer.nearest_neighbor_route(params['home'], selected_destinations)
    
    # Step 4: Build Itinerary
    itinerary = optimizer.build_itinerary(
        home=params['home'],
        home_name=params['homeName'],
        ordered_destinations=ordered,
        start_min=params['startMin'],
        end_min=params['endMin']
    )
    
    # Step 5: LLM Storytelling
    story = storyteller.generate_story(itinerary, params, lang=lang)
    
    # Step 6: Assemble response
    response = {
        **itinerary,  # steps, totalCost, totalKm, totalTime, returnKm, returnMin, arriveHome, overBudget, spareMin
        "story": story,
        "data_last_updated": data_last_updated
    }
    
    return response
```

---

### CELL 8: Markdown — Groq API Testing

```markdown
## 🧪 Testing Groq API

Sebelum full pipeline test, validasi terlebih dahulu koneksi ke Groq API
dengan prompt sederhana.
```

---

### CELL 9: Test Koneksi Groq API (Code)

```python
# Inisialisasi storyteller
storyteller = GroqStoryteller(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    max_tokens=GROQ_MAX_TOKENS,
    temperature=GROQ_TEMPERATURE
)

# Test koneksi dengan prompt sederhana
print("🔌 Testing Groq API connection...")
try:
    test_response = storyteller._call_api(
        messages=[{"role": "user", "content": "Respond with exactly: {\"status\": \"ok\", \"message\": \"Groq API berfungsi!\"}"}],
        system_prompt="You are a test API. Respond only with valid JSON as instructed."
    )
    result = json.loads(test_response)
    print(f"✅ Groq API OK: {result}")
    
except Exception as e:
    print(f"❌ Groq API Test GAGAL: {e}")
    print("Cek API key dan koneksi internet.")
    print("Notebook akan tetap berjalan dengan fallback template.")
```

---

### CELL 10: Full Integration Test — 3 Skenario (Code)

```python
# Definisikan 3 test case mencakup berbagai skenario
test_cases = [
    {
        "name": "Wisata Alam Selatan Bandung",
        "params": {
            "home": {"lat": -6.9215, "lng": 107.6071},
            "homeName": "Alun-Alun Bandung",
            "count": 3,
            "maxKm": None,
            "startMin": 7 * 60,   # 07:00
            "endMin": 18 * 60,    # 18:00
            "budget": 300000,
            "categories": ["Alam"]
        },
        "lang": "id"
    },
    {
        "name": "City Tour Kuliner & Budaya",
        "params": {
            "home": {"lat": -6.9145, "lng": 107.6020},
            "homeName": "Stasiun Bandung",
            "count": 4,
            "maxKm": 20,
            "startMin": 10 * 60,  # 10:00
            "endMin": 21 * 60,    # 21:00
            "budget": 500000,
            "categories": ["Kuliner", "Budaya"]
        },
        "lang": "id"
    },
    {
        "name": "Weekend Adventure (No Budget Limit)",
        "params": {
            "home": {"lat": -6.8126, "lng": 107.6178},
            "homeName": "Pasar Lembang",
            "count": 5,
            "maxKm": None,
            "startMin": 8 * 60,
            "endMin": 20 * 60,
            "budget": None,
            "categories": ["Alam", "Wisata", "Kuliner"]
        },
        "lang": "en"
    }
]

# Jalankan setiap test case
results = []
for tc in test_cases:
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {tc['name']}")
    print(f"{'='*70}")
    
    result = generate_full_itinerary(
        params=tc['params'],
        cbf_model=cbf_model,
        rl_data=rl_data,
        optimizer=optimizer,
        storyteller=storyteller,
        df=df,
        data_last_updated=data_last_updated,
        lang=tc['lang']
    )
    
    # Tampilkan hasil secara readable
    print(f"\n📍 Itinerary: {len(result['steps'])} destinasi")
    for step in result['steps']:
        arrive_h = step['arriveAt'] // 60
        arrive_m = step['arriveAt'] % 60
        depart_h = step['departAt'] // 60
        depart_m = step['departAt'] % 60
        print(f"  {step['idx']}. {step['dest']['name']:<30} | {arrive_h:02d}:{arrive_m:02d}–{depart_h:02d}:{depart_m:02d} | {step['travelKm']:.1f}km | Rp {step['dest']['ticket']:,}")
    
    print(f"\n💰 Total Biaya: Rp {result['totalCost']:,}")
    print(f"📏 Total Jarak: {result['totalKm']:.1f} km")
    print(f"⏱️ Total Waktu: {result['totalTime']//60} jam {result['totalTime']%60} menit")
    
    arrive_home_h = result['arriveHome'] // 60
    arrive_home_m = result['arriveHome'] % 60
    print(f"🏠 Tiba Kembali: {arrive_home_h:02d}:{arrive_home_m:02d} | {'⚠️ Overtime!' if result['overBudget'] else '✅ On time'}")
    
    print(f"\n📖 Narasi LLM:")
    print(f"  Vibe: {result['story']['vibe']}")
    print(f"  Intro: {result['story']['intro'][:200]}...")
    print(f"  Tips ({len(result['story']['tips'])} items):")
    for tip in result['story']['tips']:
        print(f"    • {tip}")
    print(f"  Closing: {result['story']['closing']}")
    
    results.append(result)

print(f"\n\n{'='*70}")
print("✅ SEMUA TEST CASE BERHASIL!")
```

---

### CELL 11: Validasi Kontrak Frontend (Code)

```python
def validate_frontend_contract(response: dict) -> bool:
    """
    Validasi bahwa response sesuai kontrak yang diharapkan frontend React.
    Ini memastikan tidak ada breaking changes antara backend dan frontend.
    """
    errors = []
    
    # Root level fields
    required_root = ['steps', 'totalCost', 'totalKm', 'totalTime', 
                      'returnKm', 'returnMin', 'arriveHome', 'overBudget', 
                      'spareMin', 'story', 'data_last_updated']
    for field in required_root:
        if field not in response:
            errors.append(f"Missing root field: {field}")
    
    # Tipe data
    if 'totalCost' in response and not isinstance(response['totalCost'], (int, float)):
        errors.append(f"totalCost must be numeric, got {type(response['totalCost'])}")
    if 'overBudget' in response and not isinstance(response['overBudget'], bool):
        errors.append(f"overBudget must be bool, got {type(response['overBudget'])}")
    
    # Steps validation
    if 'steps' in response:
        for i, step in enumerate(response['steps']):
            step_required = ['idx', 'dest', 'travelMin', 'travelKm', 'arriveAt', 'departAt']
            for field in step_required:
                if field not in step:
                    errors.append(f"steps[{i}] missing field: {field}")
            
            if 'dest' in step:
                dest_required = ['id', 'name', 'category', 'desc', 'ticket', 
                                  'duration', 'lat', 'lng', 'rating', 'tags']
                for field in dest_required:
                    if field not in step['dest']:
                        errors.append(f"steps[{i}].dest missing field: {field}")
                
                # Category harus valid
                valid_cats = {"Alam", "Kuliner", "Budaya", "Wisata", "Belanja"}
                if 'category' in step['dest'] and step['dest']['category'] not in valid_cats:
                    errors.append(f"steps[{i}].dest.category invalid: {step['dest']['category']}")
    
    # Story validation
    if 'story' in response:
        story_required = ['intro', 'highlights', 'tips', 'closing', 'vibe']
        for field in story_required:
            if field not in response['story']:
                errors.append(f"story missing field: {field}")
        
        if 'highlights' in response['story'] and 'steps' in response:
            if len(response['story']['highlights']) != len(response['steps']):
                errors.append(f"highlights count ({len(response['story']['highlights'])}) != steps count ({len(response['steps'])})")
    
    # Report
    if errors:
        print("❌ VALIDASI GAGAL:")
        for e in errors:
            print(f"  • {e}")
        return False
    else:
        print("✅ VALIDASI LULUS: Response sesuai kontrak frontend")
        return True

# Validasi semua hasil test
print("=== VALIDASI KONTRAK FRONTEND ===")
for i, (tc, result) in enumerate(zip(test_cases, results)):
    print(f"\nTest Case {i+1}: {tc['name']}")
    validate_frontend_contract(result)
```

---

### CELL 12: Bilingual Testing (Code)

```python
# Test output dalam Bahasa Inggris
print("=== BILINGUAL TEST ===")

en_params = {
    "home": {"lat": -6.9215, "lng": 107.6071},
    "homeName": "Alun-Alun Bandung",
    "count": 3,
    "maxKm": None,
    "startMin": 9 * 60,
    "endMin": 19 * 60,
    "budget": 400000,
    "categories": ["Alam", "Kuliner"]
}

en_result = generate_full_itinerary(
    params=en_params,
    cbf_model=cbf_model,
    rl_data=rl_data,
    optimizer=optimizer,
    storyteller=storyteller,
    df=df,
    data_last_updated=data_last_updated,
    lang="en"  # English
)

print("English Story:")
print(f"  Vibe: {en_result['story']['vibe']}")
print(f"  Intro: {en_result['story']['intro']}")
print(f"  Tip #1: {en_result['story']['tips'][0]}")
print(f"  Closing: {en_result['story']['closing']}")
```

---

### CELL 13: Prompt Engineering Analysis (Markdown + Code)

```markdown
## 🔬 Prompt Engineering Analysis

Dokumentasikan eksperimen prompt engineering:
1. Variasi temperature dan efeknya
2. Efek perubahan system prompt
3. Perbandingan model llama-3.1-8b-instant vs llama3-70b-8192
```

```python
# Eksperimen 1: Variasi Temperature
print("=== EKSPERIMEN: VARIASI TEMPERATURE ===\n")

temperatures = [0.3, 0.7, 1.0]
simple_params = {
    "home": {"lat": -6.9215, "lng": 107.6071},
    "homeName": "Alun-Alun Bandung",
    "count": 2,
    "maxKm": None,
    "startMin": 9 * 60,
    "endMin": 17 * 60,
    "budget": 200000,
    "categories": ["Alam"]
}

# Jalankan hanya jika API tersedia
if GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    # Ambil itinerary dulu (tanpa generate story)
    candidates = cbf_model.recommend(categories=["Alam"], top_n=10)
    selected = candidates.head(2).to_dict('records')
    ordered = optimizer.nearest_neighbor_route(simple_params['home'], selected)
    test_itin = optimizer.build_itinerary(
        home=simple_params['home'],
        home_name="Alun-Alun Bandung",
        ordered_destinations=ordered,
        start_min=simple_params['startMin'],
        end_min=simple_params['endMin']
    )
    
    for temp in temperatures:
        temp_storyteller = GroqStoryteller(api_key=GROQ_API_KEY, temperature=temp)
        story = temp_storyteller.generate_story(test_itin, simple_params)
        print(f"Temperature {temp}:")
        print(f"  Vibe: {story['vibe']}")
        print(f"  Intro (pertama 100 char): {story['intro'][:100]}...")
        print()
        time.sleep(2)  # Hindari rate limit
else:
    print("⚠️ API Key tidak tersedia, skip eksperimen temperature.")
    
# Dokumentasikan observasi dalam markdown cell berikut
```

```markdown
### Observasi Eksperimen Temperature:
| Temperature | Karakteristik Output |
|---|---|
| 0.3 | Lebih konsisten, kurang kreatif, cocok untuk produksi |
| 0.7 | Keseimbangan antara konsistensi dan kreativitas (default) |
| 1.0 | Paling kreatif, terkadang terlalu "absurd", cocok untuk variasi |

**Rekomendasi untuk produksi:** Temperature 0.7
```

---

### CELL 14: Export Sample Output (Code)

```python
import json

# Simpan satu sample output lengkap sebagai referensi untuk backend developer
sample_output = results[0]  # Test case pertama

with open("data/processed/sample_api_response.json", "w", encoding="utf-8") as f:
    json.dump(sample_output, f, ensure_ascii=False, indent=2)

print("✅ Sample API response disimpan ke data/processed/sample_api_response.json")
print(f"\nPreview (first 1000 chars):")
print(json.dumps(sample_output, ensure_ascii=False, indent=2)[:1000])
print("...")

# Juga simpan sample params
sample_params = test_cases[0]['params']
with open("data/processed/sample_api_request.json", "w", encoding="utf-8") as f:
    json.dump(sample_params, f, ensure_ascii=False, indent=2)
    
print("\n✅ Sample API request disimpan ke data/processed/sample_api_request.json")
```

---

### CELL 15: Penutup Notebook 2 (Markdown)

```markdown
## ✅ Notebook 2 Selesai — LLM Storyteller Integration

### Ringkasan:
| Komponen | Status | Keterangan |
|---|---|---|
| Model Loading | ✅ | CBF + RL + Encoders dari Notebook 1 |
| Groq API | ✅ | llama-3.1-8b-instant, JSON mode |
| Full Pipeline | ✅ | CBF → RL → Route → LLM |
| Frontend Contract | ✅ | Semua field tervalidasi |
| Bilingual | ✅ | id dan en |
| Fallback | ✅ | Template jika Groq gagal |
| Sample Output | ✅ | Tersimpan di data/processed/ |

### File Output:
| File | Keterangan |
|---|---|
| `data/processed/sample_api_request.json` | Contoh input dari frontend |
| `data/processed/sample_api_response.json` | Contoh output ke frontend |

### Langkah Selanjutnya:
1. **Backend FastAPI** — buat `backend/main.py` yang menggunakan model dan class dari notebook ini
2. **Deployment** — deploy ke VPS dengan domain .my.id
3. **Monitoring** — log request/response untuk fine-tuning future models

### Catatan Groq API:
- Free tier: 14.400 req/hari, 30 req/menit
- Tambahkan caching jika itinerary yang sama diminta berulang
- Pertimbangkan llama3-70b-8192 untuk kualitas lebih baik (lebih lambat)
```

---

---

# PETUNJUK TEKNIS UMUM (BERLAKU UNTUK KEDUA NOTEBOOK)

## Konvensi Kode

### 1. Setiap Function WAJIB memiliki:
- **Docstring** yang menjelaskan: tujuan, parameters, returns, dan contoh penggunaan
- **Type hints** untuk semua parameter dan return value
- **Error handling** yang informatif (try-except dengan pesan yang jelas)

### 2. Print statement yang informatif:
```python
# BAIK — informatif
print(f"✅ CBF model fitted. Similarity matrix: {self.similarity_matrix.shape}")
print(f"⚠️  Hanya {len(filtered)} destinasi lolos filter (dari {len(candidates)} kandidat)")
print(f"❌ OSRM timeout untuk rute {lat1},{lng1} → {lat2},{lng2}. Fallback ke Haversine.")

# BURUK — tidak informatif
print("done")
print("ok")
```

### 3. Magic numbers harus menjadi konstanta bernama:
```python
# BAIK
SPEED_KMH = 28           # kecepatan rata-rata kota Bandung
COST_PER_KM = 1200       # estimasi biaya BBM per km (Rp)
MIN_DESTINATIONS = 50    # minimum destinasi di dataset

# BURUK
travel_min = dist / 28 * 60
cost = km * 1200
```

### 4. Random seed untuk reproducibility:
```python
import random
import numpy as np
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"✅ Random seed set: {RANDOM_SEED}")
```

### 5. Progress indicator untuk operasi panjang:
```python
from tqdm import tqdm
# Gunakan tqdm untuk semua loop yang panjang (>100 iterasi)
```

## Handling Edge Cases (WAJIB ditangani)

| Edge Case | Cara Menangani |
|---|---|
| CBF returns 0 kandidat | Relax constraint: hapus budget filter, lalu max_km filter |
| RL agent state belum ada di Q-table | Fallback ke CBF score saja |
| OSRM API timeout/error | Gunakan Haversine + SPEED_KMH |
| Groq API rate limit (429) | Retry dengan exponential backoff (1s, 2s, 5s) |
| Groq response bukan valid JSON | Parse dengan fallback template |
| count > jumlah kandidat | Ambil semua kandidat yang tersedia |
| Budget = 0 atau sangat kecil | Tampilkan hanya destinasi gratis (ticket=0) |
| Semua destinasi di luar max_km | Relax max_km, ambil destinasi terdekat |
| arriveHome > endMin | Set overBudget=True (ini "overTime" sebenarnya), spareMin negatif |

## File Paths (Konsisten di Kedua Notebook)

```python
# SEMUA path relatif terhadap root project
# Bukan dari folder notebooks/

ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path(".")
# Atau jika di Jupyter:
import os
os.chdir("..")  # Dari notebooks/ ke root project

DATA_RAW = "data/raw/"
DATA_PROCESSED = "data/processed/"
MODELS_DIR = "models/"
```

## Output Formatting yang Konsisten

Gunakan format ini untuk semua menit/jam:
```python
def fmt_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def fmt_rp(amount: int) -> str:
    return f"Rp {amount:,}".replace(",", ".")
```

---

# CHECKLIST FINAL SEBELUM SUBMIT

## Notebook 1 (`01_recommendation_engine.ipynb`)
- [ ] Semua cell berjalan tanpa error dari atas ke bawah
- [ ] Dataset minimal 50 destinasi dengan semua 5 kategori
- [ ] 16 destinasi seed dari frontend `data.js` ada semua
- [ ] CBF model tersimpan di `models/cbf_model.pkl`
- [ ] RL agent tersimpan di `models/rl_agent.pkl`
- [ ] Scaler tersimpan di `models/scaler.pkl`
- [ ] Label encoders tersimpan di `models/label_encoders.pkl`
- [ ] Dataset tersimpan di `data/processed/destinations.csv`
- [ ] Feature matrix tersimpan di `data/processed/feature_matrix.npy`
- [ ] `data/last_updated.txt` berisi tanggal hari ini
- [ ] 3 visualisasi evaluasi tersimpan sebagai PNG
- [ ] Integration test dengan 3 test case berhasil
- [ ] Print summary akhir menampilkan semua file dengan ukuran

## Notebook 2 (`02_llm_storyteller.ipynb`)
- [ ] Validasi file dependensi berjalan dan semua ✅
- [ ] Model loading sukses (CBF, RL, encoders)
- [ ] Groq API test connection sukses
- [ ] 3 skenario full pipeline test berhasil
- [ ] Validasi kontrak frontend LULUS untuk semua test
- [ ] Bilingual test (id + en) berhasil
- [ ] Prompt engineering analysis terdokumentasi
- [ ] Sample output tersimpan di `data/processed/sample_api_response.json`
- [ ] Sample request tersimpan di `data/processed/sample_api_request.json`
- [ ] Fallback template berfungsi ketika Groq gagal

---

# STRUKTUR FOLDER FINAL

```
bandung-travel-ai/
├── data/
│   ├── raw/
│   │   ├── osm_raw.csv              # hasil crawling Overpass API
│   │   └── destinations_enriched.csv # sebelum cleaning
│   ├── processed/
│   │   ├── destinations.csv          # ← UTAMA: dataset bersih ≥50 dest
│   │   ├── feature_matrix.npy        # matriks fitur untuk CBF & RL
│   │   ├── cbf_similarity_heatmap.png
│   │   ├── cbf_category_scores.png
│   │   ├── rl_learning_curve.png
│   │   ├── sample_api_request.json   # contoh input dari frontend
│   │   └── sample_api_response.json  # contoh output ke frontend
│   └── last_updated.txt              # ← PENTING: tanggal update data
│
├── models/
│   ├── cbf_model.pkl                 # ← Content-Based Filtering
│   ├── rl_agent.pkl                  # ← Q-Learning RL Agent
│   ├── scaler.pkl                    # ← MinMaxScaler
│   └── label_encoders.pkl            # ← LabelEncoder + TF-IDF
│
├── notebooks/
│   ├── 01_recommendation_engine.ipynb  ← NOTEBOOK YANG DIBUAT
│   └── 02_llm_storyteller.ipynb        ← NOTEBOOK YANG DIBUAT
│
├── backend/                           # (dibuat terpisah)
│   ├── main.py
│   ├── recommender.py
│   └── ...
│
└── frontend/                          # (sudah ada)
    └── ...
```

---

*Dokumen ini adalah spesifikasi lengkap dan final. Tidak ada poin yang boleh dikurangi atau diskip. Setiap komponen, setiap edge case, setiap validasi, dan setiap format output adalah bagian integral dari sistem yang berfungsi end-to-end dengan frontend React yang sudah ada.*

**Versi:** 1.0.0  
**Dibuat untuk:** Capstone Project Bandung AI Travel Agent  
**Target:** AI Coding Model (Cursor, GitHub Copilot, Claude Code, dll)
