# 🗺️ Bandung AI Travel Agent

> **Smart itinerary planner untuk wisata Bandung Raya.**
> Input profil perjalanan → sistem rekomendasikan destinasi optimal dengan narasi cerita yang dihasilkan oleh LLM.

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![React](https://img.shields.io/badge/react-18.3-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![License](https://img.shields.io/badge/license-Academic-orange)

**Capstone Project · Program Studi Data Science · Telkom University**

| Kelompok 6 | NIM | Peran |
|---|---|---|
| Arkhan Falih Fahrie Puspita | 103052330051 | Backend — Recommendation System (CBF + RL) |
| Avatar Bintang Ramadhan | 103052300007 | Backend — LLM Storyteller |
| Azza Zukhrufa | 103052300014 | Frontend — React UI |
| Azzahra Sabryna Anggara | 103052300018 | Integrasi End-to-End |

</div>

---

## Daftar Isi

- [Demo](#demo)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Fitur Utama](#fitur-utama)
- [Dataset & Model](#dataset--model)
- [Struktur Direktori](#struktur-direktori)
- [Cara Menjalankan Lokal](#cara-menjalankan-lokal)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Evaluasi Model](#evaluasi-model)
- [Pembagian Tugas](#pembagian-tugas)

---

## Demo

| URL | Keterangan |
|---|---|
| **Frontend (Vercel)** | `https://bandung-travel.vercel.app` |
| **Backend Health** | `https://bandungaitravel-capstone-project-production.up.railway.app/api/health` |
| **API Docs (Swagger)** | `https://bandungaitravel-capstone-project-production.up.railway.app/docs` |

---

## Arsitektur Sistem

```
┌─────────────────────────────────┐         ┌────────────────────────────────────────┐
│         React Frontend          │         │         FastAPI Backend (Railway)      │
│           (Vercel)              │         │                                        │
│                                 │         │  Startup → pipeline.py load model:    │
│  WelcomeScreen (pilih lokasi)   │         │    ├── destinations.csv  (1.459 data)  │
│  FormScreen    (isi preferensi) │         │    ├── cbf_model.pkl     (1459×1459)   │
│  LoadingScreen (animasi proses) │  HTTP   │    ├── rl_agent.pkl      (Q-table)     │
│  ResultsScreen (tampil hasil)   │ ──────► │    └── scaler.pkl                      │
│                                 │  POST   │                                        │
│  ◄──── JSON {steps, story} ──── │ /api/plan├── recommender.py                      │
│                                 │         │    ├── Filter (kategori + Haversine)   │
│  Timeline rute putus-putus      │         │    ├── CBF Scoring (cosine similarity) │
│  Story card (narasi LLM)        │         │    ├── RL Q-table (greedy selection)   │
│  Google Maps link per destinasi │         │    └── TSP Route Optimizer (NN)        │
└─────────────────────────────────┘         │                                        │
                                            │── llm_storyteller.py                   │
                                            │    ├── Groq API (llama-3.1-8b-instant) │
                                            │    ├── Retry 3x (backoff 1–8 detik)    │
                                            │    └── Fallback template deterministik │
                                            └────────────────────────────────────────┘
```

### Alur Request `/api/plan`

```
User Input
    │
    ▼
[1] Filter kandidat berdasarkan kategori + jarak Haversine dari titik start
    │
    ▼
[2] CBF Scoring — cosine similarity dari feature matrix (3 one-hot + 5 numeric + 20 TF-IDF)
    │
    ▼
[3] RL Q-table — greedy selection destinasi satu per satu, reward = (rating + variety − budget_penalty)
    │
    ▼
[4] TSP Nearest-Neighbor — susun urutan rute paling efisien dari titik awal
    │
    ▼
[5] Hitung jadwal — jam tiba, durasi, jam berangkat per stop
    │
    ▼
[6] Groq LLM — generate narasi cerita perjalanan (POV orang kedua, 80–120 kata)
    │
    ▼
JSON Response → Frontend render itinerary
```

---

## Fitur Utama

- **Personalized Recommendation** — kombinasi Content-Based Filtering (CBF) dan Reinforcement Learning (Q-Learning) untuk rekomendasi destinasi yang relevan sesuai preferensi user
- **Route Optimization** — TSP Nearest-Neighbor heuristic untuk menyusun urutan kunjungan yang efisien
- **Budget & Time Aware** — hard-gate constraints untuk budget tiket dan durasi perjalanan
- **LLM Storyteller** — narasi cerita perjalanan otomatis dari Groq Llama-3.1, dengan fallback deterministik
- **1.459 Destinasi** — dataset Bandung Raya dari OpenStreetMap (Alam, Kuliner, Wisata)
- **Google Maps Integration** — setiap destinasi dilengkapi link Google Maps

---

## Dataset & Model

### Dataset

| Atribut | Detail |
|---|---|
| File | `backend/data/destinations.csv` |
| Jumlah destinasi | **1.459** |
| Distribusi kategori | Alam: 722 · Kuliner: 645 · Wisata: 92 |
| Rating range | 3.6 — 4.8 |
| Kolom | `id`, `name`, `category`, `desc`, `ticket`, `duration`, `lat`, `lng`, `rating`, `tags`, `stay_detail`, `gmaps_url`, `source` |
| Sumber | OpenStreetMap via Overpass API + enrichment manual kuliner ikonik |
| Last updated | 2026-05-25 |

Bounding box crawling: `-7.2500, 107.3500, -6.7500, 107.9000` (Bandung Raya)

### Model Artifacts

| File | Deskripsi |
|---|---|
| `backend/models/cbf_model.pkl` | Similarity matrix `(1459×1459)` + `df_index`, feature matrix `(1459, 28)` |
| `backend/models/rl_agent.pkl` | Q-table dict, dilatih 3.000 episode |
| `backend/models/scaler.pkl` | MinMaxScaler untuk fitur numerik |
| `backend/models/label_encoders.pkl` | Encoder kategori & TF-IDF tags |

**Komposisi feature CBF:**
- 3 one-hot kategori (bobot ×2.0)
- 5 fitur numerik scaled: ticket, duration, rating, lat, lng (bobot ×1.0)
- Hingga 20 TF-IDF dari kolom `tags` + `desc` (bobot ×0.5)

---

## Struktur Direktori

```
Bandung_AI_Travel-Capstone-Project/
│
├── backend/                          # FastAPI service → deploy ke Railway
│   ├── main.py                       # Entry point: /, /api/health, /api/plan
│   ├── recommender.py                # CBF + Q-Learning + TSP scheduler
│   ├── llm_storyteller.py            # Groq LLM wrapper (retry + fallback)
│   ├── data/
│   │   ├── destinations.csv          # Dataset runtime (1.459 destinasi)
│   │   └── last_updated.txt          # Tanggal update dataset
│   ├── models/                       # Pickle artifacts runtime
│   │   ├── cbf_model.pkl
│   │   ├── rl_agent.pkl
│   │   ├── scaler.pkl
│   │   └── label_encoders.pkl
│   ├── requirements.txt
│   ├── runtime.txt                   # Python 3.11
│   ├── railway.json                  # Konfigurasi deploy Railway
│   └── Procfile
│
├── frontend/                         # React app → deploy ke Vercel
│   ├── src/
│   │   ├── App.jsx                   # Root: screen flow welcome→form→loading→results
│   │   ├── components/
│   │   │   ├── WelcomeScreen.jsx     # Pilih titik keberangkatan
│   │   │   ├── FormScreen.jsx        # Input preferensi (kategori, budget, jam)
│   │   │   ├── LoadingScreen.jsx     # Animasi proses + hit API
│   │   │   └── ResultsScreen.jsx     # Timeline rute + story card + GMaps links
│   │   ├── api/
│   │   │   └── client.js             # Axios wrapper ke backend Railway
│   │   ├── data/
│   │   │   └── homeOptions.js        # Preset titik keberangkatan + koordinat
│   │   └── utils/
│   │       └── format.js             # Helper format jam, rupiah, km
│   ├── package.json
│   └── vercel.json                   # SPA rewrite rule
│
├── notebooks/
│   ├── rec-engine.ipynb              # Pipeline training CBF + RL (Kaggle)
│   └── llm-train.ipynb               # Eksperimen LLM storyteller
│
├── docs/
│   ├── api/
│   │   ├── sample_request.json
│   │   └── sample_response.json
│   └── screenshots/                  # Screenshot tampilan UI
│
├── models/                           # Mirror artifacts (untuk referensi)
├── scripts/
│   └── apply-kaggle-artifacts.sh    # Script update model dari Kaggle export
└── requirements.txt
```

---

## Cara Menjalankan Lokal

### Prasyarat

- Python 3.11+
- Node.js 18+
- API key Groq (gratis di [console.groq.com](https://console.groq.com))

### 1. Backend

```bash
cd backend

# Buat virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Konfigurasi environment
cp .env.example .env
# Edit .env: isi GROQ_API_KEY

# Jalankan server
uvicorn main:app --reload --port 8000
```

Server berjalan di `http://localhost:8000`. Cek health: `http://localhost:8000/api/health`

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Jalankan development server
npm start        # buka http://localhost:3000
```

Frontend secara default akan hit `http://localhost:8000`. Override via file `.env.local`:

```
REACT_APP_API_URL=http://localhost:8000
```

---

## API Reference

### `GET /api/health`

Cek status backend, jumlah destinasi, dan status model.

```json
{
  "status": "ok",
  "last_updated": "2026-05-25",
  "n_destinations": 1459,
  "model_loaded": true,
  "cbf_loaded": true,
  "sim_matrix_shape": [316, 316],
  "q_table_size": 1284,
  "groq_configured": true
}
```

### `POST /api/plan`

Generate itinerary + narasi perjalanan.

**Request body:**

```json
{
  "home": { "lat": -6.9218, "lng": 107.6070 },
  "homeName": "Alun-Alun Bandung",
  "count": 4,
  "startMin": 540,
  "endMin": 1260,
  "budget": 300000,
  "categories": ["Alam", "Kuliner"]
}
```

| Parameter | Tipe | Keterangan |
|---|---|---|
| `home` | object | Koordinat titik keberangkatan `{lat, lng}` |
| `homeName` | string | Nama lokasi awal (digunakan untuk narasi) |
| `count` | int (1–8) | Jumlah destinasi yang diinginkan, default 4 |
| `startMin` | int | Menit-of-day mulai, misal 540 = 09:00 |
| `endMin` | int | Menit-of-day selesai, misal 1260 = 21:00 |
| `budget` | int (opsional) | Total budget tiket (rupiah) |
| `maxKm` | float (opsional) | Jarak maksimal antar destinasi (km) |
| `categories` | array (opsional) | Subset dari `["Alam", "Kuliner", "Wisata"]`; kosong = semua |

**Response:**

```json
{
  "steps": [
    {
      "idx": 1,
      "dest": {
        "id": "tebing-keraton",
        "name": "Tebing Keraton",
        "category": "Alam",
        "rating": 4.6,
        "ticket": 15000,
        "duration": 120,
        "lat": -6.8425,
        "lng": 107.6328,
        "gmaps_url": "https://www.google.com/maps/..."
      },
      "travelMin": 25,
      "travelKm": 11.8,
      "arriveAt": 565,
      "departAt": 685
    }
  ],
  "story": {
    "story": "Trip Bandung kamu dimulai dari...",
    "vibe": "Alam & Kuliner"
  },
  "totalCost": 185000,
  "totalKm": 47.3,
  "data_last_updated": "2026-05-25"
}
```

Sample lengkap: [`docs/api/sample_response.json`](docs/api/sample_response.json)

---

## Deployment

Proyek ini di-deploy menggunakan dua platform gratis:

### Backend → Railway

| Setting | Nilai |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check | `/api/health` |

**Environment Variables yang wajib diset:**

```
GROQ_API_KEY     = gsk_xxxxxxxxxxxx
GROQ_MODEL       = llama-3.1-8b-instant
ALLOWED_ORIGINS  = https://bandung-travel.vercel.app
PYTHON_VERSION   = 3.11.9
```

### Frontend → Vercel

| Setting | Nilai |
|---|---|
| Root Directory | `frontend` |
| Framework | Create React App |
| Build Command | `npm run build` |

**Environment Variables:**

```
REACT_APP_API_URL = https://bandungaitravel-capstone-project-production.up.railway.app
```

### Auto-deploy

Setiap `git push` ke branch `main` secara otomatis men-trigger redeploy di Railway (backend) dan Vercel (frontend).

---

## Evaluasi Model

Hasil evaluasi dari 100 skenario simulasi (branch `updateVer`):

| Metrik | Nilai |
|---|---|
| Category coverage | **97.0%** |
| Distance compliance | **100.0%** |
| Budget compliance | **83.0%** |
| Average rating | **4.32 / 5.0** |
| Average total distance | 47.0 km |
| Average total cost | Rp 185.640 |
| Average variety index | **0.97** |
| Average stops per itinerary | 3.31 |

---

## Pembagian Tugas

### Arkhan Falih Fahrie Puspita — Backend Recommendation System

Bertanggung jawab atas seluruh pipeline rekomendasi di `backend/recommender.py`:
- Data collection via Overpass API (OpenStreetMap) dan preprocessing dataset 1.459 destinasi
- Feature engineering: one-hot encoding, MinMaxScaler, TF-IDF tags
- Content-Based Filtering (CBF) menggunakan cosine similarity matrix `(1459×1459)`
- Multi-Agent Reinforcement Learning (Q-Learning) dengan simulated environment (3.000 episode training)
- TSP Nearest-Neighbor heuristic untuk optimasi urutan rute
- Kategori-first reservation system untuk menjamin representasi tiap kategori
- Training notebook: `notebooks/rec-engine.ipynb`

### Avatar Bintang Ramadhan — Backend LLM Storyteller

Bertanggung jawab atas integrasi LLM di `backend/llm_storyteller.py`:
- Integrasi Groq API dengan model `llama-3.1-8b-instant`
- System prompt engineering untuk narasi POV orang kedua (80–120 kata)
- Mekanisme retry 3x dengan exponential backoff (1s → 3s → 8s)
- Fallback template deterministik saat Groq down/rate-limited
- JSON response format enforcement dan sanitasi POV
- Eksperimen LLM: `notebooks/llm-train.ipynb`

### Azza Zukhrufa — Frontend React

Bertanggung jawab atas seluruh tampilan di `frontend/src/`:
- Arsitektur screen flow linear: `WelcomeScreen → FormScreen → LoadingScreen → ResultsScreen`
- `WelcomeScreen`: deteksi lokasi (GPS/manual) dengan preset titik keberangkatan Bandung
- `FormScreen`: input preferensi kategori, budget, jam mulai/selesai, jumlah destinasi
- `LoadingScreen`: animasi proses dengan progress indicator
- `ResultsScreen`: timeline rute putus-putus, story card narasi, Google Maps hyperlink per destinasi
- Styling responsif via CSS custom properties

### Azzahra Sabryna Anggara — Integrasi End-to-End

Bertanggung jawab atas penyambungan semua komponen:
- Integrasi backend `main.py` (FastAPI) dengan Recommender + LLM Storyteller
- API client `frontend/src/api/client.js` dan schema alignment frontend–backend
- Konfigurasi CORS, environment variables, dan deployment Railway + Vercel
- Testing end-to-end: request flow, error handling, fallback behavior
- Pembaruan model artifacts dari training Kaggle ke production (`scripts/apply-kaggle-artifacts.sh`)

---

## Catatan Teknis

### Kategori-First Reservation

`recommender.py` melakukan reservasi slot per kategori secara proporsional **sebelum** Q-Learning fill. Ini mencegah DRL terlalu "rakus" pada kategori dengan reward tertinggi sehingga kategori pilihan user tetap terwakili di itinerary.

### CBF Kompatibilitas Dua Format

Loader pickle mendukung dua skema output notebook:
- Format baru: `{"similarity_matrix": ndarray, "df_index": [...]}`
- Format lama: `{"sim_matrix": ndarray, "id_to_sim_idx": {...}}`

Fallback otomatis tanpa perlu code change saat ganti versi model.

### Update Model ke Production

```bash
# Opsi A — dari branch updateVer (direkomendasikan)
git fetch origin updateVer
git cat-file -p updateVer:HASIL_TERBARU/working/models/cbf_model.pkl > backend/models/cbf_model.pkl
# (lihat README lengkap untuk semua langkah)

# Opsi B — dari Kaggle ZIP export
./scripts/apply-kaggle-artifacts.sh /path/to/bandung-travel-artifacts.zip
```

Setelah commit & push ke `main`: Railway otomatis redeploy, verifikasi via `/api/health`.

---

## Referensi

- Dataset: [OpenStreetMap](https://www.openstreetmap.org/) via Overpass API
- LLM: [Groq API](https://console.groq.com) — Llama-3.1-8b-instant
- Backend framework: [FastAPI](https://fastapi.tiangolo.com/)
- Frontend hosting: [Vercel](https://vercel.com)
- Backend hosting: [Railway](https://railway.app)
- ML: [scikit-learn](https://scikit-learn.org/) — cosine similarity, MinMaxScaler

---

<div align="center">

**Capstone Project Kelompok 6 · Program Studi Data Science · Telkom University · 2026**

</div>
