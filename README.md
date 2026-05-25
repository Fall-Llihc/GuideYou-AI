# Resume Proyek — Bandung AI Travel Agent
### Capstone Project · Program Studi Data Science · Telkom University · 2026

**Repository:** https://github.com/Fall-Llihc/Bandung_AI_Travel-Capstone-Project
**Live App:** https://bandung-travel.vercel.app
**API:** https://bandungaitravel-capstone-project-production.up.railway.app/api/health

---

## Gambaran Besar

Bandung AI Travel Agent adalah aplikasi web yang membantu pengguna merencanakan itinerary wisata di Bandung secara otomatis. Pengguna cukup memasukkan titik start, budget, jam perjalanan, dan kategori destinasi yang disukai — sistem AI akan menyusun rute optimal lengkap dengan narasi cerita perjalanan.

```
User input (lokasi, budget, waktu, kategori)
        ↓
  Backend FastAPI (Railway)
        ├── RecSys: CBF + RL → pilih destinasi terbaik
        └── TSP → susun rute optimal
        ↓
  LLM Storyteller → buat narasi perjalanan
        ↓
  Frontend React (Vercel) → tampilkan itinerary + cerita
```

---

## TAHAP 1 — Frontend (React)

### Tujuan
Membangun antarmuka pengguna yang memungkinkan user memasukkan preferensi dan melihat hasil itinerary dalam tampilan yang rapi dan interaktif.

### Teknologi
- React 18 dengan functional components dan hooks
- CSS Variables untuk theming (dark mode, warna aksen, tipografi)
- Fetch API untuk komunikasi dengan backend

### Alur Layar (Screen Flow)

```
WelcomeScreen → FormScreen → LoadingScreen → ResultsScreen
```

**WelcomeScreen** — Pengguna memilih titik keberangkatan. Tersedia 5 pilihan lokasi preset (Alun-Alun Bandung, Stasiun Bandung, Pasar Lembang, Dago, Gedung Sate) — sinkron dengan `HOME_OPTIONS` di notebook §2.6 sehingga distribusi reward RL match dengan production. Tombol deteksi GPS pakai Web Geolocation API. Error geolokasi (izin ditolak / GPS gagal / timeout) sekarang ditampilkan inline yang ramah, bukan `alert()`.

**FormScreen** — Pengguna mengisi parameter perjalanan:
- Jumlah destinasi (1–8)
- Budget total (opsional)
- Jarak maksimum dari home (opsional, hard-gate di backend)
- Jam mulai dan selesai perjalanan
- Kategori favorit: **Alam · Kuliner · Wisata** (3 kategori, sesuai notebook v3)

Validasi dilakukan inline sebelum kirim — pesan error tampil sebagai daftar di atas tombol submit, bukan satu-satu `alert()`.

**LoadingScreen** — Animasi 4 agen AI bekerja paralel dengan request backend. Jika request belum selesai dalam 12 detik, muncul hint cold-start ("server free-tier kadang butuh warm-up"). Saat error, screen ini menampilkan **error UI yang aksi-able**: ikon + judul + pesan + saran konkret + tombol *Coba lagi* / *Kembali ke form* yang dipilih sesuai jenis error (network / timeout / validation / rate-limit / server / unknown).

**ResultsScreen** — Menampilkan hasil lengkap:
- Summary cards: jumlah destinasi, total biaya, total jarak, total waktu
- Timeline itinerary: setiap destinasi dengan jam tiba, jam pergi, jarak tempuh, rating, harga tiket, dan link Google Maps
- Cerita perjalanan: satu paragraf narasi yang di-generate LLM
- Resume biaya: rincian tiket + estimasi BBM

### File Kunci
```
frontend/src/
├── App.jsx                    # State machine: welcome → form → loading → results
├── api/client.js              # POST /api/plan ke backend Railway
├── components/
│   ├── WelcomeScreen.jsx      # GPS + pilih titik start
│   ├── FormScreen.jsx         # Input parameter perjalanan
│   ├── LoadingScreen.jsx      # Animasi 4 agent cosmetic
│   └── ResultsScreen.jsx      # Tampilkan itinerary + cerita
└── index.css                  # CSS vars: --saffron, --bg, --ink, dll
```

### Kontrak Data Frontend ↔ Backend

**Request yang dikirim:**
```json
{
  "home": {"lat": -6.9215, "lng": 107.6071},
  "homeName": "Alun-Alun Bandung",
  "count": 4,
  "maxKm": null,
  "startMin": 540,
  "endMin": 1260,
  "budget": 500000,
  "categories": ["Alam", "Kuliner"]
}
```

**Response yang diterima:**
```json
{
  "steps": [{"idx":1, "dest":{...}, "travelMin":45, "arriveAt":585, "departAt":735}],
  "totalCost": 114500,
  "totalKm": 41.95,
  "arriveHome": 1012,
  "story": {"story": "Trip Bandung kamu...", "vibe": "Alam & Kuliner"}
}
```

Catatan: `arriveAt` dan `departAt` adalah menit dari tengah malam (contoh: 585 = 09:45), dikonversi di frontend dengan `Math.floor(minutes/60)` dan `minutes%60`.

---

## TAHAP 2 — Training Model LLM (Storyteller)

### Tujuan
Menghasilkan narasi perjalanan yang natural, personal, dan mengalir — bukan daftar bullet point — berdasarkan itinerary yang sudah disusun RecSys.

### Pendekatan
Proyek ini tidak melatih LLM dari nol. Sebaliknya menggunakan LLM pre-trained via API dengan teknik prompt engineering yang dirancang khusus untuk menghasilkan output yang konsisten.

### Model yang Digunakan
- Groq API dengan model `llama-3.1-8b-instant` (free tier: 14.400 request/hari)
- Alternatif: Gemini 1.5 Flash via Google AI Studio (free tier, perlu billing setup)

### Desain Prompt
System prompt dirancang dengan aturan ketat:

```
- POV orang KEDUA wajib: "kamu", "trip kamu" — dilarang "saya", "aku"
- Satu paragraf prosa mengalir — dilarang bullet, list, header
- Semua destinasi disebut natural dalam narasi
- Tips praktis disisipkan organik, bukan sebagai daftar
- Panjang 80–120 kata
- Output hanya JSON: {"story": "...", "vibe": "..."}
```

### Input ke LLM
Setiap step itinerary dikirim ke LLM dengan informasi: nama destinasi, kategori, rating, harga tiket, jam tiba, durasi kunjungan.

### Output LLM
```json
{
  "story": "Trip Bandung kamu dimulai pagi dari Alun-Alun — udara sejuk langsung menyambut sebelum kamu menanjak ke Kawah Putih yang mistis...",
  "vibe": "Alam & Kuliner"
}
```

### Mekanisme Keandalan
- Retry 3 kali dengan backoff: 1 detik → 3 detik → 8 detik
- Sanitasi post-processing: replace "saya/aku pergi" → "kamu pergi"
- Fallback template jika semua retry gagal, sehingga frontend tidak pernah crash

### File Kunci
```
backend/llm_storyteller.py     # Prompt engineering + API call + retry logic
notebooks/llm-train.ipynb      # Eksperimen dan evaluasi prompt
```

---

## TAHAP 3 — Training Model RecSys (Recommendation System)

### Tujuan
Memilih destinasi wisata terbaik yang sesuai preferensi pengguna, lalu menyusunnya dalam rute yang efisien dari segi jarak dan waktu.

### Pipeline Training (di Kaggle Notebook)

#### A. Crawling dan Pengumpulan Data

**Overpass API (OpenStreetMap):**
- Query berdasarkan tag: `tourism`, `amenity`, `leisure`, `natural`, `shop`
- Bounding box Bandung: `-7.25, 107.35, -6.75, 107.90`
- Fallback ke 3 mirror server jika server utama rate-limited
- Header wajib: `User-Agent: Bandung-AI-Travel-Capstone/1.0` untuk menghindari HTTP 406

**Seed data manual:**
- 50 destinasi ikonik Bandung dikurasi manual dengan data lengkap: nama, kategori, koordinat, harga tiket, durasi kunjungan, rating, tags, stay_detail (jadwal aktivitas detail di dalam destinasi)

**Hasil akhir:** `destinations.csv` — kolom:
```
id, name, category, desc, ticket, duration, lat, lng,
rating, tags, stay_detail, source, gmaps_url
```
Distribusi kategori (notebook v3, 3 kategori saja):
- **Alam** — gunung, curug, kebun teh, danau, tebing, taman alam
- **Kuliner** — restoran, kafe, warung, food court, kaki lima
- **Wisata** — theme park, water park, zoo, agro-edukasi, hot spring, mall, factory outlet, pasar tradisional

Kategori "Belanja" lama digabung ke "Wisata" (mall & FO secara natural adalah destinasi wisata di Bandung). Kategori "Budaya" dihapus penuh karena pada OSM banyak ter-tag dengan POI non-wisata (masjid, sekolah, kantor).

#### B. Feature Engineering
Setiap destinasi direpresentasikan sebagai vektor fitur:

| Kelompok Fitur | Dimensi | Bobot | Keterangan |
|---|---|---|---|
| One-hot kategori | 3 | ×2.0 | Alam, Kuliner, Wisata |
| Numerik ternormalisasi | 5 | ×1.0 | ticket_log, rating, duration, lat, lng |
| TF-IDF tags | s.d. 20 | ×0.5 | Keyword karakter destinasi: fotogenik, sunrise, dll |

Kategori di-bobot ×2.0 supaya similarity antar-kategori-berbeda lebih rendah — penting untuk variety guarantee. Normalisasi numerik pakai MinMaxScaler. Tags diparsing dari string Python list dengan `ast.literal_eval()`.

#### C. Content-Based Filtering (CBF)
- Cosine similarity dihitung antar semua pasangan destinasi menggunakan vektor 30 dimensi
- Hasil: similarity matrix berukuran 316×316
- Saat inference: rata-rata similarity dalam pool kategori yang dipilih user → ranking destinasi paling "representatif" dari preferensi user

#### D. Reinforcement Learning (Q-Learning)
Agent dilatih di simulated environment karena tidak ada data interaksi user nyata.

**State** agent direpresentasikan sebagai tuple 4 elemen:
```
(n_selected_bucket, budget_level, time_level, dominant_cat_idx)
- n_selected_bucket : min(8, jumlah destinasi sudah dipilih)
- budget_level      : 0–4 (0=habis, 4=≥75% sisa)
- time_level        : 0–4 berdasarkan menit tersisa
- dominant_cat_idx  : index kategori yang paling banyak dipilih
```

**Action:** memilih satu destinasi dari kandidat CBF

**Reward function (notebook v3 — 4 dimensi yang adil):**
```
reward = 0.30 × rating_score      ((rating-3)/2 clipped [0,1])
       + 0.25 × variety_bonus     (1.0 jika kategori belum dipilih)
       + 0.25 × distance_score    (1 - dist_home/max_km clipped [0,1])
       + 0.20 × budget_score      (sisa_budget / total_budget)
       - 2.0  × overmaxKm_penalty (defense in depth)
       - 1.0  × overtime_penalty
       - 0.5  × overbudget_penalty
```

`max_km` di-enforce di **4 lapis** secara HARD: filter kandidat CBF → `env.get_valid_actions` → `enforce_distance_constraint` (post-RL) → fallback fill juga cek max_km. Tidak pernah di-relax.

Training: 3.000 episode dengan epsilon-greedy (ε decay dari 1.0 → 0.05).

Saat inference: epsilon=0 (greedy), fallback ke nearest-neighbor jika state tidak ada di Q-table (Q-table sparse: 149 states dari 3.000 episode)

#### E. Route Optimization (Nearest-Neighbor TSP)
Setelah destinasi dipilih, rute disusun dengan heuristik nearest-neighbor:
- Mulai dari titik home user
- Setiap langkah: pilih destinasi terdekat yang belum dikunjungi
- Kecepatan rata-rata: 28 km/jam (estimasi urban Bandung)
- Formula waktu tempuh: `travel_min = (km / 28) × 60`

#### F. Model yang Disimpan
```
backend/models/
├── cbf_model.pkl       # similarity_matrix (N×N), feature_matrix (N×~28), df_index
├── rl_agent.pkl        # q_table, epsilon, training_history, epsilon_history
├── label_encoders.pkl  # MinMaxScaler, TfidfVectorizer, metadata kolom
└── scaler.pkl          # MinMaxScaler (numerik 5-dim)
```

Ukuran tergantung jumlah destinasi hasil crawling — biasanya cbf_model ~600KB-1MB, rl_agent ~100KB. Model dilatih dengan `scikit-learn==1.6.1` — versi harus sama persis di backend production.

### File Kunci
```
notebooks/rec-engine (4).ipynb # Crawling + feature engineering + CBF + RL + evaluasi (v3)
notebooks/_build_notebook.py   # Generator script untuk regenerate notebook
backend/recommender.py         # CBF + RL inference engine untuk serving
scripts/apply-kaggle-artifacts.sh  # Pasang artefak Kaggle ke backend/
```

---

## Workflow Update Model (Re-train di Kaggle)

Setelah notebook di-train ulang di Kaggle dan artefak hasil training di-download sebagai `bandung-travel-artifacts.zip`, gunakan script berikut untuk memasang ke backend:

```bash
# Dari root repo
bash scripts/apply-kaggle-artifacts.sh path/to/bandung-travel-artifacts.zip

# Atau auto-detect (zip diletakkan di repo root / notebooks/ / ~/Downloads/)
bash scripts/apply-kaggle-artifacts.sh
```

Script akan:
1. Verifikasi semua file kritis ada di zip (destinations.csv + 4 .pkl)
2. Backup folder `backend/data` & `backend/models` lama ke `.artifact-backup/<timestamp>/`
3. Salin `destinations.csv`, `last_updated.txt`, & `*.pkl` ke `backend/`
4. Salin `sample_request.json` & `sample_response.json` ke `docs/api/`
5. Sanity-check kategori (harus subset dari `{Alam, Kuliner, Wisata}`) dan blacklist (tidak ada nama yang lolos filter masjid/sekolah/showroom/dll)

Kalau sanity-check gagal, script keluar dengan exit code 1 — re-train notebook dengan blacklist terbaru sebelum redeploy.

**Branch flow yang direkomendasikan:**
```
1. Train notebook di Kaggle  → download bandung-travel-artifacts.zip
2. git checkout -b chore/refresh-model
3. bash scripts/apply-kaggle-artifacts.sh ~/Downloads/bandung-travel-artifacts.zip
4. git add backend/data/destinations.csv backend/data/last_updated.txt
   git add backend/models/*.pkl docs/api/*.json
5. git commit -m "chore: refresh model dari training Kaggle YYYY-MM-DD"
6. git push + buat PR ke main → merge → Railway auto-deploy
```

---

## TAHAP 4 — Integrasi dan Deployment

### Arsitektur Sistem

```
┌──────────────────────────────────────────────────┐
│  Frontend — Vercel (CDN global, free)             │
│  https://bandung-travel.vercel.app                │
└─────────────────────┬────────────────────────────┘
                      │ HTTPS POST /api/plan
                      ▼
┌──────────────────────────────────────────────────┐
│  Backend — Railway (Singapore, free trial)        │
│  FastAPI + Uvicorn, Python 3.11                  │
│                                                   │
│  recommender.py                                   │
│  ├── CBF: similarity matrix dari cbf_model.pkl   │
│  ├── RL: Q-table dari rl_agent.pkl               │
│  └── TSP: nearest-neighbor routing               │
│                                                   │
│  llm_storyteller.py                              │
│  └── Groq API → narasi cerita perjalanan         │
└──────────────────────────────────────────────────┘
```

### Tantangan Integrasi yang Ditemui dan Solusinya

**Tantangan 1 — Path model tidak ditemukan di Railway**
Railway dengan Root Directory `backend` hanya meng-copy folder `backend/` ke `/app/`. Folder `models/` yang ada di root repo tidak ikut ter-copy, sehingga saat startup muncul `FileNotFoundError`.

Solusi: pindahkan `models/` ke dalam `backend/models/`, lalu sederhanakan path:
```python
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR  = BACKEND_DIR / "models"
```

**Tantangan 2 — Key mismatch pkl training vs kode backend**
Training notebook menyimpan pkl dengan key `similarity_matrix` dan `df_index`, sementara `recommender.py` mengharapkan `sim_matrix` dan `id_to_sim_idx`.

Solusi: update `_extract_cbf()` agar support kedua format key sekaligus:
```python
sim = cbf_model.get("sim_matrix") or cbf_model.get("similarity_matrix")
idx_map = cbf_model.get("id_to_sim_idx") or {}
if not idx_map:
    df_index = cbf_model.get("df_index") or []
    idx_map = {item["id"]: i for i, item in enumerate(df_index)}
```

**Tantangan 3 — CORS error**
Frontend di Vercel tidak bisa memanggil backend Railway karena `ALLOWED_ORIGINS` masih berisi `http://localhost:3000` dari development.

Solusi: update env var `ALLOWED_ORIGINS` di Railway → isi URL Vercel yang exact dengan `https://`.

**Tantangan 4 — Story render masih format lama**
Frontend Capstone prototype menggunakan schema `{intro, highlights[], tips[], closing}` sementara backend menghasilkan `{story, vibe}`.

Solusi: update `ResultsScreen.jsx`, ganti loop render lama dengan satu `<p>{story.story}</p>`.

### Konfigurasi Environment Variables

**Railway (Backend):**
| Key | Value |
|---|---|
| `GROQ_API_KEY` | API key dari console.groq.com |
| `ALLOWED_ORIGINS` | URL Vercel frontend (exact, dengan https://) |
| `PYTHON_VERSION` | 3.11.9 |
| `GROQ_MODEL` | llama-3.1-8b-instant |

**Vercel (Frontend):**
| Key | Value |
|---|---|
| `REACT_APP_API_URL` | URL backend Railway |

### Status Deployment Saat Ini

```
✅ Backend Railway — online
   /api/health → model_loaded:true, cbf_loaded:true,
                 sim_matrix:[N,N] (N tergantung dataset hasil crawling)

✅ Frontend Vercel — online
   Aplikasi berjalan, itinerary dapat di-generate end-to-end

✅ Auto-deploy aktif
   Setiap git push ke main → Railway + Vercel redeploy otomatis
```

### Alur Lengkap Satu Request (End-to-End)

```
1. User buka https://bandung-travel.vercel.app

2. WelcomeScreen: pilih titik start (GPS atau manual)

3. FormScreen: isi jumlah destinasi, budget, jam, kategori

4. Klik "Generate Itinerary" → frontend kirim POST /api/plan ke Railway

5. Railway — recommender.py:
   a. Filter destinasi berdasarkan kategori yang dipilih user
   b. CBF scoring: hitung rata-rata cosine similarity dalam pool → ranking kandidat
   c. RL agent (greedy, ε=0): pilih destinasi satu per satu
      - encode state (budget sisa, waktu sisa, kategori dominan)
      - lookup Q-table → pilih action dengan Q tertinggi
      - fallback nearest-neighbor jika state tidak ada di Q-table
   d. Nearest-neighbor TSP: susun urutan rute paling efisien dari home
   e. Hitung jadwal: jam tiba = jam sebelumnya + travel_min, jam pergi = jam tiba + durasi

6. Railway — llm_storyteller.py:
   a. Kirim data semua step ke Groq API dengan system prompt ketat
   b. Parse response JSON → sanitasi POV
   c. Retry 3x jika gagal, fallback ke template jika semua retry habis

7. Railway return JSON lengkap ke frontend

8. ResultsScreen: render timeline + summary cards + cerita perjalanan

9. User klik nama destinasi → Google Maps terbuka di tab baru
```

---

## Ringkasan Tech Stack

| Komponen | Teknologi | Keterangan |
|---|---|---|
| Frontend | React 18, CSS Variables | Di-host di Vercel (free) |
| Backend API | FastAPI, Python 3.11, Uvicorn | Di-host di Railway (free trial) |
| CBF Model | scikit-learn 1.6.1, cosine similarity | Di-train di Kaggle |
| RL Model | Q-Learning custom Python | Di-train di Kaggle |
| Route Optimizer | Nearest-Neighbor TSP | Bagian dari backend, no library |
| LLM Storyteller | Groq API, llama-3.1-8b-instant | Free 14.400 req/hari |
| Dataset | Destinasi wisata Bandung (3 kategori) | OpenStreetMap + 24 seed manual |
| Version Control | Git + GitHub | Auto-deploy ke Railway + Vercel |

---

*Program Studi Data Science · Telkom University · Mei 2026*
