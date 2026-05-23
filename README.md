# 🗺️ Bandung AI Travel Agent

> AI-powered itinerary planner untuk wisata Bandung — pilih kategori, budget, dan jam perjalanan, agen AI menyusun rute optimal lengkap dengan cerita perjalanan.

**Live Demo:** [https://bandung-travel.vercel.app](https://bandung-travel.vercel.app)
**API:** [https://bandung-travel-api.up.railway.app](https://bandung-travel-api.up.railway.app/api/health)

---

## 🧠 Cara Kerja

```
Input User (lokasi, budget, kategori, jam)
        ↓
  FastAPI Backend (Railway)
        ├── Content-Based Filtering  ← cbf_model.pkl
        ├── RL Q-Learning Agent      ← rl_agent.pkl
        └── Nearest-Neighbor TSP
        ↓
  Gemini LLM → Narasi perjalanan
        ↓
  Itinerary JSON → Frontend React (Vercel)
```

**Tech Stack:**

| Layer | Teknologi |
|---|---|
| Frontend | React 18, CSS Variables |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Model | scikit-learn 1.6.1 (CBF + RL Q-Learning) |
| LLM | Google Gemini API — `gemini-1.5-flash` (free tier) |
| Hosting Frontend | Vercel (free) |
| Hosting Backend | Railway (free trial $5 credit/month) |
| Data | 316 destinasi wisata Bandung |

---

## 📁 Struktur Repository

```
bandung-travel-ai/
├── models/                     # Model ML (di-commit langsung, ~1 MB total)
│   ├── cbf_model.pkl           # Content-Based Filtering (872 KB)
│   ├── rl_agent.pkl            # RL Q-Learning agent (117 KB)
│   ├── label_encoders.pkl      # Scaler + TF-IDF + encoders
│   └── scaler.pkl
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── recommender.py          # CBF + RL inference engine
│   ├── llm_storyteller.py      # Gemini API integration
│   ├── requirements.txt
│   ├── railway.json            # Railway IaC (build/start/healthcheck)
│   ├── runtime.txt             # Pin Python 3.11.9 untuk Nixpacks
│   ├── Procfile                # Fallback start spec
│   └── data/
│       ├── destinations.csv    # 316 destinasi wisata Bandung
│       └── last_updated.txt
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   ├── api/
│   │   │   └── client.js       # HTTP client ke backend
│   │   └── components/
│   │       ├── WelcomeScreen.jsx
│   │       ├── FormScreen.jsx
│   │       ├── LoadingScreen.jsx
│   │       └── ResultsScreen.jsx
│   ├── package.json
│   └── .env.production
└── .gitignore
```

---

## ⚙️ Konfigurasi Lokal (Development)

### Prasyarat
- Python 3.11+
- Node.js 18+
- Akun [Google AI Studio](https://aistudio.google.com/app/apikey) untuk Gemini API key (gratis)

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/bandung-travel-ai.git
cd bandung-travel-ai
```

### 2. Setup Backend

```bash
cd backend

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Buat file `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ALLOWED_ORIGINS=http://localhost:3000
```

Jalankan backend:
```bash
uvicorn main:app --reload --port 8000
```

Verifikasi: buka [http://localhost:8000/api/health](http://localhost:8000/api/health)
```json
{"status": "ok", "last_updated": "2026-05-23", "n_destinations": 316}
```

### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install
```

Buat file `frontend/.env.development`:
```env
REACT_APP_API_URL=http://localhost:8000
```

Jalankan frontend:
```bash
npm start
```

Buka [http://localhost:3000](http://localhost:3000)

---

## 🚀 Deployment (Railway + Vercel — Gratis)

### Prasyarat Deployment
- Akun [GitHub](https://github.com) — repo sudah di-push
- Akun [Railway](https://railway.app) — daftar via GitHub
- Akun [Vercel](https://vercel.com) — daftar via GitHub
- `GEMINI_API_KEY` dari [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

### Step 1 — Pastikan Model Ikut di Repository

Model pkl (~1 MB total) harus ikut di-commit. Verifikasi:
```bash
git ls-files models/
# Harus tampil 4 file: cbf_model.pkl, rl_agent.pkl, label_encoders.pkl, scaler.pkl
```

Jika belum ada:
```bash
git add models/
git commit -m "Add trained model files"
git push
```

---

### Step 2 — Deploy Backend ke Railway

#### 2.1 Daftar Railway

Buka [railway.app](https://railway.app) → klik **"Start a New Project"** → **"Login with GitHub"** → authorize.

#### 2.2 New Project

Dashboard Railway → klik **"New Project"** → pilih **"Deploy from GitHub repo"** → cari dan pilih `Bandung_AI_Travel-Capstone-Project`.

#### 2.3 Konfigurasi Service

Railway auto-detect Python (via Nixpacks). Klik service yang baru dibuat → tab **"Settings"**:

| Field | Nilai |
|---|---|
| **Root Directory** | `backend` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

> Jika `backend/railway.json` sudah ter-commit (sudah ada di repo ini), Railway akan membaca konfigurasinya otomatis — kamu tidak perlu mengisi build/start command manual.

#### 2.4 Set Environment Variables

Tab **"Variables"** → tambah satu per satu:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | `xxxxxxxxxxxx` *(dari [aistudio.google.com](https://aistudio.google.com/app/apikey))* |
| `ALLOWED_ORIGINS` | `*` *(update setelah dapat URL Vercel)* |
| `PYTHON_VERSION` | `3.11.9` |
| `GEMINI_MODEL` | `gemini-1.5-flash` |
| `PORT` | `8000` |

#### 2.5 Generate Domain

Tab **"Settings"** → scroll ke **"Networking"** → klik **"Generate Domain"**.

Railway beri URL seperti:
```
https://bandung-travel-api.up.railway.app
```

#### 2.6 Deploy

Klik **"Deploy"** → pantau log → tunggu ~3–5 menit.

Test:
```
https://bandung-travel-api.up.railway.app/api/health
```
Harus return:
```json
{"status":"ok","n_destinations":316,"cbf_loaded":true,"sim_matrix_shape":[316,316]}
```

---

### Step 3 — Siapkan Environment Frontend

Buat file `frontend/.env.production` (ganti URL dengan URL Railway kamu):
```env
REACT_APP_API_URL=https://bandung-travel-api.up.railway.app
```

Commit dan push:
```bash
git add frontend/.env.production
git commit -m "Add production API URL"
git push
```

---

### Step 4 — Deploy Frontend ke Vercel

1. Login ke [vercel.com](https://vercel.com) → **Add New...** → **Project**
2. Import repository `bandung-travel-ai`
3. Isi konfigurasi:

   | Field | Nilai |
   |---|---|
   | **Project Name** | `bandung-travel` |
   | **Root Directory** | `frontend` ← klik Edit untuk mengubah |
   | **Framework Preset** | Create React App *(auto-detect)* |
   | **Build Command** | `npm run build` *(auto)* |
   | **Output Directory** | `build` *(auto)* |

4. Expand **Environment Variables** → tambah:

   | Name | Value |
   |---|---|
   | `REACT_APP_API_URL` | `https://bandung-travel-api.up.railway.app` |

5. Klik **Deploy** — build ~1–3 menit

6. Catat URL Vercel:
   ```
   https://bandung-travel.vercel.app
   ```

---

### Step 5 — Update CORS di Railway

Setelah URL Vercel diketahui, perbarui `ALLOWED_ORIGINS` di Railway agar lebih aman:

1. Railway dashboard → service → tab **Variables**
2. Edit nilai `ALLOWED_ORIGINS`:
   ```
   https://bandung-travel.vercel.app
   ```
3. Railway otomatis redeploy setelah variable berubah.

> Jika frontend di-redeploy ke URL berbeda (preview deployment, custom domain), tambahkan ke list dengan koma sebagai pemisah, misal:
> `https://bandung-travel.vercel.app,https://bandungtravel.my.id`

---

### Step 6 — Verifikasi End-to-End

Buka browser, buka DevTools (F12) → tab Console, lalu cek:

```
✅ https://bandung-travel-api.up.railway.app/api/health  → {"status":"ok"}
✅ https://bandung-travel.vercel.app                     → halaman Welcome tampil
✅ Isi form → klik Generate → muncul hasil itinerary
✅ Console tab kosong (tidak ada error merah CORS)
✅ Klik nama destinasi → Google Maps terbuka di tab baru
✅ Badge "vibe" muncul di bagian cerita perjalanan
```

---

### Auto-Deploy

Setelah setup selesai, setiap push ke `main` → Railway dan Vercel **otomatis redeploy** tanpa action manual:

```bash
git add .
git commit -m "deskripsi perubahan"
git push origin main
```

---

### (Opsional) Tambah Custom Domain .my.id

Jika ingin URL lebih profesional seperti `bandungtravel.my.id`:

**Di Vercel** → Settings → Domains → tambah `bandungtravel.my.id`

**Di Railway** → service → Settings → Networking → **Custom Domain** → tambah `api.bandungtravel.my.id` → Railway tampilkan target CNAME (mis. `xyz.up.railway.app`).

Tambah record di panel domain .my.id:

| Type | Name | Value |
|---|---|---|
| A | `@` | `76.76.21.21` |
| CNAME | `www` | `cname.vercel-dns.com` |
| CNAME | `api` | `<target-cname-dari-railway>` |

Update env vars:
- Vercel: `REACT_APP_API_URL` → `https://api.bandungtravel.my.id`
- Railway: `ALLOWED_ORIGINS` → `https://bandungtravel.my.id,https://www.bandungtravel.my.id`

---

## 🔑 Environment Variables — Ringkasan

### Backend (`backend/.env` untuk lokal, Railway untuk production)

| Variable | Contoh | Keterangan |
|---|---|---|
| `GEMINI_API_KEY` | `AIzaSy...` | Dari [aistudio.google.com](https://aistudio.google.com/app/apikey), wajib |
| `ALLOWED_ORIGINS` | `https://bandung-travel.vercel.app` | URL frontend, pisah koma jika lebih dari satu |
| `PYTHON_VERSION` | `3.11.9` | Pin runtime Python di Railway |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Override model Gemini (opsional) |
| `PORT` | `8000` | Diinject Railway otomatis; aman di-set manual |

> **Migrasi dari Groq:** kalau env var lama `GROQ_API_KEY` masih ter-set di Railway, app akan tetap pakai itu sebagai fallback sampai kamu rename jadi `GEMINI_API_KEY`. Tidak ada downtime saat ganti — set `GEMINI_API_KEY` lebih dulu, lalu hapus `GROQ_API_KEY` & `GROQ_MODEL`.

### Frontend (`frontend/.env.*`)

| Variable | Dev | Production |
|---|---|---|
| `REACT_APP_API_URL` | `http://localhost:8000` | `https://bandung-travel-api.up.railway.app` |

> **Jangan commit `.env`** — sudah ada di `.gitignore`

---

## ⚠️ Hal yang Perlu Diperhatikan

**Railway free trial vs Hobby plan** — Railway memberi $5 credit/bulan di trial. Service kecil seperti ini biasanya cukup, tapi pantau usage di dashboard. Untuk production jangka panjang, upgrade ke Hobby plan ($5/bulan flat) supaya tidak deprovision.

**scikit-learn versi exact** — `requirements.txt` harus `scikit-learn==1.6.1` (bukan `>=`). Model pkl dilatih dengan versi ini; versi berbeda bisa menyebabkan error saat load.

**CORS harus cocok exact** — nilai `ALLOWED_ORIGINS` di Railway harus sama persis dengan URL Vercel, termasuk `https://` dan tanpa trailing slash.

**Health endpoint** — `/api/health` mengembalikan `cbf_loaded` dan `sim_matrix_shape` untuk memastikan model benar-benar ter-load. Jika `cbf_loaded: false`, cek log Railway — kemungkinan sklearn version mismatch.

---

## 🛠️ Troubleshooting

**Error CORS di browser console**
```
Access to fetch ... has been blocked by CORS policy
```
→ Cek `ALLOWED_ORIGINS` di Railway, pastikan URL Vercel sudah benar dan tidak ada typo.

**Build Railway gagal: `ERROR: Could not find a version that satisfies scikit-learn`**
→ Pastikan `requirements.txt` berisi `scikit-learn==1.6.1` (exact, bukan range), dan `runtime.txt` berisi `python-3.11.9`.

**Model tidak ditemukan saat Railway start**
```
FileNotFoundError: models/cbf_model.pkl
```
→ Cek `git ls-files models/` — jika kosong, berarti folder `models/` tidak ter-commit. Hapus `models/*.pkl` dari `.gitignore` jika ada, lalu commit ulang.

**`cbf_loaded: false` di /api/health**
→ Model pkl ke-load tapi schema tidak sesuai. Cek log Railway untuk melihat keys apa yang ada di pkl, lalu sinkronkan dengan training notebook.

**Frontend build gagal: `Module not found`**
→ Pastikan semua import di komponen React menggunakan path yang benar. Hapus semua referensi `window.buildItinerary`, `window.DESTINATIONS`, dan `window.generateNarrative`.

**Itinerary tidak muncul meski tidak ada error**
→ Buka Network tab di DevTools, cek response dari `/api/plan`. Jika status 500, cek tab **"Deployments" → Logs** di Railway dashboard.

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan akademik dan non-komersial.

---

*Proyek Capstone — Program Studi Data Science, Telkom University · 2026*
