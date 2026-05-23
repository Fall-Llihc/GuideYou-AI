# 🗺️ Bandung AI Travel Agent

> AI-powered itinerary planner untuk wisata Bandung — pilih kategori, budget, dan jam perjalanan, agen AI menyusun rute optimal lengkap dengan cerita perjalanan.

**Live Demo:** [https://bandung-travel.vercel.app](https://bandung-travel.vercel.app)
**API:** [https://bandung-travel-api.onrender.com](https://bandung-travel-api.onrender.com/api/health)

---

## 🧠 Cara Kerja

```
Input User (lokasi, budget, kategori, jam)
        ↓
  FastAPI Backend (Render)
        ├── Content-Based Filtering  ← cbf_model.pkl
        ├── RL Q-Learning Agent      ← rl_agent.pkl
        └── Nearest-Neighbor TSP
        ↓
  Groq LLM → Narasi perjalanan
        ↓
  Itinerary JSON → Frontend React (Vercel)
```

**Tech Stack:**

| Layer | Teknologi |
|---|---|
| Frontend | React 18, CSS Variables |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Model | scikit-learn 1.6.1 (CBF + RL Q-Learning) |
| LLM | Groq API — `llama-3.1-8b-instant` (free) |
| Hosting Frontend | Vercel (free) |
| Hosting Backend | Render (free, Singapore) |
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
│   ├── llm_storyteller.py      # Groq API integration
│   ├── requirements.txt
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
- Akun [Groq](https://console.groq.com) untuk API key (gratis)

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
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

## 🚀 Deployment (Render + Vercel — Gratis)

### Prasyarat Deployment
- Akun [GitHub](https://github.com) — repo sudah di-push
- Akun [Render](https://render.com) — daftar via GitHub
- Akun [Vercel](https://vercel.com) — daftar via GitHub
- `GROQ_API_KEY` dari [console.groq.com](https://console.groq.com)

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

### Step 2 — Deploy Backend ke Render

1. Login ke [render.com](https://render.com) → **New +** → **Web Service**
2. Pilih repository `bandung-travel-ai`
3. Isi konfigurasi:

   | Field | Nilai |
   |---|---|
   | **Name** | `bandung-travel-api` |
   | **Region** | Singapore |
   | **Branch** | `main` |
   | **Root Directory** | `backend` |
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | Free |

4. Klik **Advanced** → **Add Environment Variable**:

   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | `gsk_xxxxxxxxxxxx` |
   | `ALLOWED_ORIGINS` | `*` *(update setelah dapat URL Vercel)* |

5. Klik **Create Web Service** — build ~3–5 menit

6. Setelah selesai, catat URL Render:
   ```
   https://bandung-travel-api.onrender.com
   ```
   Test: `https://bandung-travel-api.onrender.com/api/health` → harus return `{"status":"ok"}`

---

### Step 3 — Siapkan Environment Frontend

Buat file `frontend/.env.production` (ganti URL dengan URL Render kamu):
```env
REACT_APP_API_URL=https://bandung-travel-api.onrender.com
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
   | `REACT_APP_API_URL` | `https://bandung-travel-api.onrender.com` |

5. Klik **Deploy** — build ~1–3 menit

6. Catat URL Vercel:
   ```
   https://bandung-travel.vercel.app
   ```

---

### Step 5 — Update CORS di Render

Setelah URL Vercel diketahui, perbarui `ALLOWED_ORIGINS` di Render agar lebih aman:

1. Render dashboard → service → tab **Environment**
2. Edit nilai `ALLOWED_ORIGINS`:
   ```
   https://bandung-travel.vercel.app
   ```
3. Klik **Save Changes** — Render otomatis redeploy

---

### Step 6 — Verifikasi End-to-End

Buka browser, buka DevTools (F12) → tab Console, lalu cek:

```
✅ https://bandung-travel-api.onrender.com/api/health  → {"status":"ok"}
✅ https://bandung-travel.vercel.app                   → halaman Welcome tampil
✅ Isi form → klik Generate → muncul hasil itinerary
✅ Console tab kosong (tidak ada error merah CORS)
✅ Klik nama destinasi → Google Maps terbuka di tab baru
✅ Badge "vibe" muncul di bagian cerita perjalanan
```

---

### Auto-Deploy

Setelah setup selesai, setiap push ke `main` → Render dan Vercel **otomatis redeploy** tanpa action manual:

```bash
git add .
git commit -m "deskripsi perubahan"
git push origin main
```

---

### (Opsional) Tambah Custom Domain .my.id

Jika ingin URL lebih profesional seperti `bandungtravel.my.id`:

**Di Vercel** → Settings → Domains → tambah `bandungtravel.my.id`

**Di Render** → Settings → Custom Domains → tambah `api.bandungtravel.my.id`

Tambah record di panel domain .my.id:

| Type | Name | Value |
|---|---|---|
| A | `@` | `76.76.21.21` |
| CNAME | `www` | `cname.vercel-dns.com` |
| CNAME | `api` | `bandung-travel-api.onrender.com` |

Update env vars:
- Vercel: `REACT_APP_API_URL` → `https://api.bandungtravel.my.id`
- Render: `ALLOWED_ORIGINS` → `https://bandungtravel.my.id,https://www.bandungtravel.my.id`

---

## 🔑 Environment Variables — Ringkasan

### Backend (`backend/.env` untuk lokal, Render untuk production)

| Variable | Contoh | Keterangan |
|---|---|---|
| `GROQ_API_KEY` | `gsk_abc123...` | Dari [console.groq.com](https://console.groq.com), wajib |
| `ALLOWED_ORIGINS` | `https://bandung-travel.vercel.app` | URL frontend, pisah koma jika lebih dari satu |

### Frontend (`frontend/.env.*`)

| Variable | Dev | Production |
|---|---|---|
| `REACT_APP_API_URL` | `http://localhost:8000` | `https://bandung-travel-api.onrender.com` |

> **Jangan commit `.env`** — sudah ada di `.gitignore`

---

## ⚠️ Hal yang Perlu Diperhatikan

**Render free tier spin-up** — backend "tidur" setelah 15 menit tidak ada request. Request pertama setelah idle butuh ~30–50 detik untuk bangun kembali. Ini normal, bukan error.

**scikit-learn versi exact** — `requirements.txt` harus `scikit-learn==1.6.1` (bukan `>=`). Model pkl dilatih dengan versi ini; versi berbeda bisa menyebabkan error saat load.

**CORS harus cocok exact** — nilai `ALLOWED_ORIGINS` di Render harus sama persis dengan URL Vercel, termasuk `https://` dan tanpa trailing slash.

---

## 🛠️ Troubleshooting

**Error CORS di browser console**
```
Access to fetch ... has been blocked by CORS policy
```
→ Cek `ALLOWED_ORIGINS` di Render, pastikan URL Vercel sudah benar dan tidak ada typo.

**Build Render gagal: `ERROR: Could not find a version that satisfies scikit-learn`**
→ Pastikan `requirements.txt` berisi `scikit-learn==1.6.1` (exact, bukan range).

**Model tidak ditemukan saat Render start**
```
FileNotFoundError: models/cbf_model.pkl
```
→ Cek `git ls-files models/` — jika kosong, berarti folder `models/` tidak ter-commit. Hapus `models/` dari `.gitignore` jika ada, lalu commit ulang.

**Frontend build gagal: `Module not found`**
→ Pastikan semua import di komponen React menggunakan path yang benar. Hapus semua referensi `window.buildItinerary`, `window.DESTINATIONS`, dan `window.generateNarrative`.

**Itinerary tidak muncul meski tidak ada error**
→ Buka Network tab di DevTools, cek response dari `/api/plan`. Jika status 500, cek Logs di Render dashboard.

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan akademik dan non-komersial.

---

*Proyek Capstone — Program Studi Data Science, Telkom University · 2026*