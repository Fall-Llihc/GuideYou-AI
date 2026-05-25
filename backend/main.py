"""FastAPI entry point for the Bandung AI Travel Agent backend.

Exposes:
    GET  /api/health          → liveness + dataset metadata
    POST /api/plan            → CBF + RL recommendation → NN-TSP route → LLM narrative

Run locally:
    uvicorn main:app --reload --port 8000


Run in production (Render):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommender import Recommender
from llm_storyteller import generate_story

# Kategori valid sesuai notebook v3. Frontend mengirim subset dari ini.
# Permissive: kalau frontend kirim kategori unknown, kita filter & log saja
# alih-alih reject — pengalaman user lebih baik.
VALID_CATEGORIES = {"Alam", "Kuliner", "Wisata"}

# ── Setup ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bandung-travel")

app = FastAPI(
    title="Bandung AI Travel Agent API",
    description="CBF + RL recommendation + NN-TSP + LLM narrative.",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS is a comma-separated list. Use "*" only for initial setup;
# tighten to the exact Vercel URL once known.
_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
log.info("CORS allowed origins: %s", allowed_origins)

# ── Load recommender at startup ──────────────────────────────────────────
# Loading takes ~1–2s; doing it at module import means the first request
# doesn't pay the cost (good UX on Render free tier after spin-up).
recommender = Recommender()


# ── Schemas ──────────────────────────────────────────────────────────────
class HomePoint(BaseModel):
    lat: float
    lng: float


class PlanRequest(BaseModel):
    home: HomePoint
    homeName: str = Field(..., examples=["Alun-Alun Bandung"])
    count: int = Field(4, ge=1, le=8, description="Jumlah destinasi yang diinginkan")
    maxKm: Optional[float] = Field(None, ge=1, description="Jarak maks antar destinasi (km), opsional")
    startMin: int = Field(540, ge=0, le=1440, description="Menit-of-day mulai (e.g. 540 = 09:00)")
    endMin: int = Field(1260, ge=0, le=1440, description="Menit-of-day selesai (e.g. 1260 = 21:00)")
    budget: Optional[int] = Field(None, ge=0, description="Total budget tiket (rupiah), opsional")
    categories: List[str] = Field(default_factory=list, description="Kategori favorit, kosong = semua")


# ── Routes ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "bandung-travel-api",
        "version": app.version,
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "last_updated": recommender.last_updated,
        "n_destinations": recommender.n_destinations,
        "model_loaded": recommender.model_loaded,
        "cbf_loaded": recommender.cbf_loaded,
        "sim_matrix_shape": (
            list(recommender.sim_matrix.shape) if recommender.sim_matrix is not None else None
        ),
        "q_table_size": len(recommender.q_table),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/api/plan")
def plan(req: PlanRequest):
    if req.endMin <= req.startMin:
        raise HTTPException(
            status_code=400,
            detail="Jam selesai harus lebih besar dari jam mulai.",
        )
    if (req.endMin - req.startMin) < 90:
        raise HTTPException(
            status_code=400,
            detail="Total durasi perjalanan minimal 90 menit (kurang dari itu sulit memuat 1 destinasi).",
        )

    # Permissive category filter: drop yang tidak dikenal, peringatkan via log.
    # Kalau setelah filter masih kosong, biarkan recommender pakai semua kategori.
    cleaned_categories = [c for c in (req.categories or []) if c in VALID_CATEGORIES]
    if req.categories and not cleaned_categories:
        log.warning(
            "Semua kategori user (%s) tidak dikenal — fallback ke semua kategori.",
            req.categories,
        )

    try:
        itinerary = recommender.build_itinerary(
            home=(req.home.lat, req.home.lng),
            count=req.count,
            max_km=req.maxKm,
            start_min=req.startMin,
            end_min=req.endMin,
            budget=req.budget,
            categories=cleaned_categories,
        )
    except FileNotFoundError as exc:
        # Artefak hilang — error operations, bukan logic. Bantu developer.
        log.exception("Artifact missing")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum siap di server (artifact pickle/CSV tidak ditemukan). "
                "Hubungi tim untuk re-deploy."
            ),
        ) from exc
    except ValueError as exc:
        # Data malformed (mis. column hilang) — beri pesan jelas.
        log.exception("Data validation failed")
        raise HTTPException(status_code=500, detail=f"Data internal bermasalah: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Recommender failed")
        raise HTTPException(
            status_code=500,
            detail="Recommender engine gagal memproses permintaan. Coba lagi dalam beberapa saat.",
        ) from exc

    # LLM narrative is best-effort: never fail the whole response if Groq
    # is down, missing API key, or rate-limited. The fallback narrator returns
    # a generic story so the frontend always has something to render.
    # Skip LLM if no destinations were found (empty itinerary) — narrative
    # would be misleading.
    if itinerary.get("steps"):
        story = generate_story(itinerary, home_name=req.homeName, categories=cleaned_categories)
    else:
        story = {
            "story": "",
            "vibe": "",
        }

    return {
        **itinerary,
        "story": story,
        "data_last_updated": recommender.last_updated,
    }
