"""Groq LLM narrative generator with a graceful local fallback.

The frontend always expects a `story` object with shape:
    { intro, highlights[], tips[], closing, vibe }

If GROQ_API_KEY is missing or the API call fails (rate limit, timeout, etc.)
we synthesize a passable narrative locally so the UX never breaks.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import requests

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TIMEOUT_S = 12

VIBE_PHRASES = {
    "Alam": "menghirup udara segar pegunungan",
    "Kuliner": "memanjakan lidah dengan kuliner khas",
    "Budaya": "menyelami warisan budaya Sunda",
    "Wisata": "menikmati hiburan dan rekreasi",
    "Belanja": "berburu oleh-oleh dan fashion lokal",
}


def _fmt_time(mins: int) -> str:
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _fallback_story(itinerary: Dict[str, Any], home_name: str, categories: List[str]) -> Dict[str, Any]:
    steps = itinerary.get("steps", [])
    if not steps:
        return {
            "intro": "Belum ada destinasi yang bisa dirangkum.",
            "highlights": [],
            "tips": ["Coba longgarkan filter kategori atau budget."],
            "closing": "—",
            "vibe": "Campuran",
        }

    cats = list({s["dest"]["category"] for s in steps})
    vibe_text = ", lalu ".join(VIBE_PHRASES.get(c, "menjelajah Bandung") for c in cats)
    total_km = itinerary.get("totalKm", 0)
    total_time = itinerary.get("totalTime", 0)
    spare_min = itinerary.get("spareMin", 0)

    first = steps[0]["dest"]
    last = steps[-1]["dest"]

    intro = (
        f"Hari ini perjalananmu dimulai dari **{home_name}** menuju petualangan yang dirancang khusus buat kamu. "
        f"Dengan {len(steps)} destinasi terpilih, kamu akan {vibe_text}. "
        f"Sebuah rute yang {'ringkas dan padat' if total_km < 50 else 'cukup berkelana'} — total "
        f"{total_km:.1f} km dalam waktu {total_time // 60} jam {total_time % 60} menit."
    )

    highlights = [
        f"**{s['dest']['name']}** *({s['dest']['category']})* — tiba sekitar pukul "
        f"{_fmt_time(s['arriveAt'])}, alokasikan {s['dest']['duration']} menit. "
        f"{s['dest']['desc']}."
        for s in steps
    ]

    tips: List[str] = []
    if first["category"] == "Alam":
        tips.append(f"Datang lebih pagi biar view {first['name']} masih clear dan nggak terlalu ramai.")
    if "Kuliner" in cats:
        tips.append("Sisakan ruang di perut — Bandung punya cara unik buat bikin kamu nagih.")
    if total_km > 80:
        tips.append(f"Total jarak {total_km:.1f} km — sewa mobil/motor atau pakai Grab biar hemat waktu.")
    if spare_min > 60:
        tips.append(f"Masih ada {spare_min // 60} jam ekstra — bisa mampir ke kafe di Braga sebelum pulang.")
    if last["category"] in ("Kuliner", "Belanja"):
        tips.append(f"Penutup di {last['name']} pas banget — santai sambil bawa oleh-oleh.")
    if not tips:
        tips.append("Bawa power bank dan air minum yang cukup — Bandung itu indah, tapi cuacanya unpredictable.")

    closing = (
        f"Udah, gausah mikir panjang — **save itinerary ini** dan langsung cus! "
        f"Bandung lagi nunggu kamu di ujung {last['name']}."
    )

    vibe = cats[0] if len(cats) == 1 else " & ".join(cats[:2])
    return {
        "intro": intro,
        "highlights": highlights,
        "tips": tips,
        "closing": closing,
        "vibe": vibe,
    }


def _build_prompt(itinerary: Dict[str, Any], home_name: str) -> str:
    steps = itinerary.get("steps", [])
    plan_lines = [
        f"{i+1}. {s['dest']['name']} ({s['dest']['category']}, tiba {_fmt_time(s['arriveAt'])}, "
        f"durasi {s['dest']['duration']}m, tiket Rp{s['dest']['ticket']:,})"
        for i, s in enumerate(steps)
    ]
    return (
        "Kamu adalah BandungBuddy, travel buddy lokal yang santai dan informatif. "
        f"Buatkan narasi perjalanan untuk hari ini, mulai dari {home_name}, "
        f"total {itinerary.get('totalKm', 0):.1f} km, {itinerary.get('totalTime', 0)} menit.\n\n"
        "Itinerary:\n" + "\n".join(plan_lines) + "\n\n"
        "Balas HANYA dengan JSON valid (tanpa code fence atau penjelasan), schema:\n"
        '{"intro": "<2-3 kalimat sambutan>",\n'
        ' "highlights": ["<1 kalimat per destinasi, urut sesuai itinerary>"],\n'
        ' "tips": ["<2-4 tips praktis>"],\n'
        ' "closing": "<1 kalimat penutup>",\n'
        ' "vibe": "<1-2 kata, contoh: Alam & Kuliner>"}\n'
        "Pakai bahasa Indonesia kasual, sapa pembaca dengan 'kamu'. Boleh **bold** dan *italic* markdown ringan."
    )


def generate_story(
    itinerary: Dict[str, Any],
    home_name: str,
    categories: List[str],
) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.info("GROQ_API_KEY not set — using local fallback story.")
        return _fallback_story(itinerary, home_name, categories)

    if not itinerary.get("steps"):
        return _fallback_story(itinerary, home_name, categories)

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful travel writer. Always reply with valid JSON only."},
                    {"role": "user", "content": _build_prompt(itinerary, home_name)},
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        story = json.loads(content)

        # Validate minimum schema; fall back if anything is missing
        for key in ("intro", "highlights", "tips", "closing", "vibe"):
            if key not in story:
                log.warning("Groq response missing '%s' — falling back.", key)
                return _fallback_story(itinerary, home_name, categories)
        return story

    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Groq call failed (%s) — using fallback.", exc)
        return _fallback_story(itinerary, home_name, categories)
