import os, re, json, logging
import google.generativeai as genai

log = logging.getLogger(__name__)

# Konfigurasi Gemini. GROQ_API_KEY dipertahankan sebagai fallback supaya env
# var lama yang sudah di-set di Railway tetap dipakai sampai user mengganti
# nama key-nya.
_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

SYSTEM_PROMPT = """Kamu adalah travel blogger Indonesia bergaya caption Instagram.
WAJIB:
- POV orang KEDUA: "kamu", "trip kamu" — DILARANG "saya", "aku", "BandungBuddy"
- SATU paragraf prosa mengalir — TANPA bullet, list, atau header
- Sebut semua destinasi secara natural
- Panjang: 80-120 kata
- Return HANYA JSON valid: {"story": "...", "vibe": "..."}
- vibe: label singkat karakter trip (contoh: "Alam & Kuliner")"""

FALLBACK = {
    "story": "Trip Bandung kamu sudah siap! Nikmati setiap momen perjalanan.",
    "vibe": "Petualangan Bandung",
}


def _build_prompt(steps: list) -> str:
    lines = []
    for s in steps:
        d = s["dest"]
        arr = f"{s['arriveAt'] // 60:02d}:{s['arriveAt'] % 60:02d}"
        lines.append(
            f"- {d['name']} ({d['category']}, ⭐{d['rating']}, "
            f"Rp{d['ticket']:,}, tiba {arr}, stay {d['duration']}mnt)"
        )
    return "Buat narasi trip:\n" + "\n".join(lines)


def generate_story(itinerary: dict, home_name: str = "", categories: list = None) -> dict:
    if not _api_key:
        log.warning("GEMINI_API_KEY tidak di-set — pakai fallback story")
        return FALLBACK

    steps = itinerary.get("steps", [])
    if not steps:
        return FALLBACK

    prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(steps)}"

    for attempt, wait in enumerate([1, 3, 8]):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Bersihkan markdown fence kalau ada
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)

            # Sanitasi POV — sweep first-person mentions yang masih lolos
            data["story"] = re.sub(
                r"\b(saya|aku|BandungBuddy)\s+",
                "kamu ",
                data["story"],
                flags=re.IGNORECASE,
            )
            return data

        except Exception as e:  # noqa: BLE001
            log.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                import time

                time.sleep(wait)

    return FALLBACK
