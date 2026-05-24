"""Groq LLM narrative generator.

Generates a single-paragraph travel story per the schema {story, vibe}
that ResultsScreen.jsx renders directly. Falls back to a static
placeholder if GROQ_API_KEY is missing or the API call fails after the
retry budget — frontend always has something to render.
"""
import os, re, json, time, logging
import requests

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
TIMEOUT_S = 12

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
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning("GROQ_API_KEY tidak di-set — pakai fallback story")
        return FALLBACK

    steps = itinerary.get("steps", [])
    if not steps:
        return FALLBACK

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(steps)},
        ],
        "temperature": 0.7,
        # Groq's OpenAI-compatible API supports response_format on the
        # Llama-3.1 instruct models — guarantees valid JSON output.
        "response_format": {"type": "json_object"},
    }

    for attempt, wait in enumerate([1, 3, 8]):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Strip markdown fence on the off-chance the model emits one
            content = re.sub(r"```json|```", "", content).strip()
            data = json.loads(content)

            # Sanitasi POV — sweep first-person mentions yang masih lolos
            data["story"] = re.sub(
                r"\b(saya|aku|BandungBuddy)\s+",
                "kamu ",
                data["story"],
                flags=re.IGNORECASE,
            )
            return data

        except Exception as e:  # noqa: BLE001
            log.warning("Groq attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(wait)

    return FALLBACK
