#!/usr/bin/env bash
#
# apply-kaggle-artifacts.sh
# ─────────────────────────
# Pasang artifacts hasil training Kaggle (bandung-travel-artifacts.zip)
# ke struktur folder backend/. Idempotent — aman dijalankan ulang.
#
# Konteks:
#   Notebook §17 (Export Artefak) menulis ke folder /kaggle/working/data/
#   dan /kaggle/working/models/. Skrip ini menerima zip yang isinya:
#
#       bandung-travel-artifacts.zip
#       ├── data/
#       │   ├── last_updated.txt
#       │   └── processed/
#       │       ├── destinations.csv
#       │       ├── eval_report.json
#       │       ├── eval_scenarios.csv
#       │       ├── feature_matrix.npy
#       │       ├── sample_api_request.json
#       │       └── sample_api_response.json
#       └── models/
#           ├── cbf_model.pkl
#           ├── label_encoders.pkl
#           ├── rl_agent.pkl
#           └── scaler.pkl
#
#   …dan menyalin file-file kritis ke:
#
#       backend/data/destinations.csv
#       backend/data/last_updated.txt
#       backend/models/{cbf_model,rl_agent,scaler,label_encoders}.pkl
#       docs/api/sample_request.json
#       docs/api/sample_response.json
#
# Penggunaan:
#       bash scripts/apply-kaggle-artifacts.sh path/to/bandung-travel-artifacts.zip
#       bash scripts/apply-kaggle-artifacts.sh                # auto-cari di repo root
#
# Exit codes:
#       0 = sukses, 1 = error usage / file tidak ada / verifikasi gagal

set -euo pipefail

# ── 0. Lokasi repo (skrip ini ada di scripts/) ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Resolusi path zip ──────────────────────────────────────
ZIP_PATH="${1:-}"
if [[ -z "$ZIP_PATH" ]]; then
  for cand in \
      "$REPO_ROOT/bandung-travel-artifacts.zip" \
      "$REPO_ROOT/notebooks/bandung-travel-artifacts.zip" \
      "$HOME/Downloads/bandung-travel-artifacts.zip"; do
    if [[ -f "$cand" ]]; then
      ZIP_PATH="$cand"; break
    fi
  done
fi

if [[ -z "$ZIP_PATH" || ! -f "$ZIP_PATH" ]]; then
  cat >&2 <<EOF
✗ File zip artefak tidak ditemukan.

Penggunaan:
    bash scripts/apply-kaggle-artifacts.sh path/to/bandung-travel-artifacts.zip

Atau letakkan zip di salah satu lokasi berikut:
    - <repo>/bandung-travel-artifacts.zip
    - <repo>/notebooks/bandung-travel-artifacts.zip
    - ~/Downloads/bandung-travel-artifacts.zip
EOF
  exit 1
fi

echo "📦 Memakai artifact: $ZIP_PATH"

# ── 2. Extract ke folder sementara ────────────────────────────
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$ZIP_PATH" -d "$TMP"
echo "📂 Diekstrak ke $TMP"

# ── 3. Verifikasi minimal file ada ────────────────────────────
REQUIRED_FILES=(
  "data/processed/destinations.csv"
  "models/cbf_model.pkl"
  "models/rl_agent.pkl"
  "models/scaler.pkl"
  "models/label_encoders.pkl"
)

missing=()
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$TMP/$f" ]]; then
    missing+=("$f")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "✗ File berikut tidak ada di zip:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "" >&2
  echo "Pastikan §17 di notebook (Export Artefak) sudah dijalankan sebelum zip dibuat." >&2
  exit 1
fi
echo "✅ Semua file kritis ditemukan."

# ── 4. Backup folder backend/data + backend/models lama ───────
BACKUP_DIR="$REPO_ROOT/.artifact-backup/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
[[ -d backend/data   ]] && cp -r backend/data   "$BACKUP_DIR/data"
[[ -d backend/models ]] && cp -r backend/models "$BACKUP_DIR/models"
echo "🗄  Backup ke $BACKUP_DIR"

# ── 5. Salin destinations.csv + last_updated.txt ──────────────
mkdir -p backend/data
cp "$TMP/data/processed/destinations.csv" backend/data/destinations.csv
if [[ -f "$TMP/data/last_updated.txt" ]]; then
  cp "$TMP/data/last_updated.txt" backend/data/last_updated.txt
else
  date +%Y-%m-%d > backend/data/last_updated.txt
fi
echo "✅ backend/data/destinations.csv ($(wc -l < backend/data/destinations.csv) baris)"
echo "✅ backend/data/last_updated.txt ($(cat backend/data/last_updated.txt))"

# ── 6. Salin pickle ──────────────────────────────────────────
mkdir -p backend/models
for pkl in cbf_model.pkl rl_agent.pkl scaler.pkl label_encoders.pkl; do
  cp "$TMP/models/$pkl" "backend/models/$pkl"
  size=$(du -h "backend/models/$pkl" | cut -f1)
  echo "✅ backend/models/$pkl ($size)"
done

# ── 7. Salin sample API request/response (untuk docs & test) ──
mkdir -p docs/api
if [[ -f "$TMP/data/processed/sample_api_request.json" ]]; then
  cp "$TMP/data/processed/sample_api_request.json" docs/api/sample_request.json
  echo "✅ docs/api/sample_request.json"
fi
if [[ -f "$TMP/data/processed/sample_api_response.json" ]]; then
  cp "$TMP/data/processed/sample_api_response.json" docs/api/sample_response.json
  echo "✅ docs/api/sample_response.json"
fi

# ── 8. Verifikasi cepat: kategori & blacklist sanity-check ────
echo ""
echo "🔍 Sanity-check destinations.csv:"
python3 - <<'PY'
import csv, os, sys
from collections import Counter

CSV = "backend/data/destinations.csv"
ALLOWED = {"Alam", "Kuliner", "Wisata"}
BLACKLIST_HINTS = [
    "masjid", "mosque", "sekolah", "smk", "sma", "smp",
    "showroom", "dealer", "bengkel", "cuci mobil", "cuci motor",
    "supermarket", "swalayan", "indomaret", "alfamart",
]

with open(CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

cats = Counter(r["category"] for r in rows)
print(f"  Total destinasi : {len(rows)}")
print(f"  Per kategori    : {dict(cats)}")

unknown = set(cats) - ALLOWED
if unknown:
    print(f"  ⚠ Kategori tidak dikenal: {unknown}")
    sys.exit(1)

flagged = []
for r in rows:
    name = r.get("name", "").lower()
    for kw in BLACKLIST_HINTS:
        if kw in name:
            flagged.append((r["name"], kw))
            break

if flagged:
    print(f"  ⚠ {len(flagged)} destinasi terlihat seperti seharusnya kena blacklist:")
    for n, kw in flagged[:5]:
        print(f"    - {n!r}  (matches '{kw}')")
    if len(flagged) > 5:
        print(f"    ... dan {len(flagged)-5} lagi")
    print("  (Re-train notebook dengan blacklist terbaru jika ini > 0.)")
else:
    print("  ✅ Tidak ada nama yang lolos blacklist hint.")
PY

echo ""
echo "✨ Selesai. Artefak terpasang di backend/. Jangan lupa:"
echo "   1. git add backend/data/destinations.csv backend/data/last_updated.txt"
echo "   2. git add backend/models/*.pkl docs/api/*.json"
echo "   3. git commit -m 'chore: update model artifacts from Kaggle training'"
echo ""
echo "Backup model lama tersimpan di: $BACKUP_DIR"
