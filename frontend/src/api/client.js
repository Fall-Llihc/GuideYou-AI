// HTTP client untuk Bandung AI Travel Agent backend.
//
// Tujuan utama modul ini selain memanggil API:
//   1. Mengkategorikan SETIAP kemungkinan kegagalan ke "kind" yang stabil
//      sehingga UI bisa mengambil keputusan rendering tanpa parsing string.
//   2. Mengubah error mentah (CORS, ECONNREFUSED, 422 pydantic, 502 dari
//      reverse-proxy, dll.) menjadi pesan Indonesia yang ramah & aksi-able.
//
// REACT_APP_API_URL di-bake saat build oleh Create React App:
//   - .env.development → http://localhost:8000
//   - .env.production  → https://bandung-travel-api.up.railway.app
//
const RAW_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
export const API_BASE = RAW_BASE.replace(/\/+$/, "");

// Free-tier hosts yang biasanya cold-start setelah idle. Dipakai untuk
// memberi hint "tunggu 30 detik" saat network failure terjadi pertama kali.
const COLD_START_HOSTS = ["railway.app", "up.railway.app", "onrender.com", "render.com"];
const isColdStartHost = COLD_START_HOSTS.some((h) => API_BASE.includes(h));

// Default timeout per request (ms). Sengaja generous: cold-start bisa 30-50s.
// Dapat di-override per-call lewat options.timeoutMs.
const DEFAULT_TIMEOUT_MS = 60_000;

// ── Error class ──────────────────────────────────────────────
// `kind` adalah enum stabil yang dipakai LoadingScreen untuk memilih ikon,
// tone, dan tombol aksi. Nilai-nilainya:
//   "network"       → fetch throw (CORS, DNS, offline, server mati)
//   "cold-start"    → kemungkinan besar Railway/Render lagi spin-up
//   "timeout"       → request lebih lama dari timeoutMs
//   "validation"    → 400/422 input user salah
//   "rate-limit"    → 429
//   "server"        → 5xx
//   "unknown"       → fallback
export class ApiError extends Error {
  constructor(message, { kind = "unknown", status = 0, body = null, hint = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.body = body;
    this.hint = hint;
  }
}

// ── Helper: format pydantic 422 detail jadi pesan manusiawi ──
// FastAPI mengembalikan detail seperti:
//   [{"loc":["body","count"],"msg":"Input should be greater than 0","type":"greater_than"}]
function formatPydanticDetail(detail) {
  if (!Array.isArray(detail)) return null;
  const lines = detail
    .filter((d) => d && d.msg)
    .map((d) => {
      const loc = Array.isArray(d.loc) ? d.loc.filter((x) => x !== "body").join(".") : "input";
      return `• ${loc}: ${d.msg}`;
    });
  return lines.length ? lines.join("\n") : null;
}

// ── Helper: pilih copy berdasar status code ──────────────────
function classifyHttpError(status, body) {
  // Coba ambil string "detail" dari body FastAPI standard
  const rawDetail = body && (body.detail ?? body.message ?? body.error);

  if (status === 400) {
    return {
      kind: "validation",
      message:
        typeof rawDetail === "string"
          ? `Input tidak valid: ${rawDetail}`
          : "Ada parameter yang tidak valid. Silakan cek formulir lalu coba lagi.",
      hint: "Cek lagi jam, jumlah destinasi, dan budget — pastikan masuk akal.",
    };
  }

  if (status === 422) {
    const formatted = formatPydanticDetail(rawDetail);
    return {
      kind: "validation",
      message: formatted
        ? `Input belum lolos validasi:\n${formatted}`
        : "Beberapa field di formulir tidak sesuai aturan. Silakan periksa kembali.",
      hint: "Coba turunkan jumlah destinasi atau perpanjang rentang waktu.",
    };
  }

  if (status === 429) {
    return {
      kind: "rate-limit",
      message: "Terlalu banyak request dalam waktu singkat. Tunggu sebentar lalu coba lagi.",
      hint: "Tunggu ~1 menit sebelum mencoba kembali.",
    };
  }

  if (status === 408 || status === 504) {
    return {
      kind: "timeout",
      message: "Server butuh waktu lebih lama dari biasanya untuk memproses permintaan.",
      hint: "Coba ulangi — biasanya request kedua lebih cepat.",
    };
  }

  if (status >= 500) {
    return {
      kind: "server",
      message:
        typeof rawDetail === "string" && rawDetail
          ? `Server mengalami kendala: ${rawDetail}`
          : "Server sedang bermasalah. Tim sudah dapat notifikasi.",
      hint: "Coba lagi dalam 1-2 menit. Jika tetap gagal, restart browser dan coba ulang.",
    };
  }

  // 401/403 — saat ini API kita publik, tapi siapkan saja.
  if (status === 401 || status === 403) {
    return {
      kind: "validation",
      message: "Akses ditolak oleh server.",
      hint: "Hubungi developer jika kamu yakin ini bug.",
    };
  }

  // 404 — biasanya path salah, ke developer
  if (status === 404) {
    return {
      kind: "server",
      message: "Endpoint API tidak ditemukan. Mungkin versi backend belum di-deploy.",
      hint: "Pastikan REACT_APP_API_URL menunjuk ke backend yang benar.",
    };
  }

  return {
    kind: "unknown",
    message: `HTTP ${status}: ${typeof rawDetail === "string" ? rawDetail : "Unknown error"}`,
    hint: "Coba lagi atau hubungi developer.",
  };
}

// ── Core request ─────────────────────────────────────────────
async function request(path, { timeoutMs = DEFAULT_TIMEOUT_MS, ...options } = {}) {
  const url = `${API_BASE}${path}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  let resp;
  try {
    resp = await fetch(url, {
      ...options,
      signal: ctrl.signal,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
  } catch (err) {
    clearTimeout(timer);

    // AbortController fired → ini timeout, bukan network biasa
    if (err && err.name === "AbortError") {
      throw new ApiError(
        `Request melebihi batas waktu ${Math.round(timeoutMs / 1000)} detik.`,
        {
          kind: "timeout",
          hint: isColdStartHost
            ? "Server free-tier mungkin sedang bangun dari idle — tunggu 30 detik lalu coba lagi."
            : "Cek koneksi internet kamu, lalu coba ulang.",
        }
      );
    }

    // Plain network error (CORS denied, DNS, offline, server mati)
    throw new ApiError(
      isColdStartHost
        ? "Server backend belum bisa dijangkau. Kemungkinan sedang bangun dari idle."
        : "Tidak bisa terhubung ke server.",
      {
        kind: isColdStartHost ? "cold-start" : "network",
        body: String(err),
        hint: isColdStartHost
          ? "Backend free-tier butuh ~30 detik untuk warm-up. Tunggu sebentar lalu coba lagi."
          : `Pastikan backend berjalan di ${API_BASE}. Cek juga koneksi internet & CORS.`,
      }
    );
  }
  clearTimeout(timer);

  let body = null;
  try {
    body = await resp.json();
  } catch {
    /* response tidak punya JSON body — biarkan null */
  }

  if (!resp.ok) {
    const { kind, message, hint } = classifyHttpError(resp.status, body);
    throw new ApiError(message, { kind, status: resp.status, body, hint });
  }
  return body;
}

// ── Public API ───────────────────────────────────────────────
export const api = {
  /** GET /api/health — quick liveness; pakai timeout pendek */
  health: () => request("/api/health", { timeoutMs: 8_000 }),

  /** POST /api/plan — generate itinerary (timeout panjang krn cold-start) */
  plan: (params, { timeoutMs } = {}) =>
    request("/api/plan", {
      method: "POST",
      body: JSON.stringify(params),
      timeoutMs,
    }),
};

// Helper export: dipakai komponen untuk preview teks error sebelum throw
export function describeError(err) {
  if (err instanceof ApiError) {
    return { kind: err.kind, message: err.message, hint: err.hint, status: err.status };
  }
  return {
    kind: "unknown",
    message: String(err && err.message ? err.message : err),
    hint: "Coba refresh halaman lalu ulangi.",
    status: 0,
  };
}
