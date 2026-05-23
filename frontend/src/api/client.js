// HTTP client for the Bandung AI Travel Agent backend.
//
// REACT_APP_API_URL is baked in at build time by Create React App:
//   - .env.development → http://localhost:8000
//   - .env.production  → https://bandung-travel-api.onrender.com
//
// Trailing slash is stripped so callers can compose URLs without worrying.
const RAW_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
export const API_BASE = RAW_BASE.replace(/\/+$/, "");

class ApiError extends Error {
  constructor(message, { status, body } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  let resp;
  try {
    resp = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
  } catch (err) {
    // Render free tier sleeps after 15 min idle — first request can take 30–50s
    // to wake. The bare network error is opaque, so we surface a friendlier hint.
    throw new ApiError(
      `Tidak bisa terhubung ke server. ${
        API_BASE.includes("onrender.com")
          ? "Backend mungkin sedang bangun dari sleep — coba lagi dalam 30 detik."
          : `Cek apakah backend berjalan di ${API_BASE}.`
      }`,
      { status: 0, body: String(err) }
    );
  }

  let body = null;
  try {
    body = await resp.json();
  } catch {
    /* response had no JSON body */
  }

  if (!resp.ok) {
    const detail = body?.detail || resp.statusText || "Unknown error";
    throw new ApiError(`API ${resp.status}: ${detail}`, { status: resp.status, body });
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),

  /** params shape mirrors backend PlanRequest schema */
  plan: (params) =>
    request("/api/plan", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};

export { ApiError };
