import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const STAGES = [
  { ico: "⌕", name: "Filter Agent", desc: "Menyaring destinasi berdasarkan budget & waktu" },
  { ico: "✱", name: "Recommendation Agent", desc: "Cosine similarity dengan preferensi kategori" },
  { ico: "↦", name: "Route Optimizer", desc: "Nearest-neighbor + 2-opt improvement" },
  { ico: "✎", name: "Narrative Agent", desc: "Menyusun cerita perjalanan dengan LLM" },
];

export default function LoadingScreen({ params, onDone, onError }) {
  // Visual stage progression — purely cosmetic; runs in parallel with the
  // real API call. We hold at the last stage until the network finishes.
  const [active, setActive] = useState(0);
  const [error, setError] = useState(null);
  const fired = useRef(false);

  useEffect(() => {
    if (fired.current) return;
    fired.current = true;

    // Animate through stages 0→3 over ~2.4s
    const timers = [];
    STAGES.forEach((_, i) => {
      timers.push(setTimeout(() => setActive(i + 1), 600 + i * 600));
    });

    // Real backend call — Render free-tier wakes can take 30–50s so we
    // intentionally do NOT race a timeout here.
    api
      .plan(params)
      .then((data) => {
        // Make sure the user sees the final tick before we transition.
        setTimeout(() => onDone(data), 200);
      })
      .catch((err) => {
        setError(err.message || "Terjadi kesalahan saat memanggil API.");
      });

    return () => timers.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className="loader">
        <div className="loader-card rise">
          <h3 style={{ color: "var(--coral)" }}>Gagal menghasilkan itinerary</h3>
          <p style={{ marginTop: 8 }}>{error}</p>
          <button
            className="btn primary"
            style={{ marginTop: 20 }}
            onClick={() => onError && onError()}
          >
            ← Kembali ke form
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="loader">
      <div className="loader-card rise">
        <div className="spinner" />
        <h3>Menyusun itinerary terbaikmu…</h3>
        <p>Empat agen sedang bekerja sama untuk merancang rute yang sempurna.</p>
        <div className="agents">
          {STAGES.map((s, i) => {
            const isDone = i < active;
            const isActive = i === active;
            return (
              <div
                key={i}
                className={`agent-row ${isDone ? "done" : ""} ${isActive ? "active" : ""}`}
              >
                <div className="left">
                  <div className="ico">{s.ico}</div>
                  <div>
                    <div style={{ fontWeight: 500 }}>{s.name}</div>
                    <div style={{ fontSize: 12, color: "var(--ink-mute)" }}>{s.desc}</div>
                  </div>
                </div>
                <div className="status">{isDone ? "✓ done" : isActive ? "running" : "queued"}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
