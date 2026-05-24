import React, { useEffect, useRef, useState } from "react";
import { api, describeError } from "../api/client";

const STAGES = [
  { ico: "⌕", name: "Filter Agent",         desc: "Menyaring destinasi berdasar budget, jarak, & waktu" },
  { ico: "✱", name: "Recommendation Agent", desc: "Cosine similarity + RL ranking dengan preferensi user" },
  { ico: "↦", name: "Route Optimizer",      desc: "Nearest-neighbor TSP untuk minimasi total jarak" },
  { ico: "✎", name: "Narrative Agent",      desc: "Menyusun cerita perjalanan dengan LLM" },
];

// Visual config per error kind. Ikon, judul, tone, dan tombol-tombol-nya.
// Berbeda kind → berbeda urgency:
//   - cold-start / network → retry dianjurkan, biasanya self-resolve
//   - validation           → balik ke form, fix input
//   - timeout / server     → retry dengan delay
//   - rate-limit           → tunggu sebentar
const ERROR_PRESETS = {
  "cold-start": {
    icon: "⏳",
    title: "Server sedang dibangunkan",
    tone: "warn", // kuning saffron
    primaryAction: "retry",
    primaryLabel: "Coba lagi",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
  network: {
    icon: "🛰",
    title: "Koneksi ke server gagal",
    tone: "warn",
    primaryAction: "retry",
    primaryLabel: "Coba lagi",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
  timeout: {
    icon: "⏱",
    title: "Server merespon lebih lama dari biasanya",
    tone: "warn",
    primaryAction: "retry",
    primaryLabel: "Coba lagi",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
  validation: {
    icon: "✎",
    title: "Ada parameter yang perlu diperbaiki",
    tone: "info",
    primaryAction: "back",
    primaryLabel: "Perbaiki di form",
    secondaryAction: "retry",
    secondaryLabel: "Coba kirim ulang",
  },
  "rate-limit": {
    icon: "⌛",
    title: "Terlalu banyak permintaan",
    tone: "warn",
    primaryAction: "retry",
    primaryLabel: "Coba lagi (1 menit)",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
  server: {
    icon: "⚠",
    title: "Server sedang bermasalah",
    tone: "danger",
    primaryAction: "retry",
    primaryLabel: "Coba lagi",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
  unknown: {
    icon: "✕",
    title: "Terjadi error tak terduga",
    tone: "danger",
    primaryAction: "retry",
    primaryLabel: "Coba lagi",
    secondaryAction: "back",
    secondaryLabel: "Kembali ke form",
  },
};

export default function LoadingScreen({ params, onDone, onError }) {
  // Visual stage progression — purely cosmetic; runs in parallel with the
  // real API call. We hold at the last stage until the network finishes.
  const [active, setActive] = useState(0);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0); // berapa kali sudah retry
  const [coldHintShown, setColdHintShown] = useState(false);
  const fired = useRef(false);

  const callApi = () => {
    setError(null);
    setActive(0);
    fired.current = true;

    const timers = [];
    STAGES.forEach((_, i) => {
      timers.push(setTimeout(() => setActive(i + 1), 600 + i * 600));
    });

    // Tampilkan hint cold-start kalau API belum balas dalam 12 detik —
    // memberi sinyal ke user bahwa request belum hang, server masih warm-up.
    const coldHintTimer = setTimeout(() => setColdHintShown(true), 12_000);

    api
      .plan(params)
      .then((data) => {
        timers.forEach(clearTimeout);
        clearTimeout(coldHintTimer);
        // Pastikan user lihat tick terakhir sebelum transisi
        setTimeout(() => onDone(data), 200);
      })
      .catch((err) => {
        timers.forEach(clearTimeout);
        clearTimeout(coldHintTimer);
        setError(describeError(err));
      });
  };

  // Initial call (sekali pas mount)
  useEffect(() => {
    if (fired.current) return;
    callApi();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRetry = () => {
    setAttempt((n) => n + 1);
    setColdHintShown(false);
    fired.current = false;
    callApi();
  };

  // ─── Error UI ─────────────────────────────────────────────
  if (error) {
    const preset = ERROR_PRESETS[error.kind] || ERROR_PRESETS.unknown;

    const handlePrimary =
      preset.primaryAction === "retry" ? handleRetry : () => onError && onError(error);
    const handleSecondary =
      preset.secondaryAction === "retry" ? handleRetry : () => onError && onError(error);

    return (
      <div className="loader">
        <div className={`loader-card error-card error-${preset.tone} rise`}>
          <div className="error-icon" aria-hidden="true">
            {preset.icon}
          </div>

          <h3 className="error-title">{preset.title}</h3>

          {/* message bisa multi-line (pydantic detail) — pakai whiteSpace pre */}
          <p className="error-msg">{error.message}</p>

          {error.hint && (
            <div className="error-hint">
              <span className="error-hint-label">💡 Saran</span>
              <span>{error.hint}</span>
            </div>
          )}

          {error.status > 0 && (
            <div className="error-meta mono">
              status {error.status} · attempt {attempt + 1}
            </div>
          )}

          <div className="error-actions">
            <button className="btn primary" onClick={handlePrimary}>
              {preset.primaryLabel}
            </button>
            <button className="btn ghost" onClick={handleSecondary}>
              {preset.secondaryLabel}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ─── Loading UI (default) ─────────────────────────────────
  return (
    <div className="loader">
      <div className="loader-card rise">
        <div className="spinner" />
        <h3>Menyusun itinerary terbaikmu…</h3>
        <p>Empat agen sedang bekerja sama untuk merancang rute yang sempurna.</p>

        {coldHintShown && (
          <div className="cold-hint">
            ⏳ Server free-tier kadang butuh waktu warm-up. Mohon tunggu, biasanya tidak lebih
            dari 1 menit.
          </div>
        )}

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
