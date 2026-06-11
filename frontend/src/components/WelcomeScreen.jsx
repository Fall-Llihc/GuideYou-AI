import React, { useState } from "react";
import { HOME_OPTIONS } from "../data/homeOptions";

function geoErrorMessage(err) {
  switch (err && err.code) {
    case 1:
      return {
        kind: "warn",
        text: "Izin lokasi ditolak. Aktifkan izin lokasi di browser, atau pilih titik mulai manual di bawah.",
      };
    case 2:
      return {
        kind: "warn",
        text: "Lokasi tidak tersedia (GPS atau jaringan kurang akurat). Pilih titik mulai manual di bawah.",
      };
    case 3:
      return {
        kind: "warn",
        text: "Pencarian lokasi terlalu lama. Coba lagi atau pilih titik mulai manual.",
      };
    default:
      return {
        kind: "warn",
        text: "Gagal mengambil lokasi. Pilih titik mulai manual di bawah.",
      };
  }
}

export default function WelcomeScreen({ onNext }) {
  const [stage, setStage] = useState("idle"); // idle | locating | found
  const [coord, setCoord] = useState(null);
  const [homeName, setHomeName] = useState("Alun-Alun Bandung");
  const [manualId, setManualId] = useState("alun-alun");
  const [notice, setNotice] = useState(null); // { kind: "warn"|"info"|"err", text }

  const showNotice = (kind, text) => setNotice({ kind, text });
  // FIX 1: notice auto-clear when stage changes to "found"
  const clearNotice = () => setNotice(null);

  const handleDetect = () => {
    clearNotice();
    setStage("locating");

    if (!navigator.geolocation) {
      showNotice("warn", "Browser ini tidak mendukung geolokasi. Pilih titik mulai manual di bawah.");
      setStage("idle");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const c = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        setCoord(c);
        setHomeName("Lokasi Saya");
        setManualId(null);
        setStage("found");
        clearNotice(); // FIX 1: explicitly clear any lingering notice on success
      },
      (error) => {
        console.warn("Geolocation error:", error.message);
        const { kind, text } = geoErrorMessage(error);
        showNotice(kind, text);
        setStage("idle");
      },
      {
        enableHighAccuracy: true,
        timeout: 10_000,
        maximumAge: 0,
      }
    );
  };

  const handleManual = (id) => {
    clearNotice(); // FIX 1: clear notice when user picks manual option
    const opt = HOME_OPTIONS.find((o) => o.id === id);
    if (!opt) return;
    setManualId(id);
    setCoord({ lat: opt.lat, lng: opt.lng });
    setHomeName(opt.name);
    setStage("found");
  };

  const proceed = () => {
    if (manualId) {
      const opt = HOME_OPTIONS.find((o) => o.id === manualId);
      onNext({ home: { lat: opt.lat, lng: opt.lng }, homeName: opt.name });
    } else {
      onNext({ home: coord, homeName });
    }
  };

  return (
    <div className="welcome">
      <div className="welcome-left">
        <div className="kicker">GuideYou&amp;AI · Capstone Group 6</div>
        <h1 className="display">
          Rencanain harimu di <em>Bandung</em>,<br />
          bareng AI.
        </h1>
        <p className="lede">
          Kasih tahu titik berangkat, berapa lama waktunya, dan mau wisata apa —
          GuideYou&amp;AI langsung susunkan itinerary yang efisien dan sesuai selera.
          Nggak perlu buka banyak tab lagi.
        </p>
        <div className="welcome-features">
          <div className="row">
            <span className="feat-dot" style={{ background: "var(--saffron)" }} />
            Filter cerdas
          </div>
          <div className="row">
            <span className="feat-dot" style={{ background: "var(--jade)" }} />
            Rekomendasi berbasis preferensi
          </div>
          <div className="row">
            <span className="feat-dot" style={{ background: "var(--coral)" }} />
            Optimisasi rute
          </div>
        </div>
      </div>

      <div className="welcome-right">
        <div className="location-card rise">
          <div className="step-num">Langkah 1 / 2</div>
          <h3>Kita mulai dari mana?</h3>
          <p>Aktifkan lokasi atau pilih titik keberangkatan secara manual.</p>

          <div className="location-anim">
            <div className="orbit-wrap">
              <div className="orbit o1" />
              <div className="orbit o2" />
              <div className="orbit o3" />
            </div>

            {stage === "idle" && (
              <div className="loc-status loc-status-idle">
                — menunggu lokasi —
              </div>
            )}

            {stage === "locating" && (
              <div className="loc-status loc-status-locating">
                <div className="pulse" />
                <span>Mendeteksi koordinat…</span>
              </div>
            )}

            {stage === "found" && coord && (
              <div className="loc-status loc-status-found">
                <div className="home-pin">⌂</div>
                <span className="home-label">HOME · {homeName}</span>
              </div>
            )}
          </div>

          {stage === "found" && coord && (
            <div className="coord-readout">
              <span>LAT / LNG</span>
              <span>
                {coord.lat.toFixed(4)}, {coord.lng.toFixed(4)}
              </span>
            </div>
          )}

          {/* FIX 1: Notice hanya tampil saat stage bukan "found" */}
          {notice && stage !== "found" && (
            <div className={`inline-notice notice-${notice.kind}`} role="status">
              <span className="notice-ico" aria-hidden="true">
                {notice.kind === "warn" ? "⚠" : notice.kind === "err" ? "✕" : "ℹ"}
              </span>
              <span>{notice.text}</span>
              <button
                type="button"
                className="notice-close"
                onClick={clearNotice}
                aria-label="Tutup notifikasi"
              >
                ×
              </button>
            </div>
          )}

          {stage !== "found" ? (
            <button
              className="btn primary lg"
              style={{ width: "100%", justifyContent: "center" }}
              onClick={handleDetect}
              disabled={stage === "locating"}
            >
              {stage === "locating" ? "Mencari lokasi…" : "✱ Deteksi lokasi saya"}
            </button>
          ) : (
            <button
              className="btn primary lg"
              style={{ width: "100%", justifyContent: "center" }}
              onClick={proceed}
            >
              Lanjutkan →
            </button>
          )}

          <div className="manual">
            <label>Atau pilih manual</label>
            <select
              className="select"
              value={manualId ?? ""}
              onChange={(e) => handleManual(e.target.value)}
            >
              {manualId === null && (
                <option value="" disabled>
                  📍 Lokasi GPS aktif — pilih untuk override
                </option>
              )}
              {HOME_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
