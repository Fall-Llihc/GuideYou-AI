import React, { useState } from "react";
import { HOME_OPTIONS } from "../data/homeOptions";

// Map error code dari Geolocation API ke pesan Indonesia yang ramah.
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
        text: "Lokasi tidak tersedia (GPS/jaringan kurang akurat). Pilih titik mulai manual di bawah.",
      };
    case 3:
      return {
        kind: "warn",
        text: "Pencarian lokasi terlalu lama. Coba lagi atau pilih titik mulai manual di bawah.",
      };
    default:
      return {
        kind: "warn",
        text: "Terjadi error saat mengambil lokasi. Pilih titik mulai manual di bawah.",
      };
  }
}

export default function WelcomeScreen({ onNext }) {
  const [stage, setStage] = useState("idle"); // idle | locating | found
  const [coord, setCoord] = useState(null);
  const [homeName, setHomeName] = useState("Alun-Alun Bandung");
  const [manualId, setManualId] = useState("alun-alun");
  const [notice, setNotice] = useState(null);

  const showNotice = (kind, text) => setNotice({ kind, text });
  const clearNotice = () => setNotice(null);

  const handleDetect = () => {
    clearNotice();
    setStage("locating");

    if (!navigator.geolocation) {
      showNotice("warn", "Browser kamu tidak mendukung geolokasi. Pilih titik mulai manual di bawah.");
      setStage("idle");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const c = { lat: position.coords.latitude, lng: position.coords.longitude };
        setCoord(c);
        setHomeName("Lokasi Saya");
        setManualId(null);
        setStage("found");
      },
      (error) => {
        console.warn("Geolocation error:", error.message);
        const { kind, text } = geoErrorMessage(error);
        showNotice(kind, text);
        setStage("idle");
      },
      { enableHighAccuracy: true, timeout: 10_000, maximumAge: 0 }
    );
  };

  const handleManual = (id) => {
    clearNotice();
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
        {/* Updated kicker: brand + group identity */}
        <div className="kicker">GuideYou&amp;AI · Capstone Group 6</div>
        <h1 className="display">
          Rencanain harimu di <em>Bandung</em>,<br />bareng AI.
        </h1>
        <p className="lede">
          GuideYou&amp;AI membantu kamu menyusun rute perjalanan di Bandung secara
          otomatis. Cukup beri tahu titik mulai, waktu, dan preferensimu —
          AI kami akan merancang itinerary yang optimal, hemat waktu, dan
          sesuai vibe kamu.
        </p>
        <div style={{ display: "flex", gap: 28, color: "var(--ink-mute)", fontSize: 13 }}>
          <div className="row">
            <span style={{ background: "var(--saffron)", width: 6, height: 6, borderRadius: "50%", display: "inline-block" }} />{" "}
            Filter cerdas
          </div>
          <div className="row">
            <span style={{ background: "var(--jade)", width: 6, height: 6, borderRadius: "50%", display: "inline-block" }} />{" "}
            Rekomendasi berbasis preferensi
          </div>
          <div className="row">
            <span style={{ background: "var(--coral)", width: 6, height: 6, borderRadius: "50%", display: "inline-block" }} />{" "}
            Optimisasi rute
          </div>
        </div>
      </div>

      <div className="welcome-right">
        <div className="location-card rise">
          <div className="step-num">Langkah 1 / 2</div>
          <h3>Dari mana kita mulai?</h3>
          <p>Aktifkan lokasi atau pilih titik keberangkatanmu secara manual.</p>

          <div className="location-anim">
            <div className="orbit-wrap">
              <div className="orbit o1" />
              <div className="orbit o2" />
              <div className="orbit o3" />
            </div>
            {stage === "idle" && (
              <div style={{ position: "relative", color: "var(--ink-dim)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
                — menunggu lokasi —
              </div>
            )}
            {stage === "locating" && (
              <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
                <div className="pulse" />
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--saffron)" }}>
                  Mendeteksi koordinat…
                </div>
              </div>
            )}
            {stage === "found" && coord && (
              <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: "50%",
                  background: "linear-gradient(135deg, var(--saffron), var(--coral))",
                  display: "grid", placeItems: "center", fontSize: 20,
                  boxShadow: "0 0 32px rgba(232,160,74,.5)",
                }}>
                  ⌂
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-mute)" }}>
                  HOME · {homeName}
                </div>
              </div>
            )}
          </div>

          {stage === "found" && coord && (
            <div className="coord-readout">
              <span>LAT / LNG</span>
              <span>{coord.lat.toFixed(4)}, {coord.lng.toFixed(4)}</span>
            </div>
          )}

          {notice && (
            <div className={`inline-notice notice-${notice.kind}`} role="status">
              <span className="notice-ico" aria-hidden="true">
                {notice.kind === "warn" ? "⚠" : notice.kind === "err" ? "✕" : "ℹ"}
              </span>
              <span>{notice.text}</span>
              <button type="button" className="notice-close" onClick={clearNotice} aria-label="Tutup notifikasi">
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
            <select className="select" value={manualId ?? ""} onChange={(e) => handleManual(e.target.value)}>
              {manualId === null && (
                <option value="" disabled>📍 Lokasi GPS aktif — pilih untuk override</option>
              )}
              {HOME_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>{o.name}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
