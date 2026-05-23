import React, { useState } from "react";
import { HOME_OPTIONS } from "../data/homeOptions";

export default function WelcomeScreen({ onNext }) {
  const [stage, setStage] = useState("idle"); // idle | locating | found
  const [coord, setCoord] = useState(null);
  const [homeName, setHomeName] = useState("Alun-Alun Bandung");
  const [manualId, setManualId] = useState("alun-alun");

  const handleDetect = () => {
    setStage("locating");
    // Mock geolocation — pretend to be near Alun-Alun Bandung.
    // (Real geolocation would use navigator.geolocation, but the demo dataset
    // is bounded to Bandung so we hardcode a sensible center.)
    setTimeout(() => {
      const c = { lat: -6.9215, lng: 107.6071 };
      setCoord(c);
      setHomeName("Alun-Alun Bandung (terdeteksi)");
      setManualId("alun-alun");
      setStage("found");
    }, 1800);
  };

  const handleManual = (id) => {
    const opt = HOME_OPTIONS.find((o) => o.id === id);
    setManualId(id);
    setCoord({ lat: opt.lat, lng: opt.lng });
    setHomeName(opt.name);
    setStage("found");
  };

  const proceed = () => {
    const opt = HOME_OPTIONS.find((o) => o.id === manualId);
    onNext({
      home: { lat: opt.lat, lng: opt.lng },
      homeName: opt.name,
    });
  };

  return (
    <div className="welcome">
      <div className="welcome-left">
        <div className="kicker">Capstone · AI Travel Agent</div>
        <h1 className="display">
          Rencanain harimu di <em>Bandung</em>,
          <br />
          tanpa drama.
        </h1>
        <p className="lede">
          Beri tahu kami dari mana kamu berangkat, berapa lama waktumu, dan apa yang kamu suka — agen
          perjalanan kami akan menyusun rute yang masuk akal, hemat waktu, dan sesuai vibe kamu. Tanpa
          peta yang membingungkan, hanya itinerary yang siap dijalanin.
        </p>
        <div style={{ display: "flex", gap: 28, color: "var(--ink-mute)", fontSize: 13 }}>
          <div className="row">
            <span style={{ background: "var(--saffron)", width: 6, height: 6, borderRadius: "50%" }} />{" "}
            Filter cerdas
          </div>
          <div className="row">
            <span style={{ background: "var(--jade)", width: 6, height: 6, borderRadius: "50%" }} />{" "}
            Rekomendasi berbasis preferensi
          </div>
          <div className="row">
            <span style={{ background: "var(--coral)", width: 6, height: 6, borderRadius: "50%" }} />{" "}
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
              <div
                style={{
                  position: "relative",
                  color: "var(--ink-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                }}
              >
                — menunggu lokasi —
              </div>
            )}
            {stage === "locating" && (
              <div
                style={{
                  position: "relative",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <div className="pulse" />
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--saffron)" }}>
                  Mendeteksi koordinat…
                </div>
              </div>
            )}
            {stage === "found" && coord && (
              <div
                style={{
                  position: "relative",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, var(--saffron), var(--coral))",
                    display: "grid",
                    placeItems: "center",
                    fontSize: 20,
                    boxShadow: "0 0 32px rgba(232,160,74,.5)",
                  }}
                >
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
              <span>
                {coord.lat.toFixed(4)}, {coord.lng.toFixed(4)}
              </span>
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
            <select className="select" value={manualId} onChange={(e) => handleManual(e.target.value)}>
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
