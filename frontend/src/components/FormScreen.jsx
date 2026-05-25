import React, { useMemo, useState } from "react";
import { ALL_CATEGORIES, CATEGORY_DESCRIPTIONS } from "../data/homeOptions";

function ProgressLine({ label, done, active }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        alignItems: "center",
        fontSize: 13,
        color: done ? "var(--jade)" : active ? "var(--saffron)" : "var(--ink-dim)",
      }}
    >
      <span
        style={{
          width: 16,
          height: 16,
          borderRadius: "50%",
          border: `1.5px solid ${done ? "var(--jade)" : active ? "var(--saffron)" : "var(--line)"}`,
          display: "grid",
          placeItems: "center",
          background: done ? "var(--jade)" : "transparent",
          color: "var(--bg)",
          fontSize: 10,
        }}
      >
        {done ? "✓" : ""}
      </span>
      {label}
    </div>
  );
}

const parseHM = (s) => {
  const [h, m] = s.split(":").map((n) => parseInt(n, 10));
  return h * 60 + m;
};

// Validasi sebelum submit. Kembalikan array string error agar UI bisa
// menampilkannya sekaligus (lebih baik daripada alert satu-satu).
function validateForm({ count, startMin, endMin, budget, budgetOn, maxKm, maxKmOn, categories }) {
  const errors = [];

  if (count < 1 || count > 8) {
    errors.push("Jumlah destinasi harus antara 1 sampai 8.");
  }
  if (endMin <= startMin) {
    errors.push("Jam selesai harus setelah jam mulai.");
  }
  // Minimum 2 jam total — kurang dari ini biasanya tidak realistis untuk minimal 1 destinasi
  if (endMin - startMin < 90) {
    errors.push("Total waktu perjalanan minimal 1.5 jam (cukup untuk 1 destinasi + transit).");
  }
  if (budgetOn && (budget < 0 || budget > 50_000_000)) {
    errors.push("Budget harus antara 0 dan 50 juta rupiah.");
  }
  if (maxKmOn && (maxKm < 1 || maxKm > 100)) {
    errors.push("Jarak maksimal harus antara 1 dan 100 km.");
  }
  // Validasi count vs categories: kalau count < jumlah kategori dipilih,
  // cukup beri warning lewat info — tidak block submit.
  return errors;
}

export default function FormScreen({ homeData, onBack, onSubmit }) {
  const [count, setCount] = useState(4);
  const [maxKmOn, setMaxKmOn] = useState(false);
  const [maxKm, setMaxKm] = useState(40);
  const [startHM, setStartHM] = useState("09:00");
  const [endHM, setEndHM] = useState("21:00");
  const [budgetOn, setBudgetOn] = useState(true);
  const [budget, setBudget] = useState(500_000);
  // Default: Alam + Kuliner (no Belanja — kategori itu sudah dihapus)
  const [categories, setCategories] = useState(["Alam", "Kuliner"]);
  const [submitErrors, setSubmitErrors] = useState([]);

  const toggleCat = (c) => {
    setCategories((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));
    setSubmitErrors([]); // reset errors saat user mengubah input
  };

  const startMin = parseHM(startHM);
  const endMin = parseHM(endHM);
  const dur = endMin - startMin;
  const durStr = dur > 0 ? `${Math.floor(dur / 60)} jam ${dur % 60} menit` : "—";

  // Soft warning kalau count < kategori dipilih (akan ada kategori yg tidak terwakili)
  const categoryCoverageWarning = useMemo(() => {
    if (categories.length >= 2 && count < categories.length) {
      return `Kamu pilih ${categories.length} kategori tapi hanya ${count} destinasi — agen akan mengutamakan ${count} kategori dengan skor tertinggi, sisanya tidak terwakili.`;
    }
    return null;
  }, [count, categories.length]);

  // Tip kalau max_km terlalu ketat (mungkin bikin hasil kosong)
  const tightDistanceTip = useMemo(() => {
    if (!maxKmOn) return null;
    if (maxKm < 5) {
      return `Radius ${maxKm} km dari home sangat ketat untuk Bandung. Kemungkinan besar tidak ada destinasi wisata yang masuk — coba 10 km ke atas. Sistem akan memberi tahu kalau memang tidak ada yang cocok.`;
    }
    if (maxKm < 10) {
      return `Radius ${maxKm} km cukup ketat — destinasi yang muncul akan terbatas di sekitar pusat kota. Naikkan ke 20-30 km kalau mau cakupan lebih luas (Lembang, Ciwidey).`;
    }
    return null;
  }, [maxKmOn, maxKm]);

  const submit = () => {
    const errs = validateForm({
      count, startMin, endMin, budget, budgetOn, maxKm, maxKmOn, categories,
    });
    if (errs.length) {
      setSubmitErrors(errs);
      return;
    }
    setSubmitErrors([]);
    onSubmit({
      ...homeData,
      count,
      maxKm: maxKmOn ? maxKm : null,
      startMin,
      endMin,
      budget: budgetOn ? budget : null,
      categories,
    });
  };

  return (
    <div className="form-screen">
      <aside className="form-side">
        <div className="kicker" style={{ marginBottom: 14 }}>
          Langkah 2 / 2
        </div>
        <div className="you-are-here">
          <div className="pin">⌂</div>
          <div>
            <div className="label">Titik Mulai</div>
            <div className="name">{homeData.homeName}</div>
            <div className="coord">
              {homeData.home.lat.toFixed(4)}, {homeData.home.lng.toFixed(4)}
            </div>
          </div>
        </div>

        <div style={{ marginTop: 28 }}>
          <div className="kicker" style={{ marginBottom: 12 }}>
            Progres
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <ProgressLine label="Titik mulai" done />
            <ProgressLine label="Preferensi" done={false} active />
            <ProgressLine label="Generate itinerary" done={false} />
          </div>
        </div>

        <div style={{ marginTop: 36, paddingTop: 24, borderTop: "1px solid var(--line-soft)" }}>
          <div className="kicker" style={{ marginBottom: 10 }}>
            Tips
          </div>
          <p style={{ fontSize: 13, color: "var(--ink-mute)", lineHeight: 1.6, margin: 0 }}>
            Pilih jam mulai lebih pagi (07:00–09:00) supaya destinasi alam tidak terlalu ramai dan
            cuaca masih sejuk.
          </p>
        </div>

        <button className="btn ghost" style={{ marginTop: 28 }} onClick={onBack}>
          ← Ganti titik mulai
        </button>
      </aside>

      <main className="form-main">
        <h2>Atur perjalananmu.</h2>
        <p className="subtitle">
          Sesuaikan parameter di bawah. Semua opsional kecuali jumlah destinasi dan jam.
        </p>

        <div className="field-grid">
          {/* Jumlah destinasi */}
          <div className="field">
            <label>
              Jumlah Destinasi
              <span className="help">Berapa tempat yang ingin dikunjungi?</span>
            </label>
            <div className="stepper">
              <button onClick={() => setCount(Math.max(1, count - 1))}>−</button>
              <div className="val">
                {count}
                <small>tempat</small>
              </div>
              <button onClick={() => setCount(Math.min(8, count + 1))}>+</button>
            </div>
          </div>

          {/* Budget */}
          <div className="field">
            <label>
              <span>
                Budget{" "}
                <span
                  style={{
                    color: "var(--ink-dim)",
                    textTransform: "none",
                    fontFamily: "var(--font-body)",
                    letterSpacing: 0,
                  }}
                >
                  (opsional)
                </span>
              </span>
              <span
                className={`toggle ${budgetOn ? "on" : ""}`}
                onClick={() => setBudgetOn(!budgetOn)}
              >
                <span className="sw" />
                <span>{budgetOn ? "Aktif" : "Nonaktif"}</span>
              </span>
            </label>
            <div className={`num-input ${budgetOn ? "" : "is-disabled"}`}>
              <span className="num-prefix">Rp</span>
              <input
                type="number"
                min="0"
                step="50000"
                value={budget}
                onChange={(e) => setBudget(Math.max(0, parseInt(e.target.value, 10) || 0))}
                disabled={!budgetOn}
                placeholder="500000"
              />
              <div className="num-meta">{budgetOn ? budget.toLocaleString("id-ID") : "—"}</div>
            </div>
          </div>

          {/* Max distance */}
          <div className="field">
            <label>
              <span>
                Jarak Maks. dari Home{" "}
                <span
                  style={{
                    color: "var(--ink-dim)",
                    textTransform: "none",
                    fontFamily: "var(--font-body)",
                    letterSpacing: 0,
                  }}
                >
                  (opsional)
                </span>
              </span>
              <span className={`toggle ${maxKmOn ? "on" : ""}`} onClick={() => setMaxKmOn(!maxKmOn)}>
                <span className="sw" />
                <span>{maxKmOn ? "Aktif" : "Nonaktif"}</span>
              </span>
            </label>
            <div className={`num-input ${maxKmOn ? "" : "is-disabled"}`}>
              <input
                type="number"
                min="1"
                max="100"
                step="1"
                value={maxKm}
                onChange={(e) => setMaxKm(Math.max(1, parseInt(e.target.value, 10) || 1))}
                disabled={!maxKmOn}
                placeholder="40"
              />
              <span className="num-suffix">km</span>
              <div className="num-meta">
                {maxKmOn ? `Setiap destinasi ≤ ${maxKm} km dari home` : "—"}
              </div>
            </div>
            {tightDistanceTip && (
              <div className="form-info" style={{ marginTop: 8 }}>
                ℹ {tightDistanceTip}
              </div>
            )}
          </div>

          {/* Time start/end */}
          <div className="field field-full">
            <label>
              Jam Perjalanan
              <span className="help">Durasi: {durStr}</span>
            </label>
            <div className="time-pair">
              <div className="time-input">
                <span className="lab">Mulai</span>
                <input type="time" value={startHM} onChange={(e) => setStartHM(e.target.value)} />
              </div>
              <div className="arrow">→</div>
              <div className="time-input">
                <span className="lab">Selesai</span>
                <input type="time" value={endHM} onChange={(e) => setEndHM(e.target.value)} />
              </div>
            </div>
          </div>

          {/* Categories */}
          <div className="field field-full">
            <label>
              Kategori Favorit
              <span className="help">
                {categories.length === 0
                  ? "Belum ada — biarkan kosong = semua kategori"
                  : `${categories.length} dipilih`}
              </span>
            </label>
            <div className="chips">
              {ALL_CATEGORIES.map((c) => (
                <span
                  key={c}
                  className={`chip ${categories.includes(c) ? "on" : ""}`}
                  onClick={() => toggleCat(c)}
                  title={CATEGORY_DESCRIPTIONS[c] || c}
                >
                  {c}
                  <span className="x">×</span>
                </span>
              ))}
            </div>

            {categoryCoverageWarning && (
              <div className="form-info" style={{ marginTop: 10 }}>
                ℹ {categoryCoverageWarning}
              </div>
            )}
          </div>
        </div>

        {/* Inline errors di atas tombol submit */}
        {submitErrors.length > 0 && (
          <div className="form-errors" role="alert">
            <div className="form-errors-head">⚠ Tidak bisa lanjut — perbaiki dulu:</div>
            <ul>
              {submitErrors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="form-footer">
          <div className="note">
            Itinerary akan diatur ulang otomatis jika ada konflik waktu atau jarak. Kamu bisa tweak
            lagi setelah hasil keluar.
          </div>
          <button className="btn primary lg" onClick={submit}>
            ✱ Generate Itinerary →
          </button>
        </div>
      </main>
    </div>
  );
}
