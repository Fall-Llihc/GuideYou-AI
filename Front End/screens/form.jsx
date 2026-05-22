// Parameter form screen
function FormScreen({ homeData, onBack, onSubmit }) {
  const [count, setCount] = React.useState(4);
  const [maxKmOn, setMaxKmOn] = React.useState(false);
  const [maxKm, setMaxKm] = React.useState(40);
  const [startHM, setStartHM] = React.useState("09:00");
  const [endHM, setEndHM] = React.useState("21:00");
  const [budgetOn, setBudgetOn] = React.useState(true);
  const [budget, setBudget] = React.useState(500000);
  const [categories, setCategories] = React.useState(["Alam", "Kuliner"]);

  const ALL_CATS = ["Alam", "Kuliner", "Budaya", "Wisata", "Belanja"];

  const toggleCat = (c) => {
    setCategories(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);
  };

  const parseHM = (s) => {
    const [h, m] = s.split(":").map(n => parseInt(n, 10));
    return h * 60 + m;
  };

  const submit = () => {
    onSubmit({
      ...homeData,
      count,
      maxKm: maxKmOn ? maxKm : null,
      startMin: parseHM(startHM),
      endMin: parseHM(endHM),
      budget: budgetOn ? budget : null,
      categories,
    });
  };

  const dur = parseHM(endHM) - parseHM(startHM);
  const durStr = dur > 0 ? `${Math.floor(dur/60)} jam ${dur%60} menit` : "—";

  return (
    <div className="form-screen">
      <aside className="form-side">
        <div className="kicker" style={{marginBottom: 14}}>Langkah 2 / 2</div>
        <div className="you-are-here">
          <div className="pin">⌂</div>
          <div>
            <div className="label">Titik Mulai</div>
            <div className="name">{homeData.homeName}</div>
            <div className="coord">{homeData.home.lat.toFixed(4)}, {homeData.home.lng.toFixed(4)}</div>
          </div>
        </div>

        <div style={{marginTop: 28}}>
          <div className="kicker" style={{marginBottom: 12}}>Progres</div>
          <div style={{display: "flex", flexDirection: "column", gap: 10}}>
            <ProgressLine label="Titik mulai" done={true}/>
            <ProgressLine label="Preferensi" done={false} active={true}/>
            <ProgressLine label="Generate itinerary" done={false}/>
          </div>
        </div>

        <div style={{marginTop: 36, paddingTop: 24, borderTop: "1px solid var(--line-soft)"}}>
          <div className="kicker" style={{marginBottom: 10}}>Tips</div>
          <p style={{fontSize: 13, color: "var(--ink-mute)", lineHeight: 1.6, margin: 0}}>
            Pilih jam mulai lebih pagi (07:00–09:00) supaya destinasi alam tidak terlalu ramai dan cuaca masih sejuk.
          </p>
        </div>

        <button className="btn ghost" style={{marginTop: 28}} onClick={onBack}>
          ← Ganti titik mulai
        </button>
      </aside>

      <main className="form-main">
        <h2>Atur perjalananmu.</h2>
        <p className="subtitle">Sesuaikan parameter di bawah. Semua opsional kecuali jumlah destinasi dan jam.</p>

        <div className="field-grid">
          {/* Number of destinations */}
          <div className="field">
            <label>
              Jumlah Destinasi
              <span className="help">Berapa tempat yang ingin dikunjungi?</span>
            </label>
            <div className="stepper">
              <button onClick={() => setCount(Math.max(1, count - 1))}>−</button>
              <div className="val">{count}<small>tempat</small></div>
              <button onClick={() => setCount(Math.min(8, count + 1))}>+</button>
            </div>
          </div>

          {/* Budget — number input + optional toggle */}
          <div className="field">
            <label>
              <span>Budget <span style={{color: "var(--ink-dim)", textTransform: "none", fontFamily: "var(--font-body)", letterSpacing: 0}}>(opsional)</span></span>
              <span className={`toggle ${budgetOn ? "on" : ""}`} onClick={() => setBudgetOn(!budgetOn)}>
                <span className="sw"></span>
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
                onChange={e => setBudget(Math.max(0, parseInt(e.target.value) || 0))}
                disabled={!budgetOn}
                placeholder="500000"
              />
              <div className="num-meta">{budgetOn ? budget.toLocaleString("id-ID") : "—"}</div>
            </div>
          </div>

          {/* Max distance — number input + optional toggle */}
          <div className="field">
            <label>
              <span>Jarak Maks. Antar Tempat <span style={{color: "var(--ink-dim)", textTransform: "none", fontFamily: "var(--font-body)", letterSpacing: 0}}>(opsional)</span></span>
              <span className={`toggle ${maxKmOn ? "on" : ""}`} onClick={() => setMaxKmOn(!maxKmOn)}>
                <span className="sw"></span>
                <span>{maxKmOn ? "Aktif" : "Nonaktif"}</span>
              </span>
            </label>
            <div className={`num-input ${maxKmOn ? "" : "is-disabled"}`}>
              <input
                type="number"
                min="1"
                step="1"
                value={maxKm}
                onChange={e => setMaxKm(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={!maxKmOn}
                placeholder="40"
              />
              <span className="num-suffix">km</span>
              <div className="num-meta">{maxKmOn ? `≤ ${maxKm} km tiap hop` : "—"}</div>
            </div>
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
                <input type="time" value={startHM} onChange={e => setStartHM(e.target.value)} />
              </div>
              <div className="arrow">→</div>
              <div className="time-input">
                <span className="lab">Selesai</span>
                <input type="time" value={endHM} onChange={e => setEndHM(e.target.value)} />
              </div>
            </div>
          </div>

          {/* Categories */}
          <div className="field field-full">
            <label>
              Kategori Favorit
              <span className="help">{categories.length} dipilih · biarkan kosong untuk semua</span>
            </label>
            <div className="chips">
              {ALL_CATS.map(c => (
                <span key={c} className={`chip ${categories.includes(c) ? "on" : ""}`} onClick={() => toggleCat(c)}>
                  {c}
                  <span className="x">×</span>
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="form-footer">
          <div className="note">
            Itinerary akan diatur ulang otomatis jika ada konflik waktu atau jarak. Kamu bisa tweak lagi setelah hasil keluar.
          </div>
          <button className="btn primary lg" onClick={submit}>
            ✱ Generate Itinerary →
          </button>
        </div>
      </main>
    </div>
  );
}

function ProgressLine({ label, done, active }) {
  return (
    <div style={{display: "flex", gap: 10, alignItems: "center", fontSize: 13, color: done ? "var(--jade)" : active ? "var(--saffron)" : "var(--ink-dim)"}}>
      <span style={{
        width: 16, height: 16, borderRadius: "50%",
        border: `1.5px solid ${done ? "var(--jade)" : active ? "var(--saffron)" : "var(--line)"}`,
        display: "grid", placeItems: "center",
        background: done ? "var(--jade)" : "transparent",
        color: "var(--bg)",
        fontSize: 10,
      }}>
        {done ? "✓" : ""}
      </span>
      {label}
    </div>
  );
}

window.FormScreen = FormScreen;
