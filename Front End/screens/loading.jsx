// Loading screen — show agent pipeline working
function LoadingScreen({ onDone }) {
  const stages = [
    { ico: "⌕", name: "Filter Agent", desc: "Menyaring destinasi berdasarkan budget & waktu" },
    { ico: "✱", name: "Recommendation Agent", desc: "Cosine similarity dengan preferensi kategori" },
    { ico: "↦", name: "Route Optimizer", desc: "Nearest-neighbor + 2-opt improvement" },
    { ico: "✎", name: "Narrative Agent", desc: "Menyusun cerita perjalanan dengan LLM" },
  ];

  const [active, setActive] = React.useState(0);

  React.useEffect(() => {
    const timers = [];
    stages.forEach((_, i) => {
      timers.push(setTimeout(() => setActive(i + 1), 600 + i * 700));
    });
    timers.push(setTimeout(() => onDone(), 600 + stages.length * 700 + 300));
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="loader">
      <div className="loader-card rise">
        <div className="spinner"></div>
        <h3>Menyusun itinerary terbaikmu…</h3>
        <p>Empat agen sedang bekerja sama untuk merancang rute yang sempurna.</p>
        <div className="agents">
          {stages.map((s, i) => {
            const isDone = i < active;
            const isActive = i === active;
            return (
              <div key={i} className={`agent-row ${isDone ? "done" : ""} ${isActive ? "active" : ""}`}>
                <div className="left">
                  <div className="ico">{s.ico}</div>
                  <div>
                    <div style={{fontWeight: 500}}>{s.name}</div>
                    <div style={{fontSize: 12, color: "var(--ink-mute)"}}>{s.desc}</div>
                  </div>
                </div>
                <div className="status">
                  {isDone ? "✓ done" : isActive ? "running" : "queued"}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

window.LoadingScreen = LoadingScreen;
