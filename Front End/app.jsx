// Main app — orchestrate screens
function App() {
  const [view, setView] = React.useState("welcome"); // welcome | form | loading | results
  const [homeData, setHomeData] = React.useState(null);
  const [params, setParams] = React.useState(null);

  // Tweaks panel
  const [t, setTweak] = window.useTweaks(/*EDITMODE-BEGIN*/{
    "accentColor": "#e8a04a",
    "background": "warmDark",
    "demoMode": false
  }/*EDITMODE-END*/);

  // Apply accent color tweak live
  React.useEffect(() => {
    document.documentElement.style.setProperty("--saffron", t.accentColor);
    // derive a lighter version
    document.documentElement.style.setProperty("--saffron-2", t.accentColor);
  }, [t.accentColor]);

  // Apply background tweak
  React.useEffect(() => {
    const bg = t.background;
    const root = document.documentElement;
    if (bg === "warmDark") {
      root.style.setProperty("--bg", "#0f1115");
      root.style.setProperty("--bg-2", "#161a21");
      root.style.setProperty("--bg-3", "#1d222b");
      root.style.setProperty("--ink", "#f4ede2");
      root.style.setProperty("--ink-mute", "#b8b0a3");
    } else if (bg === "deepInk") {
      root.style.setProperty("--bg", "#08090c");
      root.style.setProperty("--bg-2", "#101218");
      root.style.setProperty("--bg-3", "#181b22");
      root.style.setProperty("--ink", "#eef0f5");
      root.style.setProperty("--ink-mute", "#a5a8b3");
    } else if (bg === "cream") {
      root.style.setProperty("--bg", "#f6f1e8");
      root.style.setProperty("--bg-2", "#ede5d4");
      root.style.setProperty("--bg-3", "#e0d6c0");
      root.style.setProperty("--ink", "#1f1a12");
      root.style.setProperty("--ink-mute", "#5e564a");
      root.style.setProperty("--line", "#d3c8b0");
      root.style.setProperty("--line-soft", "#e0d6c0");
    }
  }, [t.background]);

  // Demo mode: skip welcome, prefill, jump to results
  React.useEffect(() => {
    if (t.demoMode && view === "welcome") {
      const hd = { home: { lat: -6.9215, lng: 107.6071 }, homeName: "Alun-Alun Bandung" };
      setHomeData(hd);
      setParams({
        ...hd,
        count: 4,
        maxKm: null,
        startMin: 9 * 60,
        endMin: 19 * 60,
        budget: 500000,
        categories: ["Alam", "Kuliner"],
      });
      setView("results");
    }
  }, [t.demoMode]);

  const goForm = (data) => {
    setHomeData(data);
    setView("form");
  };

  const generate = (p) => {
    setParams(p);
    setView("loading");
  };

  return (
    <div className="app-shell" data-screen-label={`Bandung AI Travel Agent — ${view}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">B</div>
          <div className="brand-name">Bandung <span>AI Travel</span></div>
        </div>
        <div className="meta">
          <div className="row"><span className="dot"></span> Multi-agent online</div>
          <div className="mono" style={{fontSize: 12, color: "var(--ink-dim)"}}>capstone · v1.0</div>
        </div>
      </header>

      {view === "welcome" && (
        <window.WelcomeScreen onNext={goForm}/>
      )}
      {view === "form" && homeData && (
        <window.FormScreen
          homeData={homeData}
          onBack={() => setView("welcome")}
          onSubmit={generate}
        />
      )}
      {view === "loading" && (
        <window.LoadingScreen onDone={() => setView("results")}/>
      )}
      {view === "results" && params && (
        <window.ResultsScreen
          params={params}
          onRestart={() => { setView("welcome"); setHomeData(null); setParams(null); }}
          onEditParams={() => setView("form")}
        />
      )}

      <window.TweaksPanel>
        <window.TweakSection title="Tampilan">
          <window.TweakColor
            label="Warna Aksen"
            value={t.accentColor}
            options={["#e8a04a", "#d9684e", "#5fb09b", "#876187", "#b9a06a"]}
            onChange={v => setTweak("accentColor", v)}
          />
          <window.TweakSelect
            label="Tema"
            value={t.background}
            options={[
              {value: "warmDark", label: "Warm Dark (default)"},
              {value: "deepInk", label: "Deep Ink"},
              {value: "cream", label: "Cream (light)"},
            ]}
            onChange={v => setTweak("background", v)}
          />
        </window.TweakSection>
        <window.TweakSection title="Demo">
          <window.TweakToggle
            label="Lompat langsung ke hasil"
            value={t.demoMode}
            onChange={v => setTweak("demoMode", v)}
          />
          <window.TweakButton onClick={() => { setView("welcome"); setHomeData(null); setParams(null); }}>
            Reset ke welcome
          </window.TweakButton>
        </window.TweakSection>
      </window.TweaksPanel>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App/>);
