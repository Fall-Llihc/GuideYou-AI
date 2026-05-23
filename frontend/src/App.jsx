import React, { useState } from "react";
import WelcomeScreen from "./components/WelcomeScreen";
import FormScreen from "./components/FormScreen";
import LoadingScreen from "./components/LoadingScreen";
import ResultsScreen from "./components/ResultsScreen";

/**
 * Top-level app: linear screen flow.
 *   welcome → form → loading → results
 *
 * `homeData` survives across welcome→form, `params` survives form→results,
 * and `result` (the API response) survives loading→results so a back-button
 * doesn't refetch.
 */
export default function App() {
  const [view, setView] = useState("welcome");
  const [homeData, setHomeData] = useState(null);
  const [params, setParams] = useState(null);
  const [result, setResult] = useState(null);

  const goForm = (data) => {
    setHomeData(data);
    setView("form");
  };

  const generate = (p) => {
    setParams(p);
    setResult(null);
    setView("loading");
  };

  const restart = () => {
    setView("welcome");
    setHomeData(null);
    setParams(null);
    setResult(null);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">B</div>
          <div className="brand-name">
            Bandung <span>AI Travel</span>
          </div>
        </div>
        <div className="meta">
          <div className="row">
            <span className="dot" /> Multi-agent online
          </div>
          <div className="mono" style={{ fontSize: 12, color: "var(--ink-dim)" }}>
            capstone · v1.0
          </div>
        </div>
      </header>

      {view === "welcome" && <WelcomeScreen onNext={goForm} />}

      {view === "form" && homeData && (
        <FormScreen
          homeData={homeData}
          onBack={() => setView("welcome")}
          onSubmit={generate}
        />
      )}

      {view === "loading" && params && (
        <LoadingScreen
          params={params}
          onDone={(data) => {
            setResult(data);
            setView("results");
          }}
          onError={() => setView("form")}
        />
      )}

      {view === "results" && params && result && (
        <ResultsScreen
          params={params}
          result={result}
          onRestart={restart}
          onEditParams={() => setView("form")}
        />
      )}
    </div>
  );
}
