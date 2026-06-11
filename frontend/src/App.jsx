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

// Logo SVG component — map pin with compass orbits (matching new brand identity)
function BrandLogo({ size = 32 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 240 240"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Dark circular background */}
      <circle cx="120" cy="120" r="112" fill="#161a21" stroke="#262c37" strokeWidth="1.5" />

      {/* Glow behind pin */}
      <circle cx="120" cy="110" r="70" fill="url(#glow)" opacity=".35" />

      {/* Map pin body */}
      <path
        d="M120 42c-30.4 0-55 24.6-55 55 0 41.2 55 101 55 101s55-59.8 55-101c0-30.4-24.6-55-55-55z"
        fill="url(#pinGrad)"
        opacity=".9"
      />

      {/* Pin inner circle — dark cutout */}
      <circle cx="120" cy="95" r="26" fill="#161a21" />

      {/* Compass center dot + dashed ring */}
      <circle cx="120" cy="95" r="8" fill="#e8a04a" />
      <circle
        cx="120" cy="95" r="12"
        fill="none"
        stroke="#e8a04a"
        strokeWidth="1.5"
        strokeDasharray="3 3"
        opacity=".7"
      />

      {/* Compass spokes — jade color */}
      <line x1="120" y1="83" x2="120" y2="72" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      <line x1="132" y1="89" x2="141" y2="81" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      <line x1="132" y1="101" x2="141" y2="109" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      <line x1="120" y1="107" x2="120" y2="118" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      <line x1="108" y1="101" x2="99" y2="109" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      <line x1="108" y1="89" x2="99" y2="81" stroke="#5fb09b" strokeWidth="2" strokeLinecap="round" opacity=".8" />

      {/* Compass spoke endpoints */}
      <circle cx="120" cy="72" r="3" fill="#5fb09b" />
      <circle cx="141" cy="81" r="3" fill="#5fb09b" />
      <circle cx="141" cy="109" r="3" fill="#5fb09b" />
      <circle cx="120" cy="118" r="3" fill="#5fb09b" />
      <circle cx="99" cy="109" r="3" fill="#5fb09b" />
      <circle cx="99" cy="81" r="3" fill="#5fb09b" />

      {/* Route dots at bottom */}
      <path
        d="M90 185 Q105 170 120 175 Q135 180 150 165"
        stroke="#e8a04a"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
        strokeDasharray="6 4"
        opacity=".6"
      />
      <circle cx="90" cy="185" r="3.5" fill="#d9684e" />
      <circle cx="120" cy="175" r="3.5" fill="#e8a04a" />
      <circle cx="150" cy="165" r="3.5" fill="#5fb09b" />

      {/* Outer dashed orbit ring */}
      <ellipse
        cx="120" cy="120" rx="100" ry="100"
        fill="none"
        stroke="#f4ede2"
        strokeWidth=".6"
        strokeDasharray="4 6"
        opacity=".15"
      />

      <defs>
        <radialGradient id="glow" cx=".5" cy=".4" r=".6">
          <stop offset="0%" stopColor="#e8a04a" stopOpacity=".4" />
          <stop offset="100%" stopColor="#e8a04a" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="pinGrad" x1="120" y1="42" x2="120" y2="198" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#e8a04a" />
          <stop offset="100%" stopColor="#d9684e" />
        </linearGradient>
      </defs>
    </svg>
  );
}

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
          {/* New logo mark — round container with SVG icon */}
          <div className="brand-mark">
            <BrandLogo size={20} />
          </div>
          <div className="brand-name">
            GuideYou<span>&amp;AI</span>
          </div>
        </div>
        <div className="meta">
          <div className="row">
            <span className="dot" /> Multi-agent online
          </div>
          <div className="mono" style={{ fontSize: 12, color: "var(--ink-dim)" }}>
            Capstone Group 6 · v1.0
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
