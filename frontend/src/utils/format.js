// Display formatters used across screens.

export function fmtRp(n) {
  return "Rp " + Number(n || 0).toLocaleString("id-ID");
}

export function fmtTime(mins) {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function fmtDur(mins) {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  if (h === 0) return `${m} min`;
  if (m === 0) return `${h} jam`;
  return `${h}j ${m}m`;
}

// Tiny inline markdown renderer: **bold** and *italic* only.
// Safe enough for LLM-generated narrative; no arbitrary HTML.
export function mdInline(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}
