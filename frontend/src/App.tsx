import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export default function App() {
  const [health, setHealth] = useState<string>("checking…");

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((j) => setHealth(j.status ?? "unknown"))
      .catch(() => setHealth("unreachable"));
  }, []);

  return (
    <main className="app">
      <h1>Seestar Enhance</h1>
      <p>Backend: {health}</p>
      <p className="muted">UI placeholder. Pipeline coming soon.</p>
    </main>
  );
}
