import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev mode: vite on :5173 proxies API calls to uvicorn on :8000.
// Prod mode: the built SPA is copied into backend/app/static and served
// by FastAPI itself, so same-origin fetch paths work without config.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/process": "http://localhost:8000",
      "/status": "http://localhost:8000",
      "/result": "http://localhost:8000",
      "/preview": "http://localhost:8000",
    },
  },
});
