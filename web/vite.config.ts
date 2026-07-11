import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-time proxy: the Vite dev server forwards API calls to the FastAPI
// building on :8000, so the browser sees ONE origin in dev and prod alike.
const proxy = Object.fromEntries(
  ["/health", "/query", "/ingest", "/conversations", "/chat", "/agent"].map((p) => [
    p,
    { target: "http://localhost:8000", changeOrigin: true },
  ]),
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
});
