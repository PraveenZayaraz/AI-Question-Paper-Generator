import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api are forwarded to the FastAPI server, so the browser
// talks to one origin and no CORS setup is needed in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
