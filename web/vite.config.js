import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target =
    env.TABLE_AGENT_API_URL ||
    env.VITE_API_PROXY_TARGET ||
    "http://127.0.0.1:3636";
  const apiKey = env.TABLE_AGENT_SERVICE_API_KEY;

  return {
    plugins: [react()],
    server: {
      port: 5172,
      proxy: {
        "/v1": {
          target,
          changeOrigin: true,
          configure(proxy) {
            if (!apiKey) return;
            proxy.on("proxyReq", (proxyRequest) => {
              proxyRequest.setHeader("X-API-Key", apiKey);
            });
          },
        },
      },
    },
  };
});
