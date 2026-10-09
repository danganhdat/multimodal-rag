import { createReadStream, existsSync, statSync } from "fs";
import { extname, join, resolve } from "path";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

function serveLocal(route: string, dir: string): Plugin {
  return {
    name: "serve-local-keyframes",
    configureServer(server) {
      server.middlewares.use(route, (req, res, next) => {
        const file = join(dir, decodeURIComponent(req.url || ""));
        try {
          if (existsSync(file) && statSync(file).isFile()) {
            const ext = extname(file).toLowerCase();
            const mime: Record<string, string> = {
              ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg",
              ".png": "image/png",
              ".webp": "image/webp",
            };
            res.setHeader("Content-Type", mime[ext] || "application/octet-stream");
            res.setHeader("Cache-Control", "public, max-age=86400");
            createReadStream(file).pipe(res);
            return;
          }
        } catch {}
        next();
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, resolve(__dirname, ".."), "");
  const dataRoot = env.DATA_ROOT || "..";
  const keyframesDir = join(dataRoot, "keyframes");

  return {
    plugins: [react(), serveLocal("/keyframes", keyframesDir)],
    server: {
      port: 3000,
      proxy: {
        "/api": {
          target: "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
