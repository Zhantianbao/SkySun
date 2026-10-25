// 本文件负责前端数据与组件中的 vite.config 相关逻辑。
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        // 开发代理转给后端前移除前端统一使用的 /api 前缀。
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: { environment: "jsdom", setupFiles: "./tests/setup.ts" },
});
