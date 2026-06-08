import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["app/src/**/*.test.ts"],
    // web-txt2img is optional at runtime; skip suite when package is absent
    exclude: ["app/src/localImageGen.test.ts"],
  },
});
