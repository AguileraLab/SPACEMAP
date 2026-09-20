import { defineConfig, loadEnv, UserConfig } from "vite"

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  const port = env.VITE_PORT ? parseInt(env.VITE_PORT) : 3001

  return {
    base: "./",
    server: {
      port,
    },
    build: {
      outDir: "build",
    },
  } satisfies UserConfig
})