import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    include: ['three', '@react-three/fiber'],
  },
  build: {
    rollupOptions: {
      output: {
        // Force three.js + r3f ecosystem into its own async chunk so the
        // initial bundle stays small until a canvas actually mounts.
        manualChunks: (id) => {
          if (
            id.includes('/node_modules/three/') ||
            id.includes('/node_modules/@react-three/fiber/') ||
            id.includes('/node_modules/@react-three/drei/')
          ) {
            return 'three-vendor'
          }
        },
      },
    },
  },
})
