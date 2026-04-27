# Fast MNIST Web Demo

React 19 + Vite frontend for the Fast MNIST classifier.

## Run Locally

```sh
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://127.0.0.1:5173/`.

The app tries prediction in this order:

1. Native C++ HTTP backend at `VITE_API_BASE_URL` or `http://localhost:8080`.
2. Browser WASM artifacts from `web/public/wasm/`.
3. Browser-only JS template fallback, so the free static demo remains usable
   even when neither a backend nor staged WASM artifacts are available.

Use `Cmd+K` / `Ctrl+K` to open the animated command palette, load the sample
digit, jump between sections, toggle theme, or reset the canvas.

## Deploy

Use Vercel Hobby/free with this directory as the project root:

```sh
vercel deploy --prod --yes
```

Build command: `npm run build`. Output directory: `dist`. Leave
`VITE_API_BASE_URL` unset for a static, zero-cost deployment that relies on
WASM or the browser-only fallback.

## Checks

```sh
npm ci --no-audit --no-fund
npm run lint
npm run build
npm run format:check
npm run test:e2e
```
