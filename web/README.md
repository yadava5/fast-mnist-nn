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

## Checks

```sh
npm run lint
npm run build
```
