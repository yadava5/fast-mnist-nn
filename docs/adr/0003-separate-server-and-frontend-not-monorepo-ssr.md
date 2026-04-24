# ADR-0003: Separate C++ server + React SPA, not a monorepo Next.js SSR app

- **Status:** Accepted
- **Date:** 2026-04-22
- **Deciders:** Ayush Yadav
- **Consulted:** Shree Chaturvedi
- **Informed:** contributors

## Context and problem statement

The web experience needs (a) a way to run inference against the trained
C++ network and (b) a frontend that draws digits, visualizes
activations, and renders a revolvable 3D architecture view. There are
several plausible shapes for that combination:

- A Next.js monorepo with the inference logic inlined via N-API bindings,
  rendered server-side.
- A Next.js app that calls out to a separately-deployed inference
  service.
- A static SPA that calls a standalone HTTP inference server.
- A static SPA that runs inference locally in the browser via WebAssembly.

The project already has a mature C++17 core. The question is how to
expose it to the web.

## Decision drivers

- Deploy independence — the frontend should ship on one cadence, the
  inference on another.
- Reproducibility of the inference pipeline — the hot path should stay
  C++ with the exact same binary we benchmark.
- Single-origin deploy cost — a hobby-scale hosted demo must be free or
  near-free to keep running.
- Offline / no-server story — the demo should degrade gracefully when
  no backend is reachable.
- Platform risk — we want to avoid nodes in the architecture that depend
  on a single fragile ecosystem (e.g. N-API).

## Considered options

1. **Two deployables:** a C++ HTTP server (`fast_mnist_server`, using
   cpp-httplib + nlohmann/json) and a Vite-built React SPA that calls
   it over `VITE_API_BASE_URL`. Plus a WASM fallback compiled from the
   same C++ core.
2. **Next.js monorepo with N-API bindings** to call into the C++ core
   from Node, rendered SSR.
3. **C++ web framework** (Drogon, Crow) serving templated HTML directly.
4. **SPA with inference only in WASM**, no server.

## Decision

Option 1. Three artefacts, one source of truth:

- `fast_mnist_server` — the C++ inference binary. Same kernels, same
  serialization format. Deployed separately (Fly.io / Railway / Modal /
  any VPS).
- `web/` — a Vite-built React 19 SPA. Statically deployed on Vercel.
- WASM build — compiled from the same C++ core via Emscripten with
  `-msimd128`. Used as an offline fallback when `VITE_API_BASE_URL` is
  unreachable.

## Consequences

**Good**

- Single-purpose deployables. The server handles inference and nothing
  else; the frontend handles UI and nothing else. Each ships and scales
  on its own cadence.
- The C++ kernels stay a library and are consumed identically by three
  targets (CLI, HTTP, WASM). One source of truth for performance claims.
- Vercel hosts the SPA on its free tier. The inference server can go on
  any VPS, free-tier Fly.io, Modal, or Railway — or be omitted entirely
  in favour of the WASM fallback.
- Graceful degradation: when the server is cold or the user is offline,
  the same UI keeps working, just slower.
- No Node-in-the-hot-path. Benchmarks remain meaningful because the
  inference path is identical to the CLI path.

**Bad**

- Two deploy pipelines, two failure domains, two sets of logs to watch.
- CORS and environment-variable management across dev / preview /
  production environments. `VITE_API_BASE_URL` lives in three places
  (local `.env`, Vercel project config, hosted-demo overrides).
- Two separate version headers — the SPA could be talking to an older
  server. We mitigate with a `/health` endpoint that reports a build
  hash, but it is operational overhead.

## Alternatives considered and why they were rejected

- **Next.js monorepo with N-API bindings.** Would be one deploy, but
  N-API binaries are platform-specific, binding libraries are
  mid-maintained, and the resulting cross-compile story across Linux /
  macOS / Windows is a long tail of toolchain work. Rejected for
  fragility.
- **Next.js frontend + separate inference service.** Plausible; adds
  server-rendering machinery we do not need. The SPA rendering time is
  not a visible bottleneck; SSR adds complexity for no user-facing win.
- **C++ web framework serving templated HTML.** Good for a tech demo,
  bad for the interactive 3D visualization we want — Motion / r3f /
  Tailwind need a proper JS build. Also precludes Vercel-style hosting.
- **Pure WASM, no server.** Tempting, and we keep it as the fallback
  path. Rejected as the *primary* path because (a) the native server
  path is faster by a factor we want to showcase in benchmarks, and
  (b) the WASM binary adds 1–2 MiB of first-load payload the SPA should
  not pay up front.

## Validation

The SPA builds clean on Vercel preview. `fast_mnist_server` runs the
same Catch2 tests as the CLI. The WASM path is built in CI on every PR;
`VITE_API_BASE_URL` override is documented in
[`web/README.md`](../../web/README.md).
