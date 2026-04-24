# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 0 — repo hygiene.** Track-branch workflow on GitHub with `hygiene`, `frontend`, `backend`, `optimization` long-running branches; feature PRs target tracks, major merges target `main`.
- **Phase 0 — community health.** `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `.github/CODEOWNERS`, `.editorconfig`.
- **Phase 0 — templates.** `.github/ISSUE_TEMPLATE/` YAML forms (bug_report, feature_request, performance_regression, rfc, config) and `.github/pull_request_template.md`.
- **Phase 0 — supply chain.** GitHub Actions workflows for CodeQL (C++ + JS/TS), gitleaks secret scan, commitlint enforcement, Dependabot for npm + github-actions.
- **Phase 0 — commit discipline.** husky v9 + commitlint + lint-staged + prettier enforcement on commits.
- **Phase 1 — frontend foundations.** Motion v12, Tailwind v4 with OKLCH tokens, Geist fonts, `three` + `@react-three/fiber` + `@react-three/drei` with vendor chunk split, `cn` util, `VITE_API_BASE_URL` env support, `web/vercel.json` SPA rewrite.
- **Phase 1 — theme.** View Transitions API crossfade for dark/light toggle.
- **Phase 2 — backend activations.** `/predict` now returns `hidden_activations` + `input_grad` alongside the softmax distribution (see `apps/server.cpp`).
- **Phase 2 — 3D architecture hero.** Revolvable, scroll-linked `three`/`drei` scene rendering the 784 → 100 → 10 network live.
- **Phase 2 — canvas rewrite.** perfect-freehand SVG canvas with pressure support and undo/redo.
- **Phase 2 — real activation viz.** Saliency + heatmap + softmax components wired to the new `/predict` payload.
- **Phase 2 — paper-shaders hero backdrop.** Sticky-parallax pipeline for the landing surface.
- **Docs.** README rewritten for current project state; `BENCHMARKS.md` extracted with methodology + analysis; `docs/adr/` seeded with ADR-0001 (hand-rolled SIMD), ADR-0002 (two-layer MLP), ADR-0003 (split server + SPA).

### Changed
- `web/src/api/predict.ts` API base is now env-driven (`VITE_API_BASE_URL`) with `http://localhost:8080` fallback for dev.
- `web/src/App.tsx` wires the Phase 2 scaffolds (3D hero, canvas, activations) into the app shell.

## [1.0.0] - 2026-04-22

### Added
- Interactive React frontend at `web/` — draw a digit on a 280×280 canvas, see live predictions with 300ms debounce.
- Neural network visualisation with particle animation showing activations flowing through layers.
- Dark/light theme toggle with system preference detection and localStorage persistence.
- C++ HTTP API server (`apps/server.cpp`) exposing `/health` and `/predict` endpoints, 100-iteration timing comparison between baseline scalar and SIMD-optimized inference.

### Fixed
- ESLint error: removed `setState` in `useEffect` cycle.

### Infrastructure
- README expanded with web frontend docs; VSCode IntelliSense C++ config updated.

## [0.1.0] - 2025-12-26

### Added
- C++ core library: `Matrix` class with 64-byte aligned storage, padded leading dimension, blocked GEMM with 64×64 tiles, tiled 32×32 transpose, AXPY.
- SIMD kernels: hand-written AVX-512, AVX2, and NEON intrinsics with scalar fallback; FMA-aware where available.
- `NeuralNet` class with Xavier weight init, fused `gemv + bias + sigmoid` forward pass, in-place SGD updates via row-wise SIMD AXPY, text-format serialization.
- OpenMP parallelism on `dot`, `transpose`, `axpy` with element-count thresholds.
- CLI apps: `fast_mnist_cli` (trainer/evaluator with PGM cache), `fast_mnist_trainer` (saves `model.weights`), `fast_mnist_tests` (Catch2).
- Google Benchmark suite `benchmarks/bench_matrix.cpp` covering dot/transpose/axpy at 32–1024, plus `benchLearn` and `benchClassify` on 784→30→10.
- Python tooling: `tools/run.py` orchestrator, `tools/prepare_mnist.py` (MNIST IDX→PGM), `tools/run_benchmarks.py` (multi-config bench runner with hand-rolled SVG chart generation), `tools/bootstrap_macos.sh`.
- GitHub Actions CI on Linux/macOS/Windows plus frontend typecheck.
- Release workflow producing per-OS zip artifacts on `v*` tags.
- Published benchmark JSON runs + bench_summary.csv + light/dark SVG charts under `docs/benchmarks/`.
- Doxygen config + docs target.

[Unreleased]: https://github.com/yadava5/fast-mnist-nn/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yadava5/fast-mnist-nn/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/yadava5/fast-mnist-nn/releases/tag/v0.1.0
