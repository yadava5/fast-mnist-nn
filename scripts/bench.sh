#!/usr/bin/env bash
# Reproducibility harness for the Google Benchmark suite.
#
# Configures a Release build with OpenMP + -march=native, builds
# fast_mnist_benchmarks, and runs it with 10 repetitions per case
# writing aggregated JSON so the run can be diffed against a baseline
# (e.g. in a CodSpeed / bench-action workflow or by a human eye).
#
# Tries to pin the Linux CPU governor to "performance" before running
# so benchmark numbers are comparable across machines on the same
# kernel build. Silently skips that step on macOS / non-privileged
# shells -- the benchmarks still run, they are just noisier.
#
# Usage:
#   scripts/bench.sh                    # writes bench-<timestamp>.json
#   scripts/bench.sh out.json           # writes out.json
set -euo pipefail

OUT="${1:-bench-$(date +%Y%m%d-%H%M%S).json}"

# ----------------------------------------------------------------------
# Linux-only: best-effort CPU governor pin.
# ----------------------------------------------------------------------
if [[ "$(uname)" == "Linux" ]]; then
  if command -v cpupower >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
      sudo cpupower frequency-set -g performance >/dev/null 2>&1 \
        || echo "warn: could not set CPU governor" >&2
    else
      echo "warn: cpupower present but sudo is non-interactive; skipping" >&2
    fi
  fi
fi

# ----------------------------------------------------------------------
# Configure + build.
# ----------------------------------------------------------------------
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAST_MNIST_ENABLE_OPENMP=ON \
  -DFAST_MNIST_ENABLE_NATIVE=ON \
  -DFAST_MNIST_ENABLE_BENCHMARKS=ON

cmake --build build-bench --target fast_mnist_benchmarks --parallel

# ----------------------------------------------------------------------
# Run.
# ----------------------------------------------------------------------
# - 10 repetitions cuts noise and makes Welch's t-test downstream
#   meaningful.
# - aggregates_only=true collapses the 10 runs into mean/median/stddev
#   so the JSON stays compact.
# - json output is the format bench diff tools (googlebench_compare,
#   codspeed) accept directly.
./build-bench/fast_mnist_benchmarks \
  --benchmark_repetitions=10 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out_format=json \
  --benchmark_out="$OUT"

echo "wrote $OUT"
