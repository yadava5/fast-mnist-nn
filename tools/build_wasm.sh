#!/usr/bin/env bash
#
# Build the WebAssembly inference path and stage its artifacts under
# web/public/wasm/. Expects emsdk to be already activated in the
# current shell:
#
#   source "$EMSDK_ROOT/emsdk_env.sh"
#   ./tools/build_wasm.sh
#
# Outputs:
#   web/public/wasm/fast_mnist.js
#   web/public/wasm/fast_mnist.wasm
#   web/public/wasm/model.weights.bin   (if a native build exists)
#
set -euo pipefail

if [ -z "${EMSDK:-}" ]; then
  echo "Run: source \"\$EMSDK_ROOT/emsdk_env.sh\"  (activate emsdk first)" >&2
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

BUILD_DIR="build-wasm"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

mkdir -p web/public/wasm
cp "$BUILD_DIR/fast_mnist.js" "$BUILD_DIR/fast_mnist.wasm" web/public/wasm/

# Generate the binary weights file if a native build with
# export_weights is available. The WASM toolchain cannot run the
# exporter itself, so we assume the user has also built a regular
# native build. Try the conventional build dirs in priority order.
for NATIVE_DIR in build build-release build-debug; do
  if [ -x "$NATIVE_DIR/export_weights" ]; then
    echo "Generating binary weights via $NATIVE_DIR/export_weights"
    "$NATIVE_DIR/export_weights" model.weights web/public/wasm/model.weights.bin
    break
  fi
done

if [ ! -f web/public/wasm/model.weights.bin ]; then
  echo "warning: no native export_weights found; skipping weights export." >&2
  echo "  Build the native target first:  cmake -S . -B build && cmake --build build" >&2
fi

echo "WASM artifacts staged in web/public/wasm/"
ls -la web/public/wasm/
