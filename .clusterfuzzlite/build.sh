#!/bin/bash -eu

cd "$SRC/fast-mnist-nn"

"$CXX" $CXXFLAGS -std=c++17 -Iinclude \
  src/Matrix.cpp fuzz/matrix_stream_fuzzer.cpp \
  "$LIB_FUZZING_ENGINE" -o "$OUT/matrix_stream_fuzzer"
