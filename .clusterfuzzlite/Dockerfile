FROM gcr.io/oss-fuzz-base/base-builder:v1

COPY . $SRC/fast-mnist-nn
WORKDIR $SRC/fast-mnist-nn
COPY .clusterfuzzlite/build.sh $SRC/build.sh
