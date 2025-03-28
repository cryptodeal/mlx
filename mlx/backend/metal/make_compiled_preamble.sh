#!/bin/bash
#
# This script generates a C++ function that provides the Metal unary and binary
# ops at runtime for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.

OUTPUT_FILE=$1
INPUT_FILE=$2
CFLAGS=$3

KERNEL_DIR=$(dirname $(realpath $INPUT_FILE))
# echo "KERNEL_DIR: $KERNEL_DIR"
SRC_DIR=$(echo "$KERNEL_DIR" | cut -d'/' -f-$(($(echo "$KERNEL_DIR" | awk -F'/' '{print NF}')-4)))
# echo "SRC_DIR: $SRC_DIR"
JIT_INCLUDES="$KERNEL_DIR/jit"
# echo "JIT_INCLUDES: $JIT_INCLUDES"

SRC_BASENAME=$(basename -- "${INPUT_FILE}")
SRC_NAME="${SRC_BASENAME%.*}"

CONTENT=$(clang -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT -E -P "$INPUT_FILE" $CFLAGS 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
$CONTENT
)preamble";
}

} // namespace mlx::core::metal
EOF
