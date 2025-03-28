#!/bin/bash
#
# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.


OUTPUT_FILE=$1
HDR_FILE=$(realpath $2)

ARCH=$(uname -m)

CPUDIR=$(dirname $HDR_FILE)
# echo "CPUDIR: $CPUDIR"
SRCDIR=$(echo "$CPUDIR" | cut -d'/' -f-$(($(echo "$CPUDIR" | awk -F'/' '{print NF}')-3)))
# echo "SRCDIR: $SRCDIR"


read -r -d '' INCLUDES <<- EOM
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#include <arm_fp16.h>
#endif
EOM
CC_FLAGS="-arch ${ARCH} -nobuiltininc -nostdinc"

CONTENT=$(clang $CC_FLAGS -I "$SRCDIR" -E -P "$HDR_FILE" 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
const char* get_kernel_preamble() {
return R"preamble(
$INCLUDES
$CONTENT
using namespace mlx::core;
using namespace mlx::core::detail;
)preamble";
}
EOF
