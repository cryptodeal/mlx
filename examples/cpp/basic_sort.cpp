#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

/**
 * TODO (@cyptodeal): Implement `stablehlo::SortOp`
 *
 * N.B. We're hardcoding sort `LT` for testing purpose.
 *
 * Given the following Inputs:
 * - `inputs`: std::vector<mx::array>
 * - `dimension`: int
 * - `is_stable`: bool
 * - `comparator`: std::function (for testing, we're just hardcoding
 * less than)
 */
std::vector<mx::array> stablehlo_sort(
    const std::vector<mx::array>& inputs,
    int64_t dimension,
    bool is_stable) {
  auto indices_order = mx::argsort(
      inputs[0], static_cast<int>(dimension), mx::ComparatorType::GreaterThan);
  std::cout << indices_order << std::endl << std::endl;
  std::vector<mx::array> res(inputs.size(), mx::array({}));
  std::vector<int32_t> slice_sizes(inputs[0].ndim(), 1);
  for (size_t i = 0; i < inputs.size(); i++) {
    res[i] = mx::take_along_axis(
        inputs[i], indices_order, static_cast<int>(dimension));
  }
  return res;
}

int main() {
  std::vector<mx::array> inputs = {
      mx::array({1, 2, 3, 3, 2, 1}, {2, 3}, mx::int32),
      mx::array({3, 2, 1, 1, 2, 3}, {2, 3}, mx::int32),
  };
  int dimension = 0;
  bool is_stable = true;
  auto res = stablehlo_sort(inputs, dimension, is_stable);
  for (auto& r : res) {
    std::cout << r << std::endl << std::endl;
  }
}