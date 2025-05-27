#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/indices.hpp>
#include "mlx/mlx.h"

namespace mx = mlx::core;

template <typename T>
void testEql(const std::vector<T>& expected, const std::vector<T>& actual) {
  if (expected.size() != actual.size()) {
    std::cout << "Size mismatch: expected `" << expected.size()
              << "`, but received `" << actual.size() << "`" << std::endl;
    return;
  }
  bool matched = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cout << "Mismatch at index " << i << ": expected `" << expected[i]
                << "`, but received `" << actual[i] << "`" << std::endl;
      matched = false;
    }
  }
  if (matched)
    std::cout << "All elements match." << std::endl;
}

std::vector<mx::array> slice_func(const std::vector<mx::array>& inputs) {
  std::vector<int32_t> start = {0, 1};
  std::vector<int32_t> stop = {2, 5};
  std::vector<int32_t> strides = {1, 2};
  return std::vector<mx::array>{mx::slice(inputs[0], start, stop, strides)};
}

void testSlice() {
  auto operand = mx::array(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, std::vector<int32_t>{2, 5}, mx::float32);

  std::vector<float> expected = {1, 3, 6, 8};
  mx::array res_array = mx::compile(slice_func)({operand})[0];
  testEql<float>(
      expected,
      std::vector<float>(
          res_array.data<float>(), res_array.data<float>() + res_array.size()));
}

int main() {
  std::cout << "Running debug test fails..." << std::endl;
  // ZML slice test failure
  auto operand = mx::array(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, std::vector<int32_t>{2, 5}, mx::float32);

  std::vector<float> expected = {1, 3, 6, 8};
  mx::array res_array = mx::compile(slice_func)({operand})[0];
  res_array = mx::contiguous(res_array);
  mx::eval(res_array);

  std::cout << res_array << std::endl;
  testEql<float>(
      expected,
      std::vector<float>(
          res_array.data<float>(), res_array.data<float>() + res_array.size()));
  std::cout << "Debug test completed." << std::endl;
  return 0;
}