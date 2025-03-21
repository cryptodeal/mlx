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

// ZML supports up to 8 dimensions
auto index_space(const std::vector<int32_t>& dims) {
  using namespace ranges;
  // `MAX_RANK` for zml is 8
  std::vector<int32_t> used_dims(8, 1);
  std::copy(dims.begin(), dims.end(), used_dims.begin());
  return views::cartesian_product(
      views::indices(used_dims[0]),
      views::indices(used_dims[1]),
      views::indices(used_dims[2]),
      views::indices(used_dims[3]),
      views::indices(used_dims[4]),
      views::indices(used_dims[5]),
      views::indices(used_dims[6]),
      views::indices(used_dims[7]));
}

template <
    class Tuple,
    class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple) {
  return std::apply(
      [](auto&&... elems) {
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
      },
      std::forward<Tuple>(tuple));
}

template <
    class Tuple,
    class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple, size_t size) {
  std::vector<T> result = to_vector(tuple);
  result.resize(size);
  return result;
}

void printVector(
    const std::string& name,
    const std::vector<int32_t>& vec,
    bool indent = false) {
  if (indent) {
    std::cout << "\t";
  }
  std::cout << name.c_str() << ": { ";
  for (auto i = 0; i < vec.size(); i++) {
    std::cout << std::to_string(vec[i]);
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << " }" << std::endl;
}

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
  // Adjust for negative dimension
  int adjusted_dimension = dimension >= 0
      ? dimension
      : static_cast<int>(inputs[0].ndim()) + dimension;

  std::vector<mx::array> results = inputs;
  std::vector<mx::array> input_slices(inputs.size(), mx::array({}));
  std::vector<mx::array> comparator_args(inputs.size() * 2, mx::array({}));

  for (auto i = 0; i < inputs.size(); i++) {
    std::cout << "inputs[" << std::to_string(i) << "]: " << inputs[i]
              << std::endl;
  }

  for (const auto result_index_tuple : index_space(results[0].shape())) {
    std::vector<int32_t> result_index =
        to_vector(result_index_tuple, static_cast<size_t>(results[0].ndim()));
    std::vector<int32_t> result_slice_start = result_index;
    result_slice_start[adjusted_dimension] = 0;
    std::vector<int32_t> result_slice_stop = result_index;
    result_slice_stop[adjusted_dimension] =
        results[0].shape(adjusted_dimension);
    for (auto& d : result_slice_stop)
      d += 1;
    for (auto i = 0; i < input_slices.size(); i++) {
      input_slices[i] =
          mx::slice(inputs[i], result_slice_start, result_slice_stop);
      std::cout << "input_slices[" << std::to_string(i)
                << "]: " << input_slices[i] << std::endl;
      auto flattened_input_slice = mx::flatten(input_slices[i]);
      std::cout << "flattened_input_slice: " << flattened_input_slice
                << std::endl;
      comparator_args[i * 2] = mx::slice(flattened_input_slice, {0}, {1});
      comparator_args[i * 2 + 1] = mx::slice(flattened_input_slice, {1}, {2});
    }

    auto predicate = comparator_args[0] > comparator_args[1];
    std::cout << "predicate: " << predicate << std::endl;

    if (!predicate.item<bool>()) {
      std::vector<int32_t> scatter_indices_(input_slices[0].size());
      std::iota(scatter_indices_.begin(), scatter_indices_.end(), 0);
      for (auto i = 0; i < scatter_indices_.size(); i += 2) {
        std::iter_swap(
            scatter_indices_.begin() + i, scatter_indices_.begin() + i + 1);
      }
      auto scatter_indices = mx::array(
          scatter_indices_.data(),
          {static_cast<int>(scatter_indices_.size())},
          mx::int32);
      for (auto i = 0; i < results.size(); i++) {
        auto flattened_input_slice = mx::flatten(input_slices[i]);
        auto sorted_slice = mx::reshape(
            mx::scatter(
                flattened_input_slice,
                scatter_indices,
                mx::expand_dims(flattened_input_slice, -1),
                0),
            input_slices[i].shape());
        std::cout << std::to_string(i) << ": sorted_slice = " << sorted_slice
                  << std::endl;

        results[i] = mx::slice_update(
            results[i], sorted_slice, result_slice_start, result_slice_stop);
        mx::eval(results[i]);
      }
    }
  }

  return results;
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
    std::cout << r << std::endl;
  }
}