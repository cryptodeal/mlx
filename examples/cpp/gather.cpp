#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/indices.hpp>
#include "mlx/mlx.h"

namespace mx = mlx::core;

// ZML supports up to 8 dimensions
auto ndIndex(const std::vector<int32_t>& dims) {
  using namespace ranges;
  return views::cartesian_product(
      views::indices(dims[0]),
      views::indices(dims[1]),
      views::indices(dims[2]),
      views::indices(dims[3]),
      views::indices(dims[4]),
      views::indices(dims[5]),
      views::indices(dims[6]),
      views::indices(dims[7]));
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

int32_t currentStartIndex(mx::array& index_scalar) {
  switch (index_scalar.dtype()) {
    case mx::int64:
      return static_cast<int32_t>(index_scalar.item<int64_t>());
    case mx::int16:
      return static_cast<int32_t>(index_scalar.item<int16_t>());
    case mx::int8:
      return static_cast<int32_t>(index_scalar.item<int8_t>());
    default:
      return index_scalar.item<int32_t>();
  }
}

/**
 * TODO (@cyptodeal): Implement `stablehlo_gather`, which maps
 * `stablehlo::GatherOp` -> `mx::gather`.
 *
 * Given the following Inputs:
 * - `operand`: mx::array
 * - `start_indices`: mx::array
 * - `offset_dims`: std::vector<int64_t>
 * - `collapsed_slice_dims`: std::vector<int64_t>
 * - `operand_batching_dims`: std::vector<int64_t>
 * - `start_indices_batching_dims`: std::vector<int64_t>
 * - `start_index_map`: std::vector<int64_t>
 * - `index_vector_dim`: int64_t
 * - `slice_sizes`: std::vector<int64_t>
 * - `indices_are_sorted`: bool (maybe we implement?)
 *
 * Given we know:
 * - shape of `result`
 *
 * Use `mlx` to implement the `stablehlo::GatherOp` and return
 * a resulting `mx::array` containing the correct values in
 * the correct indices.
 */

mx::array stablehlo_gather(
    const mx::array& operand,
    const mx::array& start_indices,
    const std::vector<int64_t>& offset_dims,
    const std::vector<int64_t>& collapsed_slice_dims,
    const std::vector<int64_t>& operand_batching_dims,
    const std::vector<int64_t>& start_indices_batching_dims,
    const std::vector<int64_t>& start_index_map,
    int64_t index_vector_dim,
    const std::vector<int64_t>& slice_sizes,
    bool indices_are_sorted,
    const std::vector<int32_t>& _result_shape) {
  // Calculate batch dims
  std::vector<int32_t> batch_dims;
  for (int64_t i = 0; i < _result_shape.size(); i++) {
    if (std::find(offset_dims.begin(), offset_dims.end(), i) ==
        offset_dims.end()) {
      batch_dims.emplace_back(static_cast<int32_t>(i));
    }
  }

  mx::array result = mx::zeros(_result_shape, operand.dtype());

  // iterate over result indices (fill w 1 and ignore those values)
  std::vector<int32_t> result_shape(8, 1);
  std::copy(_result_shape.begin(), _result_shape.end(), result_shape.begin());
  for (const auto result_index_tuple : ndIndex(result_shape)) {
    std::vector<int32_t> result_index =
        to_vector(result_index_tuple, _result_shape.size());

    result_index.resize(_result_shape.size());

    std::vector<int32_t> batch_index(batch_dims.size());
    for (unsigned i = 0; i < batch_dims.size(); i++) {
      batch_index[i] = result_index[batch_dims[i]];
    }

    // Extract start index for the current batch
    std::vector<int32_t> sin_start(start_indices.ndim());
    std::vector<int32_t> sin_stop(start_indices.ndim());
    unsigned batch_idx_count = 0;
    for (auto i = 0; i < start_indices.ndim(); i++) {
      if (index_vector_dim == static_cast<int64_t>(i)) {
        sin_start[i] = 0;
        sin_stop[i] = start_indices.shape(i) + 1;
        continue;
      }
      sin_start[i] = batch_index[batch_idx_count++];
      sin_stop[i] = sin_start[i] + 1;
    }
    mx::array start_index =
        mx::flatten(mx::slice(start_indices, sin_start, sin_stop));

    // Compute full start index
    std::vector<int32_t> full_start_index(operand.ndim(), 0);
    for (auto d_start = 0; d_start < start_index_map.size(); d_start++) {
      auto d_operand = start_index_map[d_start];
      auto index_scalar = mx::slice(start_index, {d_start}, {d_start + 1});
      full_start_index[d_operand] = std::clamp(
          currentStartIndex(index_scalar),
          0,
          operand.shape(d_operand) -
              static_cast<int32_t>(slice_sizes[d_operand]));
    }

    // Compute full batching index
    std::vector<int32_t> full_batching_index(operand.ndim(), 0);
    for (auto i_batching = 0; i_batching < operand_batching_dims.size();
         i_batching++) {
      auto d_operand = static_cast<int32_t>(operand_batching_dims[i_batching]);
      auto d_start =
          static_cast<int32_t>(start_indices_batching_dims[i_batching]);
      full_batching_index[d_operand] = batch_index
          [d_start -
           (static_cast<int64_t>(d_start) < index_vector_dim ? 0 : 1)];
    }

    // Compute offset index
    std::vector<int32_t> offset_index(offset_dims.size());
    for (unsigned i = 0; i < offset_dims.size(); i++) {
      offset_index[i] = result_index[offset_dims[i]];
    }

    // Compute full offset index
    std::vector<int32_t> full_offset_index(operand.ndim(), 0);
    unsigned offset_index_count = 0;
    for (unsigned i = 0; i < full_offset_index.size(); i++) {
      if (std::find(
              operand_batching_dims.begin(),
              operand_batching_dims.end(),
              static_cast<int64_t>(i)) != operand_batching_dims.end() ||
          std::find(
              collapsed_slice_dims.begin(),
              collapsed_slice_dims.end(),
              static_cast<int64_t>(i)) != collapsed_slice_dims.end()) {
        continue;
      }
      full_offset_index[i] = offset_index[offset_index_count++];
    }

    std::vector<int32_t> operand_index(operand.ndim());
    for (unsigned i = 0; i < operand.ndim(); i++) {
      operand_index[i] =
          full_start_index[i] + full_batching_index[i] + full_offset_index[i];
    }

    // slice gathered value
    std::vector<int32_t> operand_stop = operand_index;
    for (auto& d : operand_stop)
      d += 1;
    std::vector<int32_t> result_stop = result_index;
    for (auto& d : result_stop)
      d += 1;

    result = mx::slice_update(
        result,
        mx::slice(operand, operand_index, operand_stop),
        result_index,
        result_stop);
  }
  return result;
}

int main() {
  // testing ground for
  auto operand = mx::array(
      {
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
          33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
      },
      {2, 3, 4, 2},
      mx::int32);

  auto start_indices = mx::array(
      {
          0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 9,
          0, 0, 2, 1, 2, 2, 1, 2, 0, 1, 1, 0,
      },
      {2, 2, 3, 2},
      mx::int32);

  std::vector<int64_t> offset_dims = {3, 4};
  std::vector<int64_t> collapsed_slice_dims = {1};
  std::vector<int64_t> operand_batching_dims = {0};
  std::vector<int64_t> start_indices_batching_dims = {1};
  std::vector<int64_t> start_index_map = {2, 1};
  int64_t index_vector_dim = 3;
  std::vector<int64_t> slice_sizes = {1, 1, 2, 2};
  bool indices_are_sorted = false;
  // We can query `stablehlo::GatherOp` for the shape of the result
  std::vector<int32_t> result_shape = {2, 2, 3, 2, 2};

  mx::array res = stablehlo_gather(
      operand,
      start_indices,
      offset_dims,
      collapsed_slice_dims,
      operand_batching_dims,
      start_indices_batching_dims,
      start_index_map,
      index_vector_dim,
      slice_sizes,
      indices_are_sorted,
      result_shape);

  // auto res = mx::gather(operand, );
  mx::eval(res);
  std::cout << res << std::endl;
}