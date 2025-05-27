// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

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

// mx::array cartesianProduct(const mx::Shape& shape) {
//   std::vector<mx::array> arrays;
//   for (const auto& dim : shape) {
//     arrays.push_back(mx::arange(dim));
//   }
//   return mx::reshape(mx::stack(mx::meshgrid(arrays), -1), {-1,
//   static_cast<int>(shape.size())});
// }

mx::array stablehlo_gather(
    const mx::array& operand,
    const mx::array& start_indices,
    const std::vector<int32_t>& collapsed_slice_dims,
    int32_t index_vector_dim,
    const std::vector<int32_t>& offset_dims,
    const std::vector<int32_t>& operand_batching_dims,
    const std::vector<int32_t>& result_shape,
    const std::vector<int32_t>& slice_sizes,
    const std::vector<int32_t>& start_index_map,
    const std::vector<int32_t>& start_indices_batching_dims) {
  // Calculate batch dims
  std::vector<int32_t> batch_dims;
  for (int64_t i = 0; i < result_shape.size(); i++) {
    if (std::find(offset_dims.begin(), offset_dims.end(), i) ==
        offset_dims.end()) {
      batch_dims.emplace_back(static_cast<int32_t>(i));
    }
  }

  // Calculate result indices
  std::vector<mx::array> meshgrid_arrays;
  for (const auto& dim : result_shape) {
    meshgrid_arrays.push_back(mx::arange(dim));
  }
  mx::array result_indices = mx::reshape(
      mx::stack(mx::meshgrid(meshgrid_arrays), -1),
      {-1, static_cast<int>(result_shape.size())});

  std::cout << "result_indices: " << result_indices << std::endl;

  // Calculate batch indices
  mx::array batch_indices = mx::take(
      result_indices,
      mx::array(batch_dims.data(), {static_cast<int32_t>(batch_dims.size())}),
      1);
  std::cout << "batch_indices: " << batch_indices << std::endl;

  // Calculate start indices
  auto tmp_indices = mx::split(batch_indices, batch_indices.shape(1), 1);
  for (auto& idx : tmp_indices) {
    idx = mx::flatten(idx);
  }

  std::vector<int> tmp_axes;
  for (auto i = 0; i < start_indices.ndim(); ++i) {
    if (i == index_vector_dim)
      continue;
    tmp_axes.push_back(i);
  }
  std::vector<int32_t> tmp_slice_sizes(start_indices.ndim(), 1);
  tmp_slice_sizes[index_vector_dim] = start_indices.shape(index_vector_dim);
  mx::array batch_start_indices = mx::flatten(
      mx::gather(start_indices, tmp_indices, tmp_axes, tmp_slice_sizes), 1);
  std::cout << "batch_start_indices: " << batch_start_indices << std::endl;

  // Calculate full start indicees
  mx::array full_start_indices = mx::zeros(
      {batch_start_indices.shape(0), static_cast<int32_t>(operand.ndim())},
      mx::int32);
  std::vector<int32_t> max;
  for (auto d_start = 0; d_start < start_index_map.size(); ++d_start) {
    int d_operand = start_index_map[d_start];
    max.push_back(operand.shape(d_operand) - slice_sizes[d_operand]);
  }

  printVector("max", max, true);
  auto tmp_taken_vals = mx::take(
    batch_start_indices,
    mx::arange(static_cast<int>(start_index_map.size())),
    1);
    
  auto update_values = mx::clip(
    tmp_taken_vals,
      mx::array({0}),
      mx::array(max.data(), {static_cast<int32_t>(max.size())}));
  std::cout << "update_values: " << update_values << std::endl;
  full_start_indices = mx::put_along_axis(
      full_start_indices,
      mx::broadcast_to(
          mx::array(
              start_index_map.data(),
              {static_cast<int32_t>(start_index_map.size())}),
          {full_start_indices.shape(0),
           static_cast<int32_t>(start_index_map.size())}),
      update_values,
      1);

  std::cout << "full_start_indices: " << full_start_indices << std::endl;

  // Calculate full batching indices
  mx::array full_batching_indices = mx::zeros(
      {batch_start_indices.shape(0), static_cast<int32_t>(operand.ndim())},
      mx::int32);
  mx::array i_batching =
      mx::arange(static_cast<int>(operand_batching_dims.size()));
  mx::array tmp_batch_indices = mx::array(
      operand_batching_dims.data(),
      {static_cast<int32_t>(operand_batching_dims.size())});
  mx::array d_operands = mx::take(tmp_batch_indices, i_batching);
  mx::array d_starts = mx::take(tmp_batch_indices, i_batching);
  auto tmp_batch_start_indices = mx::subtract(
      d_starts,
      mx::where(
          mx::less(d_starts, mx::full(d_starts.shape(), index_vector_dim)),
          mx::zeros(d_starts.shape(), mx::int32),
          mx::ones(d_starts.shape(), mx::int32)));

  full_batching_indices = mx::put_along_axis(
      full_batching_indices,
      mx::broadcast_to(
          d_operands, {full_batching_indices.shape(0), d_operands.shape(0)}),
      mx::take(batch_indices, tmp_batch_start_indices, 1),
      1);
  std::cout << "full_batching_indices: " << full_batching_indices << std::endl;

  // Calculate offset indices
  mx::array offset_indices = mx::take(
      result_indices,
      mx::array(offset_dims.data(), {static_cast<int32_t>(offset_dims.size())}),
      1);
  std::cout << "offset_indices: " << offset_indices << std::endl;

  // Calculate full offset indices
  mx::array full_offset_indices =
      mx::zeros(full_batching_indices.shape(), mx::int32);
  std::vector<int32_t> result_offset_idx;

  for (unsigned i = 0; i < full_offset_indices.shape(1); i++) {
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
    result_offset_idx.push_back(i);
  }
  full_offset_indices = mx::put_along_axis(
      full_offset_indices,
      mx::broadcast_to(
          mx::array(
              result_offset_idx.data(),
              {static_cast<int32_t>(result_offset_idx.size())}),
          {full_offset_indices.shape(0),
           static_cast<int32_t>(result_offset_idx.size())}),
      offset_indices,
      1);
  std::cout << "full_offset_indices: " << full_offset_indices << std::endl;

  mx::array operand_indices =
      full_start_indices + full_batching_indices + full_offset_indices;
  std::cout << "operand_indices: " << operand_indices << std::endl;

  std::vector<int32_t> gather_axes(operand.ndim());
  std::iota(gather_axes.begin(), gather_axes.end(), 0);
  return mx::reshape(
      mx::gather(
          operand,
          mx::split(operand_indices, operand_indices.shape(1), 1),
          gather_axes,
          std::vector<int32_t>(operand.ndim(), 1)),
      result_shape);

  return mx::array({});
}

int main() {
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

  auto res = stablehlo_gather(
      operand,
      start_indices,
      {1}, // collapsed_slice_dims
      3, // index_vector_dim
      {3, 4}, // offset_dims
      {0}, // operand_batching_dims
      {2, 2, 3, 2, 2}, // result_shape
      {1, 1, 2, 2}, // slice_sizes
      {2, 1},
      {1});

  std::cout << "Gather result: " << res << std::endl;
}
