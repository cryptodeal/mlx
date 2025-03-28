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
 * TODO (@cyptodeal): Implement `stablehlo_scatter`, which maps
 * `stablehlo::GatherOp` -> `mx::gather`.
 *
 * N.B. We're hardcoding scatter addition for testing purpose.
 * Normally, we'd execute a function callback on the elements,
 * with the resulting value assigned to the corresponding index
 * in the output tensor,
 *
 * Given the following Inputs:
 * - `inputs`: std::vector<mx::array>
 * - `start_indices`: mx::array
 * - `updates`: std::vector<mx::array>
 * - `update_window_dims`: std::vector<int64_t>
 * - `inserted_window_dims`: std::vector<int64_t>
 * - `input_batching_dims`: std::vector<int64_t>
 * - `scatter_indices_batching_dims`: std::vector<int64_t>
 * - `scatter_dims_to_operand_dims`: std::vector<int64_t>
 * - `index_vector_dim`: int64_t
 * - `indices_are_sorted`: bool (maybe we implement?)
 * - `indices_are_unique`: bool (maybe we implement?)
 * - `update_computation`: std::function (for testing, we're just hardcoding
 * addition)
 *
 * Given we know:
 * - shape of `result`
 *
 * Use `mlx` to implement the `stablehlo::GatherOp` and return
 * a resulting `mx::array` containing the correct values in
 * the correct indices.
 */
std::vector<mx::array> stablehlo_scatter(
    const std::vector<mx::array>& inputs,
    const mx::array& scatter_indices,
    const std::vector<mx::array>& updates,
    const std::vector<int64_t>& update_window_dims,
    const std::vector<int64_t>& inserted_window_dims,
    const std::vector<int64_t>& input_batching_dims,
    const std::vector<int64_t>& scatter_indices_batching_dims,
    const std::vector<int64_t>& scatter_dims_to_operand_dims,
    int64_t index_vector_dim,
    bool indices_are_sorted,
    bool indices_are_unique,
    const std::vector<int32_t>& result_shape) {
  std::vector<mx::array> input_indices;
  std::vector<mx::array> update_indices;

  // iterate over updates[0] index space
  for (const auto update_index_tuple : index_space(updates[0].shape())) {
    std::vector<int32_t> update_index =
        to_vector(update_index_tuple, static_cast<size_t>(updates[0].ndim()));
    

    // Calculate update scatter dims
    std::vector<int32_t> update_scatter_dims;
    for (auto i = 0; i < updates[0].ndim(); i++) {
      if (std::find(
              update_window_dims.begin(),
              update_window_dims.end(),
              static_cast<int64_t>(i)) == update_window_dims.end()) {
        update_scatter_dims.emplace_back(static_cast<int32_t>(i));
      }
    }

    // Calculate update scatter index
    std::vector<int32_t> update_scatter_index(update_scatter_dims.size());
    for (auto i = 0; i < update_scatter_dims.size(); i++) {
      update_scatter_index[i] = update_index[update_scatter_dims[i]];
    }

    // Slice start index
    std::vector<int32_t> sin_start = update_scatter_index;
    std::vector<int32_t> sin_stop = update_scatter_index;
    if (index_vector_dim < scatter_indices.ndim()) {
      sin_start.insert(sin_start.begin() + index_vector_dim, 0);
      sin_stop.insert(
          sin_stop.begin() + index_vector_dim,
          scatter_indices.shape(index_vector_dim));
    }
    for (auto& d : sin_stop)
      d += 1;
    mx::array start_index =
        mx::flatten(mx::slice(scatter_indices, sin_start, sin_stop));

    // Compute full start index
    mx::array full_start_index =
        mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
    for (auto i = 0; i < scatter_dims_to_operand_dims.size(); i++) {
      auto d_input = static_cast<int32_t>(scatter_dims_to_operand_dims[i]);
      full_start_index = mx::slice_update(
          full_start_index,
          mx::slice(start_index, {i}, {i + 1}),
          {d_input},
          {d_input + 1});
    }

    // Compute full batching index
    mx::array full_batching_index =
        mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
    for (auto i = 0; i < input_batching_dims.size(); i++) {
      int32_t d_input = input_batching_dims[i];
      int32_t d_start = scatter_indices_batching_dims[i];
      full_batching_index = mx::slice_update(
          full_batching_index,
          mx::array({update_scatter_index
                         [d_start - (d_start < index_vector_dim ? 0 : 1)]}),
          {d_input},
          {d_input + 1});
    }

    // Compute update window index
    std::vector<int32_t> update_window_index(update_window_dims.size());
    for (auto i = 0; i < update_window_dims.size(); i++) {
      update_window_index[i] = update_index[update_window_dims[i]];
    }

    // Compute full window index
    mx::array full_window_index = mx::zeros(
        {static_cast<int32_t>(
            update_window_index.size() + inserted_window_dims.size() +
            input_batching_dims.size())},
        mx::int32);
    unsigned update_window_index_count = 0;
    for (int32_t i = 0; i < full_window_index.size(); i++) {
      if (std::find(
              inserted_window_dims.begin(), inserted_window_dims.end(), i) !=
              inserted_window_dims.end() ||
          std::find(
              input_batching_dims.begin(), input_batching_dims.end(), i) !=
              input_batching_dims.end()) {
        continue;
      }
      full_window_index = mx::slice_update(
          full_window_index,
          mx::array({update_window_index[update_window_index_count++]}),
          {i},
          {i + 1});
    }

    // Compute result index
    mx::array result_index =
        full_start_index + full_batching_index + full_window_index;
    

    // TODO(@cryptodeal): need to implement so that this can
    // be checked without calling `mx::eval` (or ensure zml prevents
    // this from occurring)
    // Continue if result index is out of bounds
    if (mx::sum(
            result_index >=
            mx::array(
                reinterpret_cast<const int32_t*>(result_shape.data()),
                {static_cast<int32_t>(result_shape.size())}))
            .item<int32_t>()) {
      continue;
    }
    input_indices.push_back(result_index);
    update_indices.push_back(
        mx::array(update_index.data(), {static_cast<int>(updates[0].ndim())}));
  }
  if (update_indices.empty()) {
    return inputs;
  }

  std::vector<int32_t> scatter_axes(inputs[0].ndim());
  std::iota(scatter_axes.begin(), scatter_axes.end(), 0);
  std::vector<int32_t> gather_axes(updates[0].ndim());
  std::iota(gather_axes.begin(), gather_axes.end(), 0);
  std::vector<int32_t> gather_slice_sizes(updates[0].ndim(), 1);
  std::vector<mx::array> res;
  auto result_indices =
      mx::split(mx::stack(input_indices, 1), input_indices[0].shape(0), 0);
  auto gather_indices =
      mx::split(mx::stack(update_indices, 1), update_indices[0].shape(0), 0);
  auto idx_shape = gather_indices[0].shape();
  printVector("gather_indices shape", idx_shape);
  std::vector<int32_t> update_shape(inputs[0].ndim() + idx_shape.size(), 1);
  for (auto i = 0; i < idx_shape.size(); ++i) {
    update_shape[i] = idx_shape[i];
  }
  for (auto i = 0; i < inputs.size(); ++i) {
    auto update_vals = mx::reshape(mx::gather(updates[i], gather_indices, gather_axes, gather_slice_sizes), update_shape);
    printVector("update_vals shape", update_vals.shape());
    res.push_back(mx::scatter_add(
        inputs[i],
        result_indices,
       update_vals,
        scatter_axes));
  }

  return res;
}

int main() {
  std::vector<mx::array> inputs = {
      mx::array(
          {
              1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
              33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
          },
          {2, 3, 4, 2},
          mx::int32),
  };

  auto scatter_indices = mx::array(
      {
          0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 9,
          0, 0, 2, 1, 2, 2, 1, 2, 0, 1, 1, 0,
      },
      {2, 2, 3, 2},
      mx::int32);

  std::vector<mx::array> updates = {
      mx::full({2, 2, 3, 2, 2}, mx::array(1, mx::int32)),
  };

  std::vector<int64_t> update_window_dims = {3, 4};
  std::vector<int64_t> inserted_window_dims = {1};
  std::vector<int64_t> input_batching_dims = {0};
  std::vector<int64_t> scatter_indices_batching_dims = {1};
  std::vector<int64_t> scatter_dims_to_operand_dims = {2, 1};
  int64_t index_vector_dim = 3;
  bool indices_are_sorted = false;
  bool unique_indices = false;
  std::vector<int32_t> result_shape = {2, 3, 4, 2};

  std::vector<mx::array> res = stablehlo_scatter(
      inputs,
      scatter_indices,
      updates,
      update_window_dims,
      inserted_window_dims,
      input_batching_dims,
      scatter_indices_batching_dims,
      scatter_dims_to_operand_dims,
      index_vector_dim,
      indices_are_sorted,
      unique_indices,
      result_shape);

  std::cout << res[0] << std::endl;
}