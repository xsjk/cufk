#pragma once

#include "affine.cuh"

#include <cub/cub.cuh>
#include <cuda/std/functional>

#include <type_traits>

namespace cufk::scan {

using affine::AffineT;

constexpr int kWarpSize = 32;
template <typename T>
static __device__ __forceinline__ AffineT<T> shfl_up(const AffineT<T>& x, int offset) {
  constexpr unsigned mask = 0xffffffffu;
  AffineT<T> y;
  unroll for (int row = 0; row < 3; ++row) {
    unroll for (int col = 0; col < 3; ++col) {
      y.m[row][col] = __shfl_up_sync(mask, x.m[row][col], offset);
    }
  }
  unroll for (int row = 0; row < 3; ++row) {
    y.v[row] = __shfl_up_sync(mask, x.v[row], offset);
  }
  return y;
}

template <typename T>
static __device__ __forceinline__ AffineT<T> shfl(const AffineT<T>& x, int lane) {
  constexpr unsigned mask = 0xffffffffu;
  AffineT<T> y;
  unroll for (int row = 0; row < 3; ++row) {
    unroll for (int col = 0; col < 3; ++col) {
      y.m[row][col] = __shfl_sync(mask, x.m[row][col], lane);
    }
  }
  unroll for (int row = 0; row < 3; ++row) {
    y.v[row] = __shfl_sync(mask, x.v[row], lane);
  }
  return y;
}

template <typename T, typename Op>
static __device__ __forceinline__ T warp_inclusive(T x, Op op) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  unroll for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const T y = shfl_up(x, offset);
    if (lane >= offset) {
      x = op(y, x);
    }
  }
  return x;
}

template <typename T, int BlockThreads, typename Op>
static __device__ __forceinline__ T one_sync_carry(T thread_total, Op op, T identity) {
  static_assert(BlockThreads % kWarpSize == 0);
  constexpr int warps = BlockThreads / kWarpSize;
  __shared__ T warp_prefix[warps];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const T warp_value = warp_inclusive(thread_total, op);
  if (lane == kWarpSize - 1) {
    warp_prefix[warp] = warp_value;
  }
  __syncthreads();

  T warp_carry = identity;
  if (warp > 0) {
    if (lane == 0) {
      unroll for (int prev_warp = 0; prev_warp < warps; ++prev_warp) {
        if (prev_warp < warp) {
          warp_carry = op(warp_carry, warp_prefix[prev_warp]);
        }
      }
    }
    warp_carry = shfl(warp_carry, 0);
  }

  const T prev_thread_prefix = shfl_up(warp_value, 1);
  T carry = lane == 0 ? identity : prev_thread_prefix;
  if (warp > 0) {
    carry = lane == 0 ? warp_carry : op(warp_carry, carry);
  }
  return carry;
}

template <typename T, int BlockThreads, int ItemsPerThread, typename Op>
static __device__ __forceinline__ void custom_inclusive_one_sync(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
  static_assert(ItemsPerThread > 0);

  output[0] = input[0];
  unroll for (int item = 1; item < ItemsPerThread; ++item) {
    output[item] = op(output[item - 1], input[item]);
  }

  const T carry = one_sync_carry<T, BlockThreads>(output[ItemsPerThread - 1], op, identity);
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    output[item] = op(carry, output[item]);
  }
}

template <typename T, int BlockThreads, int ItemsPerThread, typename Op>
static __device__ __forceinline__ void custom_inclusive_two_sync(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
  static_assert(BlockThreads % kWarpSize == 0);
  static_assert(ItemsPerThread > 0);
  constexpr int warps = BlockThreads / kWarpSize;
  __shared__ T warp_prefix[warps];

  output[0] = input[0];
  unroll for (int item = 1; item < ItemsPerThread; ++item) {
    output[item] = op(output[item - 1], input[item]);
  }

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const T thread_prefix = warp_inclusive(output[ItemsPerThread - 1], op);
  if (lane == kWarpSize - 1) {
    warp_prefix[warp] = thread_prefix;
  }
  __syncthreads();

  if (warp == 0) {
    T warp_value = lane < warps ? warp_prefix[lane] : identity;
    warp_value = warp_inclusive(warp_value, op);
    if (lane < warps) {
      warp_prefix[lane] = warp_value;
    }
  }
  __syncthreads();

  const T prev_thread_prefix = shfl_up(thread_prefix, 1);
  T carry = lane == 0 ? identity : prev_thread_prefix;
  if (warp > 0) {
    carry = op(warp_prefix[warp - 1], carry);
  }
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    output[item] = op(carry, output[item]);
  }
}

template <typename T, int BlockThreads, int ItemsPerThread, cub::BlockScanAlgorithm Algorithm, typename Op>
static __device__ __forceinline__ void cub_inclusive(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op) {
  using BlockScan = cub::BlockScan<T, BlockThreads, Algorithm>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  BlockScan(scan_storage).InclusiveScan(input, output, op);
  __syncthreads();
}

// After this forward-only scan, prefix[item].v is the position prefix; prefix[item].m is scratch.
template <typename T, int BlockThreads, int ItemsPerThread>
static __device__ __forceinline__ void forward_positions_in_place(AffineT<T> (&prefix)[ItemsPerThread]) {
  if constexpr (std::is_same_v<T, double>) {
    AffineT<T> output[ItemsPerThread];
    cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_RAKING>(prefix, output, cuda::std::multiplies<>{});
    unroll for (int item = 0; item < ItemsPerThread; ++item) {
      prefix[item].v = output[item].v;
    }
  } else {
    static_assert(ItemsPerThread > 0);

    unroll for (int item = 1; item < ItemsPerThread; ++item) {
      prefix[item] = prefix[item - 1] * prefix[item];
    }

    const AffineT<T> carry = one_sync_carry<AffineT<T>, BlockThreads>(prefix[ItemsPerThread - 1], cuda::std::multiplies<>{}, AffineT<T>::identity());
    unroll for (int item = 0; item < ItemsPerThread; ++item) {
      prefix[item].v = carry * prefix[item].v;
    }
  }
}

template <typename T, int ItemsPerThread>
consteval bool uses_warp_cub_in_prefix_affine() {
  return std::is_same_v<T, double> && ItemsPerThread == 8;
}

template <typename T, int ItemsPerThread>
consteval bool uses_custom_in_prefix_affine() {
  return !std::is_same_v<T, double> && (ItemsPerThread == 2 || ItemsPerThread == 4);
}

template <typename T, int ItemsPerThread>
consteval bool uses_warp_cub_in_suffix_affine() {
  return (std::is_same_v<T, __half> && ItemsPerThread == 4) || (std::is_same_v<T, double> && ItemsPerThread == 8);
}

template <typename T, int ItemsPerThread>
consteval bool uses_custom_in_suffix_affine() {
  return (!std::is_same_v<T, double> && ItemsPerThread == 2) || (std::is_same_v<T, float> && ItemsPerThread == 4);
}

template <typename T, int BlockThreads, int ItemsPerThread, typename Op>
static __device__ __forceinline__ void prefix_path_affine(
    AffineT<T> (&input)[ItemsPerThread], AffineT<T> (&output)[ItemsPerThread], Op op, const AffineT<T>& identity) {
  if constexpr (uses_warp_cub_in_prefix_affine<T, ItemsPerThread>()) {
    cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_WARP_SCANS>(input, output, op);
  } else if constexpr (uses_custom_in_prefix_affine<T, ItemsPerThread>()) {
    custom_inclusive_one_sync<AffineT<T>, BlockThreads, ItemsPerThread>(input, output, op, identity);
  } else {
    cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_RAKING>(input, output, op);
  }
}

template <typename T, int BlockThreads, int ItemsPerThread>
static __device__ __forceinline__ void saved_prefix_affine(AffineT<T> (&input)[ItemsPerThread], AffineT<T> (&output)[ItemsPerThread]) {
  prefix_path_affine<T, BlockThreads, ItemsPerThread>(input, output, cuda::std::multiplies<>{}, AffineT<T>::identity());
}

template <typename T, int BlockThreads, int ItemsPerThread>
static __device__ __forceinline__ void suffix_affine(AffineT<T> (&input)[ItemsPerThread], AffineT<T> (&output)[ItemsPerThread]) {
  if constexpr (uses_warp_cub_in_suffix_affine<T, ItemsPerThread>()) {
    cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_WARP_SCANS>(input, output, cuda::std::plus<>{});
  } else if constexpr (uses_custom_in_suffix_affine<T, ItemsPerThread>()) {
    custom_inclusive_two_sync<AffineT<T>, BlockThreads, ItemsPerThread>(input, output, cuda::std::plus<>{}, AffineT<T>::zero());
  } else {
    cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_RAKING>(input, output, cuda::std::plus<>{});
  }
}

} // namespace cufk::scan
