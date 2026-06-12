#pragma once

#include "math.cuh"

#include <cub/cub.cuh>
#include <cuda/std/bit>

#include <cstdint>
#include <type_traits>

namespace cufk::scan {

constexpr int kWarpSize = 32;

struct CustomScanBackend {};
struct CubScanBackend {};

template <typename T>
struct WordPack {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(sizeof(T) % sizeof(uint32_t) == 0);
  static constexpr int count = sizeof(T) / sizeof(uint32_t);
  uint32_t word[count]{};
};

template <typename T>
static __device__ __forceinline__ T bitwise_shfl_up(const T& value, int offset) {
  WordPack<T> pack = cuda::std::bit_cast<WordPack<T>>(value);
  unroll for (int i = 0; i < WordPack<T>::count; ++i) {
    pack.word[i] = __shfl_up_sync(0xffffffffu, pack.word[i], offset);
  }
  return cuda::std::bit_cast<T>(pack);
}

template <typename T>
static __device__ __forceinline__ T bitwise_shfl(const T& value, int lane) {
  WordPack<T> pack = cuda::std::bit_cast<WordPack<T>>(value);
  unroll for (int i = 0; i < WordPack<T>::count; ++i) {
    pack.word[i] = __shfl_sync(0xffffffffu, pack.word[i], lane);
  }
  return cuda::std::bit_cast<T>(pack);
}

template <typename T, typename Op>
static __device__ __forceinline__ T warp_inclusive(T x, Op op) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  unroll for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const T y = bitwise_shfl_up(x, offset);
    if (lane >= offset) {
      x = op(y, x);
    }
  }
  return x;
}

template <typename T, int ItemsPerThread, typename Op>
static __device__ __forceinline__ void thread_inclusive(
    T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op) {
  static_assert(ItemsPerThread > 0);
  output[0] = input[0];
  unroll for (int item = 1; item < ItemsPerThread; ++item) {
    output[item] = op(output[item - 1], input[item]);
  }
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
    warp_carry = bitwise_shfl(warp_carry, 0);
  }

  const T prev_thread_prefix = bitwise_shfl_up(warp_value, 1);
  T carry = lane == 0 ? identity : prev_thread_prefix;
  if (warp > 0) {
    carry = lane == 0 ? warp_carry : op(warp_carry, carry);
  }
  return carry;
}

template <typename T, int BlockThreads, int ItemsPerThread, typename Op>
static __device__ __forceinline__ T inclusive_scan_carry(
    T (&input)[ItemsPerThread],
    T (&output)[ItemsPerThread],
    Op op,
    const T& identity) {
  thread_inclusive<T, ItemsPerThread>(input, output, op);
  return one_sync_carry<T, BlockThreads>(
      output[ItemsPerThread - 1], op, identity);
}

template <typename T, int BlockThreads, int ItemsPerThread, typename Op>
static __device__ __forceinline__ void custom_inclusive_one_sync(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
  const T carry =
      inclusive_scan_carry<T, BlockThreads, ItemsPerThread>(input, output, op,
                                                            identity);
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

  thread_inclusive<T, ItemsPerThread>(input, output, op);

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

  const T prev_thread_prefix = bitwise_shfl_up(thread_prefix, 1);
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
  using CubScan = cub::BlockScan<T, BlockThreads, Algorithm>;
  __shared__ typename CubScan::TempStorage scan_storage;
  CubScan(scan_storage).InclusiveScan(input, output, op);
  __syncthreads();
}

template <typename Backend, typename T, int BlockThreads>
struct BlockScanImpl;

template <typename T, int BlockThreads>
struct BlockScanImpl<CustomScanBackend, T, BlockThreads> {
  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ void InclusiveScan(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
    custom_inclusive_two_sync<T, BlockThreads, ItemsPerThread>(input, output, op, identity);
  }

  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ void InclusiveScanOneSync(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
    custom_inclusive_one_sync<T, BlockThreads, ItemsPerThread>(input, output, op, identity);
  }

  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ T InclusiveScanCarry(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
    return inclusive_scan_carry<T, BlockThreads, ItemsPerThread>(input, output, op, identity);
  }
};

template <typename T, int BlockThreads>
struct BlockScanImpl<CubScanBackend, T, BlockThreads> {
  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ void InclusiveScan(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T&) {
    cub_inclusive<T, BlockThreads, ItemsPerThread, cub::BLOCK_SCAN_RAKING>(
        input, output, op);
  }

  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ void InclusiveScanOneSync(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
    InclusiveScan(input, output, op, identity);
  }

  template <int ItemsPerThread, typename Op>
  static __device__ __forceinline__ T InclusiveScanCarry(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread], Op op, const T& identity) {
    thread_inclusive<T, ItemsPerThread>(input, output, op);
    using CubScan = cub::BlockScan<T, BlockThreads, cub::BLOCK_SCAN_RAKING>;
    __shared__ typename CubScan::TempStorage scan_storage;
    T carry;
    CubScan(scan_storage).ExclusiveScan(
        output[ItemsPerThread - 1], carry, identity, op);
    __syncthreads();
    return carry;
  }
};

template <typename Backend, typename T, int BlockThreads>
struct BlockScan
    : BlockScanImpl<Backend, T, BlockThreads> {};

} // namespace cufk::scan
