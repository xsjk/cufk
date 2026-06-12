#pragma once

#include "affine.cuh"
#include "scan.cuh"

#include <cuda/std/functional>
#include <type_traits>

namespace cufk::affine_scan {

using affine::AffineT;

template <typename ScanBackend, typename T, int BlockThreads, int ItemsPerThread>
__device__ __forceinline__ void forward_positions_in_place(
    AffineT<T> (&input)[ItemsPerThread]) {
  if constexpr (std::is_same_v<T, double>) {
    AffineT<T> prefix[ItemsPerThread];
    cufk::scan::cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread,
                              cub::BLOCK_SCAN_RAKING>(
        input, prefix, cuda::std::multiplies<>{});
    unroll for (int i = 0; i < ItemsPerThread; ++i) { input[i].v = prefix[i].v; }
  } else {
    const auto carry =
        cufk::scan::BlockScan<ScanBackend, AffineT<T>,
                              BlockThreads>::InclusiveScanCarry(
            input, input, cuda::std::multiplies<>{},
            AffineT<T>::identity());
    unroll for (int i = 0; i < ItemsPerThread; ++i) {
      input[i].v = (carry * input[i]).v;
    }
  }
}

template <typename ScanBackend, typename T, int BlockThreads, int ItemsPerThread>
__device__ __forceinline__ void saved_prefix(
    AffineT<T> (&input)[ItemsPerThread], AffineT<T> (&output)[ItemsPerThread]) {
  if constexpr (std::is_same_v<T, double> && ItemsPerThread == 8) {
    cufk::scan::cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread,
                              cub::BLOCK_SCAN_WARP_SCANS>(
        input, output, cuda::std::multiplies<>{});
  } else if constexpr (!std::is_same_v<T, double> &&
                       (ItemsPerThread == 2 || ItemsPerThread == 4)) {
    cufk::scan::BlockScan<ScanBackend, AffineT<T>,
                          BlockThreads>::InclusiveScanOneSync(
        input, output, cuda::std::multiplies<>{}, AffineT<T>::identity());
  } else {
    cufk::scan::cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread,
                              cub::BLOCK_SCAN_RAKING>(
        input, output, cuda::std::multiplies<>{});
  }
}

template <typename ScanBackend, typename T, int BlockThreads, int ItemsPerThread>
__device__ __forceinline__ void suffix(
    AffineT<T> (&input)[ItemsPerThread], AffineT<T> (&output)[ItemsPerThread]) {
  if constexpr ((std::is_same_v<T, __half> && ItemsPerThread == 4) ||
                (std::is_same_v<T, double> && ItemsPerThread == 8)) {
    cufk::scan::cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread,
                              cub::BLOCK_SCAN_WARP_SCANS>(
        input, output, cuda::std::plus<>{});
  } else if constexpr ((!std::is_same_v<T, double> && ItemsPerThread == 2) ||
                       (std::is_same_v<T, float> && ItemsPerThread == 4)) {
    cufk::scan::BlockScan<ScanBackend, AffineT<T>,
                          BlockThreads>::InclusiveScan(
        input, output, cuda::std::plus<>{}, AffineT<T>::zero());
  } else {
    cufk::scan::cub_inclusive<AffineT<T>, BlockThreads, ItemsPerThread,
                              cub::BLOCK_SCAN_RAKING>(
        input, output, cuda::std::plus<>{});
  }
}

}  // namespace cufk::affine_scan
