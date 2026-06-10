#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

#if defined(__CUDA_NO_HALF_OPERATORS__)
__host__ __device__ __forceinline__ __half operator+(__half a, __half b) {
  return __hadd(a, b);
}

__host__ __device__ __forceinline__ __half operator-(__half a, __half b) {
  return __hsub(a, b);
}

__host__ __device__ __forceinline__ __half operator-(__half x) {
  return __hsub(__float2half(0.0f), x);
}

__host__ __device__ __forceinline__ __half operator*(__half a, __half b) {
  return __hmul(a, b);
}

__host__ __device__ __forceinline__ __half operator/(__half a, __half b) {
  return __hdiv(a, b);
}
#endif

#if defined(__CUDA_NO_BFLOAT16_OPERATORS__)
__host__ __device__ __forceinline__ __nv_bfloat16 operator+(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hadd(a, b);
}

__host__ __device__ __forceinline__ __nv_bfloat16 operator-(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hsub(a, b);
}

__host__ __device__ __forceinline__ __nv_bfloat16 operator-(__nv_bfloat16 x) {
  return __hsub(__float2bfloat16(0.0f), x);
}

__host__ __device__ __forceinline__ __nv_bfloat16 operator*(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hmul(a, b);
}

__host__ __device__ __forceinline__ __nv_bfloat16 operator/(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hdiv(a, b);
}
#endif

namespace cufk {

template <typename T, typename X>
__host__ __device__ __forceinline__ T scalar(X x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(static_cast<float>(x));
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(static_cast<float>(x));
  } else {
    return static_cast<T>(x);
  }
}

template <typename T>
__host__ __device__ __forceinline__ auto to_real(T x) {
  if constexpr (std::is_same_v<T, double>) {
    return x;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else {
    return static_cast<float>(x);
  }
}

__device__ __forceinline__ float abs_real(float x) {
  return fabsf(x);
}

__device__ __forceinline__ double abs_real(double x) {
  return fabs(x);
}

__device__ __forceinline__ float sqrt_clamped(float x, float floor) {
  return sqrtf(fmaxf(x, floor));
}

__device__ __forceinline__ double sqrt_clamped(double x, double floor) {
  return sqrt(fmax(x, floor));
}

} // namespace cufk
