#pragma once

#include "math.cuh"

namespace cufk::frame {

constexpr float kFrameEps = 1.0e-12f;

using cufk::abs_real;
using cufk::scalar;
using cufk::sqrt_clamped;
using cufk::to_real;
using cufk::math::cross;
using cufk::math::dot;
using cufk::math::Vec3T;

template <typename T>
static __device__ __forceinline__ Vec3T<T> normalized(const Vec3T<T>& v, float eps = kFrameEps) {
  const auto n2 = to_real(dot(v, v));
  return v / scalar<T>(sqrt_clamped(n2, static_cast<decltype(n2)>(eps)));
}

template <typename T>
static __device__ __forceinline__ Vec3T<T> stable_perpendicular(const Vec3T<T>& v) {
  Vec3T<T> helper = { scalar<T>(1.0f), scalar<T>(0.0f), scalar<T>(0.0f) };
  if (abs_real(to_real(v[0])) >= 0.9) {
    helper[0] = scalar<T>(0.0f);
    helper[1] = scalar<T>(1.0f);
  }
  return normalized(cross(v, helper));
}

} // namespace cufk::frame
