#pragma once

#include "math.cuh"

#include <cstdint>

namespace cufk::rigid {

using cufk::scalar;
using math::cross;
using math::Vec3T;

template <typename T>
struct alignas(sizeof(uint32_t)) RigidFrameT {
  Vec3T<T> x;
  Vec3T<T> z;
  Vec3T<T> p;

  static __device__ __forceinline__ RigidFrameT identity() {
    return {
      .x = { scalar<T>(1.0f), scalar<T>(0.0f), scalar<T>(0.0f) },
      .z = { scalar<T>(0.0f), scalar<T>(0.0f), scalar<T>(1.0f) },
      .p = { scalar<T>(0.0f), scalar<T>(0.0f), scalar<T>(0.0f) },
    };
  }
};

template <typename T>
static __device__ __forceinline__ Vec3T<T> rigid_side(const RigidFrameT<T>& frame) {
  return cross(frame.z, frame.x);
}

template <typename T>
static __device__ __forceinline__ Vec3T<T> rigid_apply_with_side(const RigidFrameT<T>& frame, const Vec3T<T>& side, const Vec3T<T>& local) {
  return local[0] * frame.x + local[1] * side + local[2] * frame.z;
}

template <typename T>
static __device__ __forceinline__ RigidFrameT<T> operator*(const RigidFrameT<T>& parent, const RigidFrameT<T>& child) {
  const Vec3T<T> y = rigid_side(parent);
  return {
    .x = rigid_apply_with_side(parent, y, child.x),
    .z = rigid_apply_with_side(parent, y, child.z),
    .p = parent.p + rigid_apply_with_side(parent, y, child.p),
  };
}

} // namespace cufk::rigid
