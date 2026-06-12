#pragma once

#include "math.cuh"

#include <cstddef>
#include <type_traits>

namespace cufk::affine {

using cufk::scalar;
using cufk::to_real;
using math::dot;
using math::Mat3T;
using math::outer;
using math::sin_cos;
using math::transpose_mul;
using math::Vec3T;

template <typename T>
struct alignas(sizeof(T) * 4) AffineT {
  Mat3T<T> m;
  Vec3T<T> v;

  template <typename R>
  __host__ __device__ __forceinline__ static Vec3T<T> local_direction(R s, R c, R S, R C) {
    return { scalar<T>(-c), scalar<T>(s * C), scalar<T>(s * S) };
  }

  __host__ __device__ __forceinline__ static AffineT identity() {
    return {
      .m = Mat3T<T>::identity(),
      .v = {},
    };
  }

  template <typename R>
  static __device__ __forceinline__ AffineT local_transform(T length, R s, R c, R S, R C) {
    return {
      .m = Mat3T<T>::from_rows(
          { scalar<T>(-c), scalar<T>(-s), scalar<T>(0.0f) },
          { scalar<T>(s * C), scalar<T>(-c * C), scalar<T>(-S) },
          { scalar<T>(s * S), scalar<T>(-c * S), scalar<T>(C) }),
      .v = length * local_direction(s, c, S, C),
    };
  }

  static __device__ __forceinline__ AffineT local_transform(T length, T theta, T phi) {
    const auto [s, c] = sin_cos(to_real(theta));
    const auto [S, C] = sin_cos(to_real(phi));
    return local_transform(length, s, c, S, C);
  }

  __host__ __device__ __forceinline__ AffineT operator*(const AffineT& b) const {
    return {
      .m = m * b.m,
      .v = v + m * b.v,
    };
  }

  __host__ __device__ __forceinline__ Vec3T<T> operator*(const Vec3T<T>& x) const {
    return v + m * x;
  }

  __host__ __device__ __forceinline__ static AffineT zero() {
    return {};
  }

  __host__ __device__ __forceinline__ static AffineT suffix_from_tail(const Vec3T<T>& tail, const AffineT& curr) {
    return {
      .m = outer(tail, curr.v),
      .v = tail,
    };
  }

  __host__ __device__ __forceinline__ static AffineT local_suffix(const Mat3T<T>& frame, const T* grad_tail, const AffineT& curr) {
    return suffix_from_tail(transpose_mul(frame, grad_tail), curr);
  }

  __host__ __device__ __forceinline__ Mat3T<T> centered_suffix(const AffineT& curr) const {
    return m - outer(v, curr.v);
  }

  __host__ __device__ __forceinline__ AffineT operator+(const AffineT& b) const {
    return {
      .m = m + b.m,
      .v = v + b.v,
    };
  }
};

using Affine = AffineT<float>;

static_assert(alignof(AffineT<float>) == 16);
static_assert(sizeof(AffineT<float>) == 12 * sizeof(float));
static_assert(offsetof(AffineT<float>, v) == 9 * sizeof(float));
static_assert(std::is_standard_layout_v<AffineT<float>>);
static_assert(std::is_trivially_copyable_v<AffineT<float>>);
static_assert(sizeof(AffineT<__half>) == 12 * sizeof(__half));
static_assert(alignof(AffineT<__half>) == 8);
static_assert(sizeof(AffineT<__nv_bfloat16>) == 12 * sizeof(__nv_bfloat16));
static_assert(alignof(AffineT<__nv_bfloat16>) == 8);
static_assert(sizeof(AffineT<double>) == 12 * sizeof(double));
static_assert(alignof(AffineT<double>) == 32);

template <typename T>
struct LocalTransformGrads {
  T length;
  T angle;
  T dihedral;
};

template <typename T>
struct SinCosGrads {
  T sin;
  T cos;
};

template <typename T, typename R>
static __device__ __forceinline__ LocalTransformGrads<T> local_transform_grads(
    const AffineT<T>& prev,
    const AffineT<T>& curr,
    const AffineT<T>& suffix,
    T length,
    R s,
    R c,
    R S,
    R C) {
  const Vec3T<T> grad_t = transpose_mul(prev.m, suffix.v);
  Mat3T<T> grad_r = transpose_mul(prev.m, suffix.centered_suffix(curr)) * curr.m;
  grad_r.template column<0>() += length * grad_t;

  const Vec3T<T> direction = AffineT<T>::local_direction(s, c, S, C);
  const Vec3T<T> dtheta0 = { scalar<T>(s), scalar<T>(c * C), scalar<T>(c * S) };
  const Vec3T<T> dtheta1 = { scalar<T>(-c), scalar<T>(s * C), scalar<T>(s * S) };
  const Vec3T<T> dphi0 = { scalar<T>(0.0f), scalar<T>(-s * S), scalar<T>(s * C) };
  const Vec3T<T> dphi1 = { scalar<T>(0.0f), scalar<T>(c * S), scalar<T>(-c * C) };
  const Vec3T<T> dphi2 = { scalar<T>(0.0f), scalar<T>(-C), scalar<T>(-S) };

  return {
    .length = dot(grad_t, direction),
    .angle = dot(grad_r.template column<0>(), dtheta0) + dot(grad_r.template column<1>(), dtheta1),
    .dihedral = dot(grad_r.template column<0>(), dphi0) + dot(grad_r.template column<1>(), dphi1) + dot(grad_r.template column<2>(), dphi2),
  };
}

template <typename T, typename R>
static __device__ __forceinline__ SinCosGrads<T> local_transform_sincos_grads(
    const AffineT<T>& prev,
    const AffineT<T>& curr,
    const AffineT<T>& suffix,
    T length,
    R s,
    R c) {
  T a1[3];
  T a2[3];
  unroll for (int col = 0; col < 3; ++col) {
    const T b0 = suffix.m[0][col] - suffix.v[0] * curr.v[col];
    const T b1 = suffix.m[1][col] - suffix.v[1] * curr.v[col];
    const T b2 = suffix.m[2][col] - suffix.v[2] * curr.v[col];
    a1[col] = prev.m[0][1] * b0 + prev.m[1][1] * b1 + prev.m[2][1] * b2;
    a2[col] = prev.m[0][2] * b0 + prev.m[1][2] * b1 + prev.m[2][2] * b2;
  }

  const T grad_t1 = prev.m[0][1] * suffix.v[0] + prev.m[1][1] * suffix.v[1] + prev.m[2][1] * suffix.v[2];
  const T grad_t2 = prev.m[0][2] * suffix.v[0] + prev.m[1][2] * suffix.v[1] + prev.m[2][2] * suffix.v[2];

  const T g10 = a1[0] * curr.m[0][0] + a1[1] * curr.m[1][0] + a1[2] * curr.m[2][0] + length * grad_t1;
  const T g11 = a1[0] * curr.m[0][1] + a1[1] * curr.m[1][1] + a1[2] * curr.m[2][1];
  const T g12 = a1[0] * curr.m[0][2] + a1[1] * curr.m[1][2] + a1[2] * curr.m[2][2];
  const T g20 = a2[0] * curr.m[0][0] + a2[1] * curr.m[1][0] + a2[2] * curr.m[2][0] + length * grad_t2;
  const T g21 = a2[0] * curr.m[0][1] + a2[1] * curr.m[1][1] + a2[2] * curr.m[2][1];
  const T g22 = a2[0] * curr.m[0][2] + a2[1] * curr.m[1][2] + a2[2] * curr.m[2][2];

  const T st = scalar<T>(s);
  const T ct = scalar<T>(c);
  return {
    .sin = st * g20 - ct * g21 - g12,
    .cos = st * g10 - ct * g11 + g22,
  };
}

template <typename T>
static __device__ __forceinline__ LocalTransformGrads<T> local_transform_grads(
    const AffineT<T>& prev,
    const AffineT<T>& curr,
    const AffineT<T>& suffix,
    T length,
    T theta,
    T phi) {
  const auto [s, c] = sin_cos(to_real(theta));
  const auto [S, C] = sin_cos(to_real(phi));
  return local_transform_grads(prev, curr, suffix, length, s, c, S, C);
}

} // namespace cufk::affine
