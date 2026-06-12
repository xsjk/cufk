#pragma once

#include "scalar.cuh"

#include <type_traits>
#include <utility>

#define unroll _Pragma("unroll")

namespace cufk::math {

template <typename T>
struct Vec3T {
  T v[3]{};

  __host__ __device__ __forceinline__ T& operator[](int i) {
    return v[i];
  }

  __host__ __device__ __forceinline__ const T& operator[](int i) const {
    return v[i];
  }
};

template <typename T>
struct Mat3T {
  using Row = T[3];

  template <int Col>
  struct Column {
    static_assert(Col >= 0 && Col < 3);

    Mat3T& m;

    __host__ __device__ __forceinline__ T& operator[](int row) {
      return m[row][Col];
    }

    __host__ __device__ __forceinline__ const T& operator[](int row) const {
      return m[row][Col];
    }

    __host__ __device__ __forceinline__ operator Vec3T<T>() const {
      return { m[0][Col], m[1][Col], m[2][Col] };
    }

    __host__ __device__ __forceinline__ Column& operator=(const Vec3T<T>& x) {
      m[0][Col] = x[0];
      m[1][Col] = x[1];
      m[2][Col] = x[2];
      return *this;
    }

    __host__ __device__ __forceinline__ Column& operator+=(const Vec3T<T>& x) {
      m[0][Col] = m[0][Col] + x[0];
      m[1][Col] = m[1][Col] + x[1];
      m[2][Col] = m[2][Col] + x[2];
      return *this;
    }

    __host__ __device__ __forceinline__ Column& operator-=(const Vec3T<T>& x) {
      m[0][Col] = m[0][Col] - x[0];
      m[1][Col] = m[1][Col] - x[1];
      m[2][Col] = m[2][Col] - x[2];
      return *this;
    }
  };

  T v[3][3]{};

  __host__ __device__ __forceinline__ static Mat3T from_rows(const Vec3T<T>& row0, const Vec3T<T>& row1, const Vec3T<T>& row2) {
    return Mat3T{
      .v = {
          { row0[0], row0[1], row0[2] },
          { row1[0], row1[1], row1[2] },
          { row2[0], row2[1], row2[2] },
      },
    };
  }

  __host__ __device__ __forceinline__ static Mat3T from_columns(const Vec3T<T>& col0, const Vec3T<T>& col1, const Vec3T<T>& col2) {
    return Mat3T{
      .v = {
          { col0[0], col1[0], col2[0] },
          { col0[1], col1[1], col2[1] },
          { col0[2], col1[2], col2[2] },
      },
    };
  }

  __host__ __device__ __forceinline__ static Mat3T identity() {
    return from_rows(
        { scalar<T>(1.0f), scalar<T>(0.0f), scalar<T>(0.0f) },
        { scalar<T>(0.0f), scalar<T>(1.0f), scalar<T>(0.0f) },
        { scalar<T>(0.0f), scalar<T>(0.0f), scalar<T>(1.0f) });
  }

  __host__ __device__ __forceinline__ Row& operator[](int i) {
    return v[i];
  }

  __host__ __device__ __forceinline__ const Row& operator[](int i) const {
    return v[i];
  }

  template <int Col>
  __host__ __device__ __forceinline__ Column<Col> column() {
    return { *this };
  }

  template <int Col>
  __host__ __device__ __forceinline__ Vec3T<T> column() const {
    static_assert(Col >= 0 && Col < 3);
    return { v[0][Col], v[1][Col], v[2][Col] };
  }
};

using Vec3 = Vec3T<float>;
using Mat3 = Mat3T<float>;

static_assert(sizeof(Vec3T<float>) == 3 * sizeof(float));
static_assert(std::is_aggregate_v<Vec3T<float>>);
static_assert(std::is_standard_layout_v<Vec3T<float>>);
static_assert(std::is_trivially_copyable_v<Vec3T<float>>);
static_assert(sizeof(Mat3T<float>) == 9 * sizeof(float));
static_assert(std::is_aggregate_v<Mat3T<float>>);
static_assert(std::is_standard_layout_v<Mat3T<float>>);
static_assert(std::is_trivially_copyable_v<Mat3T<float>>);
static_assert(sizeof(Vec3T<__half>) == 3 * sizeof(__half));
static_assert(sizeof(Mat3T<__half>) == 9 * sizeof(__half));
static_assert(sizeof(Vec3T<__nv_bfloat16>) == 3 * sizeof(__nv_bfloat16));
static_assert(sizeof(Mat3T<__nv_bfloat16>) == 9 * sizeof(__nv_bfloat16));
static_assert(sizeof(Vec3T<double>) == 3 * sizeof(double));
static_assert(sizeof(Mat3T<double>) == 9 * sizeof(double));

static __device__ __forceinline__ std::pair<float, float> sin_cos(float x) {
  float s, c;
  sincosf(x, &s, &c);
  return { s, c };
}

static __device__ __forceinline__ std::pair<double, double> sin_cos(double x) {
  double s, c;
  sincos(x, &s, &c);
  return { s, c };
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> load_vec3(const T* x) {
  return { x[0], x[1], x[2] };
}

template <typename T>
__host__ __device__ __forceinline__ void store_vec3(T* out, const Vec3T<T>& x) {
  out[0] = x[0];
  out[1] = x[1];
  out[2] = x[2];
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator+(const Vec3T<T>& a, const Vec3T<T>& b) {
  return { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T>& operator+=(Vec3T<T>& a, const Vec3T<T>& b) {
  a[0] = a[0] + b[0];
  a[1] = a[1] + b[1];
  a[2] = a[2] + b[2];
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator-(const Vec3T<T>& a, const Vec3T<T>& b) {
  return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T>& operator-=(Vec3T<T>& a, const Vec3T<T>& b) {
  a[0] = a[0] - b[0];
  a[1] = a[1] - b[1];
  a[2] = a[2] - b[2];
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator-(const Vec3T<T>& x) {
  return { -x[0], -x[1], -x[2] };
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator*(T a, const Vec3T<T>& x) {
  return { a * x[0], a * x[1], a * x[2] };
}

template <typename T>
  requires(!std::is_same_v<T, float>)
__host__ __device__ __forceinline__ Vec3T<T> operator*(float a, const Vec3T<T>& x) {
  return scalar<T>(a) * x;
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator*(const Vec3T<T>& x, T a) {
  return a * x;
}

template <typename T>
  requires(!std::is_same_v<T, float>)
__host__ __device__ __forceinline__ Vec3T<T> operator*(const Vec3T<T>& x, float a) {
  return scalar<T>(a) * x;
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator/(const Vec3T<T>& x, T a) {
  return { x[0] / a, x[1] / a, x[2] / a };
}

template <typename A, typename B>
__host__ __device__ __forceinline__ auto dot(const A& a, const B& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> cross(const Vec3T<T>& a, const Vec3T<T>& b) {
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  };
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T>& operator+=(Mat3T<T>& a, const Mat3T<T>& b) {
  unroll for (int row = 0; row < 3; ++row) {
    unroll for (int col = 0; col < 3; ++col) {
      a[row][col] = a[row][col] + b[row][col];
    }
  }
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> operator+(Mat3T<T> a, const Mat3T<T>& b) {
  a += b;
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T>& operator-=(Mat3T<T>& a, const Mat3T<T>& b) {
  unroll for (int row = 0; row < 3; ++row) {
    unroll for (int col = 0; col < 3; ++col) {
      a[row][col] = a[row][col] - b[row][col];
    }
  }
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> operator-(Mat3T<T> a, const Mat3T<T>& b) {
  a -= b;
  return a;
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> outer(const Vec3T<T>& a, const Vec3T<T>& b) {
  return Mat3T<T>::from_rows(a[0] * b, a[1] * b, a[2] * b);
}

template <typename T, typename X>
__host__ __device__ __forceinline__ Vec3T<T> mat_vec_mul(const Mat3T<T>& a, const X& x) {
  return {
    a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
    a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
    a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
  };
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator*(const Mat3T<T>& a, const Vec3T<T>& x) {
  return mat_vec_mul(a, x);
}

template <typename T>
__host__ __device__ __forceinline__ Vec3T<T> operator*(const Mat3T<T>& a, const T (&x)[3]) {
  return mat_vec_mul(a, x);
}

template <typename T, typename X>
__host__ __device__ __forceinline__ Vec3T<T> transpose_mul(const Mat3T<T>& a, const X& x) {
  return {
    a[0][0] * x[0] + a[1][0] * x[1] + a[2][0] * x[2],
    a[0][1] * x[0] + a[1][1] * x[1] + a[2][1] * x[2],
    a[0][2] * x[0] + a[1][2] * x[1] + a[2][2] * x[2],
  };
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> transpose_mul(const Mat3T<T>& a, const Mat3T<T>& b) {
  return Mat3T<T>::from_columns(transpose_mul(a, b.template column<0>()), transpose_mul(a, b.template column<1>()), transpose_mul(a, b.template column<2>()));
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> mat_mul(const Mat3T<T>& a, const Mat3T<T>& b) {
  return Mat3T<T>::from_columns(mat_vec_mul(a, b.template column<0>()), mat_vec_mul(a, b.template column<1>()), mat_vec_mul(a, b.template column<2>()));
}

template <typename T>
__host__ __device__ __forceinline__ Mat3T<T> operator*(const Mat3T<T>& a, const Mat3T<T>& b) {
  return mat_mul(a, b);
}

} // namespace cufk::math
