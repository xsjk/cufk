#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace cufk::ext {

using Tensor = torch::Tensor;
using TensorList = std::vector<Tensor>;

template <typename T>
T* tensor_ptr(const Tensor& x) {
  return reinterpret_cast<T*>(x.data_ptr());
}

template <typename T>
auto kernel_arg(const auto& x) {
  if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, Tensor>) {
    return tensor_ptr<T>(x);
  } else {
    return x;
  }
}

template <typename Fn>
decltype(auto) dispatch_dtype(const Tensor& ref, Fn fn) {
  switch (ref.scalar_type()) {
  case torch::kFloat32:
    return fn.template operator()<float>();
  case torch::kFloat64:
    return fn.template operator()<double>();
  case torch::kFloat16:
    return fn.template operator()<__half>();
  case torch::kBFloat16:
    return fn.template operator()<__nv_bfloat16>();
  default:
    break;
  }
  TORCH_CHECK(false, "unsupported dtype");
  __builtin_unreachable();
}

inline bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

inline void check_cuda_float(const Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(supported_dtype(x.scalar_type()), name, " must be float32, float64, float16, or bfloat16");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

inline void check_cuda_float_row_contiguous(const Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(supported_dtype(x.scalar_type()), name, " must be float32, float64, float16, or bfloat16");
  TORCH_CHECK(x.dim() == 2, name, " must have shape (B, K)");
  TORCH_CHECK(x.stride(1) == 1, name, " must have stride(1) == 1");
}

inline void check_like(const Tensor& x, const Tensor& ref, const char* name) {
  check_cuda_float(x, name);
  TORCH_CHECK(x.scalar_type() == ref.scalar_type(), name, " must have the same dtype as reference");
  TORCH_CHECK(x.device() == ref.device(), name, " must be on the same device as reference");
}

inline void check_anchor(const Tensor& x, const Tensor& ref, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.scalar_type() == ref.scalar_type(), name, " must have the same dtype as reference");
  TORCH_CHECK(x.device() == ref.device(), name, " must be on the same device as reference");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({ 3 }), name, " must have shape (3,)");
}

inline void check_chain_anchors(const Tensor& p0, const Tensor& first_direction, const Tensor& initial_normal, const Tensor& ref) {
  check_anchor(p0, ref, "p0");
  check_anchor(first_direction, ref, "first_direction");
  check_anchor(initial_normal, ref, "initial_normal");
}

struct ChainAnchors {
  Tensor p0;
  Tensor first_direction;
  Tensor initial_normal;
};

inline ChainAnchors checked_chain_anchors(const Tensor& p0, const Tensor& first_direction, const Tensor& initial_normal, const Tensor& ref) {
  check_chain_anchors(p0, first_direction, initial_normal, ref);
  return {
    .p0 = p0.contiguous(),
    .first_direction = first_direction.contiguous(),
    .initial_normal = initial_normal.contiguous(),
  };
}

struct ChainShape {
  int batch;
  int points;
};

struct CheckedChainInputs {
  ChainShape shape;
  ChainAnchors anchors;
};

inline int chain_steps(ChainShape shape) {
  return shape.points - 3;
}

inline ChainShape chain_shape_from_steps(const Tensor& ref, int64_t steps) {
  TORCH_CHECK(ref.size(0) <= INT32_MAX && steps + 3 <= INT32_MAX, "shape exceeds int32 kernel limits");
  TORCH_CHECK(steps >= 1, "point_count must be >= 4");
  return {
    .batch = static_cast<int>(ref.size(0)),
    .points = static_cast<int>(steps + 3),
  };
}

inline void check_max_local_transforms(int steps, int max_steps, const char* op_name) {
  TORCH_CHECK(steps <= max_steps, op_name, " supports at most ", max_steps, " local transforms");
}

struct ChainLaunch {
  c10::cuda::CUDAGuard guard;
  ChainShape shape;
  int steps;
  int device;
  cudaStream_t stream;

  ChainLaunch(const Tensor& ref, ChainShape shape_, int max_steps, const char* op_name)
      : guard(ref.device()), shape(shape_), steps(chain_steps(shape)), device(ref.get_device()), stream(at::cuda::getCurrentCUDAStream()) {
    check_max_local_transforms(steps, max_steps, op_name);
  }
};

inline void set_max_dynamic_shared_memory(const auto kernel, int shared_bytes) {
  const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  TORCH_CHECK(err == cudaSuccess, "failed to opt in dynamic shared memory: ", cudaGetErrorString(err));
}

template <typename Item>
inline int max_optin_dynamic_shared_items(int device) {
  int shared_bytes = 0;
  const cudaError_t err = cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  TORCH_CHECK(err == cudaSuccess, "failed to query max opt-in shared memory: ", cudaGetErrorString(err));
  return shared_bytes / static_cast<int>(sizeof(Item));
}

inline Tensor empty_xyz_like(const Tensor& ref, ChainShape shape) {
  return torch::empty({ shape.batch, shape.points, 3 }, ref.options());
}

inline Tensor empty_strided_like(const Tensor& ref) {
  return torch::empty_strided(ref.sizes(), ref.strides(), ref.options());
}

template <typename Plan, typename Fn>
void launch_block_plan(int items, Fn fn) {
  if (items <= Plan::tiny_threads * Plan::tiny_items) {
    fn.template operator()<Plan::tiny_threads, Plan::tiny_items>();
  } else if (items <= Plan::small_threads * Plan::small_items) {
    fn.template operator()<Plan::small_threads, Plan::small_items>();
  } else if (items <= Plan::mid_threads * Plan::mid_items) {
    fn.template operator()<Plan::mid_threads, Plan::mid_items>();
  } else {
    fn.template operator()<Plan::full_threads, Plan::full_items>();
  }
}

template <typename Plan>
consteval int plan_capacity() {
  return Plan::full_threads * Plan::full_items;
}

struct DynamicSharedPlanLimit {
  int plan;
  int shared;
  int max;
};

template <typename Plan, typename Item>
inline DynamicSharedPlanLimit dynamic_shared_plan_limit(int device) {
  const int plan = plan_capacity<Plan>();
  const int shared = max_optin_dynamic_shared_items<Item>(device);
  return {
    .plan = plan,
    .shared = shared,
    .max = plan < shared ? plan : shared,
  };
}

} // namespace cufk::ext
