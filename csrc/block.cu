#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "affine.cuh"
#include "math.cuh"
#include "scan.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

using Tensor = torch::Tensor;
using TensorList = std::vector<Tensor>;

namespace {

constexpr float kEps = 1.0e-12f;

struct Shape {
  int batch;
  int points;
};

template <typename T>
struct ForwardScanPlan {
  static constexpr int tiny_threads = 128;
  static constexpr int tiny_items = 2;
  static constexpr int small_threads = 256;
  static constexpr int small_items = 2;
  static constexpr int mid_threads = 512;
  static constexpr int mid_items = 2;
  static constexpr int full_threads = 512;
  static constexpr int full_items = 4;
};

template <>
struct ForwardScanPlan<__half> : ForwardScanPlan<float> {};

template <>
struct ForwardScanPlan<__nv_bfloat16> : ForwardScanPlan<float> {};

template <>
struct ForwardScanPlan<double> {
  static constexpr int tiny_threads = 128;
  static constexpr int tiny_items = 2;
  static constexpr int small_threads = 256;
  static constexpr int small_items = 2;
  static constexpr int mid_threads = 256;
  static constexpr int mid_items = 4;
  static constexpr int full_threads = 384;
  static constexpr int full_items = 6;
};

template <typename T>
struct GradScanPlan : ForwardScanPlan<T> {};

template <>
struct GradScanPlan<double> {
  static constexpr int tiny_threads = 128;
  static constexpr int tiny_items = 2;
  static constexpr int small_threads = 256;
  static constexpr int small_items = 2;
  static constexpr int mid_threads = 256;
  static constexpr int mid_items = 4;
  static constexpr int full_threads = 256;
  static constexpr int full_items = 8;
};

struct WideRecomputeScanPlan {
  static constexpr int tiny_threads = 128;
  static constexpr int tiny_items = 2;
  static constexpr int small_threads = 128;
  static constexpr int small_items = 4;
  static constexpr int mid_threads = 128;
  static constexpr int mid_items = 8;
  static constexpr int full_threads = 128;
  static constexpr int full_items = 8;
};

template <typename T>
struct RecomputeScanPlan
    : std::conditional_t<(sizeof(cufk::affine::AffineT<T>) > sizeof(cufk::affine::AffineT<float>)), WideRecomputeScanPlan, GradScanPlan<T>> {};

template <typename Plan>
consteval int plan_capacity() {
  return Plan::full_threads * Plan::full_items;
}

constexpr int kMaxBlockScanSteps = plan_capacity<GradScanPlan<float>>();

using cufk::abs_real;
using cufk::scalar;
using cufk::sqrt_clamped;
using cufk::to_real;
using cufk::affine::add_local_transform_grads;
using cufk::affine::AffineT;
using cufk::math::cross;
using cufk::math::dot;
using cufk::math::load_vec3;
using cufk::math::Mat3T;
using cufk::math::sin_cos;
using cufk::math::store_vec3;
using cufk::math::Vec3T;

template <typename T>
struct FirstFrame {
  Vec3T<T> p1;
  Vec3T<T> p2;
  Vec3T<T> dir0;
  Vec3T<T> turn0;
  Vec3T<T> dir1;
  Vec3T<T> normal;
  Mat3T<T> frame;
};

template <typename T>
static __device__ __forceinline__ Vec3T<T> normalized(const Vec3T<T>& v) {
  const auto n2 = to_real(dot(v, v));
  return v / scalar<T>(sqrt_clamped(n2, static_cast<decltype(n2)>(kEps)));
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

template <typename T>
static __device__ __forceinline__ FirstFrame<T> compute_first_frame(
    const T* lengths, const T* angles, const T* p0, const T* first_direction, const T* initial_normal) {
  const Vec3T<T> origin = load_vec3(p0);
  const Vec3T<T> dir0 = normalized(load_vec3(first_direction));
  const Vec3T<T> p1 = origin + lengths[0] * dir0;

  Vec3T<T> normal0 = load_vec3(initial_normal);
  normal0 -= dot(normal0, dir0) * dir0;
  if (to_real(dot(normal0, normal0)) < kEps) {
    normal0 = stable_perpendicular(dir0);
  } else {
    normal0 = normalized(normal0);
  }
  const Vec3T<T> turn0 = normalized(cross(normal0, dir0));

  const auto [s, c] = sin_cos(to_real(angles[0]));
  const Vec3T<T> dir1 = -c * dir0 + s * turn0;
  const Vec3T<T> p2 = p1 + lengths[1] * dir1;
  const Vec3T<T> normal = normalized(cross(dir0, dir1));
  const Vec3T<T> side = cross(normal, dir1);

  return {
    .p1 = p1,
    .p2 = p2,
    .dir0 = dir0,
    .turn0 = turn0,
    .dir1 = dir1,
    .normal = normal,
    .frame = Mat3T<T>::from_columns(dir1, side, normal),
  };
}

template <typename T>
static __device__ __forceinline__ void write_first_points(T* out, const T* p0, const FirstFrame<T>& first) {
  store_vec3(out, load_vec3(p0));
  store_vec3(out + 3, first.p1);
  store_vec3(out + 6, first.p2);
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads) block_forward_kernel(const T* __restrict__ lengths,
    const T* __restrict__ angles,
    const T* __restrict__ dihedrals,
    const T* __restrict__ p0,
    const T* __restrict__ first_direction,
    const T* __restrict__ initial_normal,
    T* __restrict__ xyz,
    const __grid_constant__ Shape shape) {
  using Affine = AffineT<T>;
  __shared__ Vec3T<T> shared_p2;
  __shared__ Mat3T<T> shared_base_frame;

  const int points = shape.points;
  const int steps = points - 3;
  const int64_t row = blockIdx.x;
  const T* length_row = lengths + row * (points - 1);
  const T* angle_row = angles + row * (points - 2);
  const T* dihedral_row = dihedrals + row * (points - 3);
  T* out = xyz + row * points * 3;

  if (threadIdx.x == 0) {
    const FirstFrame<T> first = compute_first_frame(length_row, angle_row, p0, first_direction, initial_normal);
    write_first_points(out, p0, first);
    shared_p2 = first.p2;
    shared_base_frame = first.frame;
  }

  Affine input[ItemsPerThread];
  const int first_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    input[item] = step < steps ? Affine::local_transform(length_row[step + 2], angle_row[step + 1], dihedral_row[step]) : Affine::identity();
  }

  cufk::scan::forward_positions_in_place<T, BlockThreads, ItemsPerThread>(input);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    if (step < steps) {
      store_vec3(out + (step + 3) * 3, shared_p2 + shared_base_frame * input[item].v);
    }
  }
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads) block_forward_prefix_kernel(const T* __restrict__ lengths,
    const T* __restrict__ angles,
    const T* __restrict__ dihedrals,
    const T* __restrict__ p0,
    const T* __restrict__ first_direction,
    const T* __restrict__ initial_normal,
    T* __restrict__ xyz,
    AffineT<T>* __restrict__ prefix_out,
    const __grid_constant__ Shape shape) {
  using Affine = AffineT<T>;
  __shared__ Vec3T<T> shared_p2;
  __shared__ Mat3T<T> shared_base_frame;

  const int points = shape.points;
  const int steps = points - 3;
  const int64_t row = blockIdx.x;
  const T* length_row = lengths + row * (points - 1);
  const T* angle_row = angles + row * (points - 2);
  const T* dihedral_row = dihedrals + row * (points - 3);
  T* out = xyz + row * points * 3;
  Affine* prefix_row = prefix_out + row * steps;

  if (threadIdx.x == 0) {
    const FirstFrame<T> first = compute_first_frame(length_row, angle_row, p0, first_direction, initial_normal);
    write_first_points(out, p0, first);
    shared_p2 = first.p2;
    shared_base_frame = first.frame;
  }

  Affine input[ItemsPerThread];
  Affine prefix[ItemsPerThread];
  const int first_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    input[item] = step < steps ? Affine::local_transform(length_row[step + 2], angle_row[step + 1], dihedral_row[step]) : Affine::identity();
  }

  cufk::scan::saved_prefix_affine<T, BlockThreads, ItemsPerThread>(input, prefix);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    if (step < steps) {
      prefix_row[step] = prefix[item];
      store_vec3(out + (step + 3) * 3, shared_p2 + shared_base_frame * prefix[item].v);
    }
  }
}

template <typename T>
static __device__ __forceinline__ void write_first_frame_param_grads(const T* lengths,
    const T* angles,
    const T* p0,
    const T* first_direction,
    const T* initial_normal,
    const T* grad_points,
    const AffineT<T>& suffix,
    T* grad_lengths,
    T* grad_angles) {
  const FirstFrame<T> first = compute_first_frame(lengths, angles, p0, first_direction, initial_normal);
  const Vec3T<T> tail_v = first.frame * suffix.v;
  const Mat3T<T> tail_a = first.frame * suffix.m;
  const Vec3T<T> grad_p2 = load_vec3(grad_points + 6) + tail_v;
  grad_lengths[0] = dot(load_vec3(grad_points + 3) + grad_p2, first.dir0);
  grad_lengths[1] = dot(grad_p2, first.dir1);

  const auto [s, c] = sin_cos(to_real(angles[0]));
  const Vec3T<T> d_dir = s * first.dir0 + c * first.turn0;
  const Vec3T<T> d_side = cross(first.normal, d_dir);
  grad_angles[0] = lengths[1] * dot(grad_p2, d_dir) + dot(tail_a.template column<0>(), d_dir) + dot(tail_a.template column<1>(), d_side);
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads) block_backward_kernel(const T* __restrict__ lengths,
    const T* __restrict__ angles,
    const T* __restrict__ dihedrals,
    const T* __restrict__ p0,
    const T* __restrict__ first_direction,
    const T* __restrict__ initial_normal,
    const T* __restrict__ grad_points,
    const AffineT<T>* __restrict__ prefix,
    T* __restrict__ grad_lengths,
    T* __restrict__ grad_angles,
    T* __restrict__ grad_dihedrals,
    const __grid_constant__ Shape shape) {
  using Affine = AffineT<T>;
  __shared__ Mat3T<T> shared_base_frame;

  const int points = shape.points;
  const int steps = points - 3;
  const int64_t row = blockIdx.x;
  const T* length_row = lengths + row * (points - 1);
  const T* angle_row = angles + row * (points - 2);
  const T* dihedral_row = dihedrals + row * (points - 3);
  const T* grad_row = grad_points + row * points * 3;
  const Affine* prefix_row = prefix + row * steps;
  T* grad_length_row = grad_lengths + row * (points - 1);
  T* grad_angle_row = grad_angles + row * (points - 2);
  T* grad_dihedral_row = grad_dihedrals + row * (points - 3);

  if (threadIdx.x == 0) {
    shared_base_frame = compute_first_frame(length_row, angle_row, p0, first_direction, initial_normal).frame;
  }
  __syncthreads();

  Affine input[ItemsPerThread];
  Affine suffix[ItemsPerThread];
  const int first_reverse_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int reverse_step = first_reverse_step + item;
    const int step = steps - 1 - reverse_step;
    input[item] = Affine::zero();
    if (step >= 0) {
      const Affine& curr = prefix_row[step];
      const T* tail_grad = grad_row + (step + 3) * 3;
      input[item] = Affine::local_suffix(shared_base_frame, tail_grad, curr);
    }
  }

  cufk::scan::suffix_affine<T, BlockThreads, ItemsPerThread>(input, suffix);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int reverse_step = first_reverse_step + item;
    const int step = steps - 1 - reverse_step;
    if (step >= 0) {
      const Affine prev = step == 0 ? Affine::identity() : prefix_row[step - 1];
      const Affine curr = prefix_row[step];
      add_local_transform_grads(prev,
          curr,
          suffix[item],
          length_row[step + 2],
          angle_row[step + 1],
          dihedral_row[step],
          grad_length_row[step + 2],
          grad_angle_row[step + 1],
          grad_dihedral_row[step]);
      if (step == 0) {
        write_first_frame_param_grads(length_row, angle_row, p0, first_direction, initial_normal, grad_row, suffix[item], grad_length_row, grad_angle_row);
      }
    }
  }
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads) block_backward_recompute_kernel(const T* __restrict__ lengths,
    const T* __restrict__ angles,
    const T* __restrict__ dihedrals,
    const T* __restrict__ p0,
    const T* __restrict__ first_direction,
    const T* __restrict__ initial_normal,
    const T* __restrict__ grad_points,
    T* __restrict__ grad_lengths,
    T* __restrict__ grad_angles,
    T* __restrict__ grad_dihedrals,
    const __grid_constant__ Shape shape) {
  using Affine = AffineT<T>;
  extern __shared__ __align__(32) unsigned char shared_prefix_bytes[];
  Affine* prefix_row = reinterpret_cast<Affine*>(shared_prefix_bytes);
  __shared__ Mat3T<T> shared_base_frame;

  const int points = shape.points;
  const int steps = points - 3;
  const int64_t row = blockIdx.x;
  const T* length_row = lengths + row * (points - 1);
  const T* angle_row = angles + row * (points - 2);
  const T* dihedral_row = dihedrals + row * (points - 3);
  const T* grad_row = grad_points + row * points * 3;
  T* grad_length_row = grad_lengths + row * (points - 1);
  T* grad_angle_row = grad_angles + row * (points - 2);
  T* grad_dihedral_row = grad_dihedrals + row * (points - 3);

  if (threadIdx.x == 0) {
    shared_base_frame = compute_first_frame(length_row, angle_row, p0, first_direction, initial_normal).frame;
  }

  Affine local[ItemsPerThread];
  Affine prefix[ItemsPerThread];
  const int first_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    local[item] = step < steps ? Affine::local_transform(length_row[step + 2], angle_row[step + 1], dihedral_row[step]) : Affine::identity();
  }

  cufk::scan::saved_prefix_affine<T, BlockThreads, ItemsPerThread>(local, prefix);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    if (step < steps) {
      prefix_row[step] = prefix[item];
    }
  }
  __syncthreads();

  Affine input[ItemsPerThread];
  Affine suffix[ItemsPerThread];
  const int first_reverse_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int reverse_step = first_reverse_step + item;
    const int step = steps - 1 - reverse_step;
    input[item] = Affine::zero();
    if (step >= 0) {
      const Affine& curr = prefix_row[step];
      const T* tail_grad = grad_row + (step + 3) * 3;
      input[item] = Affine::local_suffix(shared_base_frame, tail_grad, curr);
    }
  }

  cufk::scan::suffix_affine<T, BlockThreads, ItemsPerThread>(input, suffix);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int reverse_step = first_reverse_step + item;
    const int step = steps - 1 - reverse_step;
    if (step >= 0) {
      const Affine prev = step == 0 ? Affine::identity() : prefix_row[step - 1];
      const Affine curr = prefix_row[step];
      add_local_transform_grads(prev,
          curr,
          suffix[item],
          length_row[step + 2],
          angle_row[step + 1],
          dihedral_row[step],
          grad_length_row[step + 2],
          grad_angle_row[step + 1],
          grad_dihedral_row[step]);
      if (step == 0) {
        write_first_frame_param_grads(length_row, angle_row, p0, first_direction, initial_normal, grad_row, suffix[item], grad_length_row, grad_angle_row);
      }
    }
  }
}

Shape checked_shape(const Tensor& lengths) {
  TORCH_CHECK(lengths.size(0) <= INT32_MAX && lengths.size(1) + 1 <= INT32_MAX, "shape exceeds int32 kernel limits");
  return {
    .batch = static_cast<int>(lengths.size(0)),
    .points = static_cast<int>(lengths.size(1) + 1),
  };
}

int step_count(Shape shape) {
  return shape.points - 3;
}

void check_step_count(int steps) {
  TORCH_CHECK(steps <= kMaxBlockScanSteps, "reconstruct supports at most ", kMaxBlockScanSteps, " local transforms");
}

Tensor empty_xyz_like(const Tensor& lengths, Shape shape) {
  return torch::empty({ shape.batch, shape.points, 3 }, lengths.options());
}

struct LaunchContext {
  c10::cuda::CUDAGuard guard;
  Shape shape;
  int steps;
  int device;
  cudaStream_t stream;

  explicit LaunchContext(const Tensor& lengths)
      : guard(lengths.device()), shape(checked_shape(lengths)), steps(step_count(shape)), device(lengths.get_device()),
        stream(at::cuda::getCurrentCUDAStream()) {
    check_step_count(steps);
  }
};

struct GradOutputs {
  Tensor lengths;
  Tensor angles;
  Tensor dihedrals;

  TensorList list() const {
    return { lengths, angles, dihedrals };
  }
};

GradOutputs empty_grads_like(const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals) {
  return {
    .lengths = torch::empty_like(lengths),
    .angles = torch::empty_like(angles),
    .dihedrals = torch::empty_like(dihedrals),
  };
}

template <typename T>
T* tensor_ptr(const Tensor& x) {
  return reinterpret_cast<T*>(x.data_ptr());
}

template <typename T>
AffineT<T>* affine_ptr(const Tensor& x) {
  return tensor_ptr<AffineT<T>>(x);
}

template <typename T>
auto kernel_arg(const auto& x) {
  if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, Tensor>) {
    return tensor_ptr<T>(x);
  } else {
    return x;
  }
}

enum class CudaOp {
  forward,
  forward_with_prefix,
  backward_with_prefix,
  backward_recompute,
};

template <typename Plan, typename Launch>
void launch_block_plan(int steps, Launch launch) {
  if (steps <= 256) {
    launch.template operator()<Plan::tiny_threads, Plan::tiny_items>();
  } else if (steps <= 512) {
    launch.template operator()<Plan::small_threads, Plan::small_items>();
  } else if (steps <= 1024) {
    launch.template operator()<Plan::mid_threads, Plan::mid_items>();
  } else {
    launch.template operator()<Plan::full_threads, Plan::full_items>();
  }
}

template <CudaOp Op, typename T>
void launch(const LaunchContext& run, const auto&... args) {
  const auto kernel = [&]<int Threads, int Items>() {
    if constexpr (Op == CudaOp::forward) {
      block_forward_kernel<T, Threads, Items><<<run.shape.batch, Threads, 0, run.stream>>>(args..., run.shape);
    } else if constexpr (Op == CudaOp::forward_with_prefix) {
      block_forward_prefix_kernel<T, Threads, Items><<<run.shape.batch, Threads, 0, run.stream>>>(args..., run.shape);
    } else if constexpr (Op == CudaOp::backward_with_prefix) {
      block_backward_kernel<T, Threads, Items><<<run.shape.batch, Threads, 0, run.stream>>>(args..., run.shape);
    } else if constexpr (Op == CudaOp::backward_recompute) {
      constexpr auto recompute = block_backward_recompute_kernel<T, Threads, Items>;
      const int shared_bytes = static_cast<int>(run.steps * sizeof(AffineT<T>));
      const cudaError_t err = cudaFuncSetAttribute(recompute, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
      TORCH_CHECK(err == cudaSuccess, "failed to opt in dynamic shared memory: ", cudaGetErrorString(err));
      recompute<<<run.shape.batch, Threads, shared_bytes, run.stream>>>(args..., run.shape);
    } else {
      static_assert(Op != Op, "unknown CUDA op");
    }
  };

  if constexpr (Op == CudaOp::forward) {
    launch_block_plan<ForwardScanPlan<T>>(run.steps, kernel);
  } else if constexpr (Op == CudaOp::backward_recompute) {
    launch_block_plan<RecomputeScanPlan<T>>(run.steps, kernel);
  } else {
    launch_block_plan<GradScanPlan<T>>(run.steps, kernel);
  }
}

template <CudaOp Op, typename T>
void launch_tensors(const LaunchContext& run, const auto&... args) {
  launch<Op, T>(run, kernel_arg<T>(args)...);
}

template <CudaOp Op, typename T>
void launch_grad_tensors(const LaunchContext& run, const GradOutputs& grad, const auto&... args) {
  launch_tensors<Op, T>(run, args..., grad.lengths, grad.angles, grad.dihedrals);
}

template <typename T>
void check_recompute_steps(const LaunchContext& run) {
  constexpr int plan_steps = plan_capacity<RecomputeScanPlan<T>>();
  int shared_bytes = 0;
  const cudaError_t err = cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, run.device);
  TORCH_CHECK(err == cudaSuccess, "failed to query max opt-in shared memory: ", cudaGetErrorString(err));
  const int shared_steps = shared_bytes / static_cast<int>(sizeof(AffineT<T>));
  const int max_steps = plan_steps < shared_steps ? plan_steps : shared_steps;
  TORCH_CHECK(run.steps <= max_steps,
      "backward_recompute supports at most ",
      max_steps,
      " local transforms for this dtype/device, got ",
      run.steps,
      " (plan cap ",
      plan_steps,
      ", shared-memory cap ",
      shared_steps,
      ")",
      "; use backward_with_prefix for longer chains");
}

template <typename T>
Tensor forward_cuda_impl(const Tensor& lengths, const auto&... input) {
  const LaunchContext run(lengths);
  Tensor xyz = empty_xyz_like(lengths, run.shape);
  launch_tensors<CudaOp::forward, T>(run, lengths, input..., xyz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return xyz;
}

template <typename T>
TensorList forward_with_prefix_cuda_impl(const Tensor& lengths, const auto&... input) {
  const LaunchContext run(lengths);
  Tensor xyz = empty_xyz_like(lengths, run.shape);
  Tensor prefix = torch::empty({ run.shape.batch, run.steps, 12 }, lengths.options());
  launch_tensors<CudaOp::forward_with_prefix, T>(run, lengths, input..., xyz, affine_ptr<T>(prefix));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return { xyz, prefix };
}

template <typename T>
TensorList backward_with_prefix_cuda_impl(const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal,
    const Tensor& grad_points,
    const Tensor& prefix) {
  const LaunchContext run(lengths);
  TORCH_CHECK(prefix.sizes() == torch::IntArrayRef({ run.shape.batch, run.steps, 12 }), "bad prefix shape");
  const GradOutputs grad = empty_grads_like(lengths, angles, dihedrals);
  launch_grad_tensors<CudaOp::backward_with_prefix, T>(
      run, grad, lengths, angles, dihedrals, p0, first_direction, initial_normal, grad_points, affine_ptr<T>(prefix));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad.list();
}

template <typename T>
TensorList backward_recompute_cuda_impl(const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals, const auto&... input) {
  const LaunchContext run(lengths);
  check_recompute_steps<T>(run);
  const GradOutputs grad = empty_grads_like(lengths, angles, dihedrals);
  launch_grad_tensors<CudaOp::backward_recompute, T>(run, grad, lengths, angles, dihedrals, input...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad.list();
}

template <CudaOp Op>
decltype(auto) dispatch_call(const Tensor& lengths, const auto&... args) {
  const auto run = [&]<typename T>() -> decltype(auto) {
    if constexpr (Op == CudaOp::forward) {
      return forward_cuda_impl<T>(lengths, args...);
    } else if constexpr (Op == CudaOp::forward_with_prefix) {
      return forward_with_prefix_cuda_impl<T>(lengths, args...);
    } else if constexpr (Op == CudaOp::backward_with_prefix) {
      return backward_with_prefix_cuda_impl<T>(lengths, args...);
    } else if constexpr (Op == CudaOp::backward_recompute) {
      return backward_recompute_cuda_impl<T>(lengths, args...);
    } else {
      static_assert(Op != Op, "unknown CUDA op");
    }
  };

  switch (lengths.scalar_type()) {
  case torch::kFloat32:
    return run.template operator()<float>();
  case torch::kFloat64:
    return run.template operator()<double>();
  case torch::kFloat16:
    return run.template operator()<__half>();
  case torch::kBFloat16:
    return run.template operator()<__nv_bfloat16>();
  default:
    break;
  }
  TORCH_CHECK(false, "unsupported dtype");
  __builtin_unreachable();
}

struct Anchors {
  Tensor p0;
  Tensor first_direction;
  Tensor initial_normal;
};

bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

void check_cuda_float(const Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(supported_dtype(x.scalar_type()), name, " must be float32, float64, float16, or bfloat16");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void check_like(const Tensor& x, const Tensor& ref, const char* name) {
  check_cuda_float(x, name);
  TORCH_CHECK(x.scalar_type() == ref.scalar_type(), name, " must have the same dtype as lengths");
}

void check_anchor(const Tensor& x, const Tensor& lengths, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.scalar_type() == lengths.scalar_type(), name, " must have the same dtype as lengths");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({ 3 }), name, " must have shape (3,)");
}

void check_geom(const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals) {
  check_cuda_float(lengths, "lengths");
  check_like(angles, lengths, "angles");
  check_like(dihedrals, lengths, "dihedrals");
  TORCH_CHECK(lengths.dim() == 2, "lengths must have shape (B, N-1)");
  TORCH_CHECK(angles.dim() == 2, "angles must have shape (B, N-2)");
  TORCH_CHECK(dihedrals.dim() == 2, "dihedrals must have shape (B, N-3)");
  const int64_t batch = lengths.size(0);
  const int64_t points = lengths.size(1) + 1;
  TORCH_CHECK(points >= 4, "point_count must be >= 4");
  TORCH_CHECK(angles.sizes() == torch::IntArrayRef({ batch, points - 2 }), "bad angles shape");
  TORCH_CHECK(dihedrals.sizes() == torch::IntArrayRef({ batch, points - 3 }), "bad dihedrals shape");
}

Anchors checked_inputs(
    const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals, const Tensor& p0, const Tensor& first_direction, const Tensor& initial_normal) {
  check_geom(lengths, angles, dihedrals);
  check_anchor(p0, lengths, "p0");
  check_anchor(first_direction, lengths, "first_direction");
  check_anchor(initial_normal, lengths, "initial_normal");
  return {
    .p0 = p0.contiguous(),
    .first_direction = first_direction.contiguous(),
    .initial_normal = initial_normal.contiguous(),
  };
}

template <CudaOp Op>
decltype(auto) dispatch_checked(const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal,
    const auto&... extra) {
  const Anchors anchors = checked_inputs(lengths, angles, dihedrals, p0, first_direction, initial_normal);
  return dispatch_call<Op>(lengths, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal, extra...);
}

} // namespace

Tensor forward(
    const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals, const Tensor& p0, const Tensor& first_direction, const Tensor& initial_normal) {
  return dispatch_checked<CudaOp::forward>(lengths, angles, dihedrals, p0, first_direction, initial_normal);
}

TensorList forward_with_prefix(
    const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals, const Tensor& p0, const Tensor& first_direction, const Tensor& initial_normal) {
  return dispatch_checked<CudaOp::forward_with_prefix>(lengths, angles, dihedrals, p0, first_direction, initial_normal);
}

TensorList backward_with_prefix(const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal,
    const Tensor& grad_points,
    const Tensor& prefix) {
  check_like(grad_points, lengths, "grad_points");
  check_like(prefix, lengths, "prefix");
  return dispatch_checked<CudaOp::backward_with_prefix>(lengths, angles, dihedrals, p0, first_direction, initial_normal, grad_points, prefix);
}

TensorList backward_recompute(const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal,
    const Tensor& grad_points) {
  check_like(grad_points, lengths, "grad_points");
  return dispatch_checked<CudaOp::backward_recompute>(lengths, angles, dihedrals, p0, first_direction, initial_normal, grad_points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward");
  m.def("forward_with_prefix", &forward_with_prefix, "forward with saved prefix");
  m.def("backward_with_prefix", &backward_with_prefix, "backward with saved prefix");
  m.def("backward_recompute", &backward_recompute, "backward with recomputed prefix");
}
