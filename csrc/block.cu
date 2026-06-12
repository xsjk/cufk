#include <torch/extension.h>

#include "affine.cuh"
#include "affine_scan.cuh"
#include "extension_utils.cuh"
#include "frame_utils.cuh"
#include "math.cuh"

#include <cstdint>
#include <type_traits>

using cufk::ext::chain_shape_from_steps;
using cufk::ext::ChainAnchors;
using cufk::ext::ChainLaunch;
using cufk::ext::ChainShape;
using cufk::ext::check_cuda_float;
using cufk::ext::check_like;
using cufk::ext::checked_chain_anchors;
using cufk::ext::CheckedChainInputs;
using cufk::ext::dispatch_dtype;
using cufk::ext::dynamic_shared_plan_limit;
using cufk::ext::empty_xyz_like;
using cufk::ext::kernel_arg;
using cufk::ext::launch_block_plan;
using cufk::ext::plan_capacity;
using cufk::ext::set_max_dynamic_shared_memory;
using cufk::ext::Tensor;
using cufk::ext::tensor_ptr;
using cufk::ext::TensorList;
using cufk::frame::normalized;
using cufk::frame::stable_perpendicular;

namespace {

using BlockScanBackend = cufk::scan::CustomScanBackend;
using Shape = ChainShape;

constexpr float kEps = 1.0e-12f;

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

struct WideBackwardScanPlan {
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
struct BackwardScanPlan
    : std::conditional_t<(sizeof(cufk::affine::AffineT<T>) > sizeof(cufk::affine::AffineT<float>)), WideBackwardScanPlan, GradScanPlan<T>> {};

constexpr int kMaxBlockScanSteps = plan_capacity<GradScanPlan<float>>();

using cufk::scalar;
using cufk::to_real;
using cufk::affine::AffineT;
using cufk::affine::local_transform_grads;
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
static __device__ __forceinline__ FirstFrame<T> compute_first_frame(
    const T* lengths,
    const T* angles,
    const T* p0,
    const T* first_direction,
    const T* initial_normal) {
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
__global__ void __launch_bounds__(BlockThreads) block_forward_kernel(
    const T* __restrict__ lengths,
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

  cufk::affine_scan::forward_positions_in_place<BlockScanBackend, T, BlockThreads,
                                                ItemsPerThread>(input);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    if (step < steps) {
      store_vec3(out + (step + 3) * 3, shared_p2 + shared_base_frame * input[item].v);
    }
  }
}

template <typename T>
static __device__ __forceinline__ void write_first_frame_param_grads(
    const T* lengths,
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
__global__ void __launch_bounds__(BlockThreads) block_backward_kernel(
    const T* __restrict__ lengths,
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

  cufk::affine_scan::saved_prefix<BlockScanBackend, T, BlockThreads,
                                  ItemsPerThread>(local, prefix);

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

  cufk::affine_scan::suffix<BlockScanBackend, T, BlockThreads,
                            ItemsPerThread>(input, suffix);

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int reverse_step = first_reverse_step + item;
    const int step = steps - 1 - reverse_step;
    if (step >= 0) {
      const Affine prev = step == 0 ? Affine::identity() : prefix_row[step - 1];
      const Affine curr = prefix_row[step];
      const auto [grad_length, grad_angle, grad_dihedral] =
          local_transform_grads(prev, curr, suffix[item], length_row[step + 2], angle_row[step + 1], dihedral_row[step]);
      grad_length_row[step + 2] = grad_length;
      grad_angle_row[step + 1] = grad_angle;
      grad_dihedral_row[step] = grad_dihedral;
      if (step == 0) {
        write_first_frame_param_grads(length_row, angle_row, p0, first_direction, initial_normal, grad_row, suffix[item], grad_length_row, grad_angle_row);
      }
    }
  }
}

struct LaunchContext : ChainLaunch {
  LaunchContext(const Tensor& lengths, Shape shape) : ChainLaunch(lengths, shape, kMaxBlockScanSteps, "reconstruct") {}
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
void launch_forward(const LaunchContext& run, const auto&... args) {
  const auto kernel = [&]<int Threads, int Items>() {
    block_forward_kernel<T, Threads, Items><<<run.shape.batch, Threads, 0, run.stream>>>(args..., run.shape);
  };
  launch_block_plan<ForwardScanPlan<T>>(run.steps, kernel);
}

template <typename T>
void launch_backward(const LaunchContext& run, const auto&... args) {
  const auto kernel = [&]<int Threads, int Items>() {
    constexpr auto kernel_ptr = block_backward_kernel<T, Threads, Items>;
    const int shared_bytes = static_cast<int>(run.steps * sizeof(AffineT<T>));
    set_max_dynamic_shared_memory(kernel_ptr, shared_bytes);
    kernel_ptr<<<run.shape.batch, Threads, shared_bytes, run.stream>>>(args..., run.shape);
  };
  launch_block_plan<BackwardScanPlan<T>>(run.steps, kernel);
}

template <typename T>
void launch_forward_tensors(const LaunchContext& run, const auto&... args) {
  launch_forward<T>(run, kernel_arg<T>(args)...);
}

template <typename T>
void launch_backward_tensors(const LaunchContext& run, const GradOutputs& grad, const auto&... args) {
  launch_backward<T>(
      run, kernel_arg<T>(args)..., kernel_arg<T>(grad.lengths), kernel_arg<T>(grad.angles), kernel_arg<T>(grad.dihedrals));
}

template <typename T>
void check_backward_steps(const LaunchContext& run) {
  const auto limit = dynamic_shared_plan_limit<BackwardScanPlan<T>, AffineT<T>>(run.device);
  TORCH_CHECK(
      run.steps <= limit.max,
      "backward supports at most ",
      limit.max,
      " local transforms for this dtype/device, got ",
      run.steps,
      " (plan cap ",
      limit.plan,
      ", shared-memory cap ",
      limit.shared,
      ")");
}

template <typename T>
Tensor forward_cuda_impl(const Tensor& lengths, Shape shape, const auto&... input) {
  const LaunchContext run(lengths, shape);
  Tensor xyz = empty_xyz_like(lengths, run.shape);
  launch_forward_tensors<T>(run, lengths, input..., xyz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return xyz;
}

template <typename T>
TensorList backward_cuda_impl(const Tensor& lengths, Shape shape, const Tensor& angles, const Tensor& dihedrals, const auto&... input) {
  const LaunchContext run(lengths, shape);
  check_backward_steps<T>(run);
  const GradOutputs grad = empty_grads_like(lengths, angles, dihedrals);
  launch_backward_tensors<T>(run, grad, lengths, angles, dihedrals, input...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad.list();
}

Shape check_geom(const Tensor& lengths, const Tensor& angles, const Tensor& dihedrals) {
  check_cuda_float(lengths, "lengths");
  check_like(angles, lengths, "angles");
  check_like(dihedrals, lengths, "dihedrals");
  TORCH_CHECK(lengths.dim() == 2, "lengths must have shape (B, N-1)");
  TORCH_CHECK(angles.dim() == 2, "angles must have shape (B, N-2)");
  TORCH_CHECK(dihedrals.dim() == 2, "dihedrals must have shape (B, N-3)");
  const Shape shape = chain_shape_from_steps(lengths, lengths.size(1) - 2);
  TORCH_CHECK(angles.sizes() == torch::IntArrayRef({ shape.batch, shape.points - 2 }), "bad angles shape");
  TORCH_CHECK(dihedrals.sizes() == torch::IntArrayRef({ shape.batch, shape.points - 3 }), "bad dihedrals shape");
  return shape;
}

CheckedChainInputs checked_inputs(
    const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal) {
  return {
    .shape = check_geom(lengths, angles, dihedrals),
    .anchors = checked_chain_anchors(p0, first_direction, initial_normal, lengths),
  };
}

} // namespace

Tensor forward(
    const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal) {
  const CheckedChainInputs inputs = checked_inputs(lengths, angles, dihedrals, p0, first_direction, initial_normal);
  const ChainAnchors& anchors = inputs.anchors;
  const auto run = [&]<typename T>() {
    return forward_cuda_impl<T>(lengths, inputs.shape, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal);
  };
  return dispatch_dtype(lengths, run);
}

TensorList backward(
    const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal,
    const Tensor& grad_points) {
  check_like(grad_points, lengths, "grad_points");
  const CheckedChainInputs inputs = checked_inputs(lengths, angles, dihedrals, p0, first_direction, initial_normal);
  const ChainAnchors& anchors = inputs.anchors;
  const auto run = [&]<typename T>() {
    return backward_cuda_impl<T>(
        lengths, inputs.shape, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal, grad_points);
  };
  return dispatch_dtype(lengths, run);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward");
  m.def("backward", &backward, "backward");
}
