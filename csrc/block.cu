#include <torch/autograd.h>
#include <torch/extension.h>

#include "affine.cuh"
#include "extension_utils.cuh"
#include "frame_utils.cuh"
#include "math.cuh"
#include "scan.cuh"

#include <cuda/std/functional>

#include <cstdint>

using cuda::std::multiplies;
using cuda::std::plus;
using cufk::ext::chain_shape_from_steps;
using cufk::ext::ChainAnchors;
using cufk::ext::ChainLaunch;
using cufk::ext::ChainShape;
using cufk::ext::check_cuda_float;
using cufk::ext::check_like;
using cufk::ext::checked_chain_anchors;
using cufk::ext::CheckedChainInputs;
using cufk::ext::contiguous_or_self;
using cufk::ext::dispatch_dtype;
using cufk::ext::dynamic_shared_plan_limit;
using cufk::ext::empty_xyz_like;
using cufk::ext::kernel_arg;
using cufk::ext::launch_block_plan;
using cufk::ext::plan_capacity;
using cufk::ext::set_max_dynamic_shared_memory;
using cufk::ext::Tensor;
using cufk::ext::TensorList;
using cufk::frame::normalized;
using cufk::frame::stable_perpendicular;

namespace {

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
struct BackwardScanPlan : ForwardScanPlan<T> {};

template <>
struct BackwardScanPlan<double> {
  static constexpr int tiny_threads = 128;
  static constexpr int tiny_items = 2;
  static constexpr int small_threads = 128;
  static constexpr int small_items = 4;
  static constexpr int mid_threads = 128;
  static constexpr int mid_items = 8;
  static constexpr int full_threads = 128;
  static constexpr int full_items = 8;
};

constexpr int kMaxSteps = plan_capacity<ForwardScanPlan<float>>();

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

  const Affine carry = cufk::scan::inclusive_scan_carry<BlockThreads>(input, input, multiplies<>{}, Affine::identity());
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    input[item].v = (carry * input[item]).v;
  }

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

  cufk::scan::inclusive_scan<BlockThreads>(local, prefix, multiplies<>{}, Affine::identity());

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
    input[item] = {};
    if (step >= 0) {
      const Affine& curr = prefix_row[step];
      const T* tail_grad = grad_row + (step + 3) * 3;
      input[item] = Affine::local_suffix(shared_base_frame, tail_grad, curr);
    }
  }

  cufk::scan::inclusive_scan<BlockThreads>(input, suffix, plus<>{}, Affine{});

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
  LaunchContext(const Tensor& lengths, Shape shape) : ChainLaunch(lengths, shape, kMaxSteps, "reconstruct") {}
};

template <typename T>
Tensor forward_cuda_impl(const Tensor& lengths, Shape shape, const auto&... input) {
  const LaunchContext run(lengths, shape);
  Tensor xyz = empty_xyz_like(lengths, run.shape);
  launch_block_plan<ForwardScanPlan<T>>(run.steps, [&]<int Threads, int Items>() {
    constexpr auto kernel_ptr = block_forward_kernel<T, Threads, Items>;
    kernel_ptr<<<run.shape.batch, Threads, 0, run.stream>>>(kernel_arg<T>(lengths), kernel_arg<T>(input)..., kernel_arg<T>(xyz), run.shape);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return xyz;
}

template <typename T>
TensorList backward_cuda_impl(const Tensor& lengths, Shape shape, const Tensor& angles, const Tensor& dihedrals, const auto&... input) {
  const LaunchContext run(lengths, shape);
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
  Tensor grad_lengths = torch::empty_like(lengths);
  Tensor grad_angles = torch::empty_like(angles);
  Tensor grad_dihedrals = torch::empty_like(dihedrals);
  launch_block_plan<BackwardScanPlan<T>>(run.steps, [&]<int Threads, int Items>() {
    constexpr auto kernel_ptr = block_backward_kernel<T, Threads, Items>;
    const int shared_bytes = static_cast<int>(run.steps * sizeof(AffineT<T>));
    set_max_dynamic_shared_memory(kernel_ptr, shared_bytes);
    kernel_ptr<<<run.shape.batch, Threads, shared_bytes, run.stream>>>(
        kernel_arg<T>(lengths),
        kernel_arg<T>(angles),
        kernel_arg<T>(dihedrals),
        kernel_arg<T>(input)...,
        kernel_arg<T>(grad_lengths),
        kernel_arg<T>(grad_angles),
        kernel_arg<T>(grad_dihedrals),
        run.shape);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return { grad_lengths, grad_angles, grad_dihedrals };
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

class ReconstructAutograd : public torch::autograd::Function<ReconstructAutograd> {
public:
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      Tensor lengths,
      Tensor angles,
      Tensor dihedrals,
      Tensor p0,
      Tensor first_direction,
      Tensor initial_normal) {
    const CheckedChainInputs inputs = checked_inputs(lengths, angles, dihedrals, p0, first_direction, initial_normal);
    const ChainAnchors& anchors = inputs.anchors;
    Tensor xyz = dispatch_dtype(lengths, [&]<typename T>() {
      return forward_cuda_impl<T>(lengths, inputs.shape, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal);
    });
    ctx->save_for_backward({ lengths, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal });
    return xyz;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const Tensor& lengths = saved[0];
    const Tensor& angles = saved[1];
    const Tensor& dihedrals = saved[2];
    const Tensor& p0 = saved[3];
    const Tensor& first_direction = saved[4];
    const Tensor& initial_normal = saved[5];
    Tensor grad_points = contiguous_or_self(grad_outputs[0]);
    check_like(grad_points, lengths, "grad_points");
    const CheckedChainInputs inputs = checked_inputs(lengths, angles, dihedrals, p0, first_direction, initial_normal);
    const ChainAnchors& anchors = inputs.anchors;
    TensorList grads = dispatch_dtype(lengths, [&]<typename T>() {
      return backward_cuda_impl<T>(lengths, inputs.shape, angles, dihedrals, anchors.p0, anchors.first_direction, anchors.initial_normal, grad_points);
    });
    return { grads[0], grads[1], grads[2], Tensor(), Tensor(), Tensor() };
  }
};

Tensor apply_autograd(
    const Tensor& lengths,
    const Tensor& angles,
    const Tensor& dihedrals,
    const Tensor& p0,
    const Tensor& first_direction,
    const Tensor& initial_normal) {
  return ReconstructAutograd::apply(lengths, angles, dihedrals, p0, first_direction, initial_normal);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply", &apply_autograd, "reconstruct autograd apply");
}
