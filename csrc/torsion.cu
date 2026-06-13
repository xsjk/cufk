#include <torch/autograd.h>
#include <torch/extension.h>

#include "extension_utils.cuh"
#include "math.cuh"
#include "rigid.cuh"
#include "scan.cuh"

#include <cuda/std/functional>

#include <array>
#include <cstdint>
#include <type_traits>

using cuda::std::multiplies;
using cuda::std::plus;
using cufk::ext::chain_shape_from_steps;
using cufk::ext::ChainLaunch;
using cufk::ext::ChainShape;
using cufk::ext::check_cuda_float_row_contiguous;
using cufk::ext::check_like;
using cufk::ext::contiguous_or_self;
using cufk::ext::dispatch_dtype;
using cufk::ext::empty_strided_like;
using cufk::ext::empty_xyz_like;
using cufk::ext::kernel_arg;
using cufk::ext::launch_block_plan;
using cufk::ext::plan_capacity;
using cufk::ext::Tensor;

namespace {

using Shape = ChainShape;

constexpr float kNormEps = 1.0e-8f;
constexpr int kPhaseCount = 3;

struct Angle {
  double sin;
  double cos;
};

constexpr std::array kBondLengths = {
  static_cast<float>(CUFK_TORSION_BOND_0_LENGTH), // N-CA
  static_cast<float>(CUFK_TORSION_BOND_1_LENGTH), // CA-C
  static_cast<float>(CUFK_TORSION_BOND_2_LENGTH), // C-N
};

constexpr std::array<Angle, kPhaseCount> kBondAngles = { {
    { static_cast<double>(CUFK_TORSION_ANGLE_0_SIN), static_cast<double>(CUFK_TORSION_ANGLE_0_COS) }, // N-CA-C
    { static_cast<double>(CUFK_TORSION_ANGLE_1_SIN), static_cast<double>(CUFK_TORSION_ANGLE_1_COS) }, // CA-C-N
    { static_cast<double>(CUFK_TORSION_ANGLE_2_SIN), static_cast<double>(CUFK_TORSION_ANGLE_2_COS) }, // C-N-CA
} };

static_assert(kBondLengths.size() == kPhaseCount);
static_assert(kBondAngles.size() == kPhaseCount);

consteval float bond_len(int phase) {
  return kBondLengths[phase];
}

template <int Phase>
__device__ __forceinline__ Angle bond_angle() {
  return kBondAngles[Phase];
}

consteval int len_phase(int step_phase) {
  return (step_phase + 2) % kPhaseCount;
}

consteval int angle_phase(int step_phase) {
  return (step_phase + 1) % kPhaseCount;
}

struct BackwardScanPlan {
  static constexpr int tiny_threads = 64;
  static constexpr int tiny_items = 1;
  static constexpr int small_threads = 32;
  static constexpr int small_items = 4;
  static constexpr int mid_threads = 128;
  static constexpr int mid_items = 4;
  static constexpr int full_threads = 512;
  static constexpr int full_items = 2;
};

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

constexpr int kMaxSteps = plan_capacity<ForwardScanPlan>();
static_assert((kMaxSteps + kPhaseCount - 1) / kPhaseCount <= plan_capacity<BackwardScanPlan>());

using cufk::scalar;
using cufk::to_real;
using cufk::math::cross;
using cufk::math::dot;
using cufk::math::load_vec3;
using cufk::math::Mat3T;
using cufk::math::store_vec3;
using cufk::math::Vec3T;
using cufk::rigid::rigid_apply_with_side;
using cufk::rigid::rigid_side;
using cufk::rigid::RigidFrameT;

template <typename T>
struct Pair {
  using R = decltype(to_real(T{}));
  R sin;
  R cos;
};

template <typename R>
static __device__ __forceinline__ R clamped_pair_norm2(R sin, R cos) {
  const R n2 = sin * sin + cos * cos;
  return n2 < static_cast<R>(kNormEps * kNormEps) ? static_cast<R>(kNormEps * kNormEps) : n2;
}

template <typename T>
static __device__ __forceinline__ Pair<T> normalize(T raw_s, T raw_c) {
  using R = typename Pair<T>::R;
  const R rs = to_real(raw_s);
  const R rc = to_real(raw_c);
  const R n2 = clamped_pair_norm2(rs, rc);
  R rinv;
  if constexpr (std::is_same_v<R, float>) {
    rinv = rsqrtf(n2);
  } else {
    rinv = static_cast<R>(1.0) / sqrt(n2);
  }
  return {
    .sin = rs * rinv,
    .cos = rc * rinv,
  };
}

template <typename T>
struct BaseFrame {
  Vec3T<T> p2;
  Mat3T<T> frame;
};

template <typename T>
static __device__ __forceinline__ Vec3T<T> base_p1() {
  return { scalar<T>(bond_len(0)), scalar<T>(0.0f), scalar<T>(0.0f) };
}

template <typename T>
static __device__ __forceinline__ BaseFrame<T> make_default_base_frame() {
  using R = typename Pair<T>::R;
  const auto first_angle = bond_angle<0>();
  const R s = static_cast<R>(first_angle.sin);
  const R c = static_cast<R>(first_angle.cos);
  const Vec3T<T> dir1 = { scalar<T>(-c), scalar<T>(s), scalar<T>(0.0f) };
  const Vec3T<T> side = { scalar<T>(-s), scalar<T>(-c), scalar<T>(0.0f) };
  return {
    .p2 = { scalar<T>(bond_len(0) - bond_len(1) * c), scalar<T>(bond_len(1) * s), scalar<T>(0.0f) },
    .frame = Mat3T<T>::from_columns(dir1, side, { scalar<T>(0.0f), scalar<T>(0.0f), scalar<T>(1.0f) }),
  };
}

template <typename T>
static __device__ __forceinline__ Pair<T> load_pair(const T* raw, int step) {
  return normalize(raw[2 * step], raw[2 * step + 1]);
}

template <typename T>
static __device__ __forceinline__ void store_raw_pair_grad(T* grad_raw, const T* raw, int step, T grad_torsion) {
  using R = typename Pair<T>::R;
  const R rs = to_real(raw[2 * step]);
  const R rc = to_real(raw[2 * step + 1]);
  const R n2 = clamped_pair_norm2(rs, rc);
  const R scale = to_real(grad_torsion) / n2;
  grad_raw[2 * step] = scalar<T>(rc * scale);
  grad_raw[2 * step + 1] = scalar<T>(-rs * scale);
}

template <typename T>
struct alignas(sizeof(uint32_t)) ForceMoment {
  Vec3T<T> force;
  Vec3T<T> moment;
};

template <typename T>
static __device__ __forceinline__ ForceMoment<T> operator+(ForceMoment<T> a, const ForceMoment<T>& b) {
  a.force += b.force;
  a.moment += b.moment;
  return a;
}

template <typename T>
static __device__ __forceinline__ ForceMoment<T> operator-(ForceMoment<T> a, const ForceMoment<T>& b) {
  a.force -= b.force;
  a.moment -= b.moment;
  return a;
}

template <typename T>
static __device__ __forceinline__ ForceMoment<T> world_force_moment(const Vec3T<T>& point, const T* grad) {
  const Vec3T<T> force = load_vec3(grad);
  return {
    .force = force,
    .moment = cross(point, force),
  };
}

template <int Phase, typename T>
static __device__ __forceinline__ T world_torsion_grad(const Vec3T<T>& prev, const Vec3T<T>& pivot, const ForceMoment<T>& tail) {
  static_assert(Phase >= 0 && Phase < kPhaseCount);
  return scalar<T>(1.0f / bond_len((Phase + 1) % kPhaseCount)) * dot(pivot - prev, tail.moment - cross(pivot, tail.force));
}

template <int Phase, typename T>
static __device__ __forceinline__ RigidFrameT<T> rigid_step_transform(const Pair<T>& pair) {
  static_assert(Phase >= 0 && Phase < kPhaseCount);
  using R = typename Pair<T>::R;
  constexpr int lp = len_phase(Phase);
  constexpr int ap = angle_phase(Phase);
  const auto angle = bond_angle<ap>();
  const T sin_a = scalar<T>(static_cast<R>(angle.sin));
  const T cos_a = scalar<T>(static_cast<R>(angle.cos));
  const T sin_t = scalar<T>(pair.sin);
  const T cos_t = scalar<T>(pair.cos);
  const Vec3T<T> next_x = { -cos_a, sin_a * cos_t, sin_a * sin_t };
  return {
    .x = next_x,
    .z = { scalar<T>(0.0f), -sin_t, cos_t },
    .p = scalar<T>(bond_len(lp)) * next_x,
  };
}

template <typename T>
static __device__ __forceinline__ RigidFrameT<T> load_rigid_transform(const T* raw, int step) {
  switch (step % kPhaseCount) {
  case 0:
    return rigid_step_transform<0, T>(load_pair(raw, step));
  case 1:
    return rigid_step_transform<1, T>(load_pair(raw, step));
  default:
    return rigid_step_transform<2, T>(load_pair(raw, step));
  }
}

__host__ __device__ constexpr int group_count(int steps) {
  return (steps + kPhaseCount - 1) / kPhaseCount;
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads)
    forward_kernel(const T* __restrict__ raw_pairs, int64_t raw_stride, T* __restrict__ xyz, const __grid_constant__ Shape shape) {
  using Rigid = RigidFrameT<T>;
  __shared__ Vec3T<T> base_p;
  __shared__ Mat3T<T> base_frame;

  const int points = shape.points;
  const int steps = points - 3;
  const int64_t row = blockIdx.x;
  const T* raw = raw_pairs + row * raw_stride;
  T* out = xyz + row * points * 3;

  if (threadIdx.x == 0) {
    const BaseFrame<T> base = make_default_base_frame<T>();
    store_vec3(out, Vec3T<T>{});
    store_vec3(out + 3, base_p1<T>());
    store_vec3(out + 6, base.p2);
    base_p = base.p2;
    base_frame = base.frame;
  }

  Rigid input[ItemsPerThread];
  const int first_step = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    input[item] = step < steps ? load_rigid_transform(raw, step) : Rigid::identity();
  }

  const Rigid carry = cufk::scan::inclusive_scan_carry<BlockThreads>(input, input, multiplies<>{}, Rigid::identity());
  const Vec3T<T> carry_side = rigid_side(carry);
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    input[item].p = carry.p + rigid_apply_with_side(carry, carry_side, input[item].p);
  }

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int step = first_step + item;
    if (step < steps) {
      store_vec3(out + (step + 3) * 3, base_p + base_frame * input[item].p);
    }
  }
}

template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void __launch_bounds__(BlockThreads) backward_kernel(
    const T* __restrict__ raw_pairs,
    int64_t raw_stride,
    const T* __restrict__ xyz,
    const T* __restrict__ grad_points,
    T* __restrict__ grad_raw_pairs,
    int64_t grad_raw_stride,
    const __grid_constant__ Shape shape) {
  const int points = shape.points;
  const int steps = points - 3;
  const int groups = group_count(steps);
  const int64_t row = blockIdx.x;
  const T* raw = raw_pairs + row * raw_stride;
  const T* xyz_row = xyz + row * points * 3;
  const T* grad_row = grad_points + row * points * 3;
  T* grad_raw = grad_raw_pairs + row * grad_raw_stride;

  ForceMoment<T> input[ItemsPerThread];
  ForceMoment<T> suffix[ItemsPerThread];
  ForceMoment<T> point0_items[ItemsPerThread];
  ForceMoment<T> point1_items[ItemsPerThread];
  Vec3T<T> point0_pos[ItemsPerThread];
  Vec3T<T> point1_pos[ItemsPerThread];
  const int rgroup0 = threadIdx.x * ItemsPerThread;
  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int rgroup = rgroup0 + item;
    const int group = groups - 1 - rgroup;
    input[item] = {};
    point0_items[item] = {};
    point1_items[item] = {};
    point0_pos[item] = {};
    point1_pos[item] = {};
    if (group >= 0) {
      const int step = 3 * group;
      if (step < steps) {
        point0_pos[item] = load_vec3(xyz_row + (step + 3) * 3);
        point0_items[item] = world_force_moment(point0_pos[item], grad_row + (step + 3) * 3);
        input[item] = input[item] + point0_items[item];
      }
      if (step + 1 < steps) {
        point1_pos[item] = load_vec3(xyz_row + (step + 4) * 3);
        point1_items[item] = world_force_moment(point1_pos[item], grad_row + (step + 4) * 3);
        input[item] = input[item] + point1_items[item];
      }
      if (step + 2 < steps) {
        input[item] = input[item] + world_force_moment(load_vec3(xyz_row + (step + 5) * 3), grad_row + (step + 5) * 3);
      }
    }
  }

  cufk::scan::inclusive_scan<BlockThreads>(input, suffix, plus<>{}, ForceMoment<T>{});

  unroll for (int item = 0; item < ItemsPerThread; ++item) {
    const int rgroup = rgroup0 + item;
    const int group = groups - 1 - rgroup;
    if (group >= 0) {
      const int step = 3 * group;
      const ForceMoment<T> point0 = point0_items[item];
      const ForceMoment<T> point1 = point1_items[item];
      const ForceMoment<T> tail0 = suffix[item];
      const ForceMoment<T> tail1 = tail0 - point0;
      const ForceMoment<T> tail2 = tail1 - point1;
      Vec3T<T> pivot0 = {};
      if (step < steps) {
        Vec3T<T> prev;
        if (group == 0) {
          prev = base_p1<T>();
        } else if (item + 1 < ItemsPerThread) {
          prev = point1_pos[item + 1];
        } else {
          prev = load_vec3(xyz_row + (step + 1) * 3);
        }
        pivot0 = load_vec3(xyz_row + (step + 2) * 3);
        store_raw_pair_grad(grad_raw, raw, step, world_torsion_grad<0, T>(prev, pivot0, tail0));
      }
      if (step + 1 < steps) {
        store_raw_pair_grad(grad_raw, raw, step + 1, world_torsion_grad<1, T>(pivot0, point0_pos[item], tail1));
      }
      if (step + 2 < steps) {
        store_raw_pair_grad(grad_raw, raw, step + 2, world_torsion_grad<2, T>(point0_pos[item], point1_pos[item], tail2));
      }
    }
  }
}

struct LaunchContext : ChainLaunch {
  LaunchContext(const Tensor& ref, Shape shape_) : ChainLaunch(ref, shape_, kMaxSteps, "torsion_pairs") {}
};

template <typename T>
Tensor forward_default_impl(const Tensor& raw_pairs, Shape shape) {
  const LaunchContext ctx(raw_pairs, shape);
  Tensor xyz = empty_xyz_like(raw_pairs, ctx.shape);
  launch_block_plan<ForwardScanPlan>(ctx.steps, [&]<int Threads, int Items>() {
    constexpr auto kernel_ptr = forward_kernel<T, Threads, Items>;
    kernel_ptr<<<ctx.shape.batch, Threads, 0, ctx.stream>>>(kernel_arg<T>(raw_pairs), raw_pairs.stride(0), kernel_arg<T>(xyz), ctx.shape);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return xyz;
}

template <typename T>
Tensor backward_xyz_impl(const Tensor& raw_pairs, Shape shape, const Tensor& xyz, const Tensor& grad_points) {
  const LaunchContext ctx(raw_pairs, shape);
  Tensor grad = empty_strided_like(raw_pairs);
  const int groups = group_count(ctx.steps);
  launch_block_plan<BackwardScanPlan>(groups, [&]<int Threads, int Items>() {
    constexpr auto kernel_ptr = backward_kernel<T, Threads, Items>;
    kernel_ptr<<<ctx.shape.batch, Threads, 0, ctx.stream>>>(
        kernel_arg<T>(raw_pairs), raw_pairs.stride(0), kernel_arg<T>(xyz), kernel_arg<T>(grad_points), kernel_arg<T>(grad), grad.stride(0), ctx.shape);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad;
}

Shape check_pairs(const Tensor& raw_pairs) {
  check_cuda_float_row_contiguous(raw_pairs, "raw_pairs");
  TORCH_CHECK(raw_pairs.size(1) % 2 == 0, "raw_pairs second dimension must be even");
  return chain_shape_from_steps(raw_pairs, raw_pairs.size(1) / 2);
}

class TorsionAutograd : public torch::autograd::Function<TorsionAutograd> {
public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor raw_pairs) {
    const Shape shape = check_pairs(raw_pairs);
    Tensor xyz = dispatch_dtype(raw_pairs, [&]<typename T>() { return forward_default_impl<T>(raw_pairs, shape); });
    ctx->save_for_backward({ raw_pairs, xyz });
    return xyz;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const Tensor& raw_pairs = saved[0];
    const Tensor& xyz = saved[1];
    Tensor grad_points = contiguous_or_self(grad_outputs[0]);
    check_like(grad_points, raw_pairs, "grad_points");
    const Shape shape = check_pairs(raw_pairs);
    return { dispatch_dtype(raw_pairs, [&]<typename T>() { return backward_xyz_impl<T>(raw_pairs, shape, xyz, grad_points); }) };
  }
};

Tensor apply_autograd(const Tensor& raw_pairs) {
  return TorsionAutograd::apply(raw_pairs);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply", &apply_autograd, "torsion-only autograd apply");
}
