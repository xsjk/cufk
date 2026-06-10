# cufk

CUDA/PyTorch forward kinematics for reconstructing 3D polylines from internal coordinates.

## Install

```bash
TORCH_CUDA_ARCH_LIST="<your arch>" pip install -e . --no-build-isolation
```

Set `CUDA_HOME` if PyTorch cannot find CUDA automatically. The build does not ship or assume a machine-specific CUDA path.

## API

```python
import cufk

xyz = cufk.reconstruct(lengths, angles, dihedrals)
```

```python
reconstruct(
    lengths,
    angles,
    dihedrals,
    p0=None,
    first_direction=None,
    initial_normal=None,
) -> torch.Tensor
```

Inputs are CUDA `float32`, `float64`, `float16`, or `bfloat16` tensors shaped `(B, N-1)`, `(B, N-2)`, `(B, N-3)` or single-chain `(N-1)`, `(N-2)`, `(N-3)`. Output is `(B, N, 3)` or `(N, 3)` with the same dtype.

Optional anchors `p0`, `first_direction`, and `initial_normal` are shared `(3,)` CUDA tensors with the same dtype as `lengths`. Anchor gradients are not supported. Gradients are returned for `lengths`, `angles`, and `dihedrals`.

`reconstruct` uses the fused block affine kernel for `N <= 2051`. No-grad calls use the fastest forward-only path. Calls that need non-double gradients recompute the affine prefix inside the custom backward to avoid saving the full prefix tensor. `float64` keeps the saved-prefix backward path because full-size double prefixes are too large for the shared-memory recompute path. Low-precision dtypes run the scan payload in the same low-precision type; this is experimental and can accumulate substantial error on long chains.

Long-chain and multi-kernel comparison implementations live in `coord_scan_repro`; this package keeps only the block path used as the production interface.

## Implementation

This package keeps one production route: a fused block affine scan. No-grad and non-double autograd forward use `block_forward_kernel`; non-double backward uses `block_backward_recompute_kernel`; double backward uses the saved-prefix pair `block_forward_prefix_kernel` and `block_backward_kernel`.

Scan choices are compile-time internals. Device-wide and tiled experiments live in `coord_scan_repro`; CUB is only a block-scan primitive here, not a public implementation.
