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
reconstruct_pairs = cufk.compile_torsion(
    bond_lengths=(0.14685231, 0.15365194, 0.13405086),
    bond_angles=(1.9676635, 2.062335, 2.164016),
)
xyz_torsion = reconstruct_pairs(raw_pairs)
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

`reconstruct` uses the fused block affine kernel for `N <= 2051` through a C++ autograd `apply` wrapper. Backward uses one custom kernel that rebuilds the affine prefix in shared memory. Low-precision dtypes run the scan payload in the same low-precision type; this is experimental and can accumulate substantial error on long chains.

### Torsion-only path

```python
reconstruct_pairs = cufk.compile_torsion(bond_lengths, bond_angles)
xyz = reconstruct_pairs(raw_pairs)
```

This fixed-backbone fast path is for torsion-only models with compiled bond lengths and bond angles. It accepts interleaved raw network outputs `[sin_0, cos_0, sin_1, cos_1, ...]` shaped `(B, 2*(N-3))`, normalizes each raw `(sin, cos)` pair inside the CUDA kernel, and reconstructs xyz without materializing length, angle, or dihedral tensors.

Long-chain and multi-kernel comparison implementations live in `coord_scan_repro`; this package keeps only the block path used as the production interface.

## Implementation

This package keeps one generic production route: `block_forward_kernel` plus `block_backward_kernel`.

Scan choices are compile-time internals. Device-wide and tiled experiments live in `coord_scan_repro`; CUB is only a block-scan primitive here, not a public implementation.
