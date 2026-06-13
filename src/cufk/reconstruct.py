import torch

from . import _C

BLOCK_SCAN_MAX_POINT_COUNT = 2051


def _batched_inputs(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    single = lengths.dim() == 1
    if single:
        lengths, angles, dihedrals = lengths[None], angles[None], dihedrals[None]
    return lengths, angles, dihedrals, single


def _anchor(
    x: torch.Tensor | None,
    vals: tuple[float, float, float],
    ref: torch.Tensor,
) -> torch.Tensor:
    if x is None:
        x = torch.tensor(vals, device=ref.device, dtype=ref.dtype)
    return x


def reconstruct(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
    p0: torch.Tensor | None = None,
    first_direction: torch.Tensor | None = None,
    initial_normal: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Reconstruct Cartesian coordinates from full internal coordinates.

    Args:
        lengths: Bond lengths. CUDA tensor shaped ``(N - 1,)`` for one chain
            or ``(B, N - 1)`` for a batch.
        angles: Bond angles in radians. CUDA tensor shaped ``(N - 2,)`` or
            ``(B, N - 2)``.
        dihedrals: Dihedral angles in radians. CUDA tensor shaped ``(N - 3,)``
            or ``(B, N - 3)``.
        p0: Optional first Cartesian point, shaped ``(3,)``. Defaults to
            ``(0, 0, 0)`` on the same device and dtype as ``lengths``.
        first_direction: Optional first-bond direction, shaped ``(3,)``.
            Defaults to ``(1, 0, 0)``.
        initial_normal: Optional initial plane normal, shaped ``(3,)``.
            Defaults to ``(0, 0, 1)``.

    Returns:
        Cartesian coordinates shaped ``(N, 3)`` for one chain or ``(B, N, 3)``
        for batched input. The output dtype matches ``lengths``.

    Notes:
        All inputs must be CUDA tensors with the same dtype. Supported dtypes
        are ``float16``, ``bfloat16``, ``float32``, and ``float64``.

        Gradients are supported for ``lengths``, ``angles``, and ``dihedrals``.
        Gradients for anchors are intentionally not supported.

        This is the generic reconstruction path. If bond lengths and bond
        angles are fixed constants and the model only predicts raw torsion
        pairs, use ``compile_torsion(...)`` instead.
    """
    lengths, angles, dihedrals, single = _batched_inputs(lengths, angles, dihedrals)
    p0 = _anchor(p0, (0.0, 0.0, 0.0), lengths)
    first_direction = _anchor(first_direction, (1.0, 0.0, 0.0), lengths)
    initial_normal = _anchor(initial_normal, (0.0, 0.0, 1.0), lengths)

    xyz = _C.apply(
        lengths.contiguous(),
        angles.contiguous(),
        dihedrals.contiguous(),
        p0,
        first_direction,
        initial_normal,
    )
    return xyz[0] if single else xyz
