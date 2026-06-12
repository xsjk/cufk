import torch

from . import _C

__version__ = "0.1.0"

BLOCK_SCAN_MAX_POINT_COUNT = 2051

__all__ = [
    "BLOCK_SCAN_MAX_POINT_COUNT",
    "__version__",
    "reconstruct",
]

_SUPPORTED_TYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16)


def _batched_inputs(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    assert torch.cuda.is_available(), "CUDA must be available"
    single = lengths.dim() == 1
    assert lengths.dim() in (1, 2), "lengths must have shape (N-1,) or (B, N-1)"
    assert angles.dim() == lengths.dim(), "angles must have the same rank as lengths"
    assert dihedrals.dim() == lengths.dim(), "dihedrals must have the same rank as lengths"
    assert lengths.is_cuda and angles.is_cuda and dihedrals.is_cuda, "inputs must be CUDA"
    assert lengths.dtype in _SUPPORTED_TYPES, "inputs must be float32, float64, float16, or bfloat16"
    assert angles.dtype == lengths.dtype and dihedrals.dtype == lengths.dtype, "inputs must have the same dtype"
    if single:
        lengths, angles, dihedrals = lengths[None], angles[None], dihedrals[None]
    batch, points = lengths.shape[0], lengths.shape[1] + 1
    assert points >= 4, "point_count must be >= 4"
    assert angles.shape == (batch, points - 2), "angles must have shape (B, N-2)"
    assert dihedrals.shape == (batch, points - 3), "dihedrals must have shape (B, N-3)"
    return lengths, angles, dihedrals, single


def _anchor(
    x: torch.Tensor | None,
    vals: tuple[float, float, float],
    ref: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if x is None:
        x = torch.tensor(vals, device=ref.device, dtype=ref.dtype)
    assert x.shape == (3,), f"{name} must have shape (3,)"
    assert x.device == ref.device, f"{name} must be on the same device as lengths"
    assert x.dtype == ref.dtype, f"{name} must have the same dtype as lengths"
    assert not x.requires_grad, f"{name} gradients are not supported"
    return x.contiguous()


def _anchors(
    p0: torch.Tensor | None,
    first_direction: torch.Tensor | None,
    initial_normal: torch.Tensor | None,
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _anchor(p0, (0.0, 0.0, 0.0), ref, "p0"),
        _anchor(first_direction, (1.0, 0.0, 0.0), ref, "first_direction"),
        _anchor(initial_normal, (0.0, 0.0, 1.0), ref, "initial_normal"),
    )


def _unbatch(xyz: torch.Tensor, single: bool) -> torch.Tensor:
    if single:
        return xyz[0]
    return xyz


class _ReconstructFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        dihedrals: torch.Tensor,
        p0: torch.Tensor,
        first_direction: torch.Tensor,
        initial_normal: torch.Tensor,
    ) -> torch.Tensor:
        lengths, angles, dihedrals = lengths.contiguous(), angles.contiguous(), dihedrals.contiguous()
        xyz = _C.forward(lengths, angles, dihedrals, p0, first_direction, initial_normal)
        ctx.save_for_backward(lengths, angles, dihedrals, p0, first_direction, initial_normal)
        return xyz

    @staticmethod
    def backward(ctx, grad_xyz: torch.Tensor):
        lengths, angles, dihedrals, p0, first_direction, initial_normal = ctx.saved_tensors
        grad_lengths, grad_angles, grad_dihedrals = _C.backward(
            lengths, angles, dihedrals, p0, first_direction, initial_normal, grad_xyz.contiguous()
        )
        return grad_lengths, grad_angles, grad_dihedrals, None, None, None


def reconstruct(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
    p0: torch.Tensor | None = None,
    first_direction: torch.Tensor | None = None,
    initial_normal: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Reconstruct Cartesian points from internal coordinates.

    Args:
        lengths: CUDA tensor shaped ``(N - 1,)`` or ``(B, N - 1)``.
        angles: CUDA tensor shaped ``(N - 2,)`` or ``(B, N - 2)``.
        dihedrals: CUDA tensor shaped ``(N - 3,)`` or ``(B, N - 3)``.
        p0: Shared first point, shape ``(3,)``.
        first_direction: Shared first-bond direction, shape ``(3,)``.
        initial_normal: Shared initial plane normal, shape ``(3,)``.

    Returns:
        CUDA tensor shaped ``(N, 3)`` or ``(B, N, 3)`` with the input dtype.

    Supports ``float16``, ``bfloat16``, ``float32``, and ``float64`` for up to
    ``BLOCK_SCAN_MAX_POINT_COUNT`` points. Gradients are supported for internal
    coordinates only.
    """
    lengths, angles, dihedrals, single = _batched_inputs(lengths, angles, dihedrals)
    p0, first_direction, initial_normal = _anchors(p0, first_direction, initial_normal, lengths)
    assert lengths.shape[1] + 1 <= BLOCK_SCAN_MAX_POINT_COUNT, f"reconstruct supports at most {BLOCK_SCAN_MAX_POINT_COUNT} points"
    if torch.is_grad_enabled() and (lengths.requires_grad or angles.requires_grad or dihedrals.requires_grad):
        xyz = _ReconstructFn.apply(lengths, angles, dihedrals, p0, first_direction, initial_normal)
    else:
        xyz = _C.forward(lengths.contiguous(), angles.contiguous(), dihedrals.contiguous(), p0, first_direction, initial_normal)
    return _unbatch(xyz, single)
