import hashlib
import inspect
import math
from pathlib import Path
from typing import Final, Protocol, cast

import torch
from torch.utils.cpp_extension import load

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_TORSION_SOURCES: Final[list[str]] = [str(_PROJECT_ROOT / "csrc" / "torsion.cu")]
_CUDA_DEPENDENCIES: Final[list[str]] = [str(path) for path in sorted((_PROJECT_ROOT / "csrc").glob("*.cuh"))]
_LOAD_HAS_DEPENDS: Final[bool] = "depends" in inspect.signature(load).parameters


class TorsionReconstructor(Protocol):
    def __call__(self, raw_pairs: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct Cartesian coordinates from raw torsion pairs.

        Args:
            raw_pairs: Unnormalized torsion sine/cosine pairs. CUDA floating
                tensor shaped ``(B, 2 * (N - 3))`` with interleaved
                ``[..., sin_raw, cos_raw, ...]`` values.

        Returns:
            Cartesian coordinates shaped ``(B, N, 3)``. The output device and
            dtype match ``raw_pairs``.

        Notes:
            The CUDA kernel normalizes each torsion pair internally before
            reconstruction. Gradients are supported with respect to
            ``raw_pairs``.

            Bond lengths, bond angles, and canonical anchors are fixed when
            this callable is created by ``compile_torsion(...)``. Compile once,
            then call the returned reconstructor repeatedly during training:

            ```python
            reconstruct_pairs = cufk.compile_torsion(bond_lengths, bond_angles)
            xyz = reconstruct_pairs(raw_pairs)
            ```
        """
        ...


class _TorsionExtension(Protocol):
    apply: TorsionReconstructor


_TORSION_CALLS: dict[tuple[tuple[float, float, float], tuple[float, float, float]], TorsionReconstructor] = {}
_BOND_LENGTH_MACROS: Final[tuple[str, str, str]] = (
    "CUFK_TORSION_BOND_0_LENGTH",
    "CUFK_TORSION_BOND_1_LENGTH",
    "CUFK_TORSION_BOND_2_LENGTH",
)
_ANGLE_SIN_COS_MACROS: Final[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = (
    ("CUFK_TORSION_ANGLE_0_SIN", "CUFK_TORSION_ANGLE_0_COS"),
    ("CUFK_TORSION_ANGLE_1_SIN", "CUFK_TORSION_ANGLE_1_COS"),
    ("CUFK_TORSION_ANGLE_2_SIN", "CUFK_TORSION_ANGLE_2_COS"),
)


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _as_triple(values: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    assert len(values) == 3, f"{label} must have 3 elements"
    return (values[0], values[1], values[2])


def _define_float(name: str, value: float) -> str:
    return f"-D{name}={_format_float(value)}"


def _module_name(lengths: tuple[float, float, float], angles: tuple[float, float, float]) -> str:
    text = ",".join(_format_float(value) for value in (*lengths, *angles))
    return f"cufk_torsion_{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def _compile_flags(lengths: tuple[float, float, float], angles: tuple[float, float, float]) -> list[str]:
    flags = ["-O3", "-std=c++20", "--use_fast_math"]
    flags.extend(_define_float(name, length) for name, length in zip(_BOND_LENGTH_MACROS, lengths, strict=True))
    for (sin_name, cos_name), angle in zip(_ANGLE_SIN_COS_MACROS, angles, strict=True):
        flags.extend((_define_float(sin_name, math.sin(angle)), _define_float(cos_name, math.cos(angle))))
    return flags


def compile_torsion(
    bond_lengths: tuple[float, float, float],
    bond_angles: tuple[float, float, float],
) -> TorsionReconstructor:
    r"""Compile a torsion-only reconstructor for fixed backbone geometry.

    Args:
        bond_lengths: Three repeating bond lengths compiled as CUDA constants.
        bond_angles: Three repeating bond angles in radians. Python computes
            their sine/cosine values before JIT compilation.

    Returns:
        A cached callable ``reconstruct(raw_pairs) -> xyz``. Compile once, then
        call it repeatedly during training.

    Caching:
        Repeated calls with the same ``bond_lengths`` and ``bond_angles`` reuse
        the loaded JIT extension.

    Example:
        ```python
        reconstruct = cufk.compile_torsion(
            bond_lengths=(0.14685231, 0.15365194, 0.13405086),
            bond_angles=(1.9676635, 2.062335, 2.164016),
        )
        xyz = reconstruct(raw_pairs)
        ```
    """
    lengths = _as_triple(bond_lengths, "bond_lengths")
    angles = _as_triple(bond_angles, "bond_angles")
    key = (lengths, angles)
    if key not in _TORSION_CALLS:
        load_kwargs = {"depends": _CUDA_DEPENDENCIES} if _LOAD_HAS_DEPENDS else {}
        module = cast(
            _TorsionExtension,
            load(
                name=_module_name(lengths, angles),
                sources=_TORSION_SOURCES,
                extra_cuda_cflags=_compile_flags(lengths, angles),
                extra_cflags=["-O3"],
                with_cuda=True,
                verbose=False,
                **load_kwargs,  # type: ignore[arg-type]
            ),
        )
        _TORSION_CALLS[key] = module.apply
    return _TORSION_CALLS[key]
