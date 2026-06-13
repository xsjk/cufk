from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
CSRC = ROOT / "src" / "cufk" / "csrc"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


setup(
    ext_modules=[
        CUDAExtension(
            "cufk._C",
            sources=[_rel(CSRC / "block.cu")],
            depends=[_rel(path) for path in sorted(CSRC.glob("*.cuh"))],
            extra_compile_args={"nvcc": ["-O3", "-std=c++20"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
