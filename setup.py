from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc = Path("csrc")

setup(
    ext_modules=[
        CUDAExtension(
            "cufk._C",
            sources=[str(csrc / "block.cu")],
            depends=[str(path) for path in csrc.glob("*.cuh")],
            extra_compile_args={"nvcc": ["-O3", "-std=c++20"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
