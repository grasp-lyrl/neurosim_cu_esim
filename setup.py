"""Build script for the CUDA extension.

``pyproject.toml`` declares the project metadata; this file only exists
because ``torch.utils.cpp_extension.CUDAExtension`` needs a classic
``setup()`` call to compile the CUDA kernel at install time.

Install with::

    pip install .           # release
    pip install -e .        # editable / dev
    pip install -e ".[dev]" # editable + test / lint deps
"""

import glob
import os
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = osp.dirname(osp.abspath(__file__))

# All paths relative to the project root — required by setuptools editable mode.
_csrc = osp.join(HERE, "csrc")
include_dirs = [osp.join(_csrc, "include")]

_abs_sources = glob.glob(osp.join(_csrc, "*.cpp")) + glob.glob(osp.join(_csrc, "*.cu"))
sources = [osp.relpath(s, HERE) for s in _abs_sources]

# ---- NVCC flags ----------------------------------------------------------
# Compute capabilities:
#   70 = V100          (CUDA >=11.8)
#   75 = T4 / Turing   (CUDA >=11.8)
#   80 = A100 / Ampere  (CUDA >=11.8)
#   86 = RTX 3090 etc.  (CUDA >=11.8)
#   89 = RTX 4090 / Ada (CUDA >=11.8)
#   90 = H100 / Hopper  (CUDA >=12.0)
#
# Users can extend this list via the TORCH_CUDA_ARCH_LIST env-var which
# torch.utils.cpp_extension honours automatically.

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--restrict",
    "-std=c++17",
    "-Xcompiler",
    "-fPIC",
    # Broad GPU support (CUDA >= 11.8)
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
]

# Optionally add sm_90 (Hopper) when CUDA >= 12.0 is available
try:
    import torch

    cuda_version = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    if cuda_version >= (12, 0):
        nvcc_flags.append("-gencode=arch=compute_90,code=sm_90")
except Exception:
    pass

cxx_flags = [
    "-O3",
    "-std=c++17",
    "-Wno-sign-compare",
    "-fPIC",
]

# Allow overriding the host compiler (useful in conda envs)
if "CC" not in os.environ:
    os.environ.setdefault("CC", "gcc")
if "CXX" not in os.environ:
    os.environ.setdefault("CXX", "g++")

setup(
    ext_modules=[
        CUDAExtension(
            name="_neurosim_cu_esim_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
