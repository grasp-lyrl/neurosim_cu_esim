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
import re

from setuptools import find_packages, setup
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
    # GCC 13+ rejects a missing `typename` in PyTorch's ATen/core/List_inl.h
    # (dependent decltype context).  -fpermissive is a safety net; the real
    # fix is the header patch applied by _PatchedBuildExtension below.
    "-Xcompiler",
    "-fpermissive",
    # Broad GPU support (CUDA >= 11.8)
    #"-gencode=arch=compute_70,code=sm_70",
    #"-gencode=arch=compute_75,code=sm_75",
    #"-gencode=arch=compute_80,code=sm_80",
    #"-gencode=arch=compute_86,code=sm_86",
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

# ---- GCC 13+ workaround for PyTorch header bug --------------------------
# PyTorch <= 2.11 has a bug in ATen/core/List_inl.h (operator[]):
#
#   return {impl_->list.begin() + static_cast<typename decltype(impl_->list)::difference_type>(pos)};
#
# Two issues with GCC 13+:
#   1. `typename` is placed *inside* the static_cast angle brackets but GCC
#      complains it needs `typename` before the `decltype(...)::nested` —
#      this is actually a GCC 13 parsing quirk where the `typename` is not
#      recognised in that position for dependent decltype scopes.
#   2. The brace-init return `{iterator}` can't be converted to the
#      `internal_reference_type` (a cascading failure from issue 1).
#
# The fix: rewrite the return to construct the return type explicitly,
# avoiding both the dependent-typename parse issue and the brace-init
# conversion failure.

def _patch_pytorch_list_inl_header():
    """Patch ATen/core/List_inl.h in the torch include directory if needed."""
    try:
        import torch
        torch_include = osp.join(osp.dirname(torch.__file__), "include")
        header = osp.join(torch_include, "ATen", "core", "List_inl.h")
        if not osp.isfile(header):
            return

        with open(header, "r") as f:
            content = f.read()

        # Already patched?
        if "// PATCHED-GCC13" in content:
            return

        # The buggy line (may appear once):
        buggy = (
            "return {impl_->list.begin() + "
            "static_cast<typename decltype(impl_->list)::difference_type>(pos)};"
        )

        if buggy not in content:
            return

        # Replacement: avoid decltype(impl_->list) entirely — GCC 13 under
        # nvcc rejects `typename` before dependent-decltype member scopes.
        # std::ptrdiff_t is what std::vector::difference_type resolves to.
        fixed = (
            "// PATCHED-GCC13: work around dependent-typename + brace-init bug\n"
            "  return internal_reference_type(impl_->list.begin() + static_cast<std::ptrdiff_t>(pos));"
        )

        content = content.replace(buggy, fixed)

        with open(header, "w") as f:
            f.write(content)

        print(f"[neurosim_cu_esim] Patched {header} for GCC 13+ compatibility")

    except Exception as e:
        print(f"[neurosim_cu_esim] Warning: could not patch List_inl.h: {e}")


class _PatchedBuildExtension(BuildExtension):
    """BuildExtension subclass that patches PyTorch headers before building."""

    def build_extensions(self):
        _patch_pytorch_list_inl_header()
        super().build_extensions()


setup(
    packages=find_packages(include=["neurosim_cu_esim", "neurosim_cu_esim.*"]),
    ext_modules=[
        CUDAExtension(
            name="_neurosim_cu_esim_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
    ],
    cmdclass={"build_ext": _PatchedBuildExtension},
)
