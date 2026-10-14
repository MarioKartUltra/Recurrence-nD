# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Build script for the recurrence2d pybind11 extension.

Compiles every .cpp under ../generalized/ together with src/bindings.cpp.
Handles OpenMP via libomp on macOS (Homebrew) and -fopenmp elsewhere.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


HERE = Path(__file__).parent.resolve()
GEN  = (HERE.parent / "generalized").resolve()

# setuptools requires source paths to be relative to setup.py.
def _rel(p: Path) -> str:
    return os.path.relpath(str(p), start=str(HERE))

cxx_sources = [
    _rel(GEN / "GeneralizedRecurrencePlot.cpp"),
    _rel(GEN / "SpatialRQA.cpp"),
    _rel(GEN / "SpatialRecurrenceNetwork.cpp"),
    _rel(GEN / "JointGeneralizedRecurrencePlot.cpp"),
    _rel(GEN / "CrossGeneralizedRecurrencePlot.cpp"),
    _rel(GEN / "SpatialSlidingWindow.cpp"),
    "src/bindings.cpp",
]

extra_compile = ["-O3", "-std=c++17"]
extra_link = []

# OpenMP selection — mirrors the Makefile logic for the C++ binaries.
if sys.platform == "darwin":
    # Detect Apple clang vs Homebrew gcc via $CXX.
    cxx = os.environ.get("CXX", "").lower()
    is_apple_clang = not cxx or ("clang" in cxx) or ("g++" in cxx and "gcc" not in cxx)
    try:
        out = subprocess.run(
            [os.environ.get("CXX", "g++"), "--version"],
            capture_output=True, text=True, timeout=5,
        ).stdout.lower()
        if "apple clang" in out:
            is_apple_clang = True
    except Exception:
        pass

    if is_apple_clang:
        try:
            libomp_prefix = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            raise RuntimeError(
                "libomp not found. Install with: brew install libomp"
            )
        extra_compile += [
            "-Xpreprocessor", "-fopenmp",
            f"-I{libomp_prefix}/include",
        ]
        extra_link += [f"-L{libomp_prefix}/lib", "-lomp"]
    else:
        extra_compile += ["-fopenmp"]
        extra_link += ["-fopenmp"]
else:
    extra_compile += ["-fopenmp"]
    extra_link += ["-fopenmp"]


ext_modules = [
    Pybind11Extension(
        "recurrence2d._core",
        cxx_sources,
        include_dirs=[str(GEN)],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
        cxx_std=17,
    ),
]

setup(
    name="recurrence2d",
    version="0.1.0",
    description="2D generalised recurrence plots, networks, "
                "joint/cross RP, sliding windows. OpenMP C++ backend.",
    packages=["recurrence2d"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["numpy>=1.20"],
    extras_require={
        "raster": ["rasterio>=1.3"],
        "plot":   ["matplotlib>=3.5"],
        "all":    ["rasterio>=1.3", "matplotlib>=3.5"],
    },
)
