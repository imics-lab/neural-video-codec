# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import sys
import shutil
import subprocess
import tempfile
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as cpp_ext


def _prepend_path(path):
    if not path or not os.path.isdir(path):
        return
    entries = os.environ.get("PATH", "").split(os.pathsep)
    norm_path = os.path.normcase(os.path.abspath(path))
    if any(os.path.normcase(os.path.abspath(entry)) == norm_path for entry in entries if entry):
        return
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _ensure_python_scripts_on_path():
    _prepend_path(os.path.dirname(sys.executable))


def _ensure_matching_cuda_home():
    if sys.platform != "win32" or torch.version.cuda is None:
        return
    candidate = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{torch.version.cuda}"
    if os.path.isdir(candidate):
        os.environ["CUDA_HOME"] = candidate
        os.environ["CUDA_PATH"] = candidate
        cpp_ext.CUDA_HOME = candidate
        _prepend_path(os.path.join(candidate, "bin"))


def _vcvars_candidates():
    return [
        (
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
            ["amd64", "-vcvars_ver=14.29"],
        ),
        (
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
            ["amd64", "-vcvars_ver=14.29"],
        ),
        (
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
            ["amd64", "-vcvars_ver=14.29"],
        ),
        (
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
            ["amd64", "-vcvars_ver=14.29"],
        ),
        (
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            [],
        ),
        (
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            [],
        ),
    ]


def _cl_looks_cuda_12_compatible(cl_path):
    normalized = os.path.normcase(cl_path or "")
    return "\\14.29." in normalized or "\\14.3" in normalized


def _ensure_msvc_env():
    if sys.platform != "win32":
        return
    cl_path = shutil.which("cl")
    if cl_path and _cl_looks_cuda_12_compatible(cl_path):
        return

    vcvars_path = None
    vcvars_args = []
    for candidate, args in _vcvars_candidates():
        if os.path.exists(candidate):
            vcvars_path = candidate
            vcvars_args = args
            break
    if vcvars_path is None:
        return

    script_fd, script_path = tempfile.mkstemp(suffix=".cmd")
    try:
        os.close(script_fd)
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write("@echo off\n")
            handle.write(f"call \"{vcvars_path}\" {' '.join(vcvars_args)} >nul\n")
            handle.write("set\n")
        output = subprocess.check_output(["cmd.exe", "/d", "/s", "/c", script_path], text=True)
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value
    os.environ["DISTUTILS_USE_SDK"] = "1"


_ensure_python_scripts_on_path()
_ensure_matching_cuda_home()
_ensure_msvc_env()


cxx_flags = ["-O3"]
nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-arch=native"]
if sys.platform == 'win32':
    cxx_flags = ["/O2"]
    nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"]


setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            sources=glob.glob('*.cpp') + glob.glob('*.cu'),
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
