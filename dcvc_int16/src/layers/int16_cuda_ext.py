import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache

import torch
import torch.utils.cpp_extension as cpp_ext


_EXT = None
_LOAD_ERROR = None
_DLL_DIR_HANDLES = []


def _get_cuda_arch_flags():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def _get_sources():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(src_dir, "int16_kernels_bind.cpp"),
        os.path.join(src_dir, "int16_kernels.cu"),
    ]


def _prepend_path(path):
    if not path or not os.path.isdir(path):
        return
    entries = os.environ.get("PATH", "").split(os.pathsep)
    norm_path = os.path.normcase(os.path.abspath(path))
    if any(os.path.normcase(os.path.abspath(entry)) == norm_path for entry in entries if entry):
        return
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _ensure_python_scripts_on_path():
    scripts_dir = os.path.dirname(sys.executable)
    _prepend_path(scripts_dir)


def _ensure_matching_cuda_home():
    if os.name != "nt" or torch.version.cuda is None:
        return
    version = torch.version.cuda
    candidate = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}"
    if os.path.isdir(candidate):
        os.environ["CUDA_HOME"] = candidate
        os.environ["CUDA_PATH"] = candidate
        cpp_ext.CUDA_HOME = candidate
        _prepend_path(os.path.join(candidate, "bin"))


def _get_vcvars_candidates():
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
    if os.name != "nt":
        return True
    normalized = os.path.normcase(cl_path or "")
    return "\\14.29." in normalized or "\\14.3" in normalized


def _ensure_msvc_env():
    if os.name != "nt":
        return
    cl_path = shutil.which("cl")
    if cl_path and _cl_looks_cuda_12_compatible(cl_path):
        return

    vcvars_path = None
    vcvars_args = []
    for candidate, args in _get_vcvars_candidates():
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
            args = " ".join(vcvars_args)
            handle.write(f"call \"{vcvars_path}\" {args} >nul\n")
            handle.write("set\n")
        output = subprocess.check_output(
            ["cmd.exe", "/d", "/s", "/c", script_path],
            text=True,
        )
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _ensure_windows_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    global _DLL_DIR_HANDLES
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin"))
    candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    for candidate in candidates:
        if os.path.isdir(candidate):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(candidate))


@lru_cache(maxsize=1)
def load_int16_ext():
    global _EXT
    global _LOAD_ERROR

    if _EXT is not None:
        return _EXT
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot load int16 CUDA extension.")
    _ensure_python_scripts_on_path()
    _ensure_matching_cuda_home()
    _ensure_msvc_env()
    _ensure_windows_dll_dirs()

    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    extra_cflags = []
    if os.name == "nt":
        extra_cuda_cflags.extend(["-allow-unsupported-compiler", "-Xcompiler", "/Zc:preprocessor"])
        extra_cflags.append("/Zc:preprocessor")
        cl_path = shutil.which("cl")
        if cl_path:
            extra_cuda_cflags.extend(["-ccbin", cl_path])
    extra_cuda_cflags.extend(_get_cuda_arch_flags())
    extra_ldflags = ["cublas.lib"] if os.name == "nt" else ["-lcublas"]
    if os.name == "nt":
        python_lib_dir = os.path.join(sys.base_prefix, "libs")
        if os.path.isdir(python_lib_dir):
            extra_ldflags.append(f"/LIBPATH:{python_lib_dir}")
    try:
        _EXT = cpp_ext.load(
            name="int16_cuda_ext",
            sources=_get_sources(),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        return _EXT
    except Exception as exc:  # pylint: disable=broad-except
        _LOAD_ERROR = exc
        raise


def is_available():
    try:
        load_int16_ext()
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def conv2d_int16(
    input_tensor,
    weight,
    bias,
    stride=1,
    padding=0,
    groups=1,
    weight_i8=None,
    weight_int8_scale=1,
    activation_int8_scale=1,
    k2_layer=8192,
    activation_scale_c=None,
    scale_c=None,
    residual=None,
    residual2=None,
    post_scale=None,
):
    ext = load_int16_ext()
    if (
        weight.dim() == 4
        and weight.shape[2] == 1
        and weight.shape[3] == 1
        and stride == 1
        and padding == 0
        and groups == 1
    ):
        if (
            weight_i8 is not None
            and weight.shape[0] % 4 == 0
            and weight.shape[1] % 4 == 0
        ):
            if activation_scale_c is not None and scale_c is not None:
                return ext.conv1x1_int8tc_gemm_per_channel_v2(
                    input_tensor,
                    weight_i8.reshape(weight_i8.shape[0], weight_i8.shape[1]).contiguous(),
                    bias,
                    activation_scale_c.contiguous(),
                    scale_c.contiguous(),
                )
            if scale_c is not None:
                return ext.conv1x1_int8tc_gemm_per_channel(
                    input_tensor,
                    weight_i8.reshape(weight_i8.shape[0], weight_i8.shape[1]).contiguous(),
                    bias,
                    scale_c.contiguous(),
                    int(activation_int8_scale),
                )
            return ext.conv1x1_int8tc_gemm(
                input_tensor,
                weight_i8.reshape(weight_i8.shape[0], weight_i8.shape[1]).contiguous(),
                bias,
                int(weight_int8_scale),
                int(activation_int8_scale),
                int(k2_layer),
            )
        return ext.conv1x1_int16_gemm(
            input_tensor,
            weight.reshape(weight.shape[0], weight.shape[1]).contiguous(),
            bias,
            residual.contiguous() if residual is not None else None,
            residual2.contiguous() if residual2 is not None else None,
            post_scale.contiguous() if post_scale is not None else None,
            int(k2_layer),
        )
    return ext.conv2d_int16(input_tensor, weight, bias, residual, residual2, post_scale, stride, padding, groups)


def conv1x1_int16_gemm(input_tensor, weight_2d, bias, residual=None, residual2=None, post_scale=None, k2_layer=8192):
    ext = load_int16_ext()
    return ext.conv1x1_int16_gemm(input_tensor, weight_2d, bias, residual, residual2, post_scale, int(k2_layer))


def conv1x1_int8tc_gemm(
    input_tensor,
    weight_2d_i8,
    bias,
    weight_int8_scale=1,
    activation_int8_scale=1,
    k2_layer=8192,
):
    ext = load_int16_ext()
    return ext.conv1x1_int8tc_gemm(
        input_tensor,
        weight_2d_i8,
        bias,
        int(weight_int8_scale),
        int(activation_int8_scale),
        int(k2_layer),
    )


def conv1x1_int8tc_gemm_per_channel(
    input_tensor,
    weight_2d_i8,
    bias,
    scale_c,
    activation_int8_scale=1,
):
    ext = load_int16_ext()
    return ext.conv1x1_int8tc_gemm_per_channel(
        input_tensor,
        weight_2d_i8,
        bias,
        scale_c,
        int(activation_int8_scale),
    )


def conv1x1_int8tc_gemm_per_channel_v2(
    input_tensor,
    weight_2d_i8,
    bias,
    activation_scale_c,
    eff_scale_c,
):
    ext = load_int16_ext()
    return ext.conv1x1_int8tc_gemm_per_channel_v2(
        input_tensor,
        weight_2d_i8,
        bias,
        activation_scale_c,
        eff_scale_c,
    )


def lut_lookup_int16(x, lut):
    ext = load_int16_ext()
    return ext.sigmoid_lut_int16(x, lut)


def depthwise_conv3x3_lut_fused_int16(input_tensor, weight, bias, lut, stride=1, padding=1):
    ext = load_int16_ext()
    return ext.depthwise_conv3x3_lut_fused_int16(input_tensor, weight, bias, lut, stride, padding)


def scale_index_lut_int16(scales, lut):
    ext = load_int16_ext()
    return ext.scale_index_lut_int16(scales, lut)


def clamp_reciprocal_int16(q, k1=512):
    ext = load_int16_ext()
    return ext.clamp_reciprocal_int16(q, k1)


def add_multiply_int16(a, b, scale, k1=512):
    ext = load_int16_ext()
    return ext.add_multiply_int16(a, b, scale, k1)


def multiply_int16(input_tensor, scale, k1=512):
    ext = load_int16_ext()
    return ext.multiply_int16(input_tensor, scale, k1)


def wsilu_chunk_add_int16(input_tensor, lut):
    ext = load_int16_ext()
    return ext.wsilu_chunk_add_int16(input_tensor, lut)


def add_int16(a, b):
    ext = load_int16_ext()
    return ext.add_int16(a, b)


def conv1x1_int16_gemm_wsilu_chunk(input_tensor, weight_2d, bias, lut, k2_layer=8192):
    """Fused 1x1 conv + WSiLU chunk-add.

    Input:  [B, C_in, H, W]
    Weight: [2*half_C, C_in]  (row-major)
    Bias:   [2*half_C] or empty
    LUT:    WSiLU lookup table (65536 entries, int16)
    Output: [B, half_C, H, W]
    """
    ext = load_int16_ext()
    return ext.conv1x1_int16_gemm_wsilu_chunk(
        input_tensor, weight_2d, bias, lut, int(k2_layer)
    )
