import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


USE_CUDA_KERNELS = False
INT16_CUDA_EXT_ERROR = None
try:
    from .int16_cuda_ext import (
        conv1x1_int16_gemm_wsilu_chunk as _cuda_conv1x1_int16_gemm_wsilu_chunk,
        add_multiply_int16 as _cuda_add_multiply_int16,
        clamp_reciprocal_int16 as _cuda_clamp_reciprocal_int16,
        conv2d_int16 as _cuda_conv2d_int16,
        depthwise_conv3x3_lut_fused_int16 as _cuda_depthwise_conv3x3_lut_fused_int16,
        lut_lookup_int16 as _cuda_lut_lookup_int16,
        multiply_int16 as _cuda_multiply_int16,
        scale_index_lut_int16 as _cuda_scale_index_lut_int16,
        wsilu_chunk_add_int16 as _cuda_wsilu_chunk_add_int16,
        add_int16 as _cuda_add_int16,
    )
    USE_CUDA_KERNELS = True
except Exception as exc:  # pylint: disable=broad-except
    INT16_CUDA_EXT_ERROR = exc
    USE_CUDA_KERNELS = False

if not USE_CUDA_KERNELS and "SUPPRESS_INT16_KERNEL_WARNING" not in os.environ:
    if INT16_CUDA_EXT_ERROR is None:
        print("[int16_backend] CUDA kernels not available, using slow Python reference path.")
    else:
        print(
            f"[int16_backend] CUDA kernels unavailable, using slow Python reference path: "
            f"{INT16_CUDA_EXT_ERROR}"
        )


INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1


# Layers eligible for INT8 Tensor Core routing.
# These are the heavy 1×1 pointwise convolutions in the main feature coding
# path (DepthConvBlock FFN and DC projections).  They dominate the GPU profile
# (~50–60 ms/frame each) and are not quality-critical for the entropy path.
# After QAT training the weights are already adapted to INT8 precision, so
# routing them through cublasGemmEx INT8→INT32 gives tensor-core speedup while
# integer accumulation keeps the output bit-exact across GPU architectures.
INT8_ELIGIBLE_LAYERS = frozenset((
    "ffn_conv1",
    "ffn_conv2",
    "dc_conv1",
    "dc_conv2",
))

# These layers remain in INT16 regardless of the INT8 gate.
# They cover the entropy parameter path, hyperprior, and pixel-output heads —
# all of which are quality-sensitive and calibrated to the INT16 scale contract.
INT8_BLOCKED_LAYERS = frozenset((
    "DMC.decoder.conv2",
    "y_spatial_prior",
    "y_prior_fusion",
    "feature_adaptor_i",
    "feature_adaptor_p",
    "hyper_enc",
    "hyper_encoder",
    "hyper_dec",
    "hyper_decoder",
    "temporal_prior_encoder",
    "recon_generation.head",
))


def int8_tensor_cores_enabled():
    return os.environ.get("DCVC_ENABLE_INT8_TC", "0") == "1"


def _normalize_module_name(module_name):
    return (module_name or "").replace(".params", "").replace(".down_params", ".down")


def _matches_any(module_name, patterns):
    module_name = _normalize_module_name(module_name)
    return any(pattern in module_name for pattern in patterns)


def is_int8_layer_eligible(module_name):
    single_layer = os.environ.get("DCVC_INT8_SINGLE_LAYER", "").strip()
    if single_layer:
        patterns = [_normalize_module_name(part.strip()) for part in single_layer.split(",") if part.strip()]
        normalized = _normalize_module_name(module_name)
        return any(pattern in normalized for pattern in patterns)
    normalized = _normalize_module_name(module_name)
    eligible = _matches_any(normalized, INT8_ELIGIBLE_LAYERS)
    blocked = _matches_any(normalized, INT8_BLOCKED_LAYERS)
    return eligible and not blocked


def is_int8_kernel_candidate(params):
    return (
        params.weight_int8 is not None
        and params.groups == 1
        and params.stride == 1
        and params.padding == 0
        and params.weight.dim() == 4
        and params.weight.shape[2] == 1
        and params.weight.shape[3] == 1
        and (params.weight.shape[0] % 4 == 0)
        and (params.weight.shape[1] % 4 == 0)
    )


def iter_conv_param_specs(node, module_name):
    if isinstance(node, dict):
        if "weight" in node and isinstance(node["weight"], torch.Tensor):
            yield _normalize_module_name(module_name), node
            return

        kind = node.get("kind")
        if kind == "conv2d":
            yield _normalize_module_name(module_name), node["params"]
            return
        if kind == "sequential":
            for idx, child in enumerate(node["modules"]):
                yield from iter_conv_param_specs(child, f"{module_name}.modules[{idx}]")
            return
        if kind == "depth_conv_block":
            if node["params"].get("adaptor") is not None:
                yield _normalize_module_name(f"{module_name}.adaptor"), node["params"]["adaptor"]
            for child_name in ("dc_conv1", "dc_depth_conv", "dc_conv2", "ffn_conv1", "ffn_conv2"):
                yield _normalize_module_name(f"{module_name}.{child_name}"), node["params"][child_name]
            return
        if kind == "subpel_conv2x":
            yield _normalize_module_name(f"{module_name}.conv"), node["params"]["conv"]
            return
        if kind == "residual_block_stride2":
            yield _normalize_module_name(f"{module_name}.down"), node["down_params"]
            yield from iter_conv_param_specs(node["conv"], f"{module_name}.conv")
            return
        if kind == "residual_block_upsample":
            yield from iter_conv_param_specs(node["up"], f"{module_name}.up")
            yield from iter_conv_param_specs(node["conv"], f"{module_name}.conv")
            return

        for key, value in node.items():
            if isinstance(value, (dict, list, tuple)):
                yield from iter_conv_param_specs(value, f"{module_name}.{key}")
    elif isinstance(node, (list, tuple)):
        for idx, value in enumerate(node):
            yield from iter_conv_param_specs(value, f"{module_name}[{idx}]")


def describe_int8_routing(spec, module_name, activation_scales=None):
    activation_scales = activation_scales or {}
    routes = []
    for conv_name, params in iter_conv_param_specs(spec, module_name):
        weight = params.get("weight")
        groups = params.get("groups")
        weight_scale_channel = params.get("weight_int8_channel_scale")
        activation_scale_channel = params.get("activation_int8_channel_scale")
        if activation_scale_channel is None:
            activation_scale_channel = activation_scales.get(conv_name)
        if isinstance(weight_scale_channel, torch.Tensor) and weight_scale_channel.numel() > 0:
            weight_scale_mean = float(weight_scale_channel.to(torch.float32).mean().item())
            weight_scale_max = int(weight_scale_channel.max().item())
        else:
            weight_scale_mean = float(params.get("weight_int8_scale", 1))
            weight_scale_max = int(params.get("weight_int8_scale", 1))
        if isinstance(activation_scale_channel, torch.Tensor) and activation_scale_channel.numel() > 0:
            activation_scale_mean = float(activation_scale_channel.to(torch.float32).mean().item())
            activation_scale_max = int(activation_scale_channel.max().item())
            activation_scale_pct_eq_1 = float(
                (activation_scale_channel == 1).to(torch.float32).mean().item() * 100.0
            )
        else:
            activation_scale_value = int(
                activation_scales.get(
                    conv_name,
                    params.get("activation_int8_scale", 1),
                )
            )
            activation_scale_mean = float(activation_scale_value)
            activation_scale_max = activation_scale_value
            activation_scale_pct_eq_1 = 100.0 if activation_scale_value == 1 else 0.0
        candidate = (
            isinstance(weight, torch.Tensor)
            and weight.dim() == 4
            and weight.shape[2] == 1
            and weight.shape[3] == 1
            and groups == 1
            and params.get("weight_int8") is not None
            and weight.shape[0] % 4 == 0
            and weight.shape[1] % 4 == 0
        )
        eligible = is_int8_layer_eligible(conv_name)
        routes.append(
            {
                "module_name": conv_name,
                "candidate": bool(candidate),
                "eligible": bool(eligible),
                "use_int8": bool(candidate and eligible and int8_tensor_cores_enabled()),
                "weight_int8_scale": int(params.get("weight_int8_scale", 1)),
                "weight_int8_scale_mean": weight_scale_mean,
                "weight_int8_scale_max": weight_scale_max,
                "activation_int8_scale": activation_scale_max,
                "activation_int8_scale_mean": activation_scale_mean,
                "activation_int8_scale_max": activation_scale_max,
                "activation_int8_scale_pct_eq_1": activation_scale_pct_eq_1,
            }
        )
    return routes


@dataclass(frozen=True)
class Int16QuantConfig:
    # Section 4.5 of the DCVC-RT paper uses 16-bit model integerization.
    feature_scale: int = 512
    weight_scale: int = 8192

    @property
    def bias_scale(self):
        return self.feature_scale * self.weight_scale


def clamp_int16(tensor):
    return tensor.clamp(INT16_MIN, INT16_MAX).to(torch.int16)


def quantize_feature(tensor, scale):
    return clamp_int16(torch.round(tensor * float(scale)))


def quantize_weight(tensor, scale):
    return clamp_int16(torch.round(tensor * float(scale)))


def quantize_bias(tensor, scale):
    return torch.round(tensor * float(scale)).to(torch.int32)


def dequantize_int16(tensor, scale):
    return tensor.to(torch.float32) / float(scale)


def build_sigmoid_lut(input_scale=512, output_scale=512):
    x = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.float32) / float(input_scale)
    y = torch.sigmoid(x) * float(output_scale)
    return clamp_int16(torch.round(y))


def build_wsilu_lut(feature_scale=512):
    x = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.float32) / float(feature_scale)
    y = torch.sigmoid(4.0 * x) * x
    return clamp_int16(torch.round(y * float(feature_scale)))


def _pick_power_of_two_scale(max_abs, limit=127):
    scale = 1
    while max_abs > limit * scale:
        scale <<= 1
    return scale


def pack_weight_to_int8(weight_int16):
    weight_int16 = weight_int16.to(torch.int16)
    max_abs = int(weight_int16.abs().max().item()) if weight_int16.numel() > 0 else 0
    secondary_scale = _pick_power_of_two_scale(max_abs)
    if secondary_scale == 1:
        weight_i8 = weight_int16.clamp(-127, 127).to(torch.int8)
    else:
        rounded = torch.round(weight_int16.to(torch.float32) / float(secondary_scale))
        weight_i8 = rounded.clamp(-127, 127).to(torch.int8)
    return weight_i8, secondary_scale


def pack_weight_to_int8_per_channel(weight_int16_2d):
    weight_int16_2d = weight_int16_2d.to(torch.int16)
    if weight_int16_2d.dim() != 2:
        raise RuntimeError(
            f"Expected a 2D [C_out, C_in] tensor for per-channel packing, got {tuple(weight_int16_2d.shape)}"
        )
    if weight_int16_2d.numel() == 0:
        return weight_int16_2d.to(torch.int8), torch.ones(0, dtype=torch.int32)

    max_abs = weight_int16_2d.abs().amax(dim=1).to(torch.int32)
    scales = torch.ones_like(max_abs, dtype=torch.int32)
    while True:
        too_large = max_abs > (127 * scales)
        if not bool(too_large.any()):
            break
        scales = torch.where(too_large, scales << 1, scales)

    packed = torch.round(
        weight_int16_2d.to(torch.float32) / scales.to(torch.float32).unsqueeze(1)
    ).clamp(-127, 127).to(torch.int8)
    return packed, scales.to(torch.int32)


def compute_effective_output_scale(weight_int8, weight_scale_c, activation_scale_c, weight_scale=8192):
    if weight_int8.dim() == 4:
        if weight_int8.shape[2] != 1 or weight_int8.shape[3] != 1:
            raise RuntimeError(
                "compute_effective_output_scale expects a 1x1 convolution weight tensor."
            )
        weight_int8 = weight_int8.view(weight_int8.shape[0], weight_int8.shape[1])
    if weight_int8.dim() != 2:
        raise RuntimeError(
            f"Expected weight_int8 to be 2D or 1x1 4D, got {tuple(weight_int8.shape)}"
        )

    weight_scale_c = weight_scale_c.to(dtype=torch.float32)
    activation_scale_c = activation_scale_c.to(dtype=torch.float32)
    weight_abs = weight_int8.to(dtype=torch.float32).abs()
    if weight_abs.shape[0] != weight_scale_c.numel():
        raise RuntimeError("weight_scale_c must have one value per output channel.")
    if weight_abs.shape[1] != activation_scale_c.numel():
        raise RuntimeError("activation_scale_c must have one value per input channel.")

    weight_sum = weight_abs.sum(dim=1).clamp(min=1.0)
    weighted_activation = (weight_abs * activation_scale_c.unsqueeze(0)).sum(dim=1)
    mean_activation = weighted_activation / weight_sum
    return (weight_scale_c * mean_activation / float(weight_scale)).to(torch.float32)


def choose_activation_int8_scale(tensor, current_scale=1):
    if current_scale is None or current_scale < 1:
        current_scale = 1
    max_abs = int(tensor.detach().abs().amax().item()) if tensor.numel() > 0 else 0
    return max(current_scale, _pick_power_of_two_scale(max_abs))


@dataclass
class Conv2dInt16Params:
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    k2_layer: int = 8192
    weight_int8: Optional[torch.Tensor] = None
    weight_int8_scale: int = 1
    weight_int8_channel_scale: Optional[torch.Tensor] = None
    activation_int8_scale: int = 0
    activation_int8_channel_scale: Optional[torch.Tensor] = None
    eff_scale_c: Optional[torch.Tensor] = None
    weight_int8_runtime_scale: Optional[torch.Tensor] = None
    module_name: str = ""
    int8_candidate: bool = False
    use_int8: bool = False
    activation_observer: Optional[Callable[[str, torch.Tensor], None]] = None
    stride: int = 1
    padding: int = 0
    groups: int = 1

    @classmethod
    def from_float_module(cls, module, quant_cfg):
        weight_i16 = quantize_weight(module.weight.detach().cpu(), quant_cfg.weight_scale)
        weight_i8_scale = 1
        weight_i8_channel_scale = None
        if weight_i16.dim() == 4 and weight_i16.shape[2] == 1 and weight_i16.shape[3] == 1:
            weight_i8_2d, weight_i8_channel_scale = pack_weight_to_int8_per_channel(
                weight_i16.view(weight_i16.shape[0], weight_i16.shape[1])
            )
            weight_i8 = weight_i8_2d.view_as(weight_i16)
        else:
            weight_i8, weight_i8_scale = pack_weight_to_int8(weight_i16)
        bias = None
        if module.bias is not None:
            bias = quantize_bias(module.bias.detach().cpu(), quant_cfg.bias_scale)
        return cls(
            weight=weight_i16,
            bias=bias,
            k2_layer=quant_cfg.weight_scale,
            weight_int8=weight_i8,
            weight_int8_scale=weight_i8_scale,
            weight_int8_channel_scale=weight_i8_channel_scale,
            activation_int8_scale=1,
            activation_int8_channel_scale=(
                torch.ones(weight_i16.shape[1], dtype=torch.int32)
                if weight_i16.dim() == 4 and weight_i16.shape[2] == 1 and weight_i16.shape[3] == 1
                else None
            ),
            eff_scale_c=(
                compute_effective_output_scale(
                    weight_i8_2d,
                    weight_i8_channel_scale,
                    torch.ones(weight_i16.shape[1], dtype=torch.int32),
                    weight_scale=quant_cfg.weight_scale,
                )
                if weight_i16.dim() == 4 and weight_i16.shape[2] == 1 and weight_i16.shape[3] == 1
                else None
            ),
            weight_int8_runtime_scale=None,
            module_name="",
            int8_candidate=False,
            use_int8=False,
            activation_observer=None,
            stride=module.stride[0],
            padding=module.padding[0],
            groups=module.groups,
        )

    def to_dict(self):
        return {
            "weight": self.weight,
            "bias": self.bias,
            "k2_layer": self.k2_layer,
            "weight_int8": self.weight_int8,
            "weight_int8_scale": self.weight_int8_scale,
            "weight_int8_channel_scale": self.weight_int8_channel_scale,
            "activation_int8_scale": self.activation_int8_scale,
            "activation_int8_channel_scale": self.activation_int8_channel_scale,
            "eff_scale_c": self.eff_scale_c,
            "stride": self.stride,
            "padding": self.padding,
            "groups": self.groups,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            weight=payload["weight"],
            bias=payload["bias"],
            k2_layer=payload.get("k2_layer", Int16QuantConfig().weight_scale),
            weight_int8=payload.get("weight_int8"),
            weight_int8_scale=payload.get("weight_int8_scale", 1),
            weight_int8_channel_scale=payload.get("weight_int8_channel_scale"),
            activation_int8_scale=payload.get("activation_int8_scale", 1),
            activation_int8_channel_scale=payload.get("activation_int8_channel_scale"),
            eff_scale_c=payload.get("eff_scale_c"),
            weight_int8_runtime_scale=None,
            module_name="",
            int8_candidate=False,
            use_int8=False,
            activation_observer=None,
            stride=payload["stride"],
            padding=payload["padding"],
            groups=payload["groups"],
        )

    def to(self, device):
        self.weight = self.weight.to(device=device, dtype=torch.int16, non_blocking=True)
        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=torch.int32, non_blocking=True)
        if self.weight_int8 is not None:
            self.weight_int8 = self.weight_int8.to(device=device, dtype=torch.int8, non_blocking=True)
        if self.weight_int8_channel_scale is not None:
            self.weight_int8_channel_scale = self.weight_int8_channel_scale.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if self.activation_int8_channel_scale is not None:
            self.activation_int8_channel_scale = self.activation_int8_channel_scale.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if self.eff_scale_c is not None:
            self.eff_scale_c = self.eff_scale_c.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        refresh_int8_runtime_scale(self)
        return self


@dataclass
class SubpelConv2xInt16Params:
    conv: Conv2dInt16Params

    @classmethod
    def from_float_module(cls, module, quant_cfg):
        return cls(conv=Conv2dInt16Params.from_float_module(module.conv[0], quant_cfg))

    def to_dict(self):
        return {
            "conv": self.conv.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(conv=Conv2dInt16Params.from_dict(payload["conv"]))

    def to(self, device):
        self.conv.to(device)
        return self


def configure_conv_params(
    params,
    module_name,
    activation_scales=None,
    activation_observer=None,
):
    activation_scales = activation_scales or {}
    normalized_name = _normalize_module_name(module_name)
    params.module_name = normalized_name
    params.int8_candidate = is_int8_kernel_candidate(params) and is_int8_layer_eligible(normalized_name)
    params.use_int8 = params.int8_candidate and int8_tensor_cores_enabled()
    activation_scale = activation_scales.get(normalized_name)
    if torch.is_tensor(activation_scale):
        params.activation_int8_channel_scale = activation_scale.to(dtype=torch.int32)
        if activation_scale.numel() > 0:
            params.activation_int8_scale = int(activation_scale.max().item())
    elif activation_scale is not None:
        params.activation_int8_scale = int(activation_scale)
    elif params.activation_int8_scale <= 0:
        params.activation_int8_scale = 1
    if (
        params.eff_scale_c is None
        and params.weight_int8 is not None
        and params.weight_int8_channel_scale is not None
        and params.activation_int8_channel_scale is not None
    ):
        params.eff_scale_c = compute_effective_output_scale(
            params.weight_int8,
            params.weight_int8_channel_scale,
            params.activation_int8_channel_scale,
            weight_scale=Int16QuantConfig().weight_scale,
        )
    params.activation_observer = activation_observer
    refresh_int8_runtime_scale(params)
    return params


def refresh_int8_runtime_scale(params):
    params.weight_int8_runtime_scale = None
    if params.eff_scale_c is not None:
        params.weight_int8_runtime_scale = params.eff_scale_c.to(
            device=params.eff_scale_c.device,
            dtype=torch.float32,
        ).contiguous()
        return params
    if (
        params.weight_int8_channel_scale is None
        or params.weight_int8 is None
        or params.activation_int8_scale is None
        or params.activation_int8_scale <= 0
    ):
        return params
    scale_device = params.weight_int8_channel_scale.device
    params.weight_int8_runtime_scale = (
        params.weight_int8_channel_scale.to(device=scale_device, dtype=torch.float32)
        * (float(params.activation_int8_scale) / float(Int16QuantConfig().weight_scale))
    ).contiguous()
    return params


@dataclass
class DepthConvBlockInt16Params:
    adaptor: Optional[Conv2dInt16Params]
    dc_conv1: Conv2dInt16Params
    dc_depth_conv: Conv2dInt16Params
    dc_conv2: Conv2dInt16Params
    ffn_conv1: Conv2dInt16Params
    ffn_conv2: Conv2dInt16Params
    shortcut: bool

    @classmethod
    def from_float_module(cls, module, quant_cfg):
        adaptor = None
        if module.adaptor is not None:
            adaptor = Conv2dInt16Params.from_float_module(module.adaptor, quant_cfg)
        return cls(
            adaptor=adaptor,
            dc_conv1=Conv2dInt16Params.from_float_module(module.dc[0], quant_cfg),
            dc_depth_conv=Conv2dInt16Params.from_float_module(module.dc[2], quant_cfg),
            dc_conv2=Conv2dInt16Params.from_float_module(module.dc[3], quant_cfg),
            ffn_conv1=Conv2dInt16Params.from_float_module(module.ffn[0], quant_cfg),
            ffn_conv2=Conv2dInt16Params.from_float_module(module.ffn[2], quant_cfg),
            shortcut=module.shortcut,
        )

    def to_dict(self):
        return {
            "adaptor": None if self.adaptor is None else self.adaptor.to_dict(),
            "dc_conv1": self.dc_conv1.to_dict(),
            "dc_depth_conv": self.dc_depth_conv.to_dict(),
            "dc_conv2": self.dc_conv2.to_dict(),
            "ffn_conv1": self.ffn_conv1.to_dict(),
            "ffn_conv2": self.ffn_conv2.to_dict(),
            "shortcut": self.shortcut,
        }

    @classmethod
    def from_dict(cls, payload):
        adaptor = payload["adaptor"]
        return cls(
            adaptor=None if adaptor is None else Conv2dInt16Params.from_dict(adaptor),
            dc_conv1=Conv2dInt16Params.from_dict(payload["dc_conv1"]),
            dc_depth_conv=Conv2dInt16Params.from_dict(payload["dc_depth_conv"]),
            dc_conv2=Conv2dInt16Params.from_dict(payload["dc_conv2"]),
            ffn_conv1=Conv2dInt16Params.from_dict(payload["ffn_conv1"]),
            ffn_conv2=Conv2dInt16Params.from_dict(payload["ffn_conv2"]),
            shortcut=payload["shortcut"],
        )

    def to(self, device):
        if self.adaptor is not None:
            self.adaptor.to(device)
        self.dc_conv1.to(device)
        self.dc_depth_conv.to(device)
        self.dc_conv2.to(device)
        self.ffn_conv1.to(device)
        self.ffn_conv2.to(device)
        return self


REQUIRED_SHARED_INT16_KERNELS = (
    "conv2d_1x1_int16_acc32",
    "conv2d_1x1_int8tc_acc32",
    "conv2d_3x3_int16_acc32",
    "depthwise_conv2d_3x3_int16_acc32",
    "bias_add_residual_int16",
    "bias_add_residual_quant_int16",
    "wsilu_lut_int16",
    "wsilu_chunk_add_int16",
    "pixel_shuffle_2_int16",
    "pixel_shuffle_8_int16",
)


REQUIRED_ENTROPY_INT16_KERNELS = (
    "round_and_clip_to_int8",
    "clamp_reciprocal_quant_int16",
    "process_with_mask_int16",
    "combine_for_reading_2x_int16",
    "restore_y_2x_int16",
    "restore_y_4x_int16",
    "build_index_enc_int16",
    "build_index_dec_int16",
    "add_and_multiply_int16",
)


def export_int16_manifest(quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "format_version": 3,
        "feature_scale": quant_cfg.feature_scale,
        "weight_scale": quant_cfg.weight_scale,
        "bias_scale": quant_cfg.bias_scale,
        "required_shared_kernels": list(REQUIRED_SHARED_INT16_KERNELS),
        "required_entropy_kernels": list(REQUIRED_ENTROPY_INT16_KERNELS),
        "sigmoid_lut_entries": 1 << 16,
        "wsilu_lut_entries": 1 << 16,
        "scale_index_lut_entries": 1 << 16,
    }


def quant_config_from_manifest(manifest):
    return Int16QuantConfig(
        feature_scale=manifest["feature_scale"],
        weight_scale=manifest["weight_scale"],
    )


def _shift_bits(scale):
    if scale <= 0 or scale & (scale - 1) != 0:
        raise RuntimeError(f"Expected a positive power-of-two scale, got {scale}")
    return scale.bit_length() - 1


def round_shift_right(value, bits):
    if bits == 0:
        return value
    offset = 1 << (bits - 1)
    positive = (value + offset) >> bits
    negative = -(((-value) + offset) >> bits)
    return torch.where(value >= 0, positive, negative)


def round_divide_by_scalar(value, divisor):
    if divisor <= 0:
        raise RuntimeError(f"Expected divisor > 0, got {divisor}")
    if divisor == 1:
        return value
    offset = divisor // 2
    positive = (value + offset) // divisor
    negative = -(((-value) + offset) // divisor)
    return torch.where(value >= 0, positive, negative)


def feature_to_int16(tensor, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return quantize_feature(tensor, quant_cfg.feature_scale)


def int16_to_feature(tensor, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return dequantize_int16(tensor, quant_cfg.feature_scale)


def clamp_feature_int16(tensor, min_value=0.0, max_value=1.0, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    min_scaled = int(round(min_value * quant_cfg.feature_scale))
    max_scaled = int(round(max_value * quant_cfg.feature_scale))
    return tensor.clamp(min_scaled, max_scaled).to(torch.int16)


def round_and_to_int8_int16(tensor, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    symbols = round_shift_right(tensor.to(torch.int32), _shift_bits(quant_cfg.feature_scale))
    symbols = symbols.clamp(-128, 127).to(torch.int8)
    reconstructed = clamp_int16(symbols.to(torch.int32) * quant_cfg.feature_scale)
    return reconstructed, symbols


def quantize_module_bank(tensor, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return quantize_feature(tensor.detach().cpu(), quant_cfg.feature_scale)


def quantize_scalar_to_int16(value, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return int(round(float(value) * quant_cfg.feature_scale))


def _maybe_pad_int32(x, padding):
    if padding == 0:
        return x
    return F.pad(x, (padding, padding, padding, padding))


def _conv2d_output_shape(input_i16, params):
    padded_h = input_i16.shape[2] + 2 * int(params.padding)
    padded_w = input_i16.shape[3] + 2 * int(params.padding)
    kernel_h = params.weight.shape[2]
    kernel_w = params.weight.shape[3]
    out_h = (padded_h - kernel_h) // int(params.stride) + 1
    out_w = (padded_w - kernel_w) // int(params.stride) + 1
    return (input_i16.shape[0], params.weight.shape[0], out_h, out_w)


def _validate_conv_residual_shape(residual, output_shape):
    if residual is None:
        return
    if tuple(residual.shape) != tuple(output_shape):
        raise RuntimeError(
            "Fused INT16 residual shape mismatch: "
            f"expected {tuple(output_shape)}, got {tuple(residual.shape)}"
        )


def conv2d_int16_reference(input_i16, params, quant_cfg=None, residual=None, residual2=None, post_scale=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    _validate_conv_residual_shape(residual, _conv2d_output_shape(input_i16, params))
    x = input_i16.to(torch.int32)
    w = params.weight.to(torch.int32)
    x = _maybe_pad_int32(x, params.padding)

    stride = params.stride
    _, in_channels, _, _ = x.shape
    out_channels, channels_per_group, kernel_h, kernel_w = w.shape
    groups = params.groups
    out_h = (x.shape[2] - kernel_h) // stride + 1
    out_w = (x.shape[3] - kernel_w) // stride + 1

    patches = x.unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
    patches = patches.contiguous().view(
        x.shape[0], groups, in_channels // groups, out_h, out_w, kernel_h, kernel_w
    )
    w = w.view(groups, out_channels // groups, channels_per_group, kernel_h, kernel_w)
    acc = torch.einsum("bgixymn,goimn->bgoxy", patches, w)
    acc = acc.contiguous().view(x.shape[0], out_channels, out_h, out_w)
    if params.bias is not None:
        acc = acc + params.bias.to(acc.device)[None, :, None, None]
    out = round_divide_by_scalar(acc.to(torch.int64), int(params.k2_layer))
    if residual is not None:
        out = clamp_int16(out + residual.to(torch.int32))
    if residual2 is not None:
        out = clamp_int16(out.to(torch.int32) + residual2.to(torch.int32))
    if post_scale is not None:
        post_scale = post_scale.to(device=out.device, dtype=torch.int16).reshape(1, -1, 1, 1)
        out = multiply_int16(out.to(torch.int16), post_scale, quant_cfg.feature_scale)
    return out


def conv2d_int16(input_i16, params, quant_cfg=None, residual=None, residual2=None, post_scale=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    _validate_conv_residual_shape(residual, _conv2d_output_shape(input_i16, params))
    if params.activation_observer is not None and is_int8_kernel_candidate(params):
        params.activation_observer(params.module_name, input_i16)
    if USE_CUDA_KERNELS and input_i16.is_cuda:
        bias = params.bias
        if bias is None:
            bias = torch.empty(0, dtype=torch.int32, device=input_i16.device)
        else:
            bias = bias.to(device=input_i16.device, dtype=torch.int32, non_blocking=True)
        weight = params.weight.to(device=input_i16.device, dtype=torch.int16, non_blocking=True)
        weight_i8 = None
        runtime_scale_c = None
        activation_scale_c = None
        activation_scale = params.activation_int8_scale
        if params.use_int8 and params.weight_int8 is not None and post_scale is None:
            weight_i8 = params.weight_int8.to(
                device=input_i16.device, dtype=torch.int8, non_blocking=True
            )
            if params.activation_int8_channel_scale is not None:
                activation_scale_c = params.activation_int8_channel_scale.to(
                    device=input_i16.device,
                    dtype=torch.int32,
                    non_blocking=True,
                )
            elif (
                params.stride == 1
                and params.padding == 0
                and params.groups == 1
                and weight.shape[2] == 1
                and weight.shape[3] == 1
                and (weight.shape[0] % 4 == 0)
                and (weight.shape[1] % 4 == 0)
                and activation_scale <= 0
            ):
                activation_scale = choose_activation_int8_scale(input_i16, activation_scale)
                params.activation_int8_scale = activation_scale
                refresh_int8_runtime_scale(params)
            if params.weight_int8_runtime_scale is not None:
                runtime_scale_c = params.weight_int8_runtime_scale.to(
                    device=input_i16.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
        if activation_scale <= 0:
            activation_scale = 1
        return _cuda_conv2d_int16(
            input_i16.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            params.stride,
            params.padding,
            params.groups,
            None if weight_i8 is None else weight_i8.contiguous(),
            int(params.weight_int8_scale),
            int(activation_scale),
            int(params.k2_layer),
            None if activation_scale_c is None else activation_scale_c.contiguous(),
            None if runtime_scale_c is None else runtime_scale_c.contiguous(),
            residual=residual.contiguous() if residual is not None else None,
            residual2=residual2.contiguous() if residual2 is not None else None,
            post_scale=None if post_scale is None else post_scale.to(
                device=input_i16.device,
                dtype=torch.int16,
                non_blocking=True,
            ).reshape(-1).contiguous(),
        )
    return conv2d_int16_reference(input_i16, params, quant_cfg, residual=residual, residual2=residual2, post_scale=post_scale)


def depthwise_conv3x3_lut_fused_int16(input_i16, params, lut, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    if (
        USE_CUDA_KERNELS
        and input_i16.is_cuda
        and params.groups == input_i16.shape[1]
        and params.weight.shape[0] == input_i16.shape[1]
        and params.weight.shape[1] == 1
        and params.weight.shape[2] == 3
        and params.weight.shape[3] == 3
    ):
        bias = params.bias
        if bias is None:
            bias = torch.empty(0, dtype=torch.int32, device=input_i16.device)
        else:
            bias = bias.to(device=input_i16.device, dtype=torch.int32, non_blocking=True)
        return _cuda_depthwise_conv3x3_lut_fused_int16(
            input_i16.contiguous(),
            params.weight.to(device=input_i16.device, dtype=torch.int16, non_blocking=True).contiguous(),
            bias.contiguous(),
            lut.to(device=input_i16.device, dtype=torch.int16, non_blocking=True).contiguous(),
            params.stride,
            params.padding,
        )
    activated = apply_lut_int16(input_i16, lut)
    return conv2d_int16(activated, params, quant_cfg)


def add_int16(x, y):
    if USE_CUDA_KERNELS and x.is_cuda and y.is_cuda:
        return _cuda_add_int16(x.contiguous(), y.contiguous())
    return clamp_int16(x.to(torch.int32) + y.to(torch.int32))


def multiply_int16(x, y, scale):
    y_numel = y.numel() if torch.is_tensor(y) else 0
    supports_cuda_broadcast = (
        x.dim() == 4
        and y_numel in (1, x.shape[1], x.numel())
    )
    if USE_CUDA_KERNELS and x.is_cuda and y.is_cuda and scale == 512 and supports_cuda_broadcast:
        return _cuda_multiply_int16(
            x.contiguous(),
            y.to(device=x.device, dtype=torch.int16, non_blocking=True).reshape(-1).contiguous(),
            scale,
        )
    product = x.to(torch.int32) * y.to(torch.int32)
    return clamp_int16(round_shift_right(product, _shift_bits(scale)))


def apply_lut_int16_reference(x, lut):
    idx = x.to(torch.int32) - INT16_MIN
    lut = lut.to(x.device)
    return lut[idx]


def apply_lut_int16(x, lut):
    if USE_CUDA_KERNELS and x.is_cuda:
        return _cuda_lut_lookup_int16(x.contiguous(), lut.to(device=x.device, dtype=torch.int16))
    return apply_lut_int16_reference(x, lut)


def wsilu_int16(x, wsilu_lut):
    return apply_lut_int16(x, wsilu_lut)


def wsilu_chunk_add_int16(x, wsilu_lut):
    if USE_CUDA_KERNELS and x.is_cuda:
        return _cuda_wsilu_chunk_add_int16(x.contiguous(), wsilu_lut.to(device=x.device, dtype=torch.int16))
    activated = wsilu_int16(x, wsilu_lut)
    x0, x1 = activated.chunk(2, dim=1)
    return add_int16(x0, x1)




def conv2d_wsilu_chunk_int16(input_i16, params, wsilu_lut, quant_cfg=None):
    """Fused 1x1 conv + WSiLU-chunk-add.

    Replaces the two-step sequence:
        out = conv2d_int16(x, ffn_conv1_params)
        out = wsilu_chunk_add_int16(out, wsilu_lut)

    Eligibility: 1x1 conv, stride=1, padding=0, groups=1, CUDA available.
    Falls back to the unfused path when conditions are not met.
    """
    quant_cfg = quant_cfg or Int16QuantConfig()
    weight = params.weight
    can_fuse = (
        USE_CUDA_KERNELS
        and input_i16.is_cuda
        and weight.dim() == 4
        and weight.shape[2] == 1
        and weight.shape[3] == 1
        and params.stride == 1
        and params.padding == 0
        and params.groups == 1
        and weight.shape[0] % 2 == 0
    )
    if can_fuse:
        bias = params.bias
        if bias is None:
            bias = torch.empty(0, dtype=torch.int32, device=input_i16.device)
        else:
            bias = bias.to(device=input_i16.device, dtype=torch.int32, non_blocking=True)
        weight_2d = weight.reshape(weight.shape[0], weight.shape[1]).to(
            device=input_i16.device, dtype=torch.int16, non_blocking=True
        ).contiguous()
        lut = wsilu_lut.to(device=input_i16.device, dtype=torch.int16, non_blocking=True)
        return _cuda_conv1x1_int16_gemm_wsilu_chunk(
            input_i16.contiguous(),
            weight_2d,
            bias.contiguous() if bias is not None else None,
            lut.contiguous(),
            int(params.k2_layer),
        )
    # Fallback: unfused path
    out = conv2d_int16(input_i16, params, quant_cfg)
    return wsilu_chunk_add_int16(out, wsilu_lut)


def sigmoid_int16(x, sigmoid_lut):
    return apply_lut_int16(x, sigmoid_lut)


def affine_sigmoid_int16(x, sigmoid_lut, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    sig = sigmoid_int16(x, sigmoid_lut).to(torch.int32)
    out = sig + round_shift_right(sig, 1) + quant_cfg.feature_scale // 2
    return clamp_int16(out)


def separate_prior_image_int16(params, sigmoid_lut, quant_cfg=None):
    q = params[:, :2, :, :]
    q_enc, q_dec = affine_sigmoid_int16(q, sigmoid_lut, quant_cfg).chunk(2, 1)
    scales, means = params[:, 2:, :, :].chunk(2, 1)
    return q_enc, q_dec, scales, means


def build_scale_index_lut(scale_min, scale_max, log_scale_min, log_step_recip, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    scales = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.float32)
    scales = scales / float(quant_cfg.feature_scale)
    scales = scales.clamp(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    return indexes.to(torch.int32)


def scale_index_lookup_int16(scales, scale_index_lut):
    if USE_CUDA_KERNELS and scales.is_cuda:
        lut = scale_index_lut.to(device=scales.device, dtype=torch.int32, non_blocking=True)
        return _cuda_scale_index_lut_int16(scales.contiguous(), lut.contiguous())
    flat = scales.reshape(-1).to(torch.int32) - INT16_MIN
    lut = scale_index_lut.to(device=scales.device, dtype=torch.int32)
    return lut[flat].reshape(scales.shape)


def pixel_shuffle_int16(x, upscale_factor):
    b, c, h, w = x.shape
    r = upscale_factor
    out_c = c // (r * r)
    x = x.view(b, out_c, r, r, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, out_c, h * r, w * r)


def pixel_unshuffle_int16(x, downscale_factor):
    b, c, h, w = x.shape
    r = downscale_factor
    out_c = c * r * r
    x = x.view(b, c, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, out_c, h // r, w // r)


def concat_int16(x, to_cat, cat_at_front=True):
    if to_cat is None:
        return x
    if cat_at_front:
        return torch.cat((to_cat, x), dim=1)
    return torch.cat((x, to_cat), dim=1)


def _binary_mask(mask):
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def _repeat_symbols_with_scale(y_q, repeat_factor, feature_scale):
    return torch.cat([y_q] * repeat_factor, dim=1).to(torch.int32) * int(feature_scale)


def sum_chunks_int16(x, num_chunks):
    acc = None
    for chunk in x.chunk(num_chunks, dim=1):
        chunk = chunk.to(torch.int32)
        acc = chunk if acc is None else acc + chunk
    return clamp_int16(acc)


def single_part_for_writing_2x_int16(x):
    return sum_chunks_int16(x, 2)


def single_part_for_writing_4x_int16(x):
    return sum_chunks_int16(x, 4)


def process_with_mask_int16(y, scales, means, mask, quant_cfg=None, force_zero_thres=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    mask_bool = _binary_mask(mask)
    zeros = torch.zeros_like(scales)
    scales_hat = torch.where(mask_bool, scales, zeros)
    means_hat = torch.where(mask_bool, means, torch.zeros_like(means))

    y_res = torch.where(
        mask_bool,
        y.to(torch.int32) - means_hat.to(torch.int32),
        torch.zeros_like(y, dtype=torch.int32),
    )
    y_q = round_shift_right(y_res, _shift_bits(quant_cfg.feature_scale))
    if force_zero_thres is not None:
        threshold = force_zero_thres
        if not isinstance(threshold, int):
            threshold = quantize_scalar_to_int16(threshold, quant_cfg)
        y_q = torch.where(scales_hat.to(torch.int32) > int(threshold), y_q, torch.zeros_like(y_q))
    y_q = y_q.clamp(-128, 127).to(torch.int16)
    y_hat = clamp_int16(y_q.to(torch.int32) * quant_cfg.feature_scale + means_hat.to(torch.int32))
    return y_res, y_q, y_hat, scales_hat


def combine_for_reading_2x_int16(x, mask):
    mask_bool = _binary_mask(mask)
    masked = torch.where(mask_bool, x, torch.zeros_like(x))
    return single_part_for_writing_2x_int16(masked)


def restore_y_2x_int16(y, means, mask, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    mask_bool = _binary_mask(mask)
    restored = _repeat_symbols_with_scale(y, 2, quant_cfg.feature_scale) + means.to(torch.int32)
    return clamp_int16(torch.where(mask_bool, restored, torch.zeros_like(restored)))


def restore_y_2x_with_cat_after_int16(y, means, mask, to_cat, quant_cfg=None):
    out = restore_y_2x_int16(y, means, mask, quant_cfg=quant_cfg)
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x_int16(y, means, mask, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    mask_bool = _binary_mask(mask)
    restored = _repeat_symbols_with_scale(y, 4, quant_cfg.feature_scale) + means.to(torch.int32)
    return clamp_int16(torch.where(mask_bool, restored, torch.zeros_like(restored)))


def add_and_multiply_int16_reference(y_hat_0, y_hat_1, q_dec, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    acc = y_hat_0.to(torch.int32) + y_hat_1.to(torch.int32)
    product = acc * q_dec.to(torch.int32)
    return clamp_int16(round_shift_right(product, _shift_bits(quant_cfg.feature_scale)))


def add_and_multiply_int16(y_hat_0, y_hat_1, q_dec, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    if USE_CUDA_KERNELS and y_hat_0.is_cuda:
        return _cuda_add_multiply_int16(
            y_hat_0.contiguous(),
            y_hat_1.contiguous(),
            q_dec.to(device=y_hat_0.device, dtype=torch.int16, non_blocking=True).contiguous(),
            quant_cfg.feature_scale,
        )
    return add_and_multiply_int16_reference(y_hat_0, y_hat_1, q_dec, quant_cfg)


def clamp_reciprocal_with_quant_int16_reference(q_dec, y, min_value=0.5, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    min_q = quantize_scalar_to_int16(min_value, quant_cfg)
    q_dec = q_dec.clamp(min_q, INT16_MAX).to(torch.int16)
    numerator = y.to(torch.int32) * quant_cfg.feature_scale
    y_quant = torch.div(
        numerator + torch.where(numerator >= 0, q_dec.to(torch.int32) // 2, -(q_dec.to(torch.int32) // 2)),
        q_dec.to(torch.int32),
        rounding_mode="trunc",
    )
    return q_dec, clamp_int16(y_quant)


def clamp_reciprocal_with_quant_int16(q_dec, y, min_value=0.5, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    min_q = quantize_scalar_to_int16(min_value, quant_cfg)
    q_dec = q_dec.clamp(min_q, INT16_MAX).to(torch.int16)
    if USE_CUDA_KERNELS and q_dec.is_cuda:
        q_enc = _cuda_clamp_reciprocal_int16(q_dec.contiguous(), quant_cfg.feature_scale)
        return q_dec, multiply_int16(y, q_enc, quant_cfg.feature_scale)
    return clamp_reciprocal_with_quant_int16_reference(q_dec, y, min_value, quant_cfg)


def build_index_dec_int16(scales, scale_index_lut, force_zero_thres=None):
    indexes = scale_index_lookup_int16(scales, scale_index_lut).reshape(-1).to(torch.uint8)
    skip_cond = None
    if force_zero_thres is not None:
        threshold = force_zero_thres
        if not isinstance(threshold, int):
            threshold = quantize_scalar_to_int16(threshold)
        skip_cond = scales.reshape(-1) > int(threshold)
        indexes = indexes[skip_cond]
    return indexes, skip_cond


def build_index_enc_skip_mask_int16(scales, force_zero_thres=None):
    if force_zero_thres is None:
        return None
    threshold = force_zero_thres
    if not isinstance(threshold, int):
        threshold = quantize_scalar_to_int16(threshold)
    return scales.reshape(-1) > int(threshold)


def build_index_enc_int16_full(symbols, scales, scale_index_lut):
    flat_symbols = symbols.reshape(-1).to(torch.int16)
    flat_scales = scales.reshape(-1)
    indexes = scale_index_lookup_int16(flat_scales, scale_index_lut).reshape(-1).to(torch.int16)
    packed = ((flat_symbols << 8) + indexes).to(torch.int16)
    return packed


def build_index_enc_int16(symbols, scales, scale_index_lut, force_zero_thres=None):
    packed = build_index_enc_int16_full(symbols, scales, scale_index_lut)
    keep_mask = build_index_enc_skip_mask_int16(scales, force_zero_thres=force_zero_thres)
    if keep_mask is not None:
        packed = packed[keep_mask]
    return packed


def unpack_index_symbols_int16(packed):
    packed_i32 = packed.to(torch.int32)
    symbols = (packed_i32 >> 8).to(torch.int16)
    indexes = (packed_i32 & 0xFF).to(torch.uint8)
    return symbols, indexes


def expand_decoded_symbols_int16(symbols, shape, skip_cond=None, device=None):
    device = device or symbols.device
    if skip_cond is None:
        return symbols.to(device=device, dtype=torch.int16).reshape(shape)
    flat = torch.zeros(skip_cond.numel(), dtype=torch.int16, device=device)
    flat[skip_cond.to(device)] = symbols.to(device=device, dtype=torch.int16)
    return flat.reshape(shape)


class Conv2dInt16Runner:
    def __init__(self, params, quant_cfg=None):
        self.params = params
        self.quant_cfg = quant_cfg or Int16QuantConfig()

    @classmethod
    def from_float_module(cls, module, quant_cfg=None):
        quant_cfg = quant_cfg or Int16QuantConfig()
        return cls(Conv2dInt16Params.from_float_module(module, quant_cfg), quant_cfg)

    def forward(self, x):
        return conv2d_int16(x, self.params, self.quant_cfg)

    def to(self, device):
        self.params.to(device)
        return self


class SequentialInt16Runner:
    def __init__(self, runners):
        self.runners = list(runners)

    def forward(self, x, quant_step=None):
        for i, runner in enumerate(self.runners):
            if i == len(self.runners) - 1 and quant_step is not None:
                x = _forward_with_quant_fuse(runner, x, quant_step, None)
            else:
                x = runner.forward(x)
        return x

    def to(self, device):
        for runner in self.runners:
            if hasattr(runner, "to"):
                runner.to(device)
        return self


def _forward_with_quant_fuse(runner, x, quant_step, quant_cfg):
    """Forward through runner, fusing quant_step into the final conv epilogue
    when the runner supports it. Falls back to separate multiply_int16.

    This eliminates a dedicated pointwise multiply kernel launch by folding
    the scale into the convolution's post_scale epilogue inside
    DepthConvBlockInt16Runner or SequentialInt16Runner.
    """
    if quant_step is None:
        return runner.forward(x)
    if isinstance(runner, DepthConvBlockInt16Runner):
        return runner.forward(x, quant_step=quant_step)
    if isinstance(runner, SequentialInt16Runner):
        return runner.forward(x, quant_step=quant_step)
    # Fallback: run unfused and apply multiply separately.
    out = runner.forward(x)
    cfg = quant_cfg if quant_cfg is not None else Int16QuantConfig()
    return multiply_int16(out, quant_step, cfg.feature_scale)


class SubpelConv2xInt16Runner:
    def __init__(self, params, quant_cfg=None):
        self.params = params
        self.quant_cfg = quant_cfg or Int16QuantConfig()

    @classmethod
    def from_float_module(cls, module, quant_cfg=None):
        quant_cfg = quant_cfg or Int16QuantConfig()
        return cls(SubpelConv2xInt16Params.from_float_module(module, quant_cfg), quant_cfg)

    def forward(self, x, to_cat=None, cat_at_front=True):
        out = conv2d_int16(x, self.params.conv, self.quant_cfg)
        out = pixel_shuffle_int16(out, 2)
        return concat_int16(out, to_cat, cat_at_front=cat_at_front)

    def to(self, device):
        self.params.to(device)
        return self


class DepthConvBlockInt16Runner:
    def __init__(self, params, quant_cfg=None, wsilu_lut=None):
        self.params = params
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )

    @classmethod
    def from_float_module(cls, module, quant_cfg=None, wsilu_lut=None):
        quant_cfg = quant_cfg or Int16QuantConfig()
        return cls(
            DepthConvBlockInt16Params.from_float_module(module, quant_cfg),
            quant_cfg=quant_cfg,
            wsilu_lut=wsilu_lut,
        )

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        identity = x
        if self.params.adaptor is not None:
            identity = conv2d_int16(identity, self.params.adaptor, self.quant_cfg)

        out = conv2d_int16(identity, self.params.dc_conv1, self.quant_cfg)
        if (
            self.params.dc_depth_conv.groups == out.shape[1]
            and self.params.dc_depth_conv.weight.shape[0] == out.shape[1]
            and self.params.dc_depth_conv.weight.shape[1] == 1
            and self.params.dc_depth_conv.weight.shape[2] == 3
            and self.params.dc_depth_conv.weight.shape[3] == 3
        ):
            out = depthwise_conv3x3_lut_fused_int16(
                out, self.params.dc_depth_conv, self.wsilu_lut, self.quant_cfg
            )
        else:
            out = wsilu_int16(out, self.wsilu_lut)
            out = conv2d_int16(out, self.params.dc_depth_conv, self.quant_cfg)
        out = conv2d_int16(out, self.params.dc_conv2, self.quant_cfg, residual=identity)

        ffn_identity = out
        out = conv2d_wsilu_chunk_int16(out, self.params.ffn_conv1, self.wsilu_lut, self.quant_cfg)
        fuse_quant_step = quant_step is not None
        out = conv2d_int16(
            out,
            self.params.ffn_conv2,
            self.quant_cfg,
            residual=ffn_identity,
            residual2=identity if self.params.shortcut else None,
            post_scale=quant_step if fuse_quant_step else None,
        )

        if quant_step is not None and not fuse_quant_step:
            out = multiply_int16(out, quant_step, self.quant_cfg.feature_scale)
        return concat_int16(out, to_cat, cat_at_front=cat_at_front)

    def to(self, device):
        self.params.to(device)
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16, non_blocking=True)
        return self


class ResidualBlockWithStride2Int16Runner:
    def __init__(self, down_params, conv_runner, quant_cfg=None):
        self.down_params = down_params
        self.conv_runner = conv_runner
        self.quant_cfg = quant_cfg or Int16QuantConfig()

    @classmethod
    def from_float_module(cls, module, quant_cfg=None, wsilu_lut=None):
        quant_cfg = quant_cfg or Int16QuantConfig()
        down_params = Conv2dInt16Params.from_float_module(module.down, quant_cfg)
        conv_runner = DepthConvBlockInt16Runner.from_float_module(
            module.conv, quant_cfg=quant_cfg, wsilu_lut=wsilu_lut
        )
        return cls(down_params, conv_runner, quant_cfg)

    def forward(self, x):
        x = conv2d_int16(x, self.down_params, self.quant_cfg)
        return self.conv_runner.forward(x)

    def to(self, device):
        self.down_params.to(device)
        if hasattr(self.conv_runner, "to"):
            self.conv_runner.to(device)
        return self


class ResidualBlockUpsampleInt16Runner:
    def __init__(self, up_runner, conv_runner):
        self.up_runner = up_runner
        self.conv_runner = conv_runner

    @classmethod
    def from_float_module(cls, module, quant_cfg=None, wsilu_lut=None):
        quant_cfg = quant_cfg or Int16QuantConfig()
        up_runner = SubpelConv2xInt16Runner.from_float_module(module.up, quant_cfg=quant_cfg)
        conv_runner = DepthConvBlockInt16Runner.from_float_module(
            module.conv, quant_cfg=quant_cfg, wsilu_lut=wsilu_lut
        )
        return cls(up_runner, conv_runner)

    def forward(self, x):
        x = self.up_runner.forward(x)
        return self.conv_runner.forward(x)

    def to(self, device):
        if hasattr(self.up_runner, "to"):
            self.up_runner.to(device)
        if hasattr(self.conv_runner, "to"):
            self.conv_runner.to(device)
        return self


def pack_int16_module(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()

    from .layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2, SubpelConv2x

    if isinstance(module, nn.Sequential):
        return {
            "kind": "sequential",
            "modules": [pack_int16_module(child, quant_cfg=quant_cfg) for child in module],
        }
    if isinstance(module, nn.Conv2d):
        return {
            "kind": "conv2d",
            "params": Conv2dInt16Params.from_float_module(module, quant_cfg).to_dict(),
        }
    if isinstance(module, DepthConvBlock):
        return {
            "kind": "depth_conv_block",
            "params": DepthConvBlockInt16Params.from_float_module(module, quant_cfg).to_dict(),
        }
    if isinstance(module, SubpelConv2x):
        return {
            "kind": "subpel_conv2x",
            "params": SubpelConv2xInt16Params.from_float_module(module, quant_cfg).to_dict(),
        }
    if isinstance(module, ResidualBlockWithStride2):
        return {
            "kind": "residual_block_stride2",
            "down_params": Conv2dInt16Params.from_float_module(module.down, quant_cfg).to_dict(),
            "conv": pack_int16_module(module.conv, quant_cfg=quant_cfg),
        }
    if isinstance(module, ResidualBlockUpsample):
        return {
            "kind": "residual_block_upsample",
            "up": pack_int16_module(module.up, quant_cfg=quant_cfg),
            "conv": pack_int16_module(module.conv, quant_cfg=quant_cfg),
        }

    raise TypeError(f"Unsupported module type for int16 packing: {type(module).__name__}")


def build_int16_runner(
    spec,
    quant_cfg=None,
    wsilu_lut=None,
    module_name="module",
    activation_scales=None,
    activation_observer=None,
):
    quant_cfg = quant_cfg or Int16QuantConfig()
    activation_scales = activation_scales or {}
    kind = spec["kind"]

    if kind == "sequential":
        return SequentialInt16Runner(
            [
                build_int16_runner(
                    child,
                    quant_cfg=quant_cfg,
                    wsilu_lut=wsilu_lut,
                    module_name=f"{module_name}.modules[{idx}]",
                    activation_scales=activation_scales,
                    activation_observer=activation_observer,
                )
                for idx, child in enumerate(spec["modules"])
            ]
        )
    if kind == "conv2d":
        params = Conv2dInt16Params.from_dict(spec["params"])
        configure_conv_params(
            params,
            module_name,
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        return Conv2dInt16Runner(params, quant_cfg)
    if kind == "depth_conv_block":
        params = DepthConvBlockInt16Params.from_dict(spec["params"])
        if params.adaptor is not None:
            configure_conv_params(
                params.adaptor,
                f"{module_name}.adaptor",
                activation_scales=activation_scales,
                activation_observer=activation_observer,
            )
        for child_name in ("dc_conv1", "dc_depth_conv", "dc_conv2", "ffn_conv1", "ffn_conv2"):
            configure_conv_params(
                getattr(params, child_name),
                f"{module_name}.{child_name}",
                activation_scales=activation_scales,
                activation_observer=activation_observer,
            )
        return DepthConvBlockInt16Runner(
            params,
            quant_cfg=quant_cfg,
            wsilu_lut=wsilu_lut,
        )
    if kind == "subpel_conv2x":
        params = SubpelConv2xInt16Params.from_dict(spec["params"])
        configure_conv_params(
            params.conv,
            f"{module_name}.conv",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        return SubpelConv2xInt16Runner(
            params,
            quant_cfg=quant_cfg,
        )
    if kind == "residual_block_stride2":
        down_params = Conv2dInt16Params.from_dict(spec["down_params"])
        configure_conv_params(
            down_params,
            f"{module_name}.down",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        return ResidualBlockWithStride2Int16Runner(
            down_params,
            build_int16_runner(
                spec["conv"],
                quant_cfg=quant_cfg,
                wsilu_lut=wsilu_lut,
                module_name=f"{module_name}.conv",
                activation_scales=activation_scales,
                activation_observer=activation_observer,
            ),
            quant_cfg=quant_cfg,
        )
    if kind == "residual_block_upsample":
        return ResidualBlockUpsampleInt16Runner(
            build_int16_runner(
                spec["up"],
                quant_cfg=quant_cfg,
                wsilu_lut=wsilu_lut,
                module_name=f"{module_name}.up",
                activation_scales=activation_scales,
                activation_observer=activation_observer,
            ),
            build_int16_runner(
                spec["conv"],
                quant_cfg=quant_cfg,
                wsilu_lut=wsilu_lut,
                module_name=f"{module_name}.conv",
                activation_scales=activation_scales,
                activation_observer=activation_observer,
            ),
        )

    raise RuntimeError(f"Unsupported packed int16 runner kind: {kind}")
