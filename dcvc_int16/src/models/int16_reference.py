import io
import os
import time as _time_mod

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.int16_backend import (
    Int16QuantConfig,
    _forward_with_quant_fuse,
    add_and_multiply_int16,
    add_int16,
    build_int16_runner,
    build_index_dec_int16,
    build_index_enc_int16,
    build_index_enc_int16_full,
    build_index_enc_skip_mask_int16,
    build_scale_index_lut,
    build_sigmoid_lut,
    build_wsilu_lut,
    clamp_int16,
    clamp_feature_int16,
    combine_for_reading_2x_int16,
    concat_int16,
    expand_decoded_symbols_int16,
    export_int16_manifest,
    feature_to_int16,
    int16_to_feature,
    multiply_int16,
    pack_int16_module,
    pixel_shuffle_int16,
    pixel_unshuffle_int16,
    quant_config_from_manifest,
    quantize_scalar_to_int16,
    quantize_module_bank,
    restore_y_2x_int16,
    restore_y_2x_with_cat_after_int16,
    restore_y_4x_int16,
    round_and_to_int8_int16,
    single_part_for_writing_2x_int16,
    single_part_for_writing_4x_int16,
    separate_prior_image_int16,
    process_with_mask_int16,
    clamp_reciprocal_with_quant_int16,
    unpack_index_symbols_int16,
)


def _resolve_frozen_entropy_state(model, frozen_entropy_state):
    if frozen_entropy_state is not None:
        return frozen_entropy_state
    if getattr(model, "entropy_coder", None) is not None:
        return model.export_frozen_entropy_state()
    return None


def _get_one_mask(micro_mask, height, width, device):
    mask = torch.tensor(micro_mask, dtype=torch.bool, device=device)
    mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
    mask = mask[:height, :width]
    return mask.unsqueeze(0).unsqueeze(0)


def _tree_to_cpu(payload):
    if torch.is_tensor(payload):
        return payload.detach().to(device="cpu")
    if isinstance(payload, dict):
        return {key: _tree_to_cpu(value) for key, value in payload.items()}
    if isinstance(payload, tuple):
        return tuple(_tree_to_cpu(value) for value in payload)
    if isinstance(payload, list):
        return [_tree_to_cpu(value) for value in payload]
    return payload


def _serialize_payload(payload):
    buffer = io.BytesIO()
    torch.save(_tree_to_cpu(payload), buffer)
    return buffer.getvalue()


def _deserialize_payload(bit_stream, device):
    return torch.load(io.BytesIO(bit_stream), map_location=device, weights_only=False)


def _int16_pframe_cuda_graphs_enabled():
    return os.environ.get("DCVC_ENABLE_INT16_PFRAME_GRAPHS", "0") == "1"


def _int16_pipeline_profiling_enabled():
    return os.environ.get("DCVC_PROFILE_INT16_PIPELINE", "0") == "1"


def _int16_encode_only_enabled():
    """When True, skip the decoder-sync roundtrip during encoding.

    This eliminates ~100-135 ms of GPU work per P-frame by skipping
    decompress_prior_2x and get_recon_and_feature on the decoder path.
    The encoder uses its own reconstruction for the DPB, which is
    bit-exact with the decoder's reconstruction when the entropy
    coding is lossless (rANS guarantees this).
    """
    return os.environ.get("DCVC_INT16_ENCODE_ONLY", "0") == "1"


def _int16_async_entropy_prep_enabled():
    return os.environ.get("DCVC_INT16_ASYNC_ENTROPY_PREP", "0") == "1"


class _PipelineProfiler:
    """Lightweight profiler using CUDA events for GPU timing and wall-clock for CPU."""

    def __init__(self, enabled, device):
        self.enabled = enabled
        self.device = device
        self.stages = []
        self._gpu_events = []
        self._cpu_times = []
        self._last_event = None
        self._last_cpu = None

    def mark(self, stage_name):
        if not self.enabled:
            return
        cpu_now = _time_mod.perf_counter()
        if self.device is not None and self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = None
        if self._last_event is not None or self._last_cpu is not None:
            self.stages.append(stage_name)
            self._gpu_events.append((self._last_event, event))
            self._cpu_times.append((self._last_cpu, cpu_now))
        self._last_event = event
        self._last_cpu = cpu_now

    def report(self, frame_idx):
        if not self.enabled or len(self.stages) == 0:
            return {}
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        results = {}
        total_gpu = 0.0
        total_cpu = 0.0
        lines = []
        for i, stage in enumerate(self.stages):
            start_ev, end_ev = self._gpu_events[i]
            start_cpu, end_cpu = self._cpu_times[i]
            cpu_ms = (end_cpu - start_cpu) * 1000.0
            if start_ev is not None and end_ev is not None:
                gpu_ms = start_ev.elapsed_time(end_ev)
            else:
                gpu_ms = cpu_ms
            results[stage] = {"gpu_ms": gpu_ms, "cpu_ms": cpu_ms}
            total_gpu += gpu_ms
            total_cpu += cpu_ms
            lines.append(f"  {stage:40s} GPU={gpu_ms:8.2f}ms  CPU={cpu_ms:8.2f}ms")
        results["_total"] = {"gpu_ms": total_gpu, "cpu_ms": total_cpu}
        header = f"[PROFILE FRAME {frame_idx}] Pipeline breakdown (total GPU={total_gpu:.1f}ms, CPU={total_cpu:.1f}ms):"
        print(header)
        for line in lines:
            print(line)
        return results


class _DeferredPackedStreams:
    def __init__(self, packed_streams, device, ready_event=None, host_wait=False, resolver=None):
        self.packed_streams = packed_streams
        self.device = device
        self.ready_event = ready_event
        self.host_wait = host_wait
        self.resolver = resolver

    def resolve(self):
        if self.ready_event is not None and self.device.type == "cuda":
            if self.host_wait:
                self.ready_event.synchronize()
            else:
                torch.cuda.current_stream(device=self.device).wait_event(self.ready_event)
        if self.resolver is not None:
            return self.resolver(self.packed_streams)
        return self.packed_streams


def _check_int16_range(tensor, name, frame_idx):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return 0
    max_val = int(tensor.detach().abs().amax().item())
    if max_val >= 32767:
        print(f"[FRAME {frame_idx}] SATURATION ERROR: {name} absmax={max_val}")
    elif max_val >= 28000:
        print(f"[FRAME {frame_idx}] SATURATION WARNING: {name} absmax={max_val}")
    return max_val


def _format_tensor_stats(tensor):
    if tensor is None:
        return "none"
    if not torch.is_tensor(tensor):
        return str(type(tensor))
    if tensor.numel() == 0:
        return f"shape={tuple(tensor.shape)} empty"
    det = tensor.detach()
    return (
        f"shape={tuple(det.shape)} dtype={det.dtype} "
        f"min={int(det.min().item())} max={int(det.max().item())} "
        f"absmax={int(det.abs().amax().item())}"
    )


def _log_tensor_stats(enabled, frame_idx, name, tensor):
    if not enabled:
        return
    print(f"[FRAME {frame_idx}] {name}: {_format_tensor_stats(tensor)}")
    _check_int16_range(tensor, name, frame_idx)


def _log_tensor_delta(enabled, frame_idx, name, lhs, rhs):
    if lhs is None or rhs is None or (not torch.is_tensor(lhs)) or (not torch.is_tensor(rhs)):
        return
    if lhs.shape != rhs.shape:
        print(
            f"[FRAME {frame_idx}] STATE SHAPE MISMATCH: {name} "
            f"lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)}"
        )
        return
    diff = lhs.to(torch.int32) - rhs.to(torch.int32)
    if not enabled and not bool(diff.any()):
        return
    if diff.numel() == 0:
        return
    abs_diff = diff.abs()
    max_delta = int(abs_diff.amax().item())
    nonzero = int((abs_diff != 0).sum().item())
    if enabled or nonzero > 0:
        print(
            f"[FRAME {frame_idx}] STATE DELTA {name}: "
            f"nonzero={nonzero} max_delta={max_delta}"
        )


def _optional_tensor_equal(lhs, rhs):
    if lhs is None or rhs is None:
        return lhs is None and rhs is None
    if (not torch.is_tensor(lhs)) or (not torch.is_tensor(rhs)):
        return lhs == rhs
    return torch.equal(lhs, rhs)


def _graph_cache_key(name, inputs):
    return (
        name,
        tuple(
            (tuple(t.shape), str(t.dtype), t.device.type, t.device.index if t.device.index is not None else -1)
            for t in inputs
        ),
    )


def _run_cached_cuda_graph(cache, enabled, name, fn, *inputs):
    if (not enabled) or len(inputs) == 0 or any((not torch.is_tensor(t)) or (not t.is_cuda) for t in inputs):
        return fn(*inputs)

    key = _graph_cache_key(name, inputs)
    entry = cache.get(key)
    if entry is None:
        static_inputs = [inp.clone() for inp in inputs]
        for _ in range(2):
            fn(*static_inputs)
        torch.cuda.synchronize(device=inputs[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = fn(*static_inputs)
        entry = {
            "inputs": static_inputs,
            "graph": graph,
            "outputs": static_outputs,
        }
        cache[key] = entry

    for static_input, runtime_input in zip(entry["inputs"], inputs):
        static_input.copy_(runtime_input)
    entry["graph"].replay()
    return entry["outputs"]


def _get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


class Int16RansCodecRuntime:
    def __init__(self, frozen_entropy_state):
        from .entropy_models import BitEstimator, EntropyCoder, GaussianEncoder

        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder = GaussianEncoder()
        self.gaussian_encoder.load_state(
            frozen_entropy_state["gaussian_encoder"],
            self.entropy_coder,
            force_zero_thres=frozen_entropy_state["gaussian_encoder"].get("force_zero_thres"),
        )

        bit_state = frozen_entropy_state["bit_estimator_z"]
        self.bit_estimator_z = BitEstimator(bit_state["qp_num"], bit_state["channel"])
        self.bit_estimator_z.load_state(bit_state, self.entropy_coder)
        self.z_channel = frozen_entropy_state["z_channel"]

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def reset(self):
        self.entropy_coder.reset()

    def flush(self):
        self.entropy_coder.flush()

    def get_encoded_stream(self):
        return self.entropy_coder.get_encoded_stream()

    def set_stream(self, bit_stream):
        self.entropy_coder.set_stream(bit_stream)

    def encode_z(self, z_symbols, qp):
        self.bit_estimator_z.encode_z(z_symbols, qp)

    def decode_z_hat(self, z_size, qp, device, feature_scale):
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, torch.int16)
        return clamp_int16(z_q.to(torch.int32) * feature_scale)


class Int16GaussianEntropyReference:
    def __init__(self, frozen_entropy_state, quant_cfg=None):
        if frozen_entropy_state is None:
            raise RuntimeError("Frozen entropy state is required for the int16 entropy path.")
        if "gaussian_encoder" not in frozen_entropy_state:
            raise RuntimeError("Frozen entropy bundle is missing gaussian_encoder state.")

        self.quant_cfg = quant_cfg or Int16QuantConfig()
        gaussian_state = frozen_entropy_state["gaussian_encoder"]
        self.scale_index_lut = build_scale_index_lut(
            gaussian_state["scale_min"],
            gaussian_state["scale_max"],
            gaussian_state["log_scale_min"],
            gaussian_state["log_step_recip"],
            quant_cfg=self.quant_cfg,
        )
        self.force_zero_thres = gaussian_state["force_zero_thres"]
        self.force_zero_thres_int = None
        if self.force_zero_thres is not None:
            self.force_zero_thres_int = quantize_scalar_to_int16(
                self.force_zero_thres, self.quant_cfg
            )
        self.mask_cache = {}
        self.entropy_coder = None
        self.cdf_group_index = None
        self.debug_context = {}

    def get_mask_4x(self, batch, channel, height, width, device):
        key = (batch, channel, height, width, str(device), "4x")
        if key not in self.mask_cache:
            if channel % 4 != 0:
                raise RuntimeError(f"Expected channel divisible by 4, got {channel}")
            m = torch.ones((batch, channel // 4, height, width), dtype=torch.bool, device=device)
            m0 = _get_one_mask(((1, 0), (0, 0)), height, width, device)
            m1 = _get_one_mask(((0, 1), (0, 0)), height, width, device)
            m2 = _get_one_mask(((0, 0), (1, 0)), height, width, device)
            m3 = _get_one_mask(((0, 0), (0, 1)), height, width, device)
            self.mask_cache[key] = (
                torch.cat((m & m0, m & m1, m & m2, m & m3), dim=1),
                torch.cat((m & m3, m & m2, m & m1, m & m0), dim=1),
                torch.cat((m & m2, m & m3, m & m0, m & m1), dim=1),
                torch.cat((m & m1, m & m0, m & m3, m & m2), dim=1),
            )
        return self.mask_cache[key]

    def get_mask_2x(self, batch, channel, height, width, device):
        key = (batch, channel, height, width, str(device), "2x")
        if key not in self.mask_cache:
            if channel % 2 != 0:
                raise RuntimeError(f"Expected channel divisible by 2, got {channel}")
            m = torch.ones((batch, channel // 2, height, width), dtype=torch.bool, device=device)
            m0 = _get_one_mask(((1, 0), (0, 1)), height, width, device)
            m1 = _get_one_mask(((0, 1), (1, 0)), height, width, device)
            self.mask_cache[key] = (
                torch.cat((m & m0, m & m1), dim=1),
                torch.cat((m & m1, m & m0), dim=1),
            )
        return self.mask_cache[key]

    def process_with_mask(self, y, scales, means, mask):
        return process_with_mask_int16(
            y,
            scales,
            means,
            mask,
            quant_cfg=self.quant_cfg,
            force_zero_thres=self.force_zero_thres_int,
        )

    def build_indexes_encoder(self, symbols, scales):
        return build_index_enc_int16(
            symbols,
            scales,
            self.scale_index_lut.to(scales.device),
            force_zero_thres=self.force_zero_thres_int,
        )

    def build_indexes_encoder_full(self, symbols, scales):
        return build_index_enc_int16_full(
            symbols,
            scales,
            self.scale_index_lut.to(scales.device),
        )

    def build_indexes_encoder_keep_mask(self, scales):
        return build_index_enc_skip_mask_int16(
            scales,
            force_zero_thres=self.force_zero_thres_int,
        )

    def build_indexes_decoder(self, scales):
        return build_index_dec_int16(
            scales,
            self.scale_index_lut.to(scales.device),
            force_zero_thres=self.force_zero_thres_int,
        )

    def set_debug_context(self, **kwargs):
        self.debug_context = dict(kwargs)

    def unpack_stream(self, packed_stream, shape, skip_cond=None, expected_indexes=None):
        symbols, indexes = unpack_index_symbols_int16(packed_stream)
        if expected_indexes is not None:
            expected = expected_indexes.to(device=indexes.device, dtype=indexes.dtype)
            if indexes.numel() != expected.numel():
                context_bits = []
                for key in ("frame_idx", "stage", "stream_idx"):
                    if key in self.debug_context:
                        context_bits.append(f"{key}={self.debug_context[key]}")
                context_prefix = ""
                if len(context_bits) > 0:
                    context_prefix = f"[{', '.join(context_bits)}] "
                raise RuntimeError(
                    f"{context_prefix}Packed int16 stream indexes do not match rebuilt decoder indexes: "
                    f"packed_count={indexes.numel()}, expected_count={expected.numel()}."
                )
            if torch.equal(indexes.cpu(), expected.cpu()):
                return expand_decoded_symbols_int16(
                    symbols,
                    shape,
                    skip_cond=skip_cond,
                    device=packed_stream.device,
                ), indexes
            mismatch = indexes != expected
            mismatch_count = int(mismatch.sum().item())
            first_idx = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
            actual_value = int(indexes[first_idx].item())
            expected_value = int(expected[first_idx].item())
            max_abs_delta = int(
                (indexes.to(torch.int16) - expected.to(torch.int16)).abs().amax().item()
            )
            context_bits = []
            for key in ("frame_idx", "stage", "stream_idx"):
                if key in self.debug_context:
                    context_bits.append(f"{key}={self.debug_context[key]}")
            context_prefix = ""
            if len(context_bits) > 0:
                context_prefix = f"[{', '.join(context_bits)}] "
            raise RuntimeError(
                f"{context_prefix}Packed int16 stream indexes do not match rebuilt decoder indexes: "
                f"mismatch_count={mismatch_count}, first_idx={first_idx}, "
                f"actual={actual_value}, expected={expected_value}, max_abs_delta={max_abs_delta}."
            )
        return expand_decoded_symbols_int16(
            symbols,
            shape,
            skip_cond=skip_cond,
            device=packed_stream.device,
        ), indexes

    def bind_runtime_entropy(self, entropy_coder, cdf_group_index):
        self.entropy_coder = entropy_coder
        self.cdf_group_index = cdf_group_index

    def encode_packed_stream(self, packed_stream):
        if self.entropy_coder is None or self.cdf_group_index is None:
            raise RuntimeError("Runtime entropy coder is not bound for int16 stream encoding.")
        self.entropy_coder.encode_y(
            packed_stream.to(device="cpu", dtype=torch.int16),
            self.cdf_group_index,
        )

    def decode_and_get_y_from_stream(self, scales, device=None):
        if self.entropy_coder is None or self.cdf_group_index is None:
            raise RuntimeError("Runtime entropy coder is not bound for int16 stream decoding.")
        device = device or scales.device
        indexes, skip_cond = self.build_indexes_decoder(scales)
        if len(indexes) == 0:
            return torch.zeros(scales.shape, dtype=torch.int16, device=device)
        self.entropy_coder.decode_y(indexes, self.cdf_group_index)
        symbols = self.entropy_coder.get_decoded_tensor(device, torch.int16, non_blocking=True)
        return expand_decoded_symbols_int16(
            symbols,
            scales.shape,
            skip_cond=skip_cond,
            device=device,
        )

    def to(self, device):
        self.scale_index_lut = self.scale_index_lut.to(device=device, dtype=torch.int32)
        self.mask_cache.clear()
        return self


class Int16RefFrame:
    def __init__(self, feature=None, frame=None, poc=None):
        self.feature = feature
        self.frame = frame
        self.poc = poc


def pack_intra_encoder(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "intra_encoder",
        "enc_1": pack_int16_module(module.enc_1, quant_cfg=quant_cfg),
        "enc_2": pack_int16_module(module.enc_2, quant_cfg=quant_cfg),
    }


def pack_intra_decoder(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "intra_decoder",
        "dec_1": pack_int16_module(module.dec_1, quant_cfg=quant_cfg),
        "dec_2": pack_int16_module(module.dec_2, quant_cfg=quant_cfg),
    }


def pack_feature_extractor(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "feature_extractor",
        "conv1": pack_int16_module(module.conv1, quant_cfg=quant_cfg),
        "conv2": pack_int16_module(module.conv2, quant_cfg=quant_cfg),
    }


def pack_video_encoder(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "video_encoder",
        "conv1": pack_int16_module(module.conv1, quant_cfg=quant_cfg),
        "conv2": pack_int16_module(module.conv2, quant_cfg=quant_cfg),
        "conv3": pack_int16_module(module.conv3, quant_cfg=quant_cfg),
        "down": pack_int16_module(module.down, quant_cfg=quant_cfg),
    }


def pack_video_decoder(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "video_decoder",
        "up": pack_int16_module(module.up, quant_cfg=quant_cfg),
        "conv1": pack_int16_module(module.conv1, quant_cfg=quant_cfg),
        "conv2": pack_int16_module(module.conv2, quant_cfg=quant_cfg),
    }


def pack_recon_generation(module, quant_cfg=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    return {
        "kind": "recon_generation",
        "conv": pack_int16_module(module.conv, quant_cfg=quant_cfg),
        "head": pack_int16_module(module.head, quant_cfg=quant_cfg),
    }


class IntraEncoderInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMCI.enc",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.enc_1 = build_int16_runner(
            spec["enc_1"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.enc_1",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.enc_2 = build_int16_runner(
            spec["enc_2"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.enc_2",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward(self, x, quant_step):
        out = pixel_unshuffle_int16(x, 8)
        out = self.enc_1.forward(out)
        out = multiply_int16(out, quant_step, self.quant_cfg.feature_scale)
        return self.enc_2.forward(out)

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.enc_1.to(device)
        self.enc_2.to(device)
        return self


class IntraDecoderInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMCI.dec",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.dec_1 = build_int16_runner(
            spec["dec_1"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.dec_1",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.dec_2 = build_int16_runner(
            spec["dec_2"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.dec_2",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward(self, x, quant_step):
        out = self.dec_1.forward(x)
        out = multiply_int16(out, quant_step, self.quant_cfg.feature_scale)
        out = self.dec_2.forward(out)
        return pixel_shuffle_int16(out, 8)

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.dec_1.to(device)
        self.dec_2.to(device)
        return self


class FeatureExtractorInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMC.feature_extractor",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.conv1 = build_int16_runner(
            spec["conv1"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv1",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.conv2 = build_int16_runner(
            spec["conv2"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv2",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward_part1(self, x, quant_step):
        x1 = self.conv1.forward(x)
        ctx_t = multiply_int16(x1, quant_step, self.quant_cfg.feature_scale)
        return x1, ctx_t

    def forward_part2(self, x1):
        return self.conv2.forward(x1)

    def forward(self, x, quant_step):
        x1, ctx_t = self.forward_part1(x, quant_step)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.conv1.to(device)
        self.conv2.to(device)
        return self


class EncoderInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMC.encoder",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.conv1 = build_int16_runner(
            spec["conv1"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv1",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.conv2 = build_int16_runner(
            spec["conv2"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv2",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.conv3 = build_int16_runner(
            spec["conv3"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv3",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.down = build_int16_runner(
            spec["down"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.down",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward(self, x, ctx, quant_step):
        feature = pixel_unshuffle_int16(x, 8)
        feature = self.conv1.forward(feature)
        feature = self.conv2.forward(concat_int16(feature, ctx, cat_at_front=False))
        feature = _forward_with_quant_fuse(
            self.conv3, feature, quant_step, self.quant_cfg
        )
        return self.down.forward(feature)

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.conv1.to(device)
        self.conv2.to(device)
        self.conv3.to(device)
        self.down.to(device)
        return self


class DecoderInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMC.decoder",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.up = build_int16_runner(
            spec["up"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.up",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.conv1 = build_int16_runner(
            spec["conv1"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv1",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.conv2 = build_int16_runner(
            spec["conv2"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv2",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward(self, x, ctx, quant_step):
        feature = self.up.forward(x)
        feature = self.conv1.forward(concat_int16(feature, ctx, cat_at_front=False))
        return _forward_with_quant_fuse(
            self.conv2, feature, quant_step, self.quant_cfg
        )

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.up.to(device)
        self.conv1.to(device)
        self.conv2.to(device)
        return self


class ReconGenerationInt16Runner:
    def __init__(
        self,
        spec,
        quant_cfg=None,
        wsilu_lut=None,
        module_name="DMC.recon_generation",
        activation_scales=None,
        activation_observer=None,
    ):
        self.quant_cfg = quant_cfg or Int16QuantConfig()
        self.wsilu_lut = wsilu_lut if wsilu_lut is not None else build_wsilu_lut(
            self.quant_cfg.feature_scale
        )
        self.conv = build_int16_runner(
            spec["conv"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.conv",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )
        self.head = build_int16_runner(
            spec["head"],
            self.quant_cfg,
            self.wsilu_lut,
            module_name=f"{module_name}.head",
            activation_scales=activation_scales,
            activation_observer=activation_observer,
        )

    def forward(self, x, quant_step):
        out = _forward_with_quant_fuse(
            self.conv, x, quant_step, self.quant_cfg
        )
        out = self.head.forward(out)
        out = pixel_shuffle_int16(out, 8)
        return clamp_feature_int16(out, min_value=0.0, max_value=1.0, quant_cfg=self.quant_cfg)

    def to(self, device):
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.conv.to(device)
        self.head.to(device)
        return self


def export_dmci_int16_bundle(model, quant_cfg=None, frozen_entropy_state=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    frozen_entropy_state = _resolve_frozen_entropy_state(model, frozen_entropy_state)
    bundle = {
        "format_version": 1,
        "model_type": "DMCI",
        "manifest": export_int16_manifest(quant_cfg),
        "activation_scales": {},
        "sigmoid_lut": build_sigmoid_lut(quant_cfg.feature_scale, quant_cfg.feature_scale),
        "wsilu_lut": build_wsilu_lut(quant_cfg.feature_scale),
        "q_scale_enc": quantize_module_bank(model.q_scale_enc, quant_cfg),
        "q_scale_dec": quantize_module_bank(model.q_scale_dec, quant_cfg),
        "enc": pack_intra_encoder(model.enc, quant_cfg=quant_cfg),
        "hyper_enc": pack_int16_module(model.hyper_enc, quant_cfg=quant_cfg),
        "hyper_dec": pack_int16_module(model.hyper_dec, quant_cfg=quant_cfg),
        "y_prior_fusion": pack_int16_module(model.y_prior_fusion, quant_cfg=quant_cfg),
        "y_spatial_prior_reduction": pack_int16_module(
            model.y_spatial_prior_reduction, quant_cfg=quant_cfg
        ),
        "y_spatial_prior_adaptor_1": pack_int16_module(
            model.y_spatial_prior_adaptor_1, quant_cfg=quant_cfg
        ),
        "y_spatial_prior_adaptor_2": pack_int16_module(
            model.y_spatial_prior_adaptor_2, quant_cfg=quant_cfg
        ),
        "y_spatial_prior_adaptor_3": pack_int16_module(
            model.y_spatial_prior_adaptor_3, quant_cfg=quant_cfg
        ),
        "y_spatial_prior": pack_int16_module(model.y_spatial_prior, quant_cfg=quant_cfg),
        "dec": pack_intra_decoder(model.dec, quant_cfg=quant_cfg),
    }
    if frozen_entropy_state is not None:
        bundle["frozen_entropy_state"] = frozen_entropy_state
    return bundle


def export_dmc_int16_bundle(model, quant_cfg=None, frozen_entropy_state=None):
    quant_cfg = quant_cfg or Int16QuantConfig()
    frozen_entropy_state = _resolve_frozen_entropy_state(model, frozen_entropy_state)
    bundle = {
        "format_version": 1,
        "model_type": "DMC",
        "manifest": export_int16_manifest(quant_cfg),
        "activation_scales": {},
        "wsilu_lut": build_wsilu_lut(quant_cfg.feature_scale),
        "qp_shift": list(model.qp_shift),
        "feature_adaptor_i": pack_int16_module(model.feature_adaptor_i, quant_cfg=quant_cfg),
        "feature_adaptor_p": pack_int16_module(model.feature_adaptor_p, quant_cfg=quant_cfg),
        "feature_extractor": pack_feature_extractor(model.feature_extractor, quant_cfg=quant_cfg),
        "encoder": pack_video_encoder(model.encoder, quant_cfg=quant_cfg),
        "hyper_encoder": pack_int16_module(model.hyper_encoder.conv, quant_cfg=quant_cfg),
        "hyper_decoder": pack_int16_module(model.hyper_decoder.conv, quant_cfg=quant_cfg),
        "temporal_prior_encoder": pack_int16_module(
            model.temporal_prior_encoder, quant_cfg=quant_cfg
        ),
        "y_prior_fusion": pack_int16_module(model.y_prior_fusion.conv, quant_cfg=quant_cfg),
        "y_spatial_prior": pack_int16_module(model.y_spatial_prior.conv, quant_cfg=quant_cfg),
        "decoder": pack_video_decoder(model.decoder, quant_cfg=quant_cfg),
        "recon_generation": pack_recon_generation(
            model.recon_generation_net, quant_cfg=quant_cfg
        ),
        "q_encoder": quantize_module_bank(model.q_encoder, quant_cfg),
        "q_decoder": quantize_module_bank(model.q_decoder, quant_cfg),
        "q_feature": quantize_module_bank(model.q_feature, quant_cfg),
        "q_recon": quantize_module_bank(model.q_recon, quant_cfg),
    }
    if frozen_entropy_state is not None:
        bundle["frozen_entropy_state"] = frozen_entropy_state
    return bundle


class DMCIAnalysisSubmodel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, x, qp):
        x = x.to(self.reference.q_scale_enc.device)
        if x.dtype != torch.int16:
            x = self.reference.to_int16_image(x)
        y = self.reference.encode_y(x, qp)
        z = self.reference.hyper_encode(self.reference._pad_for_y(y))
        z_hat, z_symbols = round_and_to_int8_int16(z, self.reference.quant_cfg)
        params = self.reference.fuse_y_prior(self.reference.hyper_decode(z_hat))
        _, _, y_height, y_width = y.shape
        params = params[:, :, :y_height, :y_width].contiguous()
        prior = self.reference.compress_prior_4x(y, params)
        return {
            "y_hat": prior["y_hat"],
            "z_hat": z_hat,
            "z_symbols": z_symbols,
            "packed_streams": prior["packed_streams"],
            "params": params,
        }


class DMCISynthesisSubmodel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, y_hat, z_hat, qp):  # pylint: disable=unused-argument
        y_hat = y_hat.to(self.reference.q_scale_dec.device)
        if y_hat.dtype != torch.int16:
            y_hat = feature_to_int16(y_hat, self.reference.quant_cfg)
        x_hat = self.reference.decode_x(y_hat, qp)
        return {
            "x_hat": x_hat,
        }


class DMCAnalysisSubmodel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, x, qp, use_ada_i=False, last_qp=None, update_state=True):
        x = x.to(self.reference.q_encoder.device)
        if x.dtype != torch.int16:
            x = feature_to_int16(x, self.reference.quant_cfg)

        if use_ada_i:
            if last_qp is None:
                raise RuntimeError("last_qp is required when use_ada_i=True.")
            self.reference.prepare_feature_adaptor_i(last_qp)
            self.reference.reset_ref_feature()

        ref_feature = self.reference._apply_feature_adaptor_from_dpb(self.reference.encoder_dpb)
        ctx, ctx_t = self.reference.extract_context(ref_feature, qp)
        y = self.reference.encode_y(x, ctx, qp)
        z = self.reference.hyper_encode(self.reference._pad_for_y(y))
        z_hat, z_symbols = round_and_to_int8_int16(z, self.reference.quant_cfg)
        params = self.reference.res_prior_param_decoder(z_hat, ctx_t)
        prior = self.reference.compress_prior_2x(y, params)
        feature = self.reference.decode_feature(prior["y_hat"], ctx, qp)
        if update_state:
            self.reference._add_ref_frame_to_dpb(
                self.reference.encoder_dpb,
                feature=feature,
                frame=None,
                increase_poc=True,
            )
        return {
            "y_hat": prior["y_hat"],
            "z_hat": z_hat,
            "z_symbols": z_symbols,
            "packed_streams": prior["packed_streams"],
            "ctx": ctx,
            "ctx_t": ctx_t,
            "params": params,
            "feature_encode": feature,
        }


class DMCSynthesisSubmodel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, y_hat, z_hat, qp, update_state=True):
        y_hat = y_hat.to(self.reference.q_decoder.device)
        if y_hat.dtype != torch.int16:
            y_hat = feature_to_int16(y_hat, self.reference.quant_cfg)

        ref_feature = self.reference._apply_feature_adaptor_from_dpb(self.reference.decoder_dpb)
        ctx, ctx_t = self.reference.extract_context(ref_feature, qp)
        params = None
        if z_hat is not None:
            z_hat = z_hat.to(self.reference.q_decoder.device)
            if z_hat.dtype != torch.int16:
                z_hat = feature_to_int16(z_hat, self.reference.quant_cfg)
            params = self.reference.res_prior_param_decoder(z_hat, ctx_t)
        x_hat, feature = self.reference.get_recon_and_feature(y_hat, ctx, qp)
        if update_state:
            self.reference._add_ref_frame_to_dpb(
                self.reference.decoder_dpb,
                feature=feature,
                frame=x_hat,
                increase_poc=False,
            )
        return {
            "x_hat": x_hat,
            "feature": feature,
            "ctx": ctx,
            "ctx_t": ctx_t,
            "params": params,
        }


class DMCIInt16Reference:
    def __init__(self, bundle, activation_observer=None):
        if bundle["model_type"] != "DMCI":
            raise RuntimeError(f"Expected a DMCI int16 bundle, got {bundle['model_type']}")

        self.bundle = bundle
        self.quant_cfg = quant_config_from_manifest(bundle["manifest"])
        self.wsilu_lut = bundle["wsilu_lut"]
        self.sigmoid_lut = bundle["sigmoid_lut"]
        self.activation_scales = bundle.get("activation_scales", {})

        self.enc = IntraEncoderInt16Runner(
            bundle["enc"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.enc",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.hyper_enc = build_int16_runner(
            bundle["hyper_enc"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.hyper_enc",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.hyper_dec = build_int16_runner(
            bundle["hyper_dec"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.hyper_dec",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_prior_fusion = build_int16_runner(
            bundle["y_prior_fusion"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_prior_fusion",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior_reduction = build_int16_runner(
            bundle["y_spatial_prior_reduction"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_spatial_prior_reduction",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior_adaptor_1 = build_int16_runner(
            bundle["y_spatial_prior_adaptor_1"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_spatial_prior_adaptor_1",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior_adaptor_2 = build_int16_runner(
            bundle["y_spatial_prior_adaptor_2"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_spatial_prior_adaptor_2",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior_adaptor_3 = build_int16_runner(
            bundle["y_spatial_prior_adaptor_3"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_spatial_prior_adaptor_3",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior = build_int16_runner(
            bundle["y_spatial_prior"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.y_spatial_prior",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.dec = IntraDecoderInt16Runner(
            bundle["dec"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMCI.dec",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )

        self.q_scale_enc = bundle["q_scale_enc"]
        self.q_scale_dec = bundle["q_scale_dec"]
        self.frozen_entropy_state = bundle.get("frozen_entropy_state")
        self.entropy = None
        self.rans_runtime = None
        self.use_two_entropy_coders = False
        if self.frozen_entropy_state is not None:
            self.entropy = Int16GaussianEntropyReference(
                self.frozen_entropy_state,
                quant_cfg=self.quant_cfg,
            )
            self.rans_runtime = Int16RansCodecRuntime(self.frozen_entropy_state)
            self.entropy.bind_runtime_entropy(
                self.rans_runtime.entropy_coder,
                self.rans_runtime.gaussian_encoder.cdf_group_index,
            )
        self._cuda_graph_enabled = False
        self._cuda_graph_input_shape = None
        self._cuda_graph_cache = {}

    @classmethod
    def from_float_model(cls, model, quant_cfg=None, frozen_entropy_state=None):
        return cls(
            export_dmci_int16_bundle(
                model,
                quant_cfg=quant_cfg,
                frozen_entropy_state=frozen_entropy_state,
            )
        )

    def to(self, device):
        device = torch.device(device)
        self._cuda_graph_cache.clear()
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.sigmoid_lut = self.sigmoid_lut.to(device=device, dtype=torch.int16)
        self.q_scale_enc = self.q_scale_enc.to(device=device, dtype=torch.int16)
        self.q_scale_dec = self.q_scale_dec.to(device=device, dtype=torch.int16)
        self.enc.to(device)
        self.hyper_enc.to(device)
        self.hyper_dec.to(device)
        self.y_prior_fusion.to(device)
        self.y_spatial_prior_reduction.to(device)
        self.y_spatial_prior_adaptor_1.to(device)
        self.y_spatial_prior_adaptor_2.to(device)
        self.y_spatial_prior_adaptor_3.to(device)
        self.y_spatial_prior.to(device)
        self.dec.to(device)
        if self.entropy is not None:
            self.entropy.to(device)
        return self

    def cuda(self, device=None):
        if device is None:
            return self.to(torch.device("cuda"))
        if isinstance(device, int):
            return self.to(torch.device(f"cuda:{device}"))
        return self.to(device)

    def eval(self):
        return self

    def get_analysis_submodel(self):
        return DMCIAnalysisSubmodel(self)

    def get_synthesis_submodel(self):
        return DMCISynthesisSubmodel(self)

    def parameters(self):
        yield self.q_scale_enc

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.use_two_entropy_coders = use_two_entropy_coders
        if self.rans_runtime is not None:
            self.rans_runtime.set_use_two_entropy_coders(use_two_entropy_coders)

    @property
    def device(self):
        return self.q_scale_enc.device

    @staticmethod
    def get_downsampled_shape(height, width, p):
        return _get_downsampled_shape(height, width, p)

    def get_q_scale_enc(self, qp, device=None):
        device = device or self.q_scale_enc.device
        return self.q_scale_enc[qp:qp + 1].to(device)

    def get_q_scale_dec(self, qp, device=None):
        device = device or self.q_scale_dec.device
        return self.q_scale_dec[qp:qp + 1].to(device)

    @staticmethod
    def _pad_for_y(y):
        _, _, height, width = y.shape
        pad_r = (4 - (width % 4)) % 4
        pad_b = (4 - (height % 4)) % 4
        if pad_r == 0 and pad_b == 0:
            return y
        return F.pad(y, (0, pad_r, 0, pad_b), mode="replicate")

    def to_int16_image(self, x):
        return feature_to_int16(x, self.quant_cfg)

    def to_float_image(self, x):
        return int16_to_feature(x, self.quant_cfg)

    def encode_y(self, x, qp):
        x = x.to(self.q_scale_enc.device)
        if x.dtype != torch.int16:
            x = self.to_int16_image(x)
        q_scale = self.get_q_scale_enc(qp, device=x.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_encode_y",
            lambda x_i16, q_i16: self.enc.forward(x_i16, q_i16),
            x,
            q_scale,
        )

    def hyper_encode(self, y):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_hyper_encode",
            lambda tensor: self.hyper_enc.forward(tensor),
            y,
        )

    def hyper_decode(self, z_hat):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_hyper_decode",
            lambda tensor: self.hyper_dec.forward(tensor),
            z_hat,
        )

    def fuse_y_prior(self, params):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_fuse_y_prior",
            lambda tensor: self.y_prior_fusion.forward(tensor),
            params,
        )

    def reduce_y_prior(self, params):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_reduce_y_prior",
            lambda tensor: self.y_spatial_prior_reduction.forward(tensor),
            params,
        )

    def adapt_y_spatial(self, adaptor_idx, params):
        if adaptor_idx == 1:
            return _run_cached_cuda_graph(
                self._cuda_graph_cache,
                self._cuda_graph_enabled,
                "dmci_adapt_y_spatial_1",
                lambda tensor: self.y_spatial_prior_adaptor_1.forward(tensor),
                params,
            )
        if adaptor_idx == 2:
            return _run_cached_cuda_graph(
                self._cuda_graph_cache,
                self._cuda_graph_enabled,
                "dmci_adapt_y_spatial_2",
                lambda tensor: self.y_spatial_prior_adaptor_2.forward(tensor),
                params,
            )
        if adaptor_idx == 3:
            return _run_cached_cuda_graph(
                self._cuda_graph_cache,
                self._cuda_graph_enabled,
                "dmci_adapt_y_spatial_3",
                lambda tensor: self.y_spatial_prior_adaptor_3.forward(tensor),
                params,
            )
        raise RuntimeError(f"Unsupported y_spatial adaptor index: {adaptor_idx}")

    def run_y_spatial_prior(self, params):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_run_y_spatial_prior",
            lambda tensor: self.y_spatial_prior.forward(tensor),
            params,
        )

    def separate_prior(self, params):
        return separate_prior_image_int16(
            params,
            self.sigmoid_lut.to(params.device),
            quant_cfg=self.quant_cfg,
        )

    def require_entropy(self):
        if self.entropy is None:
            raise RuntimeError(
                "This int16 reference bundle does not include frozen entropy state."
            )
        return self.entropy

    def require_rans_runtime(self):
        if self.rans_runtime is None:
            raise RuntimeError(
                "This int16 reference bundle does not include runtime entropy coding state."
            )
        return self.rans_runtime

    def compress_prior_4x(self, y, common_params):
        entropy = self.require_entropy()
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        common_params_reduced = self.reduce_y_prior(common_params)
        batch, channel, height, width = y.shape
        mask_0, mask_1, mask_2, mask_3 = entropy.get_mask_4x(
            batch, channel, height, width, y.device
        )

        y = multiply_int16(y, q_enc, self.quant_cfg.feature_scale)

        _, y_q_0, y_hat_0, s_hat_0 = entropy.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(1, params)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = entropy.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = add_int16(y_hat_so_far, y_hat_1)
        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(2, params)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = entropy.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = add_int16(y_hat_so_far, y_hat_2)
        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(3, params)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = entropy.process_with_mask(y, scales, means, mask_3)

        y_hat = multiply_int16(
            add_int16(y_hat_so_far, y_hat_3),
            q_dec,
            self.quant_cfg.feature_scale,
        )

        y_q_w_0 = single_part_for_writing_4x_int16(y_q_0)
        y_q_w_1 = single_part_for_writing_4x_int16(y_q_1)
        y_q_w_2 = single_part_for_writing_4x_int16(y_q_2)
        y_q_w_3 = single_part_for_writing_4x_int16(y_q_3)
        s_w_0 = single_part_for_writing_4x_int16(s_hat_0)
        s_w_1 = single_part_for_writing_4x_int16(s_hat_1)
        s_w_2 = single_part_for_writing_4x_int16(s_hat_2)
        s_w_3 = single_part_for_writing_4x_int16(s_hat_3)

        return {
            "y_hat": y_hat,
            "common_params_reduced": common_params_reduced,
            "packed_streams": (
                entropy.build_indexes_encoder(y_q_w_0, s_w_0),
                entropy.build_indexes_encoder(y_q_w_1, s_w_1),
                entropy.build_indexes_encoder(y_q_w_2, s_w_2),
                entropy.build_indexes_encoder(y_q_w_3, s_w_3),
            ),
            "y_q_streams": (y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3),
            "scale_streams": (s_w_0, s_w_1, s_w_2, s_w_3),
            "q_dec": q_dec,
        }

    def decompress_prior_4x(self, common_params, packed_streams=None, validate_indexes=True):
        entropy = self.require_entropy()
        _, q_dec, scales, means = self.separate_prior(common_params)
        common_params_reduced = self.reduce_y_prior(common_params)
        batch, channel, height, width = means.shape
        mask_0, mask_1, mask_2, mask_3 = entropy.get_mask_4x(
            batch, channel, height, width, means.device
        )

        scales_r = single_part_for_writing_4x_int16(torch.where(mask_0, scales, torch.zeros_like(scales)))
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[0],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_curr_step = restore_y_4x_int16(y_q_r, means, mask_0, quant_cfg=self.quant_cfg)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(1, params)).chunk(2, 1)
        scales_r = single_part_for_writing_4x_int16(torch.where(mask_1, scales, torch.zeros_like(scales)))
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[1],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_curr_step = restore_y_4x_int16(y_q_r, means, mask_1, quant_cfg=self.quant_cfg)
        y_hat_so_far = add_int16(y_hat_so_far, y_hat_curr_step)

        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(2, params)).chunk(2, 1)
        scales_r = single_part_for_writing_4x_int16(torch.where(mask_2, scales, torch.zeros_like(scales)))
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[2],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_curr_step = restore_y_4x_int16(y_q_r, means, mask_2, quant_cfg=self.quant_cfg)
        y_hat_so_far = add_int16(y_hat_so_far, y_hat_curr_step)

        params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
        scales, means = self.run_y_spatial_prior(self.adapt_y_spatial(3, params)).chunk(2, 1)
        scales_r = single_part_for_writing_4x_int16(torch.where(mask_3, scales, torch.zeros_like(scales)))
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[3],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_curr_step = restore_y_4x_int16(y_q_r, means, mask_3, quant_cfg=self.quant_cfg)
        y_hat_so_far = add_int16(y_hat_so_far, y_hat_curr_step)

        return multiply_int16(y_hat_so_far, q_dec, self.quant_cfg.feature_scale)

    def run_intra_reference(self, x, qp, validate_entropy_indexes=True):
        x = x.to(self.q_scale_enc.device)
        if x.dtype != torch.int16:
            x = self.to_int16_image(x)
        y = self.encode_y(x, qp)
        z = self.hyper_encode(self._pad_for_y(y))
        z_hat, z_symbols = round_and_to_int8_int16(z, self.quant_cfg)
        params = self.fuse_y_prior(self.hyper_decode(z_hat))
        _, _, y_height, y_width = y.shape
        params = params[:, :, :y_height, :y_width].contiguous()
        prior_enc = self.compress_prior_4x(y, params)
        y_hat_dec = self.decompress_prior_4x(
            params,
            prior_enc["packed_streams"],
            validate_indexes=validate_entropy_indexes,
        )
        x_hat_enc = self.decode_x(prior_enc["y_hat"], qp)
        x_hat_dec = self.decode_x(y_hat_dec, qp)
        return {
            "y": y,
            "z": z,
            "z_hat": z_hat,
            "z_symbols": z_symbols,
            "params": params,
            "packed_streams": prior_enc["packed_streams"],
            "y_hat_encode": prior_enc["y_hat"],
            "y_hat_decode": y_hat_dec,
            "x_hat_encode": x_hat_enc,
            "x_hat_decode": x_hat_dec,
        }

    def decode_x(self, y_hat, qp):
        q_scale = self.get_q_scale_dec(qp, device=y_hat.device)
        x_hat = _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmci_decode_x",
            lambda y_i16, q_i16: self.dec.forward(y_i16, q_i16),
            y_hat,
            q_scale,
        )
        return clamp_feature_int16(x_hat, min_value=0.0, max_value=1.0, quant_cfg=self.quant_cfg)

    def init_cuda_graph(self, height, width):
        self._cuda_graph_enabled = self.device.type == "cuda"
        self._cuda_graph_input_shape = (1, 3, height, width)
        self._cuda_graph_cache.clear()

    def run_intra_reference_fast(self, x, qp, validate_entropy_indexes=True):
        if self._cuda_graph_input_shape is not None and tuple(x.shape) != self._cuda_graph_input_shape:
            raise RuntimeError(
                f"CUDA graph input shape mismatch: expected {self._cuda_graph_input_shape}, got {tuple(x.shape)}"
            )
        return self.run_intra_reference(x, qp, validate_entropy_indexes=validate_entropy_indexes)

    def compress(self, x, qp):
        if self._cuda_graph_enabled:
            result = self.run_intra_reference_fast(x, qp)
        else:
            result = self.run_intra_reference(x, qp)
        entropy = self.require_entropy()
        rans_runtime = self.require_rans_runtime()
        rans_runtime.set_use_two_entropy_coders(self.use_two_entropy_coders)
        rans_runtime.reset()
        rans_runtime.encode_z(result["z_symbols"], qp)
        packed_streams = result["packed_streams"]
        if isinstance(packed_streams, _DeferredPackedStreams):
            packed_streams = packed_streams.resolve()
        for packed_stream in packed_streams:
            entropy.encode_packed_stream(packed_stream)
        rans_runtime.flush()
        return {
            "bit_stream": rans_runtime.get_encoded_stream(),
            "x_hat": self.to_float_image(result["x_hat_decode"]),
        }

    def decompress(self, bit_stream, sps, qp):
        rans_runtime = self.require_rans_runtime()
        rans_runtime.set_use_two_entropy_coders(sps['ec_part'] == 1)
        rans_runtime.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        z_hat = rans_runtime.decode_z_hat(
            z_size,
            qp,
            self.device,
            self.quant_cfg.feature_scale,
        )
        params = self.fuse_y_prior(self.hyper_decode(z_hat))
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(params)
        x_hat = self.decode_x(y_hat, qp)
        return {
            "x_hat": self.to_float_image(x_hat),
        }


class DMCInt16Reference:
    def __init__(self, bundle, activation_observer=None):
        if bundle["model_type"] != "DMC":
            raise RuntimeError(f"Expected a DMC int16 bundle, got {bundle['model_type']}")

        self.bundle = bundle
        self.quant_cfg = quant_config_from_manifest(bundle["manifest"])
        self.wsilu_lut = bundle["wsilu_lut"]
        self.activation_scales = bundle.get("activation_scales", {})

        self.feature_adaptor_i = build_int16_runner(
            bundle["feature_adaptor_i"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.feature_adaptor_i",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.feature_adaptor_p = build_int16_runner(
            bundle["feature_adaptor_p"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.feature_adaptor_p",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.feature_extractor = FeatureExtractorInt16Runner(
            bundle["feature_extractor"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.feature_extractor",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.encoder = EncoderInt16Runner(
            bundle["encoder"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.encoder",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.hyper_encoder = build_int16_runner(
            bundle["hyper_encoder"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.hyper_encoder",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.hyper_decoder = build_int16_runner(
            bundle["hyper_decoder"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.hyper_decoder",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.temporal_prior_encoder = build_int16_runner(
            bundle["temporal_prior_encoder"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.temporal_prior_encoder",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_prior_fusion = build_int16_runner(
            bundle["y_prior_fusion"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.y_prior_fusion",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.y_spatial_prior = build_int16_runner(
            bundle["y_spatial_prior"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.y_spatial_prior",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.decoder = DecoderInt16Runner(
            bundle["decoder"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.decoder",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )
        self.recon_generation = ReconGenerationInt16Runner(
            bundle["recon_generation"],
            quant_cfg=self.quant_cfg,
            wsilu_lut=self.wsilu_lut,
            module_name="DMC.recon_generation",
            activation_scales=self.activation_scales,
            activation_observer=activation_observer,
        )

        self.q_encoder = bundle["q_encoder"]
        self.q_decoder = bundle["q_decoder"]
        self.q_feature = bundle["q_feature"]
        self.q_recon = bundle["q_recon"]
        self.qp_shift = bundle.get("qp_shift", [0, 8, 4])
        self.frozen_entropy_state = bundle.get("frozen_entropy_state")
        self.entropy = None
        self.rans_runtime = None
        self.use_two_entropy_coders = False
        if self.frozen_entropy_state is not None:
            self.entropy = Int16GaussianEntropyReference(
                self.frozen_entropy_state,
                quant_cfg=self.quant_cfg,
            )
            self.rans_runtime = Int16RansCodecRuntime(self.frozen_entropy_state)
            self.entropy.bind_runtime_entropy(
                self.rans_runtime.entropy_coder,
                self.rans_runtime.gaussian_encoder.cdf_group_index,
            )
        self.encoder_dpb = []
        self.decoder_dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0
        self._cuda_graph_enabled = False
        self._cuda_graph_input_shape = None
        self._cuda_graph_cache = {}
        self.log_frame_stats = False
        self.sync_encoder_state_from_decoder = True
        self._last_profile_result = None
        self._entropy_prep_stream = None

    @classmethod
    def from_float_model(cls, model, quant_cfg=None, frozen_entropy_state=None):
        return cls(
            export_dmc_int16_bundle(
                model,
                quant_cfg=quant_cfg,
                frozen_entropy_state=frozen_entropy_state,
            )
        )

    def to(self, device):
        device = torch.device(device)
        self._cuda_graph_cache.clear()
        self._entropy_prep_stream = None
        self.wsilu_lut = self.wsilu_lut.to(device=device, dtype=torch.int16)
        self.feature_adaptor_i.to(device)
        self.feature_adaptor_p.to(device)
        self.feature_extractor.to(device)
        self.encoder.to(device)
        self.hyper_encoder.to(device)
        self.hyper_decoder.to(device)
        self.temporal_prior_encoder.to(device)
        self.y_prior_fusion.to(device)
        self.y_spatial_prior.to(device)
        self.decoder.to(device)
        self.recon_generation.to(device)
        self.q_encoder = self.q_encoder.to(device=device, dtype=torch.int16)
        self.q_decoder = self.q_decoder.to(device=device, dtype=torch.int16)
        self.q_feature = self.q_feature.to(device=device, dtype=torch.int16)
        self.q_recon = self.q_recon.to(device=device, dtype=torch.int16)
        if self.entropy is not None:
            self.entropy.to(device)
        for dpb in (self.encoder_dpb, self.decoder_dpb):
            for ref in dpb:
                if ref.feature is not None:
                    ref.feature = ref.feature.to(device=device, dtype=torch.int16)
                if ref.frame is not None:
                    ref.frame = ref.frame.to(device=device, dtype=torch.int16)
        return self

    def cuda(self, device=None):
        if device is None:
            return self.to(torch.device("cuda"))
        if isinstance(device, int):
            return self.to(torch.device(f"cuda:{device}"))
        return self.to(device)

    def eval(self):
        return self

    def get_analysis_submodel(self):
        return DMCAnalysisSubmodel(self)

    def get_synthesis_submodel(self):
        return DMCSynthesisSubmodel(self)

    def parameters(self):
        yield self.q_encoder

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.use_two_entropy_coders = use_two_entropy_coders
        if self.rans_runtime is not None:
            self.rans_runtime.set_use_two_entropy_coders(use_two_entropy_coders)

    @property
    def device(self):
        return self.q_encoder.device

    @staticmethod
    def get_downsampled_shape(height, width, p):
        return _get_downsampled_shape(height, width, p)

    def _get_q(self, bank, qp, device=None):
        device = device or bank.device
        return bank[qp:qp + 1].to(device)

    def require_entropy(self):
        if self.entropy is None:
            raise RuntimeError(
                "This int16 reference bundle does not include frozen entropy state."
            )
        return self.entropy

    def require_rans_runtime(self):
        if self.rans_runtime is None:
            raise RuntimeError(
                "This int16 reference bundle does not include runtime entropy coding state."
            )
        return self.rans_runtime

    def _build_packed_streams_sync(self, entropy, y_q_streams, scale_streams):
        return tuple(
            entropy.build_indexes_encoder(y_q_stream, scale_stream)
            for y_q_stream, scale_stream in zip(y_q_streams, scale_streams)
        )

    @staticmethod
    def _resolve_host_packed_streams(host_payloads):
        resolved = []
        for packed_cpu, keep_mask_cpu in host_payloads:
            if keep_mask_cpu is not None:
                resolved.append(packed_cpu[keep_mask_cpu])
            else:
                resolved.append(packed_cpu)
        return tuple(resolved)

    def is_async_entropy_prep_active(self, encode_only=False):
        # [REVERTED] Plan 13.1 async entropy prep was a net negative for wall-clock time
        # due to pinned memory copy overhead. Returning False unconditionally.
        return False

    # Reference-only Plan 13.1 experiment. Keep this available for future Nsight
    # investigations, but gate it off through is_async_entropy_prep_active().
    def _build_packed_streams_2x_async(self, entropy, y_q_parts, scale_parts):
        if self._entropy_prep_stream is None:
            self._entropy_prep_stream = torch.cuda.Stream(device=self.device)
        ready_event = torch.cuda.Event()
        current_stream = torch.cuda.current_stream(device=self.device)
        work_stream = self._entropy_prep_stream
        host_payloads = []

        with torch.cuda.stream(work_stream):
            work_stream.wait_stream(current_stream)
            for y_q_part, scale_part in zip(y_q_parts, scale_parts):
                y_q_stream = single_part_for_writing_2x_int16(y_q_part)
                scale_stream = single_part_for_writing_2x_int16(scale_part)
                packed_full = entropy.build_indexes_encoder_full(y_q_stream, scale_stream)
                keep_mask = entropy.build_indexes_encoder_keep_mask(scale_stream)

                packed_cpu = torch.empty(
                    packed_full.shape,
                    dtype=torch.int16,
                    device="cpu",
                    pin_memory=True,
                )
                packed_cpu.copy_(packed_full, non_blocking=True)

                keep_mask_cpu = None
                if keep_mask is not None:
                    keep_mask_cpu = torch.empty(
                        keep_mask.shape,
                        dtype=torch.bool,
                        device="cpu",
                        pin_memory=True,
                    )
                    keep_mask_cpu.copy_(keep_mask, non_blocking=True)

                host_payloads.append((packed_cpu, keep_mask_cpu))
            ready_event.record(work_stream)

        # Keep dynamic compaction and CPU rANS consumption outside graph capture, but
        # let the full-packed tensors and keep masks move to host while later GPU stages run.
        return _DeferredPackedStreams(
            host_payloads,
            self.device,
            ready_event=ready_event,
            host_wait=True,
            resolver=self._resolve_host_packed_streams,
        )

    def _build_packed_streams(self, entropy, y_q_streams, scale_streams, encode_only=False):
        if not self.is_async_entropy_prep_active(encode_only=encode_only):
            return self._build_packed_streams_sync(entropy, y_q_streams, scale_streams)

        if self._entropy_prep_stream is None:
            self._entropy_prep_stream = torch.cuda.Stream(device=self.device)
        ready_event = torch.cuda.Event()
        current_stream = torch.cuda.current_stream(device=self.device)
        work_stream = self._entropy_prep_stream

        with torch.cuda.stream(work_stream):
            work_stream.wait_stream(current_stream)
            packed_streams = self._build_packed_streams_sync(entropy, y_q_streams, scale_streams)
            ready_event.record(work_stream)

        # Dynamic boolean compaction and the CPU rANS handoff still keep this path
        # outside whole-frame CUDA graph capture for steady-state P-frame encoding.
        return _DeferredPackedStreams(packed_streams, self.device, ready_event=ready_event)

    @staticmethod
    def _pad_for_y(y):
        _, _, height, width = y.shape
        pad_r = (4 - (width % 4)) % 4
        pad_b = (4 - (height % 4)) % 4
        if pad_r == 0 and pad_b == 0:
            return y
        return F.pad(y, (0, pad_r, 0, pad_b), mode="replicate")

    def _add_ref_frame_to_dpb(self, dpb, feature=None, frame=None, increase_poc=True):
        if feature is not None:
            feature = feature.to(self.device)
            if feature.dtype != torch.int16:
                feature = feature_to_int16(feature, self.quant_cfg)
            feature = feature.clone()
        if frame is not None:
            frame = frame.to(self.device)
            if frame.dtype != torch.int16:
                frame = feature_to_int16(frame, self.quant_cfg)
            frame = frame.clone()
        ref_frame = Int16RefFrame(feature=feature, frame=frame, poc=self.curr_poc)
        if len(dpb) >= self.max_dpb_size:
            dpb.pop(-1)
        dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        self.encoder_dpb.clear()
        self.decoder_dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def set_log_frame_stats(self, enabled):
        self.log_frame_stats = bool(enabled)

    def consume_last_profile_result(self):
        result = self._last_profile_result
        self._last_profile_result = None
        return result

    def add_ref_frame(self, feature=None, frame=None, increase_poc=True):
        self._add_ref_frame_to_dpb(self.encoder_dpb, feature=feature, frame=frame,
                                   increase_poc=increase_poc)
        self._add_ref_frame_to_dpb(self.decoder_dpb, feature=feature, frame=frame,
                                   increase_poc=False)

    def _dpb_heads_match(self):
        if len(self.encoder_dpb) == 0 or len(self.decoder_dpb) == 0:
            return False
        enc_ref = self.encoder_dpb[0]
        dec_ref = self.decoder_dpb[0]
        return (
            _optional_tensor_equal(enc_ref.feature, dec_ref.feature)
            and _optional_tensor_equal(enc_ref.frame, dec_ref.frame)
        )

    def _apply_feature_adaptor_from_dpb(self, dpb):
        if len(dpb) == 0:
            raise RuntimeError("DPB is empty; seed it with an I-frame reconstruction first.")
        ref = dpb[0]
        if ref.feature is None:
            if ref.frame is None:
                raise RuntimeError("Reference frame has neither feature nor frame.")
            return self.apply_feature_adaptor(frame=ref.frame)
        return self.apply_feature_adaptor(feature=ref.feature)

    def reset_ref_feature(self):
        if len(self.decoder_dpb) > 0:
            self.decoder_dpb[0].feature = None

    def prepare_feature_adaptor_i(self, last_qp):
        if len(self.encoder_dpb) == 0:
            return
        ref = self.encoder_dpb[0]
        if ref.frame is not None:
            ref.feature = None
        else:
            if ref.feature is None:
                raise RuntimeError("Encoder DPB has no feature to reconstruct the frame from.")
            ref.frame = self.reconstruct_frame(ref.feature, last_qp)
            ref.feature = None

        if len(self.decoder_dpb) > 0:
            dec_ref = self.decoder_dpb[0]
            if dec_ref.frame is None:
                if dec_ref.feature is None:
                    raise RuntimeError("Decoder DPB has no feature to reconstruct the frame from.")
                dec_ref.frame = self.reconstruct_frame(dec_ref.feature, last_qp)
            dec_ref.feature = None

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]

    def apply_feature_adaptor(self, frame=None, feature=None):
        if feature is None:
            if frame is None:
                raise RuntimeError("Either frame or feature must be provided.")
            frame = frame.to(self.q_encoder.device)
            if frame.dtype != torch.int16:
                frame = feature_to_int16(frame, self.quant_cfg)
            return _run_cached_cuda_graph(
                self._cuda_graph_cache,
                self._cuda_graph_enabled,
                "dmc_feature_adaptor_i",
                lambda frame_i16: self.feature_adaptor_i.forward(pixel_unshuffle_int16(frame_i16, 8)),
                frame,
            )
        feature = feature.to(self.q_encoder.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_feature_adaptor_p",
            lambda feature_i16: self.feature_adaptor_p.forward(feature_i16),
            feature,
        )

    def extract_context(self, feature, qp):
        q_feature = self._get_q(self.q_feature, qp, device=feature.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_extract_context",
            lambda feature_i16, q_i16: self.feature_extractor.forward(feature_i16, q_i16),
            feature,
            q_feature,
        )

    def encode_y(self, x, ctx, qp):
        x = x.to(self.q_encoder.device)
        if x.dtype != torch.int16:
            x = feature_to_int16(x, self.quant_cfg)
        q_encoder = self._get_q(self.q_encoder, qp, device=x.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_encode_y",
            lambda x_i16, ctx_i16, q_i16: self.encoder.forward(x_i16, ctx_i16, q_i16),
            x,
            ctx,
            q_encoder,
        )

    def hyper_encode(self, y):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_hyper_encode",
            lambda tensor: self.hyper_encoder.forward(tensor),
            y,
        )

    def res_prior_param_decoder(self, z_hat, ctx_t):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_res_prior_param_decoder",
            lambda z_i16, ctx_t_i16: self._res_prior_param_decoder_impl(z_i16, ctx_t_i16),
            z_hat,
            ctx_t,
        )

    def _res_prior_param_decoder_impl(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder.forward(z_hat)
        temporal_params = self.temporal_prior_encoder.forward(ctx_t)
        _, _, height, width = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :height, :width].contiguous()
        return self.y_prior_fusion.forward(
            concat_int16(hierarchical_params, temporal_params, cat_at_front=False)
        )

    def run_y_spatial_prior(self, params):
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_run_y_spatial_prior",
            lambda tensor: self.y_spatial_prior.forward(tensor),
            params,
        )

    def separate_prior_for_video_encoding(self, params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant_int16(
            q_dec,
            y,
            min_value=0.5,
            quant_cfg=self.quant_cfg,
        )
        return y, q_dec, scales, means

    def separate_prior_for_video_decoding(self, params):
        q_dec, scales, means = params.chunk(3, 1)
        min_q = quantize_scalar_to_int16(0.5, self.quant_cfg)
        q_dec = q_dec.clamp(min_q, 32767).to(torch.int16)
        return q_dec, scales, means

    def compress_prior_2x(self, y, common_params, encode_only=False):
        entropy = self.require_entropy()
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        frame_idx = self.curr_poc
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y", y)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.q_dec", q_dec)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.scales_0", scales)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.means_0", means)
        batch, channel, height, width = y.shape
        mask_0, mask_1 = entropy.get_mask_2x(batch, channel, height, width, y.device)

        _, y_q_0, y_hat_0, s_hat_0 = entropy.process_with_mask(y, scales, means, mask_0)
        scales, means = self.run_y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1)).chunk(2, 1)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y_q_0", y_q_0)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y_hat_0", y_hat_0)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.scales_1", scales)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.means_1", means)
        _, y_q_1, y_hat_1, s_hat_1 = entropy.process_with_mask(y, scales, means, mask_1)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y_q_1", y_q_1)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y_hat_1", y_hat_1)

        y_hat = add_and_multiply_int16(y_hat_0, y_hat_1, q_dec, quant_cfg=self.quant_cfg)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.y_hat", y_hat)

        if self.is_async_entropy_prep_active(encode_only=encode_only):
            y_q_streams = None
            scale_streams = None
            packed_streams = self._build_packed_streams_2x_async(
                entropy,
                (y_q_0, y_q_1),
                (s_hat_0, s_hat_1),
            )
        else:
            y_q_w_0 = single_part_for_writing_2x_int16(y_q_0)
            y_q_w_1 = single_part_for_writing_2x_int16(y_q_1)
            s_w_0 = single_part_for_writing_2x_int16(s_hat_0)
            s_w_1 = single_part_for_writing_2x_int16(s_hat_1)
            y_q_streams = (y_q_w_0, y_q_w_1)
            scale_streams = (s_w_0, s_w_1)
            packed_streams = self._build_packed_streams(
                entropy,
                y_q_streams,
                scale_streams,
                encode_only=encode_only,
            )

        return {
            "y_hat": y_hat,
            "packed_streams": packed_streams,
            "y_q_streams": y_q_streams,
            "scale_streams": scale_streams,
            "q_dec": q_dec,
        }

    def decompress_prior_2x(self, common_params, packed_streams=None, validate_indexes=True, frame_idx=None):
        entropy = self.require_entropy()
        frame_idx = self.curr_poc if frame_idx is None else frame_idx
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.q_dec", q_dec)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.scales_0", scales)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.means_0", means)
        batch, channel, height, width = means.shape
        mask_0, mask_1 = entropy.get_mask_2x(batch, channel, height, width, means.device)

        scales_r = combine_for_reading_2x_int16(scales, mask_0)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.scales_r_0", scales_r)
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            entropy.set_debug_context(frame_idx=frame_idx, stage="decompress_prior_2x", stream_idx=0)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[0],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_0, cat_params = restore_y_2x_with_cat_after_int16(
            y_q_r,
            means,
            mask_0,
            common_params,
            quant_cfg=self.quant_cfg,
        )
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.y_q_0", y_q_r)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.y_hat_0", y_hat_0)

        scales, means = self.run_y_spatial_prior(cat_params).chunk(2, 1)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.scales_1", scales)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.means_1", means)
        scales_r = combine_for_reading_2x_int16(scales, mask_1)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.scales_r_1", scales_r)
        if packed_streams is None:
            y_q_r = entropy.decode_and_get_y_from_stream(scales_r, device=means.device)
        else:
            indexes, skip_cond = entropy.build_indexes_decoder(scales_r)
            entropy.set_debug_context(frame_idx=frame_idx, stage="decompress_prior_2x", stream_idx=1)
            y_q_r, _ = entropy.unpack_stream(
                packed_streams[1],
                scales_r.shape,
                skip_cond=skip_cond,
                expected_indexes=indexes if validate_indexes else None,
            )
        y_hat_1 = restore_y_2x_int16(y_q_r, means, mask_1, quant_cfg=self.quant_cfg)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.y_q_1", y_q_r)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.decode.y_hat_1", y_hat_1)

        return add_and_multiply_int16(y_hat_0, y_hat_1, q_dec, quant_cfg=self.quant_cfg)

    def decode_feature(self, y_hat, ctx, qp):
        q_decoder = self._get_q(self.q_decoder, qp, device=y_hat.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_decode_feature",
            lambda y_i16, ctx_i16, q_i16: self.decoder.forward(y_i16, ctx_i16, q_i16),
            y_hat,
            ctx,
            q_decoder,
        )

    def reconstruct_frame(self, feature, qp):
        q_recon = self._get_q(self.q_recon, qp, device=feature.device)
        return _run_cached_cuda_graph(
            self._cuda_graph_cache,
            self._cuda_graph_enabled,
            "dmc_reconstruct_frame",
            lambda feature_i16, q_i16: self.recon_generation.forward(feature_i16, q_i16),
            feature,
            q_recon,
        )

    def get_recon_and_feature(self, y_hat, ctx, qp):
        feature = self.decode_feature(y_hat, ctx, qp)
        x_hat = self.reconstruct_frame(feature, qp)
        return x_hat, feature

    def run_inter_reference(self, x, qp, use_ada_i=False, last_qp=None,
                            validate_entropy_indexes=True, encode_only=False):
        profiling = _int16_pipeline_profiling_enabled()
        prof = _PipelineProfiler(profiling, self.device if profiling else None)

        x = x.to(self.q_encoder.device)
        if x.dtype != torch.int16:
            x = feature_to_int16(x, self.quant_cfg)
        frame_idx = self.curr_poc

        if use_ada_i:
            if last_qp is None:
                raise RuntimeError("last_qp is required when use_ada_i=True.")
            self.prepare_feature_adaptor_i(last_qp)
            self.reset_ref_feature()

        prof.mark("start")
        share_ref_state = self.sync_encoder_state_from_decoder and self._dpb_heads_match()
        ref_feature_enc = self._apply_feature_adaptor_from_dpb(self.encoder_dpb)
        prof.mark("feature_adaptor_enc")
        ctx_enc, ctx_t_enc = self.extract_context(ref_feature_enc, qp)
        prof.mark("extract_context_enc")
        y = self.encode_y(x, ctx_enc, qp)
        prof.mark("encode_y")
        z = self.hyper_encode(self._pad_for_y(y))
        prof.mark("hyper_encode")
        z_hat, z_symbols = round_and_to_int8_int16(z, self.quant_cfg)
        params_enc = self.res_prior_param_decoder(z_hat, ctx_t_enc)
        prof.mark("res_prior_param_decoder_enc")
        prior_enc = self.compress_prior_2x(y, params_enc, encode_only=encode_only)
        prof.mark("compress_prior_2x")
        feature_enc = self.decode_feature(prior_enc["y_hat"], ctx_enc, qp)
        prof.mark("decode_feature_enc")
        x_hat_enc = self.reconstruct_frame(feature_enc, qp)
        prof.mark("reconstruct_frame_enc")

        if encode_only:
            # ---- ENCODE-ONLY FAST PATH ----
            # Skip decoder-side decompress_prior_2x + get_recon_and_feature.
            # Use encoder-side reconstruction for both DPBs.
            # This is safe because rANS is lossless: the encoder's y_hat
            # is bit-identical to what the decoder would reconstruct.
            self._add_ref_frame_to_dpb(
                self.encoder_dpb,
                feature=feature_enc,
                frame=x_hat_enc,
                increase_poc=True,
            )
            self._add_ref_frame_to_dpb(
                self.decoder_dpb,
                feature=feature_enc,
                frame=x_hat_enc,
                increase_poc=False,
            )
            prof.mark("dpb_update")
            self._last_profile_result = prof.report(frame_idx)

            return {
                "y": y,
                "z": z,
                "z_hat": z_hat,
                "z_symbols": z_symbols,
                "ctx_encode": ctx_enc,
                "ctx_decode": ctx_enc,
                "ctx_t_encode": ctx_t_enc,
                "ctx_t_decode": ctx_t_enc,
                "params_encode": params_enc,
                "params_decode": params_enc,
                "packed_streams": prior_enc["packed_streams"],
                "y_hat_encode": prior_enc["y_hat"],
                "y_hat_decode": prior_enc["y_hat"],
                "feature_encode": feature_enc,
                "feature_decode": feature_enc,
                "x_hat_encode": x_hat_enc,
                "x_hat_decode": x_hat_enc,
            }

        # ---- FULL ENCODE+DECODE PATH (original) ----
        if share_ref_state:
            ref_feature_dec = ref_feature_enc
            ctx_dec = ctx_enc
            ctx_t_dec = ctx_t_enc
        else:
            ref_feature_dec = self._apply_feature_adaptor_from_dpb(self.decoder_dpb)
            ctx_dec, ctx_t_dec = self.extract_context(ref_feature_dec, qp)
        prof.mark("decoder_context")
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.ref_feature_enc", ref_feature_enc)
        _log_tensor_stats(self.log_frame_stats, frame_idx, "pframe.ref_feature_dec", ref_feature_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "ref_feature", ref_feature_enc, ref_feature_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "ctx", ctx_enc, ctx_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "ctx_t", ctx_t_enc, ctx_t_dec)
        if share_ref_state:
            params_dec = params_enc
        else:
            params_dec = self.res_prior_param_decoder(z_hat, ctx_t_dec)
        y_hat_dec = self.decompress_prior_2x(
            params_dec,
            prior_enc["packed_streams"],
            validate_indexes=validate_entropy_indexes,
            frame_idx=frame_idx,
        )
        prof.mark("decompress_prior_2x")
        x_hat_dec, feature_dec = self.get_recon_and_feature(y_hat_dec, ctx_dec, qp)
        prof.mark("decode_and_recon_dec")
        _log_tensor_delta(self.log_frame_stats, frame_idx, "params", params_enc, params_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "y_hat", prior_enc["y_hat"], y_hat_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "feature", feature_enc, feature_dec)
        _log_tensor_delta(self.log_frame_stats, frame_idx, "x_hat", x_hat_enc, x_hat_dec)

        if self.sync_encoder_state_from_decoder:
            next_encoder_feature = feature_dec
            next_encoder_frame = x_hat_dec
        else:
            next_encoder_feature = feature_enc
            next_encoder_frame = None
        self._add_ref_frame_to_dpb(
            self.encoder_dpb,
            feature=next_encoder_feature,
            frame=next_encoder_frame,
            increase_poc=True,
        )
        self._add_ref_frame_to_dpb(self.decoder_dpb, feature=feature_dec, frame=x_hat_dec,
                                   increase_poc=False)
        prof.mark("dpb_update")
        self._last_profile_result = prof.report(frame_idx)

        return {
            "y": y,
            "z": z,
            "z_hat": z_hat,
            "z_symbols": z_symbols,
            "ctx_encode": ctx_enc,
            "ctx_decode": ctx_dec,
            "ctx_t_encode": ctx_t_enc,
            "ctx_t_decode": ctx_t_dec,
            "params_encode": params_enc,
            "params_decode": params_dec,
            "packed_streams": prior_enc["packed_streams"],
            "y_hat_encode": prior_enc["y_hat"],
            "y_hat_decode": y_hat_dec,
            "feature_encode": feature_enc,
            "feature_decode": feature_dec,
            "x_hat_encode": x_hat_enc,
            "x_hat_decode": x_hat_dec,
        }

    def init_cuda_graph_pframe(self, height, width):
        # The long-sequence plan10 failure reproduces only on the graph-enabled P-frame path.
        # Keep this opt-in until the graph replay path is proven stable across reset/use_ada_i frames.
        self._cuda_graph_enabled = self.device.type == "cuda" and _int16_pframe_cuda_graphs_enabled()
        self._cuda_graph_input_shape = (1, 3, height, width)
        self._cuda_graph_cache.clear()

    def run_inter_reference_fast(self, x, qp, use_ada_i=False, last_qp=None,
                                 validate_entropy_indexes=True, encode_only=False):
        if self._cuda_graph_input_shape is not None and tuple(x.shape) != self._cuda_graph_input_shape:
            raise RuntimeError(
                f"CUDA graph input shape mismatch: expected {self._cuda_graph_input_shape}, got {tuple(x.shape)}"
            )
        return self.run_inter_reference(
            x,
            qp,
            use_ada_i=use_ada_i,
            last_qp=last_qp,
            validate_entropy_indexes=validate_entropy_indexes,
            encode_only=encode_only,
        )

    def compress(self, x, qp, encode_only=None):
        profiling = _int16_pipeline_profiling_enabled()
        # Resolve encode_only: explicit parameter > env var > default (False)
        if encode_only is None:
            encode_only = _int16_encode_only_enabled()

        if profiling:
            torch.cuda.synchronize(self.device)
        entropy_t0 = _time_mod.perf_counter()

        if self._cuda_graph_enabled:
            result = self.run_inter_reference_fast(x, qp, encode_only=encode_only)
        else:
            result = self.run_inter_reference(x, qp, encode_only=encode_only)

        if profiling:
            torch.cuda.synchronize(self.device)
        entropy_t1 = _time_mod.perf_counter()

        entropy = self.require_entropy()
        rans_runtime = self.require_rans_runtime()
        rans_runtime.set_use_two_entropy_coders(self.use_two_entropy_coders)
        rans_runtime.reset()
        rans_runtime.encode_z(result["z_symbols"], qp)
        packed_streams = result["packed_streams"]
        if isinstance(packed_streams, _DeferredPackedStreams):
            packed_streams = packed_streams.resolve()
        for packed_stream in packed_streams:
            entropy.encode_packed_stream(packed_stream)
        rans_runtime.flush()

        if profiling:
            entropy_t2 = _time_mod.perf_counter()
            nn_ms = (entropy_t1 - entropy_t0) * 1000.0
            ent_ms = (entropy_t2 - entropy_t1) * 1000.0
            total_ms = (entropy_t2 - entropy_t0) * 1000.0
            mode_str = "ENCODE-ONLY" if encode_only else "FULL"
            print(
                f"[PROFILE COMPRESS {mode_str}] NN forward={nn_ms:.1f}ms  "
                f"entropy_encode={ent_ms:.1f}ms  total={total_ms:.1f}ms"
            )

        return {
            "bit_stream": rans_runtime.get_encoded_stream(),
            "x_hat": self.to_float_image(result["x_hat_decode"]),
        }

    def decompress(self, bit_stream, sps, qp):
        rans_runtime = self.require_rans_runtime()
        rans_runtime.set_use_two_entropy_coders(sps['ec_part'] == 1)
        rans_runtime.set_stream(bit_stream)
        ref_feature = self._apply_feature_adaptor_from_dpb(self.decoder_dpb)
        ctx, ctx_t = self.extract_context(ref_feature, qp)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        z_hat = rans_runtime.decode_z_hat(
            z_size,
            qp,
            self.device,
            self.quant_cfg.feature_scale,
        )
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_hat = self.decompress_prior_2x(params)
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, qp)
        self.add_ref_frame(feature, x_hat)
        return {
            "x_hat": self.to_float_image(x_hat),
        }

    def to_float_image(self, x):
        return int16_to_feature(x, self.quant_cfg)
