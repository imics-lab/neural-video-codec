#include <torch/extension.h>

torch::Tensor conv2d_int16(torch::Tensor input,
                           torch::Tensor weight,
                           c10::optional<torch::Tensor> bias_opt,
                           c10::optional<torch::Tensor> residual_opt,
                           c10::optional<torch::Tensor> residual2_opt,
                           c10::optional<torch::Tensor> post_scale_opt,
                           int64_t stride,
                           int64_t padding,
                           int64_t groups);
torch::Tensor wsilu_chunk_add_int16(torch::Tensor input, torch::Tensor lut);
torch::Tensor add_int16(torch::Tensor a, torch::Tensor b);
torch::Tensor multiply_int16(torch::Tensor input, torch::Tensor scale, int64_t k1);
torch::Tensor conv1x1_int16_gemm(torch::Tensor input,
                                 torch::Tensor weight,
                                 c10::optional<torch::Tensor> bias_opt,
                                 c10::optional<torch::Tensor> residual_opt,
                                 c10::optional<torch::Tensor> residual2_opt,
                                 c10::optional<torch::Tensor> post_scale_opt,
                                 int64_t k2_layer);
torch::Tensor conv1x1_int8tc_gemm(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int64_t weight_int8_scale,
                                  int64_t activation_int8_scale,
                                  int64_t k2_layer);
torch::Tensor conv1x1_int8tc_gemm_per_channel(torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias,
                                              torch::Tensor scale_c,
                                              int64_t activation_int8_scale);
torch::Tensor conv1x1_int8tc_gemm_per_channel_v2(torch::Tensor input,
                                                 torch::Tensor weight,
                                                 torch::Tensor bias,
                                                 torch::Tensor activation_scale_c,
                                                 torch::Tensor eff_scale_c);
torch::Tensor sigmoid_lut_int16(torch::Tensor input, torch::Tensor lut);
torch::Tensor depthwise_conv3x3_lut_fused_int16(torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias,
                                                torch::Tensor lut,
                                                int64_t stride,
                                                int64_t padding);
torch::Tensor scale_index_lut_int16(torch::Tensor scales, torch::Tensor lut);
torch::Tensor clamp_reciprocal_int16(torch::Tensor q, int64_t k1);
torch::Tensor add_multiply_int16(torch::Tensor a,
                                 torch::Tensor b,
                                 torch::Tensor scale,
                                 int64_t k1);
torch::Tensor conv1x1_int16_gemm_wsilu_chunk(torch::Tensor input,
                                             torch::Tensor weight,
                                             c10::optional<torch::Tensor> bias_opt,
                                             torch::Tensor lut,
                                             int64_t k2_layer);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_int16", &conv2d_int16, "int16 conv2d with int64 accumulator");
    m.def("wsilu_chunk_add_int16", &wsilu_chunk_add_int16, "fused wsilu + chunk + add int16");
    m.def("add_int16", &add_int16, "native add int16");
    m.def("multiply_int16", &multiply_int16, "broadcast int16 multiply");
    m.def("conv1x1_int16_gemm", &conv1x1_int16_gemm, "optimized 1x1 int16 conv");
    m.def(
        "conv1x1_int8tc_gemm",
        &conv1x1_int8tc_gemm,
        "1x1 int8 tensor-core conv");
    m.def(
        "conv1x1_int8tc_gemm_per_channel",
        &conv1x1_int8tc_gemm_per_channel,
        "1x1 int8 tensor-core conv with per-channel descale");
    m.def(
        "conv1x1_int8tc_gemm_per_channel_v2",
        &conv1x1_int8tc_gemm_per_channel_v2,
        "1x1 int8 tensor-core conv with per-channel activation cast and descale");
    m.def("sigmoid_lut_int16", &sigmoid_lut_int16, "int16 LUT lookup");
    m.def(
        "depthwise_conv3x3_lut_fused_int16",
        &depthwise_conv3x3_lut_fused_int16,
        "fused int16 LUT + depthwise 3x3 conv");
    m.def("scale_index_lut_int16", &scale_index_lut_int16, "int16 scale-index LUT lookup");
    m.def("clamp_reciprocal_int16", &clamp_reciprocal_int16, "int16 reciprocal lookup");
    m.def("add_multiply_int16", &add_multiply_int16, "int16 fused add+multiply");
    m.def(
        "conv1x1_int16_gemm_wsilu_chunk",
        &conv1x1_int16_gemm_wsilu_chunk,
        "fused 1x1 int16 conv + wsilu chunk-add");
}
