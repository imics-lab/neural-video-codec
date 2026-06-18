#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int32_t kInt16Min = -32768;
constexpr int32_t kInt16Max = 32767;
constexpr int32_t kWeightScale = 8192;
constexpr int32_t kWeightShift = 13;
constexpr int32_t kTileW = 16;
constexpr int32_t kTileH = 16;
constexpr int32_t kConv1x1TilePos = 16;
constexpr int32_t kConv1x1TileOc = 8;
constexpr int32_t kInt8Limit = 127;

cublasHandle_t g_cublas_handle = nullptr;

inline void ensure_cublas_handle(cudaStream_t stream) {
    if (g_cublas_handle == nullptr) {
        const auto status = cublasCreate(&g_cublas_handle);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                    "cublasCreate failed with status ", static_cast<int>(status));
    }
    const auto status = cublasSetStream(g_cublas_handle, stream);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasSetStream failed with status ", static_cast<int>(status));
}

inline void check_cuda_tensor(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_scalar_type(const at::Tensor& tensor,
                              at::ScalarType scalar_type,
                              const char* name) {
    TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has unexpected dtype");
}

inline int32_t power_of_two_shift(int64_t value) {
    int32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

__device__ inline int32_t clamp_int16_device(int64_t value) {
    if (value < kInt16Min) {
        return kInt16Min;
    }
    if (value > kInt16Max) {
        return kInt16Max;
    }
    return static_cast<int32_t>(value);
}

__device__ inline int64_t round_shift_right_device(int64_t value, int bits) {
    if (bits == 0) {
        return value;
    }
    const int64_t offset = static_cast<int64_t>(1) << (bits - 1);
    if (value >= 0) {
        return (value + offset) >> bits;
    }
    return -(((-value) + offset) >> bits);
}

__device__ inline int64_t round_divide_device(int64_t value, int32_t divisor) {
    if (divisor <= 1) {
        return value;
    }
    const int64_t offset = divisor / 2;
    if (value >= 0) {
        return (value + offset) / divisor;
    }
    return -(((-value) + offset) / divisor);
}

__device__ inline int64_t apply_post_scale_device(int64_t value,
                                                  const int16_t* __restrict__ post_scale,
                                                  int64_t post_scale_numel,
                                                  int32_t oc) {
    const int64_t clamped = clamp_int16_device(value);
    if (post_scale == nullptr) {
        return clamped;
    }
    const int64_t scale_idx = post_scale_numel == 1 ? 0 : static_cast<int64_t>(oc);
    const int64_t product = clamped * static_cast<int64_t>(post_scale[scale_idx]);
    return clamp_int16_device(round_shift_right_device(product, 9));
}

__global__ void conv1x1_int16_kernel(const int16_t* __restrict__ input,
                                     const int16_t* __restrict__ weight,
                                     const int32_t* __restrict__ bias,
                                     const int16_t* __restrict__ residual,
                                     const int16_t* __restrict__ residual2,
                                     const int16_t* __restrict__ post_scale,
                                     int16_t* __restrict__ output,
                                     int32_t batch,
                                     int32_t in_channels,
                                     int32_t height,
                                     int32_t width,
                                     int32_t out_channels,
                                     int32_t channels_per_group,
                                     int32_t stride,
                                     int32_t padding,
                                     int32_t groups,
                                     int32_t out_h,
                                     int32_t out_w,
                                     int64_t post_scale_numel,
                                     int32_t k2_layer,
                                     bool has_bias) {
    const int64_t total = static_cast<int64_t>(batch) * out_channels * out_h * out_w;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    int64_t tmp = linear_idx;
    const int32_t ow = tmp % out_w;
    tmp /= out_w;
    const int32_t oh = tmp % out_h;
    tmp /= out_h;
    const int32_t oc = tmp % out_channels;
    const int32_t n = tmp / out_channels;

    const int32_t out_channels_per_group = out_channels / groups;
    const int32_t group = oc / out_channels_per_group;
    const int32_t in_group_offset = group * channels_per_group;
    const int32_t ih = oh * stride - padding;
    const int32_t iw = ow * stride - padding;

    int64_t acc = has_bias ? static_cast<int64_t>(bias[oc]) : 0;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
        for (int32_t icg = 0; icg < channels_per_group; ++icg) {
            const int32_t ic = in_group_offset + icg;
            const int64_t input_idx =
                (((static_cast<int64_t>(n) * in_channels) + ic) * height + ih) * width + iw;
            const int64_t weight_idx =
                (static_cast<int64_t>(oc) * channels_per_group) + icg;
            acc += static_cast<int64_t>(input[input_idx]) *
                   static_cast<int64_t>(weight[weight_idx]);
        }
    }

    int64_t requantized = round_divide_device(acc, k2_layer);
    if (residual != nullptr) {
        requantized += static_cast<int64_t>(residual[linear_idx]);
    }
    if (residual2 != nullptr) {
        requantized += static_cast<int64_t>(residual2[linear_idx]);
    }
    output[linear_idx] = static_cast<int16_t>(apply_post_scale_device(
        requantized, post_scale, post_scale_numel, oc));
}

// Vector loads for aligned int16_t arrays
using int2_16 = int32_t;  // holds 2 int16_t
using int4_16 = int2;     // holds 4 int16_t

__global__ void conv1x1_int16_blocked_kernel(const int16_t* __restrict__ input,
                                             const int16_t* __restrict__ weight,
                                             const int32_t* __restrict__ bias,
                                             const int16_t* __restrict__ residual,
                                             const int16_t* __restrict__ residual2,
                                             const int16_t* __restrict__ post_scale,
                                             int16_t* __restrict__ output,
                                             int32_t batch,
                                             int32_t in_channels,
                                             int32_t height,
                                             int32_t width,
                                             int32_t out_channels,
                                             int64_t post_scale_numel,
                                             int32_t k2_layer,
                                             bool has_bias) {
    const int32_t M = batch * height * width;
    const int32_t N = out_channels;
    const int32_t K = in_channels;

    // Tile size: 64x64 (M x N), K stride: 16
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    
    // Each thread computes a 4x4 tile
    constexpr int TM = 4;
    constexpr int TN = 4;

    const int tx = threadIdx.x % 16; // Thread column index
    const int ty = threadIdx.x / 16; // Thread row index

    const int bx = blockIdx.x; // Block maps to N (out_channels)
    const int by = blockIdx.y; // Block maps to M (spatial/batch)
    // blockIdx.z is unused, we compute M as a flattened dimension

    const int row = by * BM + ty * TM;
    const int col = bx * BN + tx * TN;

    __shared__ int16_t As[BM * BK];
    __shared__ int16_t Bs[BK * BN];

    int32_t acc[TM][TN] = {0};

    // Pointers for global loads
    // input is [batch, C_in, height, width]. This is NOT row-major [M, K].
    // Wait, the input layout is [batch, C_in, height, width] = [N, C, H, W]
    // We need to fetch it correctly.
    // If we want [M, K] access, we have to remap indices.
    
    // Actually, in the current codebase, input is NCHW. So input[n, ic, h, w].
    // M = batch * height * width. row = m = n * height * width + h * width + w.
    // col = c_out. k = c_in.
    // So global read of A should be input[m_n, k, m_h, m_w].
    
    // Vectorized loads via int4_16
    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        // Load As from global to shared
        // As is [BM, BK] = [64, 16] = 1024 elements.
        // 256 threads -> 4 elements per thread.
        // Let's load 1 row of As per 4 threads. 64 rows * (16 / 4) = 64 * 4 = 256 threads.
        // So threadIdx.x handles row `threadIdx.x / 4` and cols `(threadIdx.x % 4) * 4`.
        int a_row = by * BM + (threadIdx.x / 4);
        int a_col = k_idx + (threadIdx.x % 4) * 4;
        
        if (a_row < M && a_col < K) {
            int m_n = a_row / (height * width);
            int m_rem = a_row % (height * width);
            
            for (int i=0; i<4; i++) {
                if (a_col + i < K) {
                    int64_t input_idx = (((int64_t)m_n * K) + (a_col + i)) * height * width + m_rem;
                    As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = input[input_idx];
                } else {
                    As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = 0;
                }
            }
        } else {
            for (int i=0; i<4; i++) {
                As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = 0;
            }
        }

        // Load Bs from global to shared
        // Bs is [BK, BN] = [16, 64] = 1024 elements.
        // We want weight[N, K] loaded into Bs[K, N], so it's transposed.
        // 1 row of Bs (length 64) takes 16 threads (4 elements each). There are 16 rows. 16 * 16 = 256 threads.
        // Wait, 16 threads * 4 elem = 64. So threadIdx.x / 16 gives the row of Bs (0..15).
        // threadIdx.x % 16 gives the chunk of 4 cols.
        int bs_row = threadIdx.x / 16;
        int bs_col = (threadIdx.x % 16) * 4;
        
        int b_row = bx * BN + bs_col;      // Maps to N (out_channels)
        int b_col = k_idx + bs_row;        // Maps to K (in_channels)
        
        for (int i=0; i<4; i++) {
            if ((b_row + i) < N && b_col < K) {
                // weight is [N, K] => weight[(b_row+i) * K + b_col]
                Bs[bs_row * BN + bs_col + i] = weight[(b_row + i) * K + b_col];
            } else {
                Bs[bs_row * BN + bs_col + i] = 0;
            }
        }
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int16_t a_frag[TM];
            int16_t b_frag[TN];
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[(ty * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k * BN + (tx * TN + j)];
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += (int32_t)a_frag[i] * (int32_t)b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int out_r = row + i; // Maps to M
            int out_c = col + j; // Maps to N
            if (out_r < M && out_c < N) {
                int64_t val = acc[i][j];
                if (has_bias) val += bias[out_c];
                
                int m_n = out_r / (height * width);
                int m_rem = out_r % (height * width);
                int m_h = m_rem / width;
                int m_w = m_rem % width;
                
                // ((n * out_channels) + oc) * height + ih) * width + iw
                int64_t output_idx = (((int64_t)m_n * N) + out_c) * height * width + m_rem;
                
                const int64_t offset = k2_layer / 2;
                if (val >= 0) val = (val + offset) / k2_layer;
                else val = -(((-val) + offset) / k2_layer);
                
                if (residual != nullptr) {
                    val += static_cast<int64_t>(residual[output_idx]);
                }
                if (residual2 != nullptr) {
                    val += static_cast<int64_t>(residual2[output_idx]);
                }
                output[output_idx] = static_cast<int16_t>(apply_post_scale_device(
                    val, post_scale, post_scale_numel, out_c));
            }
        }
    }
}

__global__ void depthwise_conv2d_int16_kernel(const int16_t* __restrict__ input,
                                              const int16_t* __restrict__ weight,
                                              const int32_t* __restrict__ bias,
                                              const int16_t* __restrict__ post_scale,
                                              int16_t* __restrict__ output,
                                              int32_t batch,
                                              int32_t channels,
                                              int32_t height,
                                              int32_t width,
                                              int32_t kernel_h,
                                              int32_t kernel_w,
                                              int32_t stride,
                                              int32_t padding,
                                              int32_t out_h,
                                              int32_t out_w,
                                              int64_t post_scale_numel,
                                              bool has_bias) {
    const int64_t total = static_cast<int64_t>(batch) * channels * out_h * out_w;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    int64_t tmp = linear_idx;
    const int32_t ow = tmp % out_w;
    tmp /= out_w;
    const int32_t oh = tmp % out_h;
    tmp /= out_h;
    const int32_t c = tmp % channels;
    const int32_t n = tmp / channels;

    int64_t acc = has_bias ? static_cast<int64_t>(bias[c]) : 0;
    for (int32_t kh = 0; kh < kernel_h; ++kh) {
        const int32_t ih = oh * stride + kh - padding;
        if (ih < 0 || ih >= height) {
            continue;
        }
        for (int32_t kw = 0; kw < kernel_w; ++kw) {
            const int32_t iw = ow * stride + kw - padding;
            if (iw < 0 || iw >= width) {
                continue;
            }
            const int64_t input_idx =
                (((static_cast<int64_t>(n) * channels) + c) * height + ih) * width + iw;
            const int64_t weight_idx =
                (((static_cast<int64_t>(c) * kernel_h) + kh) * kernel_w) + kw;
            acc += static_cast<int64_t>(input[input_idx]) *
                   static_cast<int64_t>(weight[weight_idx]);
        }
    }

    const int64_t requantized = round_shift_right_device(acc, kWeightShift);
    output[linear_idx] = static_cast<int16_t>(apply_post_scale_device(
        requantized, post_scale, post_scale_numel, c));
}

__global__ void requantize_gemm_output_int32_kernel(const int32_t* __restrict__ input,
                                                    const int32_t* __restrict__ bias,
                                                    int16_t* __restrict__ output,
                                                    int32_t rows,
                                                    int32_t cols,
                                                    bool has_bias) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    const int32_t col = static_cast<int32_t>(linear_idx % cols);
    int64_t value = static_cast<int64_t>(input[linear_idx]);
    if (has_bias) {
        value += static_cast<int64_t>(bias[col]);
    }
    output[linear_idx] = static_cast<int16_t>(clamp_int16_device(round_shift_right_device(
        value, kWeightShift)));
}

__global__ void cast_int16_to_int8_kernel(const int16_t* __restrict__ input,
                                          int8_t* __restrict__ output,
                                          int64_t total,
                                          int32_t activation_scale_shift) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int64_t value = input[idx];
    value = round_shift_right_device(value, activation_scale_shift);
    if (value < -kInt8Limit) {
        value = -kInt8Limit;
    } else if (value > kInt8Limit) {
        value = kInt8Limit;
    }
    output[idx] = static_cast<int8_t>(value);
}

__global__ void cast_int16_to_int8_per_channel_kernel(
    const int16_t* __restrict__ input,
    int8_t* __restrict__ output,
    const int32_t* __restrict__ activation_scale_c,
    int32_t rows,
    int32_t cols) {
    const int32_t col = static_cast<int32_t>(blockIdx.y);
    const int32_t row = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }

    const int64_t idx = static_cast<int64_t>(row) * cols + col;
    int32_t scale = activation_scale_c[col];
    if (scale < 1) {
        scale = 1;
    }

    int64_t value = static_cast<int64_t>(input[idx]);
    int32_t shift = 0;
    while (scale > 1) {
        scale >>= 1;
        ++shift;
    }
    value = round_shift_right_device(value, shift);
    if (value < -kInt8Limit) {
        value = -kInt8Limit;
    } else if (value > kInt8Limit) {
        value = kInt8Limit;
    }
    output[idx] = static_cast<int8_t>(value);
}

__global__ void descale_int32_to_int16_kernel(const int32_t* __restrict__ input,
                                              const int32_t* __restrict__ bias,
                                              int16_t* __restrict__ output,
                                              int32_t rows,
                                              int32_t cols,
                                              int32_t scale_multiplier,
                                              int32_t k2_layer,
                                              bool has_bias) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    const int32_t col = static_cast<int32_t>(linear_idx % cols);
    int64_t value = static_cast<int64_t>(input[linear_idx]) * scale_multiplier;
    if (has_bias) {
        value += static_cast<int64_t>(bias[col]);
    }
    output[linear_idx] = static_cast<int16_t>(
        clamp_int16_device(round_divide_device(value, k2_layer)));
}

__global__ void descale_int32_per_channel_to_int16_kernel(
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ bias,
    const float* __restrict__ scale_c,
    const int16_t* __restrict__ residual,
    int16_t* __restrict__ output,
    int32_t rows,
    int32_t cols,
    bool has_bias) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    const int32_t col = static_cast<int32_t>(linear_idx % cols);
    float value = static_cast<float>(input[linear_idx]) * scale_c[col];
    if (has_bias) {
        value += static_cast<float>(bias[col]) / static_cast<float>(kWeightScale);
    }
    int64_t val = static_cast<int64_t>(__float2int_rn(value));
    if (residual != nullptr) {
        val += static_cast<int64_t>(residual[linear_idx]);
    }
    output[linear_idx] = static_cast<int16_t>(clamp_int16_device(val));
}

__global__ void depthwise_conv3x3_lut_fused_int16_kernel(
    const int16_t* __restrict__ input,
    const int16_t* __restrict__ weight,
    const int32_t* __restrict__ bias,
    const int16_t* __restrict__ lut,
    int16_t* __restrict__ output,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t stride,
    int32_t padding,
    int32_t out_h,
    int32_t out_w,
    bool has_bias) {
    const int32_t ow = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t oh = static_cast<int32_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int32_t z = static_cast<int32_t>(blockIdx.z);
    const int32_t c = z % channels;
    const int32_t n = z / channels;

    if (n >= batch || oh >= out_h || ow >= out_w) {
        return;
    }

    int64_t acc = has_bias ? static_cast<int64_t>(bias[c]) : 0;
    const int64_t input_base = ((static_cast<int64_t>(n) * channels) + c) * height * width;
    const int64_t weight_base = static_cast<int64_t>(c) * 9;
    #pragma unroll
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t ih = oh * stride + kh - padding;
        if (ih < 0 || ih >= height) {
            continue;
        }
        #pragma unroll
        for (int32_t kw = 0; kw < 3; ++kw) {
            const int32_t iw = ow * stride + kw - padding;
            if (iw < 0 || iw >= width) {
                continue;
            }
            const int16_t raw = input[input_base + static_cast<int64_t>(ih) * width + iw];
            const uint16_t lut_idx =
                static_cast<uint16_t>(static_cast<int32_t>(raw) - kInt16Min);
            const int16_t activated = lut[lut_idx];
            acc += static_cast<int64_t>(activated) *
                   static_cast<int64_t>(weight[weight_base + kh * 3 + kw]);
        }
    }

    const int64_t output_idx =
        (((static_cast<int64_t>(n) * channels) + c) * out_h + oh) * out_w + ow;
    output[output_idx] = static_cast<int16_t>(clamp_int16_device(round_shift_right_device(
        acc, kWeightShift)));
}

template <int32_t BLOCK_C>
__global__ void conv2d_int16_tiled_kernel(const int16_t* __restrict__ input,
                                          const int16_t* __restrict__ weight,
                                          const int32_t* __restrict__ bias,
                                          const int16_t* __restrict__ post_scale,
                                          int16_t* __restrict__ output,
                                          int32_t batch,
                                          int32_t in_channels,
                                          int32_t height,
                                          int32_t width,
                                          int32_t out_channels,
                                          int32_t channels_per_group,
                                          int32_t kernel_h,
                                          int32_t kernel_w,
                                          int32_t stride,
                                          int32_t padding,
                                          int32_t groups,
                                          int32_t out_h,
                                          int32_t out_w,
                                          int64_t post_scale_numel,
                                          bool has_bias) {
    extern __shared__ int16_t shared_input[];

    const int32_t ow = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t oh = static_cast<int32_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int32_t z = static_cast<int32_t>(blockIdx.z);
    const int32_t oc = z % out_channels;
    const int32_t n = z / out_channels;

    if (n >= batch) {
        return;
    }

    const int32_t tile_input_h = blockDim.y * stride + kernel_h - 1;
    const int32_t tile_input_w = blockDim.x * stride + kernel_w - 1;
    const int32_t tile_base_h = static_cast<int32_t>(blockIdx.y) * blockDim.y * stride - padding;
    const int32_t tile_base_w = static_cast<int32_t>(blockIdx.x) * blockDim.x * stride - padding;
    const int32_t thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int32_t threads_per_block = blockDim.x * blockDim.y;

    const int32_t out_channels_per_group = out_channels / groups;
    const int32_t group = oc / out_channels_per_group;
    const int32_t in_group_offset = group * channels_per_group;

    int64_t acc = has_bias ? static_cast<int64_t>(bias[oc]) : 0;

    const int32_t tile_area = tile_input_h * tile_input_w;
    for (int32_t icg_base = 0; icg_base < channels_per_group; icg_base += BLOCK_C) {
        const int32_t ic_count = min(BLOCK_C, channels_per_group - icg_base);
        const int32_t block_elems = ic_count * tile_area;
        for (int32_t idx = thread_linear; idx < block_elems; idx += threads_per_block) {
            const int32_t icg_off = idx / tile_area;
            const int32_t tile_offset = idx % tile_area;
            const int32_t tile_y = tile_offset / tile_input_w;
            const int32_t tile_x = tile_offset % tile_input_w;
            const int32_t ic = in_group_offset + icg_base + icg_off;
            const int32_t ih = tile_base_h + tile_y;
            const int32_t iw = tile_base_w + tile_x;
            int16_t value = 0;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                const int64_t input_idx =
                    (((static_cast<int64_t>(n) * in_channels) + ic) * height + ih) * width + iw;
                value = input[input_idx];
            }
            shared_input[idx] = value;
        }
        __syncthreads();

        if (ow < out_w && oh < out_h) {
            const int32_t shared_y = threadIdx.y * stride;
            const int32_t shared_x = threadIdx.x * stride;
            
            if (kernel_h == 3 && kernel_w == 3) {
                for (int32_t icg_off = 0; icg_off < ic_count; ++icg_off) {
                    const int32_t icg = icg_base + icg_off;
                    const int64_t weight_base =
                        (static_cast<int64_t>(oc) * channels_per_group + icg) * 9;
                    const int32_t shared_base = icg_off * tile_area + shared_y * tile_input_w + shared_x;
                    
                    // Unrolled 3x3 loop
                    int64_t sum = 0;
                    sum += static_cast<int64_t>(shared_input[shared_base]) * static_cast<int64_t>(weight[weight_base]);
                    sum += static_cast<int64_t>(shared_input[shared_base + 1]) * static_cast<int64_t>(weight[weight_base + 1]);
                    sum += static_cast<int64_t>(shared_input[shared_base + 2]) * static_cast<int64_t>(weight[weight_base + 2]);
                    
                    const int32_t shared_base1 = shared_base + tile_input_w;
                    sum += static_cast<int64_t>(shared_input[shared_base1]) * static_cast<int64_t>(weight[weight_base + 3]);
                    sum += static_cast<int64_t>(shared_input[shared_base1 + 1]) * static_cast<int64_t>(weight[weight_base + 4]);
                    sum += static_cast<int64_t>(shared_input[shared_base1 + 2]) * static_cast<int64_t>(weight[weight_base + 5]);
                    
                    const int32_t shared_base2 = shared_base + 2 * tile_input_w;
                    sum += static_cast<int64_t>(shared_input[shared_base2]) * static_cast<int64_t>(weight[weight_base + 6]);
                    sum += static_cast<int64_t>(shared_input[shared_base2 + 1]) * static_cast<int64_t>(weight[weight_base + 7]);
                    sum += static_cast<int64_t>(shared_input[shared_base2 + 2]) * static_cast<int64_t>(weight[weight_base + 8]);
                    
                    acc += sum;
                }
            } else {
                for (int32_t icg_off = 0; icg_off < ic_count; ++icg_off) {
                    const int32_t icg = icg_base + icg_off;
                    const int64_t weight_base =
                        (static_cast<int64_t>(oc) * channels_per_group + icg) * kernel_h * kernel_w;
                    const int32_t shared_channel_offset = icg_off * tile_area;
                    for (int32_t kh = 0; kh < kernel_h; ++kh) {
                        for (int32_t kw = 0; kw < kernel_w; ++kw) {
                            const int32_t tile_idx =
                                shared_channel_offset + (shared_y + kh) * tile_input_w + (shared_x + kw);
                            const int64_t weight_idx = weight_base + kh * kernel_w + kw;
                            acc += static_cast<int64_t>(shared_input[tile_idx]) *
                                   static_cast<int64_t>(weight[weight_idx]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ow < out_w && oh < out_h) {
        const int64_t requantized = round_shift_right_device(acc, kWeightShift);
        const int64_t output_idx =
            (((static_cast<int64_t>(n) * out_channels) + oc) * out_h + oh) * out_w + ow;
        output[output_idx] = static_cast<int16_t>(apply_post_scale_device(
            requantized, post_scale, post_scale_numel, oc));
    }
}

__global__ void lut_lookup_int16_kernel(const int16_t* __restrict__ input,
                                        const int16_t* __restrict__ lut,
                                        int16_t* __restrict__ output,
                                        int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const uint16_t lut_idx = static_cast<uint16_t>(static_cast<int32_t>(input[idx]) - kInt16Min);
    output[idx] = lut[lut_idx];
}

__global__ void scale_index_lut_kernel(const int16_t* __restrict__ input,
                                       const int32_t* __restrict__ lut,
                                       int32_t* __restrict__ output,
                                       int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const uint16_t lut_idx = static_cast<uint16_t>(static_cast<int32_t>(input[idx]) - kInt16Min);
    output[idx] = lut[lut_idx];
}

__global__ void clamp_reciprocal_int16_kernel(const int16_t* __restrict__ input,
                                              int16_t* __restrict__ output,
                                              int32_t k1,
                                              int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int32_t q = static_cast<int32_t>(input[idx]);
    if (q < 1) {
        q = 1;
    }
    const int64_t numerator = static_cast<int64_t>(k1) * static_cast<int64_t>(k1);
    const int64_t truncated = numerator / q;
    output[idx] = static_cast<int16_t>(clamp_int16_device(truncated));
}

__global__ void add_multiply_int16_kernel(const int16_t* __restrict__ a,
                                          const int16_t* __restrict__ b,
                                          const int16_t* __restrict__ scale,
                                          int16_t* __restrict__ output,
                                          int32_t k1,
                                          int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t sum = static_cast<int64_t>(a[idx]) + static_cast<int64_t>(b[idx]);
    const int64_t product = sum * static_cast<int64_t>(scale[idx]);
    const int64_t rounded = round_shift_right_device(product, 9);
    output[idx] = static_cast<int16_t>(clamp_int16_device(rounded));
}

__global__ void multiply_int16_broadcast_kernel(const int16_t* __restrict__ input,
                                                const int16_t* __restrict__ scale,
                                                int16_t* __restrict__ output,
                                                int32_t channels,
                                                int32_t spatial_size,
                                                int64_t scale_numel,
                                                int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx / spatial_size;
    const int32_t c = static_cast<int32_t>(tmp % channels);
    const int64_t scale_idx =
        scale_numel == total ? idx : (scale_numel == 1 ? 0 : static_cast<int64_t>(c));
    const int64_t product = static_cast<int64_t>(input[idx]) * static_cast<int64_t>(scale[scale_idx]);
    output[idx] = static_cast<int16_t>(clamp_int16_device(round_shift_right_device(product, 9)));
}

int64_t num_blocks_for(int64_t total, int threads_per_block = 256) {
    return (total + threads_per_block - 1) / threads_per_block;
}

// ---- Fused conv1x1 + WSiLU-chunk-add kernel ----
__global__ void conv1x1_int16_blocked_wsilu_chunk_kernel(
        const int16_t* __restrict__ input,
        const int16_t* __restrict__ weight,
        const int32_t* __restrict__ bias,
        const int16_t* __restrict__ lut,
        int16_t* __restrict__ output,
        int32_t batch,
        int32_t in_channels,
        int32_t height,
        int32_t width,
        int32_t half_channels,
        int32_t k2_layer,
        bool has_bias) {
    const int32_t M = batch * height * width;
    const int32_t K = in_channels;
    const int32_t out_N = half_channels;
    const int32_t full_N = half_channels * 2;

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int row = by * BM + ty * TM;
    const int col = bx * BN + tx * TN;

    __shared__ int16_t As[BM * BK];
    __shared__ int16_t Bs0[BK * BN];
    __shared__ int16_t Bs1[BK * BN];

    int32_t acc_lo[TM][TN] = {0};
    int32_t acc_hi[TM][TN] = {0};

    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        int a_row = by * BM + (threadIdx.x / 4);
        int a_col = k_idx + (threadIdx.x % 4) * 4;
        if (a_row < M && a_col < K) {
            int m_n = a_row / (height * width);
            int m_rem = a_row % (height * width);
            for (int i = 0; i < 4; i++) {
                if (a_col + i < K) {
                    int64_t input_idx = (((int64_t)m_n * K) + (a_col + i)) * height * width + m_rem;
                    As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = input[input_idx];
                } else {
                    As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = 0;
                }
            }
        } else {
            for (int i = 0; i < 4; i++)
                As[(threadIdx.x / 4) * BK + (threadIdx.x % 4) * 4 + i] = 0;
        }

        int bs_row = threadIdx.x / 16;
        int bs_col = (threadIdx.x % 16) * 4;
        int b_oc_lo = bx * BN + bs_col;
        int b_oc_hi = b_oc_lo + half_channels;
        int b_k = k_idx + bs_row;
        for (int i = 0; i < 4; i++) {
            Bs0[bs_row * BN + bs_col + i] =
                ((b_oc_lo + i) < out_N && b_k < K) ? weight[(b_oc_lo + i) * K + b_k] : 0;
            Bs1[bs_row * BN + bs_col + i] =
                ((b_oc_hi + i) < full_N && b_k < K) ? weight[(b_oc_hi + i) * K + b_k] : 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int16_t a_frag[TM], b0[TN], b1[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[(ty * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b0[j] = Bs0[k * BN + (tx * TN + j)];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b1[j] = Bs1[k * BN + (tx * TN + j)];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc_lo[i][j] += (int32_t)a_frag[i] * (int32_t)b0[j];
                    acc_hi[i][j] += (int32_t)a_frag[i] * (int32_t)b1[j];
                }
            }
        }
        __syncthreads();
    }

    const int64_t offset = k2_layer / 2;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int out_r = row + i;
            int out_c = col + j;
            if (out_r < M && out_c < out_N) {
                int64_t vlo = acc_lo[i][j];
                if (has_bias) vlo += bias[out_c];
                vlo = (vlo >= 0) ? (vlo + offset) / k2_layer : -(((-vlo) + offset) / k2_layer);
                vlo = clamp_int16_device(vlo);

                int64_t vhi = acc_hi[i][j];
                if (has_bias) vhi += bias[out_c + half_channels];
                vhi = (vhi >= 0) ? (vhi + offset) / k2_layer : -(((-vhi) + offset) / k2_layer);
                vhi = clamp_int16_device(vhi);

                int64_t result = static_cast<int64_t>(lut[static_cast<int32_t>(vlo) - kInt16Min])
                               + static_cast<int64_t>(lut[static_cast<int32_t>(vhi) - kInt16Min]);
                result = clamp_int16_device(result);

                int m_n = out_r / (height * width);
                int m_rem = out_r % (height * width);
                int64_t oidx = (((int64_t)m_n * out_N) + out_c) * height * width + m_rem;
                output[oidx] = static_cast<int16_t>(result);
            }
        }
    }
}

}  // namespace

at::Tensor conv1x1_int16_gemm(at::Tensor input,
                              at::Tensor weight,
                              c10::optional<at::Tensor> bias_opt,
                              c10::optional<at::Tensor> residual_opt,
                              c10::optional<at::Tensor> residual2_opt,
                              c10::optional<at::Tensor> post_scale_opt,
                              int64_t k2_layer) {
    at::Tensor bias; if(bias_opt.has_value()) bias = bias_opt.value();
    at::Tensor residual; if(residual_opt.has_value()) residual = residual_opt.value();
    at::Tensor residual2; if(residual2_opt.has_value()) residual2 = residual2_opt.value();
    at::Tensor post_scale; if(post_scale_opt.has_value()) post_scale = post_scale_opt.value();

    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kShort, "weight");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    
    const bool has_residual = residual.defined() && residual.numel() > 0;
    if (has_residual) {
        check_cuda_tensor(residual, "residual");
        check_scalar_type(residual, at::kShort, "residual");
        residual = residual.contiguous();
    }
    const bool has_residual2 = residual2.defined() && residual2.numel() > 0;
    if (has_residual2) {
        check_cuda_tensor(residual2, "residual2");
        check_scalar_type(residual2, at::kShort, "residual2");
        residual2 = residual2.contiguous();
    }
    const bool has_post_scale = post_scale.defined() && post_scale.numel() > 0;
    if (has_post_scale) {
        check_cuda_tensor(post_scale, "post_scale");
        check_scalar_type(post_scale, at::kShort, "post_scale");
        post_scale = post_scale.contiguous();
    }
    

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions");
    TORCH_CHECK(k2_layer >= 1, "k2_layer must be >= 1");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto out_channels = static_cast<int32_t>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == out_channels, "bias must have one value per output channel");
    }
    if (has_post_scale) {
        TORCH_CHECK(
            post_scale.numel() == 1 || post_scale.numel() == out_channels,
            "post_scale must have one value or one value per output channel");
    }

    auto output = at::empty({batch, out_channels, height, width},
                            at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool has_bias = bias.defined() && bias.numel() > 0;
    const dim3 threads_per_block(256);
    const int32_t M = batch * height * width;
    const int32_t N = out_channels;
    
    const dim3 blocks(
        static_cast<uint32_t>((N + 63) / 64),
        static_cast<uint32_t>((M + 63) / 64),
        1);

    conv1x1_int16_blocked_kernel<<<blocks, threads_per_block, 0,
                                   stream.stream()>>>(
        input.data_ptr<int16_t>(),
        weight.data_ptr<int16_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        has_residual ? residual.data_ptr<int16_t>() : nullptr,
        has_residual2 ? residual2.data_ptr<int16_t>() : nullptr,
        has_post_scale ? post_scale.data_ptr<int16_t>() : nullptr,
        output.data_ptr<int16_t>(),
        batch,
        in_channels,
        height,
        width,
        out_channels,
        has_post_scale ? post_scale.numel() : 0,
        static_cast<int32_t>(k2_layer),
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


// Host launcher for fused conv1x1 + WSiLU-chunk-add.
at::Tensor conv1x1_int16_gemm_wsilu_chunk(at::Tensor input,
                                          at::Tensor weight,
                                          c10::optional<at::Tensor> bias_opt,
                                          at::Tensor lut,
                                          int64_t k2_layer) {
    at::Tensor bias; if(bias_opt.has_value()) bias = bias_opt.value();

    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(lut, "lut");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kShort, "weight");
    check_scalar_type(lut, at::kShort, "lut");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    lut = lut.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions");
    TORCH_CHECK(k2_layer >= 1, "k2_layer must be >= 1");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto full_out_channels = static_cast<int32_t>(weight.size(0));
    TORCH_CHECK(full_out_channels % 2 == 0,
                "weight out_channels must be even for wsilu_chunk_add fusion");
    const auto half_channels = full_out_channels / 2;
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == full_out_channels,
                    "bias must have one value per output channel (2*half_C)");
    }

    auto output = at::empty({batch, half_channels, height, width},
                            at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool has_bias = bias.defined() && bias.numel() > 0;
    const dim3 threads_per_block(256);
    const int32_t M = batch * height * width;

    const dim3 blocks(
        static_cast<uint32_t>((half_channels + 63) / 64),
        static_cast<uint32_t>((M + 63) / 64),
        1);

    conv1x1_int16_blocked_wsilu_chunk_kernel<<<blocks, threads_per_block, 0,
                                               stream.stream()>>>(
        input.data_ptr<int16_t>(),
        weight.data_ptr<int16_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        lut.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        batch,
        in_channels,
        height,
        width,
        half_channels,
        static_cast<int32_t>(k2_layer),
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor conv1x1_int8tc_gemm(at::Tensor input,
                               at::Tensor weight,
                               at::Tensor bias,
                               int64_t weight_int8_scale,
                               int64_t activation_int8_scale,
                               int64_t k2_layer) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kChar, "weight");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions");
    TORCH_CHECK(weight_int8_scale >= 1 && (weight_int8_scale & (weight_int8_scale - 1)) == 0,
                "weight_int8_scale must be a positive power of two");
    TORCH_CHECK(
        activation_int8_scale >= 1 && (activation_int8_scale & (activation_int8_scale - 1)) == 0,
        "activation_int8_scale must be a positive power of two");
    TORCH_CHECK(k2_layer >= 1, "k2_layer must be >= 1");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto out_channels = static_cast<int32_t>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");
    TORCH_CHECK((in_channels % 4) == 0 && (out_channels % 4) == 0,
                "INT8 Tensor Core GEMM requires in/out channels divisible by 4");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == out_channels, "bias must have one value per output channel");
    }

    const auto rows = static_cast<int32_t>(batch * height * width);
    auto input_matrix = input.permute({0, 2, 3, 1}).contiguous().view({rows, in_channels});
    auto input_i8 = at::empty({rows, in_channels},
                              at::TensorOptions().device(input.device()).dtype(at::kChar));
    auto output_i32 = at::zeros({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kInt));
    auto output_i16 = at::empty({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_cublas_handle(stream.stream());

    const int threads = 256;
    const int32_t act_shift = power_of_two_shift(activation_int8_scale);
    const int64_t cast_total = input_matrix.numel();
    const dim3 cast_blocks(static_cast<uint32_t>(num_blocks_for(cast_total, threads)));
    const dim3 threads_per_block(threads);
    cast_int16_to_int8_kernel<<<cast_blocks, threads_per_block, 0, stream.stream()>>>(
        input_matrix.data_ptr<int16_t>(),
        input_i8.data_ptr<int8_t>(),
        cast_total,
        act_shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int32_t alpha = 1;
    const int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_channels,
        rows,
        in_channels,
        &alpha,
        weight.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        input_i8.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        &beta,
        output_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        out_channels,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx int8 failed with status ", static_cast<int>(status));

    const bool has_bias = bias.defined() && bias.numel() > 0;
    const int32_t scale_multiplier = static_cast<int32_t>(weight_int8_scale * activation_int8_scale);
    const int64_t total = output_i32.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    descale_int32_to_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        output_i32.data_ptr<int32_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        output_i16.data_ptr<int16_t>(),
        rows,
        out_channels,
        scale_multiplier,
        static_cast<int32_t>(k2_layer),
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_i16.view({batch, height, width, out_channels}).permute({0, 3, 1, 2}).contiguous();
}

at::Tensor conv1x1_int8tc_gemm_per_channel(at::Tensor input,
                                           at::Tensor weight,
                                           at::Tensor bias,
                                           at::Tensor scale_c,
                                           int64_t activation_int8_scale) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(scale_c, "scale_c");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kChar, "weight");
    check_scalar_type(scale_c, at::kFloat, "scale_c");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    
    scale_c = scale_c.contiguous();

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions");
    TORCH_CHECK(
        activation_int8_scale >= 1 && (activation_int8_scale & (activation_int8_scale - 1)) == 0,
        "activation_int8_scale must be a positive power of two");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto out_channels = static_cast<int32_t>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");
    TORCH_CHECK(scale_c.numel() == out_channels, "scale_c must have one value per output channel");
    TORCH_CHECK((in_channels % 4) == 0 && (out_channels % 4) == 0,
                "INT8 Tensor Core GEMM requires in/out channels divisible by 4");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == out_channels, "bias must have one value per output channel");
    }

    const auto rows = static_cast<int32_t>(batch * height * width);
    auto input_matrix = input.permute({0, 2, 3, 1}).contiguous().view({rows, in_channels});
    auto input_i8 = at::empty({rows, in_channels},
                              at::TensorOptions().device(input.device()).dtype(at::kChar));
    auto output_i32 = at::zeros({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kInt));
    auto output_i16 = at::empty({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_cublas_handle(stream.stream());

    const int threads = 256;
    const int32_t act_shift = power_of_two_shift(activation_int8_scale);
    const int64_t cast_total = input_matrix.numel();
    const dim3 cast_blocks(static_cast<uint32_t>(num_blocks_for(cast_total, threads)));
    const dim3 threads_per_block(threads);
    cast_int16_to_int8_kernel<<<cast_blocks, threads_per_block, 0, stream.stream()>>>(
        input_matrix.data_ptr<int16_t>(),
        input_i8.data_ptr<int8_t>(),
        cast_total,
        act_shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int32_t alpha = 1;
    const int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_channels,
        rows,
        in_channels,
        &alpha,
        weight.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        input_i8.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        &beta,
        output_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        out_channels,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx int8 failed with status ", static_cast<int>(status));

    const bool has_bias = bias.defined() && bias.numel() > 0;
    const int64_t total = output_i32.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    descale_int32_per_channel_to_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        output_i32.data_ptr<int32_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        scale_c.data_ptr<float>(),
        nullptr,
        output_i16.data_ptr<int16_t>(),
        rows,
        out_channels,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_i16.view({batch, height, width, out_channels}).permute({0, 3, 1, 2}).contiguous();
}

at::Tensor conv1x1_int8tc_gemm_per_channel_v2(at::Tensor input,
                                              at::Tensor weight,
                                              at::Tensor bias,
                                              at::Tensor activation_scale_c,
                                              at::Tensor eff_scale_c) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(activation_scale_c, "activation_scale_c");
    check_cuda_tensor(eff_scale_c, "eff_scale_c");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kChar, "weight");
    check_scalar_type(activation_scale_c, at::kInt, "activation_scale_c");
    check_scalar_type(eff_scale_c, at::kFloat, "eff_scale_c");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    
    activation_scale_c = activation_scale_c.contiguous();
    eff_scale_c = eff_scale_c.contiguous();

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto out_channels = static_cast<int32_t>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");
    TORCH_CHECK(
        activation_scale_c.numel() == in_channels,
        "activation_scale_c must have one value per input channel");
    TORCH_CHECK(
        eff_scale_c.numel() == out_channels,
        "eff_scale_c must have one value per output channel");
    TORCH_CHECK((in_channels % 4) == 0 && (out_channels % 4) == 0,
                "INT8 Tensor Core GEMM requires in/out channels divisible by 4");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == out_channels, "bias must have one value per output channel");
    }

    const auto rows = static_cast<int32_t>(batch * height * width);
    auto input_matrix = input.permute({0, 2, 3, 1}).contiguous().view({rows, in_channels});
    auto input_i8 = at::empty({rows, in_channels},
                              at::TensorOptions().device(input.device()).dtype(at::kChar));
    auto output_i32 = at::zeros({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kInt));
    auto output_i16 = at::empty({rows, out_channels},
                                at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_cublas_handle(stream.stream());

    const int threads = 256;
    const dim3 threads_per_block(threads);
    const dim3 cast_blocks(
        static_cast<uint32_t>(num_blocks_for(rows, threads)),
        static_cast<uint32_t>(in_channels));
    cast_int16_to_int8_per_channel_kernel<<<cast_blocks, threads_per_block, 0, stream.stream()>>>(
        input_matrix.data_ptr<int16_t>(),
        input_i8.data_ptr<int8_t>(),
        activation_scale_c.data_ptr<int32_t>(),
        rows,
        in_channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int32_t alpha = 1;
    const int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_channels,
        rows,
        in_channels,
        &alpha,
        weight.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        input_i8.data_ptr<int8_t>(),
        CUDA_R_8I,
        in_channels,
        &beta,
        output_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        out_channels,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx int8 v2 failed with status ", static_cast<int>(status));

    const bool has_bias = bias.defined() && bias.numel() > 0;
    const int64_t total = output_i32.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    descale_int32_per_channel_to_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        output_i32.data_ptr<int32_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        eff_scale_c.data_ptr<float>(),
        nullptr,
        output_i16.data_ptr<int16_t>(),
        rows,
        out_channels,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_i16.view({batch, height, width, out_channels}).permute({0, 3, 1, 2}).contiguous();
}

at::Tensor depthwise_conv3x3_lut_fused_int16(at::Tensor input,
                                             at::Tensor weight,
                                             at::Tensor bias,
                                             at::Tensor lut,
                                             int64_t stride,
                                             int64_t padding) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_cuda_tensor(lut, "lut");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kShort, "weight");
    check_scalar_type(lut, at::kShort, "lut");
    TORCH_CHECK(lut.numel() == 65536, "lut must contain 65536 int16 entries");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    
    lut = lut.contiguous();

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 4, "weight must have 4 dimensions");
    TORCH_CHECK(weight.size(1) == 1 && weight.size(2) == 3 && weight.size(3) == 3,
                "weight must be [C, 1, 3, 3] for fused depthwise path");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto stride_i = static_cast<int32_t>(stride);
    const auto padding_i = static_cast<int32_t>(padding);
    TORCH_CHECK(weight.size(0) == channels, "depthwise weight channel mismatch");
    const auto out_h = static_cast<int32_t>((height + 2 * padding_i - 3) / stride_i + 1);
    const auto out_w = static_cast<int32_t>((width + 2 * padding_i - 3) / stride_i + 1);
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == channels, "bias must have one value per output channel");
    }

    auto output = at::empty({batch, channels, out_h, out_w},
                            at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool has_bias = bias.defined() && bias.numel() > 0;
    const dim3 threads_per_block(kTileW, kTileH);
    const dim3 blocks(
        static_cast<uint32_t>((out_w + kTileW - 1) / kTileW),
        static_cast<uint32_t>((out_h + kTileH - 1) / kTileH),
        static_cast<uint32_t>(batch * channels));
    depthwise_conv3x3_lut_fused_int16_kernel<<<blocks, threads_per_block, 0,
                                               stream.stream()>>>(
        input.data_ptr<int16_t>(),
        weight.data_ptr<int16_t>(),
        has_bias ? bias.data_ptr<int32_t>() : nullptr,
        lut.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        batch,
        channels,
        height,
        width,
        stride_i,
        padding_i,
        out_h,
        out_w,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor conv2d_int16(at::Tensor input,
                        at::Tensor weight,
                        c10::optional<at::Tensor> bias_opt,
                        c10::optional<at::Tensor> residual_opt,
                        c10::optional<at::Tensor> residual2_opt,
                        c10::optional<at::Tensor> post_scale_opt,
                        int64_t stride,
                        int64_t padding,
                        int64_t groups) {
    at::Tensor bias; if(bias_opt.has_value()) bias = bias_opt.value();
    at::Tensor residual; if(residual_opt.has_value()) residual = residual_opt.value();
    at::Tensor residual2; if(residual2_opt.has_value()) residual2 = residual2_opt.value();
    at::Tensor post_scale; if(post_scale_opt.has_value()) post_scale = post_scale_opt.value();
    check_cuda_tensor(input, "input");
    check_cuda_tensor(weight, "weight");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(weight, at::kShort, "weight");

    if (bias.defined() && bias.numel() > 0) {
        check_cuda_tensor(bias, "bias");
        check_scalar_type(bias, at::kInt, "bias");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.defined() ? bias.contiguous() : bias;
    
    const bool has_residual = residual.defined() && residual.numel() > 0;
    if (has_residual) {
        check_cuda_tensor(residual, "residual");
        check_scalar_type(residual, at::kShort, "residual");
        residual = residual.contiguous();
    }
    const bool has_residual2 = residual2.defined() && residual2.numel() > 0;
    if (has_residual2) {
        check_cuda_tensor(residual2, "residual2");
        check_scalar_type(residual2, at::kShort, "residual2");
        residual2 = residual2.contiguous();
    }
    const bool has_post_scale = post_scale.defined() && post_scale.numel() > 0;
    if (has_post_scale) {
        check_cuda_tensor(post_scale, "post_scale");
        check_scalar_type(post_scale, at::kShort, "post_scale");
        post_scale = post_scale.contiguous();
    }
    

    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(weight.dim() == 4, "weight must have 4 dimensions");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto out_channels = static_cast<int32_t>(weight.size(0));
    const auto channels_per_group = static_cast<int32_t>(weight.size(1));
    const auto kernel_h = static_cast<int32_t>(weight.size(2));
    const auto kernel_w = static_cast<int32_t>(weight.size(3));
    const auto stride_i = static_cast<int32_t>(stride);
    const auto padding_i = static_cast<int32_t>(padding);
    const auto groups_i = static_cast<int32_t>(groups);
    const auto out_h = static_cast<int32_t>((height + 2 * padding_i - kernel_h) / stride_i + 1);
    const auto out_w = static_cast<int32_t>((width + 2 * padding_i - kernel_w) / stride_i + 1);

    TORCH_CHECK(in_channels == channels_per_group * groups_i,
                "input channels do not match grouped weight shape");
    TORCH_CHECK(out_channels % groups_i == 0, "out_channels must be divisible by groups");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.numel() == out_channels, "bias must have one value per output channel");
    }
    if (has_post_scale) {
        TORCH_CHECK(
            post_scale.numel() == 1 || post_scale.numel() == out_channels,
            "post_scale must have one value or one value per output channel");
    }

    auto output = at::empty({batch, out_channels, out_h, out_w},
                            at::TensorOptions().device(input.device()).dtype(at::kShort));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (kernel_h == 1 && kernel_w == 1 && stride_i == 1 && padding_i == 0 && groups_i == 1) {
        return conv1x1_int16_gemm(
            input,
            weight.view({out_channels, channels_per_group}),
            bias,
            residual,
            residual2,
            post_scale,
            kWeightScale);
    } else if (kernel_h == 1 && kernel_w == 1) {
        const int threads = 256;
        const int64_t total = output.numel();
        const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
        const dim3 threads_per_block(threads);
        conv1x1_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
            input.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),
            has_bias ? bias.data_ptr<int32_t>() : nullptr,
            has_residual ? residual.data_ptr<int16_t>() : nullptr,
            has_residual2 ? residual2.data_ptr<int16_t>() : nullptr,
            has_post_scale ? post_scale.data_ptr<int16_t>() : nullptr,
            output.data_ptr<int16_t>(),
            batch,
            in_channels,
            height,
            width,
            out_channels,
            channels_per_group,
            stride_i,
            padding_i,
            groups_i,
            out_h,
            out_w,
            has_post_scale ? post_scale.numel() : 0,
            kWeightScale,
            has_bias);
    } else if (groups_i == in_channels && out_channels == in_channels && channels_per_group == 1) {
        const int threads = 256;
        const int64_t total = output.numel();
        const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
        const dim3 threads_per_block(threads);
        depthwise_conv2d_int16_kernel<<<blocks, threads_per_block, 0,
                                        stream.stream()>>>(
            input.data_ptr<int16_t>(),
            weight.data_ptr<int16_t>(),
            has_bias ? bias.data_ptr<int32_t>() : nullptr,
            has_post_scale ? post_scale.data_ptr<int16_t>() : nullptr,
            output.data_ptr<int16_t>(),
            batch,
            in_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_i,
            padding_i,
            out_h,
            out_w,
            has_post_scale ? post_scale.numel() : 0,
            has_bias);
    } else {
        const dim3 threads_per_block(kTileW, kTileH);
        const dim3 blocks(
            static_cast<uint32_t>((out_w + kTileW - 1) / kTileW),
            static_cast<uint32_t>((out_h + kTileH - 1) / kTileH),
            static_cast<uint32_t>(batch * out_channels));

#define LAUNCH_TILED(BLOCK_C) \
        do { \
            const size_t shared_bytes = \
                static_cast<size_t>(BLOCK_C) * \
                static_cast<size_t>(kTileH * stride_i + kernel_h - 1) * \
                static_cast<size_t>(kTileW * stride_i + kernel_w - 1) * \
                sizeof(int16_t); \
            conv2d_int16_tiled_kernel<BLOCK_C><<<blocks, threads_per_block, shared_bytes, \
                                        stream.stream()>>>( \
                input.data_ptr<int16_t>(), \
                weight.data_ptr<int16_t>(), \
                has_bias ? bias.data_ptr<int32_t>() : nullptr, \
                has_post_scale ? post_scale.data_ptr<int16_t>() : nullptr, \
                output.data_ptr<int16_t>(), \
                batch, \
                in_channels, \
                height, \
                width, \
                out_channels, \
                channels_per_group, \
                kernel_h, \
                kernel_w, \
                stride_i, \
                padding_i, \
                groups_i, \
                out_h, \
                out_w, \
                has_post_scale ? post_scale.numel() : 0, \
                has_bias); \
        } while (0)

        if (channels_per_group == 48 || channels_per_group == 96) {
            LAUNCH_TILED(6);
        } else if (channels_per_group == 64) {
            LAUNCH_TILED(4);
        } else if (channels_per_group % 8 == 0) {
            LAUNCH_TILED(8);
        } else {
            LAUNCH_TILED(4);
        }
#undef LAUNCH_TILED
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor sigmoid_lut_int16(at::Tensor input, at::Tensor lut) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(lut, "lut");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(lut, at::kShort, "lut");
    TORCH_CHECK(lut.numel() == 65536, "lut must contain 65536 int16 entries");

    input = input.contiguous();
    lut = lut.contiguous();
    auto output = at::empty_like(input);

    const int threads = 256;
    const int64_t total = input.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    lut_lookup_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        input.data_ptr<int16_t>(),
        lut.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor scale_index_lut_int16(at::Tensor scales, at::Tensor lut) {
    check_cuda_tensor(scales, "scales");
    check_cuda_tensor(lut, "lut");
    check_scalar_type(scales, at::kShort, "scales");
    check_scalar_type(lut, at::kInt, "lut");
    TORCH_CHECK(lut.numel() == 65536, "lut must contain 65536 int32 entries");

    scales = scales.contiguous();
    lut = lut.contiguous();
    auto output = at::empty(scales.sizes(),
                            at::TensorOptions().device(scales.device()).dtype(at::kInt));

    const int threads = 256;
    const int64_t total = scales.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(scales.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    scale_index_lut_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        scales.data_ptr<int16_t>(),
        lut.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor clamp_reciprocal_int16(at::Tensor q, int64_t k1) {
    check_cuda_tensor(q, "q");
    check_scalar_type(q, at::kShort, "q");
    q = q.contiguous();
    auto output = at::empty_like(q);

    const int threads = 256;
    const int64_t total = q.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(q.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    clamp_reciprocal_int16_kernel<<<blocks, threads_per_block, 0,
                                    stream.stream()>>>(
        q.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        static_cast<int32_t>(k1),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor add_multiply_int16(at::Tensor a,
                              at::Tensor b,
                              at::Tensor scale,
                              int64_t k1) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    check_cuda_tensor(scale, "scale");
    check_scalar_type(a, at::kShort, "a");
    check_scalar_type(b, at::kShort, "b");
    check_scalar_type(scale, at::kShort, "scale");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == scale.sizes(),
                "a, b, and scale must have the same shape");

    a = a.contiguous();
    b = b.contiguous();
    scale = scale.contiguous();
    auto output = at::empty_like(a);

    const int threads = 256;
    const int64_t total = a.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(a.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    add_multiply_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        a.data_ptr<int16_t>(),
        b.data_ptr<int16_t>(),
        scale.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        static_cast<int32_t>(k1),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor multiply_int16(at::Tensor input, at::Tensor scale, int64_t k1) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(scale, "scale");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(scale, at::kShort, "scale");
    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions");
    TORCH_CHECK(k1 == 512, "multiply_int16 currently expects feature scale 512");

    input = input.contiguous();
    scale = scale.contiguous();
    const auto channels = static_cast<int32_t>(input.size(1));
    TORCH_CHECK(
        scale.numel() == 1 || scale.numel() == channels || scale.numel() == input.numel(),
        "scale must have one value, one value per channel, or one value per input element");
    auto output = at::empty_like(input);

    const int threads = 256;
    const int64_t total = input.numel();
    const int32_t spatial_size = static_cast<int32_t>(input.size(2) * input.size(3));
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    multiply_int16_broadcast_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        input.data_ptr<int16_t>(),
        scale.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        channels,
        spatial_size,
        scale.numel(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void wsilu_chunk_add_int16_kernel(const int16_t* __restrict__ input,
                                             const int16_t* __restrict__ lut,
                                             int16_t* __restrict__ output,
                                             int32_t batch,
                                             int32_t half_channels,
                                             int32_t spatial_size) {
    const int64_t total = static_cast<int64_t>(batch) * half_channels * spatial_size;
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }
    
    int64_t tmp = linear_idx;
    const int32_t s = tmp % spatial_size;
    tmp /= spatial_size;
    const int32_t c = tmp % half_channels;
    const int32_t n = tmp / half_channels;
    
    // Original channels = half_channels * 2
    const int32_t in_channels = half_channels * 2;
    const int64_t idx0 = (static_cast<int64_t>(n) * in_channels + c) * spatial_size + s;
    const int64_t idx1 = idx0 + (static_cast<int64_t>(half_channels) * spatial_size);
    
    const int32_t in0 = static_cast<int32_t>(input[idx0]) - kInt16Min;
    const int32_t in1 = static_cast<int32_t>(input[idx1]) - kInt16Min;
    
    int64_t val = static_cast<int64_t>(lut[in0]) + static_cast<int64_t>(lut[in1]);
    
    if (val < kInt16Min) val = kInt16Min;
    else if (val > kInt16Max) val = kInt16Max;
    
    output[linear_idx] = static_cast<int16_t>(val);
}

__global__ void add_int16_kernel(const int16_t* __restrict__ a,
                                 const int16_t* __restrict__ b,
                                 int16_t* __restrict__ output,
                                 int64_t total) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }
    int64_t val = static_cast<int64_t>(a[linear_idx]) + static_cast<int64_t>(b[linear_idx]);
    if (val < kInt16Min) val = kInt16Min;
    else if (val > kInt16Max) val = kInt16Max;
    output[linear_idx] = static_cast<int16_t>(val);
}

at::Tensor wsilu_chunk_add_int16(at::Tensor input, at::Tensor lut) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(lut, "lut");
    check_scalar_type(input, at::kShort, "input");
    check_scalar_type(lut, at::kShort, "lut");
    
    input = input.contiguous();
    lut = lut.contiguous();
    
    const auto batch = static_cast<int32_t>(input.size(0));
    const auto in_channels = static_cast<int32_t>(input.size(1));
    const auto height = static_cast<int32_t>(input.size(2));
    const auto width = static_cast<int32_t>(input.size(3));
    const auto half_channels = in_channels / 2;
    const auto spatial_size = height * width;
    
    auto output = at::empty({batch, half_channels, height, width},
                            at::TensorOptions().device(input.device()).dtype(at::kShort));
                            
    const int threads = 256;
    const int64_t total = output.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    
    wsilu_chunk_add_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        input.data_ptr<int16_t>(),
        lut.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        batch,
        half_channels,
        spatial_size);
        
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor add_int16(at::Tensor a, at::Tensor b) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    check_scalar_type(a, at::kShort, "a");
    check_scalar_type(b, at::kShort, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    
    a = a.contiguous();
    b = b.contiguous();
    auto output = at::empty_like(a);
    
    const int threads = 256;
    const int64_t total = a.numel();
    const dim3 blocks(static_cast<uint32_t>(num_blocks_for(total, threads)));
    const dim3 threads_per_block(threads);

    const c10::cuda::CUDAGuard guard(a.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    
    add_int16_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        a.data_ptr<int16_t>(),
        b.data_ptr<int16_t>(),
        output.data_ptr<int16_t>(),
        total);
        
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
