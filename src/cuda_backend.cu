#include "uniad.h"
#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t timestamp_us, scene_hash;
    int32_t navigation_command;
    uint32_t reserved;
    float camera_intrinsics[UA_CAMERA_COUNT][9];
    float camera_to_ego[UA_CAMERA_COUNT][16];
    float ego_pose[16];
    float can_bus[18];
} production_metadata;

static_assert(sizeof(production_metadata) == UA_PRODUCTION_METADATA_BYTES,
              "production metadata ABI");

typedef struct {
    void *weights;
    void *camera_raw;
    void *metadata;
    __half *boundary_samples;
    int *stage_status;
    void *previous_bev;
    void *aligned_previous_bev;
    void *track_state_committed;
    void *track_state_candidate;
    void *arena;
    cudaStream_t stream;
    size_t weights_bytes;
    char scene_token[129];
    int has_scene;
    int previous_bev_valid;
    float temporal_shift_x;
    float temporal_shift_y;
    const __half *last_resnet_stem;
    size_t last_resnet_stem_count;
    const __half *last_resnet_layer1;
    size_t last_resnet_layer1_count;
    const __half *last_resnet_layer2;
    size_t last_resnet_layer2_count;
    const __half *last_resnet_layer3;
    size_t last_resnet_layer3_count;
    int layer3_first_nonfinite;
    const __half *last_resnet_layer4;
    size_t last_resnet_layer4_count;
    int layer4_first_nonfinite;
    const __half *last_fpn[4];
    size_t last_fpn_count[4];
    const __half *last_bevformer_flatten;
    size_t last_bevformer_flatten_count;
    const __half *last_bev_queries;
    const __half *last_bev_pos;
    const __half *last_reference_2d;
    const __half *last_reference_3d;
    const float *last_reference_camera;
    const uint8_t *last_visibility;
    const uint32_t *last_visible_indices;
    const uint32_t *last_visible_counts;
    const __half *last_temporal_attention0;
    const __half *last_encoder_norm0;
    const __half *last_spatial_attention0;
    const __half *last_encoder_norm1;
    const __half *last_encoder_ffn0;
    const __half *last_encoder_norm2;
    size_t encoder_layers_completed;
    const __half *last_track_query_pos;
    const __half *last_track_query;
    const float *initial_track_reference_points;
    const float *last_track_reference_points;
    const __half *last_track_decoder_self;
    const __half *last_track_decoder_norm0;
    const __half *last_track_decoder_cross;
    const __half *last_track_decoder_norm1;
    const __half *last_track_decoder_ffn;
    const __half *last_track_decoder_norm2;
    const __half *track_decoder_states;
    const __half *track_regressions;
    const float *track_references;
    const __half *last_track_class_logits;
    const float *last_track_boxes;
    const __half *last_track_past_trajectory;
    const float *last_track_scores;
    const uint32_t *last_track_classes;
    const uint32_t *last_track_selected_indices;
    const uint32_t *last_track_selected_count;
    ua_cuda_query_interaction_weights query_interaction_weights;
    int query_interaction_weights_valid;
    const __half *last_query_interaction_output;
    size_t last_query_interaction_queries;
    size_t track_decoder_layers_completed;
    size_t track_reference_layers_completed;
    int track_output_heads_completed;
} production_context;

#define PROD_PREVIOUS_BEV_BYTES (200u * 200u * 256u * sizeof(__half))
#define PROD_TRACK_MAX_QUERIES (901u + UA_PROD_MAX_TRACKS)
#define PROD_TRACK_QUERY_OFFSET 0u
#define PROD_TRACK_OUTPUT_OFFSET 1230000u
#define PROD_TRACK_REFERENCE_STATE_OFFSET 1845000u
#define PROD_TRACK_BOX_STATE_OFFSET 1860000u
#define PROD_TRACK_SCORE_STATE_OFFSET 1909000u
#define PROD_TRACK_ID_OFFSET 1914000u
#define PROD_TRACK_DISAPPEAR_OFFSET 1919000u
#define PROD_TRACK_MEMORY_OFFSET 1925000u
#define PROD_TRACK_MASK_OFFSET 4385000u
#define PROD_TRACK_SAVE_PERIOD_OFFSET 4390000u
#define PROD_TRACK_COUNT_OFFSET 4391204u
#define PROD_TRACK_NEXT_ID_OFFSET 4391208u
#define PROD_TRACK_STATE_BYTES 4392000u
#define PROD_ARENA_BYTES (512u * 1024u * 1024u)
#define PROD_IMAGE_WIDTH 1600u
#define PROD_IMAGE_HEIGHT 900u
#define PROD_PAD_HEIGHT 928u
#define PROD_CAMERA_RAW_BYTES \
    (UA_CAMERA_COUNT * PROD_IMAGE_WIDTH * PROD_IMAGE_HEIGHT * 3u)
#define PROD_NORMALIZED_BYTES \
    (UA_CAMERA_COUNT * 3u * PROD_PAD_HEIGHT * PROD_IMAGE_WIDTH * sizeof(__half))
#define PROD_STEM_HEIGHT 464u
#define PROD_STEM_WIDTH 800u
#define PROD_STEM_CHANNELS 64u
#define PROD_STEM_BYTES \
    (UA_CAMERA_COUNT * PROD_STEM_CHANNELS * PROD_STEM_HEIGHT * \
     PROD_STEM_WIDTH * sizeof(__half))
#define PROD_POOL_HEIGHT 232u
#define PROD_POOL_WIDTH 400u
#define PROD_POOL_BYTES \
    (UA_CAMERA_COUNT * PROD_STEM_CHANNELS * PROD_POOL_HEIGHT * \
     PROD_POOL_WIDTH * sizeof(__half))
#define PROD_BACKBONE_FPN_BOUNDARIES 16u
#define PROD_ENCODER_LAYERS 6u
#define PROD_ENCODER_BOUNDARIES_PER_LAYER 6u
#define PROD_TRACK_DECODER_LAYERS 6u
#define PROD_TRACK_BOUNDARIES_PER_LAYER 6u
#define PROD_TRACK_QUERY_COUNT (901u * 256u)
#define PROD_TRACK_STATE_OFFSET 1048576u
#define PROD_TRACK_REGRESSION_OFFSET 3817472u
#define PROD_TRACK_REFERENCE_OFFSET 3932160u
#define PROD_TRACK_CLASS_OFFSET 4000000u
#define PROD_TRACK_PAST_OFFSET 4020000u
#define PROD_TRACK_BOX_OFFSET 4050000u
#define PROD_TRACK_SCORE_OFFSET 4087000u
#define PROD_TRACK_CLASS_INDEX_OFFSET 4091000u
#define PROD_TRACK_SELECTED_OFFSET 4095000u
#define PROD_TRACK_SELECTED_COUNT_OFFSET 4096200u
#define PROD_BOUNDARY_SAMPLE_VALUES \
    ((PROD_BACKBONE_FPN_BOUNDARIES + \
      PROD_ENCODER_LAYERS * PROD_ENCODER_BOUNDARIES_PER_LAYER + \
      PROD_TRACK_DECODER_LAYERS * PROD_TRACK_BOUNDARIES_PER_LAYER) * 32u)
#define PROD_STAGE_STATUS_BYTES (sizeof(int))

static ua_status snapshot_encoder_boundary(
        production_context *c, size_t layer, size_t stage,
        const __half *source) {
    size_t window;
    if (!c || !source || layer >= PROD_ENCODER_LAYERS ||
        stage >= PROD_ENCODER_BOUNDARIES_PER_LAYER)
        return UA_ERR_ARGUMENT;
    window = PROD_BACKBONE_FPN_BOUNDARIES +
             layer * PROD_ENCODER_BOUNDARIES_PER_LAYER + stage;
    return cudaMemcpyAsync(
        c->boundary_samples + window * 32u, source,
        32u * sizeof(__half), cudaMemcpyDeviceToDevice, c->stream) ==
        cudaSuccess ? UA_OK : UA_ERR_BACKEND;
}

static ua_status snapshot_track_boundary(
        production_context *c, size_t layer, size_t stage,
        const __half *source) {
    size_t window;
    if (!c || !source || layer >= PROD_TRACK_DECODER_LAYERS ||
        stage >= PROD_TRACK_BOUNDARIES_PER_LAYER)
        return UA_ERR_ARGUMENT;
    window = PROD_BACKBONE_FPN_BOUNDARIES +
             PROD_ENCODER_LAYERS * PROD_ENCODER_BOUNDARIES_PER_LAYER +
             layer * PROD_TRACK_BOUNDARIES_PER_LAYER + stage;
    return cudaMemcpyAsync(
        c->boundary_samples + window * 32u, source,
        32u * sizeof(__half), cudaMemcpyDeviceToDevice, c->stream) ==
        cudaSuccess ? UA_OK : UA_ERR_BACKEND;
}

__global__ static void boundary_kernel(const float *x, size_t n, float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = 0.0f;
        for (size_t i = 0; i < n; ++i) v += x[i];
        *out = v;
    }
}

__global__ static void float_to_half_kernel(
        const float *source, __half *target, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) target[index] = __float2half_rn(source[index]);
}

__global__ static void half_to_float_kernel(
        const __half *source, float *target, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) target[index] = __half2float(source[index]);
}

__global__ static void record_nonfinite_half_kernel(
        const __half *values, size_t count, int block, int *first_block) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(__half2float(values[index])))
        atomicMin(first_block, block);
}

__global__ static void preprocess_bgr_kernel(
        const uint8_t *source, size_t width, size_t height, size_t row_stride,
        size_t padded_width, size_t padded_height, float3 mean, float3 inverse_std,
        __half *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = padded_width * padded_height;
    size_t count = plane * 3u;
    if (index >= count) return;
    size_t channel = index / plane;
    size_t pixel = index - channel * plane;
    size_t y = pixel / padded_width;
    size_t x = pixel - y * padded_width;
    float value = 0.0f;
    if (x < width && y < height) {
        float raw = (float)source[y * row_stride + x * 3u + channel];
        const float *mean_values = &mean.x;
        const float *inverse_values = &inverse_std.x;
        value = (raw - mean_values[channel]) * inverse_values[channel];
    }
    output[index] = __float2half_rn(value);
}

__global__ static void linear_fp16_kernel(
        const __half *x, const __half *weight, const __half *bias, __half *y,
        size_t rows, size_t in_dim, size_t out_dim, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = rows * out_dim;
    if (index >= count) return;
    size_t row = index / out_dim;
    size_t output = index - row * out_dim;
    float accumulator = has_bias ? __half2float(bias[output]) : 0.0f;
    for (size_t input = 0; input < in_dim; ++input) {
        accumulator += __half2float(x[row * in_dim + input]) *
                       __half2float(weight[output * in_dim + input]);
    }
    y[index] = __float2half_rn(accumulator);
}

__global__ static void conv2d_fp16_kernel(
        const __half *x, const __half *weight, const __half *bias, __half *y,
        size_t batches, size_t input_channels, size_t input_height,
        size_t input_width, size_t output_channels, size_t output_height,
        size_t output_width, size_t kernel_height, size_t kernel_width,
        size_t stride_height, size_t stride_width, size_t padding_height,
        size_t padding_width, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_count =
        batches * output_channels * output_height * output_width;
    if (index >= output_count) return;
    size_t output_x = index % output_width;
    size_t quotient = index / output_width;
    size_t output_y = quotient % output_height;
    quotient /= output_height;
    size_t output_channel = quotient % output_channels;
    size_t batch = quotient / output_channels;
    float accumulator = has_bias ? __half2float(bias[output_channel]) : 0.0f;
    for (size_t input_channel = 0; input_channel < input_channels;
         ++input_channel) {
        for (size_t ky = 0; ky < kernel_height; ++ky) {
            long input_y = (long)(output_y * stride_height + ky) -
                           (long)padding_height;
            if (input_y < 0 || (size_t)input_y >= input_height) continue;
            for (size_t kx = 0; kx < kernel_width; ++kx) {
                long input_x = (long)(output_x * stride_width + kx) -
                               (long)padding_width;
                if (input_x < 0 || (size_t)input_x >= input_width) continue;
                size_t input_index =
                    ((batch * input_channels + input_channel) * input_height +
                     (size_t)input_y) * input_width + (size_t)input_x;
                size_t weight_index =
                    ((output_channel * input_channels + input_channel) *
                     kernel_height + ky) * kernel_width + kx;
                accumulator += __half2float(x[input_index]) *
                               __half2float(weight[weight_index]);
            }
        }
    }
    y[index] = __float2half_rn(accumulator);
}

__device__ static float bilinear_nchw_half(
        const __half *x, size_t batch, size_t channel, size_t channels,
        size_t height, size_t width, float y, float x_coordinate) {
    long y0 = (long)floorf(y), x0 = (long)floorf(x_coordinate);
    float fy = y - (float)y0, fx = x_coordinate - (float)x0, value = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        long yy = y0 + (corner >= 2), xx = x0 + (corner & 1);
        if (yy >= 0 && xx >= 0 && (size_t)yy < height && (size_t)xx < width) {
            float wy = corner >= 2 ? fy : 1.0f - fy;
            float wx = corner & 1 ? fx : 1.0f - fx;
            size_t index = ((batch * channels + channel) * height +
                            (size_t)yy) * width + (size_t)xx;
            value += __half2float(x[index]) * wy * wx;
        }
    }
    return value;
}

__global__ static void modulated_deform_conv2d_fp16_kernel(
        const __half *x, const float *offset, const float *mask,
        const __half *weight, const __half *bias, __half *y,
        size_t batches, size_t input_channels, size_t input_height,
        size_t input_width, size_t output_channels, size_t output_height,
        size_t output_width, size_t kernel_height, size_t kernel_width,
        size_t stride_height, size_t stride_width, size_t padding_height,
        size_t padding_width, size_t dilation_height, size_t dilation_width,
        int has_bias, int packed_offset_mask) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = batches * output_channels * output_height * output_width;
    if (index >= count) return;
    size_t ox = index % output_width, quotient = index / output_width;
    size_t oy = quotient % output_height;
    quotient /= output_height;
    size_t output_channel = quotient % output_channels;
    size_t batch = quotient / output_channels;
    size_t kernel_elements = kernel_height * kernel_width;
    float accumulator = has_bias ? __half2float(bias[output_channel]) : 0.0f;
    for (size_t input_channel = 0; input_channel < input_channels;
         ++input_channel) {
        for (size_t ky = 0; ky < kernel_height; ++ky)
            for (size_t kx = 0; kx < kernel_width; ++kx) {
                size_t kernel_index = ky * kernel_width + kx;
                size_t offset_y_index;
                size_t offset_x_index;
                size_t mask_index;
                if (packed_offset_mask) {
                    offset_y_index =
                        ((batch * 3u * kernel_elements + 2u * kernel_index) *
                         output_height + oy) * output_width + ox;
                    offset_x_index =
                        offset_y_index + output_height * output_width;
                    mask_index =
                        ((batch * 3u * kernel_elements +
                          2u * kernel_elements + kernel_index) *
                         output_height + oy) * output_width + ox;
                } else {
                    offset_y_index =
                        ((batch * 2u * kernel_elements + 2u * kernel_index) *
                         output_height + oy) * output_width + ox;
                    offset_x_index =
                        offset_y_index + output_height * output_width;
                    mask_index =
                        ((batch * kernel_elements + kernel_index) *
                         output_height + oy) * output_width + ox;
                }
                float sample_y =
                    (float)(oy * stride_height + ky * dilation_height) -
                    (float)padding_height + offset[offset_y_index];
                float sample_x =
                    (float)(ox * stride_width + kx * dilation_width) -
                    (float)padding_width + offset[offset_x_index];
                float sampled = bilinear_nchw_half(
                    x, batch, input_channel, input_channels, input_height,
                    input_width, sample_y, sample_x);
                size_t weight_index =
                    ((output_channel * input_channels + input_channel) *
                     kernel_height + ky) * kernel_width + kx;
                const float *mask_values =
                    packed_offset_mask ? offset : mask;
                accumulator += sampled * mask_values[mask_index] *
                               __half2float(weight[weight_index]);
            }
    }
    y[index] = __float2half_rn(accumulator);
}

__global__ static void dcn_offset_mask_fp32_kernel(
        const __half *x, const __half *weight, const __half *bias,
        float *offset_and_mask, size_t batches, size_t input_channels,
        size_t input_height, size_t input_width, size_t output_height,
        size_t output_width, size_t stride) {
    const size_t output_channels = 27u;
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count =
        batches * output_channels * output_height * output_width;
    if (index >= count) return;
    size_t ox = index % output_width, quotient = index / output_width;
    size_t oy = quotient % output_height;
    quotient /= output_height;
    size_t output_channel = quotient % output_channels;
    size_t batch = quotient / output_channels;
    float accumulator = __half2float(bias[output_channel]);
    for (size_t input_channel = 0; input_channel < input_channels;
         ++input_channel)
        for (size_t ky = 0; ky < 3u; ++ky) {
            long iy = (long)(oy * stride + ky) - 1l;
            if (iy < 0 || (size_t)iy >= input_height) continue;
            for (size_t kx = 0; kx < 3u; ++kx) {
                long ix = (long)(ox * stride + kx) - 1l;
                if (ix < 0 || (size_t)ix >= input_width) continue;
                size_t input_index =
                    ((batch * input_channels + input_channel) *
                         input_height +
                     (size_t)iy) *
                        input_width +
                    (size_t)ix;
                size_t weight_index =
                    ((output_channel * input_channels + input_channel) * 3u +
                     ky) *
                        3u +
                    kx;
                accumulator += __half2float(x[input_index]) *
                               __half2float(weight[weight_index]);
            }
        }
    if (output_channel >= 18u)
        accumulator = 1.0f / (1.0f + expf(-accumulator));
    offset_and_mask[index] = accumulator;
}

__global__ static void nearest_upsample_add_fp16_kernel(
        const __half *low_resolution, __half *high_resolution,
        size_t batches, size_t channels, size_t low_height, size_t low_width,
        size_t high_height, size_t high_width) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = batches * channels * high_height * high_width;
    if (index >= count) return;
    size_t x = index % high_width, quotient = index / high_width;
    size_t y = quotient % high_height;
    quotient /= high_height;
    size_t channel = quotient % channels;
    size_t batch = quotient / channels;
    size_t low_y = y * low_height / high_height;
    size_t low_x = x * low_width / high_width;
    size_t low_index =
        ((batch * channels + channel) * low_height + low_y) * low_width +
        low_x;
    high_resolution[index] = __float2half_rn(
        __half2float(high_resolution[index]) +
        __half2float(low_resolution[low_index]));
}

__global__ static void bevformer_flatten_embed_kernel(
        const __half *level0, const __half *level1, const __half *level2,
        const __half *level3, const float *camera_embeds,
        const float *level_embeds, __half *output, size_t count) {
    const size_t channels = 256u;
    const size_t level_starts[4] = {0u, 23200u, 29000u, 30450u};
    const size_t heights[4] = {116u, 58u, 29u, 15u};
    const size_t widths[4] = {200u, 100u, 50u, 25u};
    const __half *inputs[4] = {level0, level1, level2, level3};
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t channel = index % channels;
    size_t token = (index / channels) % 30825u;
    size_t camera = index / (channels * 30825u);
    size_t level = token >= level_starts[3] ? 3u :
                   token >= level_starts[2] ? 2u :
                   token >= level_starts[1] ? 1u : 0u;
    size_t local = token - level_starts[level];
    size_t input_index =
        (camera * channels + channel) * heights[level] * widths[level] +
        local;
    float value = __half2float(inputs[level][input_index]) +
                  camera_embeds[camera * channels + channel] +
                  level_embeds[level * channels + channel];
    output[index] = __float2half_rn(value);
}

__global__ static void can_bus_mlp_layernorm_kernel(
        const production_metadata *metadata, const float *weight0,
        const float *bias0, const float *weight1, const float *bias1,
        const float *norm_weight, const float *norm_bias, float *output) {
    if (blockIdx.x || threadIdx.x) return;
    float hidden[128];
    float raw[256];
    for (size_t row = 0; row < 128u; ++row) {
        float value = bias0[row];
        for (size_t column = 0; column < 18u; ++column)
            value += metadata->can_bus[column] *
                     weight0[row * 18u + column];
        hidden[row] = fmaxf(value, 0.0f);
    }
    float mean = 0.0f;
    for (size_t row = 0; row < 256u; ++row) {
        float value = bias1[row];
        for (size_t column = 0; column < 128u; ++column)
            value += hidden[column] * weight1[row * 128u + column];
        raw[row] = fmaxf(value, 0.0f);
        mean += raw[row];
    }
    mean /= 256.0f;
    float variance = 0.0f;
    for (size_t row = 0; row < 256u; ++row) {
        float delta = raw[row] - mean;
        variance += delta * delta;
    }
    float inverse = rsqrtf(variance / 256.0f + 1e-5f);
    for (size_t row = 0; row < 256u; ++row)
        output[row] =
            (raw[row] - mean) * inverse * norm_weight[row] + norm_bias[row];
}

__global__ static void bev_query_can_bus_kernel(
        const __half *query_weight, const float *can_bus, __half *output,
        size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = __float2half_rn(
            __half2float(query_weight[index]) + can_bus[index % 256u]);
}

__global__ static void bev_learned_position_kernel(
        const __half *column_embed, const __half *row_embed, __half *output,
        size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t channel = index % 256u;
    size_t token = index / 256u;
    size_t y = token / 200u;
    size_t x = token - y * 200u;
    output[index] = channel < 128u
        ? column_embed[x * 128u + channel]
        : row_embed[y * 128u + channel - 128u];
}

__global__ static void bev_reference_points_kernel(
        __half *reference_2d, __half *reference_3d) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 40000u * 2u) {
        size_t coordinate = index % 2u;
        size_t query = index / 2u;
        size_t y = query / 200u;
        size_t x = query - y * 200u;
        float value = coordinate == 0u
            ? ((float)x + 0.5f) / 200.0f
            : ((float)y + 0.5f) / 200.0f;
        reference_2d[index] = __float2half_rn(value);
    }
    if (index < 4u * 40000u * 3u) {
        size_t coordinate = index % 3u;
        size_t quotient = index / 3u;
        size_t query = quotient % 40000u;
        size_t depth = quotient / 40000u;
        size_t y = query / 200u;
        size_t x = query - y * 200u;
        float value;
        if (coordinate == 0u)
            value = ((float)x + 0.5f) / 200.0f;
        else if (coordinate == 1u)
            value = ((float)y + 0.5f) / 200.0f;
        else
            value = (0.5f + (float)depth * (7.0f / 3.0f)) / 8.0f;
        reference_3d[index] = __float2half_rn(value);
    }
}

__global__ static void bev_camera_projection_kernel(
        const __half *reference_3d, const production_metadata *metadata,
        float *reference_camera, uint8_t *visibility, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t depth = index % 4u;
    size_t query = (index / 4u) % 40000u;
    size_t camera = index / (4u * 40000u);
    const __half *reference =
        reference_3d + (depth * 40000u + query) * 3u;
    float ego_x = __half2float(reference[0]) * 102.4f - 51.2f;
    float ego_y = __half2float(reference[1]) * 102.4f - 51.2f;
    float ego_z = __half2float(reference[2]) * 8.0f - 5.0f;
    const float *transform = metadata->camera_to_ego[camera];
    float dx = ego_x - transform[3];
    float dy = ego_y - transform[7];
    float dz = ego_z - transform[11];
    float camera_x =
        transform[0] * dx + transform[4] * dy + transform[8] * dz;
    float camera_y =
        transform[1] * dx + transform[5] * dy + transform[9] * dz;
    float camera_z =
        transform[2] * dx + transform[6] * dy + transform[10] * dz;
    const float *intrinsic = metadata->camera_intrinsics[camera];
    float projected_x = intrinsic[0] * camera_x +
                        intrinsic[1] * camera_y +
                        intrinsic[2] * camera_z;
    float projected_y = intrinsic[3] * camera_x +
                        intrinsic[4] * camera_y +
                        intrinsic[5] * camera_z;
    float projected_z = intrinsic[6] * camera_x +
                        intrinsic[7] * camera_y +
                        intrinsic[8] * camera_z;
    float denominator = fmaxf(projected_z, 1e-5f);
    float normalized_x = projected_x / denominator / 1600.0f;
    float normalized_y = projected_y / denominator / 900.0f;
    reference_camera[index * 2u] = normalized_x;
    reference_camera[index * 2u + 1u] = normalized_y;
    visibility[index] =
        projected_z > 1e-5f && isfinite(normalized_x) &&
        isfinite(normalized_y) && normalized_x > 0.0f &&
        normalized_x < 1.0f && normalized_y > 0.0f &&
        normalized_y < 1.0f;
}

__global__ static void compact_visible_queries_kernel(
        const uint8_t *visibility, uint32_t *indices, uint32_t *counts) {
    size_t camera = threadIdx.x;
    if (blockIdx.x || camera >= UA_CAMERA_COUNT) return;
    uint32_t count = 0;
    for (uint32_t query = 0; query < 40000u; ++query) {
        size_t base = (camera * 40000u + query) * 4u;
        if (visibility[base] || visibility[base + 1u] ||
            visibility[base + 2u] || visibility[base + 3u])
            indices[camera * 40000u + count++] = query;
    }
    counts[camera] = count;
}

__global__ static void temporal_concat_linear_kernel(
        const __half *history, const __half *query, const __half *position,
        const __half *weight, const __half *bias, __half *output,
        size_t output_dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = 40000u * output_dimensions;
    if (index >= count) return;
    size_t row = index / output_dimensions;
    size_t output_dimension = index % output_dimensions;
    float accumulator = __half2float(bias[output_dimension]);
    const __half *weight_row = weight + output_dimension * 512u;
    for (size_t column = 0; column < 256u; ++column) {
        float query_value = __half2float(query[row * 256u + column]);
        accumulator +=
            __half2float(history[row * 256u + column]) *
            __half2float(weight_row[column]);
        accumulator +=
            (query_value + __half2float(position[row * 256u + column])) *
            __half2float(weight_row[256u + column]);
    }
    output[index] = __float2half_rn(accumulator);
}

__global__ static void temporal_attention_softmax_kernel(__half *weights) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= 40000u * 8u * 2u) return;
    __half *values = weights + group * 4u;
    float maximum = -INFINITY;
    for (size_t point = 0; point < 4u; ++point)
        maximum = fmaxf(maximum, __half2float(values[point]));
    float sum = 0.0f;
    float exponentials[4];
    for (size_t point = 0; point < 4u; ++point) {
        exponentials[point] = expf(__half2float(values[point]) - maximum);
        sum += exponentials[point];
    }
    for (size_t point = 0; point < 4u; ++point)
        values[point] = __float2half_rn(exponentials[point] / sum);
}

__device__ static float temporal_bilinear_value(
        const __half *value, size_t channel, float normalized_x,
        float normalized_y) {
    float x = normalized_x * 200.0f - 0.5f;
    float y = normalized_y * 200.0f - 0.5f;
    long x0 = (long)floorf(x), y0 = (long)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0;
    float sampled = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        long yy = y0 + (corner >= 2);
        long xx = x0 + (corner & 1);
        if (yy >= 0 && xx >= 0 && yy < 200 && xx < 200) {
            float wy = corner >= 2 ? fy : 1.0f - fy;
            float wx = corner & 1 ? fx : 1.0f - fx;
            size_t token = (size_t)yy * 200u + (size_t)xx;
            sampled += __half2float(value[token * 256u + channel]) *
                       wy * wx;
        }
    }
    return sampled;
}

__global__ static void temporal_deform_sample_kernel(
        const __half *value, const __half *offsets,
        const __half *attention, const __half *reference_2d,
        float history_shift_x, float history_shift_y,
        __half *sampled_output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = 40000u * 256u;
    if (index >= count) return;
    size_t query = index / 256u;
    size_t channel = index % 256u;
    size_t head = channel / 32u;
    float output = 0.0f;
    for (size_t queue = 0; queue < 2u; ++queue) {
        float queue_output = 0.0f;
        for (size_t point = 0; point < 4u; ++point) {
            size_t sample = ((query * 8u + head) * 2u + queue) * 4u +
                            point;
            float location_x =
                __half2float(reference_2d[query * 2u]) +
                (queue == 0u ? history_shift_x : 0.0f) +
                __half2float(offsets[sample * 2u]) / 200.0f;
            float location_y =
                __half2float(reference_2d[query * 2u + 1u]) +
                (queue == 0u ? history_shift_y : 0.0f) +
                __half2float(offsets[sample * 2u + 1u]) / 200.0f;
            queue_output += temporal_bilinear_value(
                                value + queue * 40000u * 256u, channel,
                                location_x, location_y) *
                            __half2float(attention[sample]);
        }
        output += queue_output * 0.5f;
    }
    sampled_output[index] = __float2half_rn(output);
}

__global__ static void align_previous_bev_nearest_kernel(
        const __half *input, __half *output, float angle_degrees) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 40000u * 256u) return;
    size_t token = index / 256u, channel = index % 256u;
    size_t y = token / 200u, x = token % 200u;
    float radians = angle_degrees * 0.01745329251994329577f;
    float cosine = cosf(radians), sine = sinf(radians);
    float centered_x = (float)x - 99.5f;
    float centered_y = (float)y - 99.5f;
    float source_x = cosine * centered_x + sine * centered_y + 99.5f;
    float source_y = -sine * centered_x + cosine * centered_y + 99.5f;
    long nearest_x = (long)floorf(source_x + .5f);
    long nearest_y = (long)floorf(source_y + .5f);
    if (nearest_x >= 0 && nearest_x < 200 &&
        nearest_y >= 0 && nearest_y < 200)
        output[index] = input[
            ((size_t)nearest_y * 200u + (size_t)nearest_x) * 256u +
            channel];
    else
        output[index] = __float2half_rn(0.0f);
}

__global__ static void split_track_query_kernel(
        const __half *embedding, __half *query_pos, __half *query) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * 256u) return;
    size_t row = index / 256u, column = index % 256u;
    query_pos[index] = embedding[row * 512u + column];
    query[index] = embedding[row * 512u + 256u + column];
}

__global__ static void track_reference_points_kernel(
        const __half *query_pos, const float *weight, const float *bias,
        float *reference_points) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * 3u) return;
    size_t row = index / 3u, output = index % 3u;
    float accumulator = bias[output];
    for (size_t column = 0; column < 256u; ++column)
        accumulator += __half2float(query_pos[row * 256u + column]) *
                       weight[output * 256u + column];
    reference_points[index] = 1.0f / (1.0f + expf(-accumulator));
}

__device__ static float inverse_sigmoid_clamped(float value) {
    value = fminf(fmaxf(value, 1e-5f), 1.0f - 1e-5f);
    return logf(value / (1.0f - value));
}

__global__ static void track_refine_reference_kernel(
        const __half *regression, const float *old_reference,
        float *new_reference) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * 3u) return;
    size_t query = index / 3u, coordinate = index % 3u;
    size_t regression_coordinate = coordinate < 2u ? coordinate : 4u;
    float logit =
        __half2float(regression[query * 10u + regression_coordinate]) +
        inverse_sigmoid_clamped(old_reference[index]);
    new_reference[index] = 1.0f / (1.0f + expf(-logit));
}

__global__ static void track_box_output_kernel(
        const __half *regression, const float *reference, float *boxes) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * 10u) return;
    size_t query = index / 10u, coordinate = index % 10u;
    float value = __half2float(regression[index]);
    if (coordinate == 0u || coordinate == 1u || coordinate == 4u) {
        size_t reference_coordinate =
            coordinate < 2u ? coordinate : 2u;
        value += inverse_sigmoid_clamped(
            reference[query * 3u + reference_coordinate]);
        value = 1.0f / (1.0f + expf(-value));
        if (coordinate < 2u)
            value = value * 102.4f - 51.2f;
        else
            value = value * 8.0f - 5.0f;
    }
    boxes[index] = value;
}

__global__ static void track_score_class_kernel(
        const __half *logits, size_t query_count, size_t class_count,
        float *scores, uint32_t *classes) {
    size_t query = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= query_count) return;
    float best_score = -INFINITY;
    uint32_t best_class = 0u;
    for (uint32_t class_index = 0u;
         class_index < (uint32_t)class_count; ++class_index) {
        float logit =
            __half2float(logits[query * class_count + class_index]);
        float score = 1.0f / (1.0f + expf(-logit));
        if (score > best_score) {
            best_score = score;
            best_class = class_index;
        }
    }
    scores[query] = best_score;
    classes[query] = best_class;
}

__global__ static void track_filter_kernel(
        const float *scores, size_t query_count, float threshold,
        size_t capacity, uint32_t *selected, uint32_t *selected_count) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t count = 0u;
    for (uint32_t query = 0u;
         query < (uint32_t)query_count; ++query) {
        if (scores[query] >= threshold && count < (uint32_t)capacity)
            selected[count++] = query;
    }
    *selected_count = count;
}

__global__ static void tracker_state_update_kernel(
        const float *scores, int32_t *object_ids, uint32_t *disappear,
        size_t count, float score_threshold, float filter_threshold,
        uint32_t miss_tolerance, int32_t *next_object_id,
        uint32_t *active_indices, size_t active_capacity,
        uint32_t *active_count) {
    if (blockIdx.x || threadIdx.x) return;
    int32_t next_id = *next_object_id;
    uint32_t output_count = 0u;
    for (uint32_t index = 0u; index < (uint32_t)count; ++index) {
        float score = scores[index];
        int32_t object_id = object_ids[index];
        if (score >= score_threshold)
            disappear[index] = 0u;
        if (object_id == -1 && score >= score_threshold) {
            object_id = next_id++;
            object_ids[index] = object_id;
        } else if (object_id >= 0 && score < filter_threshold) {
            uint32_t missed = disappear[index] + 1u;
            disappear[index] = missed;
            if (missed >= miss_tolerance) {
                object_id = -1;
                object_ids[index] = -1;
            }
        }
        if (object_id >= 0 && score >= filter_threshold &&
            output_count < (uint32_t)active_capacity)
            active_indices[output_count++] = index;
    }
    *next_object_id = next_id;
    *active_count = output_count;
}

__global__ static void memory_bank_select_kernel(
        const float *scores, uint8_t *padding_mask, uint8_t *save_period,
        uint8_t *saved, size_t queries, size_t history, float threshold,
        uint8_t reset_period) {
    size_t query = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= queries) return;
    uint8_t period = save_period[query];
    uint8_t should_save =
        period == 0u && scores[query] > threshold ? 1u : 0u;
    if (period > 0u) --period;
    if (should_save) period = reset_period;
    save_period[query] = period;
    saved[query] = should_save;
    if (should_save) {
        for (size_t slot = 0u; slot + 1u < history; ++slot)
            padding_mask[query * history + slot] =
                padding_mask[query * history + slot + 1u];
        padding_mask[query * history + history - 1u] = 0u;
    }
}

__global__ static void memory_bank_save_kernel(
        const __half *embedding, const __half *weight, const __half *bias,
        const uint8_t *saved, __half *memory, size_t queries,
        size_t history, size_t dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= queries * dimensions) return;
    size_t query = index / dimensions, channel = index % dimensions;
    if (!saved[query]) return;
    float accumulator = __half2float(bias[channel]);
    for (size_t input_channel = 0u;
         input_channel < dimensions; ++input_channel)
        accumulator +=
            __half2float(embedding[query * dimensions + input_channel]) *
            __half2float(weight[channel * dimensions + input_channel]);
    for (size_t slot = 0u; slot + 1u < history; ++slot)
        memory[(query * history + slot) * dimensions + channel] =
            memory[(query * history + slot + 1u) * dimensions + channel];
    memory[(query * history + history - 1u) * dimensions + channel] =
        __float2half_rn(accumulator);
}

__global__ static void track_qkv_linear_kernel(
        const __half *query, const __half *query_pos, const __half *weight,
        const __half *bias, __half *qkv, size_t rows) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * 768u) return;
    size_t row = index / 768u, output = index % 768u;
    const __half *weight_row = weight + output * 256u;
    float accumulator = __half2float(bias[output]);
    for (size_t column = 0; column < 256u; ++column) {
        float value = __half2float(query[row * 256u + column]);
        if (output < 512u)
            value += __half2float(query_pos[row * 256u + column]);
        accumulator += value * __half2float(weight_row[column]);
    }
    qkv[index] = __float2half_rn(accumulator);
}

__global__ static void track_self_attention_kernel(
        const __half *qkv, __half *output, size_t rows) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= rows * 8u) return;
    size_t query_index = group / 8u, head = group % 8u;
    float maximum = -INFINITY;
    const float scale = 0.1767766952966369f;
    for (size_t key_index = 0; key_index < rows; ++key_index) {
        float score = 0.0f;
        for (size_t channel = 0; channel < 32u; ++channel)
            score += __half2float(
                         qkv[query_index * 768u + head * 32u + channel]) *
                     __half2float(
                         qkv[key_index * 768u + 256u +
                             head * 32u + channel]);
        maximum = fmaxf(maximum, score * scale);
    }
    float denominator = 0.0f;
    float accumulator[32] = {0.0f};
    for (size_t key_index = 0; key_index < rows; ++key_index) {
        float score = 0.0f;
        for (size_t channel = 0; channel < 32u; ++channel)
            score += __half2float(
                         qkv[query_index * 768u + head * 32u + channel]) *
                     __half2float(
                         qkv[key_index * 768u + 256u +
                             head * 32u + channel]);
        float probability = expf(score * scale - maximum);
        denominator += probability;
        for (size_t channel = 0; channel < 32u; ++channel)
            accumulator[channel] += probability * __half2float(
                qkv[key_index * 768u + 512u + head * 32u + channel]);
    }
    for (size_t channel = 0; channel < 32u; ++channel)
        output[query_index * 256u + head * 32u + channel] =
            __float2half_rn(accumulator[channel] / denominator);
}

__global__ static void track_query_pos_linear_kernel(
        const __half *query, const __half *query_pos, const __half *weight,
        const __half *bias, __half *output, size_t output_dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * output_dimensions) return;
    size_t row = index / output_dimensions;
    size_t output_dimension = index % output_dimensions;
    float accumulator = __half2float(bias[output_dimension]);
    const __half *weight_row = weight + output_dimension * 256u;
    for (size_t column = 0; column < 256u; ++column)
        accumulator +=
            (__half2float(query[row * 256u + column]) +
             __half2float(query_pos[row * 256u + column])) *
            __half2float(weight_row[column]);
    output[index] = __float2half_rn(accumulator);
}

__global__ static void track_cross_softmax_kernel(__half *attention) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= 901u * 8u) return;
    __half *values = attention + group * 4u;
    float maximum = -INFINITY, denominator = 0.0f, exponentials[4];
    for (size_t point = 0; point < 4u; ++point)
        maximum = fmaxf(maximum, __half2float(values[point]));
    for (size_t point = 0; point < 4u; ++point) {
        exponentials[point] = expf(__half2float(values[point]) - maximum);
        denominator += exponentials[point];
    }
    for (size_t point = 0; point < 4u; ++point)
        values[point] = __float2half_rn(exponentials[point] / denominator);
}

__global__ static void track_cross_deform_sample_kernel(
        const __half *value, const __half *offsets, const __half *attention,
        const float *reference_points, __half *sampled) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 901u * 256u) return;
    size_t query = index / 256u, channel = index % 256u;
    size_t head = channel / 32u;
    float accumulator = 0.0f;
    for (size_t point = 0; point < 4u; ++point) {
        size_t sample = (query * 8u + head) * 4u + point;
        float location_x = reference_points[query * 3u] +
            __half2float(offsets[sample * 2u]) / 200.0f;
        float location_y = reference_points[query * 3u + 1u] +
            __half2float(offsets[sample * 2u + 1u]) / 200.0f;
        accumulator += temporal_bilinear_value(
            value, channel, location_x, location_y) *
            __half2float(attention[sample]);
    }
    sampled[index] = __float2half_rn(accumulator);
}

__global__ static void linear_residual_fp16_kernel(
        const __half *input, const __half *weight, const __half *bias,
        const __half *residual, __half *output, size_t rows,
        size_t dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = rows * dimensions;
    if (index >= count) return;
    size_t row = index / dimensions;
    size_t output_dimension = index % dimensions;
    float accumulator = __half2float(bias[output_dimension]);
    for (size_t column = 0; column < dimensions; ++column)
        accumulator +=
            __half2float(input[row * dimensions + column]) *
            __half2float(
                weight[output_dimension * dimensions + column]);
    output[index] = __float2half_rn(
        accumulator + __half2float(residual[index]));
}

__global__ static void spatial_compact_linear_kernel(
        const __half *query, const uint32_t *visible_indices,
        const uint32_t *visible_counts, size_t camera, const __half *weight,
        const __half *bias, __half *output, size_t output_dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slot = index / output_dimensions;
    size_t output_dimension = index % output_dimensions;
    if (slot >= 40000u || slot >= visible_counts[camera]) return;
    size_t query_index = visible_indices[camera * 40000u + slot];
    float accumulator = __half2float(bias[output_dimension]);
    const __half *weight_row = weight + output_dimension * 256u;
    for (size_t column = 0; column < 256u; ++column)
        accumulator +=
            __half2float(query[query_index * 256u + column]) *
            __half2float(weight_row[column]);
    output[slot * output_dimensions + output_dimension] =
        __float2half_rn(accumulator);
}

__global__ static void spatial_attention_softmax_kernel(
        __half *weights, const uint32_t *visible_counts, size_t camera) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slot = group / 8u;
    if (slot >= 40000u || slot >= visible_counts[camera]) return;
    __half *values = weights + group * 32u;
    float maximum = -INFINITY, sum = 0.0f;
    float exponentials[32];
    for (size_t point = 0; point < 32u; ++point)
        maximum = fmaxf(maximum, __half2float(values[point]));
    for (size_t point = 0; point < 32u; ++point) {
        exponentials[point] = expf(__half2float(values[point]) - maximum);
        sum += exponentials[point];
    }
    for (size_t point = 0; point < 32u; ++point)
        values[point] = __float2half_rn(exponentials[point] / sum);
}

__device__ static float spatial_bilinear_value(
        const __half *value, size_t camera, size_t level, size_t channel,
        float normalized_x, float normalized_y) {
    const size_t heights[4] = {116u, 58u, 29u, 15u};
    const size_t widths[4] = {200u, 100u, 50u, 25u};
    const size_t starts[4] = {0u, 23200u, 29000u, 30450u};
    size_t height = heights[level], width = widths[level];
    float x = normalized_x * (float)width - .5f;
    float y = normalized_y * (float)height - .5f;
    long x0 = (long)floorf(x), y0 = (long)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0, sampled = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        long yy = y0 + (corner >= 2);
        long xx = x0 + (corner & 1);
        if (yy >= 0 && xx >= 0 && (size_t)yy < height &&
            (size_t)xx < width) {
            float wy = corner >= 2 ? fy : 1.0f - fy;
            float wx = corner & 1 ? fx : 1.0f - fx;
            size_t token = starts[level] + (size_t)yy * width + (size_t)xx;
            size_t index = (camera * 30825u + token) * 256u + channel;
            sampled += __half2float(value[index]) * wy * wx;
        }
    }
    return sampled;
}

__global__ static void spatial_deform_sample_kernel(
        const __half *value, const __half *offsets,
        const __half *attention, const float *reference_camera,
        const uint32_t *visible_indices, const uint32_t *visible_counts,
        size_t camera, __half *sampled_output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slot = index / 256u;
    size_t channel = index % 256u;
    if (slot >= 40000u || slot >= visible_counts[camera]) return;
    const size_t heights[4] = {116u, 58u, 29u, 15u};
    const size_t widths[4] = {200u, 100u, 50u, 25u};
    size_t query = visible_indices[camera * 40000u + slot];
    size_t head = channel / 32u;
    float accumulator = 0.0f;
    for (size_t level = 0; level < 4u; ++level)
        for (size_t point = 0; point < 8u; ++point) {
            size_t sample = ((slot * 8u + head) * 4u + level) * 8u + point;
            size_t depth = point % 4u;
            size_t reference =
                ((camera * 40000u + query) * 4u + depth) * 2u;
            float location_x = reference_camera[reference] +
                __half2float(offsets[sample * 2u]) / (float)widths[level];
            float location_y = reference_camera[reference + 1u] +
                __half2float(offsets[sample * 2u + 1u]) /
                (float)heights[level];
            accumulator += spatial_bilinear_value(
                value, camera, level, channel, location_x, location_y) *
                __half2float(attention[
                    ((slot * 8u + head) * 4u + level) * 8u + point]);
        }
    sampled_output[slot * 256u + channel] =
        __float2half_rn(accumulator);
}

__global__ static void spatial_scatter_kernel(
        const __half *sampled, const uint32_t *visible_indices,
        const uint32_t *visible_counts, size_t camera, float *slots) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slot = index / 256u;
    size_t channel = index % 256u;
    if (slot >= 40000u || slot >= visible_counts[camera]) return;
    size_t query = visible_indices[camera * 40000u + slot];
    slots[query * 256u + channel] += __half2float(sampled[index]);
}

__global__ static void spatial_output_projection_kernel(
        const float *slots, const uint8_t *visibility, const __half *weight,
        const __half *bias, const __half *residual, __half *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 40000u * 256u) return;
    size_t query = index / 256u, output_dimension = index % 256u;
    size_t observations = 0;
    for (size_t camera = 0; camera < UA_CAMERA_COUNT; ++camera) {
        size_t base = (camera * 40000u + query) * 4u;
        if (visibility[base] || visibility[base + 1u] ||
            visibility[base + 2u] || visibility[base + 3u])
            ++observations;
    }
    float inverse = observations ? 1.0f / (float)observations : 0.0f;
    float accumulator = __half2float(bias[output_dimension]);
    const __half *weight_row = weight + output_dimension * 256u;
    for (size_t column = 0; column < 256u; ++column)
        accumulator += slots[query * 256u + column] * inverse *
                       __half2float(weight_row[column]);
    output[index] = __float2half_rn(
        accumulator + __half2float(residual[index]));
}

__global__ static void relu_fp16_kernel(__half *values, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        values[index] = __float2half_rn(
            fmaxf(__half2float(values[index]), 0.0f));
}

__global__ static void linear_rect_residual_fp16_kernel(
        const __half *input, const __half *weight, const __half *bias,
        const __half *residual, __half *output, size_t rows,
        size_t input_dimensions, size_t output_dimensions) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_dimensions) return;
    size_t row = index / output_dimensions;
    size_t output_dimension = index % output_dimensions;
    float accumulator = __half2float(bias[output_dimension]);
    const __half *weight_row =
        weight + output_dimension * input_dimensions;
    for (size_t column = 0; column < input_dimensions; ++column)
        accumulator +=
            __half2float(input[row * input_dimensions + column]) *
            __half2float(weight_row[column]);
    output[index] = __float2half_rn(
        accumulator + __half2float(residual[index]));
}

__global__ static void batchnorm_relu_fp16_kernel(
        const __half *x, const void *gamma, const void *beta,
        const float *mean, const float *variance, __half *y,
        size_t channels, size_t plane, float epsilon, size_t count,
        int affine_fp16) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t channel = (index / plane) % channels;
    float value = (__half2float(x[index]) - mean[channel]) *
                  rsqrtf(variance[channel] + epsilon);
    float scale = affine_fp16
        ? __half2float(((const __half *)gamma)[channel])
        : ((const float *)gamma)[channel];
    float shift = affine_fp16
        ? __half2float(((const __half *)beta)[channel])
        : ((const float *)beta)[channel];
    value = value * scale + shift;
    y[index] = __float2half_rn(fmaxf(value, 0.0f));
}

__global__ static void maxpool2d_fp16_kernel(
        const __half *x, __half *y, size_t channels, size_t input_height,
        size_t input_width, size_t output_height, size_t output_width,
        size_t kernel_height, size_t kernel_width, size_t stride_height,
        size_t stride_width, size_t padding_height, size_t padding_width,
        size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t ox = index % output_width, quotient = index / output_width;
    size_t oy = quotient % output_height;
    quotient /= output_height;
    size_t channel = quotient % channels;
    size_t batch = quotient / channels;
    float maximum = -INFINITY;
    for (size_t ky = 0; ky < kernel_height; ++ky) {
        long iy = (long)(oy * stride_height + ky) - (long)padding_height;
        if (iy < 0 || (size_t)iy >= input_height) continue;
        for (size_t kx = 0; kx < kernel_width; ++kx) {
            long ix = (long)(ox * stride_width + kx) - (long)padding_width;
            if (ix < 0 || (size_t)ix >= input_width) continue;
            size_t input_index =
                ((batch * channels + channel) * input_height + (size_t)iy) *
                input_width + (size_t)ix;
            maximum = fmaxf(maximum, __half2float(x[input_index]));
        }
    }
    y[index] = __float2half_rn(maximum);
}

__device__ static float bn_fp16_affine(
        float value, size_t channel, const __half *gamma, const __half *beta,
        const float *mean, const float *variance) {
    return (value - mean[channel]) * rsqrtf(variance[channel] + 1e-5f) *
           __half2float(gamma[channel]) + __half2float(beta[channel]);
}

__global__ static void bottleneck_identity_final_kernel(
        const __half *main_input, const __half *weight,
        const __half *gamma, const __half *beta, const float *mean,
        const float *variance, __half *residual_and_output, size_t batches,
        size_t channels, size_t mid_channels, size_t height, size_t width) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = batches * channels * height * width;
    if (index >= count) return;
    size_t x = index % width, quotient = index / width;
    size_t y = quotient % height;
    quotient /= height;
    size_t output_channel = quotient % channels;
    size_t batch = quotient / channels;
    float accumulator = 0.0f;
    for (size_t mid = 0; mid < mid_channels; ++mid) {
        size_t input_index =
            ((batch * mid_channels + mid) * height + y) * width + x;
        accumulator += __half2float(main_input[input_index]) *
                       __half2float(weight[output_channel * mid_channels + mid]);
    }
    float value = bn_fp16_affine(
        accumulator, output_channel, gamma, beta, mean, variance);
    value += __half2float(residual_and_output[index]);
    residual_and_output[index] = __float2half_rn(fmaxf(value, 0.0f));
}

__global__ static void bottleneck_projection_final_kernel(
        const __half *main_input, const __half *main_weight,
        const __half *main_gamma, const __half *main_beta,
        const float *main_mean, const float *main_variance,
        const __half *residual_input, const __half *projection_weight,
        const __half *projection_gamma, const __half *projection_beta,
        const float *projection_mean, const float *projection_variance,
        __half *output, size_t batches, size_t input_channels,
        size_t mid_channels, size_t output_channels, size_t input_height,
        size_t input_width, size_t output_height, size_t output_width,
        size_t stride) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = batches * output_channels * output_height * output_width;
    if (index >= count) return;
    size_t x = index % output_width, quotient = index / output_width;
    size_t y = quotient % output_height;
    quotient /= output_height;
    size_t output_channel = quotient % output_channels;
    size_t batch = quotient / output_channels;
    float main_value = 0.0f;
    for (size_t mid = 0; mid < mid_channels; ++mid) {
        size_t input_index =
            ((batch * mid_channels + mid) * output_height + y) *
            output_width + x;
        main_value += __half2float(main_input[input_index]) *
                      __half2float(
                          main_weight[output_channel * mid_channels + mid]);
    }
    main_value = bn_fp16_affine(
        main_value, output_channel, main_gamma, main_beta, main_mean,
        main_variance);
    float residual_value = 0.0f;
    size_t residual_y = y * stride, residual_x = x * stride;
    for (size_t input_channel = 0; input_channel < input_channels;
         ++input_channel) {
        size_t residual_index =
            ((batch * input_channels + input_channel) * input_height +
             residual_y) * input_width + residual_x;
        residual_value += __half2float(residual_input[residual_index]) *
                          __half2float(projection_weight[
                              output_channel * input_channels + input_channel]);
    }
    residual_value = bn_fp16_affine(
        residual_value, output_channel, projection_gamma, projection_beta,
        projection_mean, projection_variance);
    output[index] = __float2half_rn(fmaxf(main_value + residual_value, 0.0f));
}

/* Correctness baseline: one thread owns one reduction. This gives a fixed
 * accumulation order for the oracle gate; optimized block reductions are a
 * separately measured candidate. */
__global__ static void layer_norm_fp16_kernel(
        const __half *x, const __half *gamma, const __half *beta, __half *y,
        size_t rows, size_t dim, float epsilon, int affine) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float mean = 0.0f;
    for (size_t column = 0; column < dim; ++column)
        mean += __half2float(x[row * dim + column]);
    mean /= (float)dim;
    float variance = 0.0f;
    for (size_t column = 0; column < dim; ++column) {
        float delta = __half2float(x[row * dim + column]) - mean;
        variance += delta * delta;
    }
    float inverse = rsqrtf(variance / (float)dim + epsilon);
    for (size_t column = 0; column < dim; ++column) {
        float value = (__half2float(x[row * dim + column]) - mean) * inverse;
        if (affine)
            value = value * __half2float(gamma[column]) +
                    __half2float(beta[column]);
        y[row * dim + column] = __float2half_rn(value);
    }
}

__global__ static void softmax_f32_kernel(
        const float *x, float *y, size_t rows, size_t cols) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float maximum = x[row * cols];
    for (size_t column = 1; column < cols; ++column)
        maximum = fmaxf(maximum, x[row * cols + column]);
    float denominator = 0.0f;
    for (size_t column = 0; column < cols; ++column)
        denominator += expf(x[row * cols + column] - maximum);
    for (size_t column = 0; column < cols; ++column)
        y[row * cols + column] =
            expf(x[row * cols + column] - maximum) / denominator;
}

__device__ static float bilinear_zero(
        const float *image, size_t height, size_t width, float y, float x,
        size_t channels, size_t channel) {
    long y0 = (long)floorf(y), x0 = (long)floorf(x);
    float fy = y - (float)y0, fx = x - (float)x0, value = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        long yy = y0 + (corner >= 2);
        long xx = x0 + (corner & 1);
        if (yy >= 0 && xx >= 0 && (size_t)yy < height && (size_t)xx < width) {
            float wy = corner >= 2 ? fy : 1.0f - fy;
            float wx = corner & 1 ? fx : 1.0f - fx;
            value += image[((size_t)yy * width + (size_t)xx) * channels +
                           channel] * wy * wx;
        }
    }
    return value;
}

__global__ static void resize_bilinear_f32_kernel(
        const float *source, size_t sh, size_t sw, float *target,
        size_t th, size_t tw) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= th * tw) return;
    size_t y = index / tw, x = index % tw;
    float sy = ((float)y + .5f) * (float)sh / (float)th - .5f;
    float sx = ((float)x + .5f) * (float)sw / (float)tw - .5f;
    target[index] = bilinear_zero(source, sh, sw, sy, sx, 1u, 0u);
}

__global__ static void deform_sample_f32_kernel(
        const float *map, size_t height, size_t width, size_t channels,
        const float *points, size_t point_count, float *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count * channels) return;
    size_t point = index / channels, channel = index % channels;
    output[index] = bilinear_zero(
        map, height, width, points[point * 2u], points[point * 2u + 1u],
        channels, channel);
}

__global__ static void stable_topk_f32_kernel(
        const float *scores, size_t count, size_t k, size_t *indices) {
    if (blockIdx.x || threadIdx.x) return;
    for (size_t output = 0; output < k; ++output) {
        size_t best = 0;
        int found = 0;
        for (size_t candidate = 0; candidate < count; ++candidate) {
            int used = 0;
            for (size_t prior = 0; prior < output; ++prior)
                if (indices[prior] == candidate) used = 1;
            if (!used && (!found || scores[candidate] > scores[best] ||
                (scores[candidate] == scores[best] && candidate < best))) {
                best = candidate;
                found = 1;
            }
        }
        indices[output] = best;
    }
}

__global__ static void ms_deform_attn_fp16_kernel(
        const __half *value, size_t total_values, size_t heads,
        size_t channels, const uint32_t *shapes, const size_t *starts,
        size_t levels, const float *locations, const float *weights,
        size_t queries, size_t points, size_t output_count, float *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    size_t per_batch = queries * heads * channels;
    size_t batch = index / per_batch;
    size_t remainder = index % per_batch;
    size_t query = remainder / (heads * channels);
    remainder %= heads * channels;
    size_t head = remainder / channels;
    size_t channel = remainder % channels;
    float accumulator = 0.0f;
    for (size_t level = 0; level < levels; ++level) {
        size_t height = shapes[level * 2u];
        size_t width = shapes[level * 2u + 1u];
        for (size_t point = 0; point < points; ++point) {
            size_t sample =
                ((((batch * queries + query) * heads + head) * levels +
                  level) * points + point);
            float x = locations[sample * 2u] * (float)width - .5f;
            float y = locations[sample * 2u + 1u] * (float)height - .5f;
            long x0 = (long)floorf(x), y0 = (long)floorf(y);
            float fx = x - (float)x0, fy = y - (float)y0;
            float sampled = 0.0f;
            for (int corner = 0; corner < 4; ++corner) {
                long yy = y0 + (corner >= 2);
                long xx = x0 + (corner & 1);
                if (yy >= 0 && xx >= 0 && (size_t)yy < height &&
                    (size_t)xx < width) {
                    size_t spatial = starts[level] + (size_t)yy * width +
                                     (size_t)xx;
                    size_t value_index =
                        ((batch * total_values + spatial) * heads + head) *
                        channels + channel;
                    float wy = corner >= 2 ? fy : 1.0f - fy;
                    float wx = corner & 1 ? fx : 1.0f - fx;
                    sampled += __half2float(value[value_index]) * wy * wx;
                }
            }
            accumulator += sampled * weights[sample];
        }
    }
    output[index] = accumulator;
}

__global__ static void deform_locations_f32_kernel(
        const float *reference, size_t reference_dims, const float *offset,
        const uint32_t *shapes, size_t queries, size_t heads, size_t levels,
        size_t points, size_t count, float *locations) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t coordinate = index & 1u;
    size_t sample = index >> 1u;
    size_t quotient = sample / points;
    size_t level = quotient % levels;
    quotient /= levels;
    quotient /= heads;
    size_t query = quotient % queries;
    size_t batch = quotient / queries;
    size_t reference_index =
        ((batch * queries + query) * levels + level) * reference_dims;
    float location;
    if (reference_dims == 2u) {
        float normalizer =
            coordinate == 0u ? (float)shapes[level * 2u + 1u]
                             : (float)shapes[level * 2u];
        location = reference[reference_index + coordinate] +
                   offset[index] / normalizer;
    } else {
        location = reference[reference_index + coordinate] +
                   offset[index] / (float)points *
                   reference[reference_index + 2u + coordinate] * .5f;
    }
    locations[index] = location;
}

__global__ static void camera_scatter_average_f32_kernel(
        const float *slots, const uint32_t *indices, const uint32_t *counts,
        size_t cameras, size_t max_queries, size_t total_queries,
        size_t dimensions, size_t count, float *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t query = index / dimensions, dimension = index % dimensions;
    float sum = 0.0f;
    size_t observations = 0;
    for (size_t camera = 0; camera < cameras; ++camera)
        for (size_t slot = 0; slot < counts[camera]; ++slot)
            if (indices[camera * max_queries + slot] == query) {
                sum += slots[(camera * max_queries + slot) * dimensions +
                             dimension];
                ++observations;
            }
    output[index] = observations ? sum / (float)observations : 0.0f;
    (void)total_queries;
}

__global__ static void queue_mean_f32_kernel(
        const float *queue, size_t queries, size_t dimensions,
        size_t queue_length, size_t count, float *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t per_batch = queries * dimensions;
    size_t batch = index / per_batch;
    size_t local = index % per_batch;
    float sum = 0.0f;
    for (size_t q = 0; q < queue_length; ++q)
        sum += queue[((batch * queue_length + q) * queries * dimensions) +
                     local];
    output[index] = sum / (float)queue_length;
}

static int finite_array(const float *values, size_t count) {
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static int fixture_count_ok(size_t count, size_t element_size) {
    return count <= SIZE_MAX / element_size &&
           count <= (size_t)UINT_MAX * 256u;
}

extern "C" ua_status ua_cuda_test_preprocess_bgr(
        const uint8_t *source, size_t width, size_t height, size_t row_stride,
        size_t padded_width, size_t padded_height, const float mean_bgr[3],
        const float std_bgr[3], float *output_chw) {
    uint8_t *device_source = NULL;
    __half *device_half = NULL;
    float *device_output = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t source_bytes, output_count;
    if (!source || !output_chw || !mean_bgr || !std_bgr || !width || !height ||
        !row_stride ||
        padded_width < width || padded_height < height)
        return UA_ERR_ARGUMENT;
    if (width > SIZE_MAX / 3u || height > SIZE_MAX / row_stride ||
        padded_height > SIZE_MAX / padded_width ||
        padded_height * padded_width > SIZE_MAX / 3u)
        return UA_ERR_CAPACITY;
    if (row_stride < width * 3u ||
        !finite_array(mean_bgr, 3) || !finite_array(std_bgr, 3) ||
        std_bgr[0] <= 0.0f || std_bgr[1] <= 0.0f || std_bgr[2] <= 0.0f)
        return UA_ERR_ARGUMENT;
    source_bytes = height * row_stride;
    output_count = padded_height * padded_width * 3u;
    if (!fixture_count_ok(output_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (cudaMalloc((void **)&device_source, source_bytes) != cudaSuccess ||
        cudaMalloc((void **)&device_half, output_count * sizeof(__half)) != cudaSuccess ||
        cudaMalloc((void **)&device_output, output_count * sizeof(float)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_source, source, source_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) goto cleanup;
    {
        float3 mean = make_float3(mean_bgr[0], mean_bgr[1], mean_bgr[2]);
        float3 inverse = make_float3(1.0f / std_bgr[0], 1.0f / std_bgr[1],
                                    1.0f / std_bgr[2]);
        size_t blocks = (output_count + 255u) / 256u;
        preprocess_bgr_kernel<<<(unsigned)blocks, 256>>>(
            device_source, width, height, row_stride, padded_width,
            padded_height, mean, inverse, device_half);
        half_to_float_kernel<<<(unsigned)blocks, 256>>>(
            device_half, device_output, output_count);
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output_chw, device_output, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_output);
    cudaFree(device_half);
    cudaFree(device_source);
    return result;
}

extern "C" ua_status ua_cuda_test_linear_fp16(
        const float *x, const float *weight, const float *bias, float *y,
        size_t rows, size_t in_dim, size_t out_dim) {
    float *device_float_x = NULL, *device_float_weight = NULL;
    float *device_float_bias = NULL, *device_float_y = NULL;
    __half *device_x = NULL, *device_weight = NULL, *device_bias = NULL;
    __half *device_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t x_count, weight_count, y_count;
    if (!rows || !in_dim || !out_dim || !x || !weight || !y)
        return UA_ERR_ARGUMENT;
    if (rows > SIZE_MAX / in_dim || out_dim > SIZE_MAX / in_dim ||
        rows > SIZE_MAX / out_dim)
        return UA_ERR_CAPACITY;
    x_count = rows * in_dim;
    weight_count = out_dim * in_dim;
    y_count = rows * out_dim;
    if (!fixture_count_ok(x_count, sizeof(float)) ||
        !fixture_count_ok(weight_count, sizeof(float)) ||
        !fixture_count_ok(y_count, sizeof(float)) ||
        !fixture_count_ok(out_dim, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, x_count) || !finite_array(weight, weight_count) ||
        (bias && !finite_array(bias, out_dim)))
        return UA_ERR_NONFINITE;
#define UA_CUDA_ALLOC(pointer, count, type) \
    if (cudaMalloc((void **)&(pointer), (count) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_CUDA_ALLOC(device_float_x, x_count, float);
    UA_CUDA_ALLOC(device_float_weight, weight_count, float);
    UA_CUDA_ALLOC(device_float_y, y_count, float);
    UA_CUDA_ALLOC(device_x, x_count, __half);
    UA_CUDA_ALLOC(device_weight, weight_count, __half);
    UA_CUDA_ALLOC(device_y, y_count, __half);
    if (bias) {
        UA_CUDA_ALLOC(device_float_bias, out_dim, float);
        UA_CUDA_ALLOC(device_bias, out_dim, __half);
    }
    if (cudaMemcpy(device_float_x, x, x_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_float_weight, weight, weight_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias && cudaMemcpy(device_float_bias, bias, out_dim * sizeof(float),
                            cudaMemcpyHostToDevice) != cudaSuccess))
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((x_count + 255u) / 256u), 256>>>(
        device_float_x, device_x, x_count);
    float_to_half_kernel<<<(unsigned)((weight_count + 255u) / 256u), 256>>>(
        device_float_weight, device_weight, weight_count);
    if (bias)
        float_to_half_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256>>>(
            device_float_bias, device_bias, out_dim);
    linear_fp16_kernel<<<(unsigned)((y_count + 255u) / 256u), 256>>>(
        device_x, device_weight, device_bias, device_y, rows, in_dim, out_dim,
        bias != NULL);
    half_to_float_kernel<<<(unsigned)((y_count + 255u) / 256u), 256>>>(
        device_y, device_float_y, y_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, device_float_y, y_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_y); cudaFree(device_bias); cudaFree(device_weight);
    cudaFree(device_x); cudaFree(device_float_y); cudaFree(device_float_bias);
    cudaFree(device_float_weight); cudaFree(device_float_x);
    return result;
#undef UA_CUDA_ALLOC
}

extern "C" ua_status ua_cuda_test_conv2d_fp16(
        const float *x, const float *weight, const float *bias, float *y,
        size_t batches, size_t input_channels, size_t input_height,
        size_t input_width, size_t output_channels, size_t kernel_height,
        size_t kernel_width, size_t stride_height, size_t stride_width,
        size_t padding_height, size_t padding_width) {
    float *float_x = NULL, *float_weight = NULL, *float_bias = NULL;
    float *float_y = NULL;
    __half *half_x = NULL, *half_weight = NULL, *half_bias = NULL;
    __half *half_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t input_count, weight_count, output_count, output_height, output_width;
    if (!x || !weight || !y || !batches || !input_channels || !input_height ||
        !input_width || !output_channels || !kernel_height || !kernel_width ||
        !stride_height || !stride_width)
        return UA_ERR_ARGUMENT;
    if (padding_height > (SIZE_MAX - input_height) / 2u ||
        padding_width > (SIZE_MAX - input_width) / 2u)
        return UA_ERR_CAPACITY;
    if (
        input_height + 2u * padding_height < kernel_height ||
        input_width + 2u * padding_width < kernel_width)
        return UA_ERR_ARGUMENT;
    output_height =
        (input_height + 2u * padding_height - kernel_height) / stride_height + 1u;
    output_width =
        (input_width + 2u * padding_width - kernel_width) / stride_width + 1u;
    if (batches > SIZE_MAX / input_channels ||
        batches * input_channels > SIZE_MAX / input_height ||
        batches * input_channels * input_height > SIZE_MAX / input_width ||
        output_channels > SIZE_MAX / input_channels ||
        output_channels * input_channels > SIZE_MAX / kernel_height ||
        output_channels * input_channels * kernel_height > SIZE_MAX / kernel_width ||
        batches > SIZE_MAX / output_channels ||
        batches * output_channels > SIZE_MAX / output_height ||
        batches * output_channels * output_height > SIZE_MAX / output_width)
        return UA_ERR_CAPACITY;
    input_count = batches * input_channels * input_height * input_width;
    weight_count =
        output_channels * input_channels * kernel_height * kernel_width;
    output_count = batches * output_channels * output_height * output_width;
    if (!fixture_count_ok(input_count, sizeof(float)) ||
        !fixture_count_ok(weight_count, sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)) ||
        !fixture_count_ok(output_channels, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, input_count) ||
        !finite_array(weight, weight_count) ||
        (bias && !finite_array(bias, output_channels)))
        return UA_ERR_NONFINITE;
#define UA_CONV_ALLOC(pointer, count, type) \
    if (cudaMalloc((void **)&(pointer), (count) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_CONV_ALLOC(float_x, input_count, float);
    UA_CONV_ALLOC(float_weight, weight_count, float);
    UA_CONV_ALLOC(float_y, output_count, float);
    UA_CONV_ALLOC(half_x, input_count, __half);
    UA_CONV_ALLOC(half_weight, weight_count, __half);
    UA_CONV_ALLOC(half_y, output_count, __half);
    if (bias) {
        UA_CONV_ALLOC(float_bias, output_channels, float);
        UA_CONV_ALLOC(half_bias, output_channels, __half);
    }
    if (cudaMemcpy(float_x, x, input_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(float_weight, weight, weight_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias && cudaMemcpy(float_bias, bias, output_channels * sizeof(float),
                            cudaMemcpyHostToDevice) != cudaSuccess))
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((input_count + 255u) / 256u), 256>>>(
        float_x, half_x, input_count);
    float_to_half_kernel<<<(unsigned)((weight_count + 255u) / 256u), 256>>>(
        float_weight, half_weight, weight_count);
    if (bias)
        float_to_half_kernel<<<
            (unsigned)((output_channels + 255u) / 256u), 256>>>(
            float_bias, half_bias, output_channels);
    conv2d_fp16_kernel<<<(unsigned)((output_count + 255u) / 256u), 256>>>(
        half_x, half_weight, half_bias, half_y, batches, input_channels,
        input_height, input_width, output_channels, output_height, output_width,
        kernel_height, kernel_width, stride_height, stride_width,
        padding_height, padding_width, bias != NULL);
    half_to_float_kernel<<<(unsigned)((output_count + 255u) / 256u), 256>>>(
        half_y, float_y, output_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, float_y, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(half_y); cudaFree(half_bias); cudaFree(half_weight);
    cudaFree(half_x); cudaFree(float_y); cudaFree(float_bias);
    cudaFree(float_weight); cudaFree(float_x);
    return result;
#undef UA_CONV_ALLOC
}

extern "C" ua_status ua_cuda_test_modulated_deform_conv2d_fp16(
        const float *x, const float *offset, const float *mask,
        const float *weight, const float *bias, float *y, size_t batches,
        size_t input_channels, size_t input_height, size_t input_width,
        size_t output_channels, size_t kernel_height, size_t kernel_width,
        size_t stride_height, size_t stride_width, size_t padding_height,
        size_t padding_width, size_t dilation_height, size_t dilation_width) {
    float *float_x = NULL, *float_weight = NULL, *float_bias = NULL;
    float *device_offset = NULL, *device_mask = NULL, *float_y = NULL;
    __half *half_x = NULL, *half_weight = NULL, *half_bias = NULL;
    __half *half_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t effective_h, effective_w, output_height, output_width;
    size_t x_count, weight_count, kernel_elements, output_count;
    size_t offset_count, mask_count;
    if (!x || !offset || !mask || !weight || !y || !batches ||
        !input_channels || !input_height || !input_width || !output_channels ||
        !kernel_height || !kernel_width || !stride_height || !stride_width ||
        !dilation_height || !dilation_width)
        return UA_ERR_ARGUMENT;
    if (kernel_height - 1u > (SIZE_MAX - 1u) / dilation_height ||
        kernel_width - 1u > (SIZE_MAX - 1u) / dilation_width)
        return UA_ERR_CAPACITY;
    effective_h = dilation_height * (kernel_height - 1u) + 1u;
    effective_w = dilation_width * (kernel_width - 1u) + 1u;
    if (padding_height > (SIZE_MAX - input_height) / 2u ||
        padding_width > (SIZE_MAX - input_width) / 2u)
        return UA_ERR_CAPACITY;
    if (input_height + 2u * padding_height < effective_h ||
        input_width + 2u * padding_width < effective_w)
        return UA_ERR_ARGUMENT;
    output_height =
        (input_height + 2u * padding_height - effective_h) / stride_height + 1u;
    output_width =
        (input_width + 2u * padding_width - effective_w) / stride_width + 1u;
    if (kernel_height > SIZE_MAX / kernel_width ||
        batches > SIZE_MAX / input_channels ||
        batches * input_channels > SIZE_MAX / input_height ||
        batches * input_channels * input_height > SIZE_MAX / input_width ||
        output_channels > SIZE_MAX / input_channels ||
        output_channels * input_channels > SIZE_MAX / kernel_height ||
        output_channels * input_channels * kernel_height > SIZE_MAX / kernel_width ||
        batches > SIZE_MAX / output_channels ||
        batches * output_channels > SIZE_MAX / output_height ||
        batches * output_channels * output_height > SIZE_MAX / output_width)
        return UA_ERR_CAPACITY;
    kernel_elements = kernel_height * kernel_width;
    if (kernel_elements > SIZE_MAX / 2u ||
        batches > SIZE_MAX / (2u * kernel_elements) ||
        batches * 2u * kernel_elements > SIZE_MAX / output_height ||
        batches * 2u * kernel_elements * output_height > SIZE_MAX / output_width)
        return UA_ERR_CAPACITY;
    x_count = batches * input_channels * input_height * input_width;
    weight_count =
        output_channels * input_channels * kernel_height * kernel_width;
    output_count = batches * output_channels * output_height * output_width;
    offset_count =
        batches * 2u * kernel_elements * output_height * output_width;
    mask_count = batches * kernel_elements * output_height * output_width;
    if (!fixture_count_ok(x_count, sizeof(float)) ||
        !fixture_count_ok(weight_count, sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)) ||
        !fixture_count_ok(offset_count, sizeof(float)) ||
        !fixture_count_ok(mask_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, x_count) || !finite_array(offset, offset_count) ||
        !finite_array(mask, mask_count) ||
        !finite_array(weight, weight_count) ||
        (bias && !finite_array(bias, output_channels)))
        return UA_ERR_NONFINITE;
#define UA_DCN_ALLOC(pointer, count, type) \
    if (cudaMalloc((void **)&(pointer), (count) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_DCN_ALLOC(float_x, x_count, float);
    UA_DCN_ALLOC(float_weight, weight_count, float);
    UA_DCN_ALLOC(device_offset, offset_count, float);
    UA_DCN_ALLOC(device_mask, mask_count, float);
    UA_DCN_ALLOC(float_y, output_count, float);
    UA_DCN_ALLOC(half_x, x_count, __half);
    UA_DCN_ALLOC(half_weight, weight_count, __half);
    UA_DCN_ALLOC(half_y, output_count, __half);
    if (bias) {
        UA_DCN_ALLOC(float_bias, output_channels, float);
        UA_DCN_ALLOC(half_bias, output_channels, __half);
    }
    if (cudaMemcpy(float_x, x, x_count * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(float_weight, weight, weight_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_offset, offset, offset_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_mask, mask, mask_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias && cudaMemcpy(float_bias, bias, output_channels * sizeof(float),
                            cudaMemcpyHostToDevice) != cudaSuccess))
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((x_count + 255u) / 256u), 256>>>(
        float_x, half_x, x_count);
    float_to_half_kernel<<<(unsigned)((weight_count + 255u) / 256u), 256>>>(
        float_weight, half_weight, weight_count);
    if (bias)
        float_to_half_kernel<<<
            (unsigned)((output_channels + 255u) / 256u), 256>>>(
            float_bias, half_bias, output_channels);
    modulated_deform_conv2d_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        half_x, device_offset, device_mask, half_weight, half_bias, half_y,
        batches, input_channels, input_height, input_width, output_channels,
        output_height, output_width, kernel_height, kernel_width, stride_height,
        stride_width, padding_height, padding_width, dilation_height,
        dilation_width, bias != NULL, 0);
    half_to_float_kernel<<<(unsigned)((output_count + 255u) / 256u), 256>>>(
        half_y, float_y, output_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, float_y, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(half_y); cudaFree(half_bias); cudaFree(half_weight);
    cudaFree(half_x); cudaFree(float_y); cudaFree(device_mask);
    cudaFree(device_offset); cudaFree(float_bias); cudaFree(float_weight);
    cudaFree(float_x);
    return result;
#undef UA_DCN_ALLOC
}

extern "C" ua_status ua_cuda_test_batchnorm_relu_fp16(
        const float *x, const float *gamma, const float *beta,
        const float *mean, const float *variance, float *y, size_t batches,
        size_t channels, size_t height, size_t width, float epsilon) {
    float *float_x = NULL, *float_y = NULL;
    float *dg = NULL, *db = NULL, *dm = NULL, *dv = NULL;
    __half *half_x = NULL, *half_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t plane, count;
    if (!x || !gamma || !beta || !mean || !variance || !y || !batches ||
        !channels || !height || !width || !isfinite(epsilon) ||
        epsilon <= 0.0f)
        return UA_ERR_ARGUMENT;
    if (height > SIZE_MAX / width || batches > SIZE_MAX / channels ||
        batches * channels > SIZE_MAX / height ||
        batches * channels * height > SIZE_MAX / width)
        return UA_ERR_CAPACITY;
    plane = height * width;
    count = batches * channels * plane;
    if (!fixture_count_ok(count, sizeof(float)) ||
        !fixture_count_ok(channels, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, count) || !finite_array(gamma, channels) ||
        !finite_array(beta, channels) || !finite_array(mean, channels) ||
        !finite_array(variance, channels))
        return UA_ERR_NONFINITE;
    for (size_t c = 0; c < channels; ++c)
        if (variance[c] < 0.0f) return UA_ERR_ARGUMENT;
#define UA_BN_ALLOC(pointer, count_, type) \
    if (cudaMalloc((void **)&(pointer), (count_) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_BN_ALLOC(float_x, count, float);
    UA_BN_ALLOC(float_y, count, float);
    UA_BN_ALLOC(half_x, count, __half);
    UA_BN_ALLOC(half_y, count, __half);
    UA_BN_ALLOC(dg, channels, float);
    UA_BN_ALLOC(db, channels, float);
    UA_BN_ALLOC(dm, channels, float);
    UA_BN_ALLOC(dv, channels, float);
    if (cudaMemcpy(float_x, x, count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dg, gamma, channels * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, beta, channels * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dm, mean, channels * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dv, variance, channels * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
        float_x, half_x, count);
    batchnorm_relu_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256>>>(
        half_x, dg, db, dm, dv, half_y, channels, plane, epsilon, count, 0);
    half_to_float_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
        half_y, float_y, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, float_y, count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(dv); cudaFree(dm); cudaFree(db); cudaFree(dg);
    cudaFree(half_y); cudaFree(half_x); cudaFree(float_y); cudaFree(float_x);
    return result;
#undef UA_BN_ALLOC
}

extern "C" ua_status ua_cuda_test_maxpool2d_fp16(
        const float *x, float *y, size_t batches, size_t channels,
        size_t input_height, size_t input_width, size_t kernel_height,
        size_t kernel_width, size_t stride_height, size_t stride_width,
        size_t padding_height, size_t padding_width) {
    float *float_x = NULL, *float_y = NULL;
    __half *half_x = NULL, *half_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t output_height, output_width, input_count, output_count;
    if (!x || !y || !batches || !channels || !input_height || !input_width ||
        !kernel_height || !kernel_width || !stride_height || !stride_width)
        return UA_ERR_ARGUMENT;
    if (padding_height > (SIZE_MAX - input_height) / 2u ||
        padding_width > (SIZE_MAX - input_width) / 2u)
        return UA_ERR_CAPACITY;
    if (input_height + 2u * padding_height < kernel_height ||
        input_width + 2u * padding_width < kernel_width)
        return UA_ERR_ARGUMENT;
    output_height =
        (input_height + 2u * padding_height - kernel_height) / stride_height + 1u;
    output_width =
        (input_width + 2u * padding_width - kernel_width) / stride_width + 1u;
    if (batches > SIZE_MAX / channels ||
        batches * channels > SIZE_MAX / input_height ||
        batches * channels * input_height > SIZE_MAX / input_width ||
        batches * channels > SIZE_MAX / output_height ||
        batches * channels * output_height > SIZE_MAX / output_width)
        return UA_ERR_CAPACITY;
    input_count = batches * channels * input_height * input_width;
    output_count = batches * channels * output_height * output_width;
    if (!fixture_count_ok(input_count, sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, input_count)) return UA_ERR_NONFINITE;
#define UA_POOL_ALLOC(pointer, count_, type) \
    if (cudaMalloc((void **)&(pointer), (count_) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_POOL_ALLOC(float_x, input_count, float);
    UA_POOL_ALLOC(float_y, output_count, float);
    UA_POOL_ALLOC(half_x, input_count, __half);
    UA_POOL_ALLOC(half_y, output_count, __half);
    if (cudaMemcpy(float_x, x, input_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((input_count + 255u) / 256u), 256>>>(
        float_x, half_x, input_count);
    maxpool2d_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        half_x, half_y, channels, input_height, input_width, output_height,
        output_width, kernel_height, kernel_width, stride_height, stride_width,
        padding_height, padding_width, output_count);
    half_to_float_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        half_y, float_y, output_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, float_y, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(half_y); cudaFree(half_x); cudaFree(float_y); cudaFree(float_x);
    return result;
#undef UA_POOL_ALLOC
}

extern "C" ua_status ua_cuda_test_layer_norm_fp16(
        const float *x, const float *gamma, const float *beta, float *y,
        size_t rows, size_t dim, float epsilon) {
    float *device_float_x = NULL, *device_float_y = NULL;
    float *device_float_gamma = NULL, *device_float_beta = NULL;
    __half *device_x = NULL, *device_y = NULL;
    __half *device_gamma = NULL, *device_beta = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t count;
    if (!x || !y || !rows || !dim || !isfinite(epsilon) || epsilon <= 0.0f)
        return UA_ERR_ARGUMENT;
    if (rows > SIZE_MAX / dim) return UA_ERR_CAPACITY;
    count = rows * dim;
    if (!fixture_count_ok(count, sizeof(float)) ||
        !fixture_count_ok(dim, sizeof(float)) ||
        !fixture_count_ok(rows, sizeof(float)))
        return UA_ERR_CAPACITY;
    if ((gamma == NULL) != (beta == NULL)) return UA_ERR_ARGUMENT;
    if (!finite_array(x, count) ||
        (gamma && (!finite_array(gamma, dim) || !finite_array(beta, dim))))
        return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_float_x, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_float_y, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_x, count * sizeof(__half)) != cudaSuccess ||
        cudaMalloc((void **)&device_y, count * sizeof(__half)) != cudaSuccess)
        goto cleanup;
    if (gamma &&
        (cudaMalloc((void **)&device_float_gamma, dim * sizeof(float)) != cudaSuccess ||
         cudaMalloc((void **)&device_float_beta, dim * sizeof(float)) != cudaSuccess ||
         cudaMalloc((void **)&device_gamma, dim * sizeof(__half)) != cudaSuccess ||
         cudaMalloc((void **)&device_beta, dim * sizeof(__half)) != cudaSuccess))
        goto cleanup;
    if (cudaMemcpy(device_float_x, x, count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (gamma &&
         (cudaMemcpy(device_float_gamma, gamma, dim * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(device_float_beta, beta, dim * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess)))
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
        device_float_x, device_x, count);
    if (gamma) {
        float_to_half_kernel<<<(unsigned)((dim + 255u) / 256u), 256>>>(
            device_float_gamma, device_gamma, dim);
        float_to_half_kernel<<<(unsigned)((dim + 255u) / 256u), 256>>>(
            device_float_beta, device_beta, dim);
    }
    layer_norm_fp16_kernel<<<(unsigned)((rows + 255u) / 256u), 256>>>(
        device_x, device_gamma, device_beta, device_y, rows, dim, epsilon,
        gamma != NULL);
    half_to_float_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
        device_y, device_float_y, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, device_float_y, count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_beta); cudaFree(device_gamma);
    cudaFree(device_y); cudaFree(device_x);
    cudaFree(device_float_beta); cudaFree(device_float_gamma);
    cudaFree(device_float_y); cudaFree(device_float_x);
    return result;
}

extern "C" ua_status ua_cuda_test_softmax_f32(
        const float *x, float *y, size_t rows, size_t cols) {
    float *device_x = NULL, *device_y = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t count;
    if (!x || !y || !rows || !cols) return UA_ERR_ARGUMENT;
    if (rows > SIZE_MAX / cols) return UA_ERR_CAPACITY;
    count = rows * cols;
    if (!fixture_count_ok(count, sizeof(float)) ||
        !fixture_count_ok(rows, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(x, count)) return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_x, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_y, count * sizeof(float)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_x, x, count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    softmax_f32_kernel<<<(unsigned)((rows + 255u) / 256u), 256>>>(
        device_x, device_y, rows, cols);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(y, device_y, count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_y); cudaFree(device_x);
    return result;
}

extern "C" ua_status ua_cuda_test_resize_bilinear_f32(
        const float *source, size_t source_height, size_t source_width,
        float *target, size_t target_height, size_t target_width) {
    float *device_source = NULL, *device_target = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t source_count, target_count;
    if (!source || !target || !source_height || !source_width ||
        !target_height || !target_width)
        return UA_ERR_ARGUMENT;
    if (source_height > SIZE_MAX / source_width ||
        target_height > SIZE_MAX / target_width)
        return UA_ERR_CAPACITY;
    source_count = source_height * source_width;
    target_count = target_height * target_width;
    if (!fixture_count_ok(source_count, sizeof(float)) ||
        !fixture_count_ok(target_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(source, source_count)) return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_source, source_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_target, target_count * sizeof(float)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_source, source, source_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    resize_bilinear_f32_kernel<<<
        (unsigned)((target_count + 255u) / 256u), 256>>>(
        device_source, source_height, source_width, device_target,
        target_height, target_width);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(target, device_target, target_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_target); cudaFree(device_source);
    return result;
}

extern "C" ua_status ua_cuda_test_deform_sample_f32(
        const float *map, size_t height, size_t width, size_t channels,
        const float *points_yx, size_t points, float *output) {
    float *device_map = NULL, *device_points = NULL, *device_output = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t map_count, output_count;
    if (!map || !points_yx || !output || !height || !width || !channels ||
        !points)
        return UA_ERR_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels ||
        points > SIZE_MAX / channels || points > SIZE_MAX / 2u)
        return UA_ERR_CAPACITY;
    map_count = height * width * channels;
    output_count = points * channels;
    if (!fixture_count_ok(map_count, sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)) ||
        !fixture_count_ok(points * 2u, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(map, map_count) || !finite_array(points_yx, points * 2u))
        return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_map, map_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_points, points * 2u * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_output, output_count * sizeof(float)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_map, map, map_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_points, points_yx, points * 2u * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    deform_sample_f32_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        device_map, height, width, channels, device_points, points,
        device_output);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, device_output, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_output); cudaFree(device_points); cudaFree(device_map);
    return result;
}

extern "C" ua_status ua_cuda_test_stable_topk_f32(
        const float *scores, size_t count, size_t k, size_t *indices) {
    float *device_scores = NULL;
    size_t *device_indices = NULL;
    ua_status result = UA_ERR_BACKEND;
    if (!scores || !indices || !count || !k || k > count)
        return UA_ERR_ARGUMENT;
    if (!fixture_count_ok(count, sizeof(float)) ||
        !fixture_count_ok(k, sizeof(size_t)))
        return UA_ERR_CAPACITY;
    if (!finite_array(scores, count)) return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_scores, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_indices, k * sizeof(size_t)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_scores, scores, count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    stable_topk_f32_kernel<<<1, 1>>>(
        device_scores, count, k, device_indices);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(indices, device_indices, k * sizeof(size_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_indices); cudaFree(device_scores);
    return result;
}

extern "C" ua_status ua_cuda_test_track_score_filter_fp16(
        const float *logits, size_t queries, size_t classes, float threshold,
        size_t capacity, float *scores, uint32_t *class_indices,
        uint32_t *selected_indices, size_t *selected_count) {
    float *device_logits_f32 = NULL, *device_scores = NULL;
    __half *device_logits = NULL;
    uint32_t *device_classes = NULL, *device_selected = NULL;
    uint32_t *device_selected_count = NULL;
    uint32_t host_selected_count = 0u;
    ua_status result = UA_ERR_BACKEND;
    size_t logit_count;
    if (!logits || !scores || !class_indices || !selected_indices ||
        !selected_count || !queries || !classes || !capacity ||
        !isfinite(threshold) || threshold < 0.0f || threshold > 1.0f ||
        queries > UINT32_MAX || classes > UINT32_MAX ||
        capacity > UINT32_MAX || queries > SIZE_MAX / classes)
        return UA_ERR_ARGUMENT;
    logit_count = queries * classes;
    if (!fixture_count_ok(logit_count, sizeof(float)) ||
        !fixture_count_ok(queries, sizeof(float)) ||
        !fixture_count_ok(capacity, sizeof(uint32_t)))
        return UA_ERR_CAPACITY;
    if (!finite_array(logits, logit_count)) return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_logits_f32,
                   logit_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_logits,
                   logit_count * sizeof(__half)) != cudaSuccess ||
        cudaMalloc((void **)&device_scores,
                   queries * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_classes,
                   queries * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_selected,
                   capacity * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_selected_count,
                   sizeof(uint32_t)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(
            device_logits_f32, logits, logit_count * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    float_to_half_kernel<<<
        (unsigned)((logit_count + 255u) / 256u), 256>>>(
        device_logits_f32, device_logits, logit_count);
    track_score_class_kernel<<<
        (unsigned)((queries + 255u) / 256u), 256>>>(
        device_logits, queries, classes, device_scores, device_classes);
    track_filter_kernel<<<1, 1>>>(
        device_scores, queries, threshold, capacity, device_selected,
        device_selected_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(scores, device_scores, queries * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(class_indices, device_classes,
                   queries * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&host_selected_count, device_selected_count,
                   sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(selected_indices, device_selected,
                   host_selected_count * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    *selected_count = host_selected_count;
    result = UA_OK;
cleanup:
    cudaFree(device_selected_count);
    cudaFree(device_selected);
    cudaFree(device_classes);
    cudaFree(device_scores);
    cudaFree(device_logits);
    cudaFree(device_logits_f32);
    return result;
}

extern "C" ua_status ua_cuda_test_tracker_state_update(
        const float *scores, int32_t *object_ids, uint32_t *disappear,
        size_t count, float score_threshold, float filter_threshold,
        uint32_t miss_tolerance, int32_t *next_object_id,
        uint32_t *active_indices, size_t active_capacity,
        size_t *active_count) {
    float *device_scores = NULL;
    int32_t *device_ids = NULL, *device_next_id = NULL;
    uint32_t *device_disappear = NULL, *device_active = NULL;
    uint32_t *device_active_count = NULL;
    uint32_t host_active_count = 0u;
    ua_status result = UA_ERR_BACKEND;
    if (!scores || !object_ids || !disappear || !next_object_id ||
        !active_indices || !active_count || !count || !active_capacity ||
        !miss_tolerance || count > UINT32_MAX ||
        active_capacity > UINT32_MAX || !isfinite(score_threshold) ||
        !isfinite(filter_threshold) || score_threshold < 0.0f ||
        score_threshold > 1.0f || filter_threshold < 0.0f ||
        filter_threshold > score_threshold ||
        !fixture_count_ok(count, sizeof(float)) ||
        !fixture_count_ok(active_capacity, sizeof(uint32_t)))
        return UA_ERR_ARGUMENT;
    if (!finite_array(scores, count)) return UA_ERR_NONFINITE;
    if (cudaMalloc(
            (void **)&device_scores, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(
            (void **)&device_ids, count * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(
            (void **)&device_disappear,
            count * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(
            (void **)&device_next_id, sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(
            (void **)&device_active,
            active_capacity * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(
            (void **)&device_active_count, sizeof(uint32_t)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(
            device_scores, scores, count * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_ids, object_ids, count * sizeof(int32_t),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_disappear, disappear, count * sizeof(uint32_t),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_next_id, next_object_id, sizeof(int32_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    tracker_state_update_kernel<<<1, 1>>>(
        device_scores, device_ids, device_disappear, count, score_threshold,
        filter_threshold, miss_tolerance, device_next_id, device_active,
        active_capacity, device_active_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(
            object_ids, device_ids, count * sizeof(int32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            disappear, device_disappear, count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            next_object_id, device_next_id, sizeof(int32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            &host_active_count, device_active_count, sizeof(uint32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            active_indices, device_active,
            host_active_count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    *active_count = host_active_count;
    result = UA_OK;
cleanup:
    cudaFree(device_active_count);
    cudaFree(device_active);
    cudaFree(device_next_id);
    cudaFree(device_disappear);
    cudaFree(device_ids);
    cudaFree(device_scores);
    return result;
}

extern "C" ua_status ua_cuda_test_memory_bank_update_fp16(
        const float *embedding, const float *scores, const float *weight,
        const float *bias, float *memory, uint8_t *padding_mask,
        uint8_t *save_period, size_t queries, size_t history,
        size_t dimensions, float threshold, uint8_t reset_period) {
    float *device_embedding_f32 = NULL, *device_weight_f32 = NULL;
    float *device_bias_f32 = NULL, *device_memory_f32 = NULL;
    float *device_scores = NULL, *device_memory_output = NULL;
    __half *device_embedding = NULL, *device_weight = NULL;
    __half *device_bias = NULL, *device_memory = NULL;
    uint8_t *device_mask = NULL, *device_period = NULL, *device_saved = NULL;
    size_t embedding_count, weight_count, memory_count;
    ua_status result = UA_ERR_BACKEND;
    if (!embedding || !scores || !weight || !bias || !memory ||
        !padding_mask || !save_period || !queries || !history ||
        !dimensions || !reset_period || !isfinite(threshold) ||
        queries > SIZE_MAX / dimensions ||
        dimensions > SIZE_MAX / dimensions ||
        queries > SIZE_MAX / history ||
        queries * history > SIZE_MAX / dimensions)
        return UA_ERR_ARGUMENT;
    embedding_count = queries * dimensions;
    weight_count = dimensions * dimensions;
    memory_count = queries * history * dimensions;
    if (!fixture_count_ok(embedding_count, sizeof(float)) ||
        !fixture_count_ok(weight_count, sizeof(float)) ||
        !fixture_count_ok(memory_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(embedding, embedding_count) ||
        !finite_array(scores, queries) ||
        !finite_array(weight, weight_count) ||
        !finite_array(bias, dimensions) ||
        !finite_array(memory, memory_count))
        return UA_ERR_NONFINITE;
#define UA_MB_MALLOC(pointer_, bytes_) \
    if (cudaMalloc((void **)&pointer_, bytes_) != cudaSuccess) goto cleanup
    UA_MB_MALLOC(device_embedding_f32, embedding_count * sizeof(float));
    UA_MB_MALLOC(device_weight_f32, weight_count * sizeof(float));
    UA_MB_MALLOC(device_bias_f32, dimensions * sizeof(float));
    UA_MB_MALLOC(device_memory_f32, memory_count * sizeof(float));
    UA_MB_MALLOC(device_scores, queries * sizeof(float));
    UA_MB_MALLOC(device_memory_output, memory_count * sizeof(float));
    UA_MB_MALLOC(device_embedding, embedding_count * sizeof(__half));
    UA_MB_MALLOC(device_weight, weight_count * sizeof(__half));
    UA_MB_MALLOC(device_bias, dimensions * sizeof(__half));
    UA_MB_MALLOC(device_memory, memory_count * sizeof(__half));
    UA_MB_MALLOC(device_mask, queries * history * sizeof(uint8_t));
    UA_MB_MALLOC(device_period, queries * sizeof(uint8_t));
    UA_MB_MALLOC(device_saved, queries * sizeof(uint8_t));
#undef UA_MB_MALLOC
    if (cudaMemcpy(
            device_embedding_f32, embedding,
            embedding_count * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_weight_f32, weight, weight_count * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_bias_f32, bias, dimensions * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_memory_f32, memory, memory_count * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_scores, scores, queries * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_mask, padding_mask, queries * history * sizeof(uint8_t),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(
            device_period, save_period, queries * sizeof(uint8_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    float_to_half_kernel<<<
        (unsigned)((embedding_count + 255u) / 256u), 256>>>(
        device_embedding_f32, device_embedding, embedding_count);
    float_to_half_kernel<<<
        (unsigned)((weight_count + 255u) / 256u), 256>>>(
        device_weight_f32, device_weight, weight_count);
    float_to_half_kernel<<<
        (unsigned)((dimensions + 255u) / 256u), 256>>>(
        device_bias_f32, device_bias, dimensions);
    float_to_half_kernel<<<
        (unsigned)((memory_count + 255u) / 256u), 256>>>(
        device_memory_f32, device_memory, memory_count);
    memory_bank_select_kernel<<<
        (unsigned)((queries + 255u) / 256u), 256>>>(
        device_scores, device_mask, device_period, device_saved, queries,
        history, threshold, reset_period);
    memory_bank_save_kernel<<<
        (unsigned)((embedding_count + 255u) / 256u), 256>>>(
        device_embedding, device_weight, device_bias, device_saved,
        device_memory, queries, history, dimensions);
    half_to_float_kernel<<<
        (unsigned)((memory_count + 255u) / 256u), 256>>>(
        device_memory, device_memory_output, memory_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(
            memory, device_memory_output, memory_count * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            padding_mask, device_mask, queries * history * sizeof(uint8_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            save_period, device_period, queries * sizeof(uint8_t),
            cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_saved); cudaFree(device_period); cudaFree(device_mask);
    cudaFree(device_memory); cudaFree(device_bias); cudaFree(device_weight);
    cudaFree(device_embedding);
    cudaFree(device_memory_output); cudaFree(device_scores);
    cudaFree(device_memory_f32); cudaFree(device_bias_f32);
    cudaFree(device_weight_f32); cudaFree(device_embedding_f32);
    return result;
}

extern "C" ua_status ua_cuda_test_ms_deform_attn_fp16(
        const float *value, size_t batches, size_t total_values,
        size_t heads, size_t channels, const uint32_t *shapes,
        const size_t *starts, size_t levels, const float *locations,
        const float *weights, size_t queries, size_t points, float *output) {
    float *float_value = NULL, *device_locations = NULL;
    float *device_weights = NULL, *device_output = NULL;
    __half *half_value = NULL;
    uint32_t *device_shapes = NULL;
    size_t *device_starts = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t value_count, sample_count, output_count, expected_total = 0;
    if (!value || !shapes || !starts || !locations || !weights || !output ||
        !batches || !total_values || !heads || !channels || !levels ||
        !queries || !points)
        return UA_ERR_ARGUMENT;
    for (size_t level = 0; level < levels; ++level) {
        size_t area;
        if (!shapes[level * 2u] || !shapes[level * 2u + 1u] ||
            (size_t)shapes[level * 2u] >
                SIZE_MAX / (size_t)shapes[level * 2u + 1u])
            return UA_ERR_CAPACITY;
        area = (size_t)shapes[level * 2u] * shapes[level * 2u + 1u];
        if (starts[level] != expected_total || area > SIZE_MAX - expected_total)
            return UA_ERR_ARGUMENT;
        expected_total += area;
    }
    if (expected_total != total_values ||
        batches > SIZE_MAX / total_values ||
        batches * total_values > SIZE_MAX / heads ||
        batches * total_values * heads > SIZE_MAX / channels ||
        batches > SIZE_MAX / queries ||
        batches * queries > SIZE_MAX / heads ||
        batches * queries * heads > SIZE_MAX / levels ||
        batches * queries * heads * levels > SIZE_MAX / points ||
        batches * queries * heads > SIZE_MAX / channels)
        return UA_ERR_CAPACITY;
    value_count = batches * total_values * heads * channels;
    sample_count = batches * queries * heads * levels * points;
    output_count = batches * queries * heads * channels;
    if (!fixture_count_ok(value_count, sizeof(float)) ||
        !fixture_count_ok(sample_count, 2u * sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)) ||
        !fixture_count_ok(levels, 2u * sizeof(uint32_t)))
        return UA_ERR_CAPACITY;
    if (!finite_array(value, value_count) ||
        !finite_array(locations, sample_count * 2u) ||
        !finite_array(weights, sample_count))
        return UA_ERR_NONFINITE;
#define UA_MS_ALLOC(pointer, count, type) \
    if (cudaMalloc((void **)&(pointer), (count) * sizeof(type)) != cudaSuccess) \
        goto cleanup
    UA_MS_ALLOC(float_value, value_count, float);
    UA_MS_ALLOC(half_value, value_count, __half);
    UA_MS_ALLOC(device_shapes, levels * 2u, uint32_t);
    UA_MS_ALLOC(device_starts, levels, size_t);
    UA_MS_ALLOC(device_locations, sample_count * 2u, float);
    UA_MS_ALLOC(device_weights, sample_count, float);
    UA_MS_ALLOC(device_output, output_count, float);
    if (cudaMemcpy(float_value, value, value_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_shapes, shapes, levels * 2u * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_starts, starts, levels * sizeof(size_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_locations, locations, sample_count * 2u * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_weights, weights, sample_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    float_to_half_kernel<<<(unsigned)((value_count + 255u) / 256u), 256>>>(
        float_value, half_value, value_count);
    ms_deform_attn_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        half_value, total_values, heads, channels, device_shapes,
        device_starts, levels, device_locations, device_weights, queries,
        points, output_count, device_output);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, device_output, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_output); cudaFree(device_weights);
    cudaFree(device_locations); cudaFree(device_starts);
    cudaFree(device_shapes); cudaFree(half_value); cudaFree(float_value);
    return result;
#undef UA_MS_ALLOC
}

extern "C" ua_status ua_cuda_test_deform_locations_f32(
        const float *reference, size_t reference_dims, const float *offset,
        const uint32_t *shapes, size_t batches, size_t queries, size_t heads,
        size_t levels, size_t points, float *locations) {
    float *device_reference = NULL, *device_offset = NULL;
    float *device_locations = NULL;
    uint32_t *device_shapes = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t reference_count, location_count;
    if (!reference || !offset || !shapes || !locations || !batches ||
        !queries || !heads || !levels || !points ||
        (reference_dims != 2u && reference_dims != 4u))
        return UA_ERR_ARGUMENT;
    if (batches > SIZE_MAX / queries ||
        batches * queries > SIZE_MAX / levels ||
        batches * queries * levels > SIZE_MAX / reference_dims ||
        batches * queries > SIZE_MAX / heads ||
        batches * queries * heads > SIZE_MAX / levels ||
        batches * queries * heads * levels > SIZE_MAX / points ||
        batches * queries * heads * levels * points > SIZE_MAX / 2u)
        return UA_ERR_CAPACITY;
    reference_count = batches * queries * levels * reference_dims;
    location_count = batches * queries * heads * levels * points * 2u;
    if (!fixture_count_ok(reference_count, sizeof(float)) ||
        !fixture_count_ok(location_count, sizeof(float)) ||
        !fixture_count_ok(levels, 2u * sizeof(uint32_t)))
        return UA_ERR_CAPACITY;
    for (size_t level = 0; level < levels; ++level)
        if (!shapes[level * 2u] || !shapes[level * 2u + 1u])
            return UA_ERR_ARGUMENT;
    if (!finite_array(reference, reference_count) ||
        !finite_array(offset, location_count))
        return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_reference,
                   reference_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_offset,
                   location_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_locations,
                   location_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_shapes,
                   levels * 2u * sizeof(uint32_t)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_reference, reference,
                   reference_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_offset, offset, location_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_shapes, shapes, levels * 2u * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    deform_locations_f32_kernel<<<
        (unsigned)((location_count + 255u) / 256u), 256>>>(
        device_reference, reference_dims, device_offset, device_shapes,
        queries, heads, levels, points, location_count, device_locations);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(locations, device_locations, location_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_shapes); cudaFree(device_locations);
    cudaFree(device_offset); cudaFree(device_reference);
    return result;
}

extern "C" ua_status ua_cuda_test_camera_scatter_average_f32(
        const float *slots, const uint32_t *indices, const uint32_t *counts,
        size_t cameras, size_t max_queries, size_t total_queries,
        size_t dimensions, float *output) {
    float *device_slots = NULL, *device_output = NULL;
    uint32_t *device_indices = NULL, *device_counts = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t slot_count, output_count;
    if (!slots || !indices || !counts || !output || !cameras ||
        !max_queries || !total_queries || !dimensions)
        return UA_ERR_ARGUMENT;
    if (cameras > SIZE_MAX / max_queries ||
        cameras * max_queries > SIZE_MAX / dimensions ||
        total_queries > SIZE_MAX / dimensions)
        return UA_ERR_CAPACITY;
    slot_count = cameras * max_queries * dimensions;
    output_count = total_queries * dimensions;
    if (!fixture_count_ok(slot_count, sizeof(float)) ||
        !fixture_count_ok(output_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(slots, slot_count)) return UA_ERR_NONFINITE;
    for (size_t camera = 0; camera < cameras; ++camera) {
        if (counts[camera] > max_queries) return UA_ERR_ARGUMENT;
        for (size_t slot = 0; slot < counts[camera]; ++slot)
            if (indices[camera * max_queries + slot] >= total_queries)
                return UA_ERR_ARGUMENT;
    }
    if (cudaMalloc((void **)&device_slots, slot_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_indices,
                   cameras * max_queries * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_counts,
                   cameras * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_output,
                   output_count * sizeof(float)) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_slots, slots, slot_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_indices, indices,
                   cameras * max_queries * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_counts, counts, cameras * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    camera_scatter_average_f32_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        device_slots, device_indices, device_counts, cameras, max_queries,
        total_queries, dimensions, output_count, device_output);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, device_output, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_output); cudaFree(device_counts);
    cudaFree(device_indices); cudaFree(device_slots);
    return result;
}

extern "C" ua_status ua_cuda_test_queue_mean_f32(
        const float *queue, size_t batches, size_t queries, size_t dimensions,
        size_t queue_length, float *output) {
    float *device_queue = NULL, *device_output = NULL;
    ua_status result = UA_ERR_BACKEND;
    size_t output_count, queue_count;
    if (!queue || !output || !batches || !queries || !dimensions ||
        !queue_length)
        return UA_ERR_ARGUMENT;
    if (batches > SIZE_MAX / queries ||
        batches * queries > SIZE_MAX / dimensions ||
        batches * queries * dimensions > SIZE_MAX / queue_length)
        return UA_ERR_CAPACITY;
    output_count = batches * queries * dimensions;
    queue_count = output_count * queue_length;
    if (!fixture_count_ok(output_count, sizeof(float)) ||
        !fixture_count_ok(queue_count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (!finite_array(queue, queue_count)) return UA_ERR_NONFINITE;
    if (cudaMalloc((void **)&device_queue, queue_count * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc((void **)&device_output, output_count * sizeof(float)) !=
            cudaSuccess)
        goto cleanup;
    if (cudaMemcpy(device_queue, queue, queue_count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        goto cleanup;
    queue_mean_f32_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256>>>(
        device_queue, queries, dimensions, queue_length, output_count,
        device_output);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, device_output, output_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    result = UA_OK;
cleanup:
    cudaFree(device_output); cudaFree(device_queue);
    return result;
}

extern "C" ua_status ua_cuda_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? UA_OK : UA_ERR_BACKEND;
}

extern "C" ua_status ua_cuda_demo(const float *input, size_t n, float *output) {
    float *dx = NULL, *dy = NULL;
    if (cudaMalloc((void **)&dx, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&dy, sizeof(float)) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_MEMORY;
    }
    if (cudaMemcpy(dx, input, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_BACKEND;
    }
    boundary_kernel<<<1, 1>>>(dx, n, dy);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, dy, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_BACKEND;
    }
    cudaFree(dx); cudaFree(dy); return UA_OK;
}

extern "C" ua_status ua_cuda_production_create(
        const void *weights, size_t bytes, void **output, size_t *device_bytes) {
    production_context *c;
    if (!weights || !bytes || !output || !device_bytes) return UA_ERR_ARGUMENT;
    *output = NULL; *device_bytes = 0;
    c = (production_context *)calloc(1, sizeof(*c));
    if (!c) return UA_ERR_MEMORY;
    if (cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&c->weights, bytes) != cudaSuccess ||
        cudaMalloc(&c->camera_raw, PROD_CAMERA_RAW_BYTES) != cudaSuccess ||
        cudaMalloc(&c->metadata, sizeof(production_metadata)) != cudaSuccess ||
        cudaMalloc(&c->boundary_samples,
                   PROD_BOUNDARY_SAMPLE_VALUES * sizeof(__half)) != cudaSuccess ||
        cudaMalloc((void **)&c->stage_status, PROD_STAGE_STATUS_BYTES) !=
            cudaSuccess ||
        cudaMalloc(&c->previous_bev, PROD_PREVIOUS_BEV_BYTES) != cudaSuccess ||
        cudaMalloc(&c->aligned_previous_bev, PROD_PREVIOUS_BEV_BYTES) !=
            cudaSuccess ||
        cudaMalloc(
            &c->track_state_committed, PROD_TRACK_STATE_BYTES) != cudaSuccess ||
        cudaMalloc(
            &c->track_state_candidate, PROD_TRACK_STATE_BYTES) != cudaSuccess ||
        cudaMalloc(&c->arena, PROD_ARENA_BYTES) != cudaSuccess ||
        cudaMemcpyAsync(c->weights, weights, bytes, cudaMemcpyHostToDevice,
                        c->stream) != cudaSuccess ||
        cudaMemsetAsync(c->previous_bev, 0, PROD_PREVIOUS_BEV_BYTES,
                        c->stream) != cudaSuccess ||
        cudaMemsetAsync(c->aligned_previous_bev, 0, PROD_PREVIOUS_BEV_BYTES,
                        c->stream) != cudaSuccess ||
        cudaMemsetAsync(
            c->track_state_committed, 0, PROD_TRACK_STATE_BYTES,
            c->stream) != cudaSuccess ||
        cudaMemsetAsync(
            c->track_state_candidate, 0, PROD_TRACK_STATE_BYTES,
            c->stream) != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess) {
        cudaFree(c->arena);
        cudaFree(c->track_state_candidate);
        cudaFree(c->track_state_committed);
        cudaFree(c->aligned_previous_bev);
        cudaFree(c->previous_bev); cudaFree(c->weights);
        cudaFree(c->stage_status); cudaFree(c->boundary_samples);
        cudaFree(c->metadata);
        cudaFree(c->camera_raw);
        if (c->stream) cudaStreamDestroy(c->stream);
        free(c); return UA_ERR_MEMORY;
    }
    c->weights_bytes = bytes;
    c->layer3_first_nonfinite = 23;
    c->layer4_first_nonfinite = 3;
    *device_bytes = bytes + PROD_CAMERA_RAW_BYTES + sizeof(production_metadata) +
                    PROD_BOUNDARY_SAMPLE_VALUES * sizeof(__half) +
                    PROD_STAGE_STATUS_BYTES +
                    2u * PROD_PREVIOUS_BEV_BYTES +
                    2u * PROD_TRACK_STATE_BYTES + PROD_ARENA_BYTES;
    *output = c;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_preprocess(
        void *context, const ua_production_input *input, size_t *h2d_bytes,
        const void **normalized_fp16) {
    production_context *c = (production_context *)context;
    const size_t row_bytes = PROD_IMAGE_WIDTH * 3u;
    const size_t camera_bytes = row_bytes * PROD_IMAGE_HEIGHT;
    const size_t plane = PROD_PAD_HEIGHT * PROD_IMAGE_WIDTH;
    production_metadata metadata;
    if (!c || !input || !input->scene_token || !h2d_bytes ||
        !normalized_fp16)
        return UA_ERR_ARGUMENT;
    *h2d_bytes = 0;
    *normalized_fp16 = NULL;
    {
        size_t scene_length = strnlen(input->scene_token, 129u);
        if (scene_length == 129u) return UA_ERR_ARGUMENT;
        if (!c->has_scene || strcmp(c->scene_token, input->scene_token)) {
            if (cudaMemsetAsync(c->previous_bev, 0, PROD_PREVIOUS_BEV_BYTES,
                                c->stream) != cudaSuccess ||
                cudaMemsetAsync(
                    c->aligned_previous_bev, 0, PROD_PREVIOUS_BEV_BYTES,
                    c->stream) != cudaSuccess ||
                cudaMemsetAsync(
                    c->track_state_committed, 0, PROD_TRACK_STATE_BYTES,
                    c->stream) != cudaSuccess ||
                cudaMemsetAsync(
                    c->track_state_candidate, 0, PROD_TRACK_STATE_BYTES,
                    c->stream) != cudaSuccess)
                return UA_ERR_BACKEND;
            memcpy(c->scene_token, input->scene_token, scene_length + 1u);
            c->has_scene = 1;
            c->previous_bev_valid = 0;
        }
    }
    c->temporal_shift_x = 0.0f;
    c->temporal_shift_y = 0.0f;
    if (c->previous_bev_valid) {
        float delta_x =
            input->ego_pose[0] * input->can_bus[0] +
            input->ego_pose[4] * input->can_bus[1] +
            input->ego_pose[8] * input->can_bus[2];
        float delta_y =
            input->ego_pose[1] * input->can_bus[0] +
            input->ego_pose[5] * input->can_bus[1] +
            input->ego_pose[9] * input->can_bus[2];
        c->temporal_shift_x = delta_x / 102.4f;
        c->temporal_shift_y = delta_y / 102.4f;
        align_previous_bev_nearest_kernel<<<
            (unsigned)((40000u * 256u + 255u) / 256u), 256, 0,
            c->stream>>>(
            (const __half *)c->previous_bev,
            (__half *)c->aligned_previous_bev, input->can_bus[17]);
        if (cudaGetLastError() != cudaSuccess) return UA_ERR_BACKEND;
    }
    for (size_t camera = 0; camera < UA_CAMERA_COUNT; ++camera) {
        const ua_bgr_image_view *view = &input->cameras[camera];
        if (!view->data || view->width != PROD_IMAGE_WIDTH ||
            view->height != PROD_IMAGE_HEIGHT ||
            view->row_stride_bytes < row_bytes)
            return UA_ERR_ARGUMENT;
        uint8_t *target =
            (uint8_t *)c->camera_raw + camera * camera_bytes;
        if (cudaMemcpy2DAsync(
                target, row_bytes, view->data, view->row_stride_bytes,
                row_bytes, PROD_IMAGE_HEIGHT, cudaMemcpyHostToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        float3 mean = make_float3(103.530f, 116.280f, 123.675f);
        float3 inverse = make_float3(1.0f, 1.0f, 1.0f);
        __half *output = (__half *)c->arena + camera * 3u * plane;
        size_t camera_output_count = 3u * plane;
        preprocess_bgr_kernel<<<
            (unsigned)((camera_output_count + 255u) / 256u), 256, 0,
            c->stream>>>(
            target, PROD_IMAGE_WIDTH, PROD_IMAGE_HEIGHT, row_bytes,
            PROD_IMAGE_WIDTH, PROD_PAD_HEIGHT, mean, inverse, output);
        if (cudaGetLastError() != cudaSuccess) return UA_ERR_BACKEND;
        *h2d_bytes += camera_bytes;
    }
    memset(&metadata, 0, sizeof(metadata));
    metadata.timestamp_us = input->timestamp_us;
    metadata.navigation_command = input->navigation_command;
    metadata.scene_hash = UINT64_C(14695981039346656037);
    for (const unsigned char *p =
             (const unsigned char *)input->scene_token; *p; ++p)
        metadata.scene_hash = (metadata.scene_hash ^ *p) *
                              UINT64_C(1099511628211);
    memcpy(metadata.camera_intrinsics, input->camera_intrinsics,
           sizeof(metadata.camera_intrinsics));
    memcpy(metadata.camera_to_ego, input->camera_to_ego,
           sizeof(metadata.camera_to_ego));
    memcpy(metadata.ego_pose, input->ego_pose, sizeof(metadata.ego_pose));
    memcpy(metadata.can_bus, input->can_bus, sizeof(metadata.can_bus));
    if (cudaMemcpyAsync(c->metadata, &metadata, sizeof(metadata),
                        cudaMemcpyHostToDevice, c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    *h2d_bytes += sizeof(metadata);
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) return UA_ERR_BACKEND;
    *normalized_fp16 = c->arena;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_resnet_stem(
        void *context, const void *normalized_fp16,
        const void *conv_weight_fp16, const void *bn_gamma_fp16,
        const void *bn_beta_fp16, const void *bn_mean_fp32,
        const void *bn_variance_fp32, const void **pooled_fp16) {
    production_context *c = (production_context *)context;
    size_t stem_count =
        UA_CAMERA_COUNT * PROD_STEM_CHANNELS * PROD_STEM_HEIGHT * PROD_STEM_WIDTH;
    size_t pool_count =
        UA_CAMERA_COUNT * PROD_STEM_CHANNELS * PROD_POOL_HEIGHT * PROD_POOL_WIDTH;
    if (!c || normalized_fp16 != c->arena || !conv_weight_fp16 ||
        !bn_gamma_fp16 || !bn_beta_fp16 || !bn_mean_fp32 ||
        !bn_variance_fp32 || !pooled_fp16)
        return UA_ERR_ARGUMENT;
    if (PROD_NORMALIZED_BYTES + PROD_STEM_BYTES + PROD_POOL_BYTES >
        PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    __half *stem = (__half *)((unsigned char *)c->arena + PROD_NORMALIZED_BYTES);
    __half *pool = (__half *)((unsigned char *)stem + PROD_STEM_BYTES);
    conv2d_fp16_kernel<<<
        (unsigned)((stem_count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)normalized_fp16, (const __half *)conv_weight_fp16,
        NULL, stem, UA_CAMERA_COUNT, 3u, PROD_PAD_HEIGHT, PROD_IMAGE_WIDTH,
        PROD_STEM_CHANNELS, PROD_STEM_HEIGHT, PROD_STEM_WIDTH, 7u, 7u, 2u,
        2u, 3u, 3u, 0);
    batchnorm_relu_fp16_kernel<<<
        (unsigned)((stem_count + 255u) / 256u), 256, 0, c->stream>>>(
        stem, bn_gamma_fp16, bn_beta_fp16,
        (const float *)bn_mean_fp32, (const float *)bn_variance_fp32, stem,
        PROD_STEM_CHANNELS, PROD_STEM_HEIGHT * PROD_STEM_WIDTH, 1e-5f,
        stem_count, 1);
    maxpool2d_fp16_kernel<<<
        (unsigned)((pool_count + 255u) / 256u), 256, 0, c->stream>>>(
        stem, pool, PROD_STEM_CHANNELS, PROD_STEM_HEIGHT, PROD_STEM_WIDTH,
        PROD_POOL_HEIGHT, PROD_POOL_WIDTH, 3u, 3u, 2u, 2u, 1u, 1u,
        pool_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    *pooled_fp16 = pool;
    c->last_resnet_stem = pool;
    c->last_resnet_stem_count = pool_count;
    c->last_resnet_layer1 = NULL;
    c->last_resnet_layer1_count = 0;
    c->last_resnet_layer2 = NULL;
    c->last_resnet_layer2_count = 0;
    c->last_resnet_layer3 = NULL;
    c->last_resnet_layer3_count = 0;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    for (size_t level = 0; level < 4u; ++level) {
        c->last_fpn[level] = NULL;
        c->last_fpn_count[level] = 0;
    }
    c->last_bevformer_flatten = NULL;
    c->last_bevformer_flatten_count = 0;
    c->last_bev_queries = NULL;
    c->last_bev_pos = NULL;
    c->last_reference_2d = NULL;
    c->last_reference_3d = NULL;
    c->last_reference_camera = NULL;
    c->last_visibility = NULL;
    c->last_visible_indices = NULL;
    c->last_visible_counts = NULL;
    c->last_temporal_attention0 = NULL;
    c->last_encoder_norm0 = NULL;
    c->last_spatial_attention0 = NULL;
    c->last_encoder_norm1 = NULL;
    c->last_encoder_ffn0 = NULL;
    c->last_encoder_norm2 = NULL;
    c->encoder_layers_completed = 0u;
    c->last_track_query_pos = NULL;
    c->last_track_query = NULL;
    c->initial_track_reference_points = NULL;
    c->last_track_reference_points = NULL;
    c->last_track_decoder_self = NULL;
    c->last_track_decoder_norm0 = NULL;
    c->last_track_decoder_cross = NULL;
    c->last_track_decoder_norm1 = NULL;
    c->last_track_decoder_ffn = NULL;
    c->last_track_decoder_norm2 = NULL;
    c->track_decoder_states = NULL;
    c->track_regressions = NULL;
    c->track_references = NULL;
    c->last_track_class_logits = NULL;
    c->last_track_boxes = NULL;
    c->last_track_past_trajectory = NULL;
    c->last_track_scores = NULL;
    c->last_track_classes = NULL;
    c->last_track_selected_indices = NULL;
    c->last_track_selected_count = NULL;
    memset(&c->query_interaction_weights, 0,
           sizeof(c->query_interaction_weights));
    c->query_interaction_weights_valid = 0;
    c->last_query_interaction_output = NULL;
    c->last_query_interaction_queries = 0u;
    c->track_decoder_layers_completed = 0u;
    c->track_reference_layers_completed = 0u;
    c->track_output_heads_completed = 0;
    return UA_OK;
}

static int valid_conv_bn(const ua_cuda_conv_bn_weights *weights) {
    return weights && weights->weight_fp16 && weights->gamma_fp16 &&
           weights->beta_fp16 && weights->mean_fp32 &&
           weights->variance_fp32;
}

static void launch_conv_bn_relu(
        cudaStream_t stream, const __half *input, __half *output,
        const ua_cuda_conv_bn_weights *weights, size_t batches,
        size_t input_channels, size_t output_channels, size_t height,
        size_t width, size_t kernel, size_t stride, size_t padding) {
    size_t output_height = (height + 2u * padding - kernel) / stride + 1u;
    size_t output_width = (width + 2u * padding - kernel) / stride + 1u;
    size_t count = batches * output_channels * output_height * output_width;
    conv2d_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, stream>>>(
        input, (const __half *)weights->weight_fp16, NULL, output, batches,
        input_channels, height, width, output_channels, output_height,
        output_width, kernel, kernel, stride, stride, padding, padding, 0);
    batchnorm_relu_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, stream>>>(
        output, weights->gamma_fp16, weights->beta_fp16,
        (const float *)weights->mean_fp32,
        (const float *)weights->variance_fp32, output, output_channels,
        output_height * output_width, 1e-5f, count, 1);
}

extern "C" ua_status ua_cuda_production_resnet_layer1(
        void *context, const void *stem_fp16,
        const ua_cuda_bottleneck_weights blocks[3], const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t batches = UA_CAMERA_COUNT, height = 232u, width = 400u;
    const size_t mid_channels = 64u, output_channels = 256u;
    const size_t mid_bytes =
        batches * mid_channels * height * width * sizeof(__half);
    const size_t output_bytes =
        batches * output_channels * height * width * sizeof(__half);
    if (!c || stem_fp16 != c->last_resnet_stem || !blocks || !output_fp16)
        return UA_ERR_ARGUMENT;
    for (size_t block = 0; block < 3u; ++block) {
        if (!valid_conv_bn(&blocks[block].conv1) ||
            !valid_conv_bn(&blocks[block].conv2) ||
            !valid_conv_bn(&blocks[block].conv3) ||
            (block == 0u &&
             (!blocks[block].has_downsample ||
              !valid_conv_bn(&blocks[block].downsample))) ||
            (block != 0u && blocks[block].has_downsample))
            return UA_ERR_ARGUMENT;
    }
    __half *scratch0 = (__half *)c->arena;
    __half *scratch1 = (__half *)((unsigned char *)c->arena + mid_bytes);
    __half *layer_output =
        (__half *)((unsigned char *)c->arena + 2u * mid_bytes);
    __half *relocated_input =
        (__half *)((unsigned char *)c->arena + PROD_ARENA_BYTES -
                   PROD_POOL_BYTES);
    if (2u * mid_bytes + output_bytes >
            PROD_ARENA_BYTES - PROD_POOL_BYTES)
        return UA_ERR_MEMORY;
    if (cudaMemcpyAsync(relocated_input, stem_fp16, PROD_POOL_BYTES,
                        cudaMemcpyDeviceToDevice, c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    launch_conv_bn_relu(
        c->stream, relocated_input, scratch0, &blocks[0].conv1, batches, 64u,
        mid_channels, height, width, 1u, 1u, 0u);
    launch_conv_bn_relu(
        c->stream, scratch0, scratch1, &blocks[0].conv2, batches,
        mid_channels, mid_channels, height, width, 3u, 1u, 1u);
    {
        size_t count = batches * output_channels * height * width;
        bottleneck_projection_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[0].conv3.weight_fp16,
            (const __half *)blocks[0].conv3.gamma_fp16,
            (const __half *)blocks[0].conv3.beta_fp16,
            (const float *)blocks[0].conv3.mean_fp32,
            (const float *)blocks[0].conv3.variance_fp32, relocated_input,
            (const __half *)blocks[0].downsample.weight_fp16,
            (const __half *)blocks[0].downsample.gamma_fp16,
            (const __half *)blocks[0].downsample.beta_fp16,
            (const float *)blocks[0].downsample.mean_fp32,
            (const float *)blocks[0].downsample.variance_fp32, layer_output,
            batches, 64u, mid_channels, output_channels, height, width, height,
            width, 1u);
        if (cudaMemcpyAsync(
                c->boundary_samples, layer_output + 92800u,
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
    }
    for (size_t block = 1; block < 3u; ++block) {
        launch_conv_bn_relu(
            c->stream, layer_output, scratch0, &blocks[block].conv1, batches,
            output_channels, mid_channels, height, width, 1u, 1u, 0u);
        launch_conv_bn_relu(
            c->stream, scratch0, scratch1, &blocks[block].conv2, batches,
            mid_channels, mid_channels, height, width, 3u, 1u, 1u);
        size_t count = batches * output_channels * height * width;
        bottleneck_identity_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[block].conv3.weight_fp16,
            (const __half *)blocks[block].conv3.gamma_fp16,
            (const __half *)blocks[block].conv3.beta_fp16,
            (const float *)blocks[block].conv3.mean_fp32,
            (const float *)blocks[block].conv3.variance_fp32, layer_output,
            batches, output_channels, mid_channels, height, width);
        if (cudaMemcpyAsync(
                c->boundary_samples + block * 32u,
                layer_output + (block < 2u ? 92800u : 0u),
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_resnet_layer1 = layer_output;
    c->last_resnet_layer1_count =
        batches * output_channels * height * width;
    c->last_resnet_layer2 = NULL;
    c->last_resnet_layer2_count = 0;
    c->last_resnet_layer3 = NULL;
    c->last_resnet_layer3_count = 0;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    c->last_resnet_stem = NULL;
    c->last_resnet_stem_count = 0;
    *output_fp16 = layer_output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_resnet_layer2(
        void *context, const void *layer1_fp16,
        const ua_cuda_bottleneck_weights blocks[4], const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t batches = UA_CAMERA_COUNT;
    const size_t input_height = 232u, input_width = 400u;
    const size_t output_height = 116u, output_width = 200u;
    const size_t input_channels = 256u, mid_channels = 128u;
    const size_t output_channels = 512u;
    const size_t input_bytes =
        batches * input_channels * input_height * input_width * sizeof(__half);
    const size_t mid_bytes =
        batches * mid_channels * output_height * output_width * sizeof(__half);
    __half *expected_input;
    __half *layer_output;
    __half *scratch0;
    __half *scratch1;
    static const size_t diagnostic_offsets[4] = {197u, 23198u, 797u, 2u};
    if (!c) return UA_ERR_ARGUMENT;
    expected_input =
        (__half *)((unsigned char *)c->arena +
                   2u * UA_CAMERA_COUNT * 64u * 232u * 400u *
                       sizeof(__half));
    layer_output = (__half *)c->arena;
    scratch0 = (__half *)((unsigned char *)expected_input + input_bytes);
    scratch1 = (__half *)((unsigned char *)scratch0 + mid_bytes);
    if (layer1_fp16 != c->last_resnet_layer1 ||
        layer1_fp16 != expected_input || !blocks || !output_fp16)
        return UA_ERR_ARGUMENT;
    if ((unsigned char *)scratch1 + mid_bytes >
        (unsigned char *)c->arena + PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    for (size_t block = 0; block < 4u; ++block) {
        if (!valid_conv_bn(&blocks[block].conv1) ||
            !valid_conv_bn(&blocks[block].conv2) ||
            !valid_conv_bn(&blocks[block].conv3) ||
            (block == 0u &&
             (!blocks[block].has_downsample ||
              !valid_conv_bn(&blocks[block].downsample))) ||
            (block != 0u && blocks[block].has_downsample))
            return UA_ERR_ARGUMENT;
    }
    launch_conv_bn_relu(
        c->stream, (const __half *)layer1_fp16, scratch0, &blocks[0].conv1,
        batches, input_channels, mid_channels, input_height, input_width, 1u,
        2u, 0u);
    launch_conv_bn_relu(
        c->stream, scratch0, scratch1, &blocks[0].conv2, batches,
        mid_channels, mid_channels, output_height, output_width, 3u, 1u, 1u);
    {
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_projection_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[0].conv3.weight_fp16,
            (const __half *)blocks[0].conv3.gamma_fp16,
            (const __half *)blocks[0].conv3.beta_fp16,
            (const float *)blocks[0].conv3.mean_fp32,
            (const float *)blocks[0].conv3.variance_fp32,
            (const __half *)layer1_fp16,
            (const __half *)blocks[0].downsample.weight_fp16,
            (const __half *)blocks[0].downsample.gamma_fp16,
            (const __half *)blocks[0].downsample.beta_fp16,
            (const float *)blocks[0].downsample.mean_fp32,
            (const float *)blocks[0].downsample.variance_fp32, layer_output,
            batches, input_channels, mid_channels, output_channels,
            input_height, input_width, output_height, output_width, 2u);
        if (cudaMemcpyAsync(
                c->boundary_samples + 3u * 32u,
                layer_output + diagnostic_offsets[0],
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
    }
    for (size_t block = 1; block < 4u; ++block) {
        launch_conv_bn_relu(
            c->stream, layer_output, scratch0, &blocks[block].conv1, batches,
            output_channels, mid_channels, output_height, output_width, 1u,
            1u, 0u);
        launch_conv_bn_relu(
            c->stream, scratch0, scratch1, &blocks[block].conv2, batches,
            mid_channels, mid_channels, output_height, output_width, 3u, 1u,
            1u);
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_identity_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[block].conv3.weight_fp16,
            (const __half *)blocks[block].conv3.gamma_fp16,
            (const __half *)blocks[block].conv3.beta_fp16,
            (const float *)blocks[block].conv3.mean_fp32,
            (const float *)blocks[block].conv3.variance_fp32, layer_output,
            batches, output_channels, mid_channels, output_height,
            output_width);
        if (cudaMemcpyAsync(
                c->boundary_samples + (3u + block) * 32u,
                layer_output + diagnostic_offsets[block],
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
    }
    if (cudaMemcpyAsync(
            c->boundary_samples + 7u * 32u, layer_output,
            32u * sizeof(__half), cudaMemcpyDeviceToDevice,
            c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_resnet_layer2 = layer_output;
    c->last_resnet_layer2_count =
        batches * output_channels * output_height * output_width;
    c->last_resnet_layer1 = NULL;
    c->last_resnet_layer1_count = 0;
    c->last_resnet_layer3 = NULL;
    c->last_resnet_layer3_count = 0;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    *output_fp16 = layer_output;
    return UA_OK;
}

static void launch_dcn_bn_relu(
        cudaStream_t stream, const __half *input, __half *output,
        float *offset_and_mask, const ua_cuda_bottleneck_weights *block,
        size_t batches, size_t channels, size_t input_height,
        size_t input_width, size_t output_height, size_t output_width,
        size_t stride) {
    size_t offset_count = batches * 27u * output_height * output_width;
    size_t output_count =
        batches * channels * output_height * output_width;
    dcn_offset_mask_fp32_kernel<<<
        (unsigned)((offset_count + 255u) / 256u), 256, 0, stream>>>(
        input, (const __half *)block->conv2_offset_weight_fp16,
        (const __half *)block->conv2_offset_bias_fp16, offset_and_mask,
        batches, channels, input_height, input_width, output_height,
        output_width, stride);
    modulated_deform_conv2d_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256, 0, stream>>>(
        input, offset_and_mask, offset_and_mask,
        (const __half *)block->conv2.weight_fp16, NULL, output, batches,
        channels, input_height, input_width, channels, output_height,
        output_width, 3u, 3u, stride, stride, 1u, 1u, 1u, 1u, 0, 1);
    batchnorm_relu_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256, 0, stream>>>(
        output, block->conv2.gamma_fp16, block->conv2.beta_fp16,
        (const float *)block->conv2.mean_fp32,
        (const float *)block->conv2.variance_fp32, output, channels,
        output_height * output_width, 1e-5f, output_count, 1);
}

extern "C" ua_status ua_cuda_production_resnet_layer3(
        void *context, const void *layer2_fp16,
        const ua_cuda_bottleneck_weights blocks[23], const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t batches = UA_CAMERA_COUNT;
    const size_t input_height = 116u, input_width = 200u;
    const size_t output_height = 58u, output_width = 100u;
    const size_t input_channels = 512u, mid_channels = 256u;
    const size_t output_channels = 1024u;
    const size_t input_bytes =
        batches * input_channels * input_height * input_width * sizeof(__half);
    const size_t mid_bytes =
        batches * mid_channels * output_height * output_width * sizeof(__half);
    const size_t offset_bytes =
        batches * 27u * output_height * output_width * sizeof(float);
    const size_t output_bytes =
        batches * output_channels * output_height * output_width *
        sizeof(__half);
    __half *scratch0;
    __half *scratch1;
    float *offset_and_mask;
    __half *layer_output;
    int first_nonfinite = 23;
    if (!c || layer2_fp16 != c->last_resnet_layer2 ||
        layer2_fp16 != c->arena || !blocks || !output_fp16)
        return UA_ERR_ARGUMENT;
    for (size_t block = 0; block < 23u; ++block) {
        if (!valid_conv_bn(&blocks[block].conv1) ||
            !valid_conv_bn(&blocks[block].conv2) ||
            !valid_conv_bn(&blocks[block].conv3) ||
            !blocks[block].conv2_is_dcn ||
            !blocks[block].conv2_offset_weight_fp16 ||
            !blocks[block].conv2_offset_bias_fp16 ||
            (block == 0u &&
             (!blocks[block].has_downsample ||
              !valid_conv_bn(&blocks[block].downsample))) ||
            (block != 0u && blocks[block].has_downsample))
            return UA_ERR_ARGUMENT;
    }
    scratch0 = (__half *)((unsigned char *)c->arena + input_bytes);
    offset_and_mask =
        (float *)((unsigned char *)scratch0 + mid_bytes);
    scratch1 =
        (__half *)((unsigned char *)offset_and_mask + offset_bytes);
    layer_output = (__half *)((unsigned char *)scratch1 + mid_bytes);
    if ((unsigned char *)layer_output + output_bytes >
        (unsigned char *)c->arena + PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    if (cudaMemcpyAsync(
            c->stage_status, &first_nonfinite, sizeof(first_nonfinite),
            cudaMemcpyHostToDevice, c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    launch_conv_bn_relu(
        c->stream, (const __half *)layer2_fp16, scratch0, &blocks[0].conv1,
        batches, input_channels, mid_channels, input_height, input_width, 1u,
        2u, 0u);
    launch_dcn_bn_relu(
        c->stream, scratch0, scratch1, offset_and_mask, &blocks[0], batches,
        mid_channels, output_height, output_width, output_height, output_width,
        1u);
    {
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_projection_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[0].conv3.weight_fp16,
            (const __half *)blocks[0].conv3.gamma_fp16,
            (const __half *)blocks[0].conv3.beta_fp16,
            (const float *)blocks[0].conv3.mean_fp32,
            (const float *)blocks[0].conv3.variance_fp32,
            (const __half *)layer2_fp16,
            (const __half *)blocks[0].downsample.weight_fp16,
            (const __half *)blocks[0].downsample.gamma_fp16,
            (const __half *)blocks[0].downsample.beta_fp16,
            (const float *)blocks[0].downsample.mean_fp32,
            (const float *)blocks[0].downsample.variance_fp32, layer_output,
            batches, input_channels, mid_channels, output_channels,
            input_height, input_width, output_height, output_width, 2u);
        if (cudaMemcpyAsync(
                c->boundary_samples + 8u * 32u, layer_output + 5697u,
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        record_nonfinite_half_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            layer_output, count, 0, c->stage_status);
    }
    for (size_t block = 1; block < 23u; ++block) {
        launch_conv_bn_relu(
            c->stream, layer_output, scratch0, &blocks[block].conv1, batches,
            output_channels, mid_channels, output_height, output_width, 1u,
            1u, 0u);
        launch_dcn_bn_relu(
            c->stream, scratch0, scratch1, offset_and_mask, &blocks[block],
            batches, mid_channels, output_height, output_width, output_height,
            output_width, 1u);
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_identity_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[block].conv3.weight_fp16,
            (const __half *)blocks[block].conv3.gamma_fp16,
            (const __half *)blocks[block].conv3.beta_fp16,
            (const float *)blocks[block].conv3.mean_fp32,
            (const float *)blocks[block].conv3.variance_fp32, layer_output,
            batches, output_channels, mid_channels, output_height,
            output_width);
        record_nonfinite_half_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            layer_output, count, (int)block, c->stage_status);
    }
    if (cudaMemcpyAsync(
            c->boundary_samples + 9u * 32u, layer_output + 195u,
            32u * sizeof(__half), cudaMemcpyDeviceToDevice,
            c->stream) != cudaSuccess ||
        cudaMemcpyAsync(
            &first_nonfinite, c->stage_status, sizeof(first_nonfinite),
            cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_resnet_layer3 = layer_output;
    c->last_resnet_layer3_count =
        batches * output_channels * output_height * output_width;
    c->layer3_first_nonfinite = first_nonfinite;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    *output_fp16 = layer_output;
    return first_nonfinite < 23 ? UA_ERR_NONFINITE : UA_OK;
}

extern "C" ua_status ua_cuda_production_resnet_layer4(
        void *context, const void *layer3_fp16,
        const ua_cuda_bottleneck_weights blocks[3], const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t batches = UA_CAMERA_COUNT;
    const size_t input_height = 58u, input_width = 100u;
    const size_t output_height = 29u, output_width = 50u;
    const size_t input_channels = 1024u, mid_channels = 512u;
    const size_t output_channels = 2048u;
    const size_t layer2_bytes =
        batches * 512u * 116u * 200u * sizeof(__half);
    const size_t input_bytes =
        batches * input_channels * input_height * input_width * sizeof(__half);
    const size_t mid_bytes =
        batches * mid_channels * output_height * output_width * sizeof(__half);
    const size_t offset_bytes =
        batches * 27u * output_height * output_width * sizeof(float);
    const size_t output_bytes =
        batches * output_channels * output_height * output_width *
        sizeof(__half);
    __half *expected_input;
    __half *scratch0;
    __half *scratch1;
    float *offset_and_mask;
    __half *layer_output;
    int first_nonfinite = 3;
    if (!c || !blocks || !output_fp16) return UA_ERR_ARGUMENT;
    expected_input =
        (__half *)((unsigned char *)c->arena + 181934400u);
    if (layer3_fp16 != c->last_resnet_layer3 ||
        layer3_fp16 != expected_input)
        return UA_ERR_ARGUMENT;
    for (size_t block = 0; block < 3u; ++block) {
        if (!valid_conv_bn(&blocks[block].conv1) ||
            !valid_conv_bn(&blocks[block].conv2) ||
            !valid_conv_bn(&blocks[block].conv3) ||
            !blocks[block].conv2_is_dcn ||
            !blocks[block].conv2_offset_weight_fp16 ||
            !blocks[block].conv2_offset_bias_fp16 ||
            (block == 0u &&
             (!blocks[block].has_downsample ||
              !valid_conv_bn(&blocks[block].downsample))) ||
            (block != 0u && blocks[block].has_downsample))
            return UA_ERR_ARGUMENT;
    }
    scratch0 = (__half *)((unsigned char *)c->arena + layer2_bytes);
    offset_and_mask =
        (float *)((unsigned char *)scratch0 + mid_bytes);
    scratch1 =
        (__half *)((unsigned char *)offset_and_mask + offset_bytes);
    layer_output = (__half *)((unsigned char *)expected_input + input_bytes);
    if ((unsigned char *)scratch1 + mid_bytes >
            (unsigned char *)expected_input ||
        (unsigned char *)layer_output + output_bytes >
            (unsigned char *)c->arena + PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    if (cudaMemcpyAsync(
            c->stage_status, &first_nonfinite, sizeof(first_nonfinite),
            cudaMemcpyHostToDevice, c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    launch_conv_bn_relu(
        c->stream, (const __half *)layer3_fp16, scratch0, &blocks[0].conv1,
        batches, input_channels, mid_channels, input_height, input_width, 1u,
        2u, 0u);
    launch_dcn_bn_relu(
        c->stream, scratch0, scratch1, offset_and_mask, &blocks[0], batches,
        mid_channels, output_height, output_width, output_height, output_width,
        1u);
    {
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_projection_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[0].conv3.weight_fp16,
            (const __half *)blocks[0].conv3.gamma_fp16,
            (const __half *)blocks[0].conv3.beta_fp16,
            (const float *)blocks[0].conv3.mean_fp32,
            (const float *)blocks[0].conv3.variance_fp32,
            (const __half *)layer3_fp16,
            (const __half *)blocks[0].downsample.weight_fp16,
            (const __half *)blocks[0].downsample.gamma_fp16,
            (const __half *)blocks[0].downsample.beta_fp16,
            (const float *)blocks[0].downsample.mean_fp32,
            (const float *)blocks[0].downsample.variance_fp32, layer_output,
            batches, input_channels, mid_channels, output_channels,
            input_height, input_width, output_height, output_width, 2u);
        if (cudaMemcpyAsync(
                c->boundary_samples + 10u * 32u, layer_output,
                32u * sizeof(__half), cudaMemcpyDeviceToDevice,
                c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        record_nonfinite_half_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            layer_output, count, 0, c->stage_status);
    }
    for (size_t block = 1; block < 3u; ++block) {
        launch_conv_bn_relu(
            c->stream, layer_output, scratch0, &blocks[block].conv1, batches,
            output_channels, mid_channels, output_height, output_width, 1u,
            1u, 0u);
        launch_dcn_bn_relu(
            c->stream, scratch0, scratch1, offset_and_mask, &blocks[block],
            batches, mid_channels, output_height, output_width, output_height,
            output_width, 1u);
        size_t count =
            batches * output_channels * output_height * output_width;
        bottleneck_identity_final_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            scratch1, (const __half *)blocks[block].conv3.weight_fp16,
            (const __half *)blocks[block].conv3.gamma_fp16,
            (const __half *)blocks[block].conv3.beta_fp16,
            (const float *)blocks[block].conv3.mean_fp32,
            (const float *)blocks[block].conv3.variance_fp32, layer_output,
            batches, output_channels, mid_channels, output_height,
            output_width);
        record_nonfinite_half_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            layer_output, count, (int)block, c->stage_status);
    }
    if (cudaMemcpyAsync(
            c->boundary_samples + 11u * 32u, layer_output + 147u,
            32u * sizeof(__half), cudaMemcpyDeviceToDevice,
            c->stream) != cudaSuccess ||
        cudaMemcpyAsync(
            &first_nonfinite, c->stage_status, sizeof(first_nonfinite),
            cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_resnet_layer4 = layer_output;
    c->last_resnet_layer4_count =
        batches * output_channels * output_height * output_width;
    c->layer4_first_nonfinite = first_nonfinite;
    *output_fp16 = layer_output;
    return first_nonfinite < 3 ? UA_ERR_NONFINITE : UA_OK;
}

static int valid_conv_bias(const ua_cuda_conv_bias_weights *weights) {
    return weights && weights->weight_fp16 && weights->bias_fp16;
}

static void launch_conv_bias(
        cudaStream_t stream, const __half *input, __half *output,
        const ua_cuda_conv_bias_weights *weights, size_t batches,
        size_t input_channels, size_t output_channels, size_t input_height,
        size_t input_width, size_t kernel, size_t stride, size_t padding) {
    size_t output_height =
        (input_height + 2u * padding - kernel) / stride + 1u;
    size_t output_width =
        (input_width + 2u * padding - kernel) / stride + 1u;
    size_t count =
        batches * output_channels * output_height * output_width;
    conv2d_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, stream>>>(
        input, (const __half *)weights->weight_fp16,
        (const __half *)weights->bias_fp16, output, batches, input_channels,
        input_height, input_width, output_channels, output_height,
        output_width, kernel, kernel, stride, stride, padding, padding, 1);
}

extern "C" ua_status ua_cuda_production_fpn(
        void *context, const void *layer2_fp16, const void *layer3_fp16,
        const void *layer4_fp16, const ua_cuda_conv_bias_weights lateral[3],
        const ua_cuda_conv_bias_weights output_convs[4],
        const void *outputs_fp16[4]) {
    production_context *c = (production_context *)context;
    const size_t batches = UA_CAMERA_COUNT, channels = 256u;
    const size_t heights[4] = {116u, 58u, 29u, 15u};
    const size_t widths[4] = {200u, 100u, 50u, 25u};
    const size_t input_channels[3] = {512u, 1024u, 2048u};
    const __half *inputs[3] = {
        (const __half *)layer2_fp16, (const __half *)layer3_fp16,
        (const __half *)layer4_fp16
    };
    __half *laterals[3];
    __half *fpn_outputs[4];
    size_t lateral_base = 288840000u;
    size_t offset = lateral_base;
    if (!c || !outputs_fp16 || layer2_fp16 != c->last_resnet_layer2 ||
        layer3_fp16 != c->last_resnet_layer3 ||
        layer4_fp16 != c->last_resnet_layer4)
        return UA_ERR_ARGUMENT;
    for (size_t level = 0; level < 3u; ++level)
        if (!valid_conv_bias(&lateral[level]) ||
            !valid_conv_bias(&output_convs[level]))
            return UA_ERR_ARGUMENT;
    if (!valid_conv_bias(&output_convs[3])) return UA_ERR_ARGUMENT;
    for (size_t level = 0; level < 3u; ++level) {
        size_t bytes =
            batches * channels * heights[level] * widths[level] *
            sizeof(__half);
        laterals[level] = (__half *)((unsigned char *)c->arena + offset);
        offset += bytes;
    }
    if (offset > PROD_ARENA_BYTES) return UA_ERR_MEMORY;
    for (size_t level = 0; level < 3u; ++level)
        launch_conv_bias(
            c->stream, inputs[level], laterals[level], &lateral[level],
            batches, input_channels[level], channels, heights[level],
            widths[level], 1u, 1u, 0u);
    for (size_t level = 2u; level > 0u; --level) {
        size_t count =
            batches * channels * heights[level - 1u] * widths[level - 1u];
        nearest_upsample_add_fp16_kernel<<<
            (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
            laterals[level], laterals[level - 1u], batches, channels,
            heights[level], widths[level], heights[level - 1u],
            widths[level - 1u]);
    }
    offset = 0;
    for (size_t level = 0; level < 4u; ++level) {
        size_t bytes =
            batches * channels * heights[level] * widths[level] *
            sizeof(__half);
        fpn_outputs[level] =
            (__half *)((unsigned char *)c->arena + offset);
        offset += bytes;
    }
    for (size_t level = 0; level < 3u; ++level)
        launch_conv_bias(
            c->stream, laterals[level], fpn_outputs[level],
            &output_convs[level], batches, channels, channels,
            heights[level], widths[level], 3u, 1u, 1u);
    launch_conv_bias(
        c->stream, fpn_outputs[2], fpn_outputs[3], &output_convs[3],
        batches, channels, channels, heights[2], widths[2], 3u, 2u, 1u);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    for (size_t level = 0; level < 4u; ++level) {
        c->last_fpn[level] = fpn_outputs[level];
        c->last_fpn_count[level] =
            batches * channels * heights[level] * widths[level];
        outputs_fp16[level] = fpn_outputs[level];
    }
    c->last_resnet_layer2 = NULL;
    c->last_resnet_layer2_count = 0;
    c->last_resnet_layer3 = NULL;
    c->last_resnet_layer3_count = 0;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_bevformer_flatten(
        void *context, const void *fpn_outputs_fp16[4],
        const void *camera_embeds_fp32, const void *level_embeds_fp32,
        const void **flattened_fp16) {
    production_context *c = (production_context *)context;
    const size_t count = UA_CAMERA_COUNT * 30825u * 256u;
    const size_t output_offset = 94694400u;
    __half *output;
    if (!c || !fpn_outputs_fp16 || !camera_embeds_fp32 ||
        !level_embeds_fp32 || !flattened_fp16)
        return UA_ERR_ARGUMENT;
    for (size_t level = 0; level < 4u; ++level)
        if (fpn_outputs_fp16[level] != c->last_fpn[level])
            return UA_ERR_ARGUMENT;
    output = (__half *)((unsigned char *)c->arena + output_offset);
    if (output_offset + count * sizeof(__half) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    for (size_t level = 0; level < 4u; ++level)
        if (cudaMemcpyAsync(
                c->boundary_samples + (12u + level) * 32u,
                c->last_fpn[level], 32u * sizeof(__half),
                cudaMemcpyDeviceToDevice, c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
    bevformer_flatten_embed_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)fpn_outputs_fp16[0],
        (const __half *)fpn_outputs_fp16[1],
        (const __half *)fpn_outputs_fp16[2],
        (const __half *)fpn_outputs_fp16[3],
        (const float *)camera_embeds_fp32,
        (const float *)level_embeds_fp32, output, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_bevformer_flatten = output;
    c->last_bevformer_flatten_count = count;
    *flattened_fp16 = output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_prepare_bev(
        void *context, const void *bev_query_weight_fp16,
        const void *col_embed_fp16, const void *row_embed_fp16,
        const ua_cuda_can_bus_weights *can_bus_weights,
        const void **bev_queries_fp16, const void **bev_pos_fp16) {
    production_context *c = (production_context *)context;
    const size_t count = 40000u * 256u;
    const size_t query_offset = 189388800u;
    const size_t position_offset = query_offset + count * sizeof(__half);
    const size_t can_bus_offset = position_offset + count * sizeof(__half);
    __half *queries;
    __half *positions;
    float *can_bus;
    if (!c || !bev_query_weight_fp16 || !col_embed_fp16 ||
        !row_embed_fp16 || !can_bus_weights || !bev_queries_fp16 ||
        !bev_pos_fp16 || !can_bus_weights->linear0_weight_fp32 ||
        !can_bus_weights->linear0_bias_fp32 ||
        !can_bus_weights->linear1_weight_fp32 ||
        !can_bus_weights->linear1_bias_fp32 ||
        !can_bus_weights->norm_weight_fp32 ||
        !can_bus_weights->norm_bias_fp32)
        return UA_ERR_ARGUMENT;
    if (!c->last_bevformer_flatten) return UA_ERR_PROFILE;
    if (can_bus_offset + 256u * sizeof(float) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    queries = (__half *)((unsigned char *)c->arena + query_offset);
    positions = (__half *)((unsigned char *)c->arena + position_offset);
    can_bus = (float *)((unsigned char *)c->arena + can_bus_offset);
    can_bus_mlp_layernorm_kernel<<<1, 1, 0, c->stream>>>(
        (const production_metadata *)c->metadata,
        (const float *)can_bus_weights->linear0_weight_fp32,
        (const float *)can_bus_weights->linear0_bias_fp32,
        (const float *)can_bus_weights->linear1_weight_fp32,
        (const float *)can_bus_weights->linear1_bias_fp32,
        (const float *)can_bus_weights->norm_weight_fp32,
        (const float *)can_bus_weights->norm_bias_fp32, can_bus);
    bev_query_can_bus_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)bev_query_weight_fp16, can_bus, queries, count);
    bev_learned_position_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)col_embed_fp16, (const __half *)row_embed_fp16,
        positions, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_bev_queries = queries;
    c->last_bev_pos = positions;
    *bev_queries_fp16 = queries;
    *bev_pos_fp16 = positions;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_prepare_bev_geometry(
        void *context, const void **reference_2d_fp16,
        const void **reference_3d_fp16, const void **reference_camera_fp32,
        const void **visibility_u8, const void **visible_indices_u32,
        const void **visible_counts_u32) {
    production_context *c = (production_context *)context;
    const size_t reference_2d_count = 40000u * 2u;
    const size_t reference_3d_count = 4u * 40000u * 3u;
    const size_t camera_count = UA_CAMERA_COUNT * 40000u * 4u;
    size_t offset = 230350080u;
    __half *reference_2d;
    __half *reference_3d;
    float *reference_camera;
    uint8_t *visibility;
    uint32_t *visible_indices;
    uint32_t *visible_counts;
    if (!c || !reference_2d_fp16 || !reference_3d_fp16 ||
        !reference_camera_fp32 || !visibility_u8 || !visible_indices_u32 ||
        !visible_counts_u32)
        return UA_ERR_ARGUMENT;
    if (!c->last_bev_queries || !c->last_bev_pos)
        return UA_ERR_PROFILE;
    reference_2d = (__half *)((unsigned char *)c->arena + offset);
    offset += reference_2d_count * sizeof(__half);
    reference_3d = (__half *)((unsigned char *)c->arena + offset);
    offset += reference_3d_count * sizeof(__half);
    reference_camera = (float *)((unsigned char *)c->arena + offset);
    offset += camera_count * 2u * sizeof(float);
    visibility = (uint8_t *)c->arena + offset;
    offset += camera_count * sizeof(uint8_t);
    visible_indices = (uint32_t *)((unsigned char *)c->arena + offset);
    offset += UA_CAMERA_COUNT * 40000u * sizeof(uint32_t);
    visible_counts = (uint32_t *)((unsigned char *)c->arena + offset);
    offset += UA_CAMERA_COUNT * sizeof(uint32_t);
    if (offset > PROD_ARENA_BYTES) return UA_ERR_MEMORY;
    bev_reference_points_kernel<<<
        (unsigned)((reference_3d_count + 255u) / 256u), 256, 0,
        c->stream>>>(reference_2d, reference_3d);
    bev_camera_projection_kernel<<<
        (unsigned)((camera_count + 255u) / 256u), 256, 0, c->stream>>>(
        reference_3d, (const production_metadata *)c->metadata,
        reference_camera, visibility, camera_count);
    compact_visible_queries_kernel<<<1, UA_CAMERA_COUNT, 0, c->stream>>>(
        visibility, visible_indices, visible_counts);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_reference_2d = reference_2d;
    c->last_reference_3d = reference_3d;
    c->last_reference_camera = reference_camera;
    c->last_visibility = visibility;
    c->last_visible_indices = visible_indices;
    c->last_visible_counts = visible_counts;
    *reference_2d_fp16 = reference_2d;
    *reference_3d_fp16 = reference_3d;
    *reference_camera_fp32 = reference_camera;
    *visibility_u8 = visibility;
    *visible_indices_u32 = visible_indices;
    *visible_counts_u32 = visible_counts;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_encoder_temporal(
        void *context, size_t layer, const void *query_fp16,
        const ua_cuda_temporal_attention_weights *weights,
        const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 40000u, dimensions = 256u;
    const size_t value_count = rows * dimensions;
    size_t offset = 0u;
    __half *projected_value;
    __half *sampling_offsets;
    __half *attention_weights;
    __half *sampled;
    __half *output;
    const __half *query = (const __half *)query_fp16;
    const __half *history;
    const __half *current_value;
    if (!c || layer >= PROD_ENCODER_LAYERS || !query || !weights ||
        !output_fp16 ||
        !c->last_bev_pos || !c->last_reference_2d ||
        !weights->value_weight_fp16 || !weights->value_bias_fp16 ||
        !weights->offset_weight_fp16 || !weights->offset_bias_fp16 ||
        !weights->attention_weight_fp16 || !weights->attention_bias_fp16 ||
        !weights->output_weight_fp16 || !weights->output_bias_fp16)
        return UA_ERR_ARGUMENT;
    history = c->previous_bev_valid
        ? (const __half *)c->aligned_previous_bev : query;
    current_value = c->previous_bev_valid
        ? c->last_bev_queries : query;
    projected_value = (__half *)((unsigned char *)c->arena + offset);
    offset += value_count * sizeof(__half);
    offset += value_count * sizeof(__half);
    offset = 241070336u;
    sampling_offsets = (__half *)((unsigned char *)c->arena + offset);
    offset += rows * 128u * sizeof(__half);
    attention_weights = (__half *)((unsigned char *)c->arena + offset);
    offset += rows * 64u * sizeof(__half);
    sampled = (__half *)((unsigned char *)c->arena + offset);
    offset += value_count * sizeof(__half);
    output = (__half *)((unsigned char *)c->arena + offset);
    offset += value_count * sizeof(__half);
    if (offset > PROD_ARENA_BYTES) return UA_ERR_MEMORY;
    linear_fp16_kernel<<<
        (unsigned)((value_count + 255u) / 256u), 256, 0, c->stream>>>(
        history,
        (const __half *)weights->value_weight_fp16,
        (const __half *)weights->value_bias_fp16, projected_value, rows,
        dimensions, dimensions, 1);
    linear_fp16_kernel<<<
        (unsigned)((value_count + 255u) / 256u), 256, 0, c->stream>>>(
        current_value,
        (const __half *)weights->value_weight_fp16,
        (const __half *)weights->value_bias_fp16,
        projected_value + value_count, rows, dimensions, dimensions, 1);
    temporal_concat_linear_kernel<<<
        (unsigned)((rows * 128u + 255u) / 256u), 256, 0, c->stream>>>(
        history, query, c->last_bev_pos,
        (const __half *)weights->offset_weight_fp16,
        (const __half *)weights->offset_bias_fp16, sampling_offsets, 128u);
    temporal_concat_linear_kernel<<<
        (unsigned)((rows * 64u + 255u) / 256u), 256, 0, c->stream>>>(
        history, query, c->last_bev_pos,
        (const __half *)weights->attention_weight_fp16,
        (const __half *)weights->attention_bias_fp16, attention_weights,
        64u);
    temporal_attention_softmax_kernel<<<
        (unsigned)((rows * 8u * 2u + 255u) / 256u), 256, 0,
        c->stream>>>(attention_weights);
    temporal_deform_sample_kernel<<<
        (unsigned)((value_count + 255u) / 256u), 256, 0, c->stream>>>(
        projected_value, sampling_offsets, attention_weights,
        c->last_reference_2d, c->temporal_shift_x,
        c->temporal_shift_y, sampled);
    linear_residual_fp16_kernel<<<
        (unsigned)((value_count + 255u) / 256u), 256, 0, c->stream>>>(
        sampled, (const __half *)weights->output_weight_fp16,
        (const __half *)weights->output_bias_fp16, query,
        output, rows, dimensions);
    if (snapshot_encoder_boundary(c, layer, 0u, output) != UA_OK ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_temporal_attention0 = output;
    c->last_encoder_norm0 = NULL;
    *output_fp16 = output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_encoder_norm_after_temporal(
        void *context, size_t layer, const void *gamma_fp16,
        const void *beta_fp16, const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 40000u, dimensions = 256u;
    const size_t output_count = rows * dimensions;
    const size_t offset = 317870336u;
    __half *output;
    if (!c || layer >= PROD_ENCODER_LAYERS || !gamma_fp16 || !beta_fp16 ||
        !output_fp16 ||
        !c->last_temporal_attention0)
        return UA_ERR_ARGUMENT;
    if (offset + output_count * sizeof(__half) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    output = (__half *)((unsigned char *)c->arena + offset);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        c->last_temporal_attention0, (const __half *)gamma_fp16,
        (const __half *)beta_fp16, output, rows, dimensions, 1e-5f, 1);
    if (snapshot_encoder_boundary(c, layer, 1u, output) != UA_OK ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_encoder_norm0 = output;
    c->last_spatial_attention0 = NULL;
    *output_fp16 = output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_encoder_spatial(
        void *context, size_t layer,
        const ua_cuda_spatial_attention_weights *weights,
        const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t visual_rows = UA_CAMERA_COUNT * 30825u;
    const size_t query_count = 40000u * 256u;
    __half *projected_value;
    __half *offsets;
    __half *attention;
    __half *sampled;
    float *slots;
    __half *output;
    if (!c || layer >= PROD_ENCODER_LAYERS || !weights || !output_fp16 ||
        !c->last_bevformer_flatten ||
        !c->last_encoder_norm0 || !c->last_reference_camera ||
        !c->last_visibility || !c->last_visible_indices ||
        !c->last_visible_counts || !weights->value_weight_fp16 ||
        !weights->value_bias_fp16 || !weights->offset_weight_fp16 ||
        !weights->offset_bias_fp16 || !weights->attention_weight_fp16 ||
        !weights->attention_bias_fp16 || !weights->output_weight_fp16 ||
        !weights->output_bias_fp16)
        return UA_ERR_ARGUMENT;
    projected_value = (__half *)((unsigned char *)c->arena);
    offsets = (__half *)((unsigned char *)c->arena + 241070336u);
    attention = (__half *)((unsigned char *)c->arena + 282030336u);
    sampled = (__half *)((unsigned char *)c->arena + 338350336u);
    slots = (float *)((unsigned char *)c->arena + 358830336u);
    output = (__half *)((unsigned char *)c->arena + 399790336u);
    if (399790336u + query_count * sizeof(__half) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    linear_fp16_kernel<<<
        (unsigned)((visual_rows * 256u + 255u) / 256u), 256, 0,
        c->stream>>>(
        c->last_bevformer_flatten,
        (const __half *)weights->value_weight_fp16,
        (const __half *)weights->value_bias_fp16, projected_value,
        visual_rows, 256u, 256u, 1);
    (void)cudaMemsetAsync(slots, 0, query_count * sizeof(float), c->stream);
    for (size_t camera = 0; camera < UA_CAMERA_COUNT; ++camera) {
        spatial_compact_linear_kernel<<<
            (unsigned)((40000u * 512u + 255u) / 256u), 256, 0,
            c->stream>>>(
            c->last_encoder_norm0, c->last_visible_indices,
            c->last_visible_counts, camera,
            (const __half *)weights->offset_weight_fp16,
            (const __half *)weights->offset_bias_fp16, offsets, 512u);
        spatial_compact_linear_kernel<<<
            (unsigned)((40000u * 256u + 255u) / 256u), 256, 0,
            c->stream>>>(
            c->last_encoder_norm0, c->last_visible_indices,
            c->last_visible_counts, camera,
            (const __half *)weights->attention_weight_fp16,
            (const __half *)weights->attention_bias_fp16, attention, 256u);
        spatial_attention_softmax_kernel<<<
            (unsigned)((40000u * 8u + 255u) / 256u), 256, 0,
            c->stream>>>(attention, c->last_visible_counts, camera);
        spatial_deform_sample_kernel<<<
            (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
            projected_value, offsets, attention, c->last_reference_camera,
            c->last_visible_indices, c->last_visible_counts, camera, sampled);
        spatial_scatter_kernel<<<
            (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
            sampled, c->last_visible_indices, c->last_visible_counts,
            camera, slots);
    }
    spatial_output_projection_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        slots, c->last_visibility,
        (const __half *)weights->output_weight_fp16,
        (const __half *)weights->output_bias_fp16, c->last_encoder_norm0,
        output);
    if (snapshot_encoder_boundary(c, layer, 2u, output) != UA_OK ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_spatial_attention0 = output;
    c->last_encoder_norm1 = NULL;
    *output_fp16 = output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_encoder_norm_after_spatial(
        void *context, size_t layer, const void *gamma_fp16,
        const void *beta_fp16, const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 40000u, dimensions = 256u;
    const size_t offset = 420270336u;
    __half *output;
    if (!c || layer >= PROD_ENCODER_LAYERS || !gamma_fp16 || !beta_fp16 ||
        !output_fp16 ||
        !c->last_spatial_attention0)
        return UA_ERR_ARGUMENT;
    if (offset + rows * dimensions * sizeof(__half) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    output = (__half *)((unsigned char *)c->arena + offset);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        c->last_spatial_attention0, (const __half *)gamma_fp16,
        (const __half *)beta_fp16, output, rows, dimensions, 1e-5f, 1);
    if (snapshot_encoder_boundary(c, layer, 3u, output) != UA_OK ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_encoder_norm1 = output;
    c->last_encoder_ffn0 = NULL;
    c->last_encoder_norm2 = NULL;
    *output_fp16 = output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_encoder_ffn(
        void *context, size_t layer,
        const ua_cuda_encoder_ffn_weights *weights,
        const void **ffn_output_fp16, const void **norm2_output_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 40000u;
    const size_t hidden_count = rows * 512u;
    const size_t output_count = rows * 256u;
    __half *hidden;
    __half *ffn_output;
    __half *norm_output;
    if (!c || layer >= PROD_ENCODER_LAYERS || !weights ||
        !ffn_output_fp16 || !norm2_output_fp16 ||
        !c->last_encoder_norm1 || !weights->linear0_weight_fp16 ||
        !weights->linear0_bias_fp16 || !weights->linear1_weight_fp16 ||
        !weights->linear1_bias_fp16 || !weights->norm_weight_fp16 ||
        !weights->norm_bias_fp16)
        return UA_ERR_ARGUMENT;
    hidden = (__half *)((unsigned char *)c->arena + 440750336u);
    ffn_output = (__half *)((unsigned char *)c->arena + 481710336u);
    norm_output = (__half *)((unsigned char *)c->arena + 502190336u);
    if (502190336u + output_count * sizeof(__half) > PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    linear_fp16_kernel<<<
        (unsigned)((hidden_count + 255u) / 256u), 256, 0, c->stream>>>(
        c->last_encoder_norm1,
        (const __half *)weights->linear0_weight_fp16,
        (const __half *)weights->linear0_bias_fp16, hidden, rows, 256u,
        512u, 1);
    relu_fp16_kernel<<<
        (unsigned)((hidden_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, hidden_count);
    linear_rect_residual_fp16_kernel<<<
        (unsigned)((output_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, (const __half *)weights->linear1_weight_fp16,
        (const __half *)weights->linear1_bias_fp16, c->last_encoder_norm1,
        ffn_output, rows, 512u, 256u);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        ffn_output, (const __half *)weights->norm_weight_fp16,
        (const __half *)weights->norm_bias_fp16, norm_output, rows, 256u,
        1e-5f, 1);
    if (snapshot_encoder_boundary(c, layer, 4u, ffn_output) != UA_OK ||
        snapshot_encoder_boundary(c, layer, 5u, norm_output) != UA_OK ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_encoder_ffn0 = ffn_output;
    c->last_encoder_norm2 = norm_output;
    c->encoder_layers_completed = layer + 1u;
    *ffn_output_fp16 = ffn_output;
    *norm2_output_fp16 = norm_output;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_commit_previous_bev(
        void *context, const void *encoder_output_fp16) {
    production_context *c = (production_context *)context;
    if (!c || !encoder_output_fp16 || c->encoder_layers_completed != 6u)
        return UA_ERR_ARGUMENT;
    if (cudaMemcpyAsync(
            c->previous_bev, encoder_output_fp16, PROD_PREVIOUS_BEV_BYTES,
            cudaMemcpyDeviceToDevice, c->stream) != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->previous_bev_valid = 1;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_prepare_track_queries(
        void *context, const void *query_embedding_fp16,
        const void *reference_weight_fp32, const void *reference_bias_fp32,
        const void **query_pos_fp16, const void **query_fp16,
        const void **reference_points_fp32) {
    production_context *c = (production_context *)context;
    const size_t query_count = 901u * 256u;
    __half *query_pos;
    __half *query;
    float *reference_points;
    if (!c || c->encoder_layers_completed != 6u ||
        !query_embedding_fp16 || !reference_weight_fp32 ||
        !reference_bias_fp32 || !query_pos_fp16 || !query_fp16 ||
        !reference_points_fp32)
        return UA_ERR_ARGUMENT;
    query_pos = (__half *)c->arena;
    query = query_pos + query_count;
    reference_points = (float *)(query + query_count);
    if ((unsigned char *)(reference_points + 901u * 3u) >
        (unsigned char *)c->arena + PROD_ARENA_BYTES)
        return UA_ERR_MEMORY;
    split_track_query_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)query_embedding_fp16, query_pos, query);
    track_reference_points_kernel<<<
        (unsigned)((901u * 3u + 255u) / 256u), 256, 0, c->stream>>>(
        query_pos, (const float *)reference_weight_fp32,
        (const float *)reference_bias_fp32, reference_points);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_track_query_pos = query_pos;
    c->last_track_query = query;
    c->initial_track_reference_points = reference_points;
    c->last_track_reference_points = reference_points;
    c->track_decoder_states = (__half *)(
        (unsigned char *)c->arena + PROD_TRACK_STATE_OFFSET);
    c->track_regressions = (__half *)(
        (unsigned char *)c->arena + PROD_TRACK_REGRESSION_OFFSET);
    c->track_references = (float *)(
        (unsigned char *)c->arena + PROD_TRACK_REFERENCE_OFFSET);
    c->track_decoder_layers_completed = 0u;
    c->track_reference_layers_completed = 0u;
    c->last_track_class_logits = NULL;
    c->last_track_boxes = NULL;
    c->last_track_past_trajectory = NULL;
    c->last_track_scores = NULL;
    c->last_track_classes = NULL;
    c->last_track_selected_indices = NULL;
    c->last_track_selected_count = NULL;
    c->last_query_interaction_output = NULL;
    c->last_query_interaction_queries = 0u;
    c->track_output_heads_completed = 0;
    *query_pos_fp16 = query_pos;
    *query_fp16 = query;
    *reference_points_fp32 = reference_points;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_track_decoder_layer(
        void *context, size_t layer, const void *bev_fp16,
        const ua_cuda_track_decoder_layer_weights *weights,
        const void **output_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 901u, dimensions = 256u;
    const size_t query_count = rows * dimensions;
    const __half *input;
    __half *qkv, *attended, *self_output, *norm0, *projected_bev;
    __half *offsets, *attention, *sampled, *cross_output, *norm1;
    __half *hidden, *ffn_output, *norm2;
    if (!c || layer >= 6u || !bev_fp16 || !weights || !output_fp16 ||
        !c->last_track_query_pos || !c->last_track_query ||
        !c->last_track_reference_points || !weights->self_in_weight_fp16 ||
        !weights->self_in_bias_fp16 || !weights->self_out_weight_fp16 ||
        !weights->self_out_bias_fp16 || !weights->norm0_weight_fp16 ||
        !weights->norm0_bias_fp16 || !weights->cross_value_weight_fp16 ||
        !weights->cross_value_bias_fp16 || !weights->cross_offset_weight_fp16 ||
        !weights->cross_offset_bias_fp16 ||
        !weights->cross_attention_weight_fp16 ||
        !weights->cross_attention_bias_fp16 ||
        !weights->cross_out_weight_fp16 || !weights->cross_out_bias_fp16 ||
        !weights->norm1_weight_fp16 || !weights->norm1_bias_fp16 ||
        !weights->ffn.linear0_weight_fp16 ||
        !weights->ffn.linear0_bias_fp16 ||
        !weights->ffn.linear1_weight_fp16 ||
        !weights->ffn.linear1_bias_fp16 || !weights->ffn.norm_weight_fp16 ||
        !weights->ffn.norm_bias_fp16)
        return UA_ERR_ARGUMENT;
    qkv = (__half *)((unsigned char *)c->arena + 4194304u);
    attended = (__half *)((unsigned char *)c->arena + 6291456u);
    self_output = (__half *)((unsigned char *)c->arena + 7340032u);
    norm0 = (__half *)((unsigned char *)c->arena + 8388608u);
    projected_bev = (__half *)((unsigned char *)c->arena + 16777216u);
    offsets = (__half *)((unsigned char *)c->arena + 37748736u);
    attention = (__half *)((unsigned char *)c->arena + 38010880u);
    sampled = (__half *)((unsigned char *)c->arena + 38273024u);
    cross_output = (__half *)((unsigned char *)c->arena + 39321600u);
    norm1 = (__half *)((unsigned char *)c->arena + 40370176u);
    hidden = (__half *)((unsigned char *)c->arena + 41943040u);
    ffn_output = (__half *)((unsigned char *)c->arena + 44040192u);
    norm2 = (__half *)((unsigned char *)c->arena + 45088768u);
    input = layer == 0u
        ? c->last_track_query
        : c->track_decoder_states + (layer - 1u) * query_count;
    if (!input || (layer > 0u &&
                   (c->track_decoder_layers_completed != layer ||
                    c->track_reference_layers_completed != layer)))
        return UA_ERR_PROFILE;
    track_qkv_linear_kernel<<<
        (unsigned)((rows * 768u + 255u) / 256u), 256, 0, c->stream>>>(
        input, c->last_track_query_pos,
        (const __half *)weights->self_in_weight_fp16,
        (const __half *)weights->self_in_bias_fp16, qkv, rows);
    track_self_attention_kernel<<<
        (unsigned)((rows * 8u + 255u) / 256u), 256, 0, c->stream>>>(
        qkv, attended, rows);
    linear_residual_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        attended, (const __half *)weights->self_out_weight_fp16,
        (const __half *)weights->self_out_bias_fp16, input, self_output,
        rows, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        self_output, (const __half *)weights->norm0_weight_fp16,
        (const __half *)weights->norm0_bias_fp16, norm0, rows, dimensions,
        1e-5f, 1);
    linear_fp16_kernel<<<
        (unsigned)((40000u * 256u + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)bev_fp16,
        (const __half *)weights->cross_value_weight_fp16,
        (const __half *)weights->cross_value_bias_fp16, projected_bev,
        40000u, dimensions, dimensions, 1);
    track_query_pos_linear_kernel<<<
        (unsigned)((rows * 64u + 255u) / 256u), 256, 0, c->stream>>>(
        norm0, c->last_track_query_pos,
        (const __half *)weights->cross_offset_weight_fp16,
        (const __half *)weights->cross_offset_bias_fp16, offsets, 64u);
    track_query_pos_linear_kernel<<<
        (unsigned)((rows * 32u + 255u) / 256u), 256, 0, c->stream>>>(
        norm0, c->last_track_query_pos,
        (const __half *)weights->cross_attention_weight_fp16,
        (const __half *)weights->cross_attention_bias_fp16, attention, 32u);
    track_cross_softmax_kernel<<<
        (unsigned)((rows * 8u + 255u) / 256u), 256, 0, c->stream>>>(
        attention);
    track_cross_deform_sample_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        projected_bev, offsets, attention, c->last_track_reference_points,
        sampled);
    linear_residual_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        sampled, (const __half *)weights->cross_out_weight_fp16,
        (const __half *)weights->cross_out_bias_fp16, norm0, cross_output,
        rows, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        cross_output, (const __half *)weights->norm1_weight_fp16,
        (const __half *)weights->norm1_bias_fp16, norm1, rows, dimensions,
        1e-5f, 1);
    linear_fp16_kernel<<<
        (unsigned)((rows * 512u + 255u) / 256u), 256, 0, c->stream>>>(
        norm1, (const __half *)weights->ffn.linear0_weight_fp16,
        (const __half *)weights->ffn.linear0_bias_fp16, hidden, rows,
        dimensions, 512u, 1);
    relu_fp16_kernel<<<
        (unsigned)((rows * 512u + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, rows * 512u);
    linear_rect_residual_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, (const __half *)weights->ffn.linear1_weight_fp16,
        (const __half *)weights->ffn.linear1_bias_fp16, norm1, ffn_output,
        rows, 512u, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        ffn_output, (const __half *)weights->ffn.norm_weight_fp16,
        (const __half *)weights->ffn.norm_bias_fp16, norm2, rows,
        dimensions, 1e-5f, 1);
    if (snapshot_track_boundary(c, layer, 0u, self_output) != UA_OK ||
        snapshot_track_boundary(c, layer, 1u, norm0) != UA_OK ||
        snapshot_track_boundary(c, layer, 2u, cross_output) != UA_OK ||
        snapshot_track_boundary(c, layer, 3u, norm1) != UA_OK ||
        snapshot_track_boundary(c, layer, 4u, ffn_output) != UA_OK ||
        snapshot_track_boundary(c, layer, 5u, norm2) != UA_OK ||
        cudaMemcpyAsync(
            (void *)(c->track_decoder_states + layer * query_count), norm2,
            query_count * sizeof(__half), cudaMemcpyDeviceToDevice,
            c->stream) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_track_decoder_self = self_output;
    c->last_track_decoder_norm0 = norm0;
    c->last_track_decoder_cross = cross_output;
    c->last_track_decoder_norm1 = norm1;
    c->last_track_decoder_ffn = ffn_output;
    c->last_track_decoder_norm2 =
        c->track_decoder_states + layer * query_count;
    c->track_decoder_layers_completed = layer + 1u;
    *output_fp16 = c->last_track_decoder_norm2;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_track_refine_references(
        void *context, size_t layer, const void *decoder_output_fp16,
        const ua_cuda_track_regression_weights *weights,
        const void **regression_fp16, const void **reference_points_fp32) {
    production_context *c = (production_context *)context;
    const size_t rows = 901u, dimensions = 256u;
    const size_t query_count = rows * dimensions;
    __half *hidden0;
    __half *hidden1;
    __half *regression;
    float *new_reference;
    if (!c || layer >= PROD_TRACK_DECODER_LAYERS ||
        !decoder_output_fp16 || !weights || !regression_fp16 ||
        !reference_points_fp32 || !c->last_track_reference_points ||
        c->track_decoder_layers_completed != layer + 1u ||
        c->track_reference_layers_completed != layer ||
        !weights->linear0_weight_fp16 || !weights->linear0_bias_fp16 ||
        !weights->linear1_weight_fp16 || !weights->linear1_bias_fp16 ||
        !weights->linear2_weight_fp16 || !weights->linear2_bias_fp16)
        return UA_ERR_ARGUMENT;
    hidden0 = (__half *)((unsigned char *)c->arena + 6291456u);
    hidden1 = (__half *)((unsigned char *)c->arena + 7340032u);
    regression = (__half *)c->track_regressions + layer * rows * 10u;
    new_reference = (float *)c->track_references + layer * rows * 3u;
    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        (const __half *)decoder_output_fp16,
        (const __half *)weights->linear0_weight_fp16,
        (const __half *)weights->linear0_bias_fp16, hidden0, rows,
        dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, query_count);
    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, (const __half *)weights->linear1_weight_fp16,
        (const __half *)weights->linear1_bias_fp16, hidden1, rows,
        dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, query_count);
    linear_fp16_kernel<<<
        (unsigned)((rows * 10u + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, (const __half *)weights->linear2_weight_fp16,
        (const __half *)weights->linear2_bias_fp16, regression, rows,
        dimensions, 10u, 1);
    track_refine_reference_kernel<<<
        (unsigned)((rows * 3u + 255u) / 256u), 256, 0, c->stream>>>(
        regression, c->last_track_reference_points, new_reference);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_track_reference_points = new_reference;
    c->track_reference_layers_completed = layer + 1u;
    *regression_fp16 = regression;
    *reference_points_fp32 = new_reference;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_track_output_heads(
        void *context,
        const ua_cuda_track_classification_weights *classification_weights,
        const ua_cuda_track_past_trajectory_weights *past_weights,
        const void **class_logits_fp16, const void **boxes_fp32,
        const void **past_trajectory_fp16) {
    production_context *c = (production_context *)context;
    const size_t rows = 901u, dimensions = 256u;
    const size_t query_count = rows * dimensions;
    const __half *state;
    __half *hidden0;
    __half *hidden1;
    __half *class_logits;
    __half *past;
    float *boxes;
    const __half *regression;
    const float *box_reference;
    if (!c || !classification_weights || !past_weights ||
        !class_logits_fp16 || !boxes_fp32 || !past_trajectory_fp16 ||
        c->track_decoder_layers_completed != PROD_TRACK_DECODER_LAYERS ||
        c->track_reference_layers_completed != PROD_TRACK_DECODER_LAYERS ||
        !classification_weights->linear0_weight_fp16 ||
        !classification_weights->linear0_bias_fp16 ||
        !classification_weights->norm0_weight_fp16 ||
        !classification_weights->norm0_bias_fp16 ||
        !classification_weights->linear1_weight_fp16 ||
        !classification_weights->linear1_bias_fp16 ||
        !classification_weights->norm1_weight_fp16 ||
        !classification_weights->norm1_bias_fp16 ||
        !classification_weights->output_weight_fp16 ||
        !classification_weights->output_bias_fp16 ||
        !past_weights->linear0_weight_fp16 ||
        !past_weights->linear0_bias_fp16 ||
        !past_weights->linear1_weight_fp16 ||
        !past_weights->linear1_bias_fp16 ||
        !past_weights->output_weight_fp16 ||
        !past_weights->output_bias_fp16)
        return UA_ERR_ARGUMENT;
    state = c->track_decoder_states +
            (PROD_TRACK_DECODER_LAYERS - 1u) * query_count;
    regression = c->track_regressions +
                 (PROD_TRACK_DECODER_LAYERS - 1u) * rows * 10u;
    box_reference = c->track_references +
                    (PROD_TRACK_DECODER_LAYERS - 2u) * rows * 3u;
    hidden0 = (__half *)((unsigned char *)c->arena + 6291456u);
    hidden1 = (__half *)((unsigned char *)c->arena + 7340032u);
    class_logits = (__half *)(
        (unsigned char *)c->arena + PROD_TRACK_CLASS_OFFSET);
    past = (__half *)((unsigned char *)c->arena + PROD_TRACK_PAST_OFFSET);
    boxes = (float *)((unsigned char *)c->arena + PROD_TRACK_BOX_OFFSET);

    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        state, (const __half *)classification_weights->linear0_weight_fp16,
        (const __half *)classification_weights->linear0_bias_fp16,
        hidden0, rows, dimensions, dimensions, 1);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, (const __half *)classification_weights->norm0_weight_fp16,
        (const __half *)classification_weights->norm0_bias_fp16,
        hidden1, rows, dimensions, 1e-5f, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, query_count);
    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, (const __half *)classification_weights->linear1_weight_fp16,
        (const __half *)classification_weights->linear1_bias_fp16,
        hidden0, rows, dimensions, dimensions, 1);
    layer_norm_fp16_kernel<<<
        (unsigned)((rows + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, (const __half *)classification_weights->norm1_weight_fp16,
        (const __half *)classification_weights->norm1_bias_fp16,
        hidden1, rows, dimensions, 1e-5f, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, query_count);
    linear_fp16_kernel<<<
        (unsigned)((rows * 10u + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1,
        (const __half *)classification_weights->output_weight_fp16,
        (const __half *)classification_weights->output_bias_fp16,
        class_logits, rows, dimensions, 10u, 1);

    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        state, (const __half *)past_weights->linear0_weight_fp16,
        (const __half *)past_weights->linear0_bias_fp16,
        hidden0, rows, dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, query_count);
    linear_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden0, (const __half *)past_weights->linear1_weight_fp16,
        (const __half *)past_weights->linear1_bias_fp16,
        hidden1, rows, dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((query_count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, query_count);
    linear_fp16_kernel<<<
        (unsigned)((rows * 16u + 255u) / 256u), 256, 0, c->stream>>>(
        hidden1, (const __half *)past_weights->output_weight_fp16,
        (const __half *)past_weights->output_bias_fp16,
        past, rows, dimensions, 16u, 1);
    track_box_output_kernel<<<
        (unsigned)((rows * 10u + 255u) / 256u), 256, 0, c->stream>>>(
        regression, box_reference, boxes);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_track_class_logits = class_logits;
    c->last_track_boxes = boxes;
    c->last_track_past_trajectory = past;
    c->track_output_heads_completed = 1;
    *class_logits_fp16 = class_logits;
    *boxes_fp32 = boxes;
    *past_trajectory_fp16 = past;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_track_score_filter(
        void *context, const void **scores_fp32, const void **classes_u32,
        const void **selected_indices_u32, const void **selected_count_u32) {
    production_context *c = (production_context *)context;
    float *scores;
    uint32_t *classes;
    uint32_t *selected;
    uint32_t *selected_count;
    if (!c || !scores_fp32 || !classes_u32 || !selected_indices_u32 ||
        !selected_count_u32 || !c->track_output_heads_completed ||
        !c->last_track_class_logits)
        return UA_ERR_ARGUMENT;
    scores = (float *)((unsigned char *)c->arena + PROD_TRACK_SCORE_OFFSET);
    classes = (uint32_t *)(
        (unsigned char *)c->arena + PROD_TRACK_CLASS_INDEX_OFFSET);
    selected = (uint32_t *)(
        (unsigned char *)c->arena + PROD_TRACK_SELECTED_OFFSET);
    selected_count = (uint32_t *)(
        (unsigned char *)c->arena + PROD_TRACK_SELECTED_COUNT_OFFSET);
    track_score_class_kernel<<<
        (900u + 255u) / 256u, 256, 0, c->stream>>>(
        c->last_track_class_logits, 900u, 10u, scores, classes);
    track_filter_kernel<<<1, 1, 0, c->stream>>>(
        scores, 900u, 0.4f, UA_PROD_MAX_TRACKS, selected, selected_count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_track_scores = scores;
    c->last_track_classes = classes;
    c->last_track_selected_indices = selected;
    c->last_track_selected_count = selected_count;
    *scores_fp32 = scores;
    *classes_u32 = classes;
    *selected_indices_u32 = selected;
    *selected_count_u32 = selected_count;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_query_interaction(
        void *context, const void *query_pos_fp16,
        const void *query_feat_fp16, const void *output_embedding_fp16,
        size_t active_queries,
        const ua_cuda_query_interaction_weights *weights,
        const void **updated_query_feat_fp16) {
    production_context *c = (production_context *)context;
    const size_t dimensions = 256u;
    size_t count;
    __half *qkv, *attended, *self_output, *norm1;
    __half *hidden, *ffn_output, *norm2;
    __half *feat_hidden, *feat_output, *norm_feat;
    if (!c || !query_pos_fp16 || !query_feat_fp16 ||
        !output_embedding_fp16 || !weights || !updated_query_feat_fp16 ||
        active_queries > UA_PROD_MAX_TRACKS ||
        !weights->self_in_weight_fp16 || !weights->self_in_bias_fp16 ||
        !weights->self_out_weight_fp16 || !weights->self_out_bias_fp16 ||
        !weights->norm1_weight_fp16 || !weights->norm1_bias_fp16 ||
        !weights->linear1_weight_fp16 || !weights->linear1_bias_fp16 ||
        !weights->linear2_weight_fp16 || !weights->linear2_bias_fp16 ||
        !weights->norm2_weight_fp16 || !weights->norm2_bias_fp16 ||
        !weights->feat1_weight_fp16 || !weights->feat1_bias_fp16 ||
        !weights->feat2_weight_fp16 || !weights->feat2_bias_fp16 ||
        !weights->norm_feat_weight_fp16 || !weights->norm_feat_bias_fp16)
        return UA_ERR_ARGUMENT;
    c->query_interaction_weights = *weights;
    c->query_interaction_weights_valid = 1;
    if (!active_queries) {
        *updated_query_feat_fp16 = query_feat_fp16;
        return UA_OK;
    }
    count = active_queries * dimensions;
    qkv = (__half *)((unsigned char *)c->arena + 4194304u);
    attended = (__half *)((unsigned char *)c->arena + 5242880u);
    self_output = (__half *)((unsigned char *)c->arena + 5505024u);
    norm1 = (__half *)((unsigned char *)c->arena + 5767168u);
    hidden = (__half *)((unsigned char *)c->arena + 6291456u);
    ffn_output = (__half *)((unsigned char *)c->arena + 6815744u);
    norm2 = (__half *)((unsigned char *)c->arena + 7340032u);
    feat_hidden = (__half *)((unsigned char *)c->arena + 7864320u);
    feat_output = (__half *)((unsigned char *)c->arena + 8388608u);
    norm_feat = (__half *)((unsigned char *)c->arena + 8912896u);
    track_qkv_linear_kernel<<<
        (unsigned)((active_queries * 768u + 255u) / 256u),
        256, 0, c->stream>>>(
        (const __half *)output_embedding_fp16,
        (const __half *)query_pos_fp16,
        (const __half *)weights->self_in_weight_fp16,
        (const __half *)weights->self_in_bias_fp16, qkv, active_queries);
    track_self_attention_kernel<<<
        (unsigned)((active_queries * 8u + 255u) / 256u),
        256, 0, c->stream>>>(qkv, attended, active_queries);
    linear_residual_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        attended, (const __half *)weights->self_out_weight_fp16,
        (const __half *)weights->self_out_bias_fp16,
        (const __half *)output_embedding_fp16, self_output,
        active_queries, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((active_queries + 255u) / 256u),
        256, 0, c->stream>>>(
        self_output, (const __half *)weights->norm1_weight_fp16,
        (const __half *)weights->norm1_bias_fp16, norm1,
        active_queries, dimensions, 1e-5f, 1);
    linear_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        norm1, (const __half *)weights->linear1_weight_fp16,
        (const __half *)weights->linear1_bias_fp16, hidden,
        active_queries, dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, count);
    linear_residual_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        hidden, (const __half *)weights->linear2_weight_fp16,
        (const __half *)weights->linear2_bias_fp16, norm1, ffn_output,
        active_queries, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((active_queries + 255u) / 256u),
        256, 0, c->stream>>>(
        ffn_output, (const __half *)weights->norm2_weight_fp16,
        (const __half *)weights->norm2_bias_fp16, norm2,
        active_queries, dimensions, 1e-5f, 1);
    linear_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        norm2, (const __half *)weights->feat1_weight_fp16,
        (const __half *)weights->feat1_bias_fp16, feat_hidden,
        active_queries, dimensions, dimensions, 1);
    relu_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        feat_hidden, count);
    linear_residual_fp16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        feat_hidden, (const __half *)weights->feat2_weight_fp16,
        (const __half *)weights->feat2_bias_fp16,
        (const __half *)query_feat_fp16, feat_output,
        active_queries, dimensions);
    layer_norm_fp16_kernel<<<
        (unsigned)((active_queries + 255u) / 256u),
        256, 0, c->stream>>>(
        feat_output, (const __half *)weights->norm_feat_weight_fp16,
        (const __half *)weights->norm_feat_bias_fp16, norm_feat,
        active_queries, dimensions, 1e-5f, 1);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess)
        return UA_ERR_BACKEND;
    c->last_query_interaction_output = norm_feat;
    c->last_query_interaction_queries = active_queries;
    *updated_query_feat_fp16 = norm_feat;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_debug_query_interaction(
        void *context, size_t active_queries) {
    production_context *c = (production_context *)context;
    const void *output = NULL;
    if (!c || !active_queries || active_queries > UA_PROD_MAX_TRACKS ||
        !c->query_interaction_weights_valid || !c->last_track_query_pos ||
        !c->last_track_query || !c->last_track_decoder_norm2)
        return UA_ERR_PROFILE;
    return ua_cuda_production_query_interaction(
        c, c->last_track_query_pos, c->last_track_query,
        c->last_track_decoder_norm2, active_queries,
        &c->query_interaction_weights, &output);
}

extern "C" ua_status ua_cuda_production_copy_boundary_f32(
        void *context, const char *name, float *values, size_t capacity,
        size_t *written) {
    production_context *c = (production_context *)context;
    float *device_values = NULL;
    size_t count;
    if (!c || !name || !values || !capacity || !written)
        return UA_ERR_ARGUMENT;
    *written = 0;
    if (!strcmp(name, "production.bevformer.previous_bev.valid")) {
        values[0] = (float)c->previous_bev_valid;
        *written = 1;
        return UA_OK;
    }
    if (!strcmp(name, "production.resnet_layer3.first_nonfinite")) {
        values[0] = (float)c->layer3_first_nonfinite;
        *written = 1;
        return UA_OK;
    }
    if (!strcmp(name, "production.resnet_layer4.first_nonfinite")) {
        values[0] = (float)c->layer4_first_nonfinite;
        *written = 1;
        return UA_OK;
    }
    if (!strcmp(
            name, "production.bevformer.reference_camera.center_d3")) {
        const size_t index = (19899u * 4u + 3u) * 2u;
        if (!c->last_reference_camera) return UA_ERR_PROFILE;
        count = capacity < 2u ? capacity : 2u;
        if (cudaMemcpyAsync(
                values, c->last_reference_camera + index,
                count * sizeof(float), cudaMemcpyDeviceToHost,
                c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        *written = count;
        return UA_OK;
    }
    if (!strcmp(
            name, "production.bevformer.visibility.center_d3")) {
        const size_t index = 19899u * 4u + 3u;
        uint8_t visible = 0;
        if (!c->last_visibility) return UA_ERR_PROFILE;
        if (cudaMemcpyAsync(
                &visible, c->last_visibility + index, sizeof(visible),
                cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        values[0] = (float)visible;
        *written = 1;
        return UA_OK;
    }
    if (!strcmp(name, "production.bevformer.visible_counts")) {
        uint32_t counts[UA_CAMERA_COUNT];
        if (!c->last_visible_counts) return UA_ERR_PROFILE;
        count = capacity < UA_CAMERA_COUNT ? capacity : UA_CAMERA_COUNT;
        if (cudaMemcpyAsync(
                counts, c->last_visible_counts,
                UA_CAMERA_COUNT * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        for (size_t camera = 0; camera < count; ++camera)
            values[camera] = (float)counts[camera];
        *written = count;
        return UA_OK;
    }
    if (!strcmp(name, "production.bevformer.visible_indices.camera0")) {
        uint32_t indices[32];
        if (!c->last_visible_indices) return UA_ERR_PROFILE;
        count = capacity < 32u ? capacity : 32u;
        if (cudaMemcpyAsync(
                indices, c->last_visible_indices,
                count * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        for (size_t index = 0; index < count; ++index)
            values[index] = (float)indices[index];
        *written = count;
        return UA_OK;
    }
    if (!strcmp(name, "production.track.reference_points")) {
        if (!c->initial_track_reference_points) return UA_ERR_PROFILE;
        count = capacity < 901u * 3u ? capacity : 901u * 3u;
        if (cudaMemcpyAsync(
                values, c->initial_track_reference_points,
                count * sizeof(float), cudaMemcpyDeviceToHost,
                c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        *written = count;
        return UA_OK;
    }
    if (!strncmp(
            name, "production.track.decoder.layer",
            strlen("production.track.decoder.layer"))) {
        const char *cursor =
            name + strlen("production.track.decoder.layer");
        size_t layer;
        if (cursor[0] < '0' || cursor[0] > '5')
            return UA_ERR_IO;
        layer = (size_t)(cursor[0] - '0');
        cursor += 1;
        if (!strcmp(cursor, ".reference")) {
            if (layer >= c->track_reference_layers_completed)
                return UA_ERR_PROFILE;
            count = capacity < 901u * 3u ? capacity : 901u * 3u;
            if (cudaMemcpyAsync(
                    values, c->track_references + layer * 901u * 3u,
                    count * sizeof(float), cudaMemcpyDeviceToHost,
                    c->stream) != cudaSuccess ||
                cudaStreamSynchronize(c->stream) != cudaSuccess)
                return UA_ERR_BACKEND;
            *written = count;
            return UA_OK;
        }
    }
    if (!strcmp(name, "production.track.outputs.box")) {
        if (!c->track_output_heads_completed || !c->last_track_boxes)
            return UA_ERR_PROFILE;
        count = capacity < 901u * 10u ? capacity : 901u * 10u;
        if (cudaMemcpyAsync(
                values, c->last_track_boxes, count * sizeof(float),
                cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        *written = count;
        return UA_OK;
    }
    if (!strcmp(name, "production.track.decode.scores")) {
        if (!c->last_track_scores) return UA_ERR_PROFILE;
        count = capacity < 900u ? capacity : 900u;
        if (cudaMemcpyAsync(
                values, c->last_track_scores, count * sizeof(float),
                cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        *written = count;
        return UA_OK;
    }
    if (!strcmp(name, "production.track.decode.selected_count")) {
        uint32_t selected_count;
        if (!c->last_track_selected_count) return UA_ERR_PROFILE;
        if (cudaMemcpyAsync(
                &selected_count, c->last_track_selected_count,
                sizeof(selected_count), cudaMemcpyDeviceToHost,
                c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        values[0] = (float)selected_count;
        *written = 1u;
        return UA_OK;
    }
    if (!strcmp(name, "production.track.decode.classes") ||
        !strcmp(name, "production.track.decode.selected_indices")) {
        uint32_t host_values[32];
        uint32_t selected_count = UA_PROD_MAX_TRACKS;
        const uint32_t *device_source =
            !strcmp(name, "production.track.decode.classes")
            ? c->last_track_classes : c->last_track_selected_indices;
        size_t available =
            !strcmp(name, "production.track.decode.classes")
            ? 900u : UA_PROD_MAX_TRACKS;
        if (!device_source) return UA_ERR_PROFILE;
        if (!strcmp(name, "production.track.decode.selected_indices")) {
            if (!c->last_track_selected_count ||
                cudaMemcpyAsync(
                    &selected_count, c->last_track_selected_count,
                    sizeof(selected_count), cudaMemcpyDeviceToHost,
                    c->stream) != cudaSuccess ||
                cudaStreamSynchronize(c->stream) != cudaSuccess)
                return UA_ERR_BACKEND;
            available = selected_count;
        }
        count = capacity < available ? capacity : available;
        if (count > 32u) count = 32u;
        if (cudaMemcpyAsync(
                host_values, device_source, count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
            cudaStreamSynchronize(c->stream) != cudaSuccess)
            return UA_ERR_BACKEND;
        for (size_t index = 0; index < count; ++index)
            values[index] = (float)host_values[index];
        *written = count;
        return UA_OK;
    }
    const __half *source = NULL;
    size_t source_count = 0;
    if (!strcmp(name, "production.resnet_stem")) {
        source = c->last_resnet_stem;
        source_count = c->last_resnet_stem_count;
    } else if (!strcmp(name, "production.resnet_layer1")) {
        source = c->last_resnet_layer1;
        source_count = c->last_resnet_layer1_count;
    } else if (!strcmp(name, "production.resnet_layer1.block0")) {
        source = c->boundary_samples;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer1.block1")) {
        source = c->boundary_samples + 32u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer1.block2")) {
        source = c->boundary_samples + 64u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer2")) {
        source = c->last_resnet_layer2;
        source_count = c->last_resnet_layer2_count;
    } else if (!strcmp(name, "production.resnet_layer2.snapshot")) {
        source = c->boundary_samples + 7u * 32u;
        source_count = 32u;
    } else if (!strncmp(
                   name, "production.resnet_layer2.block",
                   strlen("production.resnet_layer2.block"))) {
        const char *suffix =
            name + strlen("production.resnet_layer2.block");
        if (suffix[0] < '0' || suffix[0] > '3' || suffix[1] != '\0')
            return UA_ERR_IO;
        source =
            c->boundary_samples + (3u + (size_t)(suffix[0] - '0')) * 32u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer3")) {
        source = c->last_resnet_layer3;
        source_count = c->last_resnet_layer3_count;
    } else if (!strcmp(name, "production.resnet_layer3.block0")) {
        source = c->boundary_samples + 8u * 32u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer3.block22")) {
        source = c->boundary_samples + 9u * 32u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer4")) {
        source = c->last_resnet_layer4;
        source_count = c->last_resnet_layer4_count;
    } else if (!strcmp(name, "production.resnet_layer4.block0")) {
        source = c->boundary_samples + 10u * 32u;
        source_count = 32u;
    } else if (!strcmp(name, "production.resnet_layer4.block2")) {
        source = c->boundary_samples + 11u * 32u;
        source_count = 32u;
    } else if (!strncmp(name, "production.fpn.", 15u)) {
        const char *suffix = name + 15u;
        if (suffix[0] < '0' || suffix[0] > '3' || suffix[1] != '\0')
            return UA_ERR_IO;
        source = c->last_spatial_attention0
            ? c->boundary_samples +
                (12u + (size_t)(suffix[0] - '0')) * 32u
            : c->last_fpn[(size_t)(suffix[0] - '0')];
        source_count = c->last_spatial_attention0
            ? 32u
            : c->last_fpn_count[(size_t)(suffix[0] - '0')];
    } else if (!strcmp(name, "production.bevformer.feat_flatten")) {
        source = c->last_bevformer_flatten;
        source_count = c->last_bevformer_flatten_count;
    } else if (!strncmp(
                   name, "production.bevformer.feat_flatten.level",
                   strlen("production.bevformer.feat_flatten.level"))) {
        const char *suffix =
            name + strlen("production.bevformer.feat_flatten.level");
        static const size_t starts[4] = {0u, 23200u, 29000u, 30450u};
        if (suffix[0] < '0' || suffix[0] > '3' || suffix[1] != '\0')
            return UA_ERR_IO;
        if (!c->last_bevformer_flatten) return UA_ERR_PROFILE;
        source = c->last_bevformer_flatten +
                 starts[(size_t)(suffix[0] - '0')] * 256u;
        source_count = 32u;
    } else if (!strcmp(name, "production.bevformer.bev_queries")) {
        source = c->last_bev_queries;
        source_count = 40000u * 256u;
    } else if (!strcmp(name, "production.bevformer.bev_pos")) {
        source = c->last_bev_pos;
        source_count = 40000u * 256u;
    } else if (!strcmp(name, "production.bevformer.reference_2d")) {
        source = c->last_reference_2d;
        source_count = 40000u * 2u;
    } else if (!strcmp(name, "production.bevformer.reference_3d")) {
        source = c->last_reference_3d;
        source_count = 4u * 40000u * 3u;
    } else if (!strcmp(name, "production.bevformer.encoder.output")) {
        if (c->encoder_layers_completed != PROD_ENCODER_LAYERS)
            return UA_ERR_PROFILE;
        source = c->last_encoder_norm2;
        source_count = 40000u * 256u;
    } else if (!strcmp(name, "production.track.query_pos")) {
        source = c->last_track_query_pos;
        source_count = 901u * 256u;
    } else if (!strcmp(name, "production.track.query")) {
        source = c->last_track_query;
        source_count = 901u * 256u;
    } else if (!strcmp(name, "production.track.outputs.class_logits")) {
        if (!c->track_output_heads_completed) return UA_ERR_PROFILE;
        source = c->last_track_class_logits;
        source_count = 901u * 10u;
    } else if (!strcmp(name, "production.track.outputs.past_trajectory")) {
        if (!c->track_output_heads_completed) return UA_ERR_PROFILE;
        source = c->last_track_past_trajectory;
        source_count = 901u * 16u;
    } else if (!strcmp(name, "production.track.query_interaction")) {
        source = c->last_query_interaction_output;
        source_count = c->last_query_interaction_queries * 256u;
    } else if (!strncmp(
                   name, "production.track.decoder.layer",
                   strlen("production.track.decoder.layer"))) {
        static const char *const stages[6] = {
            ".self", ".norm0", ".cross", ".norm1", ".ffn", ".norm2"
        };
        const char *cursor =
            name + strlen("production.track.decoder.layer");
        size_t layer, stage, base;
        if (cursor[0] < '0' || cursor[0] > '5') return UA_ERR_IO;
        layer = (size_t)(cursor[0] - '0');
        if (layer >= c->track_decoder_layers_completed)
            return UA_ERR_PROFILE;
        cursor += 1;
        if (!strcmp(cursor, ".state")) {
            source = c->track_decoder_states + layer * PROD_TRACK_QUERY_COUNT;
            source_count = PROD_TRACK_QUERY_COUNT;
        } else if (!strcmp(cursor, ".regression")) {
            if (layer >= c->track_reference_layers_completed)
                return UA_ERR_PROFILE;
            source = c->track_regressions + layer * 901u * 10u;
            source_count = 901u * 10u;
        } else {
        for (stage = 0; stage < 6u; ++stage)
            if (!strcmp(cursor, stages[stage])) break;
        if (stage == 6u) return UA_ERR_IO;
        base = PROD_BACKBONE_FPN_BOUNDARIES +
               PROD_ENCODER_LAYERS * PROD_ENCODER_BOUNDARIES_PER_LAYER;
        source = c->boundary_samples +
            (base + layer * PROD_TRACK_BOUNDARIES_PER_LAYER + stage) * 32u;
        source_count = 32u;
        }
    } else if (!strncmp(
                   name, "production.bevformer.encoder.layer",
                   strlen("production.bevformer.encoder.layer"))) {
        static const char *const stages[6] = {
            ".temporal", ".norm0", ".spatial", ".norm1", ".ffn", ".norm2"
        };
        const char *cursor =
            name + strlen("production.bevformer.encoder.layer");
        size_t layer, stage;
        if (cursor[0] < '0' || cursor[0] > '5')
            return UA_ERR_IO;
        layer = (size_t)(cursor[0] - '0');
        if (layer >= c->encoder_layers_completed)
            return UA_ERR_PROFILE;
        cursor += 1;
        for (stage = 0; stage < 6u; ++stage)
            if (!strcmp(cursor, stages[stage])) break;
        if (stage == 6u) return UA_ERR_IO;
        source = c->boundary_samples +
            (PROD_BACKBONE_FPN_BOUNDARIES +
             layer * PROD_ENCODER_BOUNDARIES_PER_LAYER + stage) * 32u;
        source_count = 32u;
    } else {
        return UA_ERR_IO;
    }
    if (!source || !source_count) return UA_ERR_PROFILE;
    count = capacity < source_count ? capacity : source_count;
    if (!fixture_count_ok(count, sizeof(float)))
        return UA_ERR_CAPACITY;
    if (cudaMalloc((void **)&device_values, count * sizeof(float)) != cudaSuccess)
        return UA_ERR_MEMORY;
    half_to_float_kernel<<<
        (unsigned)((count + 255u) / 256u), 256, 0, c->stream>>>(
        source, device_values, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(values, device_values, count * sizeof(float),
                        cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess) {
        cudaFree(device_values);
        return UA_ERR_BACKEND;
    }
    cudaFree(device_values);
    *written = count;
    return UA_OK;
}

extern "C" ua_status ua_cuda_production_tensor_pointer(
        void *context, size_t byte_offset, size_t nbytes,
        const void **device_ptr) {
    production_context *c = (production_context *)context;
    if (!c || !device_ptr || !nbytes) return UA_ERR_ARGUMENT;
    *device_ptr = NULL;
    if (byte_offset > c->weights_bytes ||
        nbytes > c->weights_bytes - byte_offset)
        return UA_ERR_CAPACITY;
    *device_ptr = (const unsigned char *)c->weights + byte_offset;
    return UA_OK;
}

extern "C" void ua_cuda_production_reset(void *context) {
    production_context *c = (production_context *)context;
    if (!c) return;
    (void)cudaMemsetAsync(c->previous_bev, 0, PROD_PREVIOUS_BEV_BYTES, c->stream);
    (void)cudaMemsetAsync(
        c->aligned_previous_bev, 0, PROD_PREVIOUS_BEV_BYTES, c->stream);
    (void)cudaMemsetAsync(
        c->track_state_committed, 0, PROD_TRACK_STATE_BYTES, c->stream);
    (void)cudaMemsetAsync(
        c->track_state_candidate, 0, PROD_TRACK_STATE_BYTES, c->stream);
    (void)cudaStreamSynchronize(c->stream);
    memset(c->scene_token, 0, sizeof(c->scene_token));
    c->has_scene = 0;
    c->previous_bev_valid = 0;
    c->temporal_shift_x = 0.0f;
    c->temporal_shift_y = 0.0f;
    c->last_resnet_stem = NULL;
    c->last_resnet_stem_count = 0;
    c->last_resnet_layer1 = NULL;
    c->last_resnet_layer1_count = 0;
    c->last_resnet_layer2 = NULL;
    c->last_resnet_layer2_count = 0;
    c->last_resnet_layer3 = NULL;
    c->last_resnet_layer3_count = 0;
    c->layer3_first_nonfinite = 23;
    c->last_resnet_layer4 = NULL;
    c->last_resnet_layer4_count = 0;
    c->layer4_first_nonfinite = 3;
    for (size_t level = 0; level < 4u; ++level) {
        c->last_fpn[level] = NULL;
        c->last_fpn_count[level] = 0;
    }
    c->last_bevformer_flatten = NULL;
    c->last_bevformer_flatten_count = 0;
    c->last_bev_queries = NULL;
    c->last_bev_pos = NULL;
    c->last_reference_2d = NULL;
    c->last_reference_3d = NULL;
    c->last_reference_camera = NULL;
    c->last_visibility = NULL;
    c->last_visible_indices = NULL;
    c->last_visible_counts = NULL;
    c->last_temporal_attention0 = NULL;
    c->last_encoder_norm0 = NULL;
    c->last_spatial_attention0 = NULL;
    c->last_encoder_norm1 = NULL;
    c->last_encoder_ffn0 = NULL;
    c->last_encoder_norm2 = NULL;
    c->encoder_layers_completed = 0u;
    c->last_track_query_pos = NULL;
    c->last_track_query = NULL;
    c->initial_track_reference_points = NULL;
    c->last_track_reference_points = NULL;
    c->last_track_decoder_self = NULL;
    c->last_track_decoder_norm0 = NULL;
    c->last_track_decoder_cross = NULL;
    c->last_track_decoder_norm1 = NULL;
    c->last_track_decoder_ffn = NULL;
    c->last_track_decoder_norm2 = NULL;
    c->track_decoder_states = NULL;
    c->track_regressions = NULL;
    c->track_references = NULL;
    c->last_track_class_logits = NULL;
    c->last_track_boxes = NULL;
    c->last_track_past_trajectory = NULL;
    c->last_track_scores = NULL;
    c->last_track_classes = NULL;
    c->last_track_selected_indices = NULL;
    c->last_track_selected_count = NULL;
    memset(&c->query_interaction_weights, 0,
           sizeof(c->query_interaction_weights));
    c->query_interaction_weights_valid = 0;
    c->last_query_interaction_output = NULL;
    c->last_query_interaction_queries = 0u;
    c->track_decoder_layers_completed = 0u;
    c->track_reference_layers_completed = 0u;
    c->track_output_heads_completed = 0;
    (void)cudaMemsetAsync(
        c->boundary_samples, 0,
        PROD_BOUNDARY_SAMPLE_VALUES * sizeof(__half), c->stream);
    (void)cudaStreamSynchronize(c->stream);
}

extern "C" void ua_cuda_production_destroy(void *context) {
    production_context *c = (production_context *)context;
    if (!c) return;
    cudaFree(c->arena);
    cudaFree(c->track_state_candidate);
    cudaFree(c->track_state_committed);
    cudaFree(c->aligned_previous_bev);
    cudaFree(c->previous_bev); cudaFree(c->weights);
    cudaFree(c->stage_status); cudaFree(c->boundary_samples);
    cudaFree(c->metadata);
    cudaFree(c->camera_raw);
    cudaStreamDestroy(c->stream);
    free(c);
}
