#define _POSIX_C_SOURCE 200809L
#include "uniad.h"
#include "operators.h"
#ifdef UA_WITH_CUDA
#include "cuda_backend.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern ua_status ua_write_demo_assets(const char *directory);

#ifdef UA_WITH_CUDA
static int close_enough(float actual, float expected, float atol, float rtol) {
    return fabsf(actual - expected) <= atol + rtol * fabsf(expected);
}

static void reference_conv2d(
        const float *x, const float *weight, const float *bias, float *y,
        size_t batches, size_t input_channels, size_t input_height,
        size_t input_width, size_t output_channels, size_t kernel_height,
        size_t kernel_width, size_t stride_height, size_t stride_width,
        size_t padding_height, size_t padding_width) {
    size_t output_height =
        (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    size_t output_width =
        (input_width + 2 * padding_width - kernel_width) / stride_width + 1;
    size_t n, oc, oy, ox, ic, ky, kx;
    for (n = 0; n < batches; ++n)
        for (oc = 0; oc < output_channels; ++oc)
            for (oy = 0; oy < output_height; ++oy)
                for (ox = 0; ox < output_width; ++ox) {
                    float sum = bias ? bias[oc] : 0.0f;
                    for (ic = 0; ic < input_channels; ++ic)
                        for (ky = 0; ky < kernel_height; ++ky)
                            for (kx = 0; kx < kernel_width; ++kx) {
                                long iy = (long)(oy * stride_height + ky) -
                                          (long)padding_height;
                                long ix = (long)(ox * stride_width + kx) -
                                          (long)padding_width;
                                if (iy >= 0 && ix >= 0 &&
                                    (size_t)iy < input_height &&
                                    (size_t)ix < input_width)
                                    sum += x[((n * input_channels + ic) *
                                              input_height + (size_t)iy) *
                                             input_width + (size_t)ix] *
                                           weight[((oc * input_channels + ic) *
                                                   kernel_height + ky) *
                                                  kernel_width + kx];
                            }
                    y[((n * output_channels + oc) * output_height + oy) *
                      output_width + ox] = sum;
                }
}

static float reference_ms_sample(
        const float *value, size_t batch, size_t total, size_t heads,
        size_t channels, size_t head, size_t channel, size_t start,
        size_t height, size_t width, float normalized_x, float normalized_y) {
    float x = normalized_x * (float)width - .5f;
    float y = normalized_y * (float)height - .5f;
    long x0 = (long)floorf(x), y0 = (long)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0, result = 0.0f;
    int corner;
    for (corner = 0; corner < 4; ++corner) {
        long yy = y0 + (corner >= 2), xx = x0 + (corner & 1);
        if (yy >= 0 && xx >= 0 && (size_t)yy < height &&
            (size_t)xx < width) {
            float wy = corner >= 2 ? fy : 1.0f - fy;
            float wx = corner & 1 ? fx : 1.0f - fx;
            size_t spatial = start + (size_t)yy * width + (size_t)xx;
            result += value[((batch * total + spatial) * heads + head) *
                            channels + channel] * wy * wx;
        }
    }
    return result;
}

static float reference_nchw_sample(
        const float *x, size_t batch, size_t channel, size_t channels,
        size_t height, size_t width, float y, float x_coordinate) {
    float plane[64];
    size_t yy, xx;
    assert(height * width <= 64);
    for (yy = 0; yy < height; ++yy)
        for (xx = 0; xx < width; ++xx)
            plane[yy * width + xx] =
                x[((batch * channels + channel) * height + yy) * width + xx];
    return ua_op_bilinear(plane, height, width, y, x_coordinate);
}

static void test_cuda_operators(void) {
    {
        enum { WIDTH = 1600, HEIGHT = 900, STRIDE = WIDTH * 3 + 8 };
        uint8_t weights[256] = {0};
        uint8_t *image = (uint8_t *)calloc(HEIGHT, STRIDE);
        ua_production_input input;
        void *context = NULL;
        const void *normalized = NULL, *weight_pointer = NULL;
        size_t device_bytes = 0, h2d_bytes = 0;
        size_t camera;
        assert(image);
        memset(&input, 0, sizeof(input));
        input.version = 2;
        input.scene_token = "scene-a";
        for (camera = 0; camera < UA_CAMERA_COUNT; ++camera) {
            input.cameras[camera].data = image;
            input.cameras[camera].width = WIDTH;
            input.cameras[camera].height = HEIGHT;
            input.cameras[camera].row_stride_bytes = STRIDE;
        }
        assert(ua_cuda_production_create(
            weights, sizeof(weights), &context, &device_bytes) == UA_OK);
        assert(device_bytes > 512u * 1024u * 1024u);
        assert(ua_cuda_production_tensor_pointer(
            context, 16, 32, &weight_pointer) == UA_OK);
        assert(weight_pointer);
        assert(ua_cuda_production_tensor_pointer(
            context, 250, 32, &weight_pointer) == UA_ERR_CAPACITY);
        assert(ua_cuda_production_preprocess(
            context, &input, &h2d_bytes, &normalized) == UA_OK);
        assert(h2d_bytes ==
               (size_t)UA_CAMERA_COUNT * WIDTH * HEIGHT * 3u +
               UA_PRODUCTION_METADATA_BYTES);
        assert(normalized);
        input.scene_token = "scene-b";
        assert(ua_cuda_production_preprocess(
            context, &input, &h2d_bytes, &normalized) == UA_OK);
        input.cameras[4].width = WIDTH - 1;
        assert(ua_cuda_production_preprocess(
            context, &input, &h2d_bytes, &normalized) == UA_ERR_ARGUMENT);
        ua_cuda_production_destroy(context);
        free(image);
    }
    {
        /* Two rows with explicit stride padding; output is padded planar CHW. */
        const uint8_t bgr[20] = {
            10, 20, 30, 40, 50, 60, 77, 77, 77, 77,
            70, 80, 90, 100, 110, 120, 88, 88, 88, 88,
        };
        const float mean[3] = {1.0f, 2.0f, 3.0f};
        const float std[3] = {2.0f, 4.0f, 5.0f};
        float output[3 * 3 * 4];
        size_t channel, y, x;
        assert(ua_cuda_test_preprocess_bgr(
            bgr, 2, 2, 10, 4, 3, mean, std, output) == UA_OK);
        for (channel = 0; channel < 3; ++channel)
            for (y = 0; y < 3; ++y)
                for (x = 0; x < 4; ++x) {
                    float expected = 0.0f;
                    if (x < 2 && y < 2)
                        expected = ((float)bgr[y * 10 + x * 3 + channel] -
                                    mean[channel]) / std[channel];
                    assert(close_enough(
                        output[(channel * 3 + y) * 4 + x], expected,
                        5e-3f, 1e-2f));
                }
        assert(ua_cuda_test_preprocess_bgr(
            bgr, 2, 2, 5, 4, 3, mean, std, output) == UA_ERR_ARGUMENT);
    }
    {
        enum { ROWS = 3, INPUTS = 7, OUTPUTS = 5 };
        float x[ROWS * INPUTS], weight[OUTPUTS * INPUTS], bias[OUTPUTS];
        float expected[ROWS * OUTPUTS], actual[ROWS * OUTPUTS];
        size_t i;
        for (i = 0; i < ROWS * INPUTS; ++i)
            x[i] = ((int)(i % 11) - 5) * 0.125f;
        for (i = 0; i < OUTPUTS * INPUTS; ++i)
            weight[i] = ((int)(i % 7) - 3) * 0.0625f;
        for (i = 0; i < OUTPUTS; ++i) bias[i] = (float)i * 0.03125f;
        ua_op_linear(x, weight, bias, expected, ROWS, INPUTS, OUTPUTS);
        assert(ua_cuda_test_linear_fp16(
            x, weight, bias, actual, ROWS, INPUTS, OUTPUTS) == UA_OK);
        for (i = 0; i < ROWS * OUTPUTS; ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        x[0] = NAN;
        assert(ua_cuda_test_linear_fp16(
            x, weight, bias, actual, ROWS, INPUTS, OUTPUTS) ==
            UA_ERR_NONFINITE);
    }
    {
        enum {
            BATCHES = 2, INPUTS = 3, HEIGHT = 5, WIDTH = 6, OUTPUTS = 4,
            KERNEL = 3, OUT_HEIGHT = 3, OUT_WIDTH = 3
        };
        float x[BATCHES * INPUTS * HEIGHT * WIDTH];
        float weight[OUTPUTS * INPUTS * KERNEL * KERNEL], bias[OUTPUTS];
        float expected[BATCHES * OUTPUTS * OUT_HEIGHT * OUT_WIDTH];
        float actual[BATCHES * OUTPUTS * OUT_HEIGHT * OUT_WIDTH];
        size_t i;
        for (i = 0; i < sizeof(x) / sizeof(x[0]); ++i)
            x[i] = ((int)(i % 13) - 6) * .03125f;
        for (i = 0; i < sizeof(weight) / sizeof(weight[0]); ++i)
            weight[i] = ((int)(i % 9) - 4) * .015625f;
        for (i = 0; i < OUTPUTS; ++i) bias[i] = (float)i * .025f;
        reference_conv2d(
            x, weight, bias, expected, BATCHES, INPUTS, HEIGHT, WIDTH,
            OUTPUTS, KERNEL, KERNEL, 2, 2, 1, 1);
        assert(ua_cuda_test_conv2d_fp16(
            x, weight, bias, actual, BATCHES, INPUTS, HEIGHT, WIDTH,
            OUTPUTS, KERNEL, KERNEL, 2, 2, 1, 1) == UA_OK);
        for (i = 0; i < sizeof(actual) / sizeof(actual[0]); ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        assert(ua_cuda_test_conv2d_fp16(
            x, weight, bias, actual, BATCHES, INPUTS, HEIGHT, WIDTH,
            OUTPUTS, KERNEL, KERNEL, 0, 2, 1, 1) == UA_ERR_ARGUMENT);
    }
    {
        enum { B = 2, C = 3, H = 2, W = 4, COUNT = B*C*H*W };
        float x[COUNT], gamma[C], beta[C], mean[C], variance[C];
        float expected[COUNT], actual[COUNT];
        size_t i;
        for (i = 0; i < COUNT; ++i)
            x[i] = ((int)(i % 13) - 6) * .25f;
        for (i = 0; i < C; ++i) {
            gamma[i] = .75f + (float)i * .125f;
            beta[i] = ((int)i - 1) * .0625f;
            mean[i] = ((int)i - 1) * .125f;
            variance[i] = .5f + (float)i * .25f;
        }
        for (i = 0; i < COUNT; ++i) {
            size_t channel = (i / (H * W)) % C;
            float value = (x[i] - mean[channel]) /
                          sqrtf(variance[channel] + 1e-5f);
            value = value * gamma[channel] + beta[channel];
            expected[i] = value > 0.0f ? value : 0.0f;
        }
        assert(ua_cuda_test_batchnorm_relu_fp16(
            x, gamma, beta, mean, variance, actual, B, C, H, W, 1e-5f) ==
            UA_OK);
        for (i = 0; i < COUNT; ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        variance[1] = -1.0f;
        assert(ua_cuda_test_batchnorm_relu_fp16(
            x, gamma, beta, mean, variance, actual, B, C, H, W, 1e-5f) ==
            UA_ERR_ARGUMENT);
    }
    {
        enum { B = 1, C = 2, H = 4, W = 5, OH = 2, OW = 3 };
        float x[B*C*H*W], expected[B*C*OH*OW], actual[B*C*OH*OW];
        size_t i, c, oy, ox, ky, kx;
        for (i = 0; i < sizeof(x) / sizeof(x[0]); ++i)
            x[i] = ((int)(i % 17) - 8) * .125f;
        for (c = 0; c < C; ++c)
            for (oy = 0; oy < OH; ++oy)
                for (ox = 0; ox < OW; ++ox) {
                    float maximum = -INFINITY;
                    for (ky = 0; ky < 3; ++ky)
                        for (kx = 0; kx < 3; ++kx) {
                            long iy = (long)(oy * 2 + ky) - 1;
                            long ix = (long)(ox * 2 + kx) - 1;
                            if (iy >= 0 && ix >= 0 && (size_t)iy < H &&
                                (size_t)ix < W) {
                                float value = x[(c * H + (size_t)iy) * W +
                                                (size_t)ix];
                                if (value > maximum) maximum = value;
                            }
                        }
                    expected[(c * OH + oy) * OW + ox] = maximum;
                }
        assert(ua_cuda_test_maxpool2d_fp16(
            x, actual, B, C, H, W, 3, 3, 2, 2, 1, 1) == UA_OK);
        for (i = 0; i < sizeof(actual) / sizeof(actual[0]); ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
    }
    {
        enum { ROWS = 3, DIM = 256 };
        float x[ROWS * DIM], gamma[DIM], beta[DIM];
        float expected[ROWS * DIM], actual[ROWS * DIM];
        size_t i;
        for (i = 0; i < ROWS * DIM; ++i)
            x[i] = sinf((float)i * .03125f) + (float)(i % 5) * .01f;
        for (i = 0; i < DIM; ++i) {
            gamma[i] = .75f + (float)(i % 7) * .03125f;
            beta[i] = ((int)(i % 5) - 2) * .015625f;
        }
        ua_op_layer_norm(x, expected, ROWS, DIM, 1e-5f);
        for (i = 0; i < ROWS * DIM; ++i)
            expected[i] = expected[i] * gamma[i % DIM] + beta[i % DIM];
        assert(ua_cuda_test_layer_norm_fp16(
            x, gamma, beta, actual, ROWS, DIM, 1e-5f) == UA_OK);
        for (i = 0; i < ROWS * DIM; ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        assert(ua_cuda_test_layer_norm_fp16(
            x, gamma, beta, actual, 0, DIM, 1e-5f) == UA_ERR_ARGUMENT);
        assert(ua_cuda_test_layer_norm_fp16(
            x, gamma, NULL, actual, ROWS, DIM, 1e-5f) == UA_ERR_ARGUMENT);
    }
    {
        enum { ROWS = 4, COLS = 17 };
        float input[ROWS * COLS], expected[ROWS * COLS], actual[ROWS * COLS];
        size_t i;
        for (i = 0; i < ROWS * COLS; ++i)
            input[i] = 1000.0f + (float)((int)(i % COLS) - 8) * .25f;
        memcpy(expected, input, sizeof(input));
        ua_op_softmax(expected, ROWS, COLS);
        assert(ua_cuda_test_softmax_f32(
            input, actual, ROWS, COLS) == UA_OK);
        for (i = 0; i < ROWS * COLS; ++i)
            assert(close_enough(actual[i], expected[i], 1e-4f, 1e-3f));
        input[4] = INFINITY;
        assert(ua_cuda_test_softmax_f32(
            input, actual, ROWS, COLS) == UA_ERR_NONFINITE);
    }
    {
        float source[12], expected[35], actual[35];
        size_t i;
        for (i = 0; i < 12; ++i) source[i] = (float)i * .25f - 1.0f;
        ua_op_resize_bilinear(source, 3, 4, expected, 5, 7);
        assert(ua_cuda_test_resize_bilinear_f32(
            source, 3, 4, actual, 5, 7) == UA_OK);
        for (i = 0; i < 35; ++i)
            assert(close_enough(actual[i], expected[i], 1e-4f, 1e-3f));
    }
    {
        float map[4 * 5 * 3], points[] = {
            0.0f, 0.0f, 1.25f, 2.75f, -.25f, 1.5f, 3.8f, 4.2f
        };
        float expected[4 * 3], actual[4 * 3];
        size_t i;
        for (i = 0; i < sizeof(map) / sizeof(map[0]); ++i)
            map[i] = ((int)(i % 17) - 8) * .125f;
        ua_op_deform_sample(map, 4, 5, 3, points, 4, expected);
        assert(ua_cuda_test_deform_sample_f32(
            map, 4, 5, 3, points, 4, actual) == UA_OK);
        for (i = 0; i < 12; ++i)
            assert(close_enough(actual[i], expected[i], 1e-4f, 1e-3f));
    }
    {
        const float score[] = {2, 5, 5, -1, 3, 5};
        size_t expected[4], actual[4], i;
        ua_op_stable_topk(score, 6, 4, expected);
        assert(ua_cuda_test_stable_topk_f32(score, 6, 4, actual) == UA_OK);
        for (i = 0; i < 4; ++i) assert(actual[i] == expected[i]);
        assert(ua_cuda_test_stable_topk_f32(
            score, 6, 7, actual) == UA_ERR_ARGUMENT);
    }
    {
        float logits[5 * 3] = {
            0, 0, -1,
            -2, 2, 1,
            -4, -4, -4,
            -1.3249254f, -2, -3,
            10, 9, 8
        };
        const uint32_t expected_classes[5] = {0, 1, 0, 0, 0};
        const float expected_scores[5] = {
            .5f, .880797078f, .01798621f, .21f, .999954602f
        };
        float scores[5];
        uint32_t classes[5], selected[2];
        size_t selected_count = 0, i;
        assert(ua_cuda_test_track_score_filter_fp16(
            logits, 5, 3, .2f, 2, scores, classes, selected,
            &selected_count) == UA_OK);
        for (i = 0; i < 5; ++i) {
            assert(classes[i] == expected_classes[i]);
            assert(close_enough(
                scores[i], expected_scores[i], 5e-3f, 1e-2f));
        }
        assert(selected_count == 2 && selected[0] == 0 && selected[1] == 1);
        logits[0] = NAN;
        assert(ua_cuda_test_track_score_filter_fp16(
            logits, 5, 3, .2f, 2, scores, classes, selected,
            &selected_count) == UA_ERR_NONFINITE);
    }
    {
        float scores[6] = {.4f, .399f, .34f, .36f, .99f, .8f};
        int32_t object_ids[6] = {-1, -1, 5, 6, -2, -1};
        uint32_t disappear[6] = {0, 0, 4, 0, 0, 0};
        uint32_t active[3];
        int32_t next_id = 7;
        size_t active_count = 0, frame;
        assert(ua_cuda_test_tracker_state_update(
            scores, object_ids, disappear, 6, .4f, .35f, 5, &next_id,
            active, 3, &active_count) == UA_OK);
        assert(next_id == 9);
        assert(object_ids[0] == 7 && object_ids[1] == -1);
        assert(object_ids[2] == -1 && disappear[2] == 5);
        assert(object_ids[3] == 6 && object_ids[4] == -2);
        assert(object_ids[5] == 8);
        assert(active_count == 3);
        assert(active[0] == 0 && active[1] == 3 && active[2] == 5);
        scores[0] = scores[3] = scores[5] = .34f;
        for (frame = 0; frame < 5; ++frame)
            assert(ua_cuda_test_tracker_state_update(
                scores, object_ids, disappear, 6, .4f, .35f, 5, &next_id,
                active, 3, &active_count) == UA_OK);
        assert(object_ids[0] == -1 && object_ids[3] == -1 &&
               object_ids[5] == -1);
        assert(active_count == 0 && next_id == 9);
        scores[0] = NAN;
        assert(ua_cuda_test_tracker_state_update(
            scores, object_ids, disappear, 6, .4f, .35f, 5, &next_id,
            active, 3, &active_count) == UA_ERR_NONFINITE);
    }
    {
        float embedding[4] = {1, 2, 3, 4};
        float scores[2] = {.5f, .9f};
        float weight[4] = {1, 0, 0, 1};
        float bias[2] = {.5f, -.5f};
        float memory[2 * 3 * 2] = {
            1,10, 2,20, 3,30,
            4,40, 5,50, 6,60
        };
        const float expected_first[2 * 3 * 2] = {
            2,20, 3,30, 1.5f,1.5f,
            4,40, 5,50, 6,60
        };
        uint8_t mask[2 * 3] = {0,0,1, 1,1,1};
        uint8_t period[2] = {0,2};
        size_t i;
        assert(ua_cuda_test_memory_bank_update_fp16(
            embedding, scores, weight, bias, memory, mask, period,
            2, 3, 2, 0.0f, 3) == UA_OK);
        for (i = 0; i < 12; ++i)
            assert(close_enough(
                memory[i], expected_first[i], 5e-3f, 1e-2f));
        assert(mask[0] == 0 && mask[1] == 1 && mask[2] == 0);
        assert(mask[3] == 1 && mask[4] == 1 && mask[5] == 1);
        assert(period[0] == 3 && period[1] == 1);
        assert(ua_cuda_test_memory_bank_update_fp16(
            embedding, scores, weight, bias, memory, mask, period,
            2, 3, 2, 0.0f, 3) == UA_OK);
        assert(period[0] == 2 && period[1] == 0);
        assert(ua_cuda_test_memory_bank_update_fp16(
            embedding, scores, weight, bias, memory, mask, period,
            2, 3, 2, 0.0f, 3) == UA_OK);
        assert(period[0] == 1 && period[1] == 3);
        assert(mask[3] == 1 && mask[4] == 1 && mask[5] == 0);
        assert(close_enough(memory[10], 3.5f, 5e-3f, 1e-2f));
        assert(close_enough(memory[11], 3.5f, 5e-3f, 1e-2f));
        scores[0] = NAN;
        assert(ua_cuda_test_memory_bank_update_fp16(
            embedding, scores, weight, bias, memory, mask, period,
            2, 3, 2, 0.0f, 3) == UA_ERR_NONFINITE);
    }
    {
        enum {
            BATCHES = 2, TOTAL = 8, HEADS = 2, CHANNELS = 4,
            LEVELS = 2, QUERIES = 3, POINTS = 2
        };
        const uint32_t shapes[LEVELS * 2] = {2, 3, 1, 2};
        const size_t starts[LEVELS] = {0, 6};
        float value[BATCHES * TOTAL * HEADS * CHANNELS];
        float locations[BATCHES * QUERIES * HEADS * LEVELS * POINTS * 2];
        float weights[BATCHES * QUERIES * HEADS * LEVELS * POINTS];
        float expected[BATCHES * QUERIES * HEADS * CHANNELS];
        float actual[BATCHES * QUERIES * HEADS * CHANNELS];
        size_t i, b, q, h, c, l, p;
        for (i = 0; i < sizeof(value) / sizeof(value[0]); ++i)
            value[i] = ((int)(i % 19) - 9) * .0625f;
        for (i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
            weights[i] = .25f;
        for (i = 0; i < sizeof(locations) / sizeof(locations[0]); i += 2) {
            locations[i] = ((float)((i / 2) % 7) - 1.0f) / 5.0f;
            locations[i + 1] = ((float)((i / 2) % 5) + .5f) / 4.0f;
        }
        for (b = 0; b < BATCHES; ++b)
            for (q = 0; q < QUERIES; ++q)
                for (h = 0; h < HEADS; ++h)
                    for (c = 0; c < CHANNELS; ++c) {
                        float sum = 0.0f;
                        for (l = 0; l < LEVELS; ++l)
                            for (p = 0; p < POINTS; ++p) {
                                size_t sample =
                                    ((((b * QUERIES + q) * HEADS + h) *
                                      LEVELS + l) * POINTS + p);
                                sum += reference_ms_sample(
                                    value, b, TOTAL, HEADS, CHANNELS, h, c,
                                    starts[l], shapes[l * 2], shapes[l * 2 + 1],
                                    locations[sample * 2],
                                    locations[sample * 2 + 1]) *
                                    weights[sample];
                            }
                        expected[((b * QUERIES + q) * HEADS + h) *
                                 CHANNELS + c] = sum;
                    }
        assert(ua_cuda_test_ms_deform_attn_fp16(
            value, BATCHES, TOTAL, HEADS, CHANNELS, shapes, starts, LEVELS,
            locations, weights, QUERIES, POINTS, actual) == UA_OK);
        for (i = 0; i < sizeof(actual) / sizeof(actual[0]); ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        {
            const size_t bad_starts[LEVELS] = {0, 7};
            assert(ua_cuda_test_ms_deform_attn_fp16(
                value, BATCHES, TOTAL, HEADS, CHANNELS, shapes, bad_starts,
                LEVELS, locations, weights, QUERIES, POINTS, actual) ==
                UA_ERR_ARGUMENT);
        }
    }
    {
        enum { B = 1, Q = 2, H = 2, L = 2, P = 3, COUNT = B*Q*H*L*P*2 };
        const uint32_t shapes[L * 2] = {4, 8, 5, 10};
        float reference2[B * Q * L * 2], reference4[B * Q * L * 4];
        float offsets[COUNT], expected[COUNT], actual[COUNT];
        size_t i, d, sample, coordinate, level, quotient, query;
        for (i = 0; i < sizeof(reference2) / sizeof(reference2[0]); ++i)
            reference2[i] = .1f + (float)(i % 7) * .1f;
        for (i = 0; i < sizeof(reference4) / sizeof(reference4[0]); ++i)
            reference4[i] = .1f + (float)(i % 5) * .125f;
        for (i = 0; i < COUNT; ++i)
            offsets[i] = ((int)(i % 9) - 4) * .25f;
        for (d = 0; d < 2; ++d) {
            const float *reference = d ? reference4 : reference2;
            size_t dimensions = d ? 4 : 2;
            for (i = 0; i < COUNT; ++i) {
                coordinate = i & 1u;
                sample = i >> 1u;
                quotient = sample / P;
                level = quotient % L;
                quotient /= L * H;
                query = quotient % Q;
                {
                    size_t ri = (query * L + level) * dimensions;
                    if (dimensions == 2)
                        expected[i] = reference[ri + coordinate] + offsets[i] /
                            (float)shapes[level * 2 + (coordinate ? 0 : 1)];
                    else
                        expected[i] = reference[ri + coordinate] +
                            offsets[i] / (float)P *
                            reference[ri + 2 + coordinate] * .5f;
                }
            }
            assert(ua_cuda_test_deform_locations_f32(
                reference, dimensions, offsets, shapes, B, Q, H, L, P,
                actual) == UA_OK);
            for (i = 0; i < COUNT; ++i)
                assert(close_enough(actual[i], expected[i], 1e-6f, 1e-6f));
        }
    }
    {
        enum { CAMERAS = 3, MAXQ = 4, QUERIES = 5, DIM = 3 };
        const uint32_t indices[CAMERAS * MAXQ] = {
            0, 2, 4, 0, 1, 2, 0, 0, 2, 3, 4, 0
        };
        const uint32_t counts[CAMERAS] = {3, 2, 3};
        float slots[CAMERAS * MAXQ * DIM], expected[QUERIES * DIM] = {0};
        float actual[QUERIES * DIM];
        size_t observations[QUERIES] = {0}, camera, slot, d, i;
        for (i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i)
            slots[i] = (float)i * .125f;
        for (camera = 0; camera < CAMERAS; ++camera)
            for (slot = 0; slot < counts[camera]; ++slot) {
                size_t query = indices[camera * MAXQ + slot];
                ++observations[query];
                for (d = 0; d < DIM; ++d)
                    expected[query * DIM + d] +=
                        slots[(camera * MAXQ + slot) * DIM + d];
            }
        for (i = 0; i < QUERIES; ++i)
            if (observations[i])
                for (d = 0; d < DIM; ++d)
                    expected[i * DIM + d] /= (float)observations[i];
        assert(ua_cuda_test_camera_scatter_average_f32(
            slots, indices, counts, CAMERAS, MAXQ, QUERIES, DIM, actual) ==
            UA_OK);
        for (i = 0; i < QUERIES * DIM; ++i)
            assert(close_enough(actual[i], expected[i], 1e-6f, 1e-6f));
    }
    {
        enum { B = 2, Q = 3, D = 4, QUEUE = 2 };
        float input[B * QUEUE * Q * D], expected[B * Q * D];
        float actual[B * Q * D];
        size_t b, q, d, t, i;
        for (i = 0; i < sizeof(input) / sizeof(input[0]); ++i)
            input[i] = ((int)(i % 13) - 6) * .25f;
        for (b = 0; b < B; ++b)
            for (q = 0; q < Q; ++q)
                for (d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (t = 0; t < QUEUE; ++t)
                        sum += input[((b * QUEUE + t) * Q + q) * D + d];
                    expected[(b * Q + q) * D + d] = sum / (float)QUEUE;
                }
        assert(ua_cuda_test_queue_mean_f32(
            input, B, Q, D, QUEUE, actual) == UA_OK);
        for (i = 0; i < B * Q * D; ++i)
            assert(close_enough(actual[i], expected[i], 1e-6f, 1e-6f));
    }
    {
        enum {
            N = 1, IC = 2, IH = 4, IW = 5, OC = 3, KH = 3, KW = 3,
            OH = 4, OW = 5, KE = 9
        };
        float x[N * IC * IH * IW], offset[N * 2 * KE * OH * OW];
        float mask[N * KE * OH * OW], weight[OC * IC * KH * KW], bias[OC];
        float expected[N * OC * OH * OW], actual[N * OC * OH * OW];
        size_t i, n, oc, oy, ox, ic, ky, kx;
        for (i = 0; i < sizeof(x) / sizeof(x[0]); ++i)
            x[i] = ((int)(i % 17) - 8) * .125f;
        for (i = 0; i < sizeof(weight) / sizeof(weight[0]); ++i)
            weight[i] = ((int)(i % 7) - 3) * .0625f;
        for (i = 0; i < sizeof(offset) / sizeof(offset[0]); ++i) {
            size_t channel = (i / (OH * OW)) % (2 * KE);
            offset[i] = ((int)(i % 5) - 2) * .0625f +
                        ((int)(channel % 7) - 3) * .03125f;
        }
        for (i = 0; i < sizeof(mask) / sizeof(mask[0]); ++i)
            mask[i] = .25f + (float)(i % 4) * .125f;
        for (i = 0; i < OC; ++i) bias[i] = (float)i * .03125f;
        for (n = 0; n < N; ++n)
            for (oc = 0; oc < OC; ++oc)
                for (oy = 0; oy < OH; ++oy)
                    for (ox = 0; ox < OW; ++ox) {
                        float sum = bias[oc];
                        for (ic = 0; ic < IC; ++ic)
                            for (ky = 0; ky < KH; ++ky)
                                for (kx = 0; kx < KW; ++kx) {
                                    size_t k = ky * KW + kx;
                                    size_t off_y =
                                        ((n * 2 * KE + 2 * k) * OH + oy) *
                                        OW + ox;
                                    size_t off_x =
                                        off_y + OH * OW;
                                    size_t mi =
                                        ((n * KE + k) * OH + oy) * OW + ox;
                                    float sy = (float)oy + (float)ky - 1.0f +
                                               offset[off_y];
                                    float sx = (float)ox + (float)kx - 1.0f +
                                               offset[off_x];
                                    sum += reference_nchw_sample(
                                               x, n, ic, IC, IH, IW, sy, sx) *
                                           mask[mi] *
                                           weight[((oc * IC + ic) * KH + ky) *
                                                  KW + kx];
                                }
                        expected[((n * OC + oc) * OH + oy) * OW + ox] = sum;
                    }
        {
            /* Generated by the pinned mmcv modulated_deform_conv2d CUDA op. */
            const float mmcv_prefix[] = {
                -.0014762878418f, -.073802947998f, -.115325927734f,
                -.148420333862f, -.0310173034668f, -.0205535888672f,
                .0217514038086f, .0367259979248f, .00865936279297f,
                .102224349976f, -.0606002807617f, .302219390869f,
            };
            for (i = 0; i < sizeof(mmcv_prefix) / sizeof(mmcv_prefix[0]); ++i)
                assert(close_enough(expected[i], mmcv_prefix[i], 1e-6f, 1e-6f));
        }
        assert(ua_cuda_test_modulated_deform_conv2d_fp16(
            x, offset, mask, weight, bias, actual, N, IC, IH, IW, OC, KH, KW,
            1, 1, 1, 1, 1, 1) == UA_OK);
        for (i = 0; i < sizeof(actual) / sizeof(actual[0]); ++i)
            assert(close_enough(actual[i], expected[i], 5e-3f, 1e-2f));
        offset[0] = NAN;
        assert(ua_cuda_test_modulated_deform_conv2d_fp16(
            x, offset, mask, weight, bias, actual, N, IC, IH, IW, OC, KH, KW,
            1, 1, 1, 1, 1, 1) == UA_ERR_NONFINITE);
    }
    {
        const char *production_model = getenv("UA_TEST_PRODUCTION_MODEL");
        if (production_model && *production_model) {
            enum { WIDTH = 1600, HEIGHT = 900, STRIDE = WIDTH * 3 };
            ua_model *model = NULL;
            ua_context *context = NULL;
            ua_production_input input;
            ua_production_result *result;
            ua_metrics metrics;
            float stem_sample[32];
            size_t stem_written = 0, sample_index;
            uint8_t *image = (uint8_t *)calloc(HEIGHT, STRIDE);
            size_t camera;
            assert(image);
            result = (ua_production_result *)calloc(1, sizeof(*result));
            assert(result);
            memset(&input, 0, sizeof(input));
            input.version = 2;
            input.scene_token = "production-stem-fixture";
            for (camera = 0; camera < UA_CAMERA_COUNT; ++camera) {
                input.cameras[camera].data = image;
                input.cameras[camera].width = WIDTH;
                input.cameras[camera].height = HEIGHT;
                input.cameras[camera].row_stride_bytes = STRIDE;
                input.camera_intrinsics[camera][0] = 1000.0f;
                input.camera_intrinsics[camera][2] = 800.0f;
                input.camera_intrinsics[camera][4] = 1000.0f;
                input.camera_intrinsics[camera][5] = 450.0f;
                input.camera_intrinsics[camera][8] = 1.0f;
                input.camera_to_ego[camera][0] = 1.0f;
                input.camera_to_ego[camera][5] = 1.0f;
                input.camera_to_ego[camera][10] = 1.0f;
                input.camera_to_ego[camera][15] = 1.0f;
            }
            input.ego_pose[0] = 1.0f;
            input.ego_pose[5] = 1.0f;
            input.ego_pose[10] = 1.0f;
            input.ego_pose[15] = 1.0f;
            assert(ua_model_load(production_model, &model) == UA_OK);
            assert(ua_context_create(model, UA_BACKEND_CUDA, &context) == UA_OK);
            {
                ua_status production_status =
                    ua_infer_production(context, &input, result);
                if (production_status != UA_ERR_UNSUPPORTED_PROFILE)
                    fprintf(stderr, "production backbone status: %s\n",
                            ua_status_string(production_status));
                assert(production_status == UA_ERR_UNSUPPORTED_PROFILE);
            }
            if (getenv("UA_TEST_QUERY_INTERACTION")) {
                static const float qim_oracle[32] = {
                    -.480041057f,-.26444298f,-1.58191895f,-1.77307022f,
                    -2.78211832f,.642766774f,-.503648639f,-1.88503361f,
                    -.149163365f,-.95354414f,-1.84735954f,.474475414f,
                    .262166023f,1.19441462f,.311266959f,.951873243f,
                    .313104481f,.659649968f,.425093383f,-.794623673f,
                    .479144484f,.234012112f,-1.7247541f,-.124252513f,
                    -.445929378f,.157601461f,-.994906545f,
                    -.00115608796f,-1.54128075f,-2.1895988f,
                    .980407298f,-.962290525f
                };
                ua_status qim_status =
                    ua_context_debug_run_query_interaction(context, 32);
                if (qim_status != UA_OK)
                    fprintf(stderr, "query interaction status: %s\n",
                            ua_status_string(qim_status));
                assert(qim_status == UA_OK);
                assert(ua_context_debug_tensor_f32(
                    context, "production.track.query_interaction",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index], qim_oracle[sample_index],
                        5e-3f, 1e-2f));
            }
            ua_context_metrics(context, &metrics);
            assert(metrics.h2d_bytes ==
                   (size_t)UA_CAMERA_COUNT * WIDTH * HEIGHT * 3u +
                   UA_PRODUCTION_METADATA_BYTES);
            {
                const float block_oracle[2][32] = {
                    {
                        0, 0, .833156586f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f, .645292759f,
                        .645292759f, .645292759f, .645292759f
                    },
                    {
                        .345172673f, .308965772f, 1.709529877f,
                        1.575488210f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f, 1.492425680f,
                        1.492425680f, 1.492425680f
                    }
                };
                const char *block_names[2] = {
                    "production.resnet_layer1.block0",
                    "production.resnet_layer1.block1"
                };
                size_t block;
                for (block = 0; block < 2; ++block) {
                    assert(ua_context_debug_tensor_f32(
                        context, block_names[block], stem_sample, 32,
                        &stem_written) == UA_OK);
                    assert(stem_written == 32);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            block_oracle[block][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            {
                const float layer1_block2_oracle[32] = {
                    0.0f, 0.0f, .858774781f, 4.181507587f,
                    4.995261669f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f,
                    4.924468040f, 4.924468040f, 4.924468040f, 4.924468040f
                };
                size_t failures = 0, first_failure = 32;
                assert(ua_context_debug_tensor_f32(
                    context, "production.resnet_layer1.block2", stem_sample,
                    32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index) {
                    if (!close_enough(
                            stem_sample[sample_index],
                            layer1_block2_oracle[sample_index],
                            5e-3f, 1e-2f)) {
                        if (!failures) first_failure = sample_index;
                        ++failures;
                    }
                }
                assert(failures == 1 && first_failure == 2);
                assert(close_enough(
                    stem_sample[2], .882324219f, 1e-6f, 1e-6f));
            }
            assert(ua_context_debug_tensor_f32(
                context, "production.resnet_layer1", stem_sample,
                sizeof(stem_sample) / sizeof(stem_sample[0]),
                &stem_written) == UA_ERR_PROFILE);
            {
                float layer2_oracle[4][32] = {{0}};
                size_t block, failures, first_failure;
                layer2_oracle[0][2] = .628752112f;
                layer2_oracle[1][0] = 0.0f;
                layer2_oracle[1][1] = .0301198959f;
                layer2_oracle[1][2] = 1.52631879f;
                layer2_oracle[1][3] = 2.68442869f;
                layer2_oracle[1][4] = 2.61065102f;
                layer2_oracle[1][5] = 2.18902755f;
                layer2_oracle[1][6] = 2.23458219f;
                for (sample_index = 7; sample_index < 32; ++sample_index)
                    layer2_oracle[1][sample_index] = 2.23773408f;
                layer2_oracle[2][2] = .100334503f;
                layer2_oracle[3][0] = 0.0f;
                layer2_oracle[3][1] = .00414437521f;
                layer2_oracle[3][2] = .0895032585f;
                layer2_oracle[3][3] = .103559129f;
                layer2_oracle[3][4] = .111526914f;
                for (sample_index = 5; sample_index < 32; ++sample_index)
                    layer2_oracle[3][sample_index] = .112057783f;
                for (block = 0; block < 4; ++block) {
                    char boundary_name[64];
                    (void)snprintf(
                        boundary_name, sizeof(boundary_name),
                        "production.resnet_layer2.block%zu", block);
                    assert(ua_context_debug_tensor_f32(
                        context, boundary_name, stem_sample, 32,
                        &stem_written) == UA_OK);
                    failures = 0;
                    first_failure = 32;
                    for (sample_index = 0; sample_index < 32;
                         ++sample_index) {
                        if (!close_enough(
                                stem_sample[sample_index],
                                layer2_oracle[block][sample_index],
                                5e-3f, 1e-2f)) {
                            if (!failures) first_failure = sample_index;
                            ++failures;
                        }
                    }
                    if (block < 3)
                        assert(failures == 0);
                    else
                        assert(failures == 31 && first_failure == 1);
                }
            }
            {
                float layer2_final_oracle[32] = {0};
                size_t failures = 0, first_failure = 32;
                layer2_final_oracle[3] = .00414437521f;
                layer2_final_oracle[4] = .0895032585f;
                layer2_final_oracle[5] = .103559129f;
                layer2_final_oracle[6] = .111526914f;
                for (sample_index = 7; sample_index < 32; ++sample_index)
                    layer2_final_oracle[sample_index] = .112057783f;
                assert(ua_context_debug_tensor_f32(
                    context, "production.resnet_layer2.snapshot", stem_sample,
                    32,
                    &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index) {
                    if (!close_enough(
                            stem_sample[sample_index],
                            layer2_final_oracle[sample_index],
                            5e-3f, 1e-2f)) {
                        if (!failures) first_failure = sample_index;
                        ++failures;
                    }
                }
                assert(failures == 29 && first_failure == 3);
                assert(ua_context_debug_tensor_f32(
                    context, "production.resnet_layer2", stem_sample, 1,
                    &stem_written) == UA_ERR_PROFILE);
            }
            if (getenv("UA_DUMP_PRODUCTION_BOUNDARIES")) {
                const char *layer2_names[41] = {
                    "production.resnet_layer2.block0",
                    "production.resnet_layer2.block1",
                    "production.resnet_layer2.block2",
                    "production.resnet_layer2.block3",
                    "production.resnet_layer2.snapshot",
                    "production.resnet_layer3.block0",
                    "production.resnet_layer3.block22",
                    "production.resnet_layer3.first_nonfinite",
                    "production.resnet_layer4.block0",
                    "production.resnet_layer4.block2",
                    "production.resnet_layer4.first_nonfinite",
                    "production.fpn.0",
                    "production.fpn.1",
                    "production.fpn.2",
                    "production.fpn.3",
                    "production.bevformer.feat_flatten.level0",
                    "production.bevformer.feat_flatten.level1",
                    "production.bevformer.feat_flatten.level2",
                    "production.bevformer.feat_flatten.level3",
                    "production.bevformer.bev_queries",
                    "production.bevformer.bev_pos",
                    "production.bevformer.encoder.layer0.temporal",
                    "production.bevformer.encoder.layer0.norm0",
                    "production.bevformer.encoder.layer0.spatial",
                    "production.bevformer.encoder.layer0.norm1",
                    "production.bevformer.encoder.layer0.ffn",
                    "production.bevformer.encoder.layer0.norm2",
                    "production.bevformer.encoder.layer1.norm2",
                    "production.bevformer.encoder.layer2.norm2",
                    "production.bevformer.encoder.layer3.norm2",
                    "production.bevformer.encoder.layer4.norm2",
                    "production.bevformer.encoder.layer5.norm2",
                    "production.track.query_pos",
                    "production.track.query",
                    "production.track.reference_points",
                    "production.track.decoder.layer0.self",
                    "production.track.decoder.layer0.norm0",
                    "production.track.decoder.layer0.cross",
                    "production.track.decoder.layer0.norm1",
                    "production.track.decoder.layer0.ffn",
                    "production.track.decoder.layer0.norm2"
                };
                size_t boundary;
                for (boundary = 0; boundary < 41; ++boundary) {
                    assert(ua_context_debug_tensor_f32(
                        context, layer2_names[boundary], stem_sample, 32,
                        &stem_written) == UA_OK);
                    fprintf(stderr, "%s", layer2_names[boundary]);
                    for (sample_index = 0; sample_index < stem_written;
                         ++sample_index)
                        fprintf(stderr, " %.9g", stem_sample[sample_index]);
                    fprintf(stderr, "\n");
                }
                {
                    const char *track_stages[8] = {
                        "self", "norm0", "cross", "norm1", "ffn", "norm2",
                        "regression", "reference"
                    };
                    size_t layer, stage;
                    for (layer = 0; layer < 6; ++layer) {
                        for (stage = 0; stage < 8; ++stage) {
                            char boundary_name[80];
                            assert(snprintf(
                                boundary_name, sizeof(boundary_name),
                                "production.track.decoder.layer%zu.%s",
                                layer, track_stages[stage]) > 0);
                            assert(ua_context_debug_tensor_f32(
                                context, boundary_name, stem_sample, 32,
                                &stem_written) == UA_OK);
                            fprintf(stderr, "%s", boundary_name);
                            for (sample_index = 0;
                                 sample_index < stem_written;
                                 ++sample_index)
                                fprintf(
                                    stderr, " %.9g",
                                    stem_sample[sample_index]);
                            fprintf(stderr, "\n");
                        }
                    }
                    {
                        const char *output_names[7] = {
                            "production.track.outputs.class_logits",
                            "production.track.outputs.box",
                            "production.track.outputs.past_trajectory",
                            "production.track.decode.scores",
                            "production.track.decode.classes",
                            "production.track.decode.selected_count",
                            "production.track.query_interaction"
                        };
                        size_t output_count =
                            getenv("UA_TEST_QUERY_INTERACTION") ? 7u : 6u;
                        for (stage = 0; stage < output_count; ++stage) {
                            assert(ua_context_debug_tensor_f32(
                                context, output_names[stage], stem_sample,
                                32, &stem_written) == UA_OK);
                            fprintf(stderr, "%s", output_names[stage]);
                            for (sample_index = 0;
                                 sample_index < stem_written;
                                 ++sample_index)
                                fprintf(
                                    stderr, " %.9g",
                                    stem_sample[sample_index]);
                            fprintf(stderr, "\n");
                        }
                    }
                }
            }
            assert(ua_context_debug_tensor_f32(
                context, "production.resnet_stem", stem_sample, 1,
                &stem_written) == UA_ERR_PROFILE);
            assert(ua_context_debug_tensor_f32(
                context, "production.resnet_layer3.first_nonfinite",
                stem_sample, 1, &stem_written) == UA_OK);
            assert(stem_written == 1);
            assert(stem_sample[0] == 23.0f);
            {
                const char *layer3_names[2] = {
                    "production.resnet_layer3.block0",
                    "production.resnet_layer3.block22"
                };
                const float expected_nonzero[2] = {
                    .195338011f, .211586952f
                };
                size_t boundary;
                for (boundary = 0; boundary < 2; ++boundary) {
                    assert(ua_context_debug_tensor_f32(
                        context, layer3_names[boundary], stem_sample, 32,
                        &stem_written) == UA_OK);
                    assert(stem_written == 32);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            sample_index == 2
                                ? expected_nonzero[boundary] : 0.0f,
                            5e-3f, 1e-2f));
                }
            }
            {
                static const float norm2_oracle[5][32] = {
                    {
                        .365182191f,.716472685f,.113740802f,-1.15680444f,
                        -2.24810958f,-.0331797935f,-.468218327f,1.68069375f,
                        -1.870579f,-.271288246f,.0829086155f,-.416826606f,
                        1.2438314f,-.964164793f,1.04254806f,-1.53668618f,
                        -.248575822f,-.155470148f,.310545444f,-1.52540445f,
                        -2.35899329f,-1.22509897f,2.38088083f,-.125471935f,
                        -.322500765f,.097097747f,-2.52393532f,.32806322f,
                        -.62166357f,-.576438606f,-.494439334f,.879437089f
                    }, {
                        .376192987f,1.59567928f,.546116352f,-1.28681016f,
                        -1.18423152f,-.0666632727f,.0508434474f,1.29400527f,
                        -1.92102969f,.348996311f,.212038994f,.441947341f,
                        .98786509f,-.479496807f,1.09363353f,-2.69591022f,
                        -.382519096f,.234062865f,.0487613119f,-1.08449411f,
                        -2.31521249f,-1.01563239f,2.33713961f,.506079555f,
                        .519185543f,.72645396f,-2.7828815f,-.105427884f,
                        -.953949332f,-.2748487f,.496153593f,1.64286661f
                    }, {
                        .260979176f,1.53734934f,.685787678f,-1.55522871f,
                        -.381449074f,.0313585699f,.283426315f,1.11014438f,
                        -2.82386065f,.288658798f,-.805540919f,1.40855849f,
                        .723279953f,-1.28077149f,1.20845318f,-2.47253633f,
                        -.123326488f,-.453065276f,.400383621f,-.523645282f,
                        -1.92091155f,-1.59763944f,2.63357115f,-.226954907f,
                        .825895905f,.600948155f,-2.16136813f,-.235225186f,
                        -1.20675516f,-.348280132f,-.266797036f,.938477218f
                    }, {
                        .0884156525f,.983391762f,-.325036019f,-1.92665565f,
                        -.484079808f,-.334357917f,1.08940506f,.612646759f,
                        -2.94007921f,.133449912f,-.502218246f,.590327859f,
                        .529934943f,-1.80609536f,.542647481f,-2.2260344f,
                        .318092346f,-.714729369f,.742932737f,-.167987347f,
                        -1.81501698f,-1.47336638f,2.25015688f,-.424363643f,
                        .123022117f,.129111364f,-.870982766f,.0632266328f,
                        -.597901642f,-.557897925f,-1.04274774f,.806977272f
                    }, {
                        -.322827667f,.251432091f,.361058235f,-.642084181f,
                        .165644854f,.329870015f,.874504805f,.174950123f,
                        -1.16662288f,.344020873f,-.241397575f,.426225394f,
                        .257042855f,-1.76605678f,.236679628f,-1.4011296f,
                        .230596095f,-.594926894f,.81960851f,-.066196993f,
                        -.925047994f,-.996756196f,1.37400067f,-1.02062035f,
                        .281085461f,-.163471177f,-.508154809f,.360652059f,
                        -.628855765f,-.559954464f,-.76027602f,.97113812f
                    }
                };
                size_t layer;
                for (layer = 1; layer < 6; ++layer) {
                    char boundary_name[64];
                    assert(snprintf(
                        boundary_name, sizeof(boundary_name),
                        "production.bevformer.encoder.layer%zu.norm2",
                        layer) > 0);
                    assert(ua_context_debug_tensor_f32(
                        context, boundary_name, stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            norm2_oracle[layer - 1][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            {
                static const float track_oracle[3][32] = {
                    {
                        -.019995505f,2.68644309f,.127767727f,.129876271f,
                        .629823327f,.829940975f,2.11300611f,.377043426f,
                        -.461393118f,2.01579976f,1.92905796f,-.0922188163f,
                        -1.19840181f,-.543566287f,-.289550215f,-.485922784f,
                        1.93058205f,.792240262f,-.169381201f,-.102449089f,
                        -1.23494053f,1.0493505f,-1.75597894f,-.174156338f,
                        1.69554758f,-1.04936302f,.12369103f,.409826428f,
                        -.226473272f,-.679509044f,.906410754f,-.686806858f
                    }, {
                        -.0412175544f,.893709242f,-.943513095f,-1.19620049f,
                        -1.60823882f,-.308129877f,.692234993f,-1.04869771f,
                        .07020659f,-.614799201f,-.182818502f,-1.66987693f,
                        -.540927112f,.784246027f,-.756047547f,.314822704f,
                        1.60732281f,.34460035f,.969978571f,-1.40492761f,
                        -.465067565f,-1.23625636f,-.675288737f,.24529776f,
                        -1.02434206f,-.163407385f,-1.66289091f,-.214760289f,
                        -1.08037007f,1.68593013f,.976466477f,.823279738f
                    }, {
                        .599476337f,.358313262f,.484375954f,.479008347f,
                        .507003069f,.493813068f,.43891871f,.101546079f,
                        .445933431f,.390771151f,.2745727f,.470569909f,
                        .413366288f,.882731318f,.544006705f,.447766304f,
                        .534995139f,.49839139f,.678556979f,.55378145f,
                        .503202379f,.776703596f,.454459757f,.495550603f,
                        .614470541f,.55257982f,.503689766f,.245952711f,
                        .460132211f,.486274183f,.548692405f,.397038609f
                    }
                };
                const char *names[3] = {
                    "production.track.query_pos",
                    "production.track.query",
                    "production.track.reference_points"
                };
                size_t boundary;
                for (boundary = 0; boundary < 3; ++boundary) {
                    assert(ua_context_debug_tensor_f32(
                        context, names[boundary], stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            track_oracle[boundary][sample_index],
                            boundary == 2 ? 1e-4f : 5e-3f,
                            boundary == 2 ? 1e-3f : 1e-2f));
                }
            }
            {
                static const float decoder_oracle[6][32] = {
                    {
                        -1.39648068f,6.74305534f,-1.57274675f,-1.02275777f,
                        -2.10111809f,-.711836338f,-1.72470641f,1.98975563f,
                        -2.50292873f,.783831894f,-.715994596f,1.48135555f,
                        4.03156424f,-3.58632803f,-.839267135f,-.354890138f,
                        .903725207f,2.4147799f,1.18193924f,.761176944f,
                        5.39278841f,1.69868994f,-.489121556f,-.0511143506f,
                        -7.89391851f,-.925708592f,-2.51502037f,.928655148f,
                        -2.72278166f,7.81482601f,1.89969397f,-.869792581f
                    }, {
                        -.345181704f,1.95535028f,-.589376628f,-.214752764f,
                        -.814761281f,-.149017677f,-.715263128f,.574466407f,
                        -.893417597f,.258684009f,-.409581244f,.338310838f,
                        1.38797712f,-.88000083f,-.0524778217f,-.103285037f,
                        .21211271f,.445834279f,.151201621f,.237555474f,
                        1.93145919f,.302717149f,-.237591922f,.0135588786f,
                        -2.09691405f,-.211063206f,-.605902314f,.446329176f,
                        -.713951051f,2.32418704f,.629017591f,-.272524089f
                    }, {
                        -1.47887802f,1.33719766f,-.476191431f,1.35975349f,
                        -.332468808f,.402033448f,-1.30665278f,.347957969f,
                        -1.14059806f,.199007869f,-.66057241f,-.436721742f,
                        1.25410151f,-2.45021081f,.446362853f,1.28372121f,
                        .187837437f,-1.32498372f,-1.8251189f,-.752515912f,
                        2.86139441f,-.651250184f,-.0670932233f,.937495708f,
                        -.0391900539f,.802270353f,-.320339024f,-.146438777f,
                        -.806694388f,5.03931427f,-.262628675f,1.27826405f
                    }, {
                        -.861383557f,.743474722f,-.349768341f,1.00011706f,
                        -.317286462f,.283755034f,-.898197412f,.177299544f,
                        -.694895089f,.106661603f,-.419478595f,-.231983781f,
                        .873716593f,-1.275455f,.377126873f,.847783923f,
                        .103222184f,-.685621381f,-1.01320398f,-.45279026f,
                        1.65584993f,-.423345655f,-.0597154871f,.457588196f,
                        .0560931377f,.585676968f,-.203534648f,-.0134990057f,
                        -.35351935f,2.84320092f,-.0700803623f,.71786648f
                    }, {
                        -1.07703483f,.657032609f,-.542489707f,.556745708f,
                        -.656469345f,-.298930436f,-.9544487f,.586133957f,
                        -.704829812f,.727576375f,.598256171f,.051810056f,
                        1.46937418f,-1.68726277f,-.0382659435f,.323171139f,
                        1.17170835f,-.250080287f,-.423541188f,-.814526498f,
                        2.03233147f,.942438006f,.126963198f,.1287902f,
                        .650288641f,-.623818815f,-.729799807f,.411633313f,
                        -1.20334339f,4.2127018f,-1.98658013f,-.194687426f
                    }, {
                        -1.04925454f,.602662086f,-.536740482f,.490573883f,
                        -.632660031f,-.299591035f,-.839042962f,.564857602f,
                        -.63116318f,.675876737f,.611045182f,.0668837279f,
                        1.29798472f,-1.44659173f,.0635944605f,.243840665f,
                        1.11246157f,-.168572009f,-.296866f,-.722072661f,
                        1.85243261f,.850175381f,.0826230645f,.127298474f,
                        .534767926f,-.521520674f,-.674604714f,.395286053f,
                        -1.06211007f,3.95742178f,-1.76135159f,-.256041557f
                    }
                };
                const char *stages[6] = {
                    "self", "norm0", "cross", "norm1", "ffn", "norm2"
                };
                size_t stage;
                for (stage = 0; stage < 6; ++stage) {
                    char boundary_name[64];
                    assert(snprintf(
                        boundary_name, sizeof(boundary_name),
                        "production.track.decoder.layer0.%s",
                        stages[stage]) > 0);
                    assert(ua_context_debug_tensor_f32(
                        context, boundary_name, stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            decoder_oracle[stage][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            {
                static const float output_oracle[3][32] = {
                    {
                        -6.47830296f,-7.21787071f,-8.62879372f,
                        -8.55650425f,-9.53570366f,-7.94513226f,
                        -6.62638426f,-6.7707777f,-5.79417133f,
                        -6.75118589f,-5.76864338f,-6.53661156f,
                        -7.81865501f,-7.43006563f,-8.33129501f,
                        -7.41494799f,-6.57093334f,-6.61145782f,
                        -5.37166977f,-6.78034353f,-5.81477213f,
                        -6.49372721f,-7.73355198f,-7.58927011f,
                        -8.23321342f,-7.07921982f,-6.63757753f,
                        -6.57551622f,-5.35363579f,-6.56797647f,
                        -5.51920843f,-6.33017206f
                    }, {
                        31.8466911f,5.33250809f,-.551757812f,-.820800781f,
                        -1.4158175f,.496826172f,.914550781f,.0546264648f,
                        .000112235546f,.00991821289f,11.6469917f,
                        2.38884735f,-.30859375f,-.531738281f,-1.6587944f,
                        .477539062f,.942382812f,.0216674805f,
                        -.000792980194f,.00362586975f,-7.51194f,
                        -40.2396622f,-.473144531f,-.626464844f,
                        -2.33887482f,.497314453f,.920410156f,
                        .0299835205f,.00105953217f,.0238189697f,
                        -12.2572098f,-22.2995701f
                    }, {
                        -.121109322f,-.604824066f,-.236734152f,-1.22328281f,
                        -.250953048f,-1.87381029f,-.367615879f,-2.51606774f,
                        .951559126f,.970525503f,1.17511845f,1.44053543f,
                        1.40100336f,2.18087435f,1.52172637f,2.74268651f,
                        -.0782054812f,-.410984248f,-.188868999f,
                        -.861150742f,-.223970145f,-1.33497572f,
                        -.32375139f,-1.79404771f,.660236061f,2.37837887f,
                        .837831855f,2.61761308f,1.04194605f,3.18731999f,
                        1.19839716f,3.54531646f
                    }
                };
                const char *output_names[3] = {
                    "production.track.outputs.class_logits",
                    "production.track.outputs.box",
                    "production.track.outputs.past_trajectory"
                };
                size_t output_index;
                for (output_index = 0; output_index < 3; ++output_index) {
                    assert(ua_context_debug_tensor_f32(
                        context, output_names[output_index], stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            output_oracle[output_index][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            {
                static const float score_oracle[32] = {
                    .00303600752f,.00462487759f,.00470864261f,
                    .00399301201f,.00580037292f,.00420451723f,
                    .00458889455f,.00523489388f,.00320516806f,
                    .00408383971f,.0028443092f,.00533319917f,
                    .00424634293f,.00441478239f,.00443788804f,
                    .00440385565f,.00452490989f,.00546827866f,
                    .00454485184f,.00434400467f,.0044666999f,
                    .005035521f,.00440761633f,.00391780725f,
                    .00249342388f,.00411454774f,.00441532303f,
                    .00455597974f,.00543236127f,.00516259065f,
                    .00328814541f,.0047807591f
                };
                static const float class_oracle[32] = {
                    8,8,8,0,0,0,8,0,8,8,8,0,8,8,8,8,
                    0,8,8,8,8,8,8,8,8,8,8,8,8,0,8,8
                };
                assert(ua_context_debug_tensor_f32(
                    context, "production.track.decode.scores", stem_sample,
                    32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index], score_oracle[sample_index],
                        1e-4f, 1e-3f));
                assert(ua_context_debug_tensor_f32(
                    context, "production.track.decode.classes", stem_sample,
                    32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(stem_sample[sample_index] ==
                           class_oracle[sample_index]);
                assert(ua_context_debug_tensor_f32(
                    context, "production.track.decode.selected_count",
                    stem_sample, 1, &stem_written) == UA_OK);
                assert(stem_written == 1 && stem_sample[0] == 0.0f);
            }
            if (getenv("UA_DUMP_ENCODER_OUTPUT")) {
                const size_t encoder_count = 40000u * 256u;
                float *encoder_output =
                    (float *)malloc(encoder_count * sizeof(float));
                FILE *encoder_file;
                assert(encoder_output);
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.encoder.output",
                    encoder_output, encoder_count, &stem_written) == UA_OK);
                assert(stem_written == encoder_count);
                encoder_file = fopen(
                    getenv("UA_DUMP_ENCODER_OUTPUT"), "wb");
                assert(encoder_file);
                assert(fwrite(
                    encoder_output, sizeof(float), encoder_count,
                    encoder_file) == encoder_count);
                assert(fclose(encoder_file) == 0);
                free(encoder_output);
            }
            if (getenv("UA_DUMP_TRACK_HEAD_INPUTS")) {
                const char *tensor_names[3] = {
                    "production.track.decoder.layer5.state",
                    "production.track.decoder.layer5.regression",
                    "production.track.decoder.layer4.reference"
                };
                const char *suffixes[3] = {
                    ".state.f32", ".regression.f32", ".reference.f32"
                };
                const size_t counts[3] = {
                    901u * 256u, 901u * 10u, 901u * 3u
                };
                size_t tensor_index;
                for (tensor_index = 0; tensor_index < 3; ++tensor_index) {
                    float *tensor_output =
                        (float *)malloc(counts[tensor_index] * sizeof(float));
                    char output_path[512];
                    FILE *output_file;
                    assert(tensor_output);
                    assert(ua_context_debug_tensor_f32(
                        context, tensor_names[tensor_index], tensor_output,
                        counts[tensor_index], &stem_written) == UA_OK);
                    assert(stem_written == counts[tensor_index]);
                    assert(snprintf(
                        output_path, sizeof(output_path), "%s%s",
                        getenv("UA_DUMP_TRACK_HEAD_INPUTS"),
                        suffixes[tensor_index]) > 0);
                    output_file = fopen(output_path, "wb");
                    assert(output_file);
                    assert(fwrite(
                        tensor_output, sizeof(float), counts[tensor_index],
                        output_file) == counts[tensor_index]);
                    assert(fclose(output_file) == 0);
                    free(tensor_output);
                }
            }
            assert(ua_context_debug_tensor_f32(
                context, "production.resnet_layer4.first_nonfinite",
                stem_sample, 1, &stem_written) == UA_OK);
            assert(stem_written == 1 && stem_sample[0] == 3.0f);
            {
                const float layer4_block0_oracle[2] = {
                    1.35606754f, 0.0f
                };
                size_t failures = 0, first_failure = 32;
                assert(ua_context_debug_tensor_f32(
                    context, "production.resnet_layer4.block0", stem_sample,
                    32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index],
                        sample_index < 2
                            ? layer4_block0_oracle[sample_index] : 0.0f,
                        5e-3f, 1e-2f));
                assert(ua_context_debug_tensor_f32(
                    context, "production.resnet_layer4.block2", stem_sample,
                    32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index) {
                    float oracle = sample_index == 2 ? 1.65022302f :
                                   sample_index == 3 ? .634063601f : 0.0f;
                    if (!close_enough(
                            stem_sample[sample_index], oracle,
                            5e-3f, 1e-2f)) {
                        if (!failures) first_failure = sample_index;
                        ++failures;
                    }
                }
                assert(failures == 2 && first_failure == 2);
            }
            {
                static const float oracle[4][32] = {
                    {
                        -.214777857f,.600408435f,.835059881f,1.31620502f,
                        2.28122401f,2.87573075f,2.93372726f,3.36047101f,
                        3.78389955f,4.02893305f,4.08946705f,4.04948235f,
                        3.72128177f,3.51655436f,3.39026809f,3.36239719f,
                        3.55233359f,3.76678681f,3.78258204f,3.95397043f,
                        4.20297575f,4.4537158f,4.51033974f,4.71457243f,
                        4.87854195f,4.97354412f,5.02130413f,5.14165783f,
                        5.21574163f,5.25778055f,5.26724672f,5.29118776f
                    },
                    {
                        .0324153975f,-.688732088f,-.787896216f,-1.64638066f,
                        -1.41484261f,-1.0498482f,-1.23516583f,-.742133439f,
                        -.0820933208f,.109218232f,.572419047f,.920794368f,
                        1.06878078f,1.15949309f,1.27060044f,1.3400315f,
                        1.41376567f,1.39826679f,1.30792022f,1.08966911f,
                        1.06811512f,1.05539072f,1.22818816f,1.2769177f,
                        1.24111581f,1.16184211f,1.12968552f,1.09954727f,
                        1.07741177f,1.04195702f,1.00843608f,.984703481f
                    },
                    {
                        .785431147f,1.18091142f,.948239684f,.968951106f,
                        .796918511f,.823158622f,.836061835f,.808076978f,
                        .758099318f,.798000813f,.892770529f,.918704867f,
                        .95387733f,.942024589f,.982869267f,.977622151f,
                        .945878863f,.934528828f,.922981381f,.905643463f,
                        .913123488f,.925620198f,.928410172f,.929000735f,
                        .929074764f,.927165389f,.92312777f,.9200418f,
                        .913583755f,.906530619f,.896545172f,.90247345f
                    },
                    {
                        -.0492094904f,1.70581806f,1.54243672f,1.10348058f,
                        1.33353949f,1.61677063f,1.75299704f,2.10867286f,
                        2.33192945f,2.35721993f,2.31439018f,2.32760978f,
                        2.35372281f,2.37630463f,2.38217878f,2.36605382f,
                        2.29672241f,2.19930649f,2.25116682f,2.17571759f,
                        1.26080739f,.378913045f,-.259793997f,-.151380509f,
                        .362665236f,2.52020073f,6.8909421f,7.09358215f,
                        7.73367214f,7.79877377f,7.66240597f,8.3328619f
                    }
                };
                const size_t expected_failures[4] = {2u, 9u, 0u, 16u};
                size_t level;
                for (level = 0; level < 4; ++level) {
                    char name[32];
                    size_t failures = 0;
                    (void)snprintf(
                        name, sizeof(name), "production.fpn.%zu", level);
                    assert(ua_context_debug_tensor_f32(
                        context, name, stem_sample, 32, &stem_written) ==
                        UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        if (!close_enough(
                                stem_sample[sample_index],
                                oracle[level][sample_index],
                                5e-3f, 1e-2f))
                            ++failures;
                    assert(failures == expected_failures[level]);
                }
            }
            {
                static const size_t failure_count[4] = {1u, 5u, 7u, 3u};
                static const size_t failure_index[4][7] = {
                    {29u,0u,0u,0u,0u,0u,0u},
                    {0u,1u,11u,14u,29u,0u,0u},
                    {0u,2u,4u,12u,17u,27u,31u},
                    {2u,5u,11u,0u,0u,0u,0u}
                };
                static const float failure_oracle[4][7] = {
                    {-.290350795f,0,0,0,0,0,0},
                    {-.458408117f,.411177009f,1.33721817f,
                     .0671949387f,-.0151219368f,0,0},
                    {-.157915711f,-.303210586f,-.0759917498f,
                     -.570751011f,-.0811166465f,-.133810282f,
                     -.221081048f},
                    {-.126471043f,.263027161f,-.292646825f,0,0,0,0}
                };
                static const float candidate_first[4] = {
                    -.708007812f,-.469482422f,-.168457031f,-.307617188f
                };
                static const float candidate_last[4] = {
                    8.203125f,7.421875f,-.213378906f,-3.6484375f
                };
                size_t level, failure;
                for (level = 0; level < 4; ++level) {
                    char name[64];
                    (void)snprintf(
                        name, sizeof(name),
                        "production.bevformer.feat_flatten.level%zu", level);
                    assert(ua_context_debug_tensor_f32(
                        context, name, stem_sample, 32, &stem_written) ==
                        UA_OK);
                    assert(close_enough(
                        stem_sample[0], candidate_first[level],
                        1e-6f, 1e-6f));
                    assert(close_enough(
                        stem_sample[31], candidate_last[level],
                        1e-6f, 1e-6f));
                    for (failure = 0; failure < failure_count[level];
                         ++failure) {
                        size_t index = failure_index[level][failure];
                        assert(!close_enough(
                            stem_sample[index],
                            failure_oracle[level][failure],
                            5e-3f, 1e-2f));
                    }
                }
            }
            {
                static const float query_oracle[32] = {
                    -1.11320496f,.28994289f,-1.99725473f,-.00430639833f,
                    -.0376797318f,1.03104246f,-1.188627f,1.03886354f,
                    1.27279496f,.0435580611f,1.22285151f,.179040581f,
                    -.733505964f,-.38821876f,-.273934871f,.319697976f,
                    -1.37110615f,-.281122446f,.669579625f,.566161394f,
                    .826713026f,2.17078924f,.114088446f,-.251905829f,
                    -.963867486f,-.262863517f,-.841058493f,1.43859076f,
                    -.199741751f,-.741390228f,.758743465f,1.78797984f
                };
                static const float position_oracle[32] = {
                    .957006514f,.715298235f,.864209175f,-.283210218f,
                    -.0650084764f,.639823973f,-.190110266f,1.62732494f,
                    .659844398f,-.232595131f,.921410501f,-.627319217f,
                    -.343470514f,.626273751f,-.00705330912f,-.208507121f,
                    .954280853f,.0525770187f,1.21479201f,-1.60208786f,
                    -.878268361f,.46967274f,-.519861221f,-1.36508405f,
                    -2.80144048f,.306991577f,-.330807298f,-1.00548434f,
                    .219126523f,-1.35385656f,-.555027127f,1.3625803f
                };
                const char *names[2] = {
                    "production.bevformer.bev_queries",
                    "production.bevformer.bev_pos"
                };
                const float *oracles[2] = {query_oracle, position_oracle};
                size_t boundary;
                for (boundary = 0; boundary < 2; ++boundary) {
                    assert(ua_context_debug_tensor_f32(
                        context, names[boundary], stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            oracles[boundary][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            {
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.reference_2d",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 16; ++sample_index) {
                    assert(close_enough(
                        stem_sample[sample_index * 2u],
                        ((float)sample_index + .5f) / 200.0f,
                        5e-3f, 1e-2f));
                    assert(close_enough(
                        stem_sample[sample_index * 2u + 1u], .5f / 200.0f,
                        5e-3f, 1e-2f));
                }
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.reference_3d",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 10; ++sample_index) {
                    assert(close_enough(
                        stem_sample[sample_index * 3u],
                        ((float)sample_index + .5f) / 200.0f,
                        5e-3f, 1e-2f));
                    assert(close_enough(
                        stem_sample[sample_index * 3u + 1u], .5f / 200.0f,
                        5e-3f, 1e-2f));
                    assert(close_enough(
                        stem_sample[sample_index * 3u + 2u], .0625f,
                        5e-3f, 1e-2f));
                }
                assert(ua_context_debug_tensor_f32(
                    context,
                    "production.bevformer.reference_camera.center_d3",
                    stem_sample, 2, &stem_written) == UA_OK);
                assert(stem_written == 2);
                assert(close_enough(stem_sample[0], .4375f, 1e-5f, 1e-5f));
                assert(close_enough(
                    stem_sample[1], .388888889f, 1e-5f, 1e-5f));
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.visibility.center_d3",
                    stem_sample, 1, &stem_written) == UA_OK);
                assert(stem_written == 1 && stem_sample[0] == 1.0f);
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.visible_counts",
                    stem_sample, 6, &stem_written) == UA_OK);
                assert(stem_written == 6);
                for (sample_index = 0; sample_index < 6; ++sample_index)
                    assert(stem_sample[sample_index] == 32.0f);
                {
                    static const float visible_indices_oracle[32] = {
                        19696,19697,19698,19699,19700,19701,19702,19703,
                        19896,19897,19898,19899,19900,19901,19902,19903,
                        20096,20097,20098,20099,20100,20101,20102,20103,
                        20296,20297,20298,20299,20300,20301,20302,20303
                    };
                    assert(ua_context_debug_tensor_f32(
                        context,
                        "production.bevformer.visible_indices.camera0",
                        stem_sample, 32, &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(stem_sample[sample_index] ==
                               visible_indices_oracle[sample_index]);
                }
            }
            {
                static const float temporal_oracle[32] = {
                    -.437690616f,-.880314887f,-2.25096345f,-.508422554f,
                    -.755115509f,1.05218446f,-1.85406983f,1.82405162f,
                    1.04817593f,.296313703f,1.97288489f,.207867116f,
                    -1.02583313f,-.577931881f,-.31042102f,.266509265f,
                    -.96424818f,-.350552142f,.423626453f,.675806999f,
                    .818018854f,2.43115473f,.953244567f,-.0622684956f,
                    -1.33798206f,-.420865357f,-1.44936371f,.946369767f,
                    -.471825808f,-1.01157141f,.375050664f,1.14676154f
                };
                assert(ua_context_debug_tensor_f32(
                    context,
                    "production.bevformer.encoder.layer0.temporal",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index],
                        temporal_oracle[sample_index], 5e-3f, 1e-2f));
            }
            {
                static const float norm0_oracle[32] = {
                    -.282108575f,-.649311841f,-1.52733469f,-.39249903f,
                    -.486940295f,.737430394f,-1.310552f,1.30297399f,
                    .728940785f,.228391677f,1.45356476f,.197248876f,
                    -.702694416f,-.406995058f,-.196058542f,.208359942f,
                    -.632777572f,-.244204342f,.323147446f,.491761208f,
                    .630821645f,1.80415881f,.718372166f,-.0316053256f,
                    -.903759658f,-.256691664f,-.980377316f,.653170764f,
                    -.338862866f,-.697374523f,.272343934f,.877455473f
                };
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.encoder.layer0.norm0",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index], norm0_oracle[sample_index],
                        5e-3f, 1e-2f));
            }
            {
                static const float spatial_oracle[32] = {
                    -.265305519f,-.635531485f,-1.55477691f,-.411230236f,
                    -.498760909f,.739979386f,-1.32463169f,1.30027544f,
                    .690293372f,.178583086f,1.48694074f,.187166408f,
                    -.676245034f,-.422144294f,-.223511994f,.216427669f,
                    -.642291486f,-.227934629f,.321974903f,.486630052f,
                    .61115247f,1.79432118f,.698933959f,.00786045939f,
                    -.869863749f,-.25275147f,-.974697888f,.651515782f,
                    -.351006716f,-.724965036f,.242253378f,.887642205f
                };
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.encoder.layer0.spatial",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index],
                        spatial_oracle[sample_index], 5e-3f, 1e-2f));
            }
            {
                static const float norm1_oracle[32] = {
                    -.258257359f,-.643323421f,-1.57182419f,-.439605504f,
                    -.498178035f,.754717708f,-1.28877032f,1.29327965f,
                    .670128763f,.130730867f,1.46719718f,.182624683f,
                    -.700226367f,-.436866581f,-.221707493f,.206509322f,
                    -.660891473f,-.229929134f,.354683548f,.471187651f,
                    .611278176f,1.79391074f,.717232585f,.0276497304f,
                    -.817473412f,-.239993602f,-1.01380301f,.618891835f,
                    -.348555803f,-.728545666f,.233249068f,.91354996f
                };
                assert(ua_context_debug_tensor_f32(
                    context, "production.bevformer.encoder.layer0.norm1",
                    stem_sample, 32, &stem_written) == UA_OK);
                for (sample_index = 0; sample_index < 32; ++sample_index)
                    assert(close_enough(
                        stem_sample[sample_index], norm1_oracle[sample_index],
                        5e-3f, 1e-2f));
            }
            {
                static const float ffn_oracle[32] = {
                    -.168135256f,-.00336170197f,-.990188718f,-1.00511897f,
                    -1.77293479f,.318658859f,-1.57647967f,1.01151073f,
                    -1.16655445f,-.399652958f,1.91605067f,-.725988567f,
                    -.175044537f,-.255071789f,-.113246933f,-1.12929022f,
                    .376353443f,.118671373f,.728845596f,-.334148109f,
                    .0502426624f,-.444397688f,1.08080661f,-.0364615545f,
                    -1.08061481f,.683796048f,-3.33756232f,.373921782f,
                    -.498197734f,-1.31322384f,-.49845928f,1.06154871f
                };
                static const float norm2_oracle[32] = {
                    -.100918345f,-.00913847703f,-.616872132f,-.617843151f,
                    -1.18269336f,.232151717f,-.970966101f,.672396123f,
                    -.779649138f,-.23265557f,1.23580527f,-.488927037f,
                    -.0817812234f,-.133350879f,-.0397983193f,-.720405281f,
                    .262140542f,.100101106f,.463450164f,-.186285302f,
                    .0648505092f,-.358890116f,.74047935f,.0244542304f,
                    -.642050624f,.42674765f,-2.41812086f,.242969215f,
                    -.308902293f,-.82436353f,-.308314323f,.707726359f
                };
                const char *names[2] = {
                    "production.bevformer.encoder.layer0.ffn",
                    "production.bevformer.encoder.layer0.norm2"
                };
                const float *oracles[2] = {ffn_oracle, norm2_oracle};
                size_t boundary;
                for (boundary = 0; boundary < 2; ++boundary) {
                    assert(ua_context_debug_tensor_f32(
                        context, names[boundary], stem_sample, 32,
                        &stem_written) == UA_OK);
                    for (sample_index = 0; sample_index < 32; ++sample_index)
                        assert(close_enough(
                            stem_sample[sample_index],
                            oracles[boundary][sample_index],
                            5e-3f, 1e-2f));
                }
            }
            assert(ua_context_debug_tensor_f32(
                context, "production.missing", stem_sample, 1,
                &stem_written) == UA_ERR_IO);
            input.camera_intrinsics[0][0] = 0.0f;
            assert(ua_infer_production(context, &input, result) ==
                   UA_ERR_ARGUMENT);
            input.camera_intrinsics[0][0] = 1000.0f;
            input.camera_to_ego[0][15] = 0.0f;
            assert(ua_infer_production(context, &input, result) ==
                   UA_ERR_ARGUMENT);
            ua_context_destroy(context);
            ua_model_destroy(model);
            free(result);
            free(image);
        }
    }
}
#endif

static void path(char *dst, size_t n, const char *dir, const char *name) {
    assert(snprintf(dst, n, "%s/%s", dir, name) > 0);
}

int main(void) {
    char tmp[] = "/tmp/uniad-test-XXXXXX", model_path[256], f0_path[256], f1_path[256];
    ua_model *m = NULL; ua_frame *f0 = NULL, *f1 = NULL;
    ua_context *c = NULL; ua_result a, b, reset; ua_metrics metrics;
    char json[65536], tiny[8]; size_t written = 0; FILE *file; int byte;
    {
        float score[] = {1, 2, 2, -1}; size_t top[3];
        float image[] = {1, 2, 3, 4};
        float logits[] = {1000, 1000};
        ua_op_stable_topk(score, 4, 3, top);
        assert(top[0] == 1 && top[1] == 2 && top[2] == 0);
        assert(ua_op_bilinear(image, 2, 2, .5f, .5f) == 2.5f);
        assert(ua_op_bilinear(image, 2, 2, -2, 0) == 0);
        ua_op_softmax(logits, 1, 2);
        assert(logits[0] == .5f && logits[1] == .5f);
    }
#ifdef UA_WITH_CUDA
    test_cuda_operators();
#endif
    {
        int fd = mkstemp(tmp);
        assert(fd >= 0); close(fd); assert(unlink(tmp) == 0);
        assert(mkdir(tmp, 0700) == 0);
    }
    assert(ua_write_demo_assets(tmp) == UA_OK);
    path(model_path, sizeof(model_path), tmp, "demo.uaw");
    path(f0_path, sizeof(f0_path), tmp, "frame0.uaf");
    path(f1_path, sizeof(f1_path), tmp, "frame1.uaf");
    assert(ua_model_load(model_path, &m) == UA_OK);
    assert(!strcmp(ua_model_profile(m), "tiny-synthetic-v1"));
    assert(ua_frame_load(f0_path, &f0) == UA_OK);
    assert(ua_frame_load(f1_path, &f1) == UA_OK);
    assert(ua_context_create(m, UA_BACKEND_CPU, &c) == UA_OK);
    assert(ua_infer(c, f0, &a) == UA_OK);
    assert(ua_infer(c, f1, &b) == UA_OK);
    assert(b.frame_index == 1 && b.track_count == UA_MAX_TRACKS);
    ua_context_metrics(c, &metrics);
    assert(metrics.owned_host_bytes <= 256u * 1024u * 1024u);
    assert(metrics.owned_device_bytes == 0);
    ua_context_reset(c);
    assert(ua_infer(c, f1, &reset) == UA_OK);
    assert(reset.tracks[0].score != b.tracks[0].score);
    assert(ua_result_json(&b, json, sizeof(json), &written) == UA_OK && written > 100);
    assert(ua_result_json(&b, tiny, sizeof(tiny), NULL) == UA_ERR_CAPACITY);
#ifdef UA_WITH_CUDA
    {
        ua_context *cuda_context = NULL;
        assert(ua_context_create(m, UA_BACKEND_CUDA, &cuda_context) == UA_OK);
        assert(ua_infer(cuda_context, f1, &reset) == UA_OK);
        ua_context_destroy(cuda_context);
    }
#else
    assert(ua_context_create(m, UA_BACKEND_CUDA, &(ua_context *){0}) == UA_ERR_BACKEND);
#endif
    /* A one-byte payload corruption must be distinguished from bad structure. */
    file = fopen(f0_path, "r+b"); assert(file);
    assert(fseek(file, -1, SEEK_END) == 0); byte = fgetc(file); assert(byte != EOF);
    assert(fseek(file, -1, SEEK_END) == 0); assert(fputc(byte ^ 1, file) != EOF);
    assert(fclose(file) == 0);
    {
        ua_frame *bad = NULL;
        assert(ua_frame_load(f0_path, &bad) == UA_ERR_CHECKSUM);
        assert(!bad);
    }
    ua_context_destroy(c); ua_frame_destroy(f0); ua_frame_destroy(f1); ua_model_destroy(m);
    unlink(model_path); unlink(f0_path); unlink(f1_path); rmdir(tmp);
    puts("runtime tests: ok");
    return 0;
}
