#ifndef UA_CUDA_BACKEND_H
#define UA_CUDA_BACKEND_H

#include "uniad.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

ua_status ua_cuda_available(void);
ua_status ua_cuda_demo(const float *input, size_t n, float *output);
ua_status ua_cuda_production_create(
    const void *weights, size_t bytes, void **output, size_t *device_bytes);
void ua_cuda_production_reset(void *context);
void ua_cuda_production_destroy(void *context);
ua_status ua_cuda_production_tensor_pointer(
    void *context, size_t byte_offset, size_t nbytes, const void **device_ptr);
ua_status ua_cuda_production_preprocess(
    void *context, const ua_production_input *input, size_t *h2d_bytes,
    const void **normalized_fp16);
ua_status ua_cuda_production_resnet_stem(
    void *context, const void *normalized_fp16, const void *conv_weight_fp16,
    const void *bn_gamma_fp16, const void *bn_beta_fp16,
    const void *bn_mean_fp32, const void *bn_variance_fp32,
    const void **pooled_fp16);
ua_status ua_cuda_production_copy_boundary_f32(
    void *context, const char *name, float *values, size_t capacity,
    size_t *written);

typedef struct {
    const void *weight_fp16;
    const void *gamma_fp16;
    const void *beta_fp16;
    const void *mean_fp32;
    const void *variance_fp32;
} ua_cuda_conv_bn_weights;

typedef struct {
    ua_cuda_conv_bn_weights conv1, conv2, conv3, downsample;
    const void *conv2_offset_weight_fp16;
    const void *conv2_offset_bias_fp16;
    int conv2_is_dcn;
    int has_downsample;
} ua_cuda_bottleneck_weights;

typedef struct {
    const void *weight_fp16;
    const void *bias_fp16;
} ua_cuda_conv_bias_weights;

typedef struct {
    const void *linear0_weight_fp32;
    const void *linear0_bias_fp32;
    const void *linear1_weight_fp32;
    const void *linear1_bias_fp32;
    const void *norm_weight_fp32;
    const void *norm_bias_fp32;
} ua_cuda_can_bus_weights;

typedef struct {
    const void *value_weight_fp16;
    const void *value_bias_fp16;
    const void *offset_weight_fp16;
    const void *offset_bias_fp16;
    const void *attention_weight_fp16;
    const void *attention_bias_fp16;
    const void *output_weight_fp16;
    const void *output_bias_fp16;
} ua_cuda_temporal_attention_weights;

typedef struct {
    const void *value_weight_fp16;
    const void *value_bias_fp16;
    const void *offset_weight_fp16;
    const void *offset_bias_fp16;
    const void *attention_weight_fp16;
    const void *attention_bias_fp16;
    const void *output_weight_fp16;
    const void *output_bias_fp16;
} ua_cuda_spatial_attention_weights;

typedef struct {
    const void *linear0_weight_fp16;
    const void *linear0_bias_fp16;
    const void *linear1_weight_fp16;
    const void *linear1_bias_fp16;
    const void *norm_weight_fp16;
    const void *norm_bias_fp16;
} ua_cuda_encoder_ffn_weights;

typedef struct {
    ua_cuda_temporal_attention_weights temporal;
    const void *norm0_weight_fp16;
    const void *norm0_bias_fp16;
    ua_cuda_spatial_attention_weights spatial;
    const void *norm1_weight_fp16;
    const void *norm1_bias_fp16;
    ua_cuda_encoder_ffn_weights ffn;
} ua_cuda_encoder_layer_weights;

typedef struct {
    const void *self_in_weight_fp16;
    const void *self_in_bias_fp16;
    const void *self_out_weight_fp16;
    const void *self_out_bias_fp16;
    const void *norm0_weight_fp16;
    const void *norm0_bias_fp16;
    const void *cross_value_weight_fp16;
    const void *cross_value_bias_fp16;
    const void *cross_offset_weight_fp16;
    const void *cross_offset_bias_fp16;
    const void *cross_attention_weight_fp16;
    const void *cross_attention_bias_fp16;
    const void *cross_out_weight_fp16;
    const void *cross_out_bias_fp16;
    const void *norm1_weight_fp16;
    const void *norm1_bias_fp16;
    ua_cuda_encoder_ffn_weights ffn;
} ua_cuda_track_decoder_layer_weights;

typedef struct {
    const void *linear0_weight_fp16;
    const void *linear0_bias_fp16;
    const void *linear1_weight_fp16;
    const void *linear1_bias_fp16;
    const void *linear2_weight_fp16;
    const void *linear2_bias_fp16;
} ua_cuda_track_regression_weights;

typedef struct {
    const void *linear0_weight_fp16;
    const void *linear0_bias_fp16;
    const void *norm0_weight_fp16;
    const void *norm0_bias_fp16;
    const void *linear1_weight_fp16;
    const void *linear1_bias_fp16;
    const void *norm1_weight_fp16;
    const void *norm1_bias_fp16;
    const void *output_weight_fp16;
    const void *output_bias_fp16;
} ua_cuda_track_classification_weights;

typedef struct {
    const void *linear0_weight_fp16;
    const void *linear0_bias_fp16;
    const void *linear1_weight_fp16;
    const void *linear1_bias_fp16;
    const void *output_weight_fp16;
    const void *output_bias_fp16;
} ua_cuda_track_past_trajectory_weights;

typedef struct {
    const void *self_in_weight_fp16;
    const void *self_in_bias_fp16;
    const void *self_out_weight_fp16;
    const void *self_out_bias_fp16;
    const void *norm1_weight_fp16;
    const void *norm1_bias_fp16;
    const void *linear1_weight_fp16;
    const void *linear1_bias_fp16;
    const void *linear2_weight_fp16;
    const void *linear2_bias_fp16;
    const void *norm2_weight_fp16;
    const void *norm2_bias_fp16;
    const void *feat1_weight_fp16;
    const void *feat1_bias_fp16;
    const void *feat2_weight_fp16;
    const void *feat2_bias_fp16;
    const void *norm_feat_weight_fp16;
    const void *norm_feat_bias_fp16;
} ua_cuda_query_interaction_weights;

ua_status ua_cuda_production_resnet_layer1(
    void *context, const void *stem_fp16,
    const ua_cuda_bottleneck_weights blocks[3], const void **output_fp16);
ua_status ua_cuda_production_resnet_layer2(
    void *context, const void *layer1_fp16,
    const ua_cuda_bottleneck_weights blocks[4], const void **output_fp16);
ua_status ua_cuda_production_resnet_layer3(
    void *context, const void *layer2_fp16,
    const ua_cuda_bottleneck_weights blocks[23], const void **output_fp16);
ua_status ua_cuda_production_resnet_layer4(
    void *context, const void *layer3_fp16,
    const ua_cuda_bottleneck_weights blocks[3], const void **output_fp16);
ua_status ua_cuda_production_fpn(
    void *context, const void *layer2_fp16, const void *layer3_fp16,
    const void *layer4_fp16, const ua_cuda_conv_bias_weights lateral[3],
    const ua_cuda_conv_bias_weights output_convs[4],
    const void *outputs_fp16[4]);
ua_status ua_cuda_production_bevformer_flatten(
    void *context, const void *fpn_outputs_fp16[4],
    const void *camera_embeds_fp32, const void *level_embeds_fp32,
    const void **flattened_fp16);
ua_status ua_cuda_production_prepare_bev(
    void *context, const void *bev_query_weight_fp16,
    const void *col_embed_fp16, const void *row_embed_fp16,
    const ua_cuda_can_bus_weights *can_bus_weights,
    const void **bev_queries_fp16, const void **bev_pos_fp16);
ua_status ua_cuda_production_prepare_bev_geometry(
    void *context, const void **reference_2d_fp16,
    const void **reference_3d_fp16, const void **reference_camera_fp32,
    const void **visibility_u8, const void **visible_indices_u32,
    const void **visible_counts_u32);
ua_status ua_cuda_production_encoder_temporal(
    void *context, size_t layer, const void *query_fp16,
    const ua_cuda_temporal_attention_weights *weights,
    const void **output_fp16);
ua_status ua_cuda_production_encoder_norm_after_temporal(
    void *context, size_t layer, const void *gamma_fp16,
    const void *beta_fp16, const void **output_fp16);
ua_status ua_cuda_production_encoder_spatial(
    void *context, size_t layer,
    const ua_cuda_spatial_attention_weights *weights,
    const void **output_fp16);
ua_status ua_cuda_production_encoder_norm_after_spatial(
    void *context, size_t layer, const void *gamma_fp16,
    const void *beta_fp16, const void **output_fp16);
ua_status ua_cuda_production_encoder_ffn(
    void *context, size_t layer,
    const ua_cuda_encoder_ffn_weights *weights,
    const void **ffn_output_fp16, const void **norm2_output_fp16);
ua_status ua_cuda_production_commit_previous_bev(
    void *context, const void *encoder_output_fp16);
ua_status ua_cuda_production_prepare_track_queries(
    void *context, const void *query_embedding_fp16,
    const void *reference_weight_fp32, const void *reference_bias_fp32,
    const void **query_pos_fp16, const void **query_fp16,
    const void **reference_points_fp32);
ua_status ua_cuda_production_track_decoder_layer(
    void *context, size_t layer, const void *bev_fp16,
    const ua_cuda_track_decoder_layer_weights *weights,
    const void **output_fp16);
ua_status ua_cuda_production_track_refine_references(
    void *context, size_t layer, const void *decoder_output_fp16,
    const ua_cuda_track_regression_weights *weights,
    const void **regression_fp16, const void **reference_points_fp32);
ua_status ua_cuda_production_track_output_heads(
    void *context,
    const ua_cuda_track_classification_weights *classification_weights,
    const ua_cuda_track_past_trajectory_weights *past_weights,
    const void **class_logits_fp16, const void **boxes_fp32,
    const void **past_trajectory_fp16);
ua_status ua_cuda_production_track_score_filter(
    void *context, const void **scores_fp32, const void **classes_u32,
    const void **selected_indices_u32, const void **selected_count_u32);
ua_status ua_cuda_production_query_interaction(
    void *context, const void *query_pos_fp16, const void *query_feat_fp16,
    const void *output_embedding_fp16, size_t active_queries,
    const ua_cuda_query_interaction_weights *weights,
    const void **updated_query_feat_fp16);
ua_status ua_cuda_production_debug_query_interaction(
    void *context, size_t active_queries);

/* Correctness-fixture entry points. Production execution owns device buffers
 * directly; these wrappers intentionally include H2D/D2H for isolated tests. */
ua_status ua_cuda_test_preprocess_bgr(
    const uint8_t *source, size_t width, size_t height, size_t row_stride,
    size_t padded_width, size_t padded_height, const float mean_bgr[3],
    const float std_bgr[3], float *output_chw);
ua_status ua_cuda_test_linear_fp16(
    const float *x, const float *weight_out_in, const float *bias, float *y,
    size_t rows, size_t in_dim, size_t out_dim);
ua_status ua_cuda_test_conv2d_fp16(
    const float *x, const float *weight_oihw, const float *bias, float *y,
    size_t batches, size_t input_channels, size_t input_height,
    size_t input_width, size_t output_channels, size_t kernel_height,
    size_t kernel_width, size_t stride_height, size_t stride_width,
    size_t padding_height, size_t padding_width);
ua_status ua_cuda_test_modulated_deform_conv2d_fp16(
    const float *x_nchw, const float *offset_n2khkwohow,
    const float *mask_nkhkwohow, const float *weight_oihw,
    const float *bias, float *y_nchw, size_t batches, size_t input_channels,
    size_t input_height, size_t input_width, size_t output_channels,
    size_t kernel_height, size_t kernel_width, size_t stride_height,
    size_t stride_width, size_t padding_height, size_t padding_width,
    size_t dilation_height, size_t dilation_width);
ua_status ua_cuda_test_batchnorm_relu_fp16(
    const float *x_nchw, const float *gamma, const float *beta,
    const float *running_mean, const float *running_variance, float *y_nchw,
    size_t batches, size_t channels, size_t height, size_t width,
    float epsilon);
ua_status ua_cuda_test_maxpool2d_fp16(
    const float *x_nchw, float *y_nchw, size_t batches, size_t channels,
    size_t input_height, size_t input_width, size_t kernel_height,
    size_t kernel_width, size_t stride_height, size_t stride_width,
    size_t padding_height, size_t padding_width);
ua_status ua_cuda_test_layer_norm_fp16(
    const float *x, const float *gamma, const float *beta, float *y,
    size_t rows, size_t dim, float epsilon);
ua_status ua_cuda_test_softmax_f32(
    const float *x, float *y, size_t rows, size_t cols);
ua_status ua_cuda_test_resize_bilinear_f32(
    const float *source, size_t source_height, size_t source_width,
    float *target, size_t target_height, size_t target_width);
ua_status ua_cuda_test_deform_sample_f32(
    const float *map_hwc, size_t height, size_t width, size_t channels,
    const float *points_yx, size_t points, float *output);
ua_status ua_cuda_test_stable_topk_f32(
    const float *scores, size_t count, size_t k, size_t *indices);
ua_status ua_cuda_test_track_score_filter_fp16(
    const float *logits, size_t queries, size_t classes, float threshold,
    size_t capacity, float *scores, uint32_t *class_indices,
    uint32_t *selected_indices, size_t *selected_count);
ua_status ua_cuda_test_tracker_state_update(
    const float *scores, int32_t *object_ids, uint32_t *disappear,
    size_t count, float score_threshold, float filter_threshold,
    uint32_t miss_tolerance, int32_t *next_object_id,
    uint32_t *active_indices, size_t active_capacity,
    size_t *active_count);
ua_status ua_cuda_test_memory_bank_update_fp16(
    const float *embedding, const float *scores, const float *weight,
    const float *bias, float *memory, uint8_t *padding_mask,
    uint8_t *save_period, size_t queries, size_t history,
    size_t dimensions, float threshold, uint8_t reset_period);
ua_status ua_cuda_test_ms_deform_attn_fp16(
    const float *value_bshc, size_t batches, size_t total_values,
    size_t heads, size_t channels_per_head, const uint32_t *spatial_shapes_hw,
    const size_t *level_start_index, size_t levels,
    const float *sampling_locations_bqhlp2,
    const float *attention_weights_bqhlp, size_t queries, size_t points,
    float *output_bqhc);
ua_status ua_cuda_test_deform_locations_f32(
    const float *reference_bqld, size_t reference_dims,
    const float *offset_bqhlp2, const uint32_t *spatial_shapes_hw,
    size_t batches, size_t queries, size_t heads, size_t levels,
    size_t points, float *locations_bqhlp2);
ua_status ua_cuda_test_camera_scatter_average_f32(
    const float *slots_cmd, const uint32_t *query_indices_cm,
    const uint32_t *valid_counts_c, size_t cameras, size_t max_queries,
    size_t total_queries, size_t dimensions, float *output_qd);
ua_status ua_cuda_test_queue_mean_f32(
    const float *queue_bqcd, size_t batches, size_t queries,
    size_t dimensions, size_t queue_length, float *output_bqd);

#ifdef __cplusplus
}
#endif
#endif
