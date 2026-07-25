#ifndef UA_OPERATORS_H
#define UA_OPERATORS_H
#include <stddef.h>

void ua_op_linear(const float *x, const float *w, const float *bias,
                  float *y, size_t rows, size_t in_dim, size_t out_dim);
void ua_op_conv2d(const float *x, const float *w, const float *bias, float *y,
                  size_t channels, size_t height, size_t width, size_t outputs,
                  size_t kernel, size_t padding);
void ua_op_layer_norm(const float *x, float *y, size_t rows, size_t dim, float eps);
void ua_op_relu(float *x, size_t n);
void ua_op_gelu(float *x, size_t n);
void ua_op_softmax(float *x, size_t rows, size_t cols);
void ua_op_attention(const float *q, const float *k, const float *v, float *out,
                     size_t queries, size_t keys, size_t dim);
float ua_op_bilinear(const float *image, size_t height, size_t width, float y, float x);
void ua_op_resize_bilinear(const float *src, size_t sh, size_t sw,
                           float *dst, size_t dh, size_t dw);
void ua_op_deform_sample(const float *map, size_t height, size_t width, size_t channels,
                         const float *points_yx, size_t points, float *out);
void ua_op_stable_topk(const float *score, size_t n, size_t k, size_t *indices);
void ua_op_accumulate_trajectory(const float *delta_xy, size_t steps, float x0, float y0,
                                 float *trajectory_xy);
int ua_op_plan_collision(const float *plan_xy, size_t plan_steps,
                         const float *actors_xy, size_t actors, float radius,
                         float *score);
#endif
