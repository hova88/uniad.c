#include "operators.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

void ua_op_linear(const float *x, const float *w, const float *bias,
                  float *y, size_t rows, size_t in_dim, size_t out_dim) {
    size_t r, o, i;
    for (r = 0; r < rows; ++r) for (o = 0; o < out_dim; ++o) {
        float sum = bias ? bias[o] : 0.0f;
        for (i = 0; i < in_dim; ++i) sum += x[r * in_dim + i] * w[o * in_dim + i];
        y[r * out_dim + o] = sum;
    }
}

void ua_op_conv2d(const float *x, const float *w, const float *bias, float *y,
                  size_t channels, size_t height, size_t width, size_t outputs,
                  size_t kernel, size_t padding) {
    size_t o, oy, ox, c, ky, kx;
    for (o = 0; o < outputs; ++o) for (oy = 0; oy < height; ++oy)
        for (ox = 0; ox < width; ++ox) {
            float sum = bias ? bias[o] : 0.0f;
            for (c = 0; c < channels; ++c) for (ky = 0; ky < kernel; ++ky)
                for (kx = 0; kx < kernel; ++kx) {
                    long iy = (long)oy + (long)ky - (long)padding;
                    long ix = (long)ox + (long)kx - (long)padding;
                    if (iy >= 0 && ix >= 0 && (size_t)iy < height && (size_t)ix < width)
                        sum += x[(c * height + (size_t)iy) * width + (size_t)ix] *
                               w[((o * channels + c) * kernel + ky) * kernel + kx];
                }
            y[(o * height + oy) * width + ox] = sum;
        }
}

void ua_op_layer_norm(const float *x, float *y, size_t rows, size_t dim, float eps) {
    size_t r, i;
    for (r = 0; r < rows; ++r) {
        float mean = 0.0f, variance = 0.0f;
        for (i = 0; i < dim; ++i) mean += x[r * dim + i];
        mean /= (float)dim;
        for (i = 0; i < dim; ++i) {
            float d = x[r * dim + i] - mean; variance += d * d;
        }
        variance /= (float)dim;
        for (i = 0; i < dim; ++i)
            y[r * dim + i] = (x[r * dim + i] - mean) / sqrtf(variance + eps);
    }
}
void ua_op_relu(float *x, size_t n) { size_t i; for (i = 0; i < n; ++i) if (x[i] < 0) x[i] = 0; }
void ua_op_gelu(float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}
void ua_op_softmax(float *x, size_t rows, size_t cols) {
    size_t r, i;
    for (r = 0; r < rows; ++r) {
        float maximum = -FLT_MAX, sum = 0.0f;
        for (i = 0; i < cols; ++i) if (x[r * cols + i] > maximum) maximum = x[r * cols + i];
        for (i = 0; i < cols; ++i) { x[r * cols + i] = expf(x[r * cols + i] - maximum); sum += x[r * cols + i]; }
        if (sum) for (i = 0; i < cols; ++i) x[r * cols + i] /= sum;
    }
}
void ua_op_attention(const float *q, const float *k, const float *v, float *out,
                     size_t queries, size_t keys, size_t dim) {
    size_t qi, ki, d; float *scores;
    if (!queries || !keys || !dim) return;
    scores = (float *)malloc(queries * keys * sizeof(float)); if (!scores) return;
    for (qi = 0; qi < queries; ++qi) for (ki = 0; ki < keys; ++ki) {
        float sum = 0.0f;
        for (d = 0; d < dim; ++d) sum += q[qi * dim + d] * k[ki * dim + d];
        scores[qi * keys + ki] = sum / sqrtf((float)dim);
    }
    ua_op_softmax(scores, queries, keys);
    for (qi = 0; qi < queries; ++qi) for (d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (ki = 0; ki < keys; ++ki) sum += scores[qi * keys + ki] * v[ki * dim + d];
        out[qi * dim + d] = sum;
    }
    free(scores);
}
float ua_op_bilinear(const float *im, size_t h, size_t w, float y, float x) {
    long y0, x0, y1, x1; float fy, fx, result = 0.0f;
    if (!h || !w || y < -1.0f || x < -1.0f || y > (float)h || x > (float)w) return 0.0f;
    y0 = (long)floorf(y); x0 = (long)floorf(x); y1 = y0 + 1; x1 = x0 + 1;
    fy = y - (float)y0; fx = x - (float)x0;
#define SAMPLE(yy, xx, weight) do { if ((yy) >= 0 && (xx) >= 0 && (size_t)(yy) < h && (size_t)(xx) < w) result += im[(size_t)(yy) * w + (size_t)(xx)] * (weight); } while (0)
    SAMPLE(y0, x0, (1-fy)*(1-fx)); SAMPLE(y0, x1, (1-fy)*fx);
    SAMPLE(y1, x0, fy*(1-fx)); SAMPLE(y1, x1, fy*fx);
#undef SAMPLE
    return result;
}
void ua_op_resize_bilinear(const float *src, size_t sh, size_t sw,
                           float *dst, size_t dh, size_t dw) {
    size_t y, x;
    for (y = 0; y < dh; ++y) for (x = 0; x < dw; ++x) {
        float sy = ((float)y + .5f) * (float)sh / (float)dh - .5f;
        float sx = ((float)x + .5f) * (float)sw / (float)dw - .5f;
        dst[y * dw + x] = ua_op_bilinear(src, sh, sw, sy, sx);
    }
}
void ua_op_deform_sample(const float *map, size_t h, size_t w, size_t channels,
                         const float *points, size_t n, float *out) {
    size_t p, c, i;
    for (p = 0; p < n; ++p) for (c = 0; c < channels; ++c) {
        float value = 0.0f;
        /* channel planes are gathered into a temporary scalar view without allocation */
        long y0 = (long)floorf(points[p * 2]), x0 = (long)floorf(points[p * 2 + 1]);
        float fy = points[p * 2] - (float)y0, fx = points[p * 2 + 1] - (float)x0;
        const float weights[4] = {(1-fy)*(1-fx),(1-fy)*fx,fy*(1-fx),fy*fx};
        const long ys[4] = {y0,y0,y0+1,y0+1}, xs[4] = {x0,x0+1,x0,x0+1};
        for (i = 0; i < 4; ++i) if (ys[i] >= 0 && xs[i] >= 0 && (size_t)ys[i] < h && (size_t)xs[i] < w)
            value += map[((size_t)ys[i] * w + (size_t)xs[i]) * channels + c] * weights[i];
        out[p * channels + c] = value;
    }
}
void ua_op_stable_topk(const float *score, size_t n, size_t k, size_t *index) {
    size_t i, j, p;
    for (i = 0; i < k; ++i) {
        size_t best = 0; int found = 0;
        for (j = 0; j < n; ++j) {
            int used = 0; for (p = 0; p < i; ++p) if (index[p] == j) used = 1;
            if (!used && (!found || score[j] > score[best] || (score[j] == score[best] && j < best)))
                { best = j; found = 1; }
        }
        index[i] = best;
    }
}
void ua_op_accumulate_trajectory(const float *delta, size_t steps, float x, float y, float *out) {
    size_t i; for (i = 0; i < steps; ++i) { x += delta[i*2]; y += delta[i*2+1]; out[i*2]=x; out[i*2+1]=y; }
}
int ua_op_plan_collision(const float *plan, size_t steps, const float *actors,
                         size_t actor_count, float radius, float *score) {
    size_t s, a; float maximum = 0.0f, r2 = radius * radius; int collision = 0;
    for (s = 0; s < steps; ++s) for (a = 0; a < actor_count; ++a) {
        float dx=plan[s*2]-actors[(s*actor_count+a)*2], dy=plan[s*2+1]-actors[(s*actor_count+a)*2+1];
        float risk=expf(-(dx*dx+dy*dy)); if (risk>maximum) maximum=risk;
        if (dx*dx+dy*dy <= r2) collision=1;
    }
    if (score) *score = maximum;
    return collision;
}
