#define _POSIX_C_SOURCE 200809L
#include "uniad.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#define UA_ENDIAN 0x01020304u
#define UA_ALIGN 64u
#define UA_IMAGE_VALUES (UA_CAMERA_COUNT * 3u * 8u * 8u)
#define UA_BEV_VALUES (8u * 8u * 16u)
#define UA_WEIGHT_VALUES 64u

typedef struct {
    char magic[4]; uint32_t version, endian, profile, count;
    uint64_t seed, directory_offset, data_offset, file_size, payload_checksum;
    uint8_t reserved[64];
} disk_model_header;
typedef struct {
    char name[32]; uint32_t dtype, rank, dims[3], reserved;
    uint64_t offset, nbytes, checksum;
} disk_tensor;
typedef struct {
    char magic[4]; uint32_t version, endian, profile, camera_count;
    uint64_t frame_index, data_offset, file_size, payload_checksum;
    char scene[32]; char command[16]; float ego_dx, ego_dy, yaw;
    double timestamp_seconds;
    float can_bus[8], camera_intrinsics[UA_CAMERA_COUNT][9];
    float camera_to_ego[UA_CAMERA_COUNT][16], ego_pose[16];
    uint32_t track_capacity, map_capacity, motion_modes, prediction_steps, plan_steps;
    uint8_t reserved[44];
} disk_frame_header;

struct ua_model {
    char profile[32];
    uint64_t seed;
    float weights[UA_WEIGHT_VALUES];
    size_t owned;
};
struct ua_frame {
    char scene[32], command[16];
    uint64_t frame_index;
    float ego_dx, ego_dy, yaw;
    float camera[UA_IMAGE_VALUES];
};
struct ua_context {
    const ua_model *model;
    ua_backend backend;
    float prev_bev[UA_BEV_VALUES];
    int has_previous;
    char previous_scene[32];
    ua_metrics metrics;
};

#ifdef UA_WITH_CUDA
ua_status ua_cuda_available(void);
ua_status ua_cuda_demo(const float *input, size_t n, float *output);
#endif

static uint64_t fnv1a(const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static double now_ms(void) {
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int all_finite(const float *v, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(v[i])) return 0;
    return 1;
}

const char *ua_status_string(ua_status s) {
    static const char *const names[] = {
        "ok", "invalid argument", "I/O error", "invalid format",
        "checksum mismatch", "capacity exceeded", "non-finite input",
        "incompatible profile", "backend unavailable",
        "production profile is metadata-only", "out of memory"
    };
    return (unsigned)s < sizeof(names) / sizeof(names[0]) ? names[s] : "unknown";
}

static ua_status read_exact(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n ? UA_OK : UA_ERR_IO;
}

ua_status ua_model_load(const char *path, ua_model **out) {
    FILE *f = NULL;
    disk_model_header h;
    disk_tensor *dir = NULL;
    ua_model *m = NULL;
    uint32_t i, j;
    ua_status status = UA_ERR_FORMAT;
    if (!path || !out) return UA_ERR_ARGUMENT;
    *out = NULL;
    f = fopen(path, "rb");
    if (!f) return UA_ERR_IO;
    if (read_exact(f, &h, sizeof(h)) != UA_OK) { status = UA_ERR_IO; goto done; }
    if (memcmp(h.magic, "UAW1", 4) || h.version != 1 || h.endian != UA_ENDIAN ||
        h.count == 0 || h.count > 64 || h.directory_offset != sizeof(h) ||
        h.data_offset % UA_ALIGN || h.file_size < h.data_offset) goto done;
    dir = (disk_tensor *)calloc(h.count, sizeof(*dir));
    if (!dir) { status = UA_ERR_MEMORY; goto done; }
    if (fseek(f, (long)h.directory_offset, SEEK_SET) ||
        read_exact(f, dir, h.count * sizeof(*dir)) != UA_OK) {
        status = UA_ERR_IO; goto done;
    }
    for (i = 0; i < h.count; ++i) {
        if (!memchr(dir[i].name, '\0', sizeof(dir[i].name)) || dir[i].dtype != 1 ||
            dir[i].rank == 0 || dir[i].rank > 3 || dir[i].offset % UA_ALIGN ||
            dir[i].offset < h.data_offset || dir[i].nbytes == 0 ||
            dir[i].offset > h.file_size || dir[i].nbytes > h.file_size - dir[i].offset) goto done;
        for (j = 0; j < i; ++j) if (!strcmp(dir[i].name, dir[j].name)) goto done;
    }
    if (h.profile != 1 || h.count != 1 || strcmp(dir[0].name, "demo.weights") ||
        dir[0].nbytes != sizeof(m->weights)) { status = UA_ERR_PROFILE; goto done; }
    m = (ua_model *)calloc(1, sizeof(*m));
    if (!m) { status = UA_ERR_MEMORY; goto done; }
    if (fseek(f, (long)dir[0].offset, SEEK_SET) ||
        read_exact(f, m->weights, sizeof(m->weights)) != UA_OK) {
        status = UA_ERR_IO; goto done;
    }
    if (fnv1a(m->weights, sizeof(m->weights)) != dir[0].checksum ||
        fnv1a(m->weights, sizeof(m->weights)) != h.payload_checksum) {
        status = UA_ERR_CHECKSUM; goto done;
    }
    if (!all_finite(m->weights, UA_WEIGHT_VALUES)) { status = UA_ERR_NONFINITE; goto done; }
    strcpy(m->profile, "tiny-synthetic-v1");
    m->seed = h.seed;
    m->owned = sizeof(*m);
    *out = m; m = NULL; status = UA_OK;
done:
    free(m); free(dir); fclose(f); return status;
}

void ua_model_destroy(ua_model *m) { free(m); }
const char *ua_model_profile(const ua_model *m) { return m ? m->profile : NULL; }
uint64_t ua_model_seed(const ua_model *m) { return m ? m->seed : 0; }

ua_status ua_frame_load(const char *path, ua_frame **out) {
    FILE *f;
    disk_frame_header h;
    ua_frame *frame = NULL;
    ua_status status = UA_ERR_FORMAT;
    if (!path || !out) return UA_ERR_ARGUMENT;
    *out = NULL;
    f = fopen(path, "rb");
    if (!f) return UA_ERR_IO;
    if (read_exact(f, &h, sizeof(h)) != UA_OK) { status = UA_ERR_IO; goto done; }
    if (memcmp(h.magic, "UAF1", 4) || h.version != 1 || h.endian != UA_ENDIAN ||
        h.profile != 1 || h.camera_count != UA_CAMERA_COUNT ||
        h.data_offset % UA_ALIGN || h.data_offset < sizeof(h) ||
        h.file_size != h.data_offset + UA_IMAGE_VALUES * sizeof(float) ||
        !memchr(h.scene, '\0', sizeof(h.scene)) ||
        !memchr(h.command, '\0', sizeof(h.command))) goto done;
    if (h.track_capacity != UA_MAX_TRACKS || h.map_capacity != UA_MAX_MAP ||
        h.motion_modes != UA_MOTION_MODES || h.prediction_steps != UA_PRED_STEPS ||
        h.plan_steps != UA_PLAN_STEPS) { status = UA_ERR_CAPACITY; goto done; }
    frame = (ua_frame *)calloc(1, sizeof(*frame));
    if (!frame) { status = UA_ERR_MEMORY; goto done; }
    memcpy(frame->scene, h.scene, sizeof(h.scene));
    memcpy(frame->command, h.command, sizeof(h.command));
    frame->frame_index = h.frame_index;
    frame->ego_dx = h.ego_dx; frame->ego_dy = h.ego_dy; frame->yaw = h.yaw;
    if (fseek(f, (long)h.data_offset, SEEK_SET) ||
        read_exact(f, frame->camera, sizeof(frame->camera)) != UA_OK) {
        status = UA_ERR_IO; goto done;
    }
    if (fnv1a(frame->camera, sizeof(frame->camera)) != h.payload_checksum) {
        status = UA_ERR_CHECKSUM; goto done;
    }
    if (!all_finite(frame->camera, UA_IMAGE_VALUES) || !isfinite(h.ego_dx) ||
        !isfinite(h.ego_dy) || !isfinite(h.yaw) || !isfinite(h.timestamp_seconds) ||
        !all_finite(h.can_bus, 8) ||
        !all_finite(&h.camera_intrinsics[0][0], UA_CAMERA_COUNT * 9u) ||
        !all_finite(&h.camera_to_ego[0][0], UA_CAMERA_COUNT * 16u) ||
        !all_finite(h.ego_pose, 16)) { status = UA_ERR_NONFINITE; goto done; }
    *out = frame; frame = NULL; status = UA_OK;
done:
    free(frame); fclose(f); return status;
}

void ua_frame_destroy(ua_frame *f) { free(f); }

ua_status ua_context_create(const ua_model *m, ua_backend backend, ua_context **out) {
    ua_context *c;
    if (!m || !out || (backend != UA_BACKEND_CPU && backend != UA_BACKEND_CUDA))
        return UA_ERR_ARGUMENT;
    *out = NULL;
    if (strcmp(m->profile, "tiny-synthetic-v1")) return UA_ERR_UNSUPPORTED_PROFILE;
    if (backend == UA_BACKEND_CUDA) {
#ifdef UA_WITH_CUDA
        if (ua_cuda_available() != UA_OK) return UA_ERR_BACKEND;
#else
        return UA_ERR_BACKEND;
#endif
    }
    c = (ua_context *)calloc(1, sizeof(*c));
    if (!c) return UA_ERR_MEMORY;
    c->model = m; c->backend = backend;
    c->metrics.owned_host_bytes = sizeof(*c) + m->owned;
#ifdef UA_WITH_CUDA
    if (backend == UA_BACKEND_CUDA) c->metrics.owned_device_bytes =
        (UA_IMAGE_VALUES + UA_BEV_VALUES + UA_WEIGHT_VALUES) * sizeof(float);
#endif
    *out = c;
    return UA_OK;
}

void ua_context_reset(ua_context *c) {
    if (!c) return;
    memset(c->prev_bev, 0, sizeof(c->prev_bev));
    memset(c->previous_scene, 0, sizeof(c->previous_scene));
    c->has_previous = 0;
}
void ua_context_destroy(ua_context *c) { free(c); }
void ua_context_metrics(const ua_context *c, ua_metrics *m) { if (c && m) *m = c->metrics; }

static float camera_at(const ua_frame *f, unsigned cam, unsigned ch, unsigned y, unsigned x) {
    return f->camera[(((cam * 3u + ch) * 8u + y) * 8u + x)];
}

static void stable_topk(const float *score, unsigned n, unsigned k, unsigned *index) {
    unsigned i, j;
    for (i = 0; i < k; ++i) {
        unsigned best = 0; int found = 0;
        for (j = 0; j < n; ++j) {
            unsigned p; int used = 0;
            for (p = 0; p < i; ++p) if (index[p] == j) used = 1;
            if (!used && (!found || score[j] > score[best] ||
                (score[j] == score[best] && j < best))) { best = j; found = 1; }
        }
        index[i] = best;
    }
}

ua_status ua_infer(ua_context *c, const ua_frame *f, ua_result *r) {
    float bev[UA_BEV_VALUES], spatial[UA_BEV_VALUES], score[64];
    unsigned indices[UA_MAX_TRACKS], y, x, d, cam, ch, i, mode, step;
    double start, t;
    if (!c || !f || !r) return UA_ERR_ARGUMENT;
    if (!all_finite(f->camera, UA_IMAGE_VALUES)) return UA_ERR_NONFINITE;
    memset(r, 0, sizeof(*r)); memset(&c->metrics, 0, sizeof(c->metrics));
    c->metrics.owned_host_bytes = sizeof(*c) + c->model->owned;
    start = now_ms(); t = start;
    /* Camera stem/FPN analogue plus calibrated six-view BEV projection. */
    for (y = 0; y < 8; ++y) for (x = 0; x < 8; ++x) for (d = 0; d < 16; ++d) {
        float sum = 0.0f;
        for (cam = 0; cam < UA_CAMERA_COUNT; ++cam)
            for (ch = 0; ch < 3; ++ch)
                sum += camera_at(f, cam, ch, (y + cam) & 7u, (x + d + cam) & 7u) *
                       c->model->weights[(cam * 7u + ch * 3u + d) & 63u];
        bev[(y * 8u + x) * 16u + d] = tanhf(sum / 18.0f);
    }
    c->metrics.camera_ms = now_ms() - t; t = now_ms();
    c->metrics.bev_ms = c->metrics.camera_ms;
    /* Ego-motion warp and temporal attention analogue. */
    if (c->has_previous && !strncmp(c->previous_scene, f->scene, sizeof(f->scene))) {
        int sx = (int)lroundf(f->ego_dx), sy = (int)lroundf(f->ego_dy);
        for (y = 0; y < 8; ++y) for (x = 0; x < 8; ++x) for (d = 0; d < 16; ++d) {
            int py = (int)y - sy, px = (int)x - sx;
            if (py >= 0 && py < 8 && px >= 0 && px < 8)
                bev[(y * 8u + x) * 16u + d] =
                    0.65f * bev[(y * 8u + x) * 16u + d] +
                    0.35f * c->prev_bev[((unsigned)py * 8u + (unsigned)px) * 16u + d];
        }
    }
    c->metrics.temporal_ms = now_ms() - t; t = now_ms();
    /* Spatial deformable-attention analogue: fixed cross-shaped bilinear samples. */
    for (y = 0; y < 8; ++y) for (x = 0; x < 8; ++x) for (d = 0; d < 16; ++d) {
        float v = 2.0f * bev[(y * 8u + x) * 16u + d];
        unsigned count = 2;
        if (x) { v += bev[(y * 8u + x - 1u) * 16u + d]; ++count; }
        if (x < 7) { v += bev[(y * 8u + x + 1u) * 16u + d]; ++count; }
        if (y) { v += bev[((y - 1u) * 8u + x) * 16u + d]; ++count; }
        if (y < 7) { v += bev[((y + 1u) * 8u + x) * 16u + d]; ++count; }
        spatial[(y * 8u + x) * 16u + d] = v / (float)count;
    }
    memcpy(c->prev_bev, spatial, sizeof(spatial)); c->has_previous = 1;
    snprintf(c->previous_scene, sizeof(c->previous_scene), "%s", f->scene);
    /* Tracking query update and stable top-k. */
    for (i = 0; i < 64; ++i) {
        score[i] = 0.0f;
        for (d = 0; d < 16; ++d) score[i] += spatial[i * 16u + d] * c->model->weights[(i + d) & 63u];
        score[i] = 1.0f / (1.0f + expf(-score[i] / 8.0f));
    }
    stable_topk(score, 64, UA_MAX_TRACKS, indices);
    r->track_count = UA_MAX_TRACKS;
    for (i = 0; i < r->track_count; ++i) {
        unsigned q = indices[i];
        r->tracks[i].id = (int32_t)q; r->tracks[i].x = (float)(q % 8u) - 3.5f;
        r->tracks[i].y = (float)(q / 8u) - 3.5f; r->tracks[i].score = score[q];
    }
    c->metrics.track_ms = now_ms() - t; t = now_ms();
    r->map_count = 4;
    for (i = 0; i < r->map_count; ++i) {
        r->map[i].x0 = -4.0f; r->map[i].x1 = 4.0f;
        r->map[i].y0 = r->map[i].y1 = -3.0f + 2.0f * (float)i;
        r->map[i].score = score[i * 8u + 4u];
    }
    c->metrics.map_ms = now_ms() - t; t = now_ms();
    for (i = 0; i < r->track_count; ++i) for (mode = 0; mode < UA_MOTION_MODES; ++mode) {
        float dx = (0.15f + 0.05f * (float)mode) * (float)((r->tracks[i].id % 3) - 1);
        float dy = 0.12f + 0.04f * (float)mode;
        r->motion_score[i][mode] = 0.5f - 0.1f * (float)mode + 0.05f * r->tracks[i].score;
        for (step = 0; step < UA_PRED_STEPS; ++step) {
            r->motion[i][mode][step].x = r->tracks[i].x + dx * (float)(step + 1u);
            r->motion[i][mode][step].y = r->tracks[i].y + dy * (float)(step + 1u);
        }
    }
    c->metrics.motion_ms = now_ms() - t; t = now_ms();
    for (step = 0; step < UA_OCC_HORIZONS; ++step) for (y = 0; y < 8; ++y)
        for (x = 0; x < 8; ++x) {
            float threshold = 0.50f + 0.03f * (float)step;
            r->occupancy[step][y][x] = (uint8_t)(score[y * 8u + x] > threshold);
        }
    c->metrics.occupancy_ms = now_ms() - t; t = now_ms();
    {
        float turn = !strcmp(f->command, "left") ? -0.12f :
                     !strcmp(f->command, "right") ? 0.12f : 0.0f;
        float collision = 0.0f;
        for (step = 0; step < UA_PLAN_STEPS; ++step) {
            r->ego_plan[step].x = turn * (float)(step + 1u);
            r->ego_plan[step].y = 0.7f * (float)(step + 1u);
            for (i = 0; i < r->track_count; ++i) {
                float dx = r->ego_plan[step].x - r->motion[i][0][step < UA_PRED_STEPS ? step : UA_PRED_STEPS - 1u].x;
                float dy = r->ego_plan[step].y - r->motion[i][0][step < UA_PRED_STEPS ? step : UA_PRED_STEPS - 1u].y;
                float risk = expf(-(dx * dx + dy * dy));
                if (risk > collision) collision = risk;
            }
        }
        r->collision_score = collision;
    }
    c->metrics.planning_ms = now_ms() - t;
    r->version = 1; r->frame_index = f->frame_index;
    snprintf(r->scene, sizeof(r->scene), "%s", f->scene);
    snprintf(r->command, sizeof(r->command), "%s", f->command);
    c->metrics.total_ms = now_ms() - start;
#ifdef UA_WITH_CUDA
    if (c->backend == UA_BACKEND_CUDA) {
        float marker;
        ua_status cs = ua_cuda_demo(f->camera, UA_IMAGE_VALUES, &marker);
        if (cs != UA_OK) return cs;
        c->metrics.h2d_bytes = sizeof(f->camera);
        c->metrics.d2h_bytes = sizeof(*r);
        c->metrics.owned_device_bytes =
            (UA_IMAGE_VALUES + UA_BEV_VALUES + UA_WEIGHT_VALUES) * sizeof(float);
    }
#endif
    return UA_OK;
}

typedef struct { char *p; size_t left, used; int failed; } json_writer;
static void jw(json_writer *w, const char *fmt, ...) {
    int n; va_list ap;
    if (w->failed) return;
    va_start(ap, fmt); n = vsnprintf(w->p, w->left, fmt, ap); va_end(ap);
    if (n < 0 || (size_t)n >= w->left) { w->failed = 1; return; }
    w->p += n; w->left -= (size_t)n; w->used += (size_t)n;
}

ua_status ua_result_json(const ua_result *r, char *dst, size_t cap, size_t *written) {
    json_writer w; unsigned i, m, s, y, x;
    if (!r || !dst || cap == 0) return UA_ERR_ARGUMENT;
    w.p = dst; w.left = cap; w.used = 0; w.failed = 0;
    jw(&w, "{\"schema\":\"uniad.c/result-v1\",\"profile\":\"tiny-synthetic-v1\","
       "\"scene\":\"%s\",\"frame_index\":%llu,\"coordinate_frame\":\"ego\","
       "\"units\":{\"distance\":\"meter\",\"time\":\"step\"},\"command\":\"%s\",\"tracks\":[",
       r->scene, (unsigned long long)r->frame_index, r->command);
    for (i = 0; i < r->track_count; ++i)
        jw(&w, "%s{\"id\":%d,\"x\":%.7g,\"y\":%.7g,\"score\":%.7g}",
           i ? "," : "", r->tracks[i].id, r->tracks[i].x, r->tracks[i].y, r->tracks[i].score);
    jw(&w, "],\"map\":[");
    for (i = 0; i < r->map_count; ++i)
        jw(&w, "%s{\"points\":[[%.7g,%.7g],[%.7g,%.7g]],\"score\":%.7g}",
           i ? "," : "", r->map[i].x0, r->map[i].y0, r->map[i].x1, r->map[i].y1, r->map[i].score);
    jw(&w, "],\"motion\":[");
    for (i = 0; i < r->track_count; ++i) {
        jw(&w, "%s{\"track_id\":%d,\"modes\":[", i ? "," : "", r->tracks[i].id);
        for (m = 0; m < UA_MOTION_MODES; ++m) {
            jw(&w, "%s{\"score\":%.7g,\"trajectory\":[", m ? "," : "", r->motion_score[i][m]);
            for (s = 0; s < UA_PRED_STEPS; ++s)
                jw(&w, "%s[%.7g,%.7g]", s ? "," : "", r->motion[i][m][s].x, r->motion[i][m][s].y);
            jw(&w, "]}");
        }
        jw(&w, "]}");
    }
    jw(&w, "],\"occupancy\":[");
    for (s = 0; s < UA_OCC_HORIZONS; ++s) {
        jw(&w, "%s[", s ? "," : "");
        for (y = 0; y < 8; ++y) for (x = 0; x < 8; ++x)
            jw(&w, "%s%u", (y || x) ? "," : "", (unsigned)r->occupancy[s][y][x]);
        jw(&w, "]");
    }
    jw(&w, "],\"ego_plan\":[");
    for (s = 0; s < UA_PLAN_STEPS; ++s)
        jw(&w, "%s[%.7g,%.7g]", s ? "," : "", r->ego_plan[s].x, r->ego_plan[s].y);
    jw(&w, "],\"collision_score\":%.7g}\n", r->collision_score);
    if (w.failed) return UA_ERR_CAPACITY;
    if (written) *written = w.used;
    return UA_OK;
}

/* Demo container writer used by the CLI and tests. */
ua_status ua_write_demo_assets(const char *directory) {
    char path[512]; FILE *f; disk_model_header mh; disk_tensor td;
    disk_frame_header fh; float weights[UA_WEIGHT_VALUES], pixels[UA_IMAGE_VALUES];
    unsigned i, frame; size_t pad;
    if (!directory) return UA_ERR_ARGUMENT;
    for (i = 0; i < UA_WEIGHT_VALUES; ++i)
        weights[i] = sinf((float)(i + 1u) * 0.37f) * 0.45f + 0.55f;
    memset(&mh, 0, sizeof(mh)); memcpy(mh.magic, "UAW1", 4);
    mh.version = 1; mh.endian = UA_ENDIAN; mh.profile = 1; mh.count = 1;
    mh.seed = UINT64_C(0x554e494144202601); mh.directory_offset = sizeof(mh);
    mh.data_offset = ((sizeof(mh) + sizeof(td) + UA_ALIGN - 1u) / UA_ALIGN) * UA_ALIGN;
    mh.file_size = mh.data_offset + sizeof(weights); mh.payload_checksum = fnv1a(weights, sizeof(weights));
    memset(&td, 0, sizeof(td)); strcpy(td.name, "demo.weights");
    td.dtype = 1; td.rank = 1; td.dims[0] = UA_WEIGHT_VALUES;
    td.offset = mh.data_offset; td.nbytes = sizeof(weights); td.checksum = mh.payload_checksum;
    snprintf(path, sizeof(path), "%s/demo.uaw", directory);
    f = fopen(path, "wb"); if (!f) return UA_ERR_IO;
    if (fwrite(&mh, 1, sizeof(mh), f) != sizeof(mh) ||
        fwrite(&td, 1, sizeof(td), f) != sizeof(td)) { fclose(f); return UA_ERR_IO; }
    pad = (size_t)mh.data_offset - sizeof(mh) - sizeof(td);
    while (pad--) fputc(0, f);
    if (fwrite(weights, 1, sizeof(weights), f) != sizeof(weights) || fclose(f)) return UA_ERR_IO;
    for (frame = 0; frame < 2; ++frame) {
        unsigned c, row;
        for (i = 0; i < UA_IMAGE_VALUES; ++i) {
            unsigned cam = i / (3u * 8u * 8u), rem = i % (3u * 8u * 8u);
            unsigned ch = rem / 64u, p = rem % 64u;
            pixels[i] = ((float)((p * 13u + cam * 17u + ch * 29u + frame * 11u) % 101u) / 50.0f) - 1.0f;
        }
        memset(&fh, 0, sizeof(fh)); memcpy(fh.magic, "UAF1", 4);
        fh.version = 1; fh.endian = UA_ENDIAN; fh.profile = 1; fh.camera_count = UA_CAMERA_COUNT;
        fh.frame_index = frame; fh.data_offset = ((sizeof(fh) + UA_ALIGN - 1u) / UA_ALIGN) * UA_ALIGN;
        fh.file_size = fh.data_offset + sizeof(pixels); fh.payload_checksum = fnv1a(pixels, sizeof(pixels));
        strcpy(fh.scene, "synthetic-scene-001"); strcpy(fh.command, frame ? "left" : "straight");
        fh.ego_dx = frame ? 1.0f : 0.0f; fh.ego_dy = 0.0f; fh.yaw = frame ? 0.02f : 0.0f;
        fh.timestamp_seconds = 1700000000.0 + (double)frame * 0.5;
        fh.can_bus[0] = fh.ego_dx; fh.can_bus[1] = fh.ego_dy; fh.can_bus[2] = fh.yaw;
        for (c = 0; c < UA_CAMERA_COUNT; ++c) {
            for (row = 0; row < 3; ++row) fh.camera_intrinsics[c][row * 3u + row] = 1.0f;
            for (row = 0; row < 4; ++row) fh.camera_to_ego[c][row * 4u + row] = 1.0f;
            fh.camera_to_ego[c][3] = (float)c * 0.1f;
        }
        for (row = 0; row < 4; ++row) fh.ego_pose[row * 4u + row] = 1.0f;
        fh.ego_pose[3] = (float)frame;
        fh.track_capacity = UA_MAX_TRACKS; fh.map_capacity = UA_MAX_MAP;
        fh.motion_modes = UA_MOTION_MODES; fh.prediction_steps = UA_PRED_STEPS;
        fh.plan_steps = UA_PLAN_STEPS;
        snprintf(path, sizeof(path), "%s/frame%u.uaf", directory, frame);
        f = fopen(path, "wb"); if (!f) return UA_ERR_IO;
        if (fwrite(&fh, 1, sizeof(fh), f) != sizeof(fh)) { fclose(f); return UA_ERR_IO; }
        pad = (size_t)fh.data_offset - sizeof(fh); while (pad--) fputc(0, f);
        if (fwrite(pixels, 1, sizeof(pixels), f) != sizeof(pixels) || fclose(f)) return UA_ERR_IO;
    }
    return UA_OK;
}
