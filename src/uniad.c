#define _POSIX_C_SOURCE 200809L
#include "uniad.h"
#include "sha256.h"
#ifdef UA_WITH_CUDA
#include "cuda_backend.h"
#endif

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
typedef struct {
    char magic[4];
    uint32_t version, endian, profile, count, header_size;
    uint64_t directory_offset, data_offset, file_size;
    uint8_t config_sha256[32], checkpoint_sha256[32], directory_sha256[32];
    uint8_t reserved[112];
} disk_uaw2_header;
typedef struct {
    char name[128];
    uint32_t dtype, rank, flags, reserved0;
    uint64_t dims[8], offset, nbytes;
    uint8_t sha256[32], reserved[8];
} disk_uaw2_tensor;
typedef struct {
    char name[128];
    ua_tensor_info info;
} production_tensor_record;

struct ua_model {
    char profile[32];
    uint64_t seed;
    float weights[UA_WEIGHT_VALUES];
    unsigned char *production_blob;
    production_tensor_record *production_directory;
    size_t production_bytes;
    uint32_t tensor_count;
    uint8_t checkpoint_sha256[32], config_sha256[32];
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
    void *cuda_context;
};

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

static int valid_rigid_transform(const float matrix[16]) {
    const float tolerance = 2e-3f;
    size_t column, other, row;
    if (fabsf(matrix[12]) > tolerance ||
        fabsf(matrix[13]) > tolerance ||
        fabsf(matrix[14]) > tolerance ||
        fabsf(matrix[15] - 1.0f) > tolerance)
        return 0;
    for (column = 0; column < 3u; ++column)
        for (other = column; other < 3u; ++other) {
            float dot = 0.0f;
            for (row = 0; row < 3u; ++row)
                dot += matrix[row * 4u + column] *
                       matrix[row * 4u + other];
            if (fabsf(dot - (column == other ? 1.0f : 0.0f)) > tolerance)
                return 0;
        }
    return 1;
}

static int valid_camera_intrinsic(const float matrix[9]) {
    return fabsf(matrix[0]) > 1e-6f && fabsf(matrix[4]) > 1e-6f &&
           fabsf(matrix[8]) > 1e-6f;
}

const char *ua_status_string(ua_status s) {
    static const char *const names[] = {
        "ok", "invalid argument", "I/O error", "invalid format",
        "checksum mismatch", "capacity exceeded", "non-finite input",
        "incompatible profile", "backend unavailable",
        "production operator graph unavailable", "out of memory"
    };
    return (unsigned)s < sizeof(names) / sizeof(names[0]) ? names[s] : "unknown";
}

static ua_status read_exact(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n ? UA_OK : UA_ERR_IO;
}

static ua_status load_uaw1(FILE *f, ua_model **out) {
    disk_model_header h;
    disk_tensor *dir = NULL;
    ua_model *m = NULL;
    uint32_t i, j;
    ua_status status = UA_ERR_FORMAT;
    if (!f || !out) return UA_ERR_ARGUMENT;
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
    free(m); free(dir); return status;
}

static ua_status load_uaw2(FILE *f, ua_model **out) {
    disk_uaw2_header h;
    disk_uaw2_tensor *dir = NULL;
    ua_model *m = NULL;
    uint8_t digest[32];
    uint32_t i, j;
    ua_status status = UA_ERR_FORMAT;
    if (read_exact(f, &h, sizeof(h)) != UA_OK) return UA_ERR_IO;
    if (memcmp(h.magic, "UAW2", 4) || h.version != 2 || h.endian != UA_ENDIAN ||
        h.profile != 2 || h.header_size != sizeof(h) ||
        h.directory_offset != sizeof(h) || h.count == 0 || h.count > 100000 ||
        h.data_offset % 256u || h.data_offset > h.file_size) return status;
    if (h.directory_offset + (uint64_t)h.count * sizeof(*dir) > h.data_offset)
        return status;
    dir = (disk_uaw2_tensor *)calloc(h.count, sizeof(*dir));
    if (!dir) return UA_ERR_MEMORY;
    if (read_exact(f, dir, (size_t)h.count * sizeof(*dir)) != UA_OK) {
        status = UA_ERR_IO; goto done;
    }
    ua_sha256(dir, (size_t)h.count * sizeof(*dir), digest);
    if (memcmp(digest, h.directory_sha256, 32)) {
        status = UA_ERR_CHECKSUM; goto done;
    }
    for (i = 0; i < h.count; ++i) {
        uint64_t elements = 1, expected;
        size_t item_size;
        if (!memchr(dir[i].name, '\0', sizeof(dir[i].name)) ||
            dir[i].rank > 8 || dir[i].offset % 256u ||
            dir[i].offset < h.data_offset || dir[i].offset > h.file_size ||
            dir[i].nbytes > h.file_size - dir[i].offset) goto done;
        item_size = dir[i].dtype == 1 ? 2u :
                    (dir[i].dtype == 2 || dir[i].dtype == 4) ? 4u :
                    dir[i].dtype == 3 ? 8u : dir[i].dtype == 5 ? 1u : 0u;
        if (!item_size) goto done;
        for (j = 0; j < dir[i].rank; ++j) {
            if (dir[i].dims[j] && elements > UINT64_MAX / dir[i].dims[j]) goto done;
            elements *= dir[i].dims[j];
        }
        if (elements > UINT64_MAX / item_size) goto done;
        expected = elements * item_size;
        if (expected != dir[i].nbytes) goto done;
        for (j = 0; j < i; ++j)
            if (!strcmp(dir[i].name, dir[j].name)) goto done;
    }
    if (h.file_size - h.data_offset > SIZE_MAX) goto done;
    m = (ua_model *)calloc(1, sizeof(*m));
    if (!m) { status = UA_ERR_MEMORY; goto done; }
    m->production_bytes = (size_t)(h.file_size - h.data_offset);
    m->production_blob = (unsigned char *)malloc(m->production_bytes);
    if (!m->production_blob) { status = UA_ERR_MEMORY; goto done; }
    if (fseek(f, (long)h.data_offset, SEEK_SET) ||
        read_exact(f, m->production_blob, m->production_bytes) != UA_OK) {
        status = UA_ERR_IO; goto done;
    }
    for (i = 0; i < h.count; ++i) {
        const unsigned char *data =
            m->production_blob + (size_t)(dir[i].offset - h.data_offset);
        ua_sha256(data, (size_t)dir[i].nbytes, digest);
        if (memcmp(digest, dir[i].sha256, 32)) {
            status = UA_ERR_CHECKSUM; goto done;
        }
    }
    m->production_directory =
        (production_tensor_record *)calloc(h.count, sizeof(*m->production_directory));
    if (!m->production_directory) { status = UA_ERR_MEMORY; goto done; }
    for (i = 0; i < h.count; ++i) {
        production_tensor_record *record = &m->production_directory[i];
        memcpy(record->name, dir[i].name, sizeof(record->name));
        record->info.dtype = dir[i].dtype;
        record->info.rank = dir[i].rank;
        memcpy(record->info.dims, dir[i].dims, sizeof(record->info.dims));
        record->info.byte_offset = dir[i].offset - h.data_offset;
        record->info.nbytes = dir[i].nbytes;
    }
    strcpy(m->profile, "production-nuscenes-stage2-v2");
    m->tensor_count = h.count;
    memcpy(m->checkpoint_sha256, h.checkpoint_sha256, 32);
    memcpy(m->config_sha256, h.config_sha256, 32);
    m->owned = sizeof(*m) + m->production_bytes +
               (size_t)h.count * sizeof(*m->production_directory);
    *out = m; m = NULL; status = UA_OK;
done:
    if (m) {
        free(m->production_directory);
        free(m->production_blob);
        free(m);
    }
    free(dir);
    return status;
}

ua_status ua_model_load(const char *path, ua_model **out) {
    FILE *f;
    char magic[4];
    ua_status status;
    if (!path || !out) return UA_ERR_ARGUMENT;
    *out = NULL;
    f = fopen(path, "rb");
    if (!f) return UA_ERR_IO;
    if (read_exact(f, magic, sizeof(magic)) != UA_OK || fseek(f, 0, SEEK_SET)) {
        fclose(f); return UA_ERR_IO;
    }
    if (!memcmp(magic, "UAW1", 4)) status = load_uaw1(f, out);
    else if (!memcmp(magic, "UAW2", 4)) status = load_uaw2(f, out);
    else status = UA_ERR_FORMAT;
    fclose(f);
    return status;
}

void ua_model_destroy(ua_model *m) {
    if (m) {
        free(m->production_directory);
        free(m->production_blob);
    }
    free(m);
}
const char *ua_model_profile(const ua_model *m) { return m ? m->profile : NULL; }
uint64_t ua_model_seed(const ua_model *m) { return m ? m->seed : 0; }
uint32_t ua_model_tensor_count(const ua_model *m) { return m ? m->tensor_count : 0; }
ua_status ua_model_find_tensor(const ua_model *m, const char *name,
                               ua_tensor_info *info) {
    uint32_t i;
    if (!m || !name || !info) return UA_ERR_ARGUMENT;
    if (strcmp(m->profile, "production-nuscenes-stage2-v2"))
        return UA_ERR_UNSUPPORTED_PROFILE;
    for (i = 0; i < m->tensor_count; ++i) {
        if (!strcmp(m->production_directory[i].name, name)) {
            *info = m->production_directory[i].info;
            return UA_OK;
        }
    }
    return UA_ERR_IO;
}

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
    if (!strcmp(m->profile, "production-nuscenes-stage2-v2")) {
        if (backend != UA_BACKEND_CUDA) return UA_ERR_UNSUPPORTED_PROFILE;
#ifndef UA_WITH_CUDA
        return UA_ERR_BACKEND;
#endif
    } else if (strcmp(m->profile, "tiny-synthetic-v1")) {
        return UA_ERR_PROFILE;
    }
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
    if (backend == UA_BACKEND_CUDA &&
        !strcmp(m->profile, "production-nuscenes-stage2-v2")) {
        ua_status s = ua_cuda_production_create(
            m->production_blob, m->production_bytes, &c->cuda_context,
            &c->metrics.owned_device_bytes);
        if (s != UA_OK) { free(c); return s; }
    } else if (backend == UA_BACKEND_CUDA) {
        c->metrics.owned_device_bytes =
            (UA_IMAGE_VALUES + UA_BEV_VALUES + UA_WEIGHT_VALUES) * sizeof(float);
    }
#endif
    *out = c;
    return UA_OK;
}

void ua_context_reset(ua_context *c) {
    if (!c) return;
#ifdef UA_WITH_CUDA
    if (c->cuda_context) ua_cuda_production_reset(c->cuda_context);
#endif
    memset(c->prev_bev, 0, sizeof(c->prev_bev));
    memset(c->previous_scene, 0, sizeof(c->previous_scene));
    c->has_previous = 0;
}
void ua_context_destroy(ua_context *c) {
    if (!c) return;
#ifdef UA_WITH_CUDA
    if (c->cuda_context) ua_cuda_production_destroy(c->cuda_context);
#endif
    free(c);
}
void ua_context_metrics(const ua_context *c, ua_metrics *m) { if (c && m) *m = c->metrics; }
ua_status ua_context_debug_tensor_f32(
        const ua_context *c, const char *name, float *values,
        size_t capacity, size_t *written) {
    if (!c || !name || !values || !capacity || !written)
        return UA_ERR_ARGUMENT;
    if (c->backend != UA_BACKEND_CUDA || !c->cuda_context)
        return UA_ERR_BACKEND;
#ifdef UA_WITH_CUDA
    return ua_cuda_production_copy_boundary_f32(
        c->cuda_context, name, values, capacity, written);
#else
    return UA_ERR_BACKEND;
#endif
}
ua_status ua_context_debug_run_query_interaction(
        ua_context *c, size_t active_queries) {
    if (!c || !active_queries || active_queries > UA_PROD_MAX_TRACKS)
        return UA_ERR_ARGUMENT;
    if (c->backend != UA_BACKEND_CUDA || !c->cuda_context)
        return UA_ERR_BACKEND;
#ifdef UA_WITH_CUDA
    return ua_cuda_production_debug_query_interaction(
        c->cuda_context, active_queries);
#else
    return UA_ERR_BACKEND;
#endif
}

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

#ifdef UA_WITH_CUDA
static ua_status production_tensor_pointer(
        ua_context *c, const char *name, uint32_t dtype, uint32_t rank,
        const uint64_t *dims, const void **pointer) {
    ua_tensor_info info;
    ua_status status = ua_model_find_tensor(c->model, name, &info);
    if (status != UA_OK) return status;
    if (info.dtype != dtype || info.rank != rank) return UA_ERR_FORMAT;
    for (uint32_t dimension = 0; dimension < rank; ++dimension)
        if (info.dims[dimension] != dims[dimension]) return UA_ERR_FORMAT;
    return ua_cuda_production_tensor_pointer(
        c->cuda_context, (size_t)info.byte_offset, (size_t)info.nbytes,
        pointer);
}

static ua_status resolve_conv_bn(
        ua_context *c, const char *conv_name, const char *bn_prefix,
        size_t output_channels, size_t input_channels, size_t kernel,
        ua_cuda_conv_bn_weights *weights) {
    char name[192];
    uint64_t conv_dims[4] = {
        output_channels, input_channels, kernel, kernel
    };
    uint64_t channel_dims[1] = {output_channels};
    ua_status status = production_tensor_pointer(
        c, conv_name, 1u, 4u, conv_dims, &weights->weight_fp16);
    if (status != UA_OK) return status;
#define UA_RESOLVE_BN(suffix, dtype_, field) do { \
    if (snprintf(name, sizeof(name), "%s.%s", bn_prefix, suffix) < 0) \
        return UA_ERR_FORMAT; \
    status = production_tensor_pointer( \
        c, name, dtype_, 1u, channel_dims, &weights->field); \
    if (status != UA_OK) return status; \
} while (0)
    UA_RESOLVE_BN("weight", 1u, gamma_fp16);
    UA_RESOLVE_BN("bias", 1u, beta_fp16);
    UA_RESOLVE_BN("running_mean", 2u, mean_fp32);
    UA_RESOLVE_BN("running_var", 2u, variance_fp32);
#undef UA_RESOLVE_BN
    return UA_OK;
}

static ua_status resolve_dcn_offset(
        ua_context *c, const char *prefix, size_t input_channels,
        ua_cuda_bottleneck_weights *weights) {
    char name[192];
    uint64_t weight_dims[4] = {27u, input_channels, 3u, 3u};
    uint64_t bias_dims[1] = {27u};
    ua_status status;
    if (snprintf(
            name, sizeof(name), "%s.conv_offset.weight", prefix) < 0)
        return UA_ERR_FORMAT;
    status = production_tensor_pointer(
        c, name, 1u, 4u, weight_dims,
        &weights->conv2_offset_weight_fp16);
    if (status != UA_OK) return status;
    if (snprintf(name, sizeof(name), "%s.conv_offset.bias", prefix) < 0)
        return UA_ERR_FORMAT;
    status = production_tensor_pointer(
        c, name, 1u, 1u, bias_dims, &weights->conv2_offset_bias_fp16);
    if (status != UA_OK) return status;
    weights->conv2_is_dcn = 1;
    return UA_OK;
}

static ua_status resolve_conv_bias(
        ua_context *c, const char *prefix, size_t output_channels,
        size_t input_channels, size_t kernel,
        ua_cuda_conv_bias_weights *weights) {
    char name[192];
    uint64_t weight_dims[4] = {
        output_channels, input_channels, kernel, kernel
    };
    uint64_t bias_dims[1] = {output_channels};
    ua_status status;
    if (snprintf(name, sizeof(name), "%s.weight", prefix) < 0)
        return UA_ERR_FORMAT;
    status = production_tensor_pointer(
        c, name, 1u, 4u, weight_dims, &weights->weight_fp16);
    if (status != UA_OK) return status;
    if (snprintf(name, sizeof(name), "%s.bias", prefix) < 0)
        return UA_ERR_FORMAT;
    return production_tensor_pointer(
        c, name, 1u, 1u, bias_dims, &weights->bias_fp16);
}

static ua_status resolve_encoder_layer(
        ua_context *c, size_t layer,
        ua_cuda_encoder_layer_weights *weights) {
    const uint64_t projection_weight_dims[2] = {256u, 256u};
    const uint64_t projection_bias_dims[1] = {256u};
    const uint64_t temporal_offset_weight_dims[2] = {128u, 512u};
    const uint64_t temporal_offset_bias_dims[1] = {128u};
    const uint64_t temporal_attention_weight_dims[2] = {64u, 512u};
    const uint64_t temporal_attention_bias_dims[1] = {64u};
    const uint64_t spatial_offset_weight_dims[2] = {512u, 256u};
    const uint64_t spatial_offset_bias_dims[1] = {512u};
    const uint64_t spatial_attention_weight_dims[2] = {256u, 256u};
    const uint64_t ffn0_weight_dims[2] = {512u, 256u};
    const uint64_t ffn0_bias_dims[1] = {512u};
    const uint64_t ffn1_weight_dims[2] = {256u, 512u};
    char prefix[160], name[224];
    ua_status status;
    if (!c || !weights || layer >= 6u) return UA_ERR_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
    if (snprintf(
            prefix, sizeof(prefix),
            "pts_bbox_head.transformer.encoder.layers.%zu", layer) < 0)
        return UA_ERR_FORMAT;
#define UA_ENCODER_TENSOR(path_, rank_, dims_, target_) do { \
    if (snprintf(name, sizeof(name), "%s.%s", prefix, path_) < 0) \
        return UA_ERR_FORMAT; \
    status = production_tensor_pointer( \
        c, name, 1u, rank_, dims_, target_); \
    if (status != UA_OK) return status; \
} while (0)
    UA_ENCODER_TENSOR(
        "attentions.0.value_proj.weight", 2u, projection_weight_dims,
        &weights->temporal.value_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.value_proj.bias", 1u, projection_bias_dims,
        &weights->temporal.value_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.sampling_offsets.weight", 2u,
        temporal_offset_weight_dims, &weights->temporal.offset_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.sampling_offsets.bias", 1u,
        temporal_offset_bias_dims, &weights->temporal.offset_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.attention_weights.weight", 2u,
        temporal_attention_weight_dims,
        &weights->temporal.attention_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.attention_weights.bias", 1u,
        temporal_attention_bias_dims,
        &weights->temporal.attention_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.output_proj.weight", 2u, projection_weight_dims,
        &weights->temporal.output_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.0.output_proj.bias", 1u, projection_bias_dims,
        &weights->temporal.output_bias_fp16);
    UA_ENCODER_TENSOR(
        "norms.0.weight", 1u, projection_bias_dims,
        &weights->norm0_weight_fp16);
    UA_ENCODER_TENSOR(
        "norms.0.bias", 1u, projection_bias_dims,
        &weights->norm0_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.value_proj.weight", 2u,
        projection_weight_dims, &weights->spatial.value_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.value_proj.bias", 1u,
        projection_bias_dims, &weights->spatial.value_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.sampling_offsets.weight", 2u,
        spatial_offset_weight_dims, &weights->spatial.offset_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.sampling_offsets.bias", 1u,
        spatial_offset_bias_dims, &weights->spatial.offset_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.attention_weights.weight", 2u,
        spatial_attention_weight_dims,
        &weights->spatial.attention_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.deformable_attention.attention_weights.bias", 1u,
        projection_bias_dims, &weights->spatial.attention_bias_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.output_proj.weight", 2u, projection_weight_dims,
        &weights->spatial.output_weight_fp16);
    UA_ENCODER_TENSOR(
        "attentions.1.output_proj.bias", 1u, projection_bias_dims,
        &weights->spatial.output_bias_fp16);
    UA_ENCODER_TENSOR(
        "norms.1.weight", 1u, projection_bias_dims,
        &weights->norm1_weight_fp16);
    UA_ENCODER_TENSOR(
        "norms.1.bias", 1u, projection_bias_dims,
        &weights->norm1_bias_fp16);
    UA_ENCODER_TENSOR(
        "ffns.0.layers.0.0.weight", 2u, ffn0_weight_dims,
        &weights->ffn.linear0_weight_fp16);
    UA_ENCODER_TENSOR(
        "ffns.0.layers.0.0.bias", 1u, ffn0_bias_dims,
        &weights->ffn.linear0_bias_fp16);
    UA_ENCODER_TENSOR(
        "ffns.0.layers.1.weight", 2u, ffn1_weight_dims,
        &weights->ffn.linear1_weight_fp16);
    UA_ENCODER_TENSOR(
        "ffns.0.layers.1.bias", 1u, projection_bias_dims,
        &weights->ffn.linear1_bias_fp16);
    UA_ENCODER_TENSOR(
        "norms.2.weight", 1u, projection_bias_dims,
        &weights->ffn.norm_weight_fp16);
    UA_ENCODER_TENSOR(
        "norms.2.bias", 1u, projection_bias_dims,
        &weights->ffn.norm_bias_fp16);
#undef UA_ENCODER_TENSOR
    return UA_OK;
}

static ua_status resolve_track_decoder_layer(
        ua_context *c, size_t layer,
        ua_cuda_track_decoder_layer_weights *weights) {
    const uint64_t packed_weight_dims[2] = {768u, 256u};
    const uint64_t packed_bias_dims[1] = {768u};
    const uint64_t projection_weight_dims[2] = {256u, 256u};
    const uint64_t projection_bias_dims[1] = {256u};
    const uint64_t cross_offset_weight_dims[2] = {64u, 256u};
    const uint64_t cross_offset_bias_dims[1] = {64u};
    const uint64_t cross_attention_weight_dims[2] = {32u, 256u};
    const uint64_t cross_attention_bias_dims[1] = {32u};
    const uint64_t ffn0_weight_dims[2] = {512u, 256u};
    const uint64_t ffn0_bias_dims[1] = {512u};
    const uint64_t ffn1_weight_dims[2] = {256u, 512u};
    char prefix[160], name[224];
    ua_status status;
    if (!c || !weights || layer >= 6u) return UA_ERR_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
    if (snprintf(
            prefix, sizeof(prefix),
            "pts_bbox_head.transformer.decoder.layers.%zu", layer) < 0)
        return UA_ERR_FORMAT;
#define UA_TRACK_DECODER_TENSOR(path_, rank_, dims_, target_) do { \
    if (snprintf(name, sizeof(name), "%s.%s", prefix, path_) < 0) \
        return UA_ERR_FORMAT; \
    status = production_tensor_pointer( \
        c, name, 1u, rank_, dims_, target_); \
    if (status != UA_OK) return status; \
} while (0)
    UA_TRACK_DECODER_TENSOR(
        "attentions.0.attn.in_proj_weight", 2u, packed_weight_dims,
        &weights->self_in_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.0.attn.in_proj_bias", 1u, packed_bias_dims,
        &weights->self_in_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.0.attn.out_proj.weight", 2u, projection_weight_dims,
        &weights->self_out_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.0.attn.out_proj.bias", 1u, projection_bias_dims,
        &weights->self_out_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.0.weight", 1u, projection_bias_dims,
        &weights->norm0_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.0.bias", 1u, projection_bias_dims,
        &weights->norm0_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.value_proj.weight", 2u, projection_weight_dims,
        &weights->cross_value_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.value_proj.bias", 1u, projection_bias_dims,
        &weights->cross_value_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.sampling_offsets.weight", 2u,
        cross_offset_weight_dims, &weights->cross_offset_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.sampling_offsets.bias", 1u,
        cross_offset_bias_dims, &weights->cross_offset_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.attention_weights.weight", 2u,
        cross_attention_weight_dims, &weights->cross_attention_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.attention_weights.bias", 1u,
        cross_attention_bias_dims, &weights->cross_attention_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.output_proj.weight", 2u, projection_weight_dims,
        &weights->cross_out_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "attentions.1.output_proj.bias", 1u, projection_bias_dims,
        &weights->cross_out_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.1.weight", 1u, projection_bias_dims,
        &weights->norm1_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.1.bias", 1u, projection_bias_dims,
        &weights->norm1_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "ffns.0.layers.0.0.weight", 2u, ffn0_weight_dims,
        &weights->ffn.linear0_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "ffns.0.layers.0.0.bias", 1u, ffn0_bias_dims,
        &weights->ffn.linear0_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "ffns.0.layers.1.weight", 2u, ffn1_weight_dims,
        &weights->ffn.linear1_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "ffns.0.layers.1.bias", 1u, projection_bias_dims,
        &weights->ffn.linear1_bias_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.2.weight", 1u, projection_bias_dims,
        &weights->ffn.norm_weight_fp16);
    UA_TRACK_DECODER_TENSOR(
        "norms.2.bias", 1u, projection_bias_dims,
        &weights->ffn.norm_bias_fp16);
#undef UA_TRACK_DECODER_TENSOR
    return UA_OK;
}

static ua_status resolve_track_regression_branch(
        ua_context *c, size_t layer,
        ua_cuda_track_regression_weights *weights) {
    const uint64_t hidden_weight_dims[2] = {256u, 256u};
    const uint64_t hidden_bias_dims[1] = {256u};
    const uint64_t output_weight_dims[2] = {10u, 256u};
    const uint64_t output_bias_dims[1] = {10u};
    char name[160];
    ua_status status;
    if (!c || !weights || layer >= 6u) return UA_ERR_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
#define UA_TRACK_REGRESSION_TENSOR(slot_, suffix_, rank_, dims_, target_) do { \
    if (snprintf( \
            name, sizeof(name), "pts_bbox_head.reg_branches.%zu.%s", \
            layer, suffix_) < 0) \
        return UA_ERR_FORMAT; \
    status = production_tensor_pointer(c, name, 1u, rank_, dims_, target_); \
    if (status != UA_OK) return status; \
    (void)(slot_); \
} while (0)
    UA_TRACK_REGRESSION_TENSOR(
        0, "0.weight", 2u, hidden_weight_dims,
        &weights->linear0_weight_fp16);
    UA_TRACK_REGRESSION_TENSOR(
        0, "0.bias", 1u, hidden_bias_dims,
        &weights->linear0_bias_fp16);
    UA_TRACK_REGRESSION_TENSOR(
        1, "2.weight", 2u, hidden_weight_dims,
        &weights->linear1_weight_fp16);
    UA_TRACK_REGRESSION_TENSOR(
        1, "2.bias", 1u, hidden_bias_dims,
        &weights->linear1_bias_fp16);
    UA_TRACK_REGRESSION_TENSOR(
        2, "4.weight", 2u, output_weight_dims,
        &weights->linear2_weight_fp16);
    UA_TRACK_REGRESSION_TENSOR(
        2, "4.bias", 1u, output_bias_dims,
        &weights->linear2_bias_fp16);
#undef UA_TRACK_REGRESSION_TENSOR
    return UA_OK;
}

static ua_status resolve_track_output_heads(
        ua_context *c, size_t layer,
        ua_cuda_track_classification_weights *classification,
        ua_cuda_track_past_trajectory_weights *past) {
    const uint64_t hidden_weight_dims[2] = {256u, 256u};
    const uint64_t hidden_bias_dims[1] = {256u};
    const uint64_t class_weight_dims[2] = {10u, 256u};
    const uint64_t class_bias_dims[1] = {10u};
    const uint64_t past_weight_dims[2] = {16u, 256u};
    const uint64_t past_bias_dims[1] = {16u};
    char name[192];
    ua_status status;
    if (!c || !classification || !past || layer >= 6u)
        return UA_ERR_ARGUMENT;
    memset(classification, 0, sizeof(*classification));
    memset(past, 0, sizeof(*past));
#define UA_TRACK_HEAD_TENSOR(branch_, suffix_, rank_, dims_, target_) do { \
    if (snprintf( \
            name, sizeof(name), "pts_bbox_head.%s.%zu.%s", \
            branch_, layer, suffix_) < 0) \
        return UA_ERR_FORMAT; \
    status = production_tensor_pointer(c, name, 1u, rank_, dims_, target_); \
    if (status != UA_OK) return status; \
} while (0)
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "0.weight", 2u, hidden_weight_dims,
        &classification->linear0_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "0.bias", 1u, hidden_bias_dims,
        &classification->linear0_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "1.weight", 1u, hidden_bias_dims,
        &classification->norm0_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "1.bias", 1u, hidden_bias_dims,
        &classification->norm0_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "3.weight", 2u, hidden_weight_dims,
        &classification->linear1_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "3.bias", 1u, hidden_bias_dims,
        &classification->linear1_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "4.weight", 1u, hidden_bias_dims,
        &classification->norm1_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "4.bias", 1u, hidden_bias_dims,
        &classification->norm1_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "6.weight", 2u, class_weight_dims,
        &classification->output_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "cls_branches", "6.bias", 1u, class_bias_dims,
        &classification->output_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "0.weight", 2u, hidden_weight_dims,
        &past->linear0_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "0.bias", 1u, hidden_bias_dims,
        &past->linear0_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "2.weight", 2u, hidden_weight_dims,
        &past->linear1_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "2.bias", 1u, hidden_bias_dims,
        &past->linear1_bias_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "4.weight", 2u, past_weight_dims,
        &past->output_weight_fp16);
    UA_TRACK_HEAD_TENSOR(
        "past_traj_reg_branches", "4.bias", 1u, past_bias_dims,
        &past->output_bias_fp16);
#undef UA_TRACK_HEAD_TENSOR
    return UA_OK;
}

static ua_status resolve_query_interaction(
        ua_context *c, ua_cuda_query_interaction_weights *weights) {
    const uint64_t packed_weight_dims[2] = {768u, 256u};
    const uint64_t packed_bias_dims[1] = {768u};
    const uint64_t projection_weight_dims[2] = {256u, 256u};
    const uint64_t projection_bias_dims[1] = {256u};
    ua_status status;
    if (!c || !weights) return UA_ERR_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
#define UA_QIM_TENSOR(name_, rank_, dims_, target_) do { \
    status = production_tensor_pointer( \
        c, "query_interact." name_, 1u, rank_, dims_, target_); \
    if (status != UA_OK) return status; \
} while (0)
    UA_QIM_TENSOR(
        "self_attn.in_proj_weight", 2u, packed_weight_dims,
        &weights->self_in_weight_fp16);
    UA_QIM_TENSOR(
        "self_attn.in_proj_bias", 1u, packed_bias_dims,
        &weights->self_in_bias_fp16);
    UA_QIM_TENSOR(
        "self_attn.out_proj.weight", 2u, projection_weight_dims,
        &weights->self_out_weight_fp16);
    UA_QIM_TENSOR(
        "self_attn.out_proj.bias", 1u, projection_bias_dims,
        &weights->self_out_bias_fp16);
    UA_QIM_TENSOR(
        "norm1.weight", 1u, projection_bias_dims,
        &weights->norm1_weight_fp16);
    UA_QIM_TENSOR(
        "norm1.bias", 1u, projection_bias_dims,
        &weights->norm1_bias_fp16);
    UA_QIM_TENSOR(
        "linear1.weight", 2u, projection_weight_dims,
        &weights->linear1_weight_fp16);
    UA_QIM_TENSOR(
        "linear1.bias", 1u, projection_bias_dims,
        &weights->linear1_bias_fp16);
    UA_QIM_TENSOR(
        "linear2.weight", 2u, projection_weight_dims,
        &weights->linear2_weight_fp16);
    UA_QIM_TENSOR(
        "linear2.bias", 1u, projection_bias_dims,
        &weights->linear2_bias_fp16);
    UA_QIM_TENSOR(
        "norm2.weight", 1u, projection_bias_dims,
        &weights->norm2_weight_fp16);
    UA_QIM_TENSOR(
        "norm2.bias", 1u, projection_bias_dims,
        &weights->norm2_bias_fp16);
    UA_QIM_TENSOR(
        "linear_feat1.weight", 2u, projection_weight_dims,
        &weights->feat1_weight_fp16);
    UA_QIM_TENSOR(
        "linear_feat1.bias", 1u, projection_bias_dims,
        &weights->feat1_bias_fp16);
    UA_QIM_TENSOR(
        "linear_feat2.weight", 2u, projection_weight_dims,
        &weights->feat2_weight_fp16);
    UA_QIM_TENSOR(
        "linear_feat2.bias", 1u, projection_bias_dims,
        &weights->feat2_bias_fp16);
    UA_QIM_TENSOR(
        "norm_feat.weight", 1u, projection_bias_dims,
        &weights->norm_feat_weight_fp16);
    UA_QIM_TENSOR(
        "norm_feat.bias", 1u, projection_bias_dims,
        &weights->norm_feat_bias_fp16);
#undef UA_QIM_TENSOR
    return UA_OK;
}
#endif

ua_status ua_infer_production(ua_context *c, const ua_production_input *input,
                              ua_production_result *result) {
    unsigned i;
    if (!c || !input || !result || input->version != 2 ||
        !input->scene_token) return UA_ERR_ARGUMENT;
    if (strcmp(c->model->profile, "production-nuscenes-stage2-v2"))
        return UA_ERR_PROFILE;
    if (c->backend != UA_BACKEND_CUDA || !c->cuda_context)
        return UA_ERR_BACKEND;
    for (i = 0; i < UA_CAMERA_COUNT; ++i) {
        const ua_bgr_image_view *view = &input->cameras[i];
        if (!view->data || !view->width || !view->height ||
            view->row_stride_bytes < (size_t)view->width * 3u)
            return UA_ERR_ARGUMENT;
    }
    if (!all_finite(&input->camera_intrinsics[0][0], UA_CAMERA_COUNT * 9u) ||
        !all_finite(&input->camera_to_ego[0][0], UA_CAMERA_COUNT * 16u) ||
        !all_finite(input->ego_pose, 16) || !all_finite(input->can_bus, 18))
        return UA_ERR_NONFINITE;
    for (i = 0; i < UA_CAMERA_COUNT; ++i)
        if (!valid_camera_intrinsic(input->camera_intrinsics[i]) ||
            !valid_rigid_transform(input->camera_to_ego[i]))
            return UA_ERR_ARGUMENT;
    if (!valid_rigid_transform(input->ego_pose)) return UA_ERR_ARGUMENT;
    memset(result, 0, sizeof(*result));
#ifdef UA_WITH_CUDA
    {
        static const char *const names[5] = {
            "img_backbone.conv1.weight", "img_backbone.bn1.weight",
            "img_backbone.bn1.bias", "img_backbone.bn1.running_mean",
            "img_backbone.bn1.running_var"
        };
        ua_tensor_info tensors[5];
        const void *device_tensors[5] = {NULL, NULL, NULL, NULL, NULL};
        const void *normalized_images = NULL;
        const void *pooled_stem = NULL;
        const void *layer1_output = NULL;
        const void *layer2_output = NULL;
        const void *layer3_output = NULL;
        const void *layer4_output = NULL;
        ua_cuda_bottleneck_weights layer1[3];
        ua_cuda_bottleneck_weights layer2[4];
        ua_cuda_bottleneck_weights layer3[23];
        ua_cuda_bottleneck_weights layer4[3];
        ua_cuda_conv_bias_weights lateral[3], fpn_convs[4];
        const void *fpn_outputs[4] = {NULL, NULL, NULL, NULL};
        const void *camera_embeds = NULL, *level_embeds = NULL;
        const void *flattened_features = NULL;
        const void *bev_query_weight = NULL;
        const void *col_embed = NULL, *row_embed = NULL;
        const void *bev_queries = NULL, *bev_pos = NULL;
        const void *reference_2d = NULL, *reference_3d = NULL;
        const void *reference_camera = NULL, *visibility = NULL;
        const void *visible_indices = NULL, *visible_counts = NULL;
        ua_cuda_can_bus_weights can_bus_weights;
        ua_cuda_encoder_layer_weights encoder_layer;
        const void *encoder_query = NULL;
        const void *temporal_output = NULL;
        const void *norm0_output = NULL;
        const void *spatial_output = NULL;
        const void *norm1_output = NULL;
        const void *ffn_output = NULL, *norm2_output = NULL;
        const void *track_query_embedding = NULL;
        const void *track_reference_weight = NULL;
        const void *track_reference_bias = NULL;
        const void *track_query_pos = NULL, *track_query = NULL;
        const void *track_reference_points = NULL;
        ua_cuda_track_decoder_layer_weights track_decoder_layer;
        ua_cuda_track_regression_weights track_regression_weights;
        ua_cuda_track_classification_weights track_classification_weights;
        ua_cuda_track_past_trajectory_weights track_past_weights;
        const void *track_decoder_output = NULL;
        const void *track_regression = NULL;
        const void *track_class_logits = NULL;
        const void *track_boxes = NULL;
        const void *track_past_trajectory = NULL;
        const void *track_scores = NULL, *track_classes = NULL;
        const void *track_selected_indices = NULL;
        const void *track_selected_count = NULL;
        ua_cuda_query_interaction_weights query_interaction_weights;
        const void *updated_track_query_feat = NULL;
        size_t h2d_bytes = 0;
        ua_status status;
        for (i = 0; i < 5; ++i) {
            status = ua_model_find_tensor(c->model, names[i], &tensors[i]);
            if (status != UA_OK) return status;
            status = ua_cuda_production_tensor_pointer(
                c->cuda_context, (size_t)tensors[i].byte_offset,
                (size_t)tensors[i].nbytes, &device_tensors[i]);
            if (status != UA_OK || !device_tensors[i]) return status;
        }
        if (tensors[0].dtype != 1 || tensors[0].rank != 4 ||
            tensors[0].dims[0] != 64 || tensors[0].dims[1] != 3 ||
            tensors[0].dims[2] != 7 || tensors[0].dims[3] != 7)
            return UA_ERR_FORMAT;
        for (i = 1; i < 5; ++i)
            if (tensors[i].dtype != (i < 3 ? 1u : 2u) ||
                tensors[i].rank != 1 ||
                tensors[i].dims[0] != 64)
                return UA_ERR_FORMAT;
        status = ua_cuda_production_preprocess(
            c->cuda_context, input, &h2d_bytes, &normalized_images);
        if (status != UA_OK || !normalized_images) return status;
        c->metrics.h2d_bytes = h2d_bytes;
        status = ua_cuda_production_resnet_stem(
            c->cuda_context, normalized_images, device_tensors[0],
            device_tensors[1], device_tensors[2], device_tensors[3],
            device_tensors[4], &pooled_stem);
        if (status != UA_OK || !pooled_stem) return status;
        memset(layer1, 0, sizeof(layer1));
        for (i = 0; i < 3; ++i) {
            char conv_name[128], bn_name[128];
            size_t conv1_inputs = i == 0 ? 64u : 256u;
#define UA_LAYER1_RESOLVE(part, bnpart, outc, inc, kernel_, target) do { \
    (void)snprintf(conv_name, sizeof(conv_name), \
                   "img_backbone.layer1.%u.%s.weight", i, part); \
    (void)snprintf(bn_name, sizeof(bn_name), \
                   "img_backbone.layer1.%u.%s", i, bnpart); \
    status = resolve_conv_bn(c, conv_name, bn_name, outc, inc, kernel_, target); \
    if (status != UA_OK) return status; \
} while (0)
            UA_LAYER1_RESOLVE(
                "conv1", "bn1", 64u, conv1_inputs, 1u, &layer1[i].conv1);
            UA_LAYER1_RESOLVE(
                "conv2", "bn2", 64u, 64u, 3u, &layer1[i].conv2);
            UA_LAYER1_RESOLVE(
                "conv3", "bn3", 256u, 64u, 1u, &layer1[i].conv3);
#undef UA_LAYER1_RESOLVE
        }
        status = resolve_conv_bn(
            c, "img_backbone.layer1.0.downsample.0.weight",
            "img_backbone.layer1.0.downsample.1", 256u, 64u, 1u,
            &layer1[0].downsample);
        if (status != UA_OK) return status;
        layer1[0].has_downsample = 1;
        status = ua_cuda_production_resnet_layer1(
            c->cuda_context, pooled_stem, layer1, &layer1_output);
        if (status != UA_OK || !layer1_output) return status;
        memset(layer2, 0, sizeof(layer2));
        for (i = 0; i < 4; ++i) {
            char conv_name[128], bn_name[128];
            size_t conv1_inputs = i == 0 ? 256u : 512u;
#define UA_LAYER2_RESOLVE(part, bnpart, outc, inc, kernel_, target) do { \
    (void)snprintf(conv_name, sizeof(conv_name), \
                   "img_backbone.layer2.%u.%s.weight", i, part); \
    (void)snprintf(bn_name, sizeof(bn_name), \
                   "img_backbone.layer2.%u.%s", i, bnpart); \
    status = resolve_conv_bn(c, conv_name, bn_name, outc, inc, kernel_, target); \
    if (status != UA_OK) return status; \
} while (0)
            UA_LAYER2_RESOLVE(
                "conv1", "bn1", 128u, conv1_inputs, 1u, &layer2[i].conv1);
            UA_LAYER2_RESOLVE(
                "conv2", "bn2", 128u, 128u, 3u, &layer2[i].conv2);
            UA_LAYER2_RESOLVE(
                "conv3", "bn3", 512u, 128u, 1u, &layer2[i].conv3);
#undef UA_LAYER2_RESOLVE
        }
        status = resolve_conv_bn(
            c, "img_backbone.layer2.0.downsample.0.weight",
            "img_backbone.layer2.0.downsample.1", 512u, 256u, 1u,
            &layer2[0].downsample);
        if (status != UA_OK) return status;
        layer2[0].has_downsample = 1;
        status = ua_cuda_production_resnet_layer2(
            c->cuda_context, layer1_output, layer2, &layer2_output);
        if (status != UA_OK || !layer2_output) return status;
        memset(layer3, 0, sizeof(layer3));
        for (i = 0; i < 23; ++i) {
            char conv_name[128], bn_name[128], dcn_prefix[128];
            size_t conv1_inputs = i == 0 ? 512u : 1024u;
#define UA_LAYER3_RESOLVE(part, bnpart, outc, inc, kernel_, target) do { \
    (void)snprintf(conv_name, sizeof(conv_name), \
                   "img_backbone.layer3.%u.%s.weight", i, part); \
    (void)snprintf(bn_name, sizeof(bn_name), \
                   "img_backbone.layer3.%u.%s", i, bnpart); \
    status = resolve_conv_bn(c, conv_name, bn_name, outc, inc, kernel_, target); \
    if (status != UA_OK) return status; \
} while (0)
            UA_LAYER3_RESOLVE(
                "conv1", "bn1", 256u, conv1_inputs, 1u, &layer3[i].conv1);
            UA_LAYER3_RESOLVE(
                "conv2", "bn2", 256u, 256u, 3u, &layer3[i].conv2);
            UA_LAYER3_RESOLVE(
                "conv3", "bn3", 1024u, 256u, 1u, &layer3[i].conv3);
#undef UA_LAYER3_RESOLVE
            (void)snprintf(
                dcn_prefix, sizeof(dcn_prefix),
                "img_backbone.layer3.%u.conv2", i);
            status = resolve_dcn_offset(c, dcn_prefix, 256u, &layer3[i]);
            if (status != UA_OK) return status;
        }
        status = resolve_conv_bn(
            c, "img_backbone.layer3.0.downsample.0.weight",
            "img_backbone.layer3.0.downsample.1", 1024u, 512u, 1u,
            &layer3[0].downsample);
        if (status != UA_OK) return status;
        layer3[0].has_downsample = 1;
        status = ua_cuda_production_resnet_layer3(
            c->cuda_context, layer2_output, layer3, &layer3_output);
        if (status != UA_OK || !layer3_output) return status;
        memset(layer4, 0, sizeof(layer4));
        for (i = 0; i < 3; ++i) {
            char conv_name[128], bn_name[128], dcn_prefix[128];
            size_t conv1_inputs = i == 0 ? 1024u : 2048u;
#define UA_LAYER4_RESOLVE(part, bnpart, outc, inc, kernel_, target) do { \
    (void)snprintf(conv_name, sizeof(conv_name), \
                   "img_backbone.layer4.%u.%s.weight", i, part); \
    (void)snprintf(bn_name, sizeof(bn_name), \
                   "img_backbone.layer4.%u.%s", i, bnpart); \
    status = resolve_conv_bn(c, conv_name, bn_name, outc, inc, kernel_, target); \
    if (status != UA_OK) return status; \
} while (0)
            UA_LAYER4_RESOLVE(
                "conv1", "bn1", 512u, conv1_inputs, 1u, &layer4[i].conv1);
            UA_LAYER4_RESOLVE(
                "conv2", "bn2", 512u, 512u, 3u, &layer4[i].conv2);
            UA_LAYER4_RESOLVE(
                "conv3", "bn3", 2048u, 512u, 1u, &layer4[i].conv3);
#undef UA_LAYER4_RESOLVE
            (void)snprintf(
                dcn_prefix, sizeof(dcn_prefix),
                "img_backbone.layer4.%u.conv2", i);
            status = resolve_dcn_offset(c, dcn_prefix, 512u, &layer4[i]);
            if (status != UA_OK) return status;
        }
        status = resolve_conv_bn(
            c, "img_backbone.layer4.0.downsample.0.weight",
            "img_backbone.layer4.0.downsample.1", 2048u, 1024u, 1u,
            &layer4[0].downsample);
        if (status != UA_OK) return status;
        layer4[0].has_downsample = 1;
        status = ua_cuda_production_resnet_layer4(
            c->cuda_context, layer3_output, layer4, &layer4_output);
        if (status != UA_OK || !layer4_output) return status;
        memset(lateral, 0, sizeof(lateral));
        memset(fpn_convs, 0, sizeof(fpn_convs));
        for (i = 0; i < 3; ++i) {
            char prefix[128];
            static const size_t fpn_inputs[3] = {512u, 1024u, 2048u};
            (void)snprintf(
                prefix, sizeof(prefix),
                "img_neck.lateral_convs.%u.conv", i);
            status = resolve_conv_bias(
                c, prefix, 256u, fpn_inputs[i], 1u, &lateral[i]);
            if (status != UA_OK) return status;
        }
        for (i = 0; i < 4; ++i) {
            char prefix[128];
            (void)snprintf(
                prefix, sizeof(prefix), "img_neck.fpn_convs.%u.conv", i);
            status = resolve_conv_bias(
                c, prefix, 256u, 256u, 3u, &fpn_convs[i]);
            if (status != UA_OK) return status;
        }
        status = ua_cuda_production_fpn(
            c->cuda_context, layer2_output, layer3_output, layer4_output,
            lateral, fpn_convs, fpn_outputs);
        if (status != UA_OK) return status;
        for (i = 0; i < 4; ++i)
            if (!fpn_outputs[i]) return UA_ERR_BACKEND;
        {
            const uint64_t camera_dims[2] = {6u, 256u};
            const uint64_t level_dims[2] = {4u, 256u};
            status = production_tensor_pointer(
                c, "pts_bbox_head.transformer.cams_embeds", 2u, 2u,
                camera_dims, &camera_embeds);
            if (status != UA_OK) return status;
            status = production_tensor_pointer(
                c, "pts_bbox_head.transformer.level_embeds", 2u, 2u,
                level_dims, &level_embeds);
            if (status != UA_OK) return status;
        }
        status = ua_cuda_production_bevformer_flatten(
            c->cuda_context, fpn_outputs, camera_embeds, level_embeds,
            &flattened_features);
        if (status != UA_OK || !flattened_features) return status;
        memset(&can_bus_weights, 0, sizeof(can_bus_weights));
        {
            const uint64_t query_dims[2] = {40000u, 256u};
            const uint64_t position_dims[2] = {200u, 128u};
            const uint64_t linear0_weight_dims[2] = {128u, 18u};
            const uint64_t linear0_bias_dims[1] = {128u};
            const uint64_t linear1_weight_dims[2] = {256u, 128u};
            const uint64_t linear1_bias_dims[1] = {256u};
            status = production_tensor_pointer(
                c, "pts_bbox_head.bev_embedding.weight", 1u, 2u,
                query_dims, &bev_query_weight);
            if (status != UA_OK) return status;
            status = production_tensor_pointer(
                c, "pts_bbox_head.positional_encoding.col_embed.weight",
                1u, 2u, position_dims, &col_embed);
            if (status != UA_OK) return status;
            status = production_tensor_pointer(
                c, "pts_bbox_head.positional_encoding.row_embed.weight",
                1u, 2u, position_dims, &row_embed);
            if (status != UA_OK) return status;
#define UA_CAN_TENSOR(name_, rank_, dims_, field_) do { \
    status = production_tensor_pointer( \
        c, name_, 2u, rank_, dims_, &can_bus_weights.field_); \
    if (status != UA_OK) return status; \
} while (0)
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.0.weight", 2u,
                linear0_weight_dims, linear0_weight_fp32);
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.0.bias", 1u,
                linear0_bias_dims, linear0_bias_fp32);
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.2.weight", 2u,
                linear1_weight_dims, linear1_weight_fp32);
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.2.bias", 1u,
                linear1_bias_dims, linear1_bias_fp32);
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.norm.weight", 1u,
                linear1_bias_dims, norm_weight_fp32);
            UA_CAN_TENSOR(
                "pts_bbox_head.transformer.can_bus_mlp.norm.bias", 1u,
                linear1_bias_dims, norm_bias_fp32);
#undef UA_CAN_TENSOR
        }
        status = ua_cuda_production_prepare_bev(
            c->cuda_context, bev_query_weight, col_embed, row_embed,
            &can_bus_weights, &bev_queries, &bev_pos);
        if (status != UA_OK || !bev_queries || !bev_pos) return status;
        status = ua_cuda_production_prepare_bev_geometry(
            c->cuda_context, &reference_2d, &reference_3d,
            &reference_camera, &visibility, &visible_indices,
            &visible_counts);
        if (status != UA_OK || !reference_2d || !reference_3d ||
            !reference_camera || !visibility || !visible_indices ||
            !visible_counts)
            return status;
        encoder_query = bev_queries;
        for (i = 0; i < 6u; ++i) {
            status = resolve_encoder_layer(c, i, &encoder_layer);
            if (status != UA_OK) return status;
            status = ua_cuda_production_encoder_temporal(
                c->cuda_context, i, encoder_query, &encoder_layer.temporal,
                &temporal_output);
            if (status != UA_OK || !temporal_output) return status;
            status = ua_cuda_production_encoder_norm_after_temporal(
                c->cuda_context, i, encoder_layer.norm0_weight_fp16,
                encoder_layer.norm0_bias_fp16, &norm0_output);
            if (status != UA_OK || !norm0_output) return status;
            status = ua_cuda_production_encoder_spatial(
                c->cuda_context, i, &encoder_layer.spatial,
                &spatial_output);
            if (status != UA_OK || !spatial_output) return status;
            status = ua_cuda_production_encoder_norm_after_spatial(
                c->cuda_context, i, encoder_layer.norm1_weight_fp16,
                encoder_layer.norm1_bias_fp16, &norm1_output);
            if (status != UA_OK || !norm1_output) return status;
            status = ua_cuda_production_encoder_ffn(
                c->cuda_context, i, &encoder_layer.ffn, &ffn_output,
                &norm2_output);
            if (status != UA_OK || !ffn_output || !norm2_output)
                return status;
            encoder_query = norm2_output;
        }
        {
            const uint64_t query_dims[2] = {901u, 512u};
            const uint64_t reference_weight_dims[2] = {3u, 256u};
            const uint64_t reference_bias_dims[1] = {3u};
            status = production_tensor_pointer(
                c, "query_embedding.weight", 1u, 2u, query_dims,
                &track_query_embedding);
            if (status != UA_OK) return status;
            status = production_tensor_pointer(
                c, "reference_points.weight", 2u, 2u,
                reference_weight_dims, &track_reference_weight);
            if (status != UA_OK) return status;
            status = production_tensor_pointer(
                c, "reference_points.bias", 2u, 1u,
                reference_bias_dims, &track_reference_bias);
            if (status != UA_OK) return status;
        }
        status = ua_cuda_production_prepare_track_queries(
            c->cuda_context, track_query_embedding, track_reference_weight,
            track_reference_bias, &track_query_pos, &track_query,
            &track_reference_points);
        if (status != UA_OK || !track_query_pos || !track_query ||
            !track_reference_points)
            return status;
        for (i = 0; i < 6u; ++i) {
            status = resolve_track_decoder_layer(
                c, i, &track_decoder_layer);
            if (status != UA_OK) return status;
            status = ua_cuda_production_track_decoder_layer(
                c->cuda_context, i, norm2_output, &track_decoder_layer,
                &track_decoder_output);
            if (status != UA_OK || !track_decoder_output) return status;
            status = resolve_track_regression_branch(
                c, i, &track_regression_weights);
            if (status != UA_OK) return status;
            status = ua_cuda_production_track_refine_references(
                c->cuda_context, i, track_decoder_output,
                &track_regression_weights, &track_regression,
                &track_reference_points);
            if (status != UA_OK || !track_regression ||
                !track_reference_points)
                return status;
        }
        status = resolve_track_output_heads(
            c, 5u, &track_classification_weights, &track_past_weights);
        if (status != UA_OK) return status;
        status = ua_cuda_production_track_output_heads(
            c->cuda_context, &track_classification_weights,
            &track_past_weights, &track_class_logits, &track_boxes,
            &track_past_trajectory);
        if (status != UA_OK || !track_class_logits || !track_boxes ||
            !track_past_trajectory)
            return status;
        status = ua_cuda_production_track_score_filter(
            c->cuda_context, &track_scores, &track_classes,
            &track_selected_indices, &track_selected_count);
        if (status != UA_OK || !track_scores || !track_classes ||
            !track_selected_indices || !track_selected_count)
            return status;
        status = resolve_query_interaction(c, &query_interaction_weights);
        if (status != UA_OK) return status;
        status = ua_cuda_production_query_interaction(
            c->cuda_context, track_query_pos, track_query,
            track_decoder_output, 0u, &query_interaction_weights,
            &updated_track_query_feat);
        if (status != UA_OK || !updated_track_query_feat) return status;
    }
#endif
    /*
     * UAW2 parsing, validation and persistent CUDA residency are executable.
     * The official 2459-tensor operator graph is intentionally not dispatched
     * until its saved oracle fixtures pass; never substitute the tiny graph.
     */
    return UA_ERR_UNSUPPORTED_PROFILE;
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
