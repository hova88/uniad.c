#ifndef UNIAD_H
#define UNIAD_H

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define UA_API_VERSION 2u
#define UA_CAMERA_COUNT 6u
#define UA_MAX_TRACKS 8u
#define UA_MAX_MAP 8u
#define UA_MOTION_MODES 3u
#define UA_PRED_STEPS 4u
#define UA_PLAN_STEPS 6u
#define UA_OCC_HORIZONS 3u
#define UA_PROD_MAX_TRACKS 300u
#define UA_PROD_MAX_MAP_VECTORS 300u
#define UA_PROD_MAP_POINTS 20u
#define UA_PROD_MOTION_MODES 6u
#define UA_PROD_PRED_STEPS 12u
#define UA_PROD_OCC_HORIZONS 5u
#define UA_PROD_OCC_HEIGHT 200u
#define UA_PROD_OCC_WIDTH 200u
#define UA_PRODUCTION_METADATA_BYTES 760u

typedef enum {
    UA_OK = 0, UA_ERR_ARGUMENT, UA_ERR_IO, UA_ERR_FORMAT, UA_ERR_CHECKSUM,
    UA_ERR_CAPACITY, UA_ERR_NONFINITE, UA_ERR_PROFILE, UA_ERR_BACKEND,
    UA_ERR_UNSUPPORTED_PROFILE, UA_ERR_MEMORY
} ua_status;

typedef enum { UA_BACKEND_CPU = 0, UA_BACKEND_CUDA = 1 } ua_backend;

typedef struct ua_model ua_model;
typedef struct ua_frame ua_frame;
typedef struct ua_context ua_context;

typedef struct { float x, y, score; int32_t id; } ua_track;
typedef struct { float x0, y0, x1, y1, score; } ua_map_element;
typedef struct { float x, y; } ua_point;
typedef struct {
    uint32_t version;
    char scene[32];
    uint64_t frame_index;
    uint32_t track_count, map_count;
    ua_track tracks[UA_MAX_TRACKS];
    ua_map_element map[UA_MAX_MAP];
    ua_point motion[UA_MAX_TRACKS][UA_MOTION_MODES][UA_PRED_STEPS];
    float motion_score[UA_MAX_TRACKS][UA_MOTION_MODES];
    uint8_t occupancy[UA_OCC_HORIZONS][8][8];
    ua_point ego_plan[UA_PLAN_STEPS];
    float collision_score;
    char command[16];
} ua_result;

typedef struct {
    double camera_ms, bev_ms, temporal_ms, track_ms, map_ms, motion_ms;
    double occupancy_ms, planning_ms, total_ms;
    size_t owned_host_bytes, owned_device_bytes, h2d_bytes, d2h_bytes;
} ua_metrics;

typedef struct {
    uint32_t dtype, rank;
    uint64_t dims[8], byte_offset, nbytes;
} ua_tensor_info;

typedef struct {
    const uint8_t *data;
    uint32_t width, height;
    size_t row_stride_bytes;
} ua_bgr_image_view;

typedef struct {
    uint32_t version;
    ua_bgr_image_view cameras[UA_CAMERA_COUNT];
    float camera_intrinsics[UA_CAMERA_COUNT][9];
    float camera_to_ego[UA_CAMERA_COUNT][16];
    float ego_pose[16];
    float can_bus[18];
    uint64_t timestamp_us;
    const char *scene_token;
    int32_t navigation_command;
} ua_production_input;

typedef struct {
    float center[3], size[3], yaw, velocity[2], score;
    int32_t id, label;
} ua_production_track;

typedef struct {
    uint32_t point_count;
    int32_t label;
    float score;
    ua_point points[UA_PROD_MAP_POINTS];
} ua_production_map_vector;

typedef struct {
    uint32_t version, track_count, map_vector_count;
    ua_production_track tracks[UA_PROD_MAX_TRACKS];
    ua_production_map_vector map_vectors[UA_PROD_MAX_MAP_VECTORS];
    ua_point motion[UA_PROD_MAX_TRACKS][UA_PROD_MOTION_MODES][UA_PROD_PRED_STEPS];
    float motion_score[UA_PROD_MAX_TRACKS][UA_PROD_MOTION_MODES];
    uint8_t occupancy[UA_PROD_OCC_HORIZONS][UA_PROD_OCC_HEIGHT][UA_PROD_OCC_WIDTH];
    ua_point ego_plan[UA_PLAN_STEPS];
} ua_production_result;

const char *ua_status_string(ua_status status);
ua_status ua_model_load(const char *path, ua_model **out);
void ua_model_destroy(ua_model *model);
ua_status ua_frame_load(const char *path, ua_frame **out);
void ua_frame_destroy(ua_frame *frame);
ua_status ua_context_create(const ua_model *model, ua_backend backend, ua_context **out);
void ua_context_reset(ua_context *context);
void ua_context_destroy(ua_context *context);
ua_status ua_infer(ua_context *context, const ua_frame *frame, ua_result *result);
ua_status ua_infer_production(ua_context *context,
                              const ua_production_input *input,
                              ua_production_result *result);
ua_status ua_result_json(const ua_result *result, char *dst, size_t capacity,
                         size_t *written);
void ua_context_metrics(const ua_context *context, ua_metrics *metrics);
ua_status ua_context_debug_tensor_f32(
    const ua_context *context, const char *name, float *values,
    size_t capacity, size_t *written);
ua_status ua_context_debug_run_query_interaction(
    ua_context *context, size_t active_queries);
const char *ua_model_profile(const ua_model *model);
uint64_t ua_model_seed(const ua_model *model);
uint32_t ua_model_tensor_count(const ua_model *model);
ua_status ua_model_find_tensor(const ua_model *model, const char *name,
                               ua_tensor_info *info);

#ifdef __cplusplus
}
#endif
#endif
