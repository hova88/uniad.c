#define _POSIX_C_SOURCE 200809L
#include "uniad.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

extern ua_status ua_write_demo_assets(const char *directory);

static void usage(FILE *f) {
    fprintf(f,
        "usage: uniad <command> [options]\n"
        "commands:\n"
        "  doctor\n"
        "  generate-demo [directory]\n"
        "  inspect-model <model.uaw>\n"
        "  demo [--dir directory] [--backend cpu|cuda]\n"
        "  infer --profile production | --model M --frame F [--backend cpu|cuda]\n"
        "  benchmark [--dir directory] [--warmup N] [--runs N] [--backend cpu|cuda]\n");
}

static int make_dir(const char *p) {
    return mkdir(p, 0775) == 0 || errno == EEXIST;
}

static int fail(const char *where, ua_status s) {
    fprintf(stderr, "uniad: %s: %s\n", where, ua_status_string(s));
    return 1;
}

static ua_backend parse_backend(const char *s, int *ok) {
    if (!strcmp(s, "cpu")) return UA_BACKEND_CPU;
    if (!strcmp(s, "cuda")) return UA_BACKEND_CUDA;
    *ok = 0; return UA_BACKEND_CPU;
}

static int load_and_run(const char *model_path, const char *frame_path,
                        ua_backend backend, ua_context **reuse, int print_json) {
    ua_model *m = NULL; ua_frame *f = NULL; ua_context *c = reuse ? *reuse : NULL;
    ua_result r; ua_status s; char json[65536]; size_t n;
    if (!c) {
        s = ua_model_load(model_path, &m); if (s != UA_OK) return fail("load model", s);
        s = ua_context_create(m, backend, &c);
        if (s != UA_OK) { ua_model_destroy(m); return fail("create context", s); }
    }
    s = ua_frame_load(frame_path, &f);
    if (s == UA_OK) s = ua_infer(c, f, &r);
    if (s == UA_OK && print_json) {
        s = ua_result_json(&r, json, sizeof(json), &n);
        if (s == UA_OK && fwrite(json, 1, n, stdout) != n) s = UA_ERR_IO;
    }
    ua_frame_destroy(f);
    if (reuse) *reuse = c;
    else { ua_context_destroy(c); ua_model_destroy(m); }
    return s == UA_OK ? 0 : fail("infer", s);
}

static int command_demo(int argc, char **argv) {
    const char *dir = "build/demo", *backend_name = "cpu";
    char model[512], frame[512]; ua_model *m = NULL; ua_context *c = NULL;
    ua_frame *f = NULL; ua_result r; ua_status s; char json[65536]; size_t n;
    ua_backend backend; int ok = 1, i;
    for (i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend_name = argv[++i];
        else { usage(stderr); return 2; }
    }
    backend = parse_backend(backend_name, &ok); if (!ok) return 2;
    if (!make_dir("build") || !make_dir(dir)) { perror("mkdir"); return 1; }
    snprintf(model, sizeof(model), "%s/demo.uaw", dir);
    if (ua_model_load(model, &m) != UA_OK) {
        s = ua_write_demo_assets(dir); if (s != UA_OK) return fail("generate demo", s);
    }
    s = ua_model_load(model, &m); if (s != UA_OK) return fail("load model", s);
    s = ua_context_create(m, backend, &c);
    if (s != UA_OK) { ua_model_destroy(m); return fail("create context", s); }
    for (i = 0; i < 2; ++i) {
        snprintf(frame, sizeof(frame), "%s/frame%d.uaf", dir, i);
        s = ua_frame_load(frame, &f);
        if (s == UA_OK) s = ua_infer(c, f, &r);
        ua_frame_destroy(f); f = NULL;
        if (s != UA_OK) { ua_context_destroy(c); ua_model_destroy(m); return fail("demo", s); }
    }
    s = ua_result_json(&r, json, sizeof(json), &n);
    if (s == UA_OK) fwrite(json, 1, n, stdout);
    ua_context_destroy(c); ua_model_destroy(m);
    return s == UA_OK ? 0 : fail("serialize", s);
}

static int command_benchmark(int argc, char **argv) {
    const char *dir = "build/demo", *backend_name = "cpu";
    int warmup = 2, runs = 10, i, ok = 1; ua_backend backend;
    char model_path[512], frame_path[512]; ua_model *m = NULL; ua_frame *f = NULL;
    ua_context *c = NULL; ua_result r; ua_metrics mt; ua_status s;
    double sum = 0.0, cold = 0.0;
    for (i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend_name = argv[++i];
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        else return 2;
    }
    if (warmup < 0 || runs < 1) return 2;
    backend = parse_backend(backend_name, &ok); if (!ok) return 2;
    if (!make_dir("build") || !make_dir(dir)) return 1;
    snprintf(model_path, sizeof(model_path), "%s/demo.uaw", dir);
    snprintf(frame_path, sizeof(frame_path), "%s/frame1.uaf", dir);
    if (ua_model_load(model_path, &m) != UA_OK) {
        s = ua_write_demo_assets(dir); if (s != UA_OK) return fail("generate demo", s);
    }
    s = ua_model_load(model_path, &m);
    if (s == UA_OK) s = ua_frame_load(frame_path, &f);
    if (s == UA_OK) s = ua_context_create(m, backend, &c);
    if (s != UA_OK) { ua_frame_destroy(f); ua_model_destroy(m); return fail("benchmark setup", s); }
    s = ua_infer(c, f, &r); ua_context_metrics(c, &mt); cold = mt.total_ms;
    for (i = 0; s == UA_OK && i < warmup; ++i) s = ua_infer(c, f, &r);
    for (i = 0; s == UA_OK && i < runs; ++i) {
        s = ua_infer(c, f, &r); ua_context_metrics(c, &mt); sum += mt.total_ms;
    }
    if (s == UA_OK)
        printf("{\"schema\":\"uniad.c/benchmark-v1\",\"evidence\":\"synthetic\","
               "\"backend\":\"%s\",\"runs\":%d,\"cold_ms\":%.6f,\"warm_mean_ms\":%.6f,"
               "\"stages_ms\":{\"camera\":%.6f,\"bev\":%.6f,\"temporal\":%.6f,"
               "\"track\":%.6f,\"map\":%.6f,\"motion\":%.6f,\"occupancy\":%.6f,"
               "\"planning\":%.6f},\"owned_host_bytes\":%zu,\"owned_device_bytes\":%zu,"
               "\"h2d_bytes\":%zu,\"d2h_bytes\":%zu}\n",
               backend_name, runs, cold, sum / runs, mt.camera_ms, mt.bev_ms,
               mt.temporal_ms, mt.track_ms, mt.map_ms, mt.motion_ms,
               mt.occupancy_ms, mt.planning_ms, mt.owned_host_bytes,
               mt.owned_device_bytes, mt.h2d_bytes, mt.d2h_bytes);
    ua_context_destroy(c); ua_frame_destroy(f); ua_model_destroy(m);
    return s == UA_OK ? 0 : fail("benchmark", s);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(stderr); return 2; }
    if (!strcmp(argv[1], "doctor")) {
        printf("{\"api_version\":%u,\"cpu\":true,\"cuda_compiled\":%s,"
               "\"production_execution\":false,\"host_memory_gate_bytes\":268435456,"
               "\"device_memory_gate_bytes\":268435456}\n",
               UA_API_VERSION,
#ifdef UA_WITH_CUDA
               "true"
#else
               "false"
#endif
        );
        return 0;
    }
    if (!strcmp(argv[1], "generate-demo")) {
        const char *dir = argc > 2 ? argv[2] : "build/demo";
        ua_status s;
        if (!make_dir("build") || !make_dir(dir)) { perror("mkdir"); return 1; }
        s = ua_write_demo_assets(dir);
        if (s != UA_OK) return fail("generate demo", s);
        printf("generated %s/demo.uaw and two UAF frames\n", dir); return 0;
    }
    if (!strcmp(argv[1], "inspect-model")) {
        ua_model *m = NULL; ua_status s;
        if (argc != 3) return 2;
        s = ua_model_load(argv[2], &m); if (s != UA_OK) return fail("inspect", s);
        printf("{\"container\":\"UAW\",\"version\":1,\"profile\":\"%s\","
               "\"seed\":%llu,\"tensors\":[{\"name\":\"demo.weights\","
               "\"dtype\":\"f32\",\"shape\":[64],\"layout\":\"flat\"}]}\n",
               ua_model_profile(m), (unsigned long long)ua_model_seed(m));
        ua_model_destroy(m); return 0;
    }
    if (!strcmp(argv[1], "demo")) return command_demo(argc, argv);
    if (!strcmp(argv[1], "benchmark")) return command_benchmark(argc, argv);
    if (!strcmp(argv[1], "infer")) {
        const char *model = NULL, *frame = NULL, *backend_name = "cpu"; int i, ok = 1;
        for (i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--profile") && i + 1 < argc &&
                !strcmp(argv[++i], "production"))
                return fail("infer", UA_ERR_UNSUPPORTED_PROFILE);
            if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
            else if (!strcmp(argv[i], "--frame") && i + 1 < argc) frame = argv[++i];
            else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend_name = argv[++i];
        }
        if (!model || !frame) return 2;
        return load_and_run(model, frame, parse_backend(backend_name, &ok), NULL, 1);
    }
    usage(stderr); return 2;
}
