#define _POSIX_C_SOURCE 200809L
#include "uniad.h"
#include "operators.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern ua_status ua_write_demo_assets(const char *directory);

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
    assert(mkdtemp(tmp));
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
    assert(ua_context_create(m, UA_BACKEND_CUDA, &(ua_context *){0}) == UA_ERR_BACKEND);
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
