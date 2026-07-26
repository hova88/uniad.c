#ifndef UA_SHA256_H
#define UA_SHA256_H
#include <stddef.h>
#include <stdint.h>
void ua_sha256(const void *data, size_t size, uint8_t digest[32]);
#endif
