#include "uniad.h"
#include <cuda_runtime.h>

__global__ static void boundary_kernel(const float *x, size_t n, float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = 0.0f;
        for (size_t i = 0; i < n; ++i) v += x[i];
        *out = v;
    }
}

extern "C" ua_status ua_cuda_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? UA_OK : UA_ERR_BACKEND;
}

extern "C" ua_status ua_cuda_demo(const float *input, size_t n, float *output) {
    float *dx = NULL, *dy = NULL;
    if (cudaMalloc((void **)&dx, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&dy, sizeof(float)) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_MEMORY;
    }
    if (cudaMemcpy(dx, input, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_BACKEND;
    }
    boundary_kernel<<<1, 1>>>(dx, n, dy);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, dy, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy); return UA_ERR_BACKEND;
    }
    cudaFree(dx); cudaFree(dy); return UA_OK;
}
