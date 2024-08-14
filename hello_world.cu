#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to print "Hello, World!" from the GPU
__global__ void helloWorldKernel() {
    printf("Hello, World from GPU!\n");
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    helloWorldKernel<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed");

    return 0;
}




