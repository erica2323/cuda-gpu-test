#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

const int N = 10240;

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t err) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    cublasHandle_t handle;

    // Allocate memory on the device
    checkCudaError(cudaMalloc(&d_A, N * N * sizeof(float)));
    checkCudaError(cudaMalloc(&d_B, N * N * sizeof(float)));
    checkCudaError(cudaMalloc(&d_C, N * N * sizeof(float)));

    // Create cuBLAS handle
    checkCublasError(cublasCreate(&handle));

    // Initialize matrices (fill with some values, here we skip actual initialization for brevity)

    float alpha = 1.0f;
    float beta = 0.0f;

    // Timing cuBLAS GEMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "cuBLAS GEMM took " << milliseconds << " ms" << std::endl;

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
