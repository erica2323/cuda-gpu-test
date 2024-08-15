#include <stdio.h>

// Kernel function for GEMM: D = A * B + C
__global__ void gemm(float *A, float *B, float *C, float *D, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        D[row * N + col] = value + C[row * N + col];
    }
}

int main() {
    int N = 10240;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_D = (float*)malloc(size);

    // Initialize matrices A, B, and C
    for(int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 1.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_D, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(32, 32);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Launch the GEMM kernel
    gemm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D, N);

    // Copy the result matrix D back to the host
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // Verify the result
    printf("Result: %f\n", h_D[0]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}
