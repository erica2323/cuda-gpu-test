#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32 

__global__ void matrixMulSharedMemory(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < N && i * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && i * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            value += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    int N = 10240;
    size_t size = N * N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;  // Example initialization
        h_B[i] = 1.0f;  // Example initialization
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the optimized kernel
    matrixMulSharedMemory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time for the kernel: %f ms\n", milliseconds);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
