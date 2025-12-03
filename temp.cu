#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* A, const int* B, int* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 16;
    int size = n * sizeof(int);

    int h_A[16], h_B[16], h_C[16];

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threads = 8;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Result: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
