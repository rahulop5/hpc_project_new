/*
 * CUDA Implementation of PageRank Algorithm
 * Reference: "Performance Analysis of Parallelized PageRank Algorithm using OpenMP, MPI, and CUDA"
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>

// CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Configuration Constants
const int NUM_NODES = 1024;    
const int ITERATIONS = 20;     
const float DAMPING = 0.85f;   
const int BLOCK_SIZE = 256;    

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// -------------------------------------------------------------------------
// HOST HELPER: Graph Structure
// -------------------------------------------------------------------------
struct CSRGraph {
    std::vector<int> row_offsets; 
    std::vector<int> col_indices; 
    std::vector<int> out_degrees; 
};

// Generate Graph on Host
void generateRandomGraph(int num_nodes, CSRGraph &graph) {
    std::vector<std::vector<int>> adj_in(num_nodes);
    graph.out_degrees.assign(num_nodes, 0);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue; 
            // 10% probability of link i -> j
            if ((rand() % 100) < 10) { 
                adj_in[j].push_back(i);
                graph.out_degrees[i]++;
            }
        }
    }

    graph.row_offsets.push_back(0);
    for (int i = 0; i < num_nodes; ++i) {
        for (size_t k = 0; k < adj_in[i].size(); ++k) {
            graph.col_indices.push_back(adj_in[i][k]);
        }
        graph.row_offsets.push_back((int)graph.col_indices.size());
    }
}

// -------------------------------------------------------------------------
// DEVICE KERNELS
// -------------------------------------------------------------------------

// Kernel 1: Calculate Dangling Node Contribution
__global__ void calculateDanglingSum(const float* d_opg, const int* d_out_degrees, 
                                     float* d_dangling_sum, int num_nodes, float damping) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        if (d_out_degrees[tid] == 0) {
            float contribution = damping * (d_opg[tid] / (float)num_nodes);
            atomicAdd(d_dangling_sum, contribution);
        }
    }
}

// Kernel 2: Update PageRank
__global__ void updatePageRankKernel(int num_nodes, float* d_npg, const float* d_opg, 
                                     const int* d_row_offsets, const int* d_col_indices, 
                                     const int* d_out_degrees, float dangling_val, 
                                     float damping) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        float val = dangling_val + (1.0f - damping) / (float)num_nodes;

        int start = d_row_offsets[tid];
        int end = d_row_offsets[tid + 1];

        for (int i = start; i < end; ++i) {
            int source = d_col_indices[i];
            
            if (d_out_degrees[source] > 0) {
                val += damping * d_opg[source] / (float)d_out_degrees[source];
            }
        }
        d_npg[tid] = val;
    }
}

// -------------------------------------------------------------------------
// MAIN FUNCTION
// -------------------------------------------------------------------------
int main() {
    printf("Initializing PageRank (CUDA)...\n");

    // 1. Host Setup
    CSRGraph h_graph;
    generateRandomGraph(NUM_NODES, h_graph);
    int num_edges = (int)h_graph.col_indices.size();
    
    std::vector<float> h_opg(NUM_NODES); 
    float init_val = 1.0f / NUM_NODES;
    for (int i = 0; i < NUM_NODES; ++i) h_opg[i] = init_val;

    // 2. Device Allocation
    float *d_opg, *d_npg, *d_dangling_sum;
    int *d_row_offsets, *d_col_indices, *d_out_degrees;

    cudaCheckError(cudaMalloc((void**)&d_opg, NUM_NODES * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_npg, NUM_NODES * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_dangling_sum, sizeof(float)));
    
    cudaCheckError(cudaMalloc((void**)&d_row_offsets, (NUM_NODES + 1) * sizeof(int)));
    cudaCheckError(cudaMalloc((void**)&d_col_indices, num_edges * sizeof(int)));
    cudaCheckError(cudaMalloc((void**)&d_out_degrees, NUM_NODES * sizeof(int)));

    // 3. Data Copy
    cudaCheckError(cudaMemcpy(d_opg, h_opg.data(), NUM_NODES * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_row_offsets, h_graph.row_offsets.data(), (NUM_NODES + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_col_indices, h_graph.col_indices.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out_degrees, h_graph.out_degrees.data(), NUM_NODES * sizeof(int), cudaMemcpyHostToDevice));

    int gridSize = (NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 4. Execution Loop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        cudaCheckError(cudaMemset(d_dangling_sum, 0, sizeof(float)));
        
        calculateDanglingSum<<<gridSize, BLOCK_SIZE>>>(d_opg, d_out_degrees, d_dangling_sum, NUM_NODES, DAMPING);
        cudaCheckError(cudaGetLastError());

        float h_dangling_sum;
        cudaCheckError(cudaMemcpy(&h_dangling_sum, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost));

        updatePageRankKernel<<<gridSize, BLOCK_SIZE>>>(NUM_NODES, d_npg, d_opg, d_row_offsets, d_col_indices, 
                                                      d_out_degrees, h_dangling_sum, DAMPING);
        cudaCheckError(cudaGetLastError());

        // Pointer swap
        float* temp = d_opg;
        d_opg = d_npg;
        d_npg = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Results
    cudaCheckError(cudaMemcpy(h_opg.data(), d_opg, NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nExecution Time: %.2f ms\n", milliseconds);
    printf("Top 5 Node Ranks:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Node %d: %f\n", i, h_opg[i]);
    }

    // Cleanup
    cudaFree(d_opg); cudaFree(d_npg); cudaFree(d_dangling_sum);
    cudaFree(d_row_offsets); cudaFree(d_col_indices); cudaFree(d_out_degrees);

    return 0;
}