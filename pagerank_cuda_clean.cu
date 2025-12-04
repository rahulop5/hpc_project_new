#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

const int NUM_NODES = 10000;
const int ITERATIONS = 20;
const float DAMPING = 0.85f;
const int BLOCK_SIZE = 256;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct CSRGraph {
    vector<int> row_offsets;
    vector<int> col_indices;
    vector<int> out_degrees;
};

void generateRandomGraph(int num_nodes, CSRGraph &graph) {
    vector<vector<int>> adj_in(num_nodes);
    graph.out_degrees.assign(num_nodes, 0);

    srand(time(NULL));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            if ((rand() % 100) < 10) {
                adj_in[j].push_back(i);
                graph.out_degrees[i]++;
            }
        }
    }

    graph.row_offsets.push_back(0);
    for (const auto &sources : adj_in) {
        for (int src : sources) {
            graph.col_indices.push_back(src);
        }
        graph.row_offsets.push_back(graph.col_indices.size());
    }
}

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

int main() {
    cout << "Parallelizing PageRank using CUDA..." << endl;

    CSRGraph h_graph;
    generateRandomGraph(NUM_NODES, h_graph);

    int num_edges = h_graph.col_indices.size();
    cout << "Nodes: " << NUM_NODES << ", Edges: " << num_edges << endl;

    vector<float> h_opg(NUM_NODES);
    vector<float> h_npg(NUM_NODES);

    float init_val = 1.0f / NUM_NODES;
    for (int i = 0; i < NUM_NODES; ++i) h_opg[i] = init_val;

    float *d_opg, *d_npg, *d_dangling_sum;
    int *d_row_offsets, *d_col_indices, *d_out_degrees;

    cudaCheckError(cudaMalloc(&d_opg, NUM_NODES * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_npg, NUM_NODES * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_dangling_sum, sizeof(float)));

    cudaCheckError(cudaMalloc(&d_row_offsets, (NUM_NODES + 1) * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_out_degrees, NUM_NODES * sizeof(int)));

    cudaCheckError(cudaMemcpy(d_opg, h_opg.data(), NUM_NODES * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_row_offsets, h_graph.row_offsets.data(), (NUM_NODES + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_col_indices, h_graph.col_indices.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out_degrees, h_graph.out_degrees.data(), NUM_NODES * sizeof(int), cudaMemcpyHostToDevice));

    int gridSize = (NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

        updatePageRankKernel<<<gridSize, BLOCK_SIZE>>>(NUM_NODES, d_npg, d_opg, d_row_offsets,
                                                       d_col_indices, d_out_degrees,
                                                       h_dangling_sum, DAMPING);
        cudaCheckError(cudaGetLastError());

        float* temp = d_opg;
        d_opg = d_npg;
        d_npg = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaCheckError(cudaMemcpy(h_opg.data(), d_opg, NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "\nExecution Time: " << milliseconds << " ms" << endl;
    cout << "Top 5 Nodes by PageRank:" << endl;
    for (int i = 0; i < 5; ++i) {
        cout << "Node " << i << ": " << h_opg[i] << endl;
    }

    cudaFree(d_opg);
    cudaFree(d_npg);
    cudaFree(d_dangling_sum);
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_out_degrees);

    return 0;
}
