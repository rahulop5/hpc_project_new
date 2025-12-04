/*
 * Sequential Implementation of PageRank Algorithm (Detailed Profiling)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime> 
#include <chrono> 

// Configuration Constants
const int NUM_NODES = 5000;
const int ITERATIONS = 20;
const float DAMPING = 0.85f;

// Profiling Globals
double time_graph_gen = 0.0;
double time_init = 0.0;
double time_dangling = 0.0;
double time_update = 0.0;
double time_overhead = 0.0; // Includes pointer swap and loop overhead

struct Graph {
    std::vector<std::vector<int>> incoming_links; 
    std::vector<int> out_degrees;                 
};

void generateRandomGraph(int num_nodes, Graph &graph) {
    auto t1 = std::chrono::high_resolution_clock::now();

    graph.incoming_links.resize(num_nodes);
    graph.out_degrees.assign(num_nodes, 0);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            if ((rand() % 100) < 10) {
                graph.incoming_links[j].push_back(i);
                graph.out_degrees[i]++;
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_graph_gen = std::chrono::duration<double, std::milli>(t2 - t1).count();
}

__attribute__((noinline))
void initializeRanks(int num_nodes, std::vector<float> &ranks) {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    float init_val = 1.0f / num_nodes;
    for (int i = 0; i < num_nodes; ++i) {
        ranks[i] = init_val;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_init += std::chrono::duration<double, std::milli>(t2 - t1).count();
}

__attribute__((noinline))
float calculateDanglingSum(int num_nodes, const Graph &graph, const std::vector<float> &ranks) {
    auto t1 = std::chrono::high_resolution_clock::now();

    float dangling_sum = 0.0f;
    for (int i = 0; i < num_nodes; ++i) {
        if (graph.out_degrees[i] == 0) {
            dangling_sum += DAMPING * (ranks[i] / num_nodes); 
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_dangling += std::chrono::duration<double, std::milli>(t2 - t1).count();
    return dangling_sum;
}

__attribute__((noinline))
float calculateNodeRank(int p, int num_nodes, float dangling_sum, const Graph &graph, const std::vector<float> &ranks) {
    float val = dangling_sum + (1.0f - DAMPING) / num_nodes;
    for (int source : graph.incoming_links[p]) {
        if (graph.out_degrees[source] > 0) {
            val += DAMPING * ranks[source] / graph.out_degrees[source]; 
        }
    }
    return val;
}

__attribute__((noinline))
void updateRanks(int num_nodes, const Graph &graph, const std::vector<float> &ranks, std::vector<float> &new_ranks, float dangling_sum) {
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int p = 0; p < num_nodes; ++p) {
        new_ranks[p] = calculateNodeRank(p, num_nodes, dangling_sum, graph, ranks);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_update += std::chrono::duration<double, std::milli>(t2 - t1).count();
}

void pageRankSequential(int num_nodes, const Graph &graph, std::vector<float> &ranks) {
    std::vector<float> new_ranks(num_nodes);

    initializeRanks(num_nodes, ranks);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float dangling_sum = calculateDanglingSum(num_nodes, graph, ranks);
        updateRanks(num_nodes, graph, ranks, new_ranks, dangling_sum);

        // Measure Overhead (Pointer Swap + Loop mechanics)
        auto t1 = std::chrono::high_resolution_clock::now();
        ranks = new_ranks;
        auto t2 = std::chrono::high_resolution_clock::now();
        time_overhead += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
}

int main() {
    // 1. Setup Graph
    Graph graph;
    generateRandomGraph(NUM_NODES, graph);
    
    // 2. Run and Time the Algorithm
    std::vector<float> ranks(NUM_NODES);

    auto start_time = std::chrono::high_resolution_clock::now();
    pageRankSequential(NUM_NODES, graph, ranks);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;
    
    // Output Detailed Profiling
    std::cout << "Graph Gen:      " << time_graph_gen << " ms" << std::endl;
    std::cout << "Initialization: " << time_init << " ms" << std::endl;
    std::cout << "Dangling Sum:   " << time_dangling << " ms" << std::endl;
    std::cout << "Update Ranks:   " << time_update << " ms" << std::endl;
    std::cout << "Overhead:       " << time_overhead << " ms" << std::endl;

    return 0;
}
