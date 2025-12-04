/*
 * Sequential Implementation of PageRank Algorithm
 * Reference: "Performance Analysis of Parallelized PageRank Algorithm using OpenMP, MPI, and CUDA"
 * Logic follows the pseudocode in Figure 2 [cite: 174-194]
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime> 
#include <chrono> // For precise timing

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
void moncontrol(int) {} // Dummy for MSVC IntelliSense
#else
#define NOINLINE __attribute__((noinline))
// Declare gprof control function (linked via -pg)
extern "C" void moncontrol(int);
#endif

// Configuration Constants
const int NUM_NODES = 10000;
const int ITERATIONS = 20;
const float DAMPING = 0.85f;

// Profiling Globals
double time_init = 0.0;
double time_dangling = 0.0;
double time_update = 0.0;

// Structure to hold the Graph
// We use Adjacency List for the sequential version as it is straightforward
struct Graph {
    std::vector<std::vector<int>> incoming_links; // ih[p] in pseudocode [cite: 177]
    std::vector<int> out_degrees;                 // L(N) in Eq 1 [cite: 50]
};

// Function to generate the same random graph as the CUDA version
void generateRandomGraph(int num_nodes, Graph &graph) {
    graph.incoming_links.resize(num_nodes);
    graph.out_degrees.assign(num_nodes, 0);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            // 10% probability of link i -> j
            if ((rand() % 100) < 10) {
                // Link from i -> j
                // For PageRank update, j needs to know about i (incoming)
                graph.incoming_links[j].push_back(i);
                graph.out_degrees[i]++;
            }
        }
    }
}

// --- Refactored Functions for Granular Profiling ---

// Initialization: opg[p] <- 1/N [cite: 180]
NOINLINE
void initializeRanks(int num_nodes, std::vector<float> &ranks) {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    float init_val = 1.0f / num_nodes;
    for (int i = 0; i < num_nodes; ++i) {
        ranks[i] = init_val;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_init += std::chrono::duration<double, std::milli>(t2 - t1).count();
}

// 1. Calculate Dangling Node Contribution (dp)
// Corresponds to: "for all p that has no out-links do" [cite: 184]
NOINLINE
float calculateDanglingSum(int num_nodes, const Graph &graph, const std::vector<float> &ranks) {
    auto t1 = std::chrono::high_resolution_clock::now();

    float dangling_sum = 0.0f;
    for (int i = 0; i < num_nodes; ++i) {
        if (graph.out_degrees[i] == 0) {
            dangling_sum += DAMPING * (ranks[i] / num_nodes); // [cite: 186]
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_dangling += std::chrono::duration<double, std::milli>(t2 - t1).count();
    return dangling_sum;
}

// --- Granular Helper Functions for Detailed Profiling ---

NOINLINE
float getBaseTerm(float dangling_sum, int num_nodes) {
    return dangling_sum + (1.0f - DAMPING) / num_nodes;
}

NOINLINE
bool hasOutgoingLinks(int degree) {
    return degree > 0;
}

NOINLINE
float computeLinkTerm(float rank, int degree) {
    return rank / (float)degree;
}

NOINLINE
void accumulateTerm(float &current_val, float term) {
    current_val += DAMPING * term;
}

// Helper for calculating rank of a single node
// This is the inner loop logic
NOINLINE
float calculateNodeRank(int p, int num_nodes, float dangling_sum, const Graph &graph, const std::vector<float> &ranks) {
    // Base value: dp + (1-d)/N [cite: 189]
    float val = getBaseTerm(dangling_sum, num_nodes);

    // Sum incoming link contributions
    // Corresponds to: "for all ip in ih[p] do" [cite: 190]
    for (int source : graph.incoming_links[p]) {
        int degree = graph.out_degrees[source];
        if (hasOutgoingLinks(degree)) {
            float term = computeLinkTerm(ranks[source], degree);
            accumulateTerm(val, term);
        }
    }
    return val;
}

// 2. Update PageRanks
// Corresponds to: "for all p in the graph do" [cite: 187]
NOINLINE
void updateRanks(int num_nodes, const Graph &graph, const std::vector<float> &ranks, std::vector<float> &new_ranks, float dangling_sum) {
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int p = 0; p < num_nodes; ++p) {
        new_ranks[p] = calculateNodeRank(p, num_nodes, dangling_sum, graph, ranks);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    time_update += std::chrono::duration<double, std::milli>(t2 - t1).count();
}

// Main PageRank Procedure [cite: 174]
void pageRankSequential(int num_nodes, const Graph &graph, std::vector<float> &ranks) {
    std::vector<float> new_ranks(num_nodes);

    initializeRanks(num_nodes, ranks);

    // Iterative Process [cite: 182]
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        
        float dangling_sum = calculateDanglingSum(num_nodes, graph, ranks);

        updateRanks(num_nodes, graph, ranks, new_ranks, dangling_sum);

        // 3. Update references for next iteration [cite: 192]
        ranks = new_ranks;
    }
}

int main() {
    // Disable profiling initially (skip graph generation overhead)
    moncontrol(0);

    std::cout << "Sequential PageRank Implementation..." << std::endl;

    // 1. Setup Graph
    Graph graph;
    generateRandomGraph(NUM_NODES, graph);
    
    // Count edges for display
    int edge_count = 0;
    for(const auto& links : graph.incoming_links) edge_count += links.size();
    std::cout << "Nodes: " << NUM_NODES << ", Edges: " << edge_count << std::endl;

    // 2. Run and Time the Algorithm
    std::vector<float> ranks(NUM_NODES);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Enable profiling ONLY for the algorithm
    moncontrol(1);
    pageRankSequential(NUM_NODES, graph, ranks);
    moncontrol(0); // Disable profiling again
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // 3. Output Results
    std::cout << "\nExecution Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Top 5 Node Ranks:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Node " << i << ": " << ranks[i] << std::endl;
    }

    std::cout << "\n--- Profiling Results (Manual Instrumentation) ---" << std::endl;
    std::cout << "Initialization: " << time_init << " ms" << std::endl;
    std::cout << "Dangling Sum:   " << time_dangling << " ms" << std::endl;
    std::cout << "Update Ranks:   " << time_update << " ms" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
