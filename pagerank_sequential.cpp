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

// Configuration Constants
const int NUM_NODES = 1024;
const int ITERATIONS = 20;
const float DAMPING = 0.85f;

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

// Main PageRank Procedure [cite: 174]
void pageRankSequential(int num_nodes, const Graph &graph, std::vector<float> &ranks) {
    std::vector<float> new_ranks(num_nodes);

    // Initialization: opg[p] <- 1/N [cite: 180]
    float init_val = 1.0f / num_nodes;
    for (int i = 0; i < num_nodes; ++i) {
        ranks[i] = init_val;
    }

    // Iterative Process [cite: 182]
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        
        // 1. Calculate Dangling Node Contribution (dp)
        // Corresponds to: "for all p that has no out-links do" [cite: 184]
        float dangling_sum = 0.0f;
        for (int i = 0; i < num_nodes; ++i) {
            if (graph.out_degrees[i] == 0) {
                dangling_sum += DAMPING * (ranks[i] / num_nodes); // [cite: 186]
            }
        }

        // 2. Update PageRanks
        // Corresponds to: "for all p in the graph do" [cite: 187]
        for (int p = 0; p < num_nodes; ++p) {
            // Base value: dp + (1-d)/N [cite: 189]
            float val = dangling_sum + (1.0f - DAMPING) / num_nodes;

            // Sum incoming link contributions
            // Corresponds to: "for all ip in ih[p] do" [cite: 190]
            for (int source : graph.incoming_links[p]) {
                if (graph.out_degrees[source] > 0) {
                    val += DAMPING * ranks[source] / graph.out_degrees[source]; // [cite: 191]
                }
            }
            new_ranks[p] = val;
        }

        // 3. Update references for next iteration [cite: 192]
        ranks = new_ranks;
    }
}

int main() {
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
    
    pageRankSequential(NUM_NODES, graph, ranks);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // 3. Output Results
    std::cout << "\nExecution Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Top 5 Node Ranks:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Node " << i << ": " << ranks[i] << std::endl;
    }

    return 0;
}