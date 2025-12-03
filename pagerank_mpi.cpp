/*
 * MPI Implementation of PageRank Algorithm
 * Reference: "Performance Analysis of Parallelized PageRank Algorithm using OpenMP, MPI, and CUDA"
 * Implementation Strategy: Section III-D 
 */
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Configuration Constants
const int NUM_NODES = 1024;
const int ITERATIONS = 20;
const float DAMPING = 0.85f;

// Structure to hold the Local Graph Partitions
struct LocalGraph {
    // We store the full graph structure locally for simplicity in this demo,
    // but each process only ITERATES over its assigned range.
    // In a production memory-constrained env, we would only store the local slice.
    std::vector<std::vector<int>> incoming_links; 
    std::vector<int> out_degrees; 
};

void generateGraph(int num_nodes, LocalGraph &graph) {
    graph.incoming_links.resize(num_nodes);
    graph.out_degrees.assign(num_nodes, 0);

    // Using a fixed seed ensures all MPI processes generate the SAME graph structure
    // without needing complex graph distribution logic for this demo.
    srand(42); 

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            // 10% probability of link i -> j
            if ((rand() % 100) < 10) {
                graph.incoming_links[j].push_back(i);
                graph.out_degrees[i]++;
            }
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        std::cout << "MPI PageRank Implementation (" << world_size << " processes)..." << std::endl;
    }

    // 1. Setup Graph (Identical on all nodes for simplicity)
    LocalGraph graph;
    generateGraph(NUM_NODES, graph);

    // 2. Determine Local Workload (Data Decomposition)
    // "In the pageRank() function... iterative PageRank updates are distributed among processes" 
    int nodes_per_proc = NUM_NODES / world_size;
    int start_node = world_rank * nodes_per_proc;
    int end_node = (world_rank == world_size - 1) ? NUM_NODES : start_node + nodes_per_proc;

    // 3. Initialize Ranks
    // Entire vector is needed for calculation, but we only update our slice
    std::vector<float> opg(NUM_NODES);
    std::vector<float> local_npg(nodes_per_proc); // Buffer for computing local updates

    float init_val = 1.0f / NUM_NODES;
    for (int i = 0; i < NUM_NODES; ++i) opg[i] = init_val;

    double start_time = MPI_Wtime();

    // 4. Iterative PageRank
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        
        // --- Step A: Calculate Local Dangling Sum ---
        float local_dangling_sum = 0.0f;
        // Iterate only over the full graph to find dangling nodes (could be optimized)
        // A better distributed approach splits this loop, but we stick to paper's logic of distributing computation
        
        // Each process calculates a partial sum of dangling nodes from its own slice?
        // To strictly follow "distributing computation", let's have each proc calc for its assigned nodes.
        for (int i = start_node; i < end_node; ++i) {
            if (graph.out_degrees[i] == 0) {
                local_dangling_sum += DAMPING * (opg[i] / NUM_NODES);
            }
        }

        // Reduce to get global dangling sum
        // "MPI_Allreduce... used to synchronize and exchange data" 
        float global_dangling_sum = 0.0f;
        MPI_Allreduce(&local_dangling_sum, &global_dangling_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // --- Step B: Calculate New PageRanks for Assigned Slice ---
        int local_idx = 0;
        for (int p = start_node; p < end_node; ++p) {
            float val = global_dangling_sum + (1.0f - DAMPING) / NUM_NODES;

            // Iterate over incoming links (ih[p])
            for (int source : graph.incoming_links[p]) {
                if (graph.out_degrees[source] > 0) {
                    val += DAMPING * opg[source] / graph.out_degrees[source];
                }
            }
            local_npg[local_idx++] = val;
        }

        // --- Step C: Synchronize Updates ---
        // "MPI_Allgather... used to synchronize and exchange data" 
        // Every process needs the FULL updated 'npg' vector to proceed to the next iteration.
        // We gather 'local_npg' slices from all processes into the main 'opg' vector.
        MPI_Allgather(local_npg.data(), nodes_per_proc, MPI_FLOAT, 
                      opg.data(), nodes_per_proc, MPI_FLOAT, 
                      MPI_COMM_WORLD);
        
        // Note: opg is now effectively npg for the next iteration
    }

    double end_time = MPI_Wtime();

    // 5. Output Results (Only Rank 0)
    if (world_rank == 0) {
        std::cout << "\nExecution Time: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
        std::cout << "Top 5 Node Ranks:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "Node " << i << ": " << opg[i] << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}