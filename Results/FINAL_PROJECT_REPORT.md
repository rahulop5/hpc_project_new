# PageRank Algorithm Parallelization: Performance Analysis using MPI and CUDA

## 0. Abstract
The rapid growth of the World Wide Web necessitates efficient algorithms for ranking web pages, with PageRank being a fundamental metric. This project aims to achieve significant performance improvements in calculating PageRank scores by leveraging parallel computing models to overcome the limitations of sequential processing. We implemented and compared three approaches: a baseline Sequential algorithm, a distributed memory model using **MPI** (Message Passing Interface), and a massive parallelization model using **CUDA** (Compute Unified Device Architecture) on GPUs. The experiments were conducted on a heterogeneous environment comprising multi-core CPUs (via WSL for MPI) and an NVIDIA GPU, utilizing synthetic random graphs ranging from 1 to 10,000 nodes. The results demonstrate that while parallel overhead dominates at small input sizes, both MPI and CUDA achieve substantial speedups as the graph size increases, with CUDA providing the highest efficiency for large-scale matrix operations. The source code, profiling scripts, and performance datasets generated during this study are available for reproducibility.

## 1. Introduction
The PageRank algorithm, originally developed by Larry Page and Sergey Brin, stands as a cornerstone of modern information retrieval and web search engines. It revolutionized the way search results are ordered by treating links between web pages as votes of importance. In essence, a page is considered important if it is linked to by other important pages. Mathematically, this models the behavior of a "random surfer" who clicks on links at random, eventually settling into a steady-state distribution of probabilities across the web graph.

However, the computational challenge of PageRank is immense. The web graph contains billions of nodes (pages) and trillions of edges (links). The algorithm is iterative, requiring matrix-vector multiplications that scale with the size of the graph. A traditional sequential implementation, which processes nodes one by one, suffers from linear time complexity per iteration. As the dataset grows, the execution time becomes prohibitively long, making real-time or frequent updates impossible. This bottleneck provides the primary motivation for this project: to explore parallel computing techniques that can distribute the workload across multiple processing units.

Parallel computing offers two distinct paradigms to address this challenge. The first is the distributed memory model, exemplified by MPI. In this approach, the graph is partitioned across multiple independent processors (or nodes in a cluster), each with its own local memory. Processors perform calculations on their assigned chunk of the graph and exchange boundary information via network messages. This model is highly scalable and suits cluster environments but introduces communication overhead. The second paradigm is the shared memory or massive parallelization model, exemplified by CUDA for Graphics Processing Units (GPUs). GPUs consist of thousands of small, efficient cores designed for parallel throughput. By mapping graph nodes to GPU threads, we can perform thousands of PageRank updates simultaneously.

This project investigates the trade-offs between these architectures. We implement the PageRank algorithm using C++ for the sequential baseline, OpenMPI for the distributed approach, and CUDA C++ for the GPU approach. The study is set against the backdrop of modern heterogeneous computing, where the choice of hardware—CPU clusters versus GPU accelerators—can drastically impact performance, energy efficiency, and cost. Understanding these performance characteristics is crucial for designing next-generation large-scale graph processing systems.

## 2. Objective of the Work
The primary objective of this work is to evaluate and compare the performance of the PageRank algorithm across three different execution models: Sequential, MPI, and CUDA. Specifically, the project aims to:
1.  **Implement** a robust baseline sequential version of PageRank to serve as a ground truth for correctness and performance comparison.
2.  **Parallelize** the algorithm using MPI to exploit distributed memory parallelism, focusing on data decomposition and communication efficiency (using `MPI_Allgatherv`).
3.  **Accelerate** the algorithm using CUDA to leverage the massive parallelism of GPUs, optimizing for memory coalescing and thread utilization.
4.  **Analyze** the scalability of each approach by varying the graph size ($N$) from small (1 node) to moderately large (10,000 nodes).
5.  **Profile** the code using tools like `gprof` to identify computational hotspots and bottlenecks, such as communication overhead in MPI or memory transfer latency in CUDA.

## 3. Major Contributions

### 3.1 Contributions from Reference Studies
This work builds upon the findings of the reference paper *"Performance Analysis of Parallelized PageRank Algorithm using OpenMP, MPI, and CUDA"*. The reference study provided critical insights into the theoretical speedups achievable. A major contribution identified in the literature is the classification of PageRank as a memory-bound problem. The reference work highlighted that while OpenMP provides easy shared-memory parallelism, it is limited by the CPU's memory bandwidth. MPI, conversely, scales well across nodes but suffers from latency when the graph is sparse and communication frequency is high. The most significant contribution from the reference domain is the demonstration that CUDA implementations often outperform CPU-based approaches by orders of magnitude for large $N$, provided the graph fits in GPU memory. They established that the overhead of data transfer between Host (CPU) and Device (GPU) is the primary bottleneck for smaller graphs, a finding this project seeks to validate.

### 3.2 Details of Experiments
To rigorously evaluate the implementations, we designed a comprehensive experimental test-bed.

**Algorithm Implementation:**
The core PageRank logic follows the standard power iteration method:
$$ PR(p_i) = \frac{1-d}{N} + d \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} $$
Where $d=0.85$ is the damping factor, and iterations were fixed at 20 to ensure consistent profiling windows across all methods.

**Data Generation:**
We utilized a synthetic graph generator to create random graphs with varying densities. The generator ensures a consistent structure across runs by using fixed random seeds. The dataset sizes ranged exponentially: $N = \{1, 5, 10, ..., 1000, ..., 10000\}$. This wide range allows us to observe three distinct phases: the overhead-dominated phase (small $N$), the transition phase, and the compute-dominated phase (large $N$).

**Hardware & Software Environment:**
-   **Sequential & CUDA:** Executed on a Windows environment with an NVIDIA GPU. The CUDA implementation utilized `nvcc` for compilation and optimized kernel configurations (block size 256).
-   **MPI:** Executed in a WSL (Windows Subsystem for Linux) environment using the OpenMPI library. This simulated a distributed environment on a multi-core machine.
-   **Profiling:** We employed `gprof` (GNU Profiler) for the CPU-based codes (Sequential and MPI). This required compiling with the `-pg` flag. For MPI, we handled the complexity of profiling multiple processes by generating unique `gmon.out` files for each rank and aggregating the analysis.

**Methodology:**
For every $N$, the execution time was recorded. For the MPI implementation, we specifically addressed the issue of load balancing by using `MPI_Allgatherv`, ensuring that when $N$ is not perfectly divisible by the number of processes, the workload is still handled correctly without data corruption. This was a crucial implementation detail that ensured the stability of our results.

## 4. Performance Measuring Metrics
To provide a quantitative assessment, the following metrics were used:

1.  **Execution Time (ms):** The wall-clock time taken for the core PageRank iterative loop to complete 20 iterations. This excludes graph generation time to focus purely on the algorithm's computational efficiency.
2.  **Speedup:** Defined as $S = \frac{T_{sequential}}{T_{parallel}}$. This metric indicates how many times faster the parallel version is compared to the sequential baseline. It is the primary indicator of the effectiveness of parallelization.
3.  **Hotspot Analysis (%):** Using `gprof`, we identified which functions consumed the most CPU time. This helps in understanding bottlenecks. For example, in MPI, we looked for time spent in communication primitives versus computation functions like `updateRanks`.
4.  **Scalability:** We analyzed how execution time increases as $N$ increases. An ideal scalable system should show a linear (or near-linear) increase in performance proportional to resources, or a sub-linear increase in time relative to data size.

## 5. Results / Outcomes

**Sequential Approach:**
The sequential implementation showed a predictable linear increase in execution time as $N$ grew. For $N=10,000$, the execution time was approximately **4.1 seconds**. Profiling revealed that nearly 100% of the time was spent in the nested loops of the `updateRanks` function, confirming that the algorithm is compute-intensive and an ideal candidate for parallelization.

**MPI Approach:**
The MPI implementation (4 processes) demonstrated the classic trade-off of distributed computing. For small $N$ ($< 1000$), the execution time was often comparable to or slightly slower than sequential due to the overhead of `MPI_Allreduce` and `MPI_Allgatherv`. However, as $N$ approached 10,000, the speedup became evident, with execution times dropping significantly compared to the sequential baseline (approx **0.12 seconds** for N=10,000), validating the scalability of the distributed model.

**CUDA Approach:**
The CUDA implementation delivered the highest performance for large datasets. While it incurred a fixed overhead for memory allocation and kernel launch (making it slower for $N < 100$), it outperformed both Sequential and MPI methods significantly at $N=10,000$. The massive parallelism allowed thousands of nodes to be updated simultaneously, reducing the compute time to mere milliseconds.

## 6. Limitations and Future Scope
While the project successfully demonstrates the benefits of parallelization, there are limitations. First, the use of a fixed iteration count (20) rather than convergence-based termination means our absolute times are lower than those in production systems; however, this was necessary for consistent profiling. Second, the synthetic random graphs may not perfectly mimic the "small-world" properties of real web graphs (like the power-law distribution of links), which could affect load balancing in MPI. Third, the current MPI implementation replicates the full graph structure on each node (though it only computes a slice), which limits the maximum problem size to the memory of a single node.

Future work will focus on:
1.  **Memory Optimization:** Implementing a fully distributed graph storage where each MPI node only stores its local partition of the graph, enabling the processing of graphs larger than single-node memory.
2.  **Hybrid Parallelism:** Combining MPI and OpenMP (or MPI and CUDA) to utilize both multiple nodes and the multiple cores/GPUs within each node.
3.  **Real Datasets:** Testing on real-world datasets (e.g., from the SNAP library) to evaluate performance on irregular graph structures.
4.  **Dynamic Graphs:** Adapting the algorithms to handle dynamic graphs where nodes and edges are added or removed in real-time.

## 7. Observations from the Study
The study clearly answers the research question regarding the efficacy of parallel models for graph algorithms. We observed that **"one size does not fit all."** For small datasets, the simplicity of the Sequential algorithm is unbeatable due to the absence of overhead. As data grows, MPI provides a scalable path using commodity hardware. However, for compute-intensive matrix operations like PageRank, the GPU (CUDA) architecture offers superior throughput.

A key observation was the impact of communication overhead. In MPI, the `Allgather` step is a synchronization barrier that can severely limit performance if the network is slow or if the workload is unbalanced. Similarly, in CUDA, data transfer between Host and Device is a bottleneck. The study confirms that efficient parallelization requires not just dividing the work, but minimizing the movement of data. The original question of "how much speedup is possible" is answered: orders of magnitude are possible, but they are strictly bounded by the non-parallelizable portions of the code (Amdahl's Law) and communication latency.
