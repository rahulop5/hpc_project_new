# PageRank Performance Analysis Project

This project implements the PageRank algorithm using three different approaches to analyze performance scaling:
1.  **Sequential** (C++)
2.  **Parallel MPI** (Message Passing Interface)
3.  **Parallel CUDA** (NVIDIA GPU)

## Prerequisites

*   **Windows 10/11**
*   **Visual Studio Code**
*   **MinGW-w64** (for `g++` and `gprof`)
*   **WSL (Windows Subsystem for Linux)** with `mpic++` and `mpirun` installed (for MPI)
*   **NVIDIA CUDA Toolkit** (for `nvcc`)

---

## 1. Sequential Implementation

The sequential version serves as the baseline for performance comparison.

*   **Source File (Clean):** `pagerank_sequential.cpp` (For code review/presentation)
*   **Source File (Profiling):** `pagerank_sequential_gprof.cpp` (Instrumented for gprof)
*   **Script:** `run_seq_profiling.ps1`

### Automated Run (Recommended)
Open a PowerShell terminal in VS Code and run:
```powershell
.\run_seq_profiling.ps1
```
**Output:**
*   Generates `FINAL_REPORT_SEQUENTIAL.txt` containing execution times for N=1 to 10000.
*   Generates individual `analysis_seq_gprof_*.txt` files for detailed profiling.

### Manual Compilation & Execution
To compile and run the code manually (e.g., for a single node count):

**Clean Version (No Profiling):**
```powershell
g++ -O2 -o pagerank_seq pagerank_sequential.cpp
.\pagerank_seq.exe
```

**Profiling Version (With gprof):**
```powershell
g++ -pg -O2 -o pagerank_seq_gprof pagerank_sequential_gprof.cpp
.\pagerank_seq_gprof.exe
gprof pagerank_seq_gprof.exe gmon.out > analysis.txt
```

---

## 2. MPI Implementation (Parallel CPU)

The MPI version distributes the graph across multiple processes. This **must** be run inside the WSL environment.

*   **Source File:** `pagerank_mpi.cpp`
*   **Script:** `run_mpi_analysis.sh`

### Automated Run (Recommended)
You can run this directly from the Windows PowerShell terminal by invoking WSL:
```powershell
wsl ./run_mpi_analysis.sh
```
*Alternatively, if you are already inside a WSL terminal:*
```bash
./run_mpi_analysis.sh
```

**Output:**
*   Generates `MPI_Result/Final_MPI_Report.txt`.

### Manual Compilation & Execution (WSL)
To compile and run manually inside WSL:

```bash
# Compile with profiling enabled
mpic++ -pg -O2 -o pagerank_mpi pagerank_mpi.cpp

# Run with 4 processes
mpirun --oversubscribe -np 4 ./pagerank_mpi

# Generate gprof report
gprof pagerank_mpi gmon.out > mpi_analysis.txt
```

---

## 3. CUDA Implementation (Parallel GPU)

The CUDA version utilizes the GPU for massive parallelism.

*   **Source File:** `pagerank_cuda.cu`
*   **Script:** `run_cuda_gprof.ps1`

### Automated Run (Recommended)
Open a PowerShell terminal in VS Code and run:
```powershell
.\run_cuda_gprof.ps1
```

**Output:**
*   Generates `CUDA_Final_GPROF_Report.txt`.

### Manual Compilation & Execution
To compile and run manually:

```powershell
# Compile (allowing unsupported compiler if necessary for VS integration)
nvcc -allow-unsupported-compiler -o pagerank_cuda pagerank_cuda.cu

# Run
.\pagerank_cuda.exe
```

To compile with profiling support (for host-side gprof):
```powershell
nvcc -allow-unsupported-compiler -o pagerank_cuda_gprof pagerank_cuda.cu -Xcompiler "-pg"
.\pagerank_cuda_gprof.exe
gprof pagerank_cuda_gprof.exe gmon.out > cuda_analysis.txt
```

---

## Summary of Commands

| Implementation | Automated Script (PowerShell) | Manual Compile Command |
| :--- | :--- | :--- |
| **Sequential** | `.\run_seq_profiling.ps1` | `g++ -O2 -o seq pagerank_sequential.cpp` |
| **MPI** | `wsl ./run_mpi_analysis.sh` | `mpic++ -O2 -o mpi pagerank_mpi.cpp` |
| **CUDA** | `.\run_cuda_gprof.ps1` | `nvcc -o cuda pagerank_cuda.cu` |
