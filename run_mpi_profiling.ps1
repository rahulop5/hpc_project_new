$node_counts = @(1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)
$cpp_file = "pagerank_mpi_gprof.cpp"
$exe_file = "pagerank_mpi_gprof.exe"
$results_file = "MPI_Result/Final_MPI_Report.txt"
$num_procs = 4

# Initialize results file
"PERFORMANCE ANALYSIS SUMMARY (MPI - $num_procs Processes)" | Out-File -FilePath $results_file -Encoding ascii
"Nodes`tExecution Time (ms)`tTop Hotspot (%)" | Out-File -FilePath $results_file -Append -Encoding ascii

# Read original content
$content = Get-Content -Path $cpp_file -Raw

foreach ($nodes in $node_counts) {
    Write-Host "Running MPI for $nodes nodes..."
    
    # Replace NUM_NODES
    $new_content = $content -replace "const int NUM_NODES = \d+;", "const int NUM_NODES = $nodes;"
    $new_content | Set-Content -Path $cpp_file -NoNewline
    
    # Compile
    # We use cmd /c to execute the compilation command. 
    # Ensure mpic++ is in your PATH.
    cmd /c "mpic++ -pg -o $exe_file $cpp_file"
    if ($LASTEXITCODE -ne 0) { 
        Write-Error "Compilation failed for $nodes. Ensure MPI is installed and mpic++ is in PATH."
        break 
    }
    
    # Run
    # Set GMON_OUT_PREFIX to generate separate profile files for each process (e.g. gmon.out.pid)
    $env:GMON_OUT_PREFIX = "gmon.out"
    
    # Run with mpiexec
    $output = cmd /c "mpiexec -n $num_procs ./$exe_file" 2>&1
    
    # Parse Time (Only from Rank 0 output)
    $time = "Error"
    foreach ($line in $output) {
        if ($line -match "Execution Time: ([\d\.]+) ms") {
            $time = $matches[1]
            break
        }
    }
    
    Write-Host "Time: $time ms"

    # Run gprof
    $hotspot = "0.00"
    $gmon_files = Get-ChildItem "gmon.out*" | Sort-Object LastWriteTime -Descending
    if ($gmon_files) {
        # Analyze the first found gmon file
        $gmon_file = $gmon_files[0].Name
        $analysis_file = "analysis_mpi_$nodes.txt"
        cmd /c "gprof $exe_file $gmon_file > $analysis_file"

        # Parse Hotspot
        # Look for updateRanks or calculateNodeRank or similar
        $patterns = @("updateRanks", "calculateNodeRank", "pageRankSequential", "MPI_Allgather", "MPI_Allreduce")
        foreach ($p in $patterns) {
            $match = Select-String -Path $analysis_file -Pattern $p -SimpleMatch -Quiet
            if ($match) {
                $line = Select-String -Path $analysis_file -Pattern $p -SimpleMatch | Select-Object -First 1
                if ($line) {
                    $tokens = ($line.Line -split '\s+') | Where-Object { $_ -ne '' }
                    if ($tokens.Count -gt 0) { $hotspot = $tokens[0] }
                }
                break
            }
        }
    } else {
        Write-Warning "No gmon.out file found for analysis."
    }

    # Append to results
    "$nodes`t$time`t$hotspot" | Out-File -FilePath $results_file -Append -Encoding ascii
    
    # Clean up gmon files to avoid confusion for next run
    Remove-Item "gmon.out*" -ErrorAction SilentlyContinue
}

Write-Host "MPI Profiling complete. Results saved to $results_file"
