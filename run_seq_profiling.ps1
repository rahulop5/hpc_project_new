$node_counts = @(1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)
$cpp_file = "pagerank_sequential_gprof.cpp"
$exe_file = "pagerank_seq_gprof.exe"
$results_file = "FINAL_REPORT_SEQUENTIAL.txt"

# Initialize results file
"Nodes,Time_ms" | Out-File -FilePath $results_file -Encoding ascii

# Read original content
$content = Get-Content -Path $cpp_file -Raw

foreach ($nodes in $node_counts) {
    Write-Host "Running for $nodes nodes..."
    
    # Replace NUM_NODES
    $new_content = $content -replace "const int NUM_NODES = \d+;", "const int NUM_NODES = $nodes;"
    $new_content | Set-Content -Path $cpp_file -NoNewline
    
    # Compile with -pg for gprof
    # Using -O2 to be realistic, but noinline attribute ensures functions are kept
    & g++ -pg -O2 -o pagerank_seq_gprof $cpp_file
    if ($LASTEXITCODE -ne 0) { 
        Write-Error "Compilation failed for $nodes"
        break 
    }
    
    # Run and capture output
    # gmon.out is generated in the current directory
    $output = & .\$exe_file 2>&1
    
    # Parse Time
    $time = "Error"
    foreach ($line in $output) {
        if ($line -match "Execution Time: ([\d\.]+) ms") {
            $time = $matches[1]
            break
        }
    }
    
    Write-Host "Time: $time ms"

    # Append to results
    "$nodes,$time" | Out-File -FilePath $results_file -Append -Encoding ascii
    
    # Run gprof
    # We use cmd /c to ensure redirection works
    cmd /c "gprof $exe_file gmon.out > analysis_seq_gprof_$nodes.txt"
}

Write-Host "Profiling complete. Results saved to $results_file and analysis_seq_gprof_*.txt"
