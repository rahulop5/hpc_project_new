# PowerShell script to compile and profile the CUDA "clean" pagerank program with gprof
# Creates per-N gprof analysis files and a consolidated report `CUDA_Final_GPROF_Report.txt`

$node_counts = @(1,5,10,20,50,100,200,500,1000,2000,5000,6000,7000,8000,9000,10000)
$cu_file = "pagerank_cuda.cu"
$exe_file = "pagerank_cuda_clean_gprof.exe"
$results_file = "CUDA_Final_GPROF_Report.txt"

"PERFORMANCE ANALYSIS SUMMARY (CUDA - Host gprof)" | Out-File -FilePath $results_file -Encoding ascii
"Nodes`tExecution Time (ms)`tTop Hotspot (%)" | Out-File -FilePath $results_file -Append -Encoding ascii

# Read original file content so we can replace NUM_NODES each iteration
$content = Get-Content -Path $cu_file -Raw

foreach ($nodes in $node_counts) {
    Write-Host "Processing $nodes nodes..."

    # Replace NUM_NODES = <number>;
    $new_content = $content -replace "const int NUM_NODES = \d+;", "const int NUM_NODES = $nodes;"
    $new_content | Set-Content -Path $cu_file -NoNewline

    # ------------------------------
    #       IMPORTANT CHANGE
    # ------------------------------
    # Added: -allow-unsupported-compiler
    # Added: forcing host compiler to MSVC with -Xcompiler "-pg"
    # ------------------------------
    Write-Host "Compiling with nvcc..."
    & nvcc -allow-unsupported-compiler -o $exe_file $cu_file -Xcompiler "-pg" 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Compilation failed for $nodes nodes."
        break
    }

    # remove gmon files
    Remove-Item "gmon.out*" -ErrorAction SilentlyContinue

    # run executable
    Write-Host "Running $exe_file ..."
    $output = & .\$exe_file 2>&1

    # extract time
    $time = "0.00"
    foreach ($line in $output) {
        if ($line -match "Execution Time: ([\d\.]+) ms") {
            $time = $matches[1]
            break
        }
    }

    # if time not displayed → measure manually
    if ($time -eq "0.00") {
        $t = Measure-Command { & .\$exe_file > $null }
        $time = [math]::Round($t.TotalMilliseconds, 4)
    }

    # run gprof only if gmon exists
    $analysis_file = "analysis_cuda_$nodes.txt"
    $gmon_found = Get-ChildItem -Filter "gmon.out*" | Select-Object -First 1

    if ($gmon_found) {
        cmd /c "gprof $exe_file $($gmon_found.Name) > $analysis_file"

        # detect hotspot
        $patterns = @("updateRanks","calculateNodeRank","updatePageRank","computePageRank")
        $hotspot = "0.00"

        foreach ($p in $patterns) {
            $hit = Select-String -Path $analysis_file -Pattern $p -SimpleMatch -Quiet
            if ($hit) {
                $line = Select-String -Path $analysis_file -Pattern $p -SimpleMatch | Select-Object -First 1
                if ($line) {
                    $tokens = ($line.Line -split '\s+') | Where-Object { $_ -ne '' }
                    if ($tokens.Count -gt 0) { $hotspot = $tokens[0] }
                }
                break
            }
        }
    }
    else {
        Write-Warning "No gmon.out produced for $nodes nodes."
        $hotspot = "0.00"
    }

    # append results
    $line_out = "{0}`t{1}`t{2}" -f $nodes, $time, $hotspot
    $line_out | Out-File -FilePath $results_file -Append -Encoding ascii

    # cleanup
    Remove-Item "gmon.out*" -ErrorAction SilentlyContinue
}

Write-Host "DONE. Results saved in $results_file"
