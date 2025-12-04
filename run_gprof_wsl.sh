#!/bin/bash

NODE_COUNTS=(10 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000)
SOURCE_FILE="pagerank_sequential_gprof.cpp"
TEMP_CPP="pagerank_wsl_temp.cpp"
EXE_FILE="pagerank_wsl"
RESULTS_FILE="final_gprof_results.txt"

echo "WSL GPROF PROFILING RESULTS" > $RESULTS_FILE
echo "===========================" >> $RESULTS_FILE
echo "Nodes	Iterations	Hotspot(%)" >> $RESULTS_FILE

for NODES in "${NODE_COUNTS[@]}"; do
    echo "Processing $NODES nodes..."
    
    # Calculate iterations (dynamic scaling logic)
    # Avoid division by zero or issues with small numbers
    if [ $NODES -eq 0 ]; then continue; fi
    
    # Bash integer arithmetic
    # 100,000,000 / (N*N)
    DENOM=$((NODES * NODES))
    HIGH_ITERS=$((100000000 / DENOM))
    
    if [ $HIGH_ITERS -lt 20 ]; then HIGH_ITERS=20; fi
    if [ $HIGH_ITERS -gt 1000000 ]; then HIGH_ITERS=1000000; fi
    
    # Create temp file with modified constants
    # We use sed to replace the lines. 
    sed "s/const int NUM_NODES = [0-9]*;/const int NUM_NODES = $NODES;/g" $SOURCE_FILE > $TEMP_CPP
    # Use a second sed pass for iterations to avoid overwriting
    sed -i "s/const int ITERATIONS = [0-9]*;/const int ITERATIONS = $HIGH_ITERS;/g" $TEMP_CPP
    
    # Compile
    g++ -pg -no-pie -O2 -o $EXE_FILE $TEMP_CPP
    
    # Run
    ./$EXE_FILE > /dev/null
    
    # Gprof
    gprof $EXE_FILE gmon.out > analysis_wsl_$NODES.txt
    
    # Parse
    # Look for updateRanks or calculateNodeRank. 
    # gprof output format: % time | cumulative | self | calls | self/call | total/call | name
    # We want the first column (% time)
    HOTSPOT_PCT=$(grep -E "updateRanks|calculateNodeRank" analysis_wsl_$NODES.txt | head -n 1 | awk '{print $1}')
    
    if [ -z "$HOTSPOT_PCT" ]; then
        HOTSPOT_PCT="0.00"
    fi
    
    echo "$NODES	$HIGH_ITERS	$HOTSPOT_PCT" >> $RESULTS_FILE
done

echo "Done."
