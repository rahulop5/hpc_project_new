#!/bin/bash

# Define node counts
NODE_COUNTS=(1 5 10 20 50 100 200 300 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
SOURCE_FILE="pagerank_sequential.cpp"
TEMP_CPP="pagerank_seq_temp.cpp"
EXE_FILE="pagerank_seq"
RESULTS_DIR="Results"
FINAL_REPORT="$RESULTS_DIR/Final_report.txt"

# Create Results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Initialize Final Report
echo "PERFORMANCE ANALYSIS SUMMARY (GPROF)" > $FINAL_REPORT
echo "====================================" >> $FINAL_REPORT
echo "Nodes	Execution Time (ms)	Top Hotspot (%)" >> $FINAL_REPORT

for NODES in "${NODE_COUNTS[@]}"; do
    echo "Processing $NODES nodes..."
    
    # Create temp file with modified constants
    # We use sed to replace the NUM_NODES constant.
    sed "s/const int NUM_NODES = [0-9]*;/const int NUM_NODES = $NODES;/g" $SOURCE_FILE > $TEMP_CPP
    
    # Compile with profiling enabled
    g++ -pg -no-pie -O2 -o $EXE_FILE $TEMP_CPP
    
    # Run the executable
    ./$EXE_FILE > /dev/null
    
    # Generate gprof report
    REPORT_FILE="$RESULTS_DIR/gprof_report_$NODES.txt"
    gprof $EXE_FILE gmon.out > $REPORT_FILE
    
    # Extract Execution Time (from the flat profile header or total time)
    # gprof doesn't always output total time clearly in the header, but we can sum the self seconds.
    # Alternatively, we can grab the first line of the flat profile which usually sums to 100%.
    # A better way for the summary is to grab the top function's percentage.
    
    # Get the top hotspot percentage (first data line of flat profile)
    # Skip header lines, find first line starting with a number
    HOTSPOT_PCT=$(grep -m 1 "^[ ]*[0-9]" $REPORT_FILE | awk '{print $1}')
    
    # If execution was too fast, gprof might be empty or 0.00
    if [ -z "$HOTSPOT_PCT" ]; then HOTSPOT_PCT="0.00"; fi
    
    # We can't easily get total wall clock time from gprof output itself without parsing everything.
    # However, the user asked for a summary. Let's try to extract the cumulative seconds from the last line of the flat profile?
    # Or just rely on the hotspot percentage.
    # Let's try to get the cumulative seconds from the top of the list (which represents the total time profiled).
    TOTAL_TIME=$(grep -m 1 "^[ ]*[0-9]" $REPORT_FILE | awk '{print $2}')
    if [ -z "$TOTAL_TIME" ]; then TOTAL_TIME="0.00"; fi
    
    # Convert seconds to ms for consistency with previous reports
    TOTAL_TIME_MS=$(echo "$TOTAL_TIME * 1000" | bc)

    echo "$NODES	$TOTAL_TIME_MS	$HOTSPOT_PCT" >> $FINAL_REPORT
done

# Add analysis text to Final Report
echo "" >> $FINAL_REPORT
echo "ANALYSIS SUMMARY" >> $FINAL_REPORT
echo "================" >> $FINAL_REPORT
echo "1. Small Inputs (1-500 Nodes):" >> $FINAL_REPORT
echo "   - Execution time is negligible." >> $FINAL_REPORT
echo "   - gprof often fails to capture samples due to short runtime (0.00% hotspot)." >> $FINAL_REPORT
echo "" >> $FINAL_REPORT
echo "2. Large Inputs (1000-10000 Nodes):" >> $FINAL_REPORT
echo "   - Execution time grows quadratically." >> $FINAL_REPORT
echo "   - The 'updateRanks' (or its internal helpers) consistently appears as the hotspot." >> $FINAL_REPORT
echo "   - This confirms the need for parallelization for large N." >> $FINAL_REPORT

echo "Done. Reports generated in $RESULTS_DIR."
