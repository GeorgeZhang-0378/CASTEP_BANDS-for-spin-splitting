#!/bin/bash
# Run_All_Bands.sh

for i in $(seq 0 9); do
    seed="AM_${i}"

    if [ -f "${seed}.bands" ]; then
        echo "Processing ${seed} ..."
        python Plot_Bands.py "${seed}"
    else
        echo "Skipping ${seed} (no ${seed}.bands found)"
    fi
done
