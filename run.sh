#!/bin/bash

particle_sizes=(1 100 1000 100000 1000000 10000000 50000000 100000000 500000000 750000000) 

# https://stackoverflow.com/questions/17066250/create-timestamp-variable-in-bash-script
timestamp_hash=$(date +%s)
output_dir="./results/cout"
mkdir -p $output_dir
output_file="${output_dir}/result_${timestamp_hash}.txt"

# for i in "${array[@]}"
for n in "${particle_sizes[@]}"
do
    echo "---------------------------" | tee -a $output_file
    echo "Running with $n Particles:" | tee -a $output_file
    # https://phoenixnap.com/kb/bash-redirect-stderr-to-stdout#
    make run n=$n >> $output_file 2>&1
    echo "Finished running" | tee -a $output_file
done

echo "Finished all simulations"