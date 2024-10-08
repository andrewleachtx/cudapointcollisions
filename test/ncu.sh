#!/bin/bash

# reducing particle & thread sizes for times sake
thread_sizes=(256 512)
# particle_sizes=(1 1000 1000000 10000000 50000000 100000000 500000000 750000000)
particle_sizes=(1 1000 100000)

# https://stackoverflow.com/questions/17066250/create-timestamp-variable-in-bash-script
timestamp_hash=$(date +%s)

# for i in "${array[@]}"
for n in "${particle_sizes[@]}"; do
    for threads in "${thread_sizes[@]}"; do
        output_dir="./results/cout/${n}"
        mkdir -p $output_dir
        output_file="${output_dir}/${n}_particles_${threads}_threads_${timestamp_hash}.txt"

        echo "---------------------------" | tee -a $output_file
        echo "Running with $n Particles and $threads Threads per block:" | tee -a $output_file

        # 1000 iterations should be fine?
        ncu --launch-count 500 --target-processes all -o "prof_$timestamp_hash" build/Debug/CUDAPOINTCOLLISIONS $n $threads

        echo "Finished running with $n particles and $threads threads per block" | tee -a $output_file
    done
done

echo "Finished all simulations"

# CURRENT SCRIPT (500 IS WAY TOO MUCH BUT I USED IT INITIALLY)
# ncu --launch-count 500 --target-processes all --set full --import-source on -o "profile_whatever" build/Debug/CUDAPOINTCOLLISIONS 10000 256