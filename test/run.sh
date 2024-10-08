#!/bin/bash

thread_sizes=(32 64 128 256 512 1024)
particle_sizes=(1 100 1000 100000 1000000 10000000 50000000 100000000 500000000 750000000) 

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

        build/CUDAPOINTCOLLISIONS $n $threads >> $output_file 2>&1

        echo "Finished running with $n Particles and $threads Threads per block" | tee -a $output_file
    done
done

echo "Finished all simulations"

