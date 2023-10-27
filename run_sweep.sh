#!/usr/bin/env bash

# first argument is data dir
data_dir=$1

# second argument is output dir
output_dir=$2

seeds=( 1 2 3 )
cache_sizes=( 0 1 2 3 4 5 6 7 8 9 )

for seed in "${seeds[@]}"
do
    for cache_size in "${cache_sizes[@]}"
    do
        python3 run_benchmark.py --seed "$seed" --cache-limit-gib "$cache_size" --data-dir "$data_dir" --output-dir "$output_dir"
    done
done


