#!/usr/bin/env bash

seeds=( 1 2 3 )
cache_sizes=( 0 1 2 3 4 5 6 7 8 9 )

for seed in "${seeds[@]}"
do
    for cache_size in "${cache_sizes[@]}"
    do
        python3 run_benchmark.py --seed "$seed" --cache-limit-gib "$cache_size"
    done
done


