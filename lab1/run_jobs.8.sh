#!/bin/bash

## Give the Job a descriptive name
#PBS -N GoL-8

## Output and error files
#PBS -o run_job.out
#PBS -e run_job.err

## Limit memory, runtime etc.
#PBS -l walltime=01:00:00

## How many nodes:processors_per_node should we get?
#PBS -l nodes=1:ppn=8


## Start
## Load appropriate module
module load openmpi/1.8.3

## Setup working dir and clear results file
cd /home/parallel/parlab17/a1/
mkdir -p results
> ./results/game_8.out

## Run the job (use full paths to make sure we execute the correct thing)
## Execute each run 10 times

# 64x64
for i in $(seq 1 10);
do
    ./game_of_life 64 1000 8 1>>./results/game_8.out
done

# 1024x1024
for i in $(seq 1 10);
do
    ./game_of_life 1024 1000 8 1>>./results/game_8.out
done
# 4096x4096
for i in $(seq 1 10);
do
    ./game_of_life 4096 1000 8 1>>./results/game_8.out
done