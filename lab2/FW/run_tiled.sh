#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw_tiled

## Output and error files
#PBS -o run_fw_tiled.out
#PBS -e run_fw_tiled.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:59:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/FW

# Create the result-directories
mkdir -p results
echo "THREADS, BATCH_SIZE, TILE_SIZE, TIME" > results/fw_tiled.csv

export GOMP_CPU_AFFINITY="0-63"

THREADS=(1 2 4 8 16 32 64)
SIZES=(4096)
TILE_SIZES=(16 32 64 128)

for b in ${TILE_SIZES[@]}
do
    for n in ${SIZES[@]}
    do
        for t in ${THREADS[@]}
        do
            for i in {1..5}
            do
                export OMP_NUM_THREADS=$t
                ./fw_tiled $n $b  1>>./results/fw_tiled.csv
            done
        done
    done
done