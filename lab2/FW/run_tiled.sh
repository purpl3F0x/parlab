#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw_tiled

## Output and error files
#PBS -o run_fw_tiled.out
#PBS -e run_fw_tiled.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:12:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/FW

# Create the result-directories
mkdir -p results
echo "THREADS, BATCH_SIZE, TILE_SIZE, TIME" > results/fw_tiled.csv


THREADS=(1 2 4 8)
SIZES=(1024 2048 4096)
TILE_SIZES=(32 64 128)

for b in ${TILE_SIZES[@]}
do
    for n in ${SIZES[@]}
    do
        for t in ${THREADS[@]}
        do
            export OMP_NUM_THREADS=$t
            ./fw_sr $n $b  1>>./results/fw_tiled.csv
        done
    done
done


