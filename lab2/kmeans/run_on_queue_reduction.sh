#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans_reduction_parlab17

## Output and error files
#PBS -o run_kmeans_reduction.out
#PBS -e run_kmeans_reduction.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/kmeans

SIZE=256
COORDS=16
CLUSTERS=32
LOOPS=10

# Create results directory
mkdir -p ./results
> ./results/reduction.out
> ./results/reduction_small.out


./kmeans_seq -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction.out

export GOMP_CPU_AFFINITY="0-15 16-31 32-47 48-63"

for i in 1 2 4 8 16 32 64
do
    export  OMP_NUM_THREADS=$i
    ./kmeans_omp_reduction -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction.out
done


# Run rediction for {256, 1, 4, 10}
SIZE=256
COORDS=1
CLUSTERS=4
LOOPS=10

export GOMP_CPU_AFFINITY="0-15 16-31 32-47 48-63"

./kmeans_seq -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction.out

for i in 1 2 4 8 16 32 64
do
    export  OMP_NUM_THREADS=$i
    ./kmeans_omp_reduction -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction_small.out
done


# Run reduction with numa aware io

./kmeans_seq -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction.out

for i in 1 2 4 8 16 32 64
do
    export  OMP_NUM_THREADS=$i
    ./kmeans_omp_reduction_numa_aware_io -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/reduction_small.out
done
