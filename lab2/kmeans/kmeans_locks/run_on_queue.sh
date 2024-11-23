#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_kmeans.out
#PBS -e run_kmeans.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:40:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/kmeans/kmeans_locks
THREADS=(1 2 4 8 16 32 64)
SIZE=32
COORDS=16
CLUSTERS=32
LOOPS=10


mkdir -p ./results

export GOMP_CPU_AFFINITY="0-15 16-31 32-47 48-63"

for thread_num in "${THREADS[@]}"; do
	export OMP_NUM_THREADS=$thread_num
	./kmeans_omp_critical -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1 >> ./results/kmeans_critical.out
	./kmeans_omp_array_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1 >> ./results/kmeans_array_lock.out
	./kmeans_omp_pthread_spin_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_pthread_spin_lock.out         
	./kmeans_omp_clh_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_clh_lock.out
	./kmeans_omp_nosync_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_no_sync_lock.out
	./kmeans_omp_tas_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_tas_lock.out
	./kmeans_omp_pthread_mutex_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_pthread_mutex_lock.out
	./kmeans_omp_ttas_lock -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS 1>>./results/kmeans_ttas_lock.out
done
