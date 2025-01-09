#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_kmeans.out
#PBS -e run_kmeans.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

cd /home/parallel/parlab17/a3_stavros
export CUDA_VISIBLE_DEVICES=1
# sizes='32 64 128 256 512 1024 2048'
sizes='1024'

#coordinates='4'
coordinates='32'

#centers='64'
centers='64'

loop_threashold='10'
# loop_threashold='100''

block_size='32 64 128 238 256 512 1024'

progs=(
	kmeans_seq
	kmeans_cuda_naive
	kmeans_cuda_transpose
	kmeans_cuda_shared
	kmeans_cuda_all_gpu
	kmeans_cuda_all_gpu_delta_reduction
)



for size in $sizes; do
	for coord in $coordinates; do
		for center in $centers; do
			filename=Execution_logs/Sz-${size}_Coo-${coord}_Cl-${center}.csv 
			echo "Implementation,blockSize,avg_loop_t,min_loop_t,max_loop_t," >> $filename

			filename=Execution_logs/silver1-V100_Sz-${size}_Coo-${coord}_Cl-${center}.csv 
			> filename
			echo "Implementation,blockSize,loop_total,avg_loop_t,min_loop_t,max_loop_t,avg_cpu_time,avg_gpu_time,transfers_time,alloc_time,gpu_alloc_time,gpu_get_time" >> $filename

			for prog in "${progs[@]}"; do
				if [[ $prog == 'kmeans_seq' ]]; then
					./${prog} -s $size -n $coord -c $center -l $loop_threashold
				fi
				for bs in $block_size; do
					if [[ $prog == 'kmeans_cuda_naive' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_transpose' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_shared' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_all_gpu' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_all_gpu_delta_reduction' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_all_gpu_reduction' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					elif [[ $prog == 'kmeans_cuda_gitman' ]]; then
						./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
					fi
				done
			done
		done
	done
done
