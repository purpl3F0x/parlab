#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_conc_ll

## Output and error files
#PBS -o run_conc.out
#PBS -e run_conc.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=02:00:00

## Start
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/conc_ll
THREADS=(1 2 4 8 16 32 64 128)
LIST_SIZES=(1024 8192)
CONTAINS=("100 0 0" "80 10 10" "20 40 40" "0 50 50")
TOTAL_CORES=32

mkdir -p ./results

for thread_num in "${THREADS[@]}"; do

    cores=""
  for (( i=0; i<thread_num; i++ )); do
    if [ -z "$cores" ]; then
      cores=$((i % TOTAL_CORES))
    else
      cores="$cores,$((i % TOTAL_CORES))"
    fi
  done

  export MT_CONF=$cores
  echo "$MT_CONF"

        for triplet in "${CONTAINS[@]}"; do
                for list_size in "${LIST_SIZES[@]}"; do
                        read -r num1 num2 num3 <<< "$triplet"
                        echo "$list_size $num1 $num2 $num3"
                        ./x.cgl $list_size $num1 $num2 $num3 1>>./results/cgl.out
                        ./x.fgl $list_size $num1 $num2 $num3 1>>./results/fgl.out
                        ./x.lazy $list_size $num1 $num2 $num3 1>>./results/lazy.out
                        ./x.nb $list_size $num1 $num2 $num3 1>>./results/nb.out
                        ./x.opt $list_size $num1 $num2 $num3 1>>./results/opt.out

                        if [ $thread_num -eq 1 ]; then
                            ./x.serial $list_size $num1 $num2 $num3 1>>./results/serial.out
                        fi
                done
        done
done

