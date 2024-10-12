parlab17@scirouter:~/a1$ cat run_job.1.sh
#!/bin/bash

## Give the Job a descriptive name
#PBS -N GoL-1

## Output and error files
#PBS -o run_job.out
#PBS -e run_job.err

## Limit memory, runtime etc.
#PBS -l walltime=00:35:00

## How many nodes:processors_per_node should we get?
#PBS -l nodes=1:ppn=1


## Start
## Load appropriate module
module load openmpi/1.8.3

## Setup working dir and clear results file
cd /home/parallel/parlab17/a1/
mkdir -p results
> ./results/game_1.out

## Run the job (use full paths to make sure we execute the correct thing)
## Execute each run 10 times

# 64x64
for i in $(seq 1 10);
do
    ./game_of_life 64 1000 1 1>>./results/game_1.out
done

# 1024x1024
for i in $(seq 1 10);
do
    ./game_of_life 1024 1000 1 1>>./results/game_1.out
done
# 4096x4096
for i in $(seq 1 10);
do
    ./game_of_life 4096 1000 1 1>>./results/game_1.out
done