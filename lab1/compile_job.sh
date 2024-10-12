#!/bin/bash

## Give the Job a descriptive name
#PBS -N makejob

## Output and error files
#PBS -o compile_job.out
#PBS -e compile_job.err

## How many machines should we get?
#PBS -l nodes=1

## Start
## Load appropriate module
module load openmpi/1.8.3

## Run make in the src folder (modify properly)
cd /home/parallel/parlab17/a1/
make