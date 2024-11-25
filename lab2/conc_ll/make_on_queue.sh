#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_conc

## Output and error files
#PBS -o make_conc.out
#PBS -e make_conc.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:01:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab17/a2/conc_ll/
make
