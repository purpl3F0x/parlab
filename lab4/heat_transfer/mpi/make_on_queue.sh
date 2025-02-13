# Give the Job a descriptive name
#PBS -N make_heat_transfer_mpi

## Output and error files
#PBS -o make_heat_transfer_mpi.out
#PBS -e make_heat_transfer_mpi.err

## How many machines should we get?
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:01:00

## Start
## Run make in the src folder

module load openmpi/1.8.3
cd $HOME/a4/heat_transfer/mpi
make clean && make