# Give the Job a descriptive name
#PBS -N make_kmeans_mpi

## Output and error files
#PBS -o make_kmeans.out
#PBS -e make_kmeans.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:00:59

## Start 
## Run make in the src folder

cd $HOME/a4/kmeans
make