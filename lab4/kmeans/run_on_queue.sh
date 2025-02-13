# Give the Job a descriptive name
#PBS -N run_kmeans_mpi

## Output and error files
#PBS -o run_kmeans_mpi.out
#PBS -e run_kmeans_mpi.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8

##How long should the job run for?
#PBS -l walltime=00:15:00

## Start
## Load appropriate module
module load openmpi/1.8.3


# cd to the directory where the qsub command was issued
cd $HOME/a4/kmeans

# Clear the output file
mkdir -p output
> ./output/kmeans.out


# Run the job to PROCS times to get an average


# Usage: ./kmeans_mpi [switches]
#        -c num_clusters    : number of clusters (must be > 1)
#        -s size            : size of examined dataset
#        -n num_coords      : number of coordinates
#        -t threshold       : threshold value (default : 0.001)
#        -l loop_threshold  : iterations threshold (default : 10)
#        -d                 : enable debug mode
#        -h                 : print this help information

for i in 1 2 4 8 16 32 64
do

        mpirun --mca btl tcp,self -np ${i} ./kmeans_mpi -s 256 -n 16 -c 32 -l 10 1>>./output/kmeans.out
done