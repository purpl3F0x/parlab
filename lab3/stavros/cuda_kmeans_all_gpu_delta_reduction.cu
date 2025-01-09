#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "alloc.h"
#include "error.h"
#include "kmeans.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid() {
    return blockDim.x * blockIdx.x + threadIdx.x;
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static double euclid_dist_2_transpose(
  int     numCoords,
  int     numObjs,
  int     numClusters,
  double* objects,  // [numCoords][numObjs]
  double* clusters, // [numCoords][numClusters]
  int     objectId,
  int     clusterId) {
    int    i;
    double ans = 0.0;

    // TODO: Copy from full-offload
    for (i = 0; i < numCoords; i++) {
        auto diff = objects[objectId + i * numObjs] - clusters[clusterId + i * numClusters];
        ans += diff * diff;
    }
    return (ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static void find_nearest_cluster(int     numCoords,
                                            int     numObjs,
                                            int     numClusters,
                                            double* objects,        //  [numCoords][numObjs]
                                            double* deviceClusters, //  [numCoords][numClusters]
                                            int*    devicenewClusterSize, //  [numClusters]
                                            double* devicenewClusters, //  [numCoords][numClusters]
                                            int*    deviceMembership,  //  [numObjs]
                                            double* devdelta) {
    extern __shared__ double shmem_total[];
    double*                  shmemClusters = shmem_total; // [numCoords][numClusters]
    double* delta_reduce_buff              = shmem_total + numClusters * numCoords; // [BLOCK_SIZE]

    /* Get the global ID of the thread. */
    int tid = get_tid();

    //  TODO: Replacing (*devdelta)+= 1.0; with reduction:
    //  - each thread updates the single element of delta_reduce_buff
    //       corresponding to its local id (threadIdx.x) -> 1.0 if membership changes, otherwise 0.
    //  - Then, ensuring delta_reduce_buff is fully updated, its containts must be summed in
    //       delta_reduce_buff[0] either by one thread (lower perf) or with a tree-based reduction (similar to dot reduction example in slides)
    //  - Finally, delta_reduce_buff[0] (local value in block) must be added to devdelta (global
    //       delta value), ensuring write dependencies!

    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            shmemClusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    if (tid < numObjs) {
        int    index, i;
        double dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        min_dist =
          euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, 0);


        for (i = 1; i < numClusters; i++) {
            dist = euclid_dist_2_transpose(
              numCoords, numObjs, numClusters, objects, shmemClusters, tid, i);

            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        // TODO: Add delta to delta_reduce_buff[threadIdx.x]
        // This is branchless, and not initalizing the array reduces the number of writes
        delta_reduce_buff[threadIdx.x] = (deviceMembership[tid] != index) ? 1.0 : 0.0;

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;

        atomicAdd(&devicenewClusterSize[index], 1.0);

        for (int j = 0; j < numCoords; j++) {
            atomicAdd(&devicenewClusters[j * numClusters + index], objects[tid + j * numObjs]);
        }

        // Adder Tree for delta_reduce_buff
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                delta_reduce_buff[threadIdx.x] += delta_reduce_buff[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(devdelta, delta_reduce_buff[0]);
        }
    }
}

__global__ static void update_centroids(int     numCoords,
                                        int     numClusters,
                                        int*    devicenewClusterSize, //  [numClusters]
                                        double* devicenewClusters,    //  [numCoords][numClusters]
                                        double* deviceClusters)       //  [numCoords][numClusters])
{

    const int tid       = get_tid();
    const int clusterId = tid % numClusters;

    if (tid < numCoords * numClusters) {
        if (devicenewClusterSize[clusterId] > 0) {
            deviceClusters[tid] = devicenewClusters[tid] / devicenewClusterSize[clusterId];
        }
        devicenewClusters[tid] = 0;
    }
    __syncthreads();
    devicenewClusterSize[clusterId] = 0;
}

//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
void kmeans_gpu(double* objects,        /* in: [numObjs][numCoords] */
                int     numCoords,      /* no. features */
                int     numObjs,        /* no. objects */
                int     numClusters,    /* no. clusters */
                double  threshold,      /* % objects change membership */
                long    loop_threshold, /* maximum number of iterations */
                int*    membership,     /* out: [numObjs] */
                double* clusters,       /* out: [numClusters][numCoords] */
                int     blockSize) {

    double timing = wtime();
    double timing_internal;
    double timer_min = DBL_MAX, timer_max = 0;
    double timing_gpu, timing_cpu, timing_transfers;
    double alloc_time, gpu_alloc_time, gpu_get_time;
    double transfers_time = 0, cpu_time = 0, gpu_time = 0;

    int    i, j, loop = 0;
    double delta = 0, *dev_delta_ptr; /* % of objects change their clusters */

    // TODO: Copy me from transpose version
    double** dimObjects  = (double**)calloc_2d(numCoords, numObjs, sizeof(double));
    double** dimClusters = (double**)calloc_2d(numCoords, numClusters, sizeof(double));
    double** newClusters = (double**)calloc_2d(numCoords, numClusters, sizeof(double));

    printf("\n|-----------Full-offload Delta Reduction GPU Kmeans------------|\n\n");

#pragma omp parallel for private(i, j) schedule(static)
    for (i = 0; i < numObjs; i++) {
        for (j = 0; j < numCoords; j++) {
            dimObjects[j][i] = objects[i * numCoords + j];
        }
    }

    double* deviceObjects;
    double *deviceClusters, *devicenewClusters;
    int*    deviceMembership;
    int*    devicenewClusterSize; /* [numClusters]: no. objects assigned in each new cluster */

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i = 0; i < numObjs; i++)
        membership[i] = -1;

    timing     = wtime() - timing;
    alloc_time = timing * 1000;
    printf("t_alloc: %lf ms\n\n", 1000 * timing);
    timing                                       = wtime();
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize) ? blockSize : numObjs;
    const unsigned int numClusterBlocks =
      (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;


    /*	Define the shared memory needed per block.
      - BEWARE: Also add extra shmem for delta buffer.
      - BEWARE: We can overrun our shared memory here if there are too many
      clusters or too many coordinates!
      - This can lead to occupancy problems or even inability to run.
      - Your exercise implementation is not requested to account for that (e.g. always assume
      deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize =
      numClusters * numCoords * sizeof(double) +  // deviceClusters[numCoords][numClusters]
      numThreadsPerClusterBlock * sizeof(double); // delta_reduce_buff[BLOCK_SIZE]

    const unsigned int update_centroids_block_sz =
      (numCoords * numClusters > blockSize) ? blockSize : numCoords * numClusters;

    const unsigned int update_centroids_dim_sz =
      (numCoords * numClusters + update_centroids_block_sz - 1) / update_centroids_block_sz + 1;


    cudaDeviceProp deviceProp;
    int            deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster "
              "centroids\n");
    }

    checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(double)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters * numCoords * sizeof(double)));
    checkCuda(cudaMalloc(&devicenewClusters, numClusters * numCoords * sizeof(double)));
    checkCuda(cudaMalloc(&devicenewClusterSize, numClusters * sizeof(int)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));


    timing         = wtime() - timing;
    gpu_alloc_time = timing * 1000;
    printf("t_alloc_gpu: %lf ms\n\n", 1000 * timing);
    timing = wtime();

    checkCuda(cudaMemcpy(
      deviceObjects, dimObjects[0], numObjs * numCoords * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(
      cudaMemcpy(deviceMembership, membership, numObjs * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceClusters,
                         dimClusters[0],
                         numClusters * numCoords * sizeof(double),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters * sizeof(int)));
    free(dimObjects[0]);

    timing       = wtime() - timing;
    gpu_get_time = timing * 1000;
    printf("t_get_gpu: %lf ms\n\n", 1000 * timing);
    timing = wtime();


    do {
        timing_internal = wtime();
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));
        timing_gpu = wtime();

        // printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d,
        // shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock,
        // clusterBlockSharedDataSize/1000);
        find_nearest_cluster<<<numClusterBlocks,
                               numThreadsPerClusterBlock,
                               clusterBlockSharedDataSize>>>(numCoords,
                                                             numObjs,
                                                             numClusters,
                                                             deviceObjects,
                                                             deviceClusters,
                                                             devicenewClusterSize,
                                                             devicenewClusters,
                                                             deviceMembership,
                                                             dev_delta_ptr);
        cudaDeviceSynchronize();
        checkLastCudaError();

        gpu_time += wtime() - timing_gpu;

        // printf("Kernels complete for itter %d, updating data in CPU\n", loop);

        timing_transfers = wtime();

        checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));

        transfers_time += wtime() - timing_transfers;


        timing_gpu = wtime();


        update_centroids<<<update_centroids_dim_sz, update_centroids_block_sz, 0>>>(
          numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);

        cudaDeviceSynchronize();
        checkLastCudaError();
        gpu_time += wtime() - timing_gpu;

        timing_cpu = wtime();
        delta /= numObjs;
        // printf("delta is %f - ", delta);
        loop++;
        // printf("completed loop %d\n", loop);
        cpu_time += wtime() - timing_cpu;

        timing_internal = wtime() - timing_internal;
        if (timing_internal < timer_min)
            timer_min = timing_internal;
        if (timing_internal > timer_max)
            timer_max = timing_internal;
    } while (delta > threshold && loop < loop_threshold);


    checkCuda(
      cudaMemcpy(membership, deviceMembership, numObjs * sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(dimClusters[0],
                         deviceClusters,
                         numClusters * numCoords * sizeof(double),
                         cudaMemcpyDeviceToHost));

    for (i = 0; i < numClusters; i++) {
        // if (newClusterSize[i] > 0) {
        for (j = 0; j < numCoords; j++) {
            clusters[i * numCoords + j] = dimClusters[j][i];
        }
        //}
    }

    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf "
           "ms\n\t-> t_loop_max = %lf ms\n\t"
           "-> t_cpu_avg = %lf ms\n\t-> t_gpu_avg = %lf ms\n\t-> t_transfers_avg = %lf "
           "ms\n\n|-------------------------------------------|\n",
           loop,
           1000 * timing,
           1000 * timing / loop,
           1000 * timer_min,
           1000 * timer_max,
           1000 * cpu_time / loop,
           1000 * gpu_time / loop,
           1000 * transfers_time / loop);

    char outfile_name[1024] = { 0 };
    sprintf(outfile_name,
            "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv",
            numObjs * numCoords * sizeof(double) / (1024 * 1024),
            numCoords,
            numClusters);
    FILE* fp = fopen(outfile_name, "a+");
    if (!fp)
        error("Filename %s did not open succesfully, no logging performed\n", outfile_name);
    fprintf(fp,
            "%s, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
            "All_GPU_Delta_Reduction",
            blockSize,
            timing * 1000,
            timing / loop * 1000,
            timer_min * 1000,
            timer_max * 1000,
            cpu_time / loop * 1000,
            gpu_time / loop * 1000,
            transfers_time / loop * 1000,
            alloc_time,
            gpu_alloc_time,
            gpu_get_time);
    fclose(fp);

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(devicenewClusterSize));
    checkCuda(cudaFree(deviceMembership));

    return;
}
