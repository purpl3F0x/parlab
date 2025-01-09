

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strtok() */
#include <sys/stat.h>
#include <sys/types.h> /* open() */
#include <unistd.h>    /* getopt() */

int _debug;
#include "alloc.h"
#include "error.h"
#include "kmeans.h"


/// This is the validation eps that will be used for result comparisson.
/// If set too low implementations with very divergent double operation order might lead to errors!
/// Always check error difference in these cases and adjust eps accordingly!
#ifdef VALIDATE
double validation_eps = 1e-2;
#endif

static void usage(char* argv0) {
    char* help = "Usage: %s [switches]\n"
                 "       -c num_clusters    : number of clusters (must be > 1)\n"
                 "       -s size            : size of examined dataset\n"
                 "       -n num_coords      : number of coordinates\n"
                 "       -t threshold       : threshold value (default : 0.001)\n"
                 "       -l loop_threshold  : iterations threshold (default : 10)\n"
                 "       -d                 : enable debug mode\n"
                 "       -h                 : print this help information\n"
                 "GPU extras:\n"
                 "       -b                 : blocksize\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

void DisplayHeader() {
    const int kb = 1024;
    const int mb = kb * kb;
    printf("NBody.GPU\n=========\n\n");

    printf("CUDA version:   v%d\n", CUDART_VERSION);
    printf("Thrust version: v%d.%d\n\n", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION);

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Devices: \n\n");

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("%d: %s: %d.%d\n", i, props.name, props.major, props.minor);
        printf("  Global memory:   %ldmb\n", props.totalGlobalMem / mb);
        printf("  Shared memory:   %dkb\n", props.sharedMemPerBlock / kb);
        printf("  Constant memory: %dkb\n", props.totalConstMem / kb);
        printf("  Block registers: %d\n\n", props.regsPerBlock);

        printf("  Warp size:         %d\n", props.warpSize);
        printf("  Threads per block: %d\n", props.maxThreadsPerBlock);
        printf("  Max block dimensions: [ %d, %d, %d ]\n",
               props.maxThreadsDim[0],
               props.maxThreadsDim[1],
               props.maxThreadsDim[2]);
        printf("  Max grid dimensions:  [ %d, %d, %d ]\n",
               props.maxGridSize[0],
               props.maxGridSize[1],
               props.maxGridSize[2]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    long         i, j, opt;
    extern char* optarg;
    extern int   optind;

    int block_size = 0;

    long    numClusters = 0, numCoords = 0, numObjs = 0;
    int*    membership; // [numObjs]
    double* objects;    // [numObjs * numCoords] data  objects
    double* clusters;   // [numClusters * numCoords] cluster center
    double  dataset_size = 0, threshold;
    long    loop_threshold;
    double  io_timing_read;

    /* some default values */
    _debug         = 0;
    threshold      = 0.001;
    loop_threshold = 10;
    numClusters    = 0;

    printf(
      "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    while ((opt = getopt(argc, argv, "b:n:t:l:c:s:dh")) != EOF) {
        switch (opt) {
            case 'b':
                block_size = atol(optarg);
                break;
            case 'c':
                numClusters = atol(optarg);
                break;
            case 't':
                threshold = atof(optarg);
                break;
            case 'l':
                loop_threshold = atol(optarg);
                break;
            case 's':
                dataset_size = atof(optarg);
                break;
            case 'n':
                numCoords = atol(optarg);
                break;
            case 'd':
                _debug = 1;
                break;
            case 'h':
            default:
                usage(argv[0]);
                break;
        }
    }
    if (!block_size)
        error("block_size not provided for GPU version, terminating\n");

    if (numClusters <= 1)
        usage(argv[0]);

    numObjs = (dataset_size * 1024 * 1024) / (numCoords * sizeof(double));

    if (numObjs < numClusters) {
        printf("Error: number of clusters must be larger than the number of data points to be "
               "clustered.\n");
        return 1;
    }

    DisplayHeader();

    puts("\n\n");

    printf("dataset_size = %.2f MB    numObjs = %ld    numCoords = %ld    numClusters = %ld, "
           "block_size = %d\n",
           dataset_size,
           numObjs,
           numCoords,
           numClusters,
           block_size);


    objects = dataset_generation(numObjs, numCoords);

    // Allocate space for clusters (coordinates of cluster centers)
    clusters = (double*)malloc(numClusters * numCoords * sizeof(double));

#ifdef VALIDATE
    // Allocate space for validation clusters (coordinates of cluster centers)
    double* validation_clusters = (double*)malloc(numClusters * numCoords * sizeof(double));
#endif

    // The first numClusters elements are selected as initial centers
    for (i = 0; i < numClusters; i++)
        for (j = 0; j < numCoords; j++) {
            clusters[i * numCoords + j] = objects[i * numCoords + j];
#ifdef VALIDATE
            validation_clusters[i * numCoords + j] = clusters[i * numCoords + j];
#endif
        }
    // check initial cluster centers for repetition
    if (check_repeated_clusters(numClusters, numCoords, clusters) == 0) {
        printf(
          "Error: some initial clusters are repeated. Please select distinct initial centers\n");
        return 1;
    }


    // printf("Initial cluster centers:\n");
    // for (i=0; i<numClusters; i++) {
    //     printf("clusters[%ld] =",i);
    //     for (j=0; j<numCoords; j++)
    //         printf(" %6.2f", clusters[i*numCoords + j]);
    //     printf("\n");
    // }


    // membership: the cluster id for each data object
    membership = (int*)malloc(numObjs * sizeof(int));

#ifdef VALIDATE
    // Perform validation run
    kmeans(objects,
           numCoords,
           numObjs,
           numClusters,
           threshold,
           loop_threshold,
           membership,
           validation_clusters);
#endif
    // start the core computation
    printf("\n");
    kmeans_gpu(objects,
               numCoords,
               numObjs,
               numClusters,
               threshold,
               loop_threshold,
               membership,
               clusters,
               block_size);
    printf("\n");


    // printf("Final cluster centers:\n");
    // for (i=0; i<numClusters; i++) {
    //     printf("clusters[%ld] = ",i);
    //    for (j=0; j<numCoords; j++)
    //         printf("%6.2f ", clusters[i*numCoords + j]);
    //     printf("\n");
    // }

#ifdef VALIDATE
    printf("Performing validation....");
    int ik, checked[numClusters][numCoords];
    for (i = 0; i < numClusters; i++)
        for (j = 0; j < numCoords; j++)
            checked[i][j] = 0;
    for (i = 0; i < numClusters; i++)
        for (j = 0; j < numCoords; j++)
            if (!checked[i][j]) {
                for (ik = 0; ik < numClusters; ik++)
                    if (!checked[i][j]) {
                        if (abs((validation_clusters[i * numCoords + j] -
                                 clusters[ik * numCoords + j]) /
                                validation_clusters[i * numCoords + j]) < validation_eps) {
                            checked[i][j] = 1;
                            break;
                        }
                    }
                if (!checked[i][j])
                    error("Validation failed: cluster[%d][%d]: %lf instead of %lf\n",
                          i,
                          j,
                          clusters[i * numCoords + j],
                          validation_clusters[i * numCoords + j]);
                // else printf("Validation ok: cluster[%d][%d]: %lf instead of %lf\n", i, j,
                // clusters[ik*numCoords + j],  validation_clusters[i*numCoords + j]);
            }
    printf("PASSED!\n");
    free(validation_clusters);
#endif
    free(objects);
    free(membership);
    free(clusters);

    return 0;
}
