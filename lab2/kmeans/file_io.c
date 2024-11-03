#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* strtok() */
#include <sys/stat.h>
#include <sys/types.h> /* open() */
#include <unistd.h>    /* read(), close() */

#ifdef _OPENMP
    #include <omp.h>
#endif
#include "kmeans.h"

#ifdef _NUMA_AWARE
const char numa_aware[] = "NUMA-Aware";
#else
const char numa_aware[] = "Non-NUMA-Aware";
#endif

double *dataset_generation(int numObjs, int numCoords)
{
    double *objects = NULL;
    long    i, j;
    // Random values that will be generated will be between 0 and 10.
    double val_range = 10;

    /* allocate space for objects[][] and read all objects */
    objects = (typeof(objects))malloc(numObjs * numCoords * sizeof(*objects));

    /*
     * Hint : Could dataset generation be performed in a more "NUMA-Aware" way?
     *        Need to place data "close" to the threads that will perform operations on them.
     *        reminder : First-touch data placement policy
     */

    for (i = 0; i < numObjs; i++) {
        unsigned int seed = i;

        // clang-format off
		#if  defined(_OPENMP) && defined(_NUMA_AWARE)
			#pragma message "NUMA-Aware dataset generation"
			#pragma omp parallel for schedule(static)
		#endif
        // clang-format on
        for (j = 0; j < numCoords; j++) {
            objects[i * numCoords + j] = (rand_r(&seed) / ((double)RAND_MAX)) * val_range;
            if (_debug && i == 0)
                LOG("object[i=%ld][j=%ld]=%f\n", i, j, objects[i * numCoords + j]);
        }
    }

    return objects;
}
