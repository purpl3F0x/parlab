#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

#ifndef _NO_LOG
    #define LOG(...)                                                                                                                                 \
        do {                                                                                                                                         \
            fprintf(stderr, __VA_ARGS__);                                                                                                            \
        } while (0);

    #define LOG_FLUSH()                                                                                                                              \
        {                                                                                                                                            \
            fflush(stderr);                                                                                                                          \
        }

#else
    #define LOG(...) {}
    #define LOG_FLUSH() {}
#endif

void kmeans(double *objects, int numCoords, int numObjs, int numClusters, double threshold, long loop_threshold, int *membership, double *clusters);

double *dataset_generation(int numObjs, int numCoords);

int check_repeated_clusters(int, int, double *);

double wtime(void);

extern int _debug;

#endif
