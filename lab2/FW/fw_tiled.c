/*
 * Tiled version of the Floyd-Warshall algorithm.
 * command-line arguments: N, B
 * N = size of graph
 * B = size of tile
 * works only when N is a multiple of B
 */
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

inline int  min(int a, int b);
inline void FW(int** A, int K, int I, int J, int N);
inline void FW_SSE(int** A, int K, int I, int J, int N);
// inline void FW_AVX1(int** A, int K, int I, int J, int N);
// inline void FW_AVX2(int** A, int K, int I, int J, int N);


void FW_AVX2(int** A, int K, int I, int J, int N);

int main(int argc, char** argv) {
    int**          A;
    int            i, j, k;
    struct timeval t1, t2;
    double         time;
    int            B = 64;
    int            N = 1024;

    if (argc != 3) {
        fprintf(stdout, "Usage %s N B\n", argv[0]);
        exit(0);
    }

    N = atoi(argv[1]);
    B = atoi(argv[2]);

    A = (int**)malloc(N * sizeof(int*));
    for (i = 0; i < N; i++) {
        A[i] = (int*)malloc(N * sizeof(int));
    }

    graph_init_random(A, -1, N, 128 * N);

    gettimeofday(&t1, 0);

    for (k = 0; k < N; k += B) {
        FW(A, k, k, k, B);

        // clang-format off
            #pragma omp parallel default (none) shared(A, N, B, k) private(i, j)
            {

                #pragma omp for private(i)   nowait
                for (i = 0; i < k; i += B)
                    FW(A, k, i, k, B);

                #pragma omp for private(i) schedule(static) nowait
                for (i = k + B; i < N; i += B)
                    FW(A, k, i, k, B);

                #pragma omp for private(j) schedule(static) nowait
                for (j = 0; j < k; j += B)
                    FW(A, k, k, j, B);

                #pragma omp for private(j) schedule(static) nowait
                for (j = k + B; j < N; j += B)
                    FW(A, k, k, j, B);

                #pragma omp taskwait

                #pragma omp for private(i, j) collapse(2) schedule(static) nowait
                for (i = 0; i < k; i += B)
                    for (j = 0; j < k; j += B)
                        FW(A, k, i, j, B);

                #pragma omp for private(i, j) collapse(2) schedule(static) nowait
                for (i = 0; i < k; i += B)
                    for (j = k + B; j < N; j += B)
                        FW(A, k, i, j, B);

                #pragma omp for private(i, j) collapse(2) schedule(static)  nowait
                for (i = k + B; i < N; i += B)
                    for (j = 0; j < k; j += B)
                        FW(A, k, i, j, B);

                #pragma omp for private(i, j) collapse(2) schedule(static) nowait
                for (i = k + B; i < N; i += B)
                    for (j = k + B; j < N; j += B)
                        FW(A, k, i, j, B);

                #pragma omp taskwait
            }
        }
    // clang-format on

    gettimeofday(&t2, 0);

    time = (double)((t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec) / 1000000;
    printf("FW_TILED,%d,%d,%.4f\n", N, B, time);

    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < N; j++) {
    //         fprintf(stdout, "%d\t", A[i][j]);
    //     }
    //     fprintf(stdout, "\n");
    // }

    return 0;
}

inline int min(int a, int b) {
    if (a <= b) {
        return a;
    } else {
        return b;
    }
}

void FW(int** A, int K, int I, int J, int N) {
    // int i, j, k;
    // FW_AVX1(A, K, I, J, N);
    FW_SSE(A, K, I, J, N);

    // for (k = K; k < K + N; k++)
    // for (i = I; i < I + N; i++)
    // for (j = J; j < J + N; j++) {
    // A[i][j] = min(A[i][j], A[i][k] + A[k][j]);
    // }
}
#include <immintrin.h>

void FW_SSE(int** A, int K, int I, int J, int N) {
    for (int k = K; k < K + N; k++) {
        for (int i = I; i < I + N; i++) {
            __m128i a_ik = _mm_set1_epi32(A[i][k]); // Broadcast A[i][k] across all elements

            int j = J;
            for (; j <= J + N - 32; j += 32) { // Unroll by 8 (8 * 4 = 32 elements)

                // Load blocks of A[i][j:j+31] and A[k][j:j+31]
                __m128i a_ij0 = _mm_loadu_si128((__m128i*)&A[i][j]);
                __m128i a_ij1 = _mm_loadu_si128((__m128i*)&A[i][j + 4]);
                __m128i a_ij2 = _mm_loadu_si128((__m128i*)&A[i][j + 8]);
                __m128i a_ij3 = _mm_loadu_si128((__m128i*)&A[i][j + 12]);
                __m128i a_ij4 = _mm_loadu_si128((__m128i*)&A[i][j + 16]);
                __m128i a_ij5 = _mm_loadu_si128((__m128i*)&A[i][j + 20]);
                __m128i a_ij6 = _mm_loadu_si128((__m128i*)&A[i][j + 24]);
                __m128i a_ij7 = _mm_loadu_si128((__m128i*)&A[i][j + 28]);

                __m128i a_kj0 = _mm_loadu_si128((__m128i*)&A[k][j]);
                __m128i a_kj1 = _mm_loadu_si128((__m128i*)&A[k][j + 4]);
                __m128i a_kj2 = _mm_loadu_si128((__m128i*)&A[k][j + 8]);
                __m128i a_kj3 = _mm_loadu_si128((__m128i*)&A[k][j + 12]);
                __m128i a_kj4 = _mm_loadu_si128((__m128i*)&A[k][j + 16]);
                __m128i a_kj5 = _mm_loadu_si128((__m128i*)&A[k][j + 20]);
                __m128i a_kj6 = _mm_loadu_si128((__m128i*)&A[k][j + 24]);
                __m128i a_kj7 = _mm_loadu_si128((__m128i*)&A[k][j + 28]);

                // Compute A[i][k] + A[k][j:j+31]
                __m128i sum0 = _mm_add_epi32(a_ik, a_kj0);
                __m128i sum1 = _mm_add_epi32(a_ik, a_kj1);
                __m128i sum2 = _mm_add_epi32(a_ik, a_kj2);
                __m128i sum3 = _mm_add_epi32(a_ik, a_kj3);
                __m128i sum4 = _mm_add_epi32(a_ik, a_kj4);
                __m128i sum5 = _mm_add_epi32(a_ik, a_kj5);
                __m128i sum6 = _mm_add_epi32(a_ik, a_kj6);
                __m128i sum7 = _mm_add_epi32(a_ik, a_kj7);

                // Compute the minimum values
                __m128i min_val0 = _mm_min_epi32(a_ij0, sum0);
                __m128i min_val1 = _mm_min_epi32(a_ij1, sum1);
                __m128i min_val2 = _mm_min_epi32(a_ij2, sum2);
                __m128i min_val3 = _mm_min_epi32(a_ij3, sum3);
                __m128i min_val4 = _mm_min_epi32(a_ij4, sum4);
                __m128i min_val5 = _mm_min_epi32(a_ij5, sum5);
                __m128i min_val6 = _mm_min_epi32(a_ij6, sum6);
                __m128i min_val7 = _mm_min_epi32(a_ij7, sum7);

                // Store the results back to A[i][j]
                _mm_storeu_si128((__m128i*)&A[i][j], min_val0);
                _mm_storeu_si128((__m128i*)&A[i][j + 4], min_val1);
                _mm_storeu_si128((__m128i*)&A[i][j + 8], min_val2);
                _mm_storeu_si128((__m128i*)&A[i][j + 12], min_val3);
                _mm_storeu_si128((__m128i*)&A[i][j + 16], min_val4);
                _mm_storeu_si128((__m128i*)&A[i][j + 20], min_val5);
                _mm_storeu_si128((__m128i*)&A[i][j + 24], min_val6);
                _mm_storeu_si128((__m128i*)&A[i][j + 28], min_val7);
            }

            // Handle remaining elements (if N is not a multiple of 32)
            for (; j < J + N; j++) {
                A[i][j] = A[i][j] < (A[i][k] + A[k][j]) ? A[i][j] : (A[i][k] + A[k][j]);
            }
        }
    }
}


// void FW_AVX1(int** A, int K, int I, int J, int N) {
//     int k, i, j;
//     // __m256 _msb_mask = (__m256)_mm256_set1_epi32(0x80000000);
//     __m256 _msb_mask = _mm256_set1_ps(-1.0f);

//     for (k = K; k < K + N; k++) {
//         for (i = I; i < I + N; i++) {
//             __m256i a_ik = _mm256_set1_epi32(A[i][k]); // Broadcast A[i][k] across all elements

//             j = J;
//             for (; j <= J + N - 32; j += 32) { // Unroll by 4 (4 * 8 = 32 elements)
//                 // Load blocks of A[i][j:j+31] and A[k][j:j+31]

//                 // This will enforce the compiler to use lea and YMMWORD PTR,
//                 // instead of using 4 moves intersecting the avx stores
//                 void* const a_addr_ij0 = &A[i][j];
//                 void* const a_addr_ij1 = &A[i][j + 8];
//                 void* const a_addr_ij2 = &A[i][j + 16];
//                 void* const a_addr_ij3 = &A[i][j + 24];

//                 void* const a_addr_kj0 = &A[k][j];
//                 void* const a_addr_kj1 = &A[k][j + 8];
//                 void* const a_addr_kj2 = &A[k][j + 16];
//                 void* const a_addr_kj3 = &A[k][j + 24];


//                 __m256i a_ij0 = _mm256_loadu_si256((__m256i*)a_addr_ij0);
//                 __m256i a_ij1 = _mm256_loadu_si256((__m256i*)a_addr_ij1);
//                 __m256i a_ij2 = _mm256_loadu_si256((__m256i*)a_addr_ij2);
//                 __m256i a_ij3 = _mm256_loadu_si256((__m256i*)a_addr_ij3);

//                 __m256i a_kj0 = _mm256_loadu_si256((__m256i*)a_addr_kj0);
//                 __m256i a_kj1 = _mm256_loadu_si256((__m256i*)a_addr_kj1);
//                 __m256i a_kj2 = _mm256_loadu_si256((__m256i*)a_addr_kj2);
//                 __m256i a_kj3 = _mm256_loadu_si256((__m256i*)a_addr_kj3);

//                 // Compute A[i][k] + A[k][j:j+31]
//                 __m256i sum0 = _mm256_add_epi32(a_ik, a_kj0);
//                 __m256i sum1 = _mm256_add_epi32(a_ik, a_kj1);
//                 __m256i sum2 = _mm256_add_epi32(a_ik, a_kj2);
//                 __m256i sum3 = _mm256_add_epi32(a_ik, a_kj3);

//                 // Compute the minimum values, using poor man's min
//                 __m256i sub_val1 = _mm256_sub_epi32(sum0, a_ij0);
//                 __m256i sub_val2 = _mm256_sub_epi32(sum1, a_ij1);
//                 __m256i sub_val3 = _mm256_sub_epi32(sum2, a_ij2);
//                 __m256i sub_val4 = _mm256_sub_epi32(sum3, a_ij3);

//                 __m256i cmp1 = (__m256i)_mm256_and_ps((__m256)sub_val1, _msb_mask);
//                 __m256i cmp2 = (__m256i)_mm256_and_ps((__m256)sub_val2, _msb_mask);
//                 __m256i cmp3 = (__m256i)_mm256_and_ps((__m256)sub_val3, _msb_mask);
//                 __m256i cmp4 = (__m256i)_mm256_and_ps((__m256)sub_val4, _msb_mask);

//                 // Store the results back to A[i][j]
//                 _mm256_maskstore_epi32(a_addr_ij0, cmp1, sum0);
//                 _mm256_maskstore_epi32(a_addr_ij1, cmp2, sum1);
//                 _mm256_maskstore_epi32(a_addr_ij2, cmp3, sum2);
//                 _mm256_maskstore_epi32(a_addr_ij3, cmp4, sum3);
//             }

//             // Handle remaining elements (if N is not a multiple of 32)
//             for (; j < J + N; j++) {
//                 A[i][j] = min(A[i][j], A[i][k] + A[k][j]);
//             }
//         }
//     }
// }

// void FW_AVX2(int** A, int K, int I, int J, int N) {
//     for (int k = K; k < K + N; k++) {
//         for (int i = I; i < I + N; i++) {
//             __m256i a_ik = _mm256_set1_epi32(A[i][k]); // Broadcast A[i][k] across all elements

//             int j = J;
//             for (; j <= J + N - 32; j += 32) { // Unroll by 4 (4 * 8 = 32 elements)
//                 // Load blocks of A[i][j:j+31] and A[k][j:j+31]

//                 // This will enforce the compiler to use lea and YMMWORD PTR,
//                 // instead of using 4 moves intersecting the avx stores
//                 void* const a_addr_ij0 = &A[i][j];
//                 void* const a_addr_ij1 = &A[i][j + 8];
//                 void* const a_addr_ij2 = &A[i][j + 16];
//                 void* const a_addr_ij3 = &A[i][j + 24];

//                 void* const a_addr_kj0 = &A[k][j];
//                 void* const a_addr_kj1 = &A[k][j + 8];
//                 void* const a_addr_kj2 = &A[k][j + 16];
//                 void* const a_addr_kj3 = &A[k][j + 24];


//                 __m256i a_ij0 = _mm256_loadu_si256((__m256i*)a_addr_ij0);
//                 __m256i a_ij1 = _mm256_loadu_si256((__m256i*)a_addr_ij1);
//                 __m256i a_ij2 = _mm256_loadu_si256((__m256i*)a_addr_ij2);
//                 __m256i a_ij3 = _mm256_loadu_si256((__m256i*)a_addr_ij3);

//                 __m256i a_kj0 = _mm256_loadu_si256((__m256i*)a_addr_kj0);
//                 __m256i a_kj1 = _mm256_loadu_si256((__m256i*)a_addr_kj1);
//                 __m256i a_kj2 = _mm256_loadu_si256((__m256i*)a_addr_kj2);
//                 __m256i a_kj3 = _mm256_loadu_si256((__m256i*)a_addr_kj3);

//                 // Compute A[i][k] + A[k][j:j+31]
//                 __m256i sum0 = _mm256_add_epi32(a_ik, a_kj0);
//                 __m256i sum1 = _mm256_add_epi32(a_ik, a_kj1);
//                 __m256i sum2 = _mm256_add_epi32(a_ik, a_kj2);
//                 __m256i sum3 = _mm256_add_epi32(a_ik, a_kj3);

//                 // Compute the minimum values
//                 __m256i min_val0 = _mm256_min_epi32(a_ij0, sum0);
//                 __m256i min_val1 = _mm256_min_epi32(a_ij1, sum1);
//                 __m256i min_val2 = _mm256_min_epi32(a_ij2, sum2);
//                 __m256i min_val3 = _mm256_min_epi32(a_ij3, sum3);

//                 // Store the results back to A[i][j]
//                 _mm256_storeu_si256((__m256i*)a_addr_ij0, min_val0);
//                 _mm256_storeu_si256((__m256i*)a_addr_ij1, min_val1);
//                 _mm256_storeu_si256((__m256i*)a_addr_ij2, min_val2);
//                 _mm256_storeu_si256((__m256i*)a_addr_ij3, min_val3);
//             }

//             // Handle remaining elements (if N is not a multiple of 32)
//             for (; j < J + N; j++) {
//                 A[i][j] = min(A[i][j], A[i][k] + A[k][j]);
//             }
//         }
//     }
// }
