#include "transpose.cuh"

#include <sys/time.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

double wtime(void) {
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +             // in seconds
               ((double)etstart.tv_usec) / 1000000.0; // in microseconds
    return now_time;
}

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);
    }

    void Stop() {
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

int main(int argc, char** argv) {
    const size_t M    = atoi(argv[1]);
    const size_t N    = atoi(argv[2]);
    const size_t size = M * N * sizeof(double);

    double* data     = (double*)malloc(size);
    double* data_cpu = (double*)malloc(size);

    GpuTimer timer;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            data[i * N + j]     = i * N + j;
            data_cpu[i * N + j] = i * N + j;
        }
    }

    // GPU Transpose
    double* d_data;

    timer.Start();
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    transpose_inplace(d_data, M, N);
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);
    timer.Stop();

    // CPU Transpose
    auto start = wtime();
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < i; j++) {
            auto tmp            = data_cpu[i * N + j];
            data_cpu[i * N + j] = data_cpu[j * M + i];
            data_cpu[j * M + i] = tmp;
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
    }
    auto stop = wtime();


    printf("Transpose test passed\n");
    printf("GPU Time: %.3f ms\n", timer.Elapsed());
    printf("CPU Time: %.3f ms\n", (stop - start) * 1000);

    // Compare
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            if (data[i * N + j] != data_cpu[i * N + j]) {
                printf(
                  "Mismatch at (%d, %d): %f != %f\n", i, j, data[i * N + j], data_cpu[i * N + j]);
                return 1;
            }
        }
    }

    // // print gpu result
    // for (size_t i = 0; i < M; i++) {
    //     for (size_t j = 0; j < N; j++) {
    //         printf("%.0f ", data[i * N + j]);
    //     }
    //     printf("\n");
    // }

    free(data);
    free(data_cpu);
    cudaFree(d_data);

    return 0;
}