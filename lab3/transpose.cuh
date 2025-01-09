#pragma once

static constexpr int TILE_DIM   = 32;
static constexpr int BLOCK_ROWS = 16;


template<typename T> __global__ void iptransposeCoalesced(T* data) {
    __shared__ T tile_s[TILE_DIM][TILE_DIM + 1];
    __shared__ T tile_d[TILE_DIM][TILE_DIM + 1];

    int x     = blockIdx.x * TILE_DIM + threadIdx.x;
    int y     = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
        int dx = blockIdx.y * TILE_DIM + threadIdx.x;
        int dy = blockIdx.x * TILE_DIM + threadIdx.y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_d[threadIdx.y + j][threadIdx.x] = data[(dy + j) * width + dx];
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(dy + j) * width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(y + j) * width + x] = tile_d[threadIdx.x][threadIdx.y + j];
    }

    else if (blockIdx.y == blockIdx.x) { // handle on-diagonal case
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            data[(y + j) * width + x] = tile_s[threadIdx.x][threadIdx.y + j];
    }
}

template<typename T> __host__ void transpose_inplace(T* data, const size_t M, const size_t N) {
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(N / TILE_DIM, M / TILE_DIM);

    iptransposeCoalesced<T><<<blocks, threads>>>(data);
}
