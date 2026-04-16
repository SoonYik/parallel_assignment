#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define BLOCK_SIZE (TILE_SIZE + 2 * KERNEL_RADIUS)

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    // Shared memory buffer to hold the tile + halo
    __shared__ unsigned char sharedMem[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x - KERNEL_RADIUS;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y - KERNEL_RADIUS;

    // 1. Load data into Shared Memory (including halos)
    if (x >= 0 && x < width && y >= 0 && y < height)
        sharedMem[threadIdx.y][threadIdx.x] = input[y * width + x];
    else
        sharedMem[threadIdx.y][threadIdx.x] = 0;

    __syncthreads(); // Ensure all threads finished loading

    // 2. Only compute if thread is within the "inner" tile
    if (threadIdx.x >= KERNEL_RADIUS && threadIdx.x < BLOCK_SIZE - KERNEL_RADIUS &&
        threadIdx.y >= KERNEL_RADIUS && threadIdx.y < BLOCK_SIZE - KERNEL_RADIUS) {

        int outX = blockIdx.x * TILE_SIZE + (threadIdx.x - KERNEL_RADIUS);
        int outY = blockIdx.y * TILE_SIZE + (threadIdx.y - KERNEL_RADIUS);

        if (outX < width && outY < height) {
            float sum = 0;
            int kernel[5][5] = { {1, 4, 7, 4, 1}, 
                {4, 16, 26, 16, 4}, 
                {7, 26, 41, 26, 7},
                {4, 16, 26, 16, 4},
                {1,  4,  7,  4, 1} 
            }; // (Truncated for brevity)

            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    sum += sharedMem[threadIdx.y + i][threadIdx.x + j] * kernel[i + 2][j + 2];
                }
            }
            output[outY * width + outX] = (unsigned char)(sum / 273.0f);
            //output[outY * width + outX] = 255;
        }
    }
}

extern "C" void launch_gaussian_blur(unsigned char* h_input, unsigned char* h_output, int w, int h) {

    //cudaError_t err;

    unsigned char* d_in, * d_out;
    size_t size = w * h * sizeof(unsigned char);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((w + TILE_SIZE - 1) / TILE_SIZE, (h + TILE_SIZE - 1) / TILE_SIZE);

   /* err = cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("CUDA Copy In Error: %s\n", cudaGetErrorString(err));*/

    gaussianBlurKernel << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, w, h);

    /*err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));*/

    cudaDeviceSynchronize();

   /* err = cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("CUDA Copy Out Error: %s\n", cudaGetErrorString(err));*/

    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}