#include <iostream>
#include <cmath>
#include <omp.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

//Sobel Filter using 1D arrays
void applyEdgeDetection(unsigned char* inputImage, unsigned char* outputEdges, int width, int height) 
{
    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    cout << "Starting OpenMP Edge Detection with " << omp_get_max_threads() << " threads..." << endl;

    //parallelize the outer loop
    #pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    // Convert 2D coordinates to 1D array
                    int pixelIndex = ((y + i) * width) + (x + j);
                    int pixelValue = inputImage[pixelIndex];

                    sumX += pixelValue * Gx[i + 1][j + 1];
                    sumY += pixelValue * Gy[i + 1][j + 1];
                }
            }

            int magnitude = sqrt((sumX * sumX) + (sumY * sumY));
            if (magnitude > 255) magnitude = 255;

            // Store the result
            outputEdges[(y * width) + x] = (unsigned char)magnitude;
        }
    }
}

// This is YOUR territory. Soon Yik doesn't need to touch this.
void runParallelIntegration(unsigned char* fullImage, int width, int height, int rank, int size) {

    // 1. Calculate how to slice the banana
    int rows_per_rank = height / size;
    int chunk_size = width * rows_per_rank;

    // 2. Every rank (worker) needs a small local bucket to hold their slice
    unsigned char* localInput = new unsigned char[chunk_size];
    unsigned char* localOutput = new unsigned char[chunk_size];

    // 3. SCATTER: Rank 0 cuts the image and sends pieces to everyone else
    // If rank != 0, fullImage is ignored, and they just wait to receive.
    MPI_Scatter(fullImage, chunk_size, MPI_UNSIGNED_CHAR,
        localInput, chunk_size, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // 4. THE HAND-OFF: Call Soon Yik's code on the local chunk
    // Notice we pass 'rows_per_rank' as the height!
    applyEdgeDetection(localInput, localOutput, width, rows_per_rank);

    // 5. GATHER: Rank 0 collects all the finished slices back into fullImage
    MPI_Gather(localOutput, chunk_size, MPI_UNSIGNED_CHAR,
        fullImage, chunk_size, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Clean up local memory
    delete[] localInput;
    delete[] localOutput;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, channels;
    unsigned char* imgData = NULL;

    // Only Rank 0 loads the file
    if (rank == 0) {
        imgData = stbi_load("banana.jpg", &width, &height, &channels, 1);
        if (!imgData) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // Broadcast dimensions so everyone knows the "slice" math
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // RUN YOUR WRAPPER
    runParallelIntegration(imgData, width, height, rank, size);

    // Only Rank 0 saves the final combined result
    if (rank == 0) {
        stbi_write_jpg("output_edges_parallel.jpg", width, height, 1, imgData, 100);
        stbi_image_free(imgData);
    }

    MPI_Finalize();

    //unsigned char* imgData = stbi_load("banana.jpg", &width, &height, &channels, 1);

    //if (imgData == NULL) {
    //    cout << "Error loading banana.jpg." << endl;
    //    return 1;
    //}
    //cout << "Loaded image: " << width << "x" << height << " pixels." << endl;

    //size_t imgSize = width * height;
    //unsigned char* outputData = new unsigned char[imgSize];
    //
    //// Initialize output array to pure black (0)
    //for (size_t i = 0; i < imgSize; ++i) {
    //    outputData[i] = 0;
    //}

    //omp_set_num_threads(4);
    //double startTime = omp_get_wtime();
    //applyEdgeDetection(imgData, outputData, width, height);
    //double endTime = omp_get_wtime();
    //cout << "Processing Time: " << (endTime - startTime) << " seconds." << endl;

    ////Save result
    //int writeSuccess = stbi_write_jpg("output_edges.jpg", width, height, 1, outputData, 100);

    //if (writeSuccess) {
    //    cout << "Successfully saved edge detection result to 'output_edges.jpg'." << endl;
    //}
    //else {
    //    cout << "Failed to write the output image." << endl;
    //}

    ////Clean up memory
    //stbi_image_free(imgData);
    //delete[] outputData;

    return 0;
}

//#include <mpi.h>
//#include <iostream>
//
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    std::cout << "Hello from Rank " << world_rank << " out of " << world_size << " processors!" << std::endl;
//
//    MPI_Finalize();
//    return 0;
//}