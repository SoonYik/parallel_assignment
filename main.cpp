#include <iostream>
#include <cmath>
#include <omp.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

//Gaussian Blur
void applyGaussianBlur(unsigned char* inputImage, unsigned char* outputImage, int width, int height)
{
    cout << "Running Step 1: Gaussian Blur..." << endl;

    int kernel[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    int kernelWeight = 273;

#pragma omp parallel for schedule(dynamic)
    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < width - 2; ++x) {
            int sum = 0;
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    // Convert 2D coordinates to the 1D array index
                    int pixelIndex = ((y + i) * width) + (x + j);
                    sum += inputImage[pixelIndex] * kernel[i + 2][j + 2];
                }
            }
            outputImage[(y * width) + x] = (unsigned char)(sum / kernelWeight);
        }
    }
}

//Sobel Filter
void applyEdgeDetection(unsigned char* inputImage, unsigned char* outputEdges, float* outputAngles, int width, int height)
{
    cout << "Running Step 2: Edge Detection..." << endl;

    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    //parallelize the outer loop
#pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int pixelIndex = ((y + i) * width) + (x + j);
                    int pixelValue = inputImage[pixelIndex];

                    sumX += pixelValue * Gx[i + 1][j + 1];
                    sumY += pixelValue * Gy[i + 1][j + 1];
                }
            }

            int magnitude = sqrt((sumX * sumX) + (sumY * sumY));
            float angle = atan2(sumY, sumX) * (180.0 / 3.14159265);
            if (angle < 0) {
                angle += 180.0;
            }
            outputAngles[(y * width) + x] = angle;
            if (magnitude > 255) magnitude = 255;
            outputEdges[(y * width) + x] = (unsigned char)magnitude;
        }
    }
}

//Non-Maximum Suppression
void applyNonMaxSuppression(unsigned char* magnitudeImg, float* angleImg, unsigned char* outputImg, int width, int height)
{
    cout << "Running Step 3: Non-Maximum Suppression..." << endl;
#pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = (y * width) + x;
            float angle = angleImg[idx];
            unsigned char mag = magnitudeImg[idx];
            unsigned char pixel1 = 255;
            unsigned char pixel2 = 255;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                pixel1 = magnitudeImg[y * width + (x + 1)];
                pixel2 = magnitudeImg[y * width + (x - 1)];
            }
            else if (angle >= 22.5 && angle < 67.5) {
                pixel1 = magnitudeImg[(y + 1) * width + (x - 1)];
                pixel2 = magnitudeImg[(y - 1) * width + (x + 1)];
            }
            else if (angle >= 67.5 && angle < 112.5) {
                pixel1 = magnitudeImg[(y + 1) * width + x];
                pixel2 = magnitudeImg[(y - 1) * width + x];
            }
            else if (angle >= 112.5 && angle < 157.5) {
                pixel1 = magnitudeImg[(y - 1) * width + (x - 1)];
                pixel2 = magnitudeImg[(y + 1) * width + (x + 1)];
            }

            if (mag >= pixel1 && mag >= pixel2) {
                outputImg[idx] = mag;
            }
            else {
                outputImg[idx] = 0;
            }
        }
    }
}

// Finalize edges using High and Low thresholds
void applyHysteresis(unsigned char* inputImg, unsigned char* outputImg, int width, int height, int lowThresh, int highThresh)
{
    cout << "Running Step 4: Hysteresis Thresholding..." << endl;

    unsigned char* tempThresh = new unsigned char[width * height];

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < width * height; ++i) {
        if (inputImg[i] >= highThresh) {
            tempThresh[i] = 255;
            outputImg[i] = 255;
        }
        else if (inputImg[i] >= lowThresh) {
            tempThresh[i] = 50;
        }
        else {
            tempThresh[i] = 0;
            outputImg[i] = 0;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = (y * width) + x;
            if (tempThresh[idx] == 50) {
                bool touchesStrongEdge = false;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        int neighborIdx = ((y + i) * width) + (x + j);
                        if (tempThresh[neighborIdx] == 255) {
                            touchesStrongEdge = true;
                        }
                    }
                }
                if (touchesStrongEdge) {
                    outputImg[idx] = 255;
                }
                else {
                    outputImg[idx] = 0;
                }
            }
        }
    }

    delete[] tempThresh; // Clean up memory
}

void runHybridProcessing(unsigned char* fullImage, int width, int height, int rank, int size) {

    int rows_per_rank = height / size;
    int chunk_size = width * rows_per_rank;

    unsigned char* localInput = new unsigned char[chunk_size];
    unsigned char* localBlur = new unsigned char[chunk_size];
    unsigned char* localSobel = new unsigned char[chunk_size];
    float* localAngles = new float[chunk_size];
    unsigned char* localNms = new unsigned char[chunk_size];
    unsigned char* localOutput = new unsigned char[chunk_size];

    MPI_Scatter(fullImage, chunk_size, MPI_UNSIGNED_CHAR,
        localInput, chunk_size, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // 4. THE HAND-OFF: Call Soon Yik's code on the local chunk
    // Notice we pass 'rows_per_rank' as the height!
    applyGaussianBlur(localInput, localBlur, width, rows_per_rank);
    applyEdgeDetection(localBlur, localSobel, localAngles, width, rows_per_rank);
    applyNonMaxSuppression(localSobel, localAngles, localNms, width, rows_per_rank);
    applyHysteresis(localNms, localOutput, width, rows_per_rank, 50, 150);

    // 5. GATHER: Rank 0 collects all the finished slices back into fullImage
    MPI_Gather(localOutput, chunk_size, MPI_UNSIGNED_CHAR,
        fullImage, chunk_size, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Clean up local memory
    delete[] localInput; delete[] localBlur; delete[] localSobel;
    delete[] localAngles; delete[] localNms; delete[] localOutput;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, channels;
    unsigned char* rawImg = NULL;

    // Only Rank 0 loads the image
    if (rank == 0) {
        rawImg = stbi_load("road.jpg", &width, &height, &channels, 1);
        if (!rawImg) {
            cout << "Failed to load image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Share dimensions with all ranks
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = omp_get_wtime();

    // Run your hybrid orchestration
    runHybridProcessing(rawImg, width, height, rank, size);

    if (rank == 0) {
        double endTime = omp_get_wtime();
        cout << "Total Parallel Execution time: " << (endTime - startTime) << " seconds" << endl;
        stbi_write_jpg("4_final_edges.jpg", width, height, 1, rawImg, 100);
        stbi_image_free(rawImg);
    }

    MPI_Finalize();
    return 0;
}

//int main() {
//    int width, height, channels;
//    unsigned char* rawImg = stbi_load("road.jpg", &width, &height, &channels, 1);
//    size_t imgSize = width * height;
//    unsigned char* blurredImg = new unsigned char[imgSize];
//    float* angleImg = new float[imgSize];
//    unsigned char* sobelImg = new unsigned char[imgSize];
//    unsigned char* nmsImg = new unsigned char[imgSize];
//    unsigned char* finalImg = new unsigned char[imgSize];
//
//    double startTime = omp_get_wtime();
//
//    //1: Gaussian Blur
//    applyGaussianBlur(rawImg, blurredImg, width, height);
//    stbi_write_jpg("1_blurred.jpg", width, height, 1, blurredImg, 100);
//
//    //2: Sobel Filter
//    applyEdgeDetection(blurredImg, sobelImg, angleImg, width, height);
//    stbi_write_jpg("2_sobel_edges.jpg", width, height, 1, sobelImg, 100);
//
//    //3: Non-Maximum Suppression
//    applyNonMaxSuppression(sobelImg, angleImg, nmsImg, width, height);
//    stbi_write_jpg("3_nms_thinned.jpg", width, height, 1, nmsImg, 100);
//
//    //4: Hysteresis Thresholding
//    applyHysteresis(nmsImg, finalImg, width, height, 50, 150);
//    stbi_write_jpg("4_final_edges.jpg", width, height, 1, finalImg, 100);
//
//    double endTime = omp_get_wtime();
//    cout << "Total execution time: " << (endTime - startTime) << " seconds" << endl;
//
//    // Clean up memory
//    stbi_image_free(rawImg);
//    delete[] blurredImg;
//    delete[] sobelImg;
//    delete[] nmsImg;
//    delete[] finalImg;
//    delete[] angleImg;
//
//    return 0;
//}
