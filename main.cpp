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
    //cout << "Running Step 1: Gaussian Blur..." << endl;

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
    //cout << "Running Step 2: Edge Detection..." << endl;
    
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
    //cout << "Running Step 3: Non-Maximum Suppression..." << endl;
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
    //cout << "Running Step 4: Hysteresis Thresholding..." << endl;

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

void runHybridProcessing(unsigned char* fullVideoBuffer, int width, int height, int totalFrames, int rank, int size) {

    int frames_per_rank = totalFrames / size;
    size_t frame_size = (size_t)width * height;
    size_t batch_size = frame_size * frames_per_rank;

    // Buffer to hold the frames assigned to this rank
    unsigned char* localBatchInput = new unsigned char[batch_size];
    unsigned char* localBatchOutput = new unsigned char[batch_size];

    // 1. Distribute the frames
    MPI_Scatter(fullVideoBuffer, batch_size, MPI_UNSIGNED_CHAR,
        localBatchInput, batch_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 2. The Worker Loop: Process each frame in the batch
    for (int f = 0; f < frames_per_rank; ++f)
    {
        unsigned char* currentFrameIn = &localBatchInput[f * frame_size];
        unsigned char* currentFrameOut = &localBatchOutput[f * frame_size];

        // Temporary buffers for OpenMP pipeline
        unsigned char* tempBlur = new unsigned char[frame_size];
        unsigned char* tempSobel = new unsigned char[frame_size];
        unsigned char* tempNms = new unsigned char[frame_size];
        float* tempAngles = new float[frame_size];

        // Call the pipeline
        applyGaussianBlur(currentFrameIn, tempBlur, width, height);
        applyEdgeDetection(tempBlur, tempSobel, tempAngles, width, height);
        applyNonMaxSuppression(tempSobel, tempAngles, currentFrameOut/*tempNms*/, width, height);
        //applyHysteresis(tempNms, currentFrameOut, width, height, 50, 150);

        delete[] tempBlur; delete[] tempSobel; delete[] tempAngles; delete[] tempNms;
    }

    // 3. Gather the processed frames
    MPI_Gather(localBatchOutput, batch_size, MPI_UNSIGNED_CHAR,
        fullVideoBuffer, batch_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    delete[] localBatchInput;
    delete[] localBatchOutput;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totalFrames = 0;
    int width = 0, height = 0, channels = 0;
    unsigned char* videoBuffer = NULL;

    if (rank == 0)
    {
        while (true)
        {
            char filename[64];
            sprintf(filename, "frames/%05d.jpg", totalFrames + 1);
            if (stbi_info(filename, &width, &height, &channels))
            {
                totalFrames++;
            }
            else {
                break;
            }
        }

        if (totalFrames == 0)
        {
            cout << "Error: No frames found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        totalFrames = (totalFrames / size) * size;

        cout << "Detected " << totalFrames << " frames. Resolution: " << width << "x" << height << endl;

        size_t totalSize = (size_t)width * height * totalFrames;
        videoBuffer = new unsigned char[totalSize];

        for (int i = 0; i < totalFrames; i++)
        {
            char filename[64];
            sprintf(filename, "frames/%05d.jpg", i + 1);
            int w, h, c;
            unsigned char* data = stbi_load(filename, &w, &h, &c, 1);
            memcpy(videoBuffer + ((size_t)i * width * height), data, (size_t)width * height);
            stbi_image_free(data);
        }
    }

    // 4. Share dimensions so all workers can prepare their local memory
    MPI_Bcast(&totalFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = omp_get_wtime();

    // 5. Run the processing
    runHybridProcessing(videoBuffer, width, height, totalFrames, rank, size);

    if (rank == 0) {
        double endTime = omp_get_wtime();
        cout << "Total Parallel Execution time: " << (endTime - startTime) << " seconds" << endl;
        cout << "Total Parallel Execution time per frame: " << (endTime - startTime)/ totalFrames << " seconds" << endl;
        // 6. Save results
        for (int i = 0; i < totalFrames; i++) {
            char outName[64];
            sprintf(outName, "output/processed_%05d.jpg", i + 1);
            stbi_write_jpg(outName, width, height, 1, videoBuffer + ((size_t)i * width * height), 100);
        }

        delete[] videoBuffer;
    }

    MPI_Finalize();
    return 0;
}