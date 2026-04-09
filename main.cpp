#include <iostream>
#include <cmath>
#include <omp.h>

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

int main() 
{
    int width, height, channels;
    unsigned char* imgData = stbi_load("banana.jpg", &width, &height, &channels, 1);

    if (imgData == NULL) {
        cout << "Error loading banana.jpg." << endl;
        return 1;
    }
    cout << "Loaded image: " << width << "x" << height << " pixels." << endl;

    size_t imgSize = width * height;
    unsigned char* outputData = new unsigned char[imgSize];
    
    // Initialize output array to pure black (0)
    for (size_t i = 0; i < imgSize; ++i) {
        outputData[i] = 0;
    }

    omp_set_num_threads(4);
    double startTime = omp_get_wtime();
    applyEdgeDetection(imgData, outputData, width, height);
    double endTime = omp_get_wtime();
    cout << "Processing Time: " << (endTime - startTime) << " seconds." << endl;

    //Save result
    int writeSuccess = stbi_write_jpg("output_edges.jpg", width, height, 1, outputData, 100);

    if (writeSuccess) {
        cout << "Successfully saved edge detection result to 'output_edges.jpg'." << endl;
    }
    else {
        cout << "Failed to write the output image." << endl;
    }

    //Clean up memory
    stbi_image_free(imgData);
    delete[] outputData;

    return 0;
}