#include "non_max_supp.h"
//#include <stdio.h>

// This is a helper function to test for max.
// If center is the max point, return 0. Return center otherwise
__host__ __device__ float maxSupp(float center, float p1, float p2, float p3 = -1.0, float p4 = -1.0)
{
    if (center >= p1 && center >= p2 && center >= p3 && center >= p4)
    {
        return center;
    }
    else
    {
        return 0.0;
    }
}

// This is a helper function to test for bounds within the image
// If the pixel indicies are within the image dimensions, return the image at that location
// Return 0 otherwise.
__host__ __device__ float getPoint(float *img, int cIdx, int rIdx, int height, int width)
{
    if (!img || rIdx < 0 || rIdx >= height || cIdx < 0 || cIdx >= width)
    {
        return 0.0;
    }
    return *(img + cIdx + rIdx * width);
}

// Serial implementation of the non-maximum supression funciont
void nms(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
    // FILE *quantFile = fopen("quantNms.txt", "w");

    // Loop through the image pixel by pixel
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            // Determine the gradient angle for the current pixel
            float angle = *(gradImg + j * width + i);
            float p1 = -1.0; //, p3 = -1.0;
            float p2 = -1.0; //, p4 = -1.0;
            unsigned int fAngle = 0;
            if (angle > 180)
            {
                angle = angle - 180;
            }

            // Quantize the gradient angle at this pixel location
            if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
                fAngle = 0;
            else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
                fAngle = 135;
            else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
                fAngle = 90;
            else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
                fAngle = 45;

            // fprintf(quantFile, "%d,", fAngle);

            // Based on the quantized gradient angle, select neighboring pixels
            // to compare the current pixel against to see if it's a local maximum
            //
            // The getPoint returns a min value if the pixel falls out-of-bounds
            switch (fAngle)
            {
            case 0:
                p1 = getPoint(inImg, i, j + 1, height, width);
                p2 = getPoint(inImg, i, j - 1, height, width);
                // p3 = getPoint(inImg, i, j+2, height, width);
                // p4 = getPoint(inImg, i, j-2, height, width);
                break;
            case 45:
                p1 = getPoint(inImg, i - 1, j - 1, height, width);
                p2 = getPoint(inImg, i + 1, j + 1, height, width);
                // p3 = getPoint(inImg, i-2, j-2, height, width);
                // p4 = getPoint(inImg, i+2, j+2, height, width);
                break;
            case 90:
                p1 = getPoint(inImg, i + 1, j, height, width);
                p2 = getPoint(inImg, i - 1, j, height, width);
                // p3 = getPoint(inImg, i+2, j, height, width);
                // p4 = getPoint(inImg, i-2, j, height, width);
                break;
            case 135:
                p1 = getPoint(inImg, i + 1, j - 1, height, width);
                p2 = getPoint(inImg, i - 1, j + 1, height, width);
                // p3 = getPoint(inImg, i+2, j-2, height, width);
                // p4 = getPoint(inImg, i-2, j+2, height, width);
                break;
            default:
                break;
            }

            // Get the center point
            float center = getPoint(inImg, i, j, height, width);
            //*(nmsImg + i + j*width) = maxSupp(center, p1, p2, p3, p4);

            // If center is a local max, output center. Set output pixel to zero otherwise
            *(nmsImg + i + j * width) = maxSupp(center, p1, p2); 
        }
        // fprintf(quantFile, "\n");
    }
    // fclose(quantFile);
}

// Naive implementation of the non-maximum supression kernel
// Kernel assumes a thread will be generated for each pixel in the image. 
__global__ void nms_global(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
    // Determine thread location within the image
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;

    // Initialize default points for testing against
    float p1 = -1.0; //, p3 = -1.0;
    float p2 = -1.0; //, p4 = -1.0;
    unsigned int fAngle = 0;

    // Thread will only participate if it is in the image
    if (col < width && row < height) // Since size_t is unsigned, it can't fall below 0
    {
        // Get the gradient angle for the pixel for this thread from global memory
        float angle = *(gradImg + row * width + col);

        // Quantize gradient angle based on nearest neighbor angle
        if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
            fAngle = 0;
        else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
            fAngle = 135;
        else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
            fAngle = 90;
        else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
            fAngle = 45;

        // Based on the quantized gradient angle, select neighboring pixels
        // to compare the current pixel against to see if it's a local maximum
        // These pixels will read from global memory.
        //
        // The getPoint returns a min value if the pixel falls out-of-bounds
        switch (fAngle)
        {
            case 0:
                p1 = getPoint(inImg, col, row + 1, height, width);
                p2 = getPoint(inImg, col, row - 1, height, width);
                // p3 = getPoint(inImg, col, row+2, height, width);
                // p4 = getPoint(inImg, col, row-2, height, width);
                break;
            case 45:
                p1 = getPoint(inImg, col - 1, row - 1, height, width);
                p2 = getPoint(inImg, col + 1, row + 1, height, width);
                // p3 = getPoint(inImg, col-2, row-2, height, width);
                // p4 = getPoint(inImg, col+2, row+2, height, width);
                break;
            case 90:
                p1 = getPoint(inImg, col + 1, row, height, width);
                p2 = getPoint(inImg, col - 1, row, height, width);
                // p3 = getPoint(inImg, col+2, row, height, width);
                // p4 = getPoint(inImg, col-2, row, height, width);
                break;
            case 135:
                p1 = getPoint(inImg, col + 1, row - 1, height, width);
                p2 = getPoint(inImg, col - 1, row + 1, height, width);
                // p3 = getPoint(inImg, col+2, row-2, height, width);
                // p4 = getPoint(inImg, col-2, row+2, height, width);
                break;
            default:
                break;
        }

        // Get the center point
        float center = getPoint(inImg, col, row, height, width);
        //*(nmsImg + i + j*width) = maxSupp(center, p1, p2, p3, p4);

        // If center is a local max, output center. Set output pixel to zero otherwise
        *(nmsImg + col + row * width) = maxSupp(center, p1, p2);
    }
}

// Optimized implementation of the non-maximum supression kernel
// Kernel assumes a thread will be generated for each pixel in the image. 
// Thread blocks will be 16x16
__global__ void nms_opt(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
    // Calculate global index position within the image
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;

    // Initialize default points for testing against
    float p1 = -1.0; //, p3 = -1.0;
    float p2 = -1.0; //, p4 = -1.0;

    // Define tile size for reading the local image into shared memory
    const size_t TILE_SIZE = 16;
    const size_t P_IMG_SIZE = TILE_SIZE + 2; // Handle overrun on edges

    // Expand tile for magnitude image to account for local neighbors
    // still within the image
    __shared__ float pImage[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ float pAngle[TILE_SIZE][TILE_SIZE];

    // Read image gradient angle and magnitude into shared memory tile
    // Need to include threads that fall outside of image dimensions for reading in tile
    if (col < width + 2 && row < height + 2) // Since size_t is unsigned, it can't fall below 0
    {
        // Read in magnitude
        // Since there are only 16x16 threads, but the tile is 18x18 - stride some threads
        // to read in multiple pixels
        for (size_t i = 0; threadIdx.x + i < P_IMG_SIZE; i += TILE_SIZE)
        {
            for (size_t j = 0; threadIdx.y + j < P_IMG_SIZE; j += TILE_SIZE)
            {
                // Shift input image up and to the left 1 to account for neighbors along the tile
                pImage[threadIdx.x + i][threadIdx.y + j] = getPoint(inImg, col + i - 1, row + j - 1, height, width);
            }
        }

        // Read in gradient angle tile - this is only 16x16
        pAngle[threadIdx.x][threadIdx.y] = getPoint(gradImg, col, row, height, width); // gradImg[row*width + col];
    }
    __syncthreads(); // Finish memory read prior to continuing

    // Only threads witin the image participate in the image supression
    if (col < width && row < height) // Since size_t is unsigned, it can't fall below 0
    {
        // Read in the gradient angle from shared memory
        float angle = pAngle[threadIdx.x][threadIdx.y]; //*(gradImg + row*width + col);
        
        // Determine magnitude index witin tile
        // This +1 accounts for the image shift done at tile read
        size_t i = threadIdx.x + 1;
        size_t j = threadIdx.y + 1;

        // Read in the pixels for comparison from shared memory based on quantized angle calculation
        // Note that none of the neighbor pixels should fall outside the tile for gradient magnitude
        if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
        {
            p1 = pImage[i][j + 1];
            p2 = pImage[i][j - 1];
            // p3 = getPoint(inImg, i, j+2, 16, 16);
            // p4 = getPoint(inImg, i, j-2, 16, 16);
        }
        else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
        {
            p1 = pImage[i + 1][j - 1];
            p2 = pImage[i - 1][j + 1];
            // p3 = getPoint(inImg, i+2, j-2, 16, 16);
            // p4 = getPoint(inImg, i-2, j+2, 16, 16);
        }
        else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
        {
            p1 = pImage[i + 1][j];
            p2 = pImage[i - 1][j];
            // p3 = getPoint(inImg, i+2, j, 16, 16);
            // p4 = getPoint(inImg, i-2, j, 16, 16);
        }
        else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
        {
            p1 = pImage[i - 1][j - 1];
            p2 = pImage[i + 1][j + 1];
            // p3 = getPoint(inImg, i-2, j-2, 16, 16);
            // p4 = getPoint(inImg, i+2, j+2, 16, 16);
        }

        // Get the center point
        float center = pImage[i][j];

        // If center is a local max, output center. Set output pixel to zero otherwise
        nmsImg[col + row * width] = maxSupp(center, p1, p2);
    }
}

