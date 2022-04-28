// To calculate the Grayscale image = ColorToGrayscale
// For the gaussian blur = Conv2D
// and the sobel filter which gives the
//  gradient descent = GradientSobel
#include "filters.h"

#define FILTERSIZE 3
#define BLOCKSIZE 16

////////////////////////////////////////////////////////
// Conv2DOpt inputs: inImg: float  grayscale image, the gaussian filter, and the
// image width and height
//and outputs outImg:  blurred image of width*height
// This is a slightly optimized version
/////////////////////////////////////////////////////////
__global__ void Conv2DOpt(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int halfFilter = (int)filterSize/2;

   // boundary check if it's in the image
   if(row >= 0 && row < height && col >= 0 && col < width) {
     float pixelvalue = 0;
     int start_col = col - halfFilter;
     int start_row = row - halfFilter;

     // now do the filtering
     for (int j = 0; j < filterSize; ++j) {
       for (int k = 0; k < filterSize; ++k) {
         int cur_row = start_row + j;
         int cur_col = start_col + k;

         // only count the ones that are inside the boundaries
         if (cur_row >=0 && cur_row < height && cur_col >= 0 && cur_col < width) {
           pixelvalue += inImg[cur_row*width + cur_col] * filter[j * filterSize + k]; //[k][j];
         }

       }
     }
     __threadfence();     
     outImg[row*width + col] = pixelvalue;
   }
>>>>>>> Stashed changes

                // only count the ones that are inside the boundaries
                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                {
                    pixelvalue += inImg[cur_row * width + cur_col] * filter[j * filterSize + k]; //[k][j];
                }
            }
        }
        __threadfence();
        outImg[row * width + col] = pixelvalue;
    }
}

////////////////////////////////////////////////////////
// Conv2DOptRow inputs: inImg: float  grayscale image, the gaussian filter, and the
// image width and height
//and outputs outImg:  a row dimension blurred image of width*height
// This is half of the Conv2D function, needs to be paired with Conv2DOptCol
/////////////////////////////////////////////////////////
__global__ void Conv2DOptRow(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int halfFilter =(int)filterSize/2;

   // boundary check if it's in the image
   if(row >= 0 && row < height && col >= 0 && col < width) {
     float pixelvalue = 0;
     int start_col = col - halfFilter;

    // boundary check if it's in the image
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        float pixelvalue = 0;
        int start_col = col - halfFilter;

        // now do the filtering
        for (int j = 0; j < filterSize; ++j)
        {
            int cur_row = row;
            int cur_col = start_col + j;

            // only count the ones that are inside the boundaries
            if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
            {
                pixelvalue += inImg[cur_row * width + cur_col] * filter[j * filterSize + 1] * 3; //[k][j];
            }
        }
        __threadfence();
        outImg[row * width + col] = pixelvalue;
    }
}

////////////////////////////////////////////////////////
// Conv2DOptCol inputs: inImg: float  row-wise blurred image, the gaussian filter, and the
// image width and height
//and outputs outImg:  a gaussian blurred image of width*height
// This is a partner with the Conv2DOoptRow, which must be run first
/////////////////////////////////////////////////////////
__global__ void Conv2DOptCol(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int halfFilter = 1; //(int)filterSize/2;

    // boundary check if it's in the image
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        float pixelvalue = 0;
        int start_row = row - halfFilter;

        // now do the filtering
        for (int j = 0; j < filterSize; ++j)
        {
            int cur_row = start_row + j;
            int cur_col = col;

            // only count the ones that are inside the boundaries
            if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
            {
                pixelvalue += inImg[cur_row * width + cur_col] * filter[filterSize + j]; //[k][j];
            }
        }
        __threadfence();
        outImg[row * width + col] = pixelvalue;
    }
}

////////////////////////////////////////////////////////
// GradientSobelOpt inputs: inImg: float  grayscale gaussian blurred image, 
// , and the image width and height 
// and outputs sobelImg:  magnitude of the gradients image of width*height
// output: gradientImg: the phase of the gradients in an image of width*height
// This is an unrolled optimized version
/////////////////////////////////////////////////////////
__global__ void GradientSobelOpt(float *inImg, float *sobelImg, float *gradientImg, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // To detect horizontal lines, G_x. 
/*
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    // To detect vertical lines, G_y 
    const int fmat_y[3][3]  = {
        {-1, -2, -1},
        {0,   0,  0},
        {1,   2,  1}
    };
*/
    // now do the filtering
    // halfFitler is how many are on each side

    // now do the filtering
    // halfFitler is how many are on each side
    float sumx = 0;
    float sumy = 0;
    //// DO THE SOBEL FILTERING ///////////

    // boundary check if it's in the image
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        // only count the ones that are inside the boundaries
        if ((row - 1) >= 0)
        {
            sumy += -2 * inImg[(row - 1) * width + col];
            if ((col - 1) >= 0)
            {
                sumy += -1 * inImg[(row - 1) * width + (col - 1)];
                sumx += -1 * inImg[(row - 1) * width + (col - 1)];
                sumx += -2 * inImg[(row)*width + (col - 1)];
            }
            if ((col + 1) < width)
            {
                sumy += -1 * inImg[(row - 1) * width + (col + 1)];
                sumx += 1 * inImg[(row - 1) * width + (col + 1)];
            }
        }
        if ((row + 1) < height)
        {
            sumy += 2 * inImg[(row + 1) * width + col];
            if ((col - 1) >= 0)
            {
                sumx += -1 * inImg[(row + 1) * width + (col - 1)];
                sumy += 1 * inImg[(row + 1) * width + (col - 1)];
            }
            if ((col + 1) < width)
            {
                sumx += 1 * inImg[(row + 1) * width + (col + 1)];
                sumy += 1 * inImg[(row + 1) * width + (col + 1)];
                sumx += 2 * inImg[(row)*width + (col + 1)];
            }
        }
        // now calculate the sobel output and gradients
        sobelImg[row * width + col] = sqrt(sumx * sumx + sumy * sumy);                         // output of the sobel filter
        gradientImg[row * width + col] = atan(__fdividef(sumx, sumy)) * __fdividef(180, M_PI); // the gradient calculateion
    }
}

////////////////////////////////////////////////////////
// Conv2DOpt inputs: inImg: float  grayscale image, the gaussian filter, and the
// image width and height
//and outputs outImg:  blurred image of width*height
// This is a slightly optimized version--it's only faster if the filtersize is bigger than 3.
/////////////////////////////////////////////////////////
__global__ void Conv2DTiled(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
    int halfFilter = (int)filterSize/2;
    // first make a shared memory filter
    float sharedfilter[FILTERSIZE][FILTERSIZE];
    for (int i = 0; i < filterSize; i++)
    {
        for (int j = 0; j < filterSize; j++)
        {
            sharedfilter[i][j] = filter[i * filterSize + j];
        }
    }

    int TILESIZE = BLOCKSIZE - FILTERSIZE + 1;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    // int bdx = blockDim.x; int bdy = blockDim.y;

    // then do a tiled convolution
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE];
    int row = ty + by * TILESIZE;
    int col = tx + bx * TILESIZE;
    int startrow = row - halfFilter;
    int startcol = col - halfFilter;

    // load the tile elements
    if (startrow >= 0 && startrow < height && startcol >= 0 && startcol < width)
    {
        tile[ty][tx] = inImg[startrow * width + startcol];
    }
    else
    {
        tile[ty][tx] = 0.0f;
    }
    // then wait for the whole tile to load
    __syncthreads();

    float pval = 0.0;

    // int num_pixel = 0
    int cornerrow = ty;
    int cornercol = tx;
    ;
    // then compute if youre in the tile
    if (tx < TILESIZE && ty < TILESIZE)
    {
        for (int i = 0; i < filterSize; i++)
        {
            for (int j = 0; j < filterSize; j++)
            {
                int currentrow = i + cornerrow;
                int currentcol = j + cornercol;
                if (currentrow >= 0 && currentcol >= 0 && currentrow < height && currentcol < width)
                {
                    pval += tile[currentrow][currentcol] * sharedfilter[j][i];
                    // num_pixel++;
                }
            }
        }
        __syncthreads();
        // after every iteration then write to the output
        if (row < height && col < width)
            outImg[row * width + col] = pval; // * __fdividef(FILTERSIZE*FILTERSIZE,num_pixel);
    }

    // then make sure the threads are all done
    //__syncthreads();
}

////////////////////////////////////////////////////////
// GradientSobelOpt inputs: inImg: float  grayscale gaussian blurred image, 
// , and the image width and height 
// and outputs sobelImg:  magnitude of the gradients image of width*height
// output: gradientImg: the phase of the gradients in an image of width*height
// This is a Tiled version: because the filters are 3x3 it's a slower version
/////////////////////////////////////////////////////////
__global__ void GradientSobelTiled(float *inImg, float *sobelImg, float *gradientImg, int height, int width) {

    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    int filterSize = (int)FILTERSIZE;
    int TILESIZE = BLOCKSIZE - FILTERSIZE + 1;
    // To detect horizontal lines, G_x.
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    // To detect vertical lines, G_y
    const int fmat_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    // set up the tile
    int halfFilter = (int)filterSize / 2;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;

    // do a tiled convolution
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE];
    int row = ty + by * TILESIZE;
    int col = tx + bx * TILESIZE;
    int startrow = row - halfFilter;
    int startcol = col - halfFilter;

    // load the tile elements
    if (startrow >= 0 && startrow < height && startcol >= 0 && startcol < width)
    {
        tile[ty][tx] = inImg[startrow * width + startcol];
    }
    else
    {
        tile[ty][tx] = 0.0f;
    }
    // then wait for the whole tile to load
    __syncthreads();

    // now do the filtering
    double sumx = 0;
    double sumy = 0;
    //// DO THE SOBEL FILTERING ///////////

    // boundary check if it's in the image
    if (ty < TILESIZE && tx < TILESIZE)
    {

        // now do the filtering
        for (int j = 0; j < filterSize; j++)
        {
            for (int k = 0; k < filterSize; k++)
            {
                sumy += tile[j + ty][k + tx] * fmat_y[j][k];
                sumx += tile[j + ty][k + tx] * fmat_x[j][k];
            }
        }

        // then write to output for that element
        if (row < height && col < width)
        {
            // now calculate the sobel output and gradients
            sobelImg[row * width + col] = sqrt(sumx * sumx + sumy * sumy); // output of the sobel filter
            double value = __fdividef(sumx, sumy);
            gradientImg[row * width + col] = atan(value) * __fdividef(180, M_PI); // the gradient calculateion
        }
    }
}

// To calculate the Grayscale image = ColorToGrayscale
// For the gaussian blur = Conv2D
// and the sobel filter which gives the
//  gradient descent = GradientSobel
#include "filters.h"

////////////////////////////////////////////////////////
// populate_blur_filter inputs: filterEdgeLength: the size of the filter (square filter) 
// and outputs outFilter: a gaussian calculated that is FilterEdgeLen x FilterEdgeLen sized
/////////////////////////////////////////////////////////

void populate_blur_filter(double *outFilter, size_t filterEdgeLen)
{
    // double scaleVal = 1;
    // double stDev = (double)FILTERSIZE/3;

    double stDevSq = 0.6;
    double pi = M_PI;
    double scaleFac = (1 / (2 * pi * stDevSq));

    for (int i = 0; i < filterEdgeLen; ++i)
    {
        for (int j = 0; j < filterEdgeLen; ++j)
        {

            // pow() is slow so just multiply out
            double xComp = (i + 1 - (filterEdgeLen + 1) / 2) * (i + 1 - (filterEdgeLen + 1) / 2);
            double yComp = (j + 1 - (filterEdgeLen + 1) / 2) * (j + 1 - (filterEdgeLen + 1) / 2);

            // calculate the value at each index of the Kernel
            double filterVal = exp(-(xComp + yComp) / (2 * stDevSq));
            filterVal = scaleFac * filterVal;

            // populate Kernel
            outFilter[i + j * filterEdgeLen] = filterVal;
        }
    }
}

////////////////////////////////////////////////////////
// ColorToGrayscaleSerial inputs: input: a RGB image in ppm format, 
// , and the image width and height (x & y)
// and outputs output:  grayscale image of width*height
// This is a serialized version
/////////////////////////////////////////////////////////
void ColorToGrayscaleSerial(float *input, float *output,
                            unsigned int y, unsigned int x)
{
    for (unsigned int ii = 0; ii < y; ii++)
    {
        for (unsigned int jj = 0; jj < x; jj++)
        {
            unsigned int idx = ii * x + jj;
            float r = input[3 * idx];     // red value for pixel
            float g = input[3 * idx + 1]; // green value for pixel
            float b = input[3 * idx + 2];
            output[idx] = (float)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}



// convert the image to grayscale
////////////////////////////////////////////////////////
// ColorToGrayscaleSerial inputs: input: a RGB image in ppm format, 
// , and the image width and height (x & y)
// and outputs output:  grayscale image of width*height
// This is a CUDA version
/////////////////////////////////////////////////////////
__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height) {
   int idx, grayidx;
   int col = blockDim.x * blockIdx.x + threadIdx.x;
   int row  = blockDim.y * blockIdx.y + threadIdx.y;
   int numchannel = 3;

   // x = col and y = row
   // make sure we're within the boundary of the image
   if (col >= 0 && col < width && row >=0 && row < height) {
      // each spot is 3 big (rgb) so get the number of spots
      grayidx = row * width + col;
      idx     = grayidx * numchannel; // and multiply by three
      // to calculate the beginning of the 3 for that pixel
      float r = inImg[idx];           //red
      float g = inImg[idx + 1];       //green
      float b = inImg[idx + 2];       //blue
      outImg[grayidx]  = (0.21*r + 0.71*g + 0.07*b);
   }
}

// the gaussian blur is just a conv2d with a filter
////////////////////////////////////////////////////////
// Conv2DOpt inputs: inImg: float  grayscale image, the gaussian filter, and the
// image width and height
//and outputs outImg:  blurred image of width*height
// This is an unoptimized CUDA version
/////////////////////////////////////////////////////////
__global__ void Conv2D(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int halfFilter = (int)filterSize/2;

   // boundary check if it's in the image
   if(row >= 0 && row < height && col >= 0 && col < width) {
     float pixelvalue = 0;
     int start_col = col - halfFilter;
     int start_row = row - halfFilter;

     // now do the filtering
     for (int j = 0; j < filterSize; ++j) {
       for (int k = 0; k < filterSize; ++k) {
         int cur_row = start_row + j;
         int cur_col = start_col + k;

         // only count the ones that are inside the boundaries
         if (cur_row >=0 && cur_row < height && cur_col >= 0 && cur_col < width) {
           pixelvalue += inImg[cur_row*width + cur_col] * filter[j + k*filterSize];
         }

       }
     }
    // wait for threads to finish, then write to the output image 
    __threadfence();
     outImg[row*width + col] = pixelvalue;      
   }
}


////////////////////////////////////////////////////////
// GradientSobel inputs: inImg: float  grayscale gaussian blurred image, 
// , and the image width and height 
// and outputs sobelImg:  magnitude of the gradients image of width*height
// output: gradientImg: the phase of the gradients in an image of width*height
// This is an unoptimized CUDA  version
/////////////////////////////////////////////////////////
__global__ void GradientSobel(float *inImg, float *sobelImg, float *gradientImg, int height, int width) {
    int filterSize = (int)FILTERSIZE;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // To detect horizontal lines, G_x.
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    // To detect vertical lines, G_y
    const int fmat_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    // now do the filtering
    // halfFitler is how many are on each side
    int halfFilter = (int)filterSize / 2;
    double sumx = 0;
    double sumy = 0;
    //// DO THE SOBEL FILTERING ///////////

    // boundary check if it's in the image
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        int start_col = col - halfFilter;
        int start_row = row - halfFilter;

        // now do the filtering
        for (int j = 0; j < filterSize; ++j)
        {
            for (int k = 0; k < filterSize; ++k)
            {
                int cur_row = start_row + j;
                int cur_col = start_col + k;

                // only count the ones that are inside the boundaries
                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                {
                    sumy += inImg[cur_row * width + cur_col] * fmat_y[j][k];
                    sumx += inImg[cur_row * width + cur_col] * fmat_x[j][k];
                }
            }
        }

        // now calculate the sobel output and gradients
        sobelImg[row * width + col] = sqrt(sumx * sumx + sumy * sumy); // output of the sobel filter

        gradientImg[row * width + col] = atan(sumx / sumy) * 180 / M_PI; // the gradient calculateion
    }
}

////////////////////////////////////////////////////////
// Conv2DOpt inputs: inImg: float  grayscale image, the gaussian filter, and the
// image width and height
//and outputs outImg:  blurred image of width*height
// This is an unoptimized serial implementation version
/////////////////////////////////////////////////////////
void Conv2DSerial(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {

    // find center position of kernel (half of kernel size)
    int filterHalf = filterSize / 2;

    // iterate over rows and coluns of the image
    for (int row = 0; row < height; ++row) // rows
    {
        for (int col = 0; col < width; ++col) // columns
        {
            int start_col = col - filterHalf;
            int start_row = row - filterHalf;
            float pixelvalue = 0;

            // then for each pixel iterate through the filter
            for (int j = 0; j < filterSize; ++j) // filter rows
            {
                for (int k = 0; k < filterSize; ++k) // kernel columns
                {
                    int cur_row = start_row + j;
                    int cur_col = start_col + k;
                    if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                    {
                        pixelvalue += inImg[cur_row * width + cur_col] * filter[j + k * filterSize];
                    }
                }
            }
            outImg[row * width + col] = pixelvalue;
        }
    }
}

////////////////////////////////////////////////////////
// GradientSobelSerial inputs: inImg: float  grayscale gaussian blurred image, 
// , and the image width and height 
// and outputs sobelImg:  magnitude of the gradients image of width*height
// output: gradientImg: the phase of the gradients in an image of width*height
// This is an unoptimized serial version
/////////////////////////////////////////////////////////
void GradientSobelSerial(float *inImg, float *mag, float *phase, int height, int width)
{
	int filterSize = (int)FILTERSIZE;
	int halfFilter = (int)(filterSize)/2;
	
	// To detect horizontal lines, G_x. 
	const int fmat_x[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	// To detect vertical lines, G_y 
	const int fmat_y[3][3]  = {
		{-1, -2, -1},
		{0,   0,  0},
		{1,   2,  1}
	};

	// iterate over rows and columns of the image
	for(int row=0; row < height; ++row)              // rows
	{
		for(int col=0; col < width; ++col)          // columns
		{

			double sumx = 0;
			double sumy = 0;

			int start_col = col - halfFilter;
			int start_row = row - halfFilter;

			// now do the filtering
			for (int j = 0; j < filterSize; ++j)
			{
				for (int k = 0; k < filterSize; ++k)
				{

					int cur_row = start_row + j;
					int cur_col = start_col + k;

					// only count the ones that are inside the boundaries
					if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
					{
						sumy += inImg[cur_row*width + cur_col] * fmat_y[j][k];
						sumx += inImg[cur_row*width + cur_col] * fmat_x[j][k];
					}

				}
			}
			// now calculate the magnitude and the phase
			mag[row*width + col] = sqrt(sumx * sumx + sumy * sumy); // output of the sobel filt at this index
			phase[row*width+col] = atan(sumx/sumy) * 180/M_PI; // gradient at pixel

		}
	}
}
