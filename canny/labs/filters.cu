// For the gaussian blur
// and the sobel filter which gives the 
//  gradient descent

#define FILTERSIZE 3

// convert the image to grayscale
__global__ void ColorToGrayscale(float *inImg, int *outImg, int width, int height) {
   int idx, grayidx;
   int col = blockDim.x * blockIdx.x + threadIdx.x;
   int row  = blockDim.y * blockIdx.y + threadIdx.y;
   int numchannel = 3;

   // x = col and y = row
   if (col >= 0 && col < width && row >=0 && row < height) {
      // each spot is 3 big (rgb) so get the number of spots
      grayidx = row * width + col;
      idx     = grayidx * numchannel; // and multiply by three
      // to calculate the beginning of the 3 for that pixel
      float r = inImg[idx];           //red
      float g = inImg[idx + 1];       //green
      float b = inImg[idx + 2];       //blue
      outImg[grayidx]  = (int)((0.21*r + 0.71*g + 0.07*b)*255);
   }
}


// the gaussian blur is just a conv2d with a filter
__global__ void Conv2D(int *inImg, int *outImg, double filter[FILTERSIZE][FILTERSIZE], int width, int height, int filterSize) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int halfFilter = (int)filterSize/2;

   // boundary check if it's in the image
   if(row > 0 && row < height && col > 0 && col < width) {
      int pixelvalue = 0;
      int start_col = col - halfFilter;
      int start_row = row - halfFilter;
      
      // now do the filtering
      for (int j = 0; j < filterSize; ++j) {
         for (int k = 0; k < filterSize; ++k) {
	    int cur_row = start_row + j;
            int cur_col = start_col + k;
           
            // only count the ones that are inside the boundaries
            if (cur_row >=0 && cur_row < height && cur_col >= 0 && cur_col < width) {
               pixelvalue += inImg[cur_row*width + cur_col] * filter[j][k];
	    }
           
         }
      }
      __syncthreads();
      outImg[row*width + col] = (unsigned char)(pixelvalue);      
   }

}

__global__ void GradientSobel(int *inImg, float *sobelImg, float *gradientImg, int height, int width) {
   int filterSize = (int)FILTERSIZE;
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   
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

   // now do the filtering
   // halfFitler is how many are on each side
   int halfFilter = (int)filterSize/2;
   double sumx = 0;
   double sumy = 0;
   //// DO THE SOBEL FILTERING ///////////

   // boundary check if it's in the image
   if(row > 0 && row < height && col > 0 && col < width) {
      int start_col = col - halfFilter;
      int start_row = row - halfFilter;

      // now do the filtering
      for (int j = 0; j < filterSize; ++j) {
         for (int k = 0; k < filterSize; ++k) {
            int cur_row = start_row + j;
            int cur_col = start_col + k;

            // only count the ones that are inside the boundaries
            if (cur_row >=0 && cur_row < height) {
               sumy += inImg[cur_row*width + cur_col] * fmat_y[j][k];
            }
	    if ( cur_col >= 0 && cur_col < width) {
               sumx += inImg[cur_row*width + cur_col] * fmat_x[j][k];
            }

         }
      }

      // now calculate the sobel output and gradients
      __syncthreads();
      int value = sqrt(sumx * sumx + sumy*sumy);
      if (value > 255) {
          value = 255;
      } 
      if (value < 0) {
          value = 0;
      }

      sobelImg[row*width + col] = value; // output of the sobel filter
      gradientImg[row*width + col] = atan(sumx/sumy) * 180/M_PI; // the gradient calculateion
   }
 
   __syncthreads();

}


/*
__global__ void GaussianFilter(float *inImg, float *outImg, float *filter, int width, int height, int filterSize) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        //int y = blockDim.y * blockIdx.y + threadIdx.y;

        // read in kernel image into shared memory?
        __shared__ double inFilter[FILTERSIZE][FILTERSIZE];

        for (int fx = 0; fx < filterSize; ++fx) {
                for (int fy = 0; fy < filterSize; ++fy) {
                        inFilter[fx][fy] = filter[fx * filterSize + fy];
                }
        }

        __syncthreads();

        int filterHalf = (filterSize - 1) / 2;
        // check boundaries, do I need to do width - filterSize?
        if (x > 0 && x < width*height ) {
                double kernelval = 0;
                double sumval    = 0;
                //multiply every value of the filter with the corresponding image pixel
                for (int fy = 0; fy < filterSize; ++fy) {
                        for (int fx = 0; fx < filterSize; ++fx) {
                                // check edge cases, if it's within the boundary then apply filter
                                if ((x + ((fy -filterHalf)*width)+fx - filterHalf >= 0) &&
                                   (x + ((fy -filterHalf)*width)+fx - filterHalf <= width*height-1) &&
                                   (((x % width) + fx - filterHalf) >= 0) &&
                                   (((x % width) + fx - filterHalf) <= width-1)) {
                                        int xval = x + filterHalf*width + fx - filterHalf;
                                        sumval   += inImg[xval] * inFilter[fy][fx];
                                        kernelval++;
                                }
                        }
                }


                outImg[x] = sumval/ kernelval;
        }
        __syncthreads();
}
*/

/*
__global__ void SobelFilterGradient(float *inImg, float *outImg, float *gradientDir, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // these are just const values? idk where they come from
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


        int filterSize = FILTERSIZE;
        double halfFilter = filterSize/2;

        double sumx = 0;
        double sumy = 0;
        // make sure to skip the ones that are on the edges.
        // since the filter is 3 wide, just skip the 1 edges and you'll miss the others
        if (x < 0 || x > width - 1 || y < 0 || y > height - 1) {
                return;
        }
     for (int fy = y - halfFilter; fy < (y + filterSize - halfFilter); fy++) {
                for (int fx = x - halfFilter; fx < (x + filterSize - halfFilter); fx++) {
                        sumx += (double)(fmat_x[fy - y + halfFilter][x - fx + halfFilter] * inImg[fy * width + fx]);
                        sumy += (double)(fmat_y[fy - y + halfFilter][x - fx + halfFilter] * inImg[fy * width + fx]);
                }
        }

        __syncthreads();

        // get magnitude then clip it to 0-255
        // sqrt (x^2 + y^2) 
        int value = sqrt(sumx * sumx + sumy * sumy);
        if (value > 255)
                value = 255;
        if (value < 0)
                value = 0;

        outImg[y * width + x]      = value; // output of the sobel filter
        gradientDir[y * width + x] = atan(sumx/sumy) * 180/M_PI; // the gradient calculation

        }
        __syncthreads();
}
*/




