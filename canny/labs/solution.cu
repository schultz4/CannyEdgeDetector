#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ INSERT DEVICE CODE HERE

__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height) {
        int idx, grayidx;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int row  = blockDim.y * blockIdx.y + threadIdx.y;
        int numchannel = 3;

        // x = col and y = row
        if (col < width && row < height) {
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

__global__ void Make2DGaussFilter( float *inFilter, int filterSize) { 
	int x = blockDim.x * blockIdx.x + threadIdx.x;
        
	filterCenter = (filterSize - 1.0)/2.0;
        int filterSum = 0;

	//first make the filter
	//filter = (float *)malloc(fltSize*fltSize*sizeof(float));
	// filters are square
	if (x < width*height) {
	        for (int i = 0; i < filterSize; ++i) {
			inFilter[i*filterSize + j] = exp(-(pow(i-filterCenter,2)+pow(j-filterCenter,2))
                                /(2*filterSize*filterSize))/(2*M_PI*filterSize*filterSize);	
			filterSum += inFilter[i*filterSize + j];
		}
	}
	//__syncthreads__;

	for (i = 0; i < filterSize*filterSize; ++i) {
		inFilter[i] /= filterSum;
	}

}

__global__ void Make1DGaussFilter(float *inFilter, int filterSize) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	filterCenter = (filterSize - 1.0)/2.0;
        int filterSum = 0;

	for (int i = 0; i < filterSize; ++i) {
		inFilter[i] = exp(-pow((i-filterCenter)/filterSize, 2)/2)/(filterSize * sqrt(2*M_PI));
		filterSum += inFilter[i];
	}

	// then normalize the filter
        for (i = 0; i < filterSize*filterSize; ++i) {
                inFilter[i] /= filterSum;
        }

}

__global__ void GaussianFilter(float *inImg, float *outImg, float *filter, int width, int height, int filterSize) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;

	// read in kernel image into shared memory?
	__shared__ double inFilter[filterSize][filterSize];
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
		double sumval	 = 0;
		//multiply every value of the filter with the corresponding image pixel
		for (int fy = 0; fy < filterSize; ++fy) {
			for (fx = 0; fx < filterSize; ++fx) {
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

__global__ void SobelFilterGradient(float *inImg, float *outImg, float *gradientDir, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// these are just const values? idk where they come from
	/* To detect horizontal lines, G_x. */
    	const int fmat_x[3][3] = {
        	{-1, 0, 1},
        	{-2, 0, 2},
        	{-1, 0, 1}
    	};
    	/* To detect vertical lines, G_y */
    	const int fmat_y[3][3]  = {
        	{-1, -2, -1},
        	{0,   0,  0},
        	{1,   2,  1}
    	};


	double filterSize = 3;
	double halfFilter = filterSize/2;

	double sumx = 0;
	double sumy = 0;
	// make sure to skip the ones that are on the edges.
	// since the filter is 3 wide, just skip the 1 edges and you'll miss the others
	if (x < 0 || x > width - 1 || y < 0 || y > height - 1) {
		return;
	}

	for (fy = y - halfFilter; fy < (y + filterSize - filterHalf); fy++) {
		for (fx = x - halfFilter; fx < (x + filterSize - filterHalf); fx++) {
			sumx += (double) fmat_x[fy - y + halfFilter][x - fx + halfFilter] * inImg[fy * width + fx];
			sumy += (double) fmat_y[fy - y + halfFilter][x - fx + halfFilter] * inImg[fy * width + fx];
		}
	}

	__syncthreads();

	// get magnitude then clip it to 0-255
	// sqrt (x^2 + y^2) 
	int value = sqrt(sumx * sumx + sumy * sumy);
	if (value > 255) 
		value = 255
	if (value < 0)
		value = 0

	outImg[y * width + x]      = value; // output of the sobel filter
	gradientDir[y * width + x] = atan(sumx/sumy) * 180/M_PI); // the gradient calculation

	}	
	__syncthreads();
} 


// Also modify the main function to launch thekernel.
int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;

  float *hostInputImageData;
  float *hostGrayImageData;
  float *hostBlurImageData;
  float *hostGradientImageData;
  float *hostSobelImageData;
  float *hostOutputImageData;

  float *deviceInputImageData;
  float *deviceGrayImageData;
  float *deviceBlurImageData;
  float *deviceGradientImageData;
  float *deviceSobelImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
//  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  ////////////////////////////////////////////////
// GRAYSCALE MEMORY SETUP
  ///////////////////////////////////////////////
  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceGrayImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  // GRAYSCALE

  // 256 = 16 * 16
  dim3 BlockDim(16,16);
  dim3 GridDim;

  GridDim.x = (imageWidth + BlockDim.x - 1) / BlockDim.x;
  GridDim.y = (imageHeight + BlockDim.y - 1) / BlockDim.y;

  // call the greyscale function
  ColorToGrayscale<<<GridDim, BlockDim>>>(deviceInputImageData, deviceGrayImageData, imageWidth, imageHeight);


  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostGrayImageData, deviceGrayImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");
 
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  //wbSolution(args, outputImage);
  cudaFree(deviceInputImageData);
  //////////////////////////////////////////////////////////////
  // END GRAYSCALE
  ///////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////
  // START GAUSSIAN BLUR 
  ///////////////////////////////////////////////////////
  cudaMalloc((void**)&deviceBlurImageData, imageWidth*imageHeight*sizeof(float));
  cudaMemcpy(deviceGrayImageData, hostGrayImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);
  int block_dim  = 1024;
  int num_blocks = (int)ceil((imageWidth*imageHeight) / block_dim);
  dim3 gridDim(gridsize);
  dim3 blockDim(blocksize);
  // todo get the acutal filter here
  float *hostFilter = gaussianFilter();
  float *deviceFilter;
  cudaMemcpy(deviceFilter, hostFilter, filtersize*filterSize*sizeof(float), cudaMemcpyHostToDevice);
  GaussianFilter<<<gridDim, blockDim>>>(deviceGrayImageData, deviceBlurImageData, deviceFilter, imageWidth, imageHeight, filterSize);

  cudaMemcpy(hostBlurImageData, deviceBlurImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceGrayImageData);

  /////////// ///////////////////////////////////////
  // END GAUSS BLUR
  //////////////////////////////////////////////////
  //////////////////////////////////////////////////
  // START SOBEL
  /////////////////////////////////////////////////

  cudaMalloc((void**)&deviceSobelImageData, imageWidth*imageHeight*sizeof(float));
  cudaMalloc((void**)&deviceGradientImageData, imageWidth*imageHeight*sizeof(float));
  cudaMemcpy(deviceBlurImageData, hostBlurImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 BlockDim(16,16);
  dim3 GridDim;

  GridDim.x = (imageWidth + BlockDim.x - 1) / BlockDim.x;
  GridDim.y = (imageHeight + BlockDim.y - 1) / BlockDim.y;

  SobelFilterGradient(deviceBlurImageData, deviceSobelImageData, deviceGradientImageData, imageWidth, imageHeight);

  cudaMemcpy(hostSobelImageData, devideSobelImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostGradientImageData, deviceGradientImageData, imageWidth*imageHeight*sizeof(float), cuaMemcpyDeviceToHost);

  ///////////////////////////////////////////////
  // END SOBEL AND GRADIENT CALC
  //////////////////////////////////////////////
  char *oFile = wbArg_getOutputFile(args);
  //wbExport(oFile, hostOutputImageData, imageWidth, imageHeight);
  wbExport(oFile, outputImage);

  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
