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
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int filterHalf = filterSize / 2;	
	// check boundaries, do I need to do width - filterSize?
	if (x > width - filterHalf || y > height - filterHalf) {
		return;
	}
	
	double sumval = 0;
	//multiply every value of the filter with the corresponding image pixel
	for (int fy = 0; fy < filterSize; fy++) {
		for (fx = 0; fx < filterSize; fx++) {
			//first center the filter on the image
			int xval = x - filterSize / 2.0 + fx;
			int yval = y - filterSize / 2.0 + fy;
			sumval   = inImg[yval*width + xval] * filter[fy][fx]
		}
	}
	
}

__global__ void 


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
  float *hostOutputImageData;
  float *deviceInputImageData;
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
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  // 256 = 16 * 16
  dim3 BlockDim(16,16);
  dim3 GridDim;

  GridDim.x = (imageWidth + BlockDim.x - 1) / BlockDim.x;
  GridDim.y = (imageHeight + BlockDim.y - 1) / BlockDim.y;

  // call the greyscale function
  ColorToGrayscale<<<GridDim, BlockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);


  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  //wbSolution(args, outputImage);

  char *oFile = wbArg_getOutputFile(args);
  //wbExport(oFile, hostOutputImageData, imageWidth, imageHeight);
  wbExport(oFile, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
