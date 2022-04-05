#include <wb.h>
#include "Otsus_Method.h"

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
  //float *hostGrayImageData;
  //float *hostBlurImageData;
  //float *hostGradientImageData;
  //float *hostSobelImageData;
  float *hostOutputImageData;

  float *deviceInputImageData;
  //float *deviceGrayImageData;
  //float *deviceBlurImageData;
  //float *deviceGradientImageData;
  //float *deviceSobelImageData;
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
  // GRAYSCALE

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
  cudaFree(deviceInputImageData);
  //////////////////////////////////////////////////////////////
  // END GRAYSCALE
  ///////////////////////////////////////////////////////////
/*
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
*/
  char *oFile = wbArg_getOutputFile(args);
  //wbExport(oFile, hostOutputImageData, imageWidth, imageHeight);
  wbExport(oFile, outputImage);

  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
