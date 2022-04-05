#include <wb.h>
#include "filters.cu"
#include "Otsus_Method.h"

#define FILTERSIZE 3

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// make a void sequential here

// make a void parallel

// make a void tiled/shared memory here

// Also modify the main function to launch thekernel.
int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  //wbImage_t outputImage;

  float *hostInputImageData;
  int *hostGrayImageData;
  int *hostBlurImageData;
  float *hostGradientImageData;
  float *hostSobelImageData;
  //int *hostOutputImageData;

  float *deviceInputImageData;
  int *deviceGrayImageData;
  int *deviceBlurImageData;
  float *deviceGradientImageData;
  float *deviceSobelImageData;
  //int *deviceOutputImageData;

  unsigned int *histogram;
  histogram = (unsigned int *)calloc(256, sizeof(unsigned int));

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  // set up the images
  //outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData    = wbImage_getData(inputImage);
  //hostOutputImageData = wbImage_getData(outputImage);
  hostGrayImageData     = (int *)malloc(imageHeight*imageWidth*sizeof(int));
  hostBlurImageData     = (int *)malloc(imageHeight*imageWidth*sizeof(int));
  hostSobelImageData    = (float *)malloc(imageHeight*imageWidth*sizeof(float));
  hostGradientImageData = (float *)malloc(imageHeight*imageWidth*sizeof(float));

  //hostOutputImageData = (int *)malloc(imageHeight*imageWidth*sizeof(int));

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  ////////////////////////////////////////////////
// GRAYSCALE MEMORY SETUP
  ///////////////////////////////////////////////
  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceGrayImageData, imageWidth*imageHeight*sizeof(int));
  cudaMalloc((void **)&deviceBlurImageData, imageWidth*imageHeight*sizeof(int));
  cudaMalloc((void **)&deviceSobelImageData, imageWidth*imageHeight*sizeof(int));
  cudaMalloc((void **)&deviceGradientImageData, imageWidth*imageHeight*sizeof(int));
//  cudaMalloc((void **)&deviceOutputImageData,
//             imageWidth * imageHeight * sizeof(int));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  double filter[FILTERSIZE][FILTERSIZE];
  populate_blur_filter(filter);
  int filterSize = (int)FILTERSIZE;
  
  // 256 = 16 * 16
  int blocksize = 16;
  dim3 BlockDim(blocksize,blocksize);
  dim3 GridDim(ceil(imageWidth/blocksize), ceil(imageHeight/blocksize));
  //dim3 GridDim(((imageWidth+BlockDim.x-1)/BlockDim.x), ((imageHeight+BlockDim.y-1)/BlockDim.y));  

  // call the greyscale function
  ColorToGrayscale<<<GridDim, BlockDim>>>(deviceInputImageData, deviceGrayImageData, imageWidth, imageHeight);
  // and then blur the image
  Conv2D<<<GridDim, BlockDim>>>(deviceGrayImageData, deviceBlurImageData, filter, imageWidth, imageHeight, filterSize);
  GradientSobel<<<GridDim, BlockDim>>>(deviceBlurImageData, deviceSobelImageData, deviceGradientImageData, imageHeight, imageWidth); 


  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostBlurImageData, deviceBlurImageData, imageWidth*imageHeight*sizeof(int), cudaMemcpyDeviceToHost);
  //cudaMemcpy(hostOutputImageData, deviceOutputImageData,
  //           imageWidth * imageHeight * sizeof(int),
  //           cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");
 
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
  cudaMemcpy(hostGrayImageData, deviceGrayImageData, imageWidth*imageHeight*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(hostSobelImageData, deviceSobelImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(hostGradientImageData, deviceGradientImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);  
 
  ///////////////////////////////////////////////////////////
  // test the serial implementation with the grayscale image
  ///////////////////////////////////////////////////////////
  int *BlurImageData;
  float *SobelImageData;
  float *GradientImageData;
  BlurImageData     = (int *)malloc(imageHeight*imageWidth*sizeof(int));
  SobelImageData    = (float *)malloc(imageHeight*imageWidth*sizeof(float));
  GradientImageData = (float *)malloc(imageHeight*imageWidth*sizeof(float));

  Conv2DSerial(hostGrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);
  GradientSobelSerial(BlurImageData, SobelImageData, GradientImageData, imageHeight, imageWidth);

  std::cout << "Finished with Serial" << std::endl;
  //////////////////////////////////////////////////////////
  // end serial implementation
  //////////////////////////////////////////////////////////
  
  Histogram_Sequential(deviceGrayImageData, histogram, imageWidth, imageHeight);

  double thresh = Otsu_Sequential(histogram);

  printf("\n");
  printf("Width = %u\n",imageWidth);
  printf("Height = %u\n",imageHeight);
  printf("Histogram[0] = %u\n",histogram[0]);
  printf("Histogram[1] = %u\n",histogram[1]);
  printf("Histogram[20] = %u\n",histogram[20]);
  printf("Histogram[45] = %u\n",histogram[45]);
  printf("Histogram[56] = %u\n",histogram[56]);
  printf("Image[0] = %f\n",hostGrayImageData[0]);
  printf("Image[1] = %f\n",hostGrayImageData[1]);
  printf("Image[20] = %f\n",hostGrayImageData[20]);
  printf("Otsu's Threshold = %f\n", thresh);
  printf("\n");
  
  //wbSolution(args, outputImage);
  cudaFree(deviceInputImageData);
  cudaFree(deviceGrayImageData);
  cudaFree(deviceBlurImageData);
  cudaFree(deviceSobelImageData);
  cudaFree(deviceGradientImageData);


  //char *oFile = wbArg_getOutputFile(args);
  //wbExport(oFile, hostOutputImageData, imageWidth, imageHeight);
  //wbExport(oFile, outputImage);

  //cudaFree(deviceOutputImageData);

  //wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
