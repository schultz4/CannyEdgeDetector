#include <stdio.h>
#include <wb.h>
#include "Otsus_Method.h"
#include "filters.h"

// Also modify the main function to launch thekernel.
int main(int argc, char *argv[]) {
  printf("Canny Serial Solution\n");

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



  ///////////////////////////////////////////////////////////
  // test the serial implementation with the grayscale image
  ///////////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing Computation (memory + compute)");

  double filter[FILTERSIZE][FILTERSIZE];
  populate_blur_filter(filter);
  int filterSize = (int)FILTERSIZE;

  int *BlurImageData;
  float *SobelImageData;
  float *GradientImageData;
  BlurImageData     = (int *)malloc(imageHeight*imageWidth*sizeof(int));
  SobelImageData    = (float *)malloc(imageHeight*imageWidth*sizeof(float));
  GradientImageData = (float *)malloc(imageHeight*imageWidth*sizeof(float));

  Conv2DSerial(hostGrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);
  GradientSobelSerial(BlurImageData, SobelImageData, GradientImageData, imageHeight, imageWidth);
  
  Histogram_Sequential(hostGrayImageData, histogram, imageWidth, imageHeight); // Switched from device

  double thresh = Otsu_Sequential(histogram);

  wbTime_stop(Compute, "Doing the serial computation");

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

  std::cout << "Finished with Serial" << std::endl;
  //////////////////////////////////////////////////////////
  // end serial implementation
  //////////////////////////////////////////////////////////
  
  //char *oFile = wbArg_getOutputFile(args);
  //wbExport(oFile, hostGrayImageData, imageWidth, imageHeight);
  //wbExport(oFile, outputImage);

  //cudaFree(deviceOutputImageData);

  //wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

