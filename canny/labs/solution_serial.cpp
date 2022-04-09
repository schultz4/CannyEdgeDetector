#include <stdio.h>
#include <wb.h>
#include "Otsus_Method.h"
#include "filters.h"
#include "non_max_supp.h"

// Also modify the main function to launch thekernel.
int main(int argc, char *argv[]) {

	//////////////////////////////
	// Parameter Initialization //
	//////////////////////////////


	// Image parameters for wbLib
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;

	// Host side parameters
	float *hostInputImageData;

	// Filtering parameters
  float *GrayImageData;
	float *BlurImageData;
	float *GradMagData;
	float *GradPhaseData;
	float *NmsImageData;

	// Otsu's Method parameters
	unsigned int *histogram;


	////////////////////
	// Image Handling //
	////////////////////


	// Parse the input arguments
	args = wbArg_read(argc, argv);

	// Read input file
	inputImageFile = wbArg_getInputFile(args, 0);

	// Import input image 
	inputImage = wbImport(inputImageFile);

	// Scrape info from input image
	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	
	// Define output image data
	hostInputImageData = wbImage_getData(inputImage);

	////////////////////////////////
	// Host Memory Initialization //
	////////////////////////////////

	// Allocate memory for serial filtering
  GrayImageData     = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	BlurImageData     = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	GradMagData    		= (float *)calloc(imageHeight*imageWidth, sizeof(float));
	GradPhaseData 		= (float *)calloc(imageHeight*imageWidth, sizeof(float));
	NmsImageData      = (float *)calloc(imageHeight*imageWidth, sizeof(float));

  // Set output image for phase desired
  // Note - input image is 3 channels. Other phases only have 1 channel
  float *outData = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	outputImage = wbImage_new(imageWidth, imageHeight, 1, outData);
  
	histogram = (unsigned int *)calloc(256, sizeof(unsigned int));


	/////////////////////////
	// Image Preprocessing //
	/////////////////////////


	// Create filter skeleton
	double filter[FILTERSIZE][FILTERSIZE];

	// Fill the gaussian filter
	populate_blur_filter(filter);

	// ?????
	int filterSize = (int)FILTERSIZE;


	////////////////////
	// Host Execution //
	////////////////////

  // GrayImageData Serial
  ColorToGrayscaleSerial(hostInputImageData, GrayImageData, imageWidth, imageHeight);
  
	// Blur image using Gaussian Kernel
	Conv2DSerial(GrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);

	// Calculate gradient using Sobel Operators
	GradientSobelSerial(BlurImageData, GradMagData, GradPhaseData, imageHeight, imageWidth);

  // Suppress non-maximum pixels along gradient
  nms(GradMagData, NmsImageData, GradPhaseData, imageHeight, imageWidth);

	// Calculate histogram of blurred image
	Histogram_Sequential(BlurImageData, histogram, imageWidth, imageHeight);

	// Calculate threshold using Otsu's Method
	double thresh = Otsu_Sequential(histogram);

  // Copy image data for output image (choose 1 - can only log one at a time for now
  memcpy((void *)outData, (void *) GrayImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, BlurImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, SobelImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, NmsImageData, imageHeight*imageWidth*sizeof(float));

	////////////////////
	// Debugging Info //
	////////////////////


	// Print info
	printf("\n");
	printf("Width = %u\n",imageWidth);
	printf("Height = %u\n",imageHeight);
	printf("InputImage[0] = %f\n",hostInputImageData[0]);
	printf("Histogram[0] = %u\n",histogram[0]);
	printf("Histogram[1] = %u\n",histogram[1]);
	printf("Histogram[20] = %u\n",histogram[20]);
	printf("Histogram[45] = %u\n",histogram[45]);
	printf("Histogram[56] = %u\n",histogram[56]);
	printf("Image[0] = %f\n",GrayImageData[0]);
	printf("Image[1] = %f\n",GrayImageData[1]);
	printf("Image[36] = %f\n",GrayImageData[36]);
	printf("Image[400] = %f\n",GrayImageData[400]);
	printf("Image[900] = %f\n",GrayImageData[900]);
	printf("Image[1405] = %f\n",GrayImageData[1405]);
	printf("Image[85000] = %f\n",GrayImageData[85000]);
	printf("First row of Gaussian filter = %f %f %f\n",filter[0][0], filter[0][1], filter[0][2]);
	printf("Second row of Gaussian filter = %f %f %f\n",filter[1][0], filter[1][1], filter[1][2]);
	printf("Third row of Gaussian filter = %f %f %f\n",filter[2][0], filter[2][1], filter[2][2]);
	printf("Blurred Image[0] = %f\n",BlurImageData[0]*255);
	printf("Blurred [25] = %f\n", BlurImageData[25]*255);
	printf("Blurred Image[290] = %f\n",BlurImageData[290]*255);
	printf("Gradient magnitude at [0] = %f\n",GradMagData[0]);
	printf("Gradient magnitude at [20] = %f\n",GradMagData[20]);
	printf("Gradient magnitude at [9000] = %f\n",GradMagData[9000]);
	printf("Gradient phase at [0] = %f\n",GradPhaseData[0]);
	printf("Gradient phase at [20] = %f\n",GradPhaseData[20]);
	printf("Gradient phase at [290] = %f\n",GradPhaseData[290]);
	printf("Otsu's Threshold = %f\n", thresh);
	printf("\n");

	// Export image
	char *oFile = wbArg_getOutputFile(args);
	wbExport(oFile, outputImage);


	//////////////
	// Clean Up //
	//////////////


	// Destroy all host memory
	free(GradMagData);
	free(GradPhaseData);
	free(NmsImageData);
  free(GrayImageData);
	free(BlurImageData);
	free(histogram);

	// Destroy images
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
