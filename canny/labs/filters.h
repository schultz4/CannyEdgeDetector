#ifndef __FILTERS_H__
#define __FILTERS_H__
#define FILTERSIZE 3

void Conv2DSerial(float *inImg, float *outImg, double filter[FILTERSIZE][FILTERSIZE], int width, int height, int filterSize);
void populate_blur_filter(double outFilter[FILTERSIZE][FILTERSIZE]);
void GradientSobelSerial(float *inImg, float *sobelImg, float *gradientImg, int height, int width); 
void ColorToGrayscaleSerial(float *input, float *output, unsigned int y, unsigned int x); 

#ifdef __CUDACC__
__global__ void ColorToGrayscale(float *inImg, int *outImg, int width, int height); 
#endif

#endif // __FILTERS_H__
