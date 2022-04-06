#ifndef __FILTERS_H__
#define __FILTERS_H__
#define FILTERSIZE 3

void Conv2DSerial(int *inImg, int *outImg, double filter[FILTERSIZE][FILTERSIZE], int width, int height, int filterSize); 
void populate_blur_filter(double outFilter[FILTERSIZE][FILTERSIZE]);
void GradientSobelSerial(int *inImg, float *sobelImg, float *gradientImg, int height, int width); 
#ifdef __CUDACC__
__global__ void ColorToGrayscale(float *inImg, int *outImg, int width, int height); 
#endif

#endif // __FILTERS_H__
