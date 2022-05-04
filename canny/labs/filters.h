#ifndef __FILTERS_H__
#define __FILTERS_H__


void populate_blur_filter(double *outFilter, size_t filterEdgeLen, float stdev);
void ColorToGrayscaleSerial(float *input, float *output, unsigned int y, unsigned int x);

void Conv2DSerial(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);

void GradientSobelSerial(float *inImg, float *sobelImg, float *gradientImg, int height, int width);

#ifdef __CUDACC__
__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height);

__global__ void Conv2D(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
__global__ void Conv2DOptRow(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
__global__ void Conv2DOptCol(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);

__global__ void GradientSobel(float *inImg, float *sobelImg, float *gradientImg, int height, int width);
__global__ void GradientSobelTiled(float *inImg, float *sobelImg, float *gradientImg, int height, int width);
__global__ void GradientSobelOpt(float *inImg, float *sobelImg, float *gradientImg, int height, int width);


#endif

#endif // __FILTERS_H__
