#pragma once
#include <iostream>
#include <chrono>
using namespace std;

void Histogram_Sequential(float* image, unsigned int* hist, int width, int height);
float Otsu_Sequential_Optimized(unsigned int* hist, int width, int height);
float Otsu_Sequential(unsigned int* hist, int width, int height);

#ifdef __CUDACC__
__global__ void NaiveHistogram(float* image, unsigned int* hist, int width, int height);
__global__ void NaiveOtsu(unsigned int* histogram, float* thresh, int width, int height);
__global__ void OptimizedHistogram(float* image, unsigned int* hist, int width, int height);
__global__ void OptimizedHistogramReplication(float* image, unsigned int* hist, int width, int height);
__global__ void OptimizedOtsu(unsigned int* histogram, float* thresh, int width, int height);
#endif