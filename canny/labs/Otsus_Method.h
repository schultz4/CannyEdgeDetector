#pragma once
#include <iostream>
#include <chrono>
using namespace std;

void Histogram_Sequential(unsigned char* image, int width, int height, int* hist);
double Otsu_Sequential(int* hist);
void Threshold_Sequential(unsigned char *image, unsigned char *strong_edges, unsigned char *weak_edges, int width, int height, double thresh);
