#pragma once
#include <iostream>
#include <chrono>
using namespace std;

void Histogram_Sequential(unsigned char* image, int width, int height, int* hist);
char Threshold_Sequential(int* hist);
