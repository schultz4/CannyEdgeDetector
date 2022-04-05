#pragma once
#include <iostream>
#include <chrono>
using namespace std;

void Histogram_Sequential(int* image, unsigned int* hist, int width, int height);
double Otsu_Sequential(unsigned int* hist);
