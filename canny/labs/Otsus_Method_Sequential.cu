#include "Otsus_Method.h"

void Histogram_Sequential(unsigned char *image, int width, int height, int *hist)
{
	// Row pointer
	unsigned char* matrix = image;

	// Loop through every pixel
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// Update histogram
			hist[matrix[j]]++;
		}

		// Update row pointer
		matrix += height;

	}

}

double Otsu_Sequential(int* histogram)
{

	// Calculate the bin_edges
	double bin_edges[256];
	bin_edges[0] = 0.0;
	double increment = 0.99609375;
	for (int i = 1; i < 256; i++)
		bin_edges[i] = bin_edges[i - 1] + increment;

	// Calculate bin_mids
	double bin_mids[256];
	for (int i = 0; i < 256; i++)
		bin_mids[i] = (bin_edges[i] + bin_edges[i + 1]) / 2;

	// Iterate over all thresholds (indices) and get the probabilities weight1, weight2
	double weight1[256];
	weight1[0] = histogram[0];
	for (int i = 1; i < 256; i++)
		weight1[i] = histogram[i] + weight1[i - 1];

	int total_sum = 0;
	for (int i = 0; i < 256; i++)
		total_sum = total_sum + histogram[i];
	double weight2[256];
	weight2[0] = total_sum;
	for (int i = 1; i < 256; i++)
		weight2[i] = weight2[i - 1] - histogram[i - 1];

	// Calculate the class means: mean1 and mean2
	double histogram_bin_mids[256];
	for (int i = 0; i < 256; i++)
		histogram_bin_mids[i] = histogram[i] * bin_mids[i];

	double cumsum_mean1[256];
	cumsum_mean1[0] = histogram_bin_mids[0];
	for (int i = 1; i < 256; i++)
		cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];

	double cumsum_mean2[256];
	cumsum_mean2[0] = histogram_bin_mids[255];
	for (int i = 1, j = 254; i < 256 && j >= 0; i++, j--)
		cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[j];

	double mean1[256];
	for (int i = 0; i < 256; i++)
		mean1[i] = cumsum_mean1[i] / weight1[i];

	double mean2[256];
	for (int i = 0, j = 255; i < 256 && j >= 0; i++, j--)
		mean2[j] = cumsum_mean2[i] / weight2[j];

	// Calculate Inter_class_variance
	double Inter_class_variance[255];
	double dnum = 10000000000;
	for (int i = 0; i < 255; i++)
		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) / dnum) * (mean1[i] - mean2[i + 1]);

	// Maximize interclass variance
	double maxi = 0;
	int getmax = 0;
	for (int i = 0; i < 255; i++) {
		if (maxi < Inter_class_variance[i]) {
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}

	return bin_mids[getmax];

}

void Threshold_Sequential(unsigned char *image, unsigned char *strong_edges, unsigned char *weak_edges, int width, int height, double thresh)
{
	// Row pointer
	unsigned char *matrix = image;

	// Threshold from Otsu's method
	double upper_thresh = thresh;

	// Lower threshold calculated from Canny
	double lower_thresh = thresh - 0.2;

	// Loop through all pixels
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// Strong edges
			if (matrix[j] > upper_thresh)
			{
				// Write pixel value to strong_edges matrix
				strong_edges[j] = matrix[j];
			}
			// weak edges
			else if(matrix[j] <= upper_thresh && matrix[j] > lower_thresh)
			{
				// Write pixel value to weak_edges matrix
				weak_edges[j] = matrix[j];
			}
		}

		// Update row pointers
		matrix += height;
		strong_edges += height;
		weak_edges += height;

	}
}
