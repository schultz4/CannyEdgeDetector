#include "Otsus_Method.h"
#include <cmath>

#define OUTPUT_VAL 200
#define NUM_BINS 256

void Histogram_Sequential(float *image, unsigned int *hist, int width, int height)
{
	// Loop through every pixel
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pos = image[row*width + col] * 255;

			pos = (pos > 255) ? 255 : pos;

			hist[pos]++;
		}

	}

}

double Otsu_Sequential(unsigned int* histogram, int width, int height)
{

	float histogram_bin_mids[256];
	float weight1[256];
	float weight2[256];
	float cumsum_mean1[256];
	float cumsum_mean2[256];
	float mean1[256];
	float mean2[256];
	float inter_class_variance[255];
	float max_variance = 0;

	int thresh = 0;

	float bin_length = 255.0f/256.0f;
	float half_bin_length = 255.0f/512.0f;

	// Calculate bin mids
	for(int i = 0; i < 256; i++)
	{
		histogram_bin_mids[i] = histogram[i] * (half_bin_length + bin_length * i);
	}

	weight1[0] = histogram[0];
	weight2[0] = width * height;

	// Calculate class probabilities
	for(int i = 1; i < 256; i++)
	{
		weight1[i] = histogram[i] + weight1[i-1];
		weight2[i] = weight2[i-1] - histogram[i-1];
	}

	cumsum_mean1[0] = histogram_bin_mids[0];
	cumsum_mean2[0] = histogram_bin_mids[255];

	// Calculate class means
	for(int i = 1; i < 256; i++)
	{
		cumsum_mean1[i] = cumsum_mean1[i-1] + histogram_bin_mids[i];
		cumsum_mean2[i] = cumsum_mean2[i-1] + histogram_bin_mids[256 - i - 1];
		mean1[i] = cumsum_mean1[i] / weight1[i];
		mean2[256 - i - 1] = cumsum_mean2[i] / weight2[256 - i - 1];
	}

	for (int i = 1; i < 256; i++)
	{
		mean1[i] = (weight1[i] == 0) ? 0 : mean1[i];
		mean2[i] = (weight2[i] == 0) ? 0 : mean2[i];	
	} 

	// Calculate Inter_class_variance
	for(int i = 0; i < 255; i++)
	{
		inter_class_variance[i] = (weight1[i] * weight2[i] * (mean1[i] - mean2[i+1])) * (mean1[i] - mean2[i+1]);	
	}

	// Maximize interclass variance
	for(int i = 0;i < 255; i++){
		if(max_variance < inter_class_variance[i])
		{
			max_variance = inter_class_variance[i];
			thresh = i;
		}
	}

	// Return normalized threshold
	//return bin_mids[thresh]; //This is the actual Otsu's threshold
	//return cumsum_mean1[OUTPUT_VAL]; //This is a test value
	return half_bin_length + bin_length * thresh; // This is also a test value and equivalent to key[0]

}


double Otsu_Sequential_Optimized(unsigned int* histogram, int width, int height)
{

	float histogram_bin_mids[256];
	float weight1[256];
	float weight2[256];
	float cumsum_mean1[256];
	float cumsum_mean2[256];
	float mean1[256];
	float mean2[256];
	float inter_class_variance[255];
	float max_variance = 0;

	int thresh = 0;

	float bin_length = 255.0f/256.0f;
	float half_bin_length = 255.0f/512.0f;

	// Calculate bin mids
	#pragma omp parallel for
	for(int i = 0; i < 256; i++)
	{
		histogram_bin_mids[i] = histogram[i] * (half_bin_length + bin_length * i);
	}

	weight1[0] = histogram[0];
	weight2[0] = width * height;

	float w1 = histogram[0];

	// Calculate class probabilities
	#pragma omp parallel for simd reduction(inscan, + : w1)
	for(int i = 1; i < 256; i++)
	{
		w1 += histogram[i];
		#pragma omp scan inclusive
		weight1[i] = w1;
	}

	for(int i = 1; i < 256; i++)
	{
		weight2[i] = weight2[i-1] - histogram[i-1];
	}

	cumsum_mean1[0] = histogram_bin_mids[0];
	cumsum_mean2[0] = histogram_bin_mids[255];

	// Calculate class means
	for(int i = 1; i < 256; i++)
	{
		cumsum_mean1[i] = cumsum_mean1[i-1] + histogram_bin_mids[i];
		cumsum_mean2[i] = cumsum_mean2[i-1] + histogram_bin_mids[256 - i - 1];
		mean1[i] = cumsum_mean1[i] / weight1[i];
		mean2[256 - i - 1] = cumsum_mean2[i] / weight2[256 - i - 1];
	}

	// Calculate Inter_class_variance
	#pragma omp parallel for
	for(int i = 0; i < 255; i++)
	{
		inter_class_variance[i] = (weight1[i] * weight2[i] * (mean1[i] - mean2[i+1])) * (mean1[i] - mean2[i+1]);	
	}

	// Maximize interclass variance
	#pragma omp parallel for reduction( 
	for(int i = 0;i < 255; i++){
		if(max_variance < inter_class_variance[i])
		{
			max_variance = inter_class_variance[i];
			thresh = i;
		}
	}

	// Return normalized threshold
	//return bin_mids[thresh]; //This is the actual Otsu's threshold
	//return cumsum_mean1[OUTPUT_VAL]; //This is a test value
	return half_bin_length + bin_length * thresh; // This is also a test value and equivalent to key[0]

}


__global__ void NaiveHistogram(float* image, unsigned int* histogram, int width, int height)
{
	// insert your code here
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = blockDim.x * gridDim.x;

	while(tid < width * height)
	{
		int position = int(image[tid]*255);

		if (position >= 0 && position < 256)
		{
			atomicAdd(&(histogram[position]),1);
		}

		tid += stride;

	}
}

__global__ void OptimizedHistogram(float* image, unsigned int* histogram, int width, int height)
{
	__shared__ unsigned int histogram_private[256];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = blockDim.x * gridDim.x;

	for(int bin = threadIdx.x; bin < 256; bin += blockDim.x)
	{
		histogram_private[bin] = 0;
	}

	__syncthreads();

	while(tid < width * height)
	{
		int position = int(image[tid]*255);

		if (position >= 0 && position < 256)
		{
			atomicAdd(&(histogram_private[position]),1);
		}

		tid += stride;

	}

	__syncthreads();

	for(int bin = threadIdx.x; bin < 256; bin += blockDim.x)
	{
		atomicAdd(&(histogram[bin]), histogram_private[bin]);
	}

}

__global__ void OptimizedHistogramReplication(float* image, unsigned int* histogram, int width, int height)
{
	//Make sure to call this kernel with 1024 threads per block
	const int R = 20;

	__shared__ unsigned int hist_private[(256 + 1) * R];

	// warp indexes
	const int warp_id = (int)( __fdividef(threadIdx.x, 32) );
	const int lane = threadIdx.x & 31;
	const int warps_per_block = (int)( __fdividef(blockDim.x, 32) );

	// offset to per-block sub histogram
	const int off_rep = (256 + 1) * (threadIdx.x % R);

	// set const for interleaved read access
	// to reduce the number of overlapping warps
	// account for the case where warp doesnt divide into number of elements
	const int elem_per_warp = (width * height - 1)/warps_per_block + 1;
	const int begin = elem_per_warp * warp_id + 32 * blockIdx.x + lane;
	const int end = elem_per_warp * (warp_id + 1);
	const int step = 32 * gridDim.x; 

	// Initialize
	for (int pos = threadIdx.x; pos < (256 + 1) * R; pos += blockDim.x)
	{
		hist_private[pos] = 0;
	}

	// wait for all threads to complete
	__syncthreads();

	// Main loop
	for (int i = begin; i < end; i += step)
	{
		int pos = i < width * height ? (image[i] * 255) : 0;
			pos = pos > 255 ? 255 : pos;
		int inc = i < width * height ? 1 : 0; // read the global mem
		atomicAdd(&hist_private[off_rep + pos], inc); // vote in the shared memory
	}

	// wait for threads to end
	//__threadfence();
	__syncthreads();

	//merge per_block sub histograms and write to global memory
	for (int pos = threadIdx.x; pos < 256; pos += blockDim.x)
	{
		int sum = 0;

		for(int base = 0; base < (256 + 1) * R; base += (256 + 1))
		{
			sum += hist_private[base + pos];
		}		

		atomicAdd(histogram + pos, sum);
	}	
}



__global__ void NaiveOtsu(unsigned int *histogram, float* thresh, int width, int height)
{
	__shared__ float weight1[256];
	__shared__ float weight2[256];

	__shared__ float histogram_bin_mids[256];

	__shared__ float mean1[256];
	__shared__ float mean2[257];

	__shared__ float inter_class_variance[256];
	__shared__ int key[256];

	float bin_length = 255.0f/256.0f;
	float half_bin_length = 255.0f/512.0f;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < 256)
	{
		histogram_bin_mids[tid] = histogram[tid] * (half_bin_length + bin_length * tid); //CHANGE

		__syncthreads();

		float w1 = histogram[0];
		float w2 = width * height;

		float cs_mean1 = histogram_bin_mids[0];
		float cs_mean2 = histogram_bin_mids[255];

		// Calculate class probabilities and means
		for(int i = 1; i < tid + 1; i++)
		{
			w1 += histogram[i];
			w2 -= histogram[i-1];
			cs_mean1 += histogram_bin_mids[i];
			cs_mean2 += histogram_bin_mids[256-i-1];
		}

		weight1[tid] = w1;
		weight2[tid] = w2;

		__syncthreads();

		mean1[tid] = cs_mean1 / weight1[tid];
		mean2[256 - tid - 1] = cs_mean2 / weight2[256 - tid - 1];

		key[tid] = tid;

		__syncthreads();
		//__threadfence();
	
		if (weight1[tid] == 0 || weight2[tid] == 0)
		{
			mean1[tid] = 0;
			mean2[tid] = 0;
		}

		__threadfence();
		//__syncthreads();

		inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid+1])) * (mean1[tid] - mean2[tid+1]) * 0.0000001f;

		//__threadfence();
		__syncthreads();

		for (int stride = 1; stride < 256; stride *= 2)
		{
			if(tid % (2*stride) == 0)
			{
				if(inter_class_variance[tid] < inter_class_variance[tid+stride])
				{
					inter_class_variance[tid] = inter_class_variance[tid+stride];
					key[tid] = key[tid+stride];
				}
			}
			__syncthreads();
		}

		__threadfence();
		//__syncthreads();

		if(threadIdx.x == 0)
		{
			thresh[0] = half_bin_length + bin_length * key[0];
		}

	}
}

template <typename T>
__device__ T scan_warp(volatile T *ptr, const unsigned int idx = threadIdx.x)
{
	const unsigned int lane = threadIdx.x & 31;

	if ( lane >= 1)  ptr[ idx ] += ptr[idx - 1];
	if ( lane >= 2)  ptr[ idx ] += ptr[idx - 2];
	if ( lane >= 4)  ptr[ idx ] += ptr[idx - 4];
	if ( lane >= 8)  ptr[ idx ] += ptr[idx - 8];
	if ( lane >= 16) ptr[ idx ] += ptr[idx - 16];

	return ptr[idx];

}

template <typename T> 
__device__ T scan_block(volatile T *ptr , const unsigned int idx = threadIdx.x)
{

	const unsigned int lane = idx & 31;
	const unsigned int warpid = idx >> 5;

	// Step 1: Intra - warp scan in each warp
	T val = scan_warp( ptr , idx );
	__syncthreads ();

	// Step 2: Collect per - warp partial results
	if( lane == 31 ) ptr[ warpid ] = ptr[ idx ];
	__syncthreads ();

	// Step 3: Use 1 st warp to scan per - warp results
	if( warpid == 0 ) scan_warp( ptr , idx );
	__syncthreads ();

	// Step 4: Accumulate results from Steps 1 and 3
	if ( warpid > 0 ) val += ptr [ warpid - 1 ];
	__syncthreads ();

	// Step 5: Write and return the final result
	ptr [ idx ] = val ;
	__syncthreads ();

	return val ;

}

__global__ void OptimizedOtsu(unsigned int *histogram, float* thresh, int width, int height)
{

	__shared__ unsigned int weight1[256];
	__shared__ unsigned int weight2[256];

	__shared__ float mean1[256];
	__shared__ float mean2[257];

	__shared__ float inter_class_variance[256];
	__shared__ int key[256];

	// Faster division
	float bin_length = __fdividef(255.0, 256.0f);
	float half_bin_length = __fdividef(255.0f, 512.0f);

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < 256)
	{
		unsigned int hist_private = histogram[tid];

		weight1[tid] = hist_private;

		mean1[tid] = hist_private * (half_bin_length + bin_length * tid);
		mean2[255-tid] = hist_private * (half_bin_length + bin_length * tid);
		
		//__syncthreads();
		__threadfence();

		weight1[tid] = scan_block(weight1, tid);
		mean1[tid] = scan_block(mean1, tid);
		mean2[tid] = scan_block(mean2, tid);

		//__syncthreads();
		__threadfence();

		weight2[tid] = width * height - weight1[tid] + hist_private;
		float cs_mean2 = mean2[tid];

		__syncthreads();

		mean1[tid] = __fdividef(mean1[tid], weight1[tid]);
		mean2[255 - tid] = __fdividef(cs_mean2, weight2[255 - tid]);
	
		// Make an ordered vector 0-255
		key[tid] = tid;

		__syncthreads();
		//__threadfence();
	
		if (weight1[tid] == 0 || weight2[tid] == 0)
		{
			mean1[tid] = 0;
			mean2[tid] = 0;
		}

		//__threadfence();
		__syncthreads();

		inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid+1])) * (mean1[tid] - mean2[tid+1]) * 0.0000001f;

		//__threadfence();
		__syncthreads();

		for (int stride = (blockDim.x / 2); stride > 0; stride >>= 1)
		{
			if(threadIdx.x < stride)
			{
				if(inter_class_variance[threadIdx.x] < inter_class_variance[threadIdx.x+stride])
				{
					inter_class_variance[threadIdx.x] = inter_class_variance[threadIdx.x+stride];
					key[threadIdx.x] = key[threadIdx.x+stride];
				}
			}
			__syncthreads();
		}

		__threadfence();
		__syncthreads();

		if(threadIdx.x == 0)
		{
			thresh[0] = half_bin_length + bin_length * key[0];
		}

	}	

}
