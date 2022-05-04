#include "Otsus_Method.h"
#include <cmath>


///////////////////////////////////////////////////////////////////
// Histogram_Sequential inputs: image: an 8-bit grayscale image, //
// and the image width and height (width & height)               //
// and outputs hist: the histogram of image, and clipped at 255  //
// This is a sequential version operating only on the CPU        //
///////////////////////////////////////////////////////////////////

void Histogram_Sequential(float *image, unsigned int *hist, int width, int height)
{
    // Loop through every row
    for (int row = 0; row < height; row++)
    {
        // Loop through every column
        for (int col = 0; col < width; col++)
        {
            // Determine pixel intensity between 0 and 255
            int pos = image[row * width + col] * 255;

            // Clip the value if it's larger than 255
            pos = (pos > 255) ? 255 : pos;

            // Increment histogram count
            hist[pos]++;
        }
    }
}


///////////////////////////////////////////////////////////////////
// Otsu_Sequential inputs: histogram: a 256 bin image histogram, //
// and the image width and height (width & height)               //
// and returns the mathematically optimal threshold              //
// This is a sequential version operating only on the CPU        //
///////////////////////////////////////////////////////////////////

float Otsu_Sequential(unsigned int *histogram, int width, int height)
{

    // Initialize parameters
    float histogram_bin_mids[256];
    unsigned int weight1[256];
    unsigned int weight2[256];
    float cumsum_mean1[256];
    float cumsum_mean2[256];
    float mean1[256] {};
    float mean2[256] {};
    float inter_class_variance[255];
    float max_variance = 0;
    int thresh = 0;

    // Calculate values for bin mids
    float bin_length = 255.0f / 256.0f;
    float half_bin_length = 255.0f / 512.0f;

    // Calculate bin mids
    for (int i = 0; i < 256; i++)
    {
        histogram_bin_mids[i] = histogram[i] * (half_bin_length + bin_length * i);
    }

    // Set the first value of each weight
    weight1[0] = histogram[0];
    weight2[0] = width * height;

    // Calculate class weights
    for (int i = 1; i < 256; i++)
    {
        weight1[i] = histogram[i] + weight1[i - 1];
        weight2[i] = weight2[i - 1] - histogram[i - 1];
    }

    // Set the first value of each cumsum_mean
    cumsum_mean1[0] = histogram_bin_mids[0];
    cumsum_mean2[0] = histogram_bin_mids[255];

    // Calculate class means
    for (int i = 1; i < 256; i++)
    {
        cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];
        cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[256 - i - 1];
        mean1[i] = cumsum_mean1[i] / weight1[i];
        mean2[256 - i - 1] = cumsum_mean2[i] / weight2[256 - i - 1];
    }

    // Overwrite means in case of divide by zero
    for (int i = 1; i < 256; i++)
    {
        mean1[i] = (weight1[i] == 0) ? 0 : mean1[i];
        mean2[i] = (weight2[i] == 0) ? 0 : mean2[i];
    }

    // Calculate inter-class variance
    for (int i = 0; i < 255; i++)
    {
        inter_class_variance[i] = (weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) * (mean1[i] - mean2[i + 1]);
    }

    // Maximize inter-class variance
    for (int i = 0; i < 255; i++)
    {
        // Update max variance if next variance is larger
        if (max_variance < inter_class_variance[i])
        {
            max_variance = inter_class_variance[i];
            thresh = i;
        }
    }

    // Return normalized threshold
    return (half_bin_length + bin_length * thresh) / 255;
}


/////////////////////////////////////////////////////////////////////////////
// Otsu_Sequential_Optimized inputs: histogram: a 256 bin image histogram, //
// and the image width and height (width & height)                         //
// and returns the mathematically optimal threshold                        //
// This is a sequential version operating only on the CPU                  //
// OpenMP is used to speed up reduction and some loops                     //
/////////////////////////////////////////////////////////////////////////////

float Otsu_Sequential_Optimized(unsigned int *histogram, int width, int height)
{

    // Initialize parameters
    float histogram_bin_mids[256];
    unsigned int weight1[256];
    unsigned int weight2[256];
    float cumsum_mean1[256];
    float cumsum_mean2[256];
    float mean1[256] {};
    float mean2[256] {};
    float inter_class_variance[255];
    float max_variance = 0;
    int thresh = 0;

    // Calculate values for bin mids
    float bin_length = 255.0f / 256.0f;
    float half_bin_length = 255.0f / 512.0f;

    // Calculate bin mids
    #pragma omp parallel for private(i)
    for (int i = 0; i < 256; i++)
    {
        histogram_bin_mids[i] = histogram[i] * (half_bin_length + bin_length * i);
    }

    // Set the first value of each weight
    weight1[0] = histogram[0];
    weight2[0] = width * height;

    // Calculate class probabilities
    for (int i = 1; i < 256; i++)
    {
        weight1[i] = histogram[i] + weight1[i - 1];
        weight2[i] = weight2[i - 1] - histogram[i - 1];
    }

    // Set the first value of each cumsum_mean
    cumsum_mean1[0] = histogram_bin_mids[0];
    cumsum_mean2[0] = histogram_bin_mids[255];

    // Calculate class means
    for (int i = 1; i < 256; i++)
    {
        cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];
        cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[256 - i - 1];
        mean1[i] = cumsum_mean1[i] / weight1[i];
        mean2[256 - i - 1] = cumsum_mean2[i] / weight2[256 - i - 1];
    }

    // Overwrite means in case of divide by zero
    #pragma omp parallel for
    for (int i = 1; i < 256; i++)
    {
        mean1[i] = (weight1[i] == 0) ? 0 : mean1[i];
        mean2[i] = (weight2[i] == 0) ? 0 : mean2[i];
    }

    // Calculate inter-class variance
    #pragma omp parallel for
    for (int i = 0; i < 255; i++)
    {
        inter_class_variance[i] = (weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) * (mean1[i] - mean2[i + 1]);
    }

    // Maximize interclass variance using OpenMP reduction
    // The reduction is performed on both max_variance and thresh simultaneously
    #pragma omp parallel for private(i) reduction(max : max_variance, thresh)
    for (int i = 0; i < 255; i++)
    {
        // Update max variance if next variance is larger
        if (max_variance < inter_class_variance[i])
        {
            max_variance = inter_class_variance[i];
            thresh = i;
        }
    }

    // Return normalized threshold
    return (half_bin_length + bin_length * thresh) / 255;
}


///////////////////////////////////////////////////////////////////////
// NaiveHistogram inputs: image: an 8-bit grayscale image,           //
// and the image width and height (width & height)                   //
// and outputs histogram: the histogram of image, and clipped at 255 //
// This is a basic CUDA version operating only with global memory    //
///////////////////////////////////////////////////////////////////////

__global__ void NaiveHistogram(float *image, unsigned int *histogram, int width, int height)
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate stride
    int stride = blockDim.x * gridDim.x;

    // Let each thread stride through elements
    while (tid < width * height)
    {
        // Determine pixel intensity between 0 and 255
        unsigned int position = image[tid] * 255;

        // Clip the value if it's larger than 255
        position = (position > 255) ? 255 : position;

        // Increment histogram count
        atomicAdd(&(histogram[position]), 1);

        // Stride to next element for tid
        tid += stride;
    }
}


///////////////////////////////////////////////////////////////////////
// OptimizedHistogram inputs: image: an 8-bit grayscale image,       //
// and the image width and height (width & height)                   //
// and outputs histogram: the histogram of image, and clipped at 255 //
// This is an optimized CUDA version operating with shared memory    //
///////////////////////////////////////////////////////////////////////

__global__ void OptimizedHistogram(float *image, unsigned int *histogram, int width, int height)
{
    // Allocate shared copy of histogram
    __shared__ unsigned int histogram_private[256];

    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate stride
    int stride = blockDim.x * gridDim.x;

    // Initialize shared histogram with zeros
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        histogram_private[bin] = 0;
    }

    // Wait for all threads to finish initializing histogram
    __syncthreads();

    // Let each thread stride through elements
    while (tid < width * height)
    {
        // Determine pixel intensity between 0 and 255
        unsigned int position = image[tid] * 255;

        // Clip the value if it's larger than 255
        position = (position > 255) ? 255 : position;

        // Increment histogram count
        atomicAdd(&(histogram_private[position]), 1);

        // Stride to next element for tid
        tid += stride;
    }

    // Let all threads finish updating shared histogram
    __syncthreads();

    // Add all copies of shared histogram back to global histogram
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        atomicAdd(&(histogram[bin]), histogram_private[bin]);
    }
}


////////////////////////////////////////////////////////////////////////////
// OptimizedHistogramReplication inputs: image: an 8-bit grayscale image, //
// and the image width and height (width & height)                        //
// and outputs histogram: the histogram of image, and clipped at 255      //
// This is an advanced CUDA version operating with shared memory          //
// Many replicated copies of histogram are stored in shared memory        //
// Each copy of histogram is padded with one more byte to offset banks    //
////////////////////////////////////////////////////////////////////////////

__global__ void OptimizedHistogramReplication(float *image, unsigned int *histogram, int width, int height)
{
    //Replication factor R
    const int R = 8;

    // Allocate replicated padded sub histograms in shared mem
    __shared__ unsigned int hist_private[(256 + 1) * R];

    // Warp indexes
    const int warp_id = (int)(__fdividef(threadIdx.x, 32));
    const int lane = threadIdx.x & 31;
    const int warps_per_block = (int)(__fdividef(blockDim.x, 32));

    // Offset to per-block sub histogram
    const int off_rep = (256 + 1) * (threadIdx.x % R);

    // Set const for interleaved read access
    // to reduce the number of overlapping warps
    // account for the case where warp doesnt divide into number of elements
    const int elem_per_warp = (width * height - 1) / warps_per_block + 1;
    const int begin = elem_per_warp * warp_id + 32 * blockIdx.x + lane;
    const int end = elem_per_warp * (warp_id + 1);
    const int step = 32 * gridDim.x;

    // Initialize shared histogram with zeros
    for (int pos = threadIdx.x; pos < (256 + 1) * R; pos += blockDim.x)
    {
        hist_private[pos] = 0;
    }

    // Wait for all threads to finish initializing histogram
    __syncthreads();

    // Main loop
    for (int i = begin; i < end; i += step)
    {

        // Determine pixel intensity between 0 and 255
        int pos = i < width * height ? (image[i] * 255) : 0;

        // Clip the value if it's larger than 255
        pos = pos > 255 ? 255 : pos;

        // Determine if the position should be incremented
        int inc = i < width * height ? 1 : 0;        

        // Increment histogram count
        atomicAdd(&hist_private[off_rep + pos], inc);
    }

    // Let all threads finish updating shared histograms
    __syncthreads();

    // Add all copies of shared histogram back to global histogram
    for (int pos = threadIdx.x; pos < 256; pos += blockDim.x)
    {
        int sum = 0;

        // Sum all replicated copies of sub histogram
        for (int base = 0; base < (256 + 1) * R; base += (256 + 1))
        {
            sum += hist_private[base + pos];
        }

        atomicAdd(histogram + pos, sum);
    }
}


//////////////////////////////////////////////////////////////
// NaiveOtsu inputs: histogram: a 256 bin image histogram,  //
// and the image width and height (width & height)          //
// and outputs thresh: the mathematically optimal threshold //
// This is a basic CUDA version                             //
// Uses ineffecient parallel scan and interleaved reduction //
//////////////////////////////////////////////////////////////

__global__ void NaiveOtsu(unsigned int *histogram, float *thresh, int width, int height)
{
    // Allocate shared memories
    __shared__ unsigned int weight1[256];
    __shared__ unsigned int weight2[256];

    __shared__ float histogram_bin_mids[256];

    __shared__ float mean1[256];
    __shared__ float mean2[257];

    __shared__ float inter_class_variance[256];
    __shared__ int key[256];

    // Calculate values for bin mids
    float bin_length = 255.0f / 256.0f;
    float half_bin_length = 255.0f / 512.0f;

    // Calculate global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 256)
    {
        // Calculate bin mids
        histogram_bin_mids[tid] = histogram[tid] * (half_bin_length + bin_length * tid);

        // Wait for all threads to finish initializing histogram bin mids
        __syncthreads();

        // Set the first value of each weight
        unsigned int w1 = histogram[0];
        unsigned int w2 = width * height;

        // Set the first value of each cumsum_mean
        float cs_mean1 = histogram_bin_mids[0];
        float cs_mean2 = histogram_bin_mids[255];

        // Calculate class probabilities and cumulative sum means
        for (int i = 1; i < tid + 1; i++)
        {
            w1 += histogram[i];
            w2 -= histogram[i - 1];
            cs_mean1 += histogram_bin_mids[i];
            cs_mean2 += histogram_bin_mids[256 - i - 1];
        }

        // Calculate class weights
        weight1[tid] = w1;
        weight2[tid] = w2;

        // Wait for all threads to finish initializing updating weights
        __syncthreads();

        // Calculate class means  
        mean1[tid] = cs_mean1 / weight1[tid];
        mean2[256 - tid - 1] = cs_mean2 / weight2[256 - tid - 1];

        // Initialize key for reduction
        key[tid] = tid;

        // Wait for all threads to finish initializing initializing key
        __syncthreads();

        // Overwrite means in case of divide by zero
        if (weight1[tid] == 0 || weight2[tid] == 0)
        {
            mean1[tid] = 0;
            mean2[tid] = 0;
        }

        // Wait for all threads to finish updating means
        __syncthreads();

        // Calculate inter-class variance
        inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid + 1])) * (mean1[tid] - mean2[tid + 1]) * 0.0000001f;

        // Wait for all threads to finish updating inter-class variance      
        __syncthreads();

        // Maximize interclass variance using interleaved reduction with key
        for (int stride = 1; stride < 256; stride *= 2)
        {
            if (tid % (2 * stride) == 0)
            {
                if (inter_class_variance[tid] < inter_class_variance[tid + stride])
                {
                    inter_class_variance[tid] = inter_class_variance[tid + stride];
                    key[tid] = key[tid + stride];
                }
            }
            __syncthreads();
        }

        // Wait for all threads to finish reductions
        __syncthreads();

        // Return normalized threshold
        if (threadIdx.x == 0)
        {
            thresh[0] = (half_bin_length + bin_length * key[0]) / 255;
        }
    }
}


//////////////////////////////////////////////////////////////////
// scan_warp inputs: ptr: data for scan to be performed on,     //
// and the idx: the threadID of a thread assigned parallel scan //
// and returns ptr: data after scan has been performed          //
// This is a per warp scan, so no synchronization is needed     //
// Scan type is inclusive, and performs in Nlog(N) steps        //
//////////////////////////////////////////////////////////////////

template <typename T>
__device__ T scan_warp(volatile T *ptr, const unsigned int idx = threadIdx.x)
{
    const unsigned int lane = threadIdx.x & 31;

    // Unroll last iteration of inclusive scan
    // No synchronization because it's within a warp
    if (lane >= 1)
        ptr[idx] += ptr[idx - 1];
    if (lane >= 2)
        ptr[idx] += ptr[idx - 2];
    if (lane >= 4)
        ptr[idx] += ptr[idx - 4];
    if (lane >= 8)
        ptr[idx] += ptr[idx - 8];
    if (lane >= 16)
        ptr[idx] += ptr[idx - 16];

    return ptr[idx];
}


//////////////////////////////////////////////////////////////////
// scan_block inputs: ptr: data for scan to be performed on,    //
// and the idx: the threadID of a thread assigned parallel scan //
// and returns val: data after scan has been performed          //
// This function uses scan_warp to scan a block of data         //
// Scan type is inclusive, and performs in Nlog(N) steps        //
//////////////////////////////////////////////////////////////////

template <typename T>
__device__ T scan_block(volatile T *ptr, const unsigned int idx = threadIdx.x)
{

    const unsigned int lane = idx & 31;
    const unsigned int warpid = idx >> 5;

    // Step 1: Intra - warp scan in each warp
    T val = scan_warp(ptr, idx);
    __syncthreads();

    // Step 2: Collect per - warp partial results
    if (lane == 31)
        ptr[warpid] = ptr[idx];
    __syncthreads();

    // Step 3: Use 1 st warp to scan per - warp results
    if (warpid == 0)
        scan_warp(ptr, idx);
    __syncthreads();

    // Step 4: Accumulate results from Steps 1 and 3
    if (warpid > 0)
        val += ptr[warpid - 1];
    __syncthreads();

    // Step 5: Write and return the final result
    ptr[idx] = val;
    __syncthreads();

    return val;
}


/////////////////////////////////////////////////////////////////////
// OptimizedOtsu inputs: histogram: a 256 bin image histogram,     //
// and the image width and height (width & height)                 //
// and outputs thresh: the mathematically optimal threshold        //
// This is an advanced CUDA version                                //
// Uses intra-warp parallel scan and sequential parallel reduction //
/////////////////////////////////////////////////////////////////////

__global__ void OptimizedOtsu(unsigned int *histogram, float *thresh, int width, int height)
{
    // Allocate shared memories
    __shared__ unsigned int weight1[256];
    __shared__ unsigned int weight2[256];

    __shared__ float mean1[256];
    __shared__ float mean2[257];

    __shared__ float inter_class_variance[256];
    __shared__ int key[256];

    // Calculate values for bin mids with faster division
    float bin_length = __fdividef(255.0, 256.0f);
    float half_bin_length = __fdividef(255.0f, 512.0f);

    // Calculate global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 256)
    {
        // Read one value from histogram into local memory
        unsigned int hist_private = histogram[tid];

        // Set the first value of weight1
        weight1[tid] = hist_private;

        // Populate shared memory class means
        mean1[tid] = hist_private * (half_bin_length + bin_length * tid);
        mean2[255 - tid] = hist_private * (half_bin_length + bin_length * tid);

        // Wait for threads to finish memory transactions
        __threadfence();

        // Perform in-place inclusive scan
        weight1[tid] = scan_block(weight1, tid);
        mean1[tid] = scan_block(mean1, tid);
        mean2[tid] = scan_block(mean2, tid);

        // Wait for threads to finish memory transactions
        __threadfence();

        // Calculate class 2 weight and temporary mean
        weight2[tid] = width * height - weight1[tid] + hist_private;
        float cs_mean2 = mean2[tid];

        // Wait for threads to finish writing to weight 2
        __syncthreads();

        // Calculate class means
        mean1[tid] = __fdividef(mean1[tid], weight1[tid]);
        mean2[255 - tid] = __fdividef(cs_mean2, weight2[255 - tid]);

        // Initialize key for reduction
        key[tid] = tid;

        // Wait for all threads to finish initializing initializing key
        __syncthreads();

        // Overwrite means in case of divide by zero
        if (weight1[tid] == 0 || weight2[tid] == 0)
        {
            mean1[tid] = 0;
            mean2[tid] = 0;
        }

        // Wait for all threads to finish updating means
        __syncthreads();

        // Calculate inter-class variance
        inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid + 1])) * (mean1[tid] - mean2[tid + 1]) * 0.0000001f;

        // Wait for all threads to finish updating inter-class variance  
        __syncthreads();

        // Maximize interclass variance using coalesced reduction with key
        for (int stride = (blockDim.x / 2); stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                if (inter_class_variance[threadIdx.x] < inter_class_variance[threadIdx.x + stride])
                {
                    inter_class_variance[threadIdx.x] = inter_class_variance[threadIdx.x + stride];
                    key[threadIdx.x] = key[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        // Wait for all threads to finish reductions
        __syncthreads();

        // Return normalized threshold
        if (threadIdx.x == 0)
        {
            thresh[0] = __fdividef(half_bin_length + bin_length * key[0], 255.0f);
        }
    }
}