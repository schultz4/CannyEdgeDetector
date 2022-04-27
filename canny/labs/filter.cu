

bal__ void Make2DGaussFilter(float *inFilter, int filterSize)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    filterCenter = (filterSize - 1.0) / 2.0;
    int filterSum = 0;

    // first make the filter
    // filter = (float *)malloc(fltSize*fltSize*sizeof(float));
    //  filters are square
    if (x < width * height)
    {
        for (int i = 0; i < filterSize; ++i)
        {
            inFilter[i * filterSize + j] = exp(-(pow(i - filterCenter, 2) + pow(j - filterCenter, 2)) / (2 * filterSize * filterSize)) / (2 * M_PI * filterSize * filterSize);
            filterSum += inFilter[i * filterSize + j];
        }
    }
    //__syncthreads__;

    for (i = 0; i < filterSize * filterSize; ++i)
    {
        inFilter[i] /= filterSum;
    }
}

__global__ void Make1DGaussFilter(float *inFilter, int filterSize)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    filterCenter = (filterSize - 1.0) / 2.0;
    int filterSum = 0;

    for (int i = 0; i < filterSize; ++i)
    {
        inFilter[i] = exp(-pow((i - filterCenter) / filterSize, 2) / 2) / (filterSize * sqrt(2 * M_PI));
        filterSum += inFilter[i];
    }

    // then normalize the filter
    for (i = 0; i < filterSize * filterSize; ++i)
    {
        inFilter[i] /= filterSum;
    }
}
