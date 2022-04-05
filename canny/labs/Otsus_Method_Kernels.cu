__global__ void NaiveHistogram(unsigned char *image, int width, int height, int *hist)
{
    // Calculate threadID in x and y directions
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate linear threadID
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Calculate number of threads in x and y directions
    int num_threads_x = blockDim.x * gridDim.x;
    int num_threads_y = blockDim.y * gridDim.y;

    // Calculate linear blockID
    int bid = blockIdx.x + blockIdx.y * gridDim.x;

    // Loop through all pixels
    for (int col = tid_x; col < width; col += num_threads_x)
    {
        for (int row = tid_y; row < height; row += num_threads_y)
        {
            unsigned char pos = image[tid];
            atomicAdd(&hist[pos], 1);
        }
    }

}