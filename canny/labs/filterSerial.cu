// For the gaussian Blur = Conv2DSerial
// for the sobel filter and Gradients = GradientSobelSerial
//
#define FILTERSIZE 3

__global__ void Conv2DSerial(int *inImg, int *outImg, double filter[FILTERSIZE][FILTERSIZE], int width, int height, int filterSize)
{
    // find center position of kernel (half of kernel size)
    int filterHalf = filterSize / 2;

    // iterate over rows and coluns of the image
    for (int row = 0; row < height; ++row) // rows
    {
        for (int col = 0; col < width; ++col) // columns
        {
            int start_col = col - halfFilter;
            int start_row = row - halfFilter;
            int pixelvalue = 0;

            // then for each pixel iterate through the filter
            for (int j = 0; j < filterSize; ++j) // filter rows
            {
                for (int k = 0; k < filterSize; ++k) // kernel columns
                {
                    int cur_row = start_row + j;
                    int cur_col = start_col + k;
                    if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                    {
                        pixelvalue += inImg[cur_row * width + cur_col] * filter[j][k];
                    }
                }
            }
            outImg[row * width + col] = (int)(pixelvalue);
        }
    }
}
__global__ void GradientSobelSerial(int *inImg, float *sobelImg, float *gradientImg, int height, int width)
{

    int filterSize = (int)FILTERSIZE;
    int halfFilter = (int)filterSize / 2;

    // To detect horizontal lines, G_x.
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    // To detect vertical lines, G_y
    const int fmat_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    // iterate over rows and coluns of the image
    for (int row = 0; row < height; ++row) // rows
    {
        for (int col = 0; col < width; ++col) // columns
        {
            double sumx = 0;
            double sumy = 0;

            int start_col = col - halfFilter;
            int start_row = row - halfFilter;

            // now do the filtering
            for (int j = 0; j < filterSize; ++j)
            {
                for (int k = 0; k < filterSize; ++k)
                {
                    int cur_row = start_row + j;
                    int cur_col = start_col + k;

                    // only count the ones that are inside the boundaries
                    if (cur_row >= 0 && cur_row < height)
                    {
                        sumy += inImg[cur_row * width + cur_col] * fmat_y[j][k];
                    }
                    if (cur_col >= 0 && cur_col < width)
                    {
                        sumx += inImg[cur_row * width + cur_col] * fmat_x[j][k];
                    }
                }
            }
            int value = sqrt(sumx * sumx + sumy * sumy);
            if (value > 255)
            {
                value = 255;
            }
            if (value < 0)
            {
                value = 0;
            }
            sobelImg[row * width + col] = value;                             // output of the sobel filt at this index
            gradientImg[row * width + col] = atan(sumx / sumy) * 180 / M_PI; // graient at pixel
        }
    }
}
