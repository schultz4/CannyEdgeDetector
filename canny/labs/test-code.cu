#include "filters.h"
#include "test-code.h"

void GradientSobelPhaseOnlySerial(float *inImg, float *phase, int height, int width, size_t filterSize)
{

    int halfFilter = (int)(filterSize) / 2;

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

    // iterate over rows and columns of the image
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
                    if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                    {
                        sumy += inImg[cur_row * width + cur_col] * fmat_y[j][k] * 255;
                        sumx += inImg[cur_row * width + cur_col] * fmat_x[j][k] * 255;
                    }
                }
            }

            phase[row * width + col] = atan(sumx / sumy) * 180 / M_PI; // gradient at pixel
        }
    }
}
