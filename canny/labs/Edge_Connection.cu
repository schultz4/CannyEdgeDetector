#include "Edge_Connection.h"


void threshold_detection_serial(float *image, float *weak_img, float *edges_img, 
                        double thresh_high, int width, int height) {

    //Define lower threshold from higher threshold                      
    double thresh_low = thresh_high / 2;

	
     //Loop to all pixels in image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            // Mark strong edges
            if (image[i * width + j] >= thresh_high) {
                edges_img[i * width + j] = 1;
                weak_img[i * width + j] = 0;

            // Mark possible weak edges
            } else if (image[i * width + j] < thresh_high && image[i * width + j] >= thresh_low) {
                edges_img[i * width + j] = 0;
                weak_img[i * width + j] = 1;

            // Mark none edges
            } else if ( image[i * width + j] < thresh_low) {
                edges_img[i * width + j] = 0;
                weak_img[i * width + j] = 0;
            }
        }
    } 
}


void edge_connection_serial(float *weak_img, float *edge_img, int width, int height) {

    //Size of edge screach
    int edge_size = 1;

    // Loop to all pixels
    for (int i = 0; i < height; i++)
	{
        for (int j = 0; j < width; j++)
		{

            // find weak pixel and determine if it is adjacent to edge pixel
            // changed to add correct boundary
            if (weak_img[i * width + j] == 1)
			{
				int sum = 0;

                // Scan adjacent pixels
                for (int edge_row = -edge_size; edge_row < edge_size+1; ++edge_row)
				{
                    for ( int edge_col = -edge_size; edge_col < edge_size+1; ++edge_col)
					{
                        int curRow = i + edge_row;
                        int curCol = j + edge_col;

                        // Make sure adjacent pixels are not beyond the boundary of the image
                        if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
						{
							// Sum all pixels in 3x3 neighborhood
                            sum += edge_img[curRow * width + curCol];
                        }
                	}
				}	
		
				// Subtract center pixel from sum
				sum = sum - edge_img[i * width + j];
				
		        if (sum > 0)
				{
					weak_img[i * width + j] = 0;
		            edge_img[i * width + j] = 1;  
		        }

            }
        }
    }
}


__global__ void thresh_detection_global(float *image, float *weak_img, float *edge_img, float *thresh_high,
                                        int width, int height) {

    // Set lower threshold from high threshold
    float thresh_low = thresh_high[0] / 2;

    // Set up thread ID
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    // Go through all of the pixels and mark them as edge, non edge, or weak edge
    if ((Col < width) && (Row < height)) {

        // Edge pixels
        if (image[Row*width+Col] >= thresh_high[0]){
            edge_img[Row*width+Col] = 1;
            weak_img[Row*width+Col] = 0;

        // Weak pixels
        } else if (image[Row*width+Col] < thresh_high[0] && image[Row*width+Col] >= thresh_low) {
            edge_img[Row*width+Col] = 0;
            weak_img[Row*width+Col] = 1;

        // Non pixels
        } else if (image[Row*width+Col] < thresh_low){
            edge_img[Row*width+Col] = 0;
            weak_img[Row*width+Col] = 0;
        }
    }
}


__global__ void edge_connection_global(float *weak_img, float *edge_img, int width, int height) {

    // Size of edge screach
    int edge_size = 1;

    // Set up thread ID
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    // setup for multiple iteration
    for(int i = 0; i < 4; i++){

    if ((Col < width) && (Row < height)) 
    {

        // Find weak pixel and determine if it is adjacent to edge pixel
        // Changed to add correct boundary
        if (weak_img[Row * width + Col] == 1) 
        {

            int sum = 0;

             // Scan adjacent pixels
            for (int edge_row = -edge_size; edge_row < edge_size+1; ++edge_row) 
            {
                for ( int edge_col = -edge_size; edge_col < edge_size+1; ++edge_col) 
                {
                    int curRow = Row + edge_row;
                    int curCol = Col + edge_col;

                    // Make sure adjacent pixels are not beyond the boundary of the image
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
                    {
                        // Sum all pixels in 3x3 neighborhood
                        sum += edge_img[curRow * width + curCol];
                    }
                }
            }

            // Subtract center pixel from sum
			sum = sum - edge_img[Row * width + Col];
				
		    if (sum > 0)
			{
				weak_img[Row * width + Col] = 0;
		        edge_img[Row * width + Col] = 1;  
		    }
        }
    }
    }
}

__global__ void thresh_detection_shared(float *image, float *weak_img, float *edge_img, float *thresh_high,
                                        int width, int height) {

    // Most pixels will be zero so now default to zero and only write if weak or edge pixel

    // Set lower threshold from high threshold
    float thresh_low = __fdivide(thresh_high[0], 2);

    // Set up thread ID
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    // Go through all of the pixels and mark them as edge, non edge, or weak edge
    if ((Col < width) && (Row < height)) {

        // Edge pixels
        if (image[Row*width+Col] >= thresh_high[0]){
            edge_img[Row*width+Col] = 1;
            //weak_img[Row*width+Col] = 0;

        // Weak pixels
        } else if (image[Row*width+Col] < thresh_high[0] && image[Row*width+Col] >= thresh_low) {
            //edge_img[Row*width+Col] = 0;
            weak_img[Row*width+Col] = 1;

        // Non pixels
        //} else if (image[Row*width+Col] < thresh_low){
            //edge_img[Row*width+Col] = 0;
            //weak_img[Row*width+Col] = 0;
        }
    }
}



__global__ void edge_connection_shared(float *weak_img, float *edge_img, int width, int height) {

    // Set Tile wiidth
    const int TILE_WIDTH = 14;

    // Size of edge screach
    const int edge_size = 1;

    // Set up thread ID
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Set up row and column with edge
    int Col = tx + bx * (blockDim.x - 2 * edge_size);
    int Row = ty + by * (blockDim.y - 2 * edge_size);

    // run multiple iterations
    for(int i = 0; i < 4; i++){

    // Set bounds for edges of image
    if ((Col < width + edge_size) && (Row < height + edge_size)) {

        // Allocate shared memory
        // Has an extra pixel on each side of the tile for neibouring image seach
        __shared__ float edge_chunk[TILE_WIDTH + (2 * edge_size)][TILE_WIDTH + (2 * edge_size)];
        __shared__ float weak_chunk[TILE_WIDTH + (2 * edge_size)][TILE_WIDTH + (2 * edge_size)];

        // Set row and column of image without extra boundary pixel
        int rel_row = Row - edge_size;
        int rel_col = Col - edge_size;

        // Read image data into tile
        // If pixel is outside image, set to zero
        if ((rel_row < height) && (rel_col < width) && (rel_row >= 0) && (rel_col >=0)) {
            edge_chunk[ty][tx] = edge_img[rel_row * width + rel_col];
            weak_chunk[ty][tx] = weak_img[rel_row * width + rel_col];
        } else {
            edge_chunk[ty][tx] = 0;
            weak_chunk[ty][tx] = 0;
        }

        __syncthreads();

        // Filter out-of-bounds threads
        if ((tx >= edge_size) && (ty >= edge_size) && (ty < blockDim.y - edge_size) && (tx < blockDim.x - edge_size)) {

            // Find weak pixel and determine if it is adjacent to edge pixel
            // Changed to add correct boundary
            if (weak_chunk[ty][tx] == 1) { 
                int sum = 0;

                // Scan adjacent pixels
                for (int edge_row = -edge_size; edge_row < edge_size+1; ++edge_row) {
                    for (int edge_col = -edge_size; edge_col < edge_size+1; ++edge_col) {
                        int curRow = ty + edge_row;
                        int curCol = tx + edge_col;

                        // Make sure adjacent pixels are not beyond the boundary of the image
                        if ((curRow >= -1) && (curRow < height) && (curCol >= -1) && (curCol < width)) {

						    // Sum all pixels in 3x3 neighborhood
                            sum += edge_chunk[curRow][curCol];
                        }
                    }
                }
                // Subtract center pixel from sum
			    sum = sum - edge_chunk[ty][tx];
    
                // Change weak edge to strong edge
		        if (sum > 0) {
				    weak_img[rel_row * width + rel_col]= 0;
	                edge_img[rel_row * width + rel_col]= 1; 
                }
            }
        }
    }
    }
}
