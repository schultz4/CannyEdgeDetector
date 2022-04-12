#include "Edge_Connection.h"

// Note: I have set this code so that the edges are the value of 1 and non edges are the value of 0.
// If this needs to change just replace the values in edge_img to whatever they need to be.

void threshold_detection_serial(float *image, float *weak_img, float *edges_img, 
                        double thresh_high, int width, int height) {

    //Define lower threshold from higher threshold                       
    double thresh_low = thresh_high - 0.2;

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
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            // find weak pixel and determine if it is adjacent to edge pixel
            // changed to add correct boundary
            if (weak_img[i * width + j] == 1) {
                int strong_edge = 0;

                // Scan adjacent pixels
                for (int edge_row = -edge_size; edge_row < edge_size+1; ++edge_row) {
                    for ( int edge_col = -edge_size; edge_col < edge_size+1; ++edge_col) {
                        int curRow = i + edge_row;
                        int curCol = j + edge_col;

                        // Make sure adjacent pixels are not beyond the boundary of the image
                        if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                            if (edge_img[curRow * width + curCol] == 1) {
                                strong_edge = 1;
                            }
                        }

                        // Make as edge if adjacent pixel is and edge pixel
                        if (strong_edge == 1) {
                            edge_img[curRow * width + curCol] = 1;  
                        }  
                    }
                }
            }
        }
    }
}


/*
__global__ void thresh_detection_global_kernel(float *image, float *weak_img, float *edge_img, double thresh_high,
                                        int width, int height) {

    // Set lower threshold from high threshold
    float thresh_low = thresh_high / 2;

    // Set up thread ID
    int Col = threadIdx.x + blockIdx.x * blockDim.x
    int Row = threadIdx.y + blockIdx.y * blockDim.y

    // Go through all of the pixels and mark them as edge, non edge, or weak edge
    if ((Col < width) && (Row < heigth)) {

        // Edge pixels
        if (image[Row*width+Col] >= thresh_high){
            edge_img[Row*width+Col] = 1;
            weak_img[Row*width+Col] = 0;

        // Weak pixels
        } else if (image[Row*width+Col] < thresh_high && image[Row*width+Col] >= thresh_low) {
            edge_img[Row*width+Col] = 0;
            weak_img[Row*width+Col] = 1;

        // Non pixels
        } else if (image[Row*width+Col] < thresh_low){
            edge_img[Row*width+Col] = 0;
            weak_img[Row*width+Col] = 0;
        }
    }
}


__global__ void edge_connection_global_kernel(float *weak_img, float *edge_img, int width, int height) {

    // Size of edge screach
    int edge_size = 1;

    // Set up thread ID
<<<<<<< Updated upstream
    int Col = threadIdx.x + blockIdx.x * blockDim.x
    int Row = threadIdx.y + blockIdx.y * blockDim.y
    if ((Col < width) && (Row < heigth)) {
=======
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if ((Col < width) && (Row < height)) 
    {
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                        if (edge_img[curRow * width + curCol] == 1) {
                            strong_edge = 1;
                        }
                    }

                    // Make as edge if adjacent pixel is and edge pixel
                    if (strong_edge == 1) {
                        edge_img[curRow * width + curCol] = 1;  
                        
=======
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
                    {
                        // Sum all pixels in a 3X3 area
                        sum += edge_img[curRow * width + curCol]; 
>>>>>>> Stashed changes
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

*/
