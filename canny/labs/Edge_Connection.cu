#include "Edge_Connection.h"

// Note: I have set this code so that the edges are the value of 1 and non edges are the value of 0.
// If this needs to change just replace the values in edge_img to whatever they need to be.

void threshold_detection_serial(float *image, unsigned char *weak_img, unsigned char *edges_img, 
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


void edge_connection_serial(unsigned char *weak_img, unsigned char *edge_img, int width, int height) {

    //Size of edge screach
    int edge_size = 1;

    // Loop to all pixels
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            // find weak pixel and determine if it is adjacent to edge pixel
            // changed to add correct boundary
            if (weak_img[i * width + j] = 1) {
                int strong_edge = 0;

                // Scan adjacent pixels
                for (int edge_row = -edge_size; edge_row < edge_size+1; ++edge_row) {
                    for ( int edge_col = -edge_size; edge_col < edge_size+1; ++edge_col) {
                        int curRow = i + edge_row;
                        int curCol = j + edge_col;

                        // Make sure adjacent pixels are not beyond the boundary of the image
                        if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                            if (edge_img[curRow * width + curCol] = 1) {
                                strong_edge = 1;
                            }

                            // Make as edge if adjacent pixel is and edge pixel
                            if (strong_edge = 1) {
                                edge_img[curRow * width + curCol] = 1;  
                            }
                        }
                    }
                }
            }
        }
    }
}
