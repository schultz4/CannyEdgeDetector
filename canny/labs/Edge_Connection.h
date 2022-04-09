#ifndef __EDGE_CONNECTION_H__
#define __EDGE_CONNECTION_H__

void threshold_detection_serial(float *image, float *weak_img, float *edges_img, 
                        double thresh_high, int width, int height);

void edge_connection_serial(float *weak_img, float *edge_img, int width, int height);



#endif
