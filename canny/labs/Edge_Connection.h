#ifndef __EDGE_CONNECTION_H__
#define __EDGE_CONNECTION_H__

void threshold_detection_serial(float *image, float *weak_img, float *edges_img,
                                float thresh_high, int width, int height);

void edge_connection_serial(float *weak_img, float *edge_img, int width, int height);

#ifdef __CUDACC__

__global__ void thresh_detection_global(float *image, float *weak_img, float *edge_img, float *thresh_high, int width, int height);
__global__ void edge_connection_global(float *weak_img, float *edge_img, int width, int height);
__global__ void thresh_detection_shared(float *image, float *weak_img, float *edge_img, float *thresh_high, int width, int height);
__global__ void edge_connection_shared(float *weak_img, float *edge_img, int width, int height);

#endif

#endif
