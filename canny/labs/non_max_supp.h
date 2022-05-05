#ifndef __NON_MAX_SUPP_H__
#define __NON_MAX_SUPP_H__

// This function implements a serial version of the non-maximum supression
// function to be run on the CPU
//
// Take in gradient magnitude and phase angles for an image and output the
// gradient magnitude image with points that are not local maximums along
// the gradient edges set to 0.0
// 
// \params
//     inImage pointer to a 2D input gradient magnitude image
//     nmsImg  pointer to a 2D output non-maximum supressed image
//     gradImg pointer to a 2D input gradient angle image
//     height  number of rows of pixels in the input images
//     width   number of columns of pixels in the input images
//           
void nms(float *inImg, float *nmsImg, float *gradImg, int height, int width);

#ifdef __CUDACC__
// This function implements a CUDA version of the non-maximum supression
// function to be run on the GPU
//
// Take in gradient magnitude and phase angles for an image and output the
// gradient magnitude image with points that are not local maximums along
// the gradient edges set to 0.0
// 
// \params
//     inImage pointer to a 2D input gradient magnitude image
//     nmsImg  pointer to a 2D output non-maximum supressed image
//     gradImg pointer to a 2D input gradient angle image
//     height  number of rows of pixels in the input images
//     width   number of columns of pixels in the input images
//
// \cuda_params
//     block_dimensions Thread blocks must be 2D
//     grid_dimensions  Grid should be sized to cover image dimensions based on block dimensions
__global__ void nms_global(float *inImg, float *nmsImg, float *gradImg, int height, int width);

// This function implements a CUDA version of the non-maximum supression
// function to be run on the GPU
//
// Take in gradient magnitude and phase angles for an image and output the
// gradient magnitude image with points that are not local maximums along
// the gradient edges set to 0.0
// 
// \params
//     inImage pointer to a 2D input gradient magnitude image
//     nmsImg  pointer to a 2D output non-maximum supressed image
//     gradImg pointer to a 2D input gradient angle image
//     height  number of rows of pixels in the input images
//     width   number of columns of pixels in the input images
//
// \cuda_params
//     block_dimensions Thread blocks must be sized 16x16
//     grid_dimensions  Grid should be sized to cover image dimensions based on 16x16 blocks
__global__ void nms_opt(float *inImg, float *nmsImg, float *gradImg, int height, int width);
#endif // __CUDACC__

#endif // __NON_MAX_SUPP_H__
