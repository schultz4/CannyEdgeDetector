#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>

#define WIDTH 300
#define HEIGHT 300
#define IN_FILE  "/home/u9/catmcqueen/ece569_hpc/proj/CannyEdgeDetector/BSDS300/images/train/resized/100075_resized.jpg"
#define REQUIRED_COMPS 3




class ppm {
         bool flag_alloc;
         void init();
         //info about the PPM file (height and width)
         unsigned int nr_lines;
         unsigned int nr_columns;

     public:
         //arrays for storing the R,G,B values
         unsigned char *r;
         unsigned char *g;
         unsigned char *b;
         //
         unsigned int height;
         unsigned int width;
         unsigned int max_col_val;
         //total number of elements (pixels)
         unsigned int size;

         ppm();
         //create a PPM object and fill it with data stored in fname
         ppm(const std::string &fname);
         //create an "empty" PPM image with a given width and height;the R,G,B arrays are filled with zeros
         ppm(const unsigned int _width, const unsigned int _height);
         //free the memory used by the R,G,B vectors when the object is destroyed
         ~ppm();
         //read the PPM image from fname
         void read(const std::string &fname);
         //write the PPM image in fname
         void write(const std::string &fname);
     };





int main (int argc, char *argv[]) {
    //unsigned char* image = NULL;
    //char *inputImageFile;
    //float *hostInputImageData;
    //float *hostOutputImageData;
    //float *deviceInputImageData;
    //float *deviceOutputImageData;
    //int width = 0;
    //int height = 0;
    //int actualComps = 0;

    ppm image(IN_FILE);
    std::cout << "width: " << image.width << "and height: " <<image.height << std::endl;

}
