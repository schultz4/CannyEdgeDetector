#include "non_max_supp.h"

float maxSupp(float center, float p1, float p2, float p3, float p4)
{
  if (center >= p1 && center >= p2 && center >= p3 && center >= p4)
  {
    return center;
  }
  else
  {
    return 0.0;
  }
}

void nms(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
  //Mat             sMatGradPad;
  //vector<Point2i> vecIndex;
  //int             s32Pad      = ks32NonMaxFiltLen >> 1;
  //Point pOne, pTwo, pThree, pFour;
  //vector<uchar> vecPixel(ks32NonMaxFiltLen);

  //copyMakeBorder(sMatGrad, sMatGradPad, s32Pad, s32Pad, s32Pad, s32Pad, BORDER_REPLICATE);
  //sMatOutputGrad.create(Size(sMatGrad.cols, sMatGrad.rows), CV_8UC1);

  for(int i = 0; i < width; ++i)
  {
    for(int j = 0; j < height; ++j)
    {
      float angle = *(gradImg + j*width + i);
      float p1 = -1.0, p2 = -1.0, p3 = -1.0, p4 = -1.0;
      unsigned int fAngle = 0;
      if (angle > 180)
        angle = angle - 180;

      if ((angle > 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180))
        fAngle = 0;
      else if (angle > 22.5 && angle <= 67.5)
        fAngle = 45;
      else if (angle > 67.5 && angle <= 112.5)
        fAngle = 90;
      else if (angle > 112.5 && angle <= 157.5)
        fAngle = 135;

      switch( fAngle ) 
      {
        case 0:
  //        pOne   = (Point(0, 1)); pTwo   = (Point(0, 2)); pThree = (Point(-1, 0)); pFour  = (Point(-2, 0));
          p1 = *(grad + (j + 1)*width + i);
          p2 = *(grad + (j + 2)*witdh + i);
          p3 = *(grad + j*witdh + i - 1);
          p4 = *(grad + j*witdh + i - 2);
          *(nmsImg + j*width + i) = maxSupp(angle, p1, p2, p3, p4);
          break;
        case 45:
  //        pOne   = (Point(-1, -1)); pTwo   = (Point(-2, -2)); pThree = (Point(1, 1)); pFour  = (Point(2, 2));
          break;
        case 90:
  //        pOne   = (Point(-1, 0)); pTwo   = (Point(-2, 0)); pThree = (Point(1, 0)); pFour  = (Point(2, 0));
          break;
        case 135:
  //        pOne   = (Point(-1, 1)); pTwo   = (Point(-2, 2)); pThree = (Point(1, -1)); pFour  = (Point(2, -2));
          break;
        default:
          break;
      }
    }
  }
  //for (int i = s32Pad; i < sMatGradPad.rows - s32Pad; i++){
  //  for (int j = s32Pad; j < sMatGradPad.cols - s32Pad; j++){
  //    int ii = i - s32Pad;
  //    int jj = j - s32Pad;
  //    unsigned short u16Angle = sMatEdgeOrientation.at<unsigned short>(ii, jj);
  //    switch ( u16Angle ) {
  //      case 0: 
  //        pOne   = (Point(0, 1)); pTwo   = (Point(0, 2)); pThree = (Point(-1, 0)); pFour  = (Point(-2, 0));
  //        break;
  //      case 45:  
  //        pOne   = (Point(-1, -1)); pTwo   = (Point(-2, -2)); pThree = (Point(1, 1)); pFour  = (Point(2, 2));
  //        break;
  //      case 90:  
  //        pOne   = (Point(-1, 0)); pTwo   = (Point(-2, 0)); pThree = (Point(1, 0)); pFour  = (Point(2, 0));
  //        break;
  //      case 135: 
  //        pOne   = (Point(-1, 1)); pTwo   = (Point(-2, 2)); pThree = (Point(1, -1)); pFour  = (Point(2, -2));
  //        break;
  //      default:  
  //        break;
  //    }       /* -----  end switch  ----- */

  //    //four neighboring pixels
  //    vecPixel.at(0) = sMatGradPad.at<uchar>(i + pTwo.x   , j + pTwo.y);
  //    vecPixel.at(1) = sMatGradPad.at<uchar>(i + pOne.x   , j + pOne.y);
  //    vecPixel.at(2) = sMatGradPad.at<uchar>(i , j);
  //    vecPixel.at(3) = sMatGradPad.at<uchar>(i + pThree.x , j + pThree.y);
  //    vecPixel.at(4) = sMatGradPad.at<uchar>(i + pFour.x  , j + pFour.y);

  //    vector<uchar>::iterator itr = max_element(vecPixel.begin(), vecPixel.end());
  //    int s32Index = itr - vecPixel.begin();
  //    // Index of the current pixel in stored at index=2
  //    if ( 2 != s32Index ) {
  //      sMatOutputGrad.at<uchar>(ii, jj) = 0;
  //    } else {
  //      sMatOutputGrad.at<uchar>(ii, jj) = sMatGradPad.at<uchar>(i, j);
  //    }

  //  }
  //}
}

