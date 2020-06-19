//
// Created by lab on 20-3-24.
//

#ifndef DSO_MODULES_PIXELGRADENT_H
#define DSO_MODULES_PIXELGRADENT_H
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "Utils.h"

using namespace cv;



enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};
typedef Eigen::Matrix<float,2,1> Vec2f_;

class PixelGradient
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PixelGradient();
  ~PixelGradient();
  float* absSquaredGrad[PYR_LEVELS];
  Eigen::Vector3f* dI;
  Eigen::Vector3f* dIp[PYR_LEVELS];
  int wG[PYR_LEVELS];
  int hG[PYR_LEVELS];
  int pyrLevelsUsed;

  void computeGradents(const Mat img, Mat &gradents, int level = 1);//, CalibHessian* HCalib);
 private:

};


template<typename T>
class MinimalImage
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int w;
  int h;
  T* data;

  /*
   * creates minimal image with own memory
   */
  inline MinimalImage(int w_, int h_) : w(w_), h(h_)
  {
    data = new T[w*h];
    ownData=true;
  }

  /*
   * creates minimal image wrapping around existing memory
   */
  inline MinimalImage(int w_, int h_, T* data_) : w(w_), h(h_)
  {
    data = data_;
    ownData=false;
  }

  inline ~MinimalImage()
  {
    if(ownData) delete [] data;
  }

  inline MinimalImage* getClone()
  {
    MinimalImage* clone = new MinimalImage(w,h);
    memcpy(clone->data, data, sizeof(T)*w*h);
    return clone;
  }


  inline T& at(int x, int y) {return data[(int)x+((int)y)*w];}
  inline T& at(int i) {return data[i];}

  inline void setBlack()
  {
    memset(data, 0, sizeof(T)*w*h);
  }

  inline void setConst(T val)
  {
    for(int i=0;i<w*h;i++) data[i] = val;
  }

  inline void setPixel1(const float &u, const float &v, T val)
  {
    at(u+0.5f,v+0.5f) = val;
  }

  inline void setPixel4(const float &u, const float &v, T val)
  {
    at(u+1.0f,v+1.0f) = val;
    at(u+1.0f,v) = val;
    at(u,v+1.0f) = val;
    at(u,v) = val;
  }

  inline void setPixel9(const int &u, const int &v, T val)
  {
    at(u+1,v-1) = val;
    at(u+1,v) = val;
    at(u+1,v+1) = val;
    at(u,v-1) = val;
    at(u,v) = val;
    at(u,v+1) = val;
    at(u-1,v-1) = val;
    at(u-1,v) = val;
    at(u-1,v+1) = val;
  }

  inline void setPixelCirc(const int &u, const int &v, T val)
  {
    int w_ = 1;
    int h_ = 2;
    for(int i=-h_;i<=h_;i++)
    {
      at(u+h_,v+i) = val;
      at(u-h_,v+i) = val;
      at(u+w_,v+i) = val;
      at(u-w_,v+i) = val;

      at(u+i,v-h_) = val;
      at(u+i,v+h_) = val;
      at(u+i,v-w_) = val;
      at(u+i,v+w_) = val;
    }
  }

 private:
  bool ownData;
};

typedef Eigen::Matrix<unsigned char,3,1> Vec3b_;
typedef MinimalImage<float> MinimalImageF;
typedef MinimalImage<Vec3f> MinimalImageF3;
typedef MinimalImage<unsigned char> MinimalImageB;
typedef MinimalImage<Vec3b_> MinimalImageB3;
typedef MinimalImage<unsigned short> MinimalImageB16;

//####################








#endif //DSO_MODULES_PIXELGRADENT_H
