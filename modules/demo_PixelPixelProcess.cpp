#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "PixelGradient.h"
#include "PixelSelector.h"

using namespace std;
using namespace cv;

int main(){

  std::string path = "../data";
  std::string data_path;

  if (path.find_last_of("/") != path.size() - 1)
    data_path = path + "/";

//  cv::Mat img0 = cv::imread(data_path + "rect.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//  cv::Mat img0 = cv::imread(data_path + "dist_tum0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//  cv::Mat img0 = cv::imread(data_path + "00223.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//  cv::Mat img0 = cv::imread(data_path + "02183.png",CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img0 = cv::imread(data_path + "tum0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//  cvResize(img0, img0, 0.5);
//  img0.resize(Size(55, 22));
//  resize(img0, img0, Size(img0.rows/2,img0.cols/2));


//TODO 计算梯度
  PixelGradient *pixelGradent_ = new PixelGradient;
  cv::Mat gradent_0;
//  cv::Mat gradent_1;
  pixelGradent_->computeGradents(img0, gradent_0);

  imshow("gradent_1", gradent_0);


//TODO 选择像素
  PixelSelector sel(pixelGradent_->wG[0],pixelGradent_->hG[0]);
  Pnt* points[PYR_LEVELS]; 		//!< 每一层上的点类, 是第一帧提取出来的
  int numPoints[PYR_LEVELS];  	//!< 每一层的点数目

  float *statusMap = new float[pixelGradent_->wG[0] * pixelGradent_->hG[0]];

  bool *statusMapB = new bool[pixelGradent_->wG[0] * pixelGradent_->hG[0]];
  float densities[] = {0.03,0.05,0.15,0.5,1}; // 不同层取得点密度
  sel.currentPotential = 3; // 设置网格大小，3*3大小格
  int *w = &pixelGradent_->wG[0];
  int *h = &pixelGradent_->hG[0];

  for(int lvl=0; lvl<pixelGradent_->pyrLevelsUsed; lvl++)
  {
    int npts;
    if(lvl == 0) // 第0层提取特征像素
      npts = sel.makeMaps(pixelGradent_, statusMap, densities[lvl] * w[0] * h[0], 1, true, 2);
    else  // 其它层则选出goodpoints
      npts = makePixelStatus(pixelGradent_->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);

    points[lvl] = new Pnt[npts];

    // set idepth map to initially 1 everywhere.
    int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
    Pnt *pl = points[lvl];  // 每一层上的点
    int nl = 0;
    // 要留出pattern的空间, 2 border
//[ ***step 3*** ] 在选出的像素中, 添加点信息
    for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
      for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
        //if(x==2) printf("y=%d!\n",y);
        // 如果是被选中的像素
        if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
          //assert(patternNum==9);
          pl[nl].u = x + 0.1;   //? 加0.1干啥
          pl[nl].v = y + 0.1;
          pl[nl].idepth = 1;
          pl[nl].iR = 1;
          pl[nl].isGood = true;
          pl[nl].energy.setZero();
          pl[nl].lastHessian = 0;
          pl[nl].lastHessian_new = 0;
          pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

          Eigen::Vector3f *cpt = pixelGradent_->dIp[lvl] + x + y * w[lvl]; // 该像素梯度
          float sumGrad2 = 0;
          // 计算pattern内像素梯度和
          for (int idx = 0; idx < patternNum; idx++) {
            int dx = patternP[idx][0]; // pattern 的偏移
            int dy = patternP[idx][1];
            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
            sumGrad2 += absgrad;
          }

          // float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
          // pl[nl].outlierTH = patternNum*gth*gth;
          //! 外点的阈值与pattern的大小有关, 一个像素是12*12
          //? 这个阈值怎么确定的...
          pl[nl].outlierTH = patternNum * setting_outlierTH;

          nl++;
          assert(nl <= npts);
        }
      }

    numPoints[lvl] = nl; // 点的数目,  去掉了一些边界上的点
  }

//  imshow("diff", diff);
//  imshow("gradent_0", gradent_0);
//  cvWaitKey(0);

  return 0;
}

