//
// Created by lab on 20-3-24.
//
#include "PixelGradient.h"
#include "Utils.h"
PixelGradient::PixelGradient() {
  pyrLevelsUsed = 4;
}


PixelGradient::~PixelGradient() {
}

void PixelGradient::computeGradents(const Mat img, Mat &gradents, int level)//, CalibHessian* HCalib)
{
  //*******************//  pre parameter
  int w = img.cols;
  int h = img.rows;

//  Eigen::Vector3f *dI;

  //int wG[PYR_LEVELS], hG[PYR_LEVELS];

  wG[0] = w;
  hG[0] = h;

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    wG[level] = wG[level - 1] / 2;
    hG[level] = hG[level - 1] / 2;
  }

  int tolPixel = img.cols * img.rows;

  unsigned char *color = new unsigned char[tolPixel];

//  for (int j = 0; j < img.rows; ++j) {
//    for (int i = 0; i < img.cols; ++i) {
//      color[j*img.cols+i] = (float)(img.at<uchar>(j, i));
//    }
//  }

//  for (int i = 0; i < tolPixel; ++i) {
//    color[i] = (float) img.data[i];
//  }

  memcpy(color, img.data, img.rows*img.cols);


  //*******************// 每一层创建图像值, 和图像梯度的存储空间
  for (int i = 0; i < pyrLevelsUsed; i++) {
    dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
    absSquaredGrad[i] = new float[wG[i] * hG[i]];
  }
  dI = dIp[0]; // 原来他们指向同一个地方,图像导数


  // make d0


  for (int i = 0; i < w * h; i++)
    dI[i][0] = (float)color[i];

  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小
    Eigen::Vector3f *dI_l = dIp[lvl];

    float *dabs_l = absSquaredGrad[lvl];
    if (lvl > 0) {
      int lvlm1 = lvl - 1;
      int wlm1 = wG[lvlm1]; // 列数
      Eigen::Vector3f *dI_lm = dIp[lvlm1];


      // 像素4合1, 生成金字塔
      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
              dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        }
    }

    float mean_grident = 0.0f;
    for (int idx = wl; idx < wl * (hl - 1); idx++) // 第二行开始
    {
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      dI_l[idx][1] = dx; // 梯度
      dI_l[idx][2] = dy;

      dabs_l[idx] = dx * dx + dy * dy; // 梯度平方
      mean_grident += sqrtf(dabs_l[idx]);
//      if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
//      {
//        //! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) / V(x)
//        float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
//        dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
//      }

    }
    std::cout << "mean grident of level" << lvl << "=" << mean_grident / (wl * hl) << std::endl;

  }

  int wl = wG[level], hl = hG[level]; // 该层图像大小
  Eigen::Vector3f *dI_l = dIp[level];
  float *dabs_l = absSquaredGrad[level];
  gradents = cv::Mat(hl, wl, CV_LOAD_IMAGE_GRAYSCALE);
  toCvMat(dabs_l, gradents);
//  for (int j = 0; j < hl; ++j)
//    for (int i = 0; i < wl; ++i) {
////        img_gradent.at<uchar>(j, i) = (uchar) dI_l[j * wl + i][1];
//      gradents.at<uchar>(j, i) = (uchar) dabs_l[j * wl + i];
//    }

//  delete []color;
//  delete dI;
}