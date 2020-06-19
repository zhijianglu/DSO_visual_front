//
// Created by lab on 20-3-25.
//

#ifndef DSO_MODULES_BASE_H
#define DSO_MODULES_BASE_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

#define PYR_LEVELS 6
#define patternNum 8
#define patternP staticPattern[8] // 取第8种pattern
#define patternPadding 2
extern int staticPattern[10][40][2];
extern float setting_outlierTH;

typedef Eigen::Matrix<float,2,1> Vec2f_;

extern float setting_minGradHistCut;
extern float setting_minGradHistAdd;
extern float setting_gradDownweightPerLevel;
extern bool  setting_selectDirectionDistribution;
extern float minUseGrad_pixsel;
extern int sparsityFactor;

//void toCvMat(unsigned char *data, cv::Mat &image);
void toCvMat(float *data, cv::Mat &image);

void drawMat(cv::Mat &image, cv::Point2i UL, Scalar color = Scalar(255, 255, 0), int lineWidth = 1);
#endif //DSO_MODULES_BASE_H
