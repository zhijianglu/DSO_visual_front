#include "photometric.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <iterator>

hw::PhotometricUndistorter::PhotometricUndistorter(std::string &path) {
  std::string data_path = path;

  if (path.find_last_of("/") != path.size() - 1)
    data_path = path + "/";

  std::string response_file = data_path + "pcalib.txt";
  std::string vignette_file = data_path + "vignette.png";

  // read G
  std::ifstream rf(response_file.c_str());
  assert(rf.good());
  std::cout << "[INFO]: Reading Photometric Calibration from " << response_file << std::endl;

  std::string line;
  std::getline(rf, line);
  std::istringstream l1i(line);
  std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());

  G_depth_ = Gvec.size();
  if (G_depth_ < 256) {
    std::cout << "[ERROR]: invalid format! got num of entries less than 256 in response time file" << std::endl;
    return;
  }

  // normalized
  G_.resize(G_depth_);
  float min = Gvec[0];
  float max = Gvec[G_depth_ - 1];
  for (int i = 0; i < G_depth_; ++i) {
    G_[i] = 255.0f * (Gvec[i] - min) / (max - min);
  }


  // read V
  cv::Mat v_img = cv::imread(vignette_file, CV_LOAD_IMAGE_UNCHANGED);

  w_ = v_img.cols;
  h_ = v_img.rows;

  vignette_ = cv::Mat(h_, w_, CV_32FC1);
  vignette_inv_ = cv::Mat(h_, w_, CV_32FC1);
  cv::Mat v_8U = cv::Mat(h_, w_, CV_8U);

  double max_v;
  cv::minMaxLoc(v_img, 0, &max_v);
  // vignette_ = v_img / (float)max_v;
  v_img.convertTo(vignette_, CV_32F, 1.f / max_v);
  // cv::imshow("vignette", vignette_);

  vignette_inv_ = 1.0f / vignette_;
  // vignette_inv_.convertTo(v_8U, CV_8U);
  // cv::imshow("vignette", v_8U);
  // cv::waitKey(0);

}


void hw::PhotometricUndistorter::processFrame(cv::Mat& image) {
  int w_img = image.cols;
  int h_img = image.rows;
//   cv::imshow("image", image);
//   cv::waitKey(0);

  if (w_img == 0 || h_img == 0 || w_ != w_img || h_ != h_img) {
    std::cout << "[ERROR]: wrong image solution !!" << std::endl;
    return;
  }

  if (image.type() == CV_16S) {
    std::cout << "[Warning]: input is a 16bit image" << std::endl;
    return;
  }

  //TODO: use 'vignette_inv_' and 'G_' to finish photometric undistort on 'image'
  uchar* ptr_image = 0;
  for(int i = 0; i < h_img; ++i) {
    ptr_image = image.ptr<uchar>(i);
    for (int j = 0; j < w_img; ++j) {
      ptr_image[j] = (uchar)G_[ptr_image[j]];
      ptr_image[j] *= (uchar)vignette_inv_.at<float>(i, j);
    }
  }




}