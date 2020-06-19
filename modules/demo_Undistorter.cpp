#include "geometry.hpp"
#include "photometric.hpp"
 
#include <string>
#include <iostream>
#include <vector>

int main(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "[ERROR]: Wrong num of parameters" << std::endl;
    argv[1] = "../data";
  }

  std::string path = argv[1];
  std::string data_path;

  if (path.find_last_of("/") != path.size() - 1)
    data_path = path + "/";

  cv::Mat img = cv::imread(data_path + "1.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_rect;

  hw::GeometryUndistorter g_u(path);
  hw::PhotometricUndistorter p_u(path);

  p_u.processFrame(img);
  cv::imshow("photometric", img);
  cv::waitKey(33);

  g_u.processFrame(img, img_rect);
  cv::imshow("geometric", img_rect);
  cv::waitKey(33);

  cv::imwrite(data_path + "rect.jpg", img_rect);
  cvWaitKey(0);
  return 0;
}
