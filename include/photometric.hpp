
#ifndef PHOTOMETRIC_HPP_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace hw
{
    
class PhotometricUndistorter
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string &path);

    void processFrame(cv::Mat& image);

private:
    std::vector<float> G_;
    int G_depth_;
    cv::Mat vignette_;
    cv::Mat vignette_inv_;
    int w_, h_;
};


} // end namespace


#endif // !PHOTOMETRIC_HPP_



