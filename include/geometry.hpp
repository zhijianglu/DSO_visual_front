#ifndef _GEOMETRY_HPP_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>


namespace hw
{

class GeometryUndistorter
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    GeometryUndistorter(std::string &path);

    void processFrame(cv::Mat &src, cv::Mat &dst);

private:
    int w_, h_, w_rect_, h_rect_;
    Eigen::Matrix3d K_, K_rect_;
    float k1_, k2_, k3_, k4_;  // for Equid
    float omega_;  // for FoV
    std::vector<Eigen::Vector2f> remap_;

    bool crop;

    void makeOptimalK_crop();
    void initRectifyMap();
    void distortCoordinates(std::vector<Eigen::Vector2f> &in);
    
};

} // end namespace

#endif // !_GEOMETRY_HPP_
