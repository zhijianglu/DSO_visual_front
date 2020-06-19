#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "PixelGradient.h"
#include "PixelSelector.h"
#include "/home/lab/cpp_lib/getfile.h"

using namespace std;
using namespace cv;

int main(){

    std::string source = "/media/lab/S_disk/Paper_src/data/smooth";
    std::string output_path = "/media/lab/S_disk/Paper_src/data/gradient";

    std::vector<std::string> files;
    if (getdir(source, files) >= 0) {
        printf("found %d image files in folder %s!\n", (int) files.size(), source.c_str());
    } else if (getFile(source, files) >= 0) {
        printf("found %d image files in file %s!\n", (int) files.size(), source.c_str());
    } else {
        printf("could not load file list! wrong path / file?\n");
    }

    int image_count = files.size();

    for (int index = 0; index < image_count; ++index)
    {
        Mat imageInput = imread(files[index]);

//TODO 计算梯度

        PixelGradient *pixelGradient_ = new PixelGradient;

        cv::Mat gradient_img;

//  cv::Mat gradent_1;

        pixelGradient_->computeGradents(imageInput, gradient_img,0);

        imshow("gradent_1", gradient_img);

        imwrite(output_path+"/"+to_string(index)+".jpg",gradient_img);
    }

    return 0;
}
