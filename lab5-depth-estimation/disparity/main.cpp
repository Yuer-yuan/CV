//
// Created by guo on 23-5-29.
//

#include <opencv2/opencv.hpp>
#include <string>

#include "stereo_vision.h"

int main() {
    std::string dataset = "cones";  // "cones", "teddy", "tsukuba", "map"
    std::string data_dir = "/home/guo/CLionProjects/disparity/data/";
    std::string img1_path, img2_path;
    img1_path = data_dir + dataset + "/im0.png";
    img2_path = data_dir + dataset + "/im1.png";
    cv::Mat img1 = cv::imread(img1_path);
    cv::Mat img2 = cv::imread(img2_path);
    CV_Assert(img1.data && img2.data);
    CV_Assert(img1.cols == img2.cols && img1.rows == img1.rows);

    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    cv::Mat disparity, disparity_norm;
    get_disparity(img1_gray, img2_gray, disparity,  disparity_norm, DISP_LOCAL, DISP_NC);
//    get_disparity(img1_gray, img2_gray, disparity,  disparity_norm, DISP_SEMI);

    cv::Mat disparity_norm_heat;
    cv::applyColorMap(disparity_norm, disparity_norm_heat, cv::COLORMAP_JET);
    cv::imshow("disparity", disparity_norm);
    cv::imshow("disparity_norm_heat", disparity_norm_heat);
    cv::waitKey(0);
    return 0;
}