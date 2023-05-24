//
// Created by guo on 23-5-24.
//

#ifndef MYSIFT_UTIL_H
#define MYSIFT_UTIL_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

static void show_image(const cv::Mat& img, const std::string& title = "image", bool norm = false) {
    if (norm) {
        cv::Mat norm_img;
        cv::normalize(img, norm_img, 0, 255, cv::NORM_MINMAX);
        if (img.type() == CV_32FC1) {
            norm_img.convertTo(norm_img, CV_8UC1);
        } else if (img.type() == CV_32FC3) {
            norm_img.convertTo(norm_img, CV_8UC3);
        } else if (img.type() == CV_64FC1) {
            norm_img.convertTo(norm_img, CV_8UC1);
        } else if (img.type() == CV_64FC3) {
            norm_img.convertTo(norm_img, CV_8UC3);
        } else {
            std::cout << "Unknown image type!" << std::endl;
            return;
        }
        cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
        cv::imshow(title, norm_img);
        cv::waitKey(0);
        return;
    }
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::imshow(title, img);
    cv::waitKey(0);
}

#endif //MYSIFT_UTIL_H
