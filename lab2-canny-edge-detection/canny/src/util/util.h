#ifndef __UTIL_H__
#define __UTIL_H__

#include <opencv2/opencv.hpp>

cv::Mat normalize(cv::Mat &img);    // normalize to [0, 255]
cv::Mat normalize_to_8U(cv::Mat &img);  // normalize to [0, 255] and convert to CV_8UC1

std::string get_file_name(const std::string &path);

cv::Mat color_with_grad(cv::Mat &grad, cv::Mat &orien);

float wrap(float value, float min, float max, float step);

#endif