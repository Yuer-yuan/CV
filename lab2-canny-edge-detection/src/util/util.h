#ifndef __UTIL_H__
#define __UTIL_H__

#include <opencv2/opencv.hpp>

cv::Mat blur(cv::Mat &img);
cv::Mat normalize(cv::Mat &img);
cv::Mat normalize_to_8U(cv::Mat &img);

#endif