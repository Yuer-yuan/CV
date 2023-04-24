#ifndef __CANNY_H__
#define __CANNY_H__

#include <opencv2/opencv.hpp>

static void blur();
static void get_gradient();
static void non_maximum_suppress();
static void double_threshold();
static void hysteresis();
cv::Mat canny(cv::Mat &img_gray, double low_threshold = 30, double high_threshold = 60, bool linear_interpolation = true);

#endif