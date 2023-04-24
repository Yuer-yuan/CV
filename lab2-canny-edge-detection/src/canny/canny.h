#ifndef __CANNY_H__
#define __CANNY_H__

#include <opencv2/opencv.hpp>

static void get_gradient();
static void non_maximum_suppress();
static void double_threshold();
static void hysteresis();
cv::Mat canny(cv::Mat &img_gray, double low_threshold = 0.3 * 255, double high_threshold = 0.6 * 255, const char *save_dir = "./save");

#endif