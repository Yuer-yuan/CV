#ifndef __CANNY_H__
#define __CANNY_H__

#include <opencv2/opencv.hpp>

void get_gradient(cv::Mat &img_gray, cv::Mat &img_grad_abs, cv::Mat &img_angle);
cv::Mat non_maximum_suppress(cv::Mat &img_grad_abs, cv::Mat &img_angle);
void double_threshold(cv::Mat &img_suppressed, float low_threshold, float high_threshold, cv::Mat &img_strong, std::vector<std::pair<int, int>> &weak_points_idx);
cv::Mat hysteresis(cv::Mat &img_suppressed, cv::Mat &img_strong, std::vector<std::pair<int, int>> &weak_points_idx);
cv::Mat canny(cv::Mat &img_gray, double low_threshold = 0.3 * 255, double high_threshold = 0.6 * 255, const char *save_dir = "./save");

#endif