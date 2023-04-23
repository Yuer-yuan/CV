#include "canny/canny.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

cv::Mat canny(cv::Mat &img_gray, double low_threshold, double high_threshold, const char *save_dir) {
    cv::Mat img_canny;
    CV_Assert(!img_gray.empty());

    // get gradient and direction by applying sobel operator
    cv::Mat img_grad_abs, img_angle;
    get_gradient(img_gray, img_grad_abs, img_angle);

    // non-maximum suppression
    cv::Mat img_suppressed = non_maximum_suppress(img_grad_abs, img_angle);
    img_suppressed = normalize(img_suppressed);

    // double threshold
    cv::Mat img_strong;
    std::vector<std::pair<int, int>> weak_points_idx;
    double_threshold(img_suppressed, low_threshold, high_threshold, img_strong, weak_points_idx);

    // hysteresis
    img_canny = hysteresis(img_suppressed, img_strong, weak_points_idx);
    img_canny = normalize_to_8U(img_canny);
    cv::imwrite(std::string(save_dir) + "/canny.png", img_canny);
    return img_canny;
}

void get_gradient(cv::Mat &img_gray, cv::Mat &img_grad_abs, cv::Mat &img_angle) {
    img_grad_abs.create(img_gray.size(), CV_32FC1);
    img_angle.create(img_gray.size(), CV_32FC1);

    cv::Mat kernel_sobel_x, kernel_sobel_y;
    cv::Mat img_grad_x, img_grad_y;
    kernel_sobel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    kernel_sobel_y = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::filter2D(img_gray, img_grad_x, CV_32FC1, kernel_sobel_x);
    cv::filter2D(img_gray, img_grad_y, CV_32FC1, kernel_sobel_y);
    cv::magnitude(img_grad_x, img_grad_y, img_grad_abs);    // grad_abs = sqrt(src1^2 + src2^2)
    cv::phase(img_grad_x, img_grad_y, img_angle, true);     // angle = atan2(src1, src2) (in degree [0, 360))
}

cv::Mat non_maximum_suppress(cv::Mat &img_grad_abs, cv::Mat &img_angle) {
    cv::Mat img_suppressed(img_grad_abs.size(), CV_32FC1);

    int rows = img_grad_abs.rows, cols = img_grad_abs.cols;
    float angle, grad, grad1, grad2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                img_suppressed.at<float>(i, j) = 0;
                continue;
            }
            angle = img_angle.at<float>(i, j);
            grad = img_grad_abs.at<float>(i, j);
            if (angle >= 337.5 || angle < 22.5 || (angle >= 157.5 && angle < 202.5)) {
                grad1 = img_grad_abs.at<float>(i, j - 1);
                grad2 = img_grad_abs.at<float>(i, j + 1);
            } else if ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5)) {
                grad1 = img_grad_abs.at<float>(i - 1, j + 1);
                grad2 = img_grad_abs.at<float>(i + 1, j - 1);
            } else if ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5)) {
                grad1 = img_grad_abs.at<float>(i - 1, j);
                grad2 = img_grad_abs.at<float>(i + 1, j);
            } else if ((angle >= 112.5 && angle < 157.5) || (angle >= 292.5 && angle < 337.5)) {
                grad1 = img_grad_abs.at<float>(i - 1, j - 1);
                grad2 = img_grad_abs.at<float>(i + 1, j + 1);
            } else {
                std::cout << "angle: " << angle << std::endl;
                CV_Assert(false);
            }
            if (grad >= grad1 && grad >= grad2) {
                img_suppressed.at<float>(i, j) = grad;
            } else {
                img_suppressed.at<float>(i, j) = 0;
            }
        }
    }
    return img_suppressed;
}

void double_threshold(cv::Mat &img_suppressed, float low_threshold, float high_threshold, cv::Mat &img_strong, std::vector<std::pair<int, int>> &weak_points_idx) {
    img_strong.create(img_suppressed.size(), CV_32FC1);
    img_strong.setTo(0);

    int rows = img_suppressed.rows, cols = img_suppressed.cols;
    float grad;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad = img_suppressed.at<float>(i, j);
            if (grad >= high_threshold) {
                img_strong.at<float>(i, j) = 255;
            } else if (grad >= low_threshold) {
                weak_points_idx.push_back(std::make_pair(i, j));
            }
        }
    }
}

cv::Mat hysteresis(cv::Mat &img_suppressed, cv::Mat &img_strong, std::vector<std::pair<int, int>> &weak_points_idx) {
    cv::Mat img_canny = img_strong.clone();

    int rows = img_suppressed.rows, cols = img_suppressed.cols;
    for (auto &idx : weak_points_idx) {
        int i = idx.first, j = idx.second;
        if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
            continue;
        }
        if (img_strong.at<float>(i - 1, j - 1) > 0 || img_strong.at<float>(i - 1, j) > 0 || img_strong.at<float>(i - 1, j + 1) > 0 ||
            img_strong.at<float>(i, j - 1) > 0 || img_strong.at<float>(i, j + 1) > 0 ||
            img_strong.at<float>(i + 1, j - 1) > 0 || img_strong.at<float>(i + 1, j) > 0 || img_strong.at<float>(i + 1, j + 1) > 0) {
            img_canny.at<float>(i, j) = 255;
        }
    }
    return img_canny;
}
