#include "canny/canny.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

enum wnd_name { GRAD, DIR, SUPP, STRONG, WEAK, FILTERED, HYST };
const char *wnd_name[] = { "gradient", "direction", "suppressed", "strong", "weak", "filtered", "hysteresis" };
static cv::Mat img_gray_local;
static cv::Mat img_grad_norm, img_grad_abs, img_angle, img_grad_x, img_grad_y;
static cv::Mat img_suppressed;
static cv::Mat img_strong, img_weak, img_filtered;
static cv::Mat img_hysteresis;

static int sobel_ksize = 3, sobel_ksize_max = 7;    // sobel kernel size must be 1, 3, 5, or 7
static int linear_interpolation = 0, linear_interpolation_max = 1;  // 0: no linear interpolation, 1: linear interpolation
static int low_threshold, high_threshold, low_threshold_max = 255, high_threshold_max = 255;

static void create_trackbars();
static void on_trackbar_canny(int, void *);

cv::Mat canny(cv::Mat &img_gray, double low_thresh, double high_thresh, const char *save_dir) {
    CV_Assert(!img_gray.empty());

    img_gray_local = img_gray.clone();
    low_threshold = low_thresh, high_threshold = high_thresh;
    if (low_threshold > high_threshold) std::swap(low_threshold, high_threshold);

    // get gradient and direction by applying sobel operator
    get_gradient();

    // non-maximum suppression
    img_suppressed = cv::Mat::zeros(img_grad_abs.size(), CV_8UC1);
    non_maximum_suppress();

    // double threshold
    img_strong = cv::Mat::zeros(img_suppressed.size(), CV_8UC1);
    img_weak = cv::Mat::zeros(img_suppressed.size(), CV_8UC1);
    img_filtered = cv::Mat::zeros(img_suppressed.size(), CV_8UC1);
    double_threshold();

    // hysteresis
    hysteresis();

    create_trackbars();

    return img_hysteresis;
}

static void get_gradient() {
    if (sobel_ksize % 2 == 0) sobel_ksize++;
    cv::Sobel(img_gray_local, img_grad_x, CV_32FC1, 1, 0, sobel_ksize);
    cv::Sobel(img_gray_local, img_grad_y, CV_32FC1, 0, 1, sobel_ksize);
    cv::magnitude(img_grad_x, img_grad_y, img_grad_abs);    // grad_abs = sqrt(src1^2 + src2^2)
    cv::phase(img_grad_x, img_grad_y, img_angle, true);     // angle = atan2(src1, src2) (in degree 0~360)
    img_grad_norm = normalize_to_8U(img_grad_abs);
    cv::imshow(wnd_name[GRAD], img_grad_norm);
    cv::imshow(wnd_name[DIR], normalize_to_8U(img_angle));
}

static void non_maximum_suppress() {
    img_suppressed.setTo(0);
    int rows = img_grad_norm.rows, cols = img_grad_norm.cols;
    if (!linear_interpolation) {
        float angle;
        uint8_t grad, grad1, grad2;
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                angle = img_angle.at<float>(i, j);
                grad = img_grad_norm.at<uint8_t>(i, j);
                if (angle >= 337.5 || angle < 22.5 || (angle >= 157.5 && angle < 202.5)) {
                    grad1 = img_grad_norm.at<uint8_t>(i, j - 1);
                    grad2 = img_grad_norm.at<uint8_t>(i, j + 1);
                } else if ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5)) {
                    grad1 = img_grad_norm.at<uint8_t>(i - 1, j + 1);
                    grad2 = img_grad_norm.at<uint8_t>(i + 1, j - 1);
                } else if ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5)) {
                    grad1 = img_grad_norm.at<uint8_t>(i - 1, j);
                    grad2 = img_grad_norm.at<uint8_t>(i + 1, j);
                } else if ((angle >= 112.5 && angle < 157.5) || (angle >= 292.5 && angle < 337.5)) {
                    grad1 = img_grad_norm.at<uint8_t>(i - 1, j - 1);
                    grad2 = img_grad_norm.at<uint8_t>(i + 1, j + 1);
                } else {
                    std::cout << "angle: " << angle << std::endl;
                    CV_Assert(false);
                }
                if (grad >= grad1 && grad >= grad2) {
                    img_suppressed.at<uint8_t>(i, j) = grad;
                } else {
                    img_suppressed.at<uint8_t>(i, j) = 0;
                }
            }
        }
    } else {    // ref: https://blog.csdn.net/kezunhai/article/details/11620357
        float grad_x, grad_y, weight;
        float grad, grad_tmp1, grad_tmp2;
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                grad_x = img_grad_x.at<float>(i, j);
                grad_y = img_grad_y.at<float>(i, j);
                grad = img_grad_norm.at<uint8_t>(i, j);
                if (grad_x == 0 && grad_y == 0) {
                    img_suppressed.at<uint8_t>(i, j) = 0;
                    continue;    
                }
                if (grad_x == 0) {
                    grad_tmp1 = img_grad_norm.at<uint8_t>(i - 1, j);
                    grad_tmp2 = img_grad_norm.at<uint8_t>(i + 1, j);
                } else if (grad_y == 0) {
                    grad_tmp1 = img_grad_norm.at<uint8_t>(i, j - 1);
                    grad_tmp2 = img_grad_norm.at<uint8_t>(i, j + 1);
                } else {
                    weight = grad_y / grad_x;
                    if (weight >= 1) {
                        weight = 1 / weight;
                        grad_tmp1 = weight * img_grad_norm.at<uint8_t>(i - 1, j - 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i - 1, j);
                        grad_tmp2 = weight * img_grad_norm.at<uint8_t>(i + 1, j + 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i + 1, j);
                    } else if (weight < -1) {
                        weight = -1 / weight;
                        grad_tmp1 = weight * img_grad_norm.at<uint8_t>(i - 1, j + 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i - 1, j);
                        grad_tmp2 = weight * img_grad_norm.at<uint8_t>(i + 1, j - 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i + 1, j);
                    } else if (weight >= 0) {
                        grad_tmp1 = weight * img_grad_norm.at<uint8_t>(i + 1, j - 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i, j - 1);
                        grad_tmp2 = weight * img_grad_norm.at<uint8_t>(i - 1, j + 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i, j + 1);
                    } else if (weight < 0) {
                        weight = -weight;
                        grad_tmp1 = weight * img_grad_norm.at<uint8_t>(i - 1, j - 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i, j - 1);
                        grad_tmp2 = weight * img_grad_norm.at<uint8_t>(i + 1, j + 1) + (1 - weight) * img_grad_norm.at<uint8_t>(i, j + 1);
                    } else {
                        CV_Assert(0);
                    }
                }
                if (grad >= grad_tmp1 && grad >= grad_tmp2) {
                    img_suppressed.at<uint8_t>(i, j) = grad;
                } else {
                    img_suppressed.at<uint8_t>(i, j) = 0;
                }
            }
                
        }
    }
    cv::imshow(wnd_name[SUPP], img_suppressed);
}

static void double_threshold() {
    img_strong.setTo(0), img_weak.setTo(0), img_filtered.setTo(0);
    int rows = img_suppressed.rows, cols = img_suppressed.cols;
    uint8_t grad, fill;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad = img_suppressed.at<uint8_t>(i, j);
            fill = grad;
            if (grad >= high_threshold) {
                img_strong.at<uint8_t>(i, j) = fill;
                img_weak.at<uint8_t>(i, j) = 0;
                img_filtered.at<uint8_t>(i, j) = 0;
            } else if (grad >= low_threshold) {
                img_strong.at<uint8_t>(i, j) = 0;
                img_weak.at<uint8_t>(i, j) = fill;
                img_filtered.at<uint8_t>(i, j) = 0;
            } else {
                img_strong.at<uint8_t>(i, j) = 0;
                img_weak.at<uint8_t>(i, j) = 0;
                img_filtered.at<uint8_t>(i, j) = grad > 0 ? fill : 0;
            }
        }
    }
    cv::imshow(wnd_name[STRONG], img_strong);
    cv::imshow(wnd_name[WEAK], img_weak);
    cv::imshow(wnd_name[FILTERED], img_filtered);
}

static void hysteresis() {
    img_hysteresis = img_strong.clone();
    int rows = img_suppressed.rows, cols = img_suppressed.cols;
    uint8_t weak_pt;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((i == 0) || (i == rows - 1) || (j == 0) || (j == cols - 1)) {
                img_hysteresis.at<uint8_t>(i, j) = 0;
                continue;
            }
            weak_pt = img_weak.at<uint8_t>(i, j);
            if (weak_pt > 0) {
                if (img_strong.at<uint8_t>(i - 1, j - 1) > 0 || img_strong.at<uint8_t>(i - 1, j) > 0 || img_strong.at<uint8_t>(i - 1, j + 1) > 0 ||
                    img_strong.at<uint8_t>(i, j - 1) > 0 || img_strong.at<uint8_t>(i, j + 1) > 0 ||
                    img_strong.at<uint8_t>(i + 1, j - 1) > 0 || img_strong.at<uint8_t>(i + 1, j) > 0 || img_strong.at<uint8_t>(i + 1, j + 1) > 0) {
                        img_hysteresis.at<uint8_t>(i, j) = weak_pt;
                }
            }
        }
    }
    cv::imshow(wnd_name[HYST], img_hysteresis);
}

static void create_trackbars() {
    cv::namedWindow(wnd_name[GRAD], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[SUPP], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[STRONG], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[WEAK], cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("sobel_size", wnd_name[GRAD], &sobel_ksize, sobel_ksize_max, on_trackbar_canny);
    cv::createTrackbar("linear_interpolation", wnd_name[SUPP], &linear_interpolation, linear_interpolation_max, on_trackbar_canny);
    cv::createTrackbar("low_threshold", wnd_name[WEAK], &low_threshold, low_threshold_max, on_trackbar_canny);
    cv::createTrackbar("high_threshold", wnd_name[STRONG], &high_threshold, high_threshold_max, on_trackbar_canny);
}

static void on_trackbar_canny(int, void *) {
    if (sobel_ksize % 2 == 0) {
        sobel_ksize++;
        cv::setTrackbarPos("sobel_size", wnd_name[GRAD], sobel_ksize);
    }
    if (high_threshold < low_threshold) {
        std::swap(high_threshold, low_threshold);
        cv::setTrackbarPos("low_threshold", wnd_name[WEAK], low_threshold);
        cv::setTrackbarPos("high_threshold", wnd_name[STRONG], high_threshold);
    }
    get_gradient();
    non_maximum_suppress();
    double_threshold();
    hysteresis();
}