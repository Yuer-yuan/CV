#include "canny/canny.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

extern bool interactive;

enum wnd_name { BLUR, GRAD, DIR, SUPP, STRONG, WEAK, FILTERED, HYST, VOTE, LINK };
const char *wnd_name[] = { "blur", "gradient", "direction", "suppressed", "strong", "weak", "filtered", "hysteresis", "edge_vote", "edge_link" };

static cv::Mat img_gray_local;
static cv::Mat img_blur;
static cv::Mat img_grad_norm, img_grad_abs, img_angle, img_grad_x, img_grad_y;
static cv::Mat img_suppressed;
static cv::Mat img_strong, img_weak, img_filtered;
static cv::Mat img_hysteresis;

static cv::Mat img_edge_link_grad, img_edge_link_orien, img_edge_vote_grad, img_edge_vote_orien;
static std::map<std::pair<int, int>, std::vector<std::pair<float, float>>> vec_pixel_grad; // <(i, j), (grad1, orien1), (grad2, orien2)>

static int blur_ksize = 5, blur_ksize_max = 31;
static int blur_sigma = 1, blur_sigma_max = 10;

static int sobel_ksize = 3, sobel_ksize_max = 7;    // sobel kernel size must be 1, 3, 5, or 7
static int linear_interpolation = 0, linear_interpolation_max = 1;  // 0: no linear interpolation, 1: linear interpolation
static int low_threshold, high_threshold, low_threshold_max = 255, high_threshold_max = 255;

static int edge_linking = 0, edge_linking_max = 1;  // 0: no edge linking, 1: link edge
static int edge_linking_threshold, edge_linking_threshold_max = 255;

static void create_trackbars();
static void on_trackbar_canny(int, void *);

cv::Mat canny(cv::Mat &img_gray, double low_thresh, double high_thresh, bool linear_inter, bool edge_linking_param) {
    CV_Assert(!img_gray.empty());

    img_gray_local = img_gray.clone();
    low_threshold = low_thresh, high_threshold = high_thresh;
    linear_interpolation = linear_inter;
    if (low_threshold > high_threshold) std::swap(low_threshold, high_threshold);
    edge_linking = edge_linking_param;
    edge_linking_threshold = (low_threshold + high_threshold) >> 1;

    // eliminate noise by applying gaussian blur
    blur();

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

    // edge link
    img_edge_link_grad = cv::Mat::zeros(img_hysteresis.size(), CV_8UC1);
    img_edge_link_orien = cv::Mat::zeros(img_hysteresis.size(), CV_32FC1);
    img_edge_vote_grad = cv::Mat::zeros(img_hysteresis.size(), CV_8UC1);
    img_edge_vote_orien = cv::Mat::zeros(img_hysteresis.size(), CV_32FC1);
    if (edge_linking) edge_link();

b:  // set for quick break point when debugging with gdb. a usage may like `b canny:b`
    if (interactive) create_trackbars();

    if (edge_linking) return img_edge_link_grad;
    return img_hysteresis;
}

static void blur() {
    if (blur_ksize % 2 == 0) blur_ksize += 1;
    cv::GaussianBlur(img_gray_local, img_blur, cv::Size(blur_ksize, blur_ksize), (double)blur_sigma);
    if (interactive) cv::imshow(wnd_name[BLUR], img_blur);
}

static void get_gradient() {
    if (sobel_ksize % 2 == 0) sobel_ksize++;
    cv::Sobel(img_blur, img_grad_x, CV_32FC1, 1, 0, sobel_ksize);
    cv::Sobel(img_blur, img_grad_y, CV_32FC1, 0, 1, sobel_ksize);
    cv::magnitude(img_grad_x, img_grad_y, img_grad_abs);    // grad_abs = sqrt(src1^2 + src2^2)
    cv::phase(img_grad_x, img_grad_y, img_angle, true);     // angle = atan2(src1, src2) (in degree 0~360)
    img_grad_norm = normalize_to_8U(img_grad_abs);
    if (interactive) {
        cv::imshow(wnd_name[GRAD], img_grad_norm);
        cv::imshow(wnd_name[DIR], normalize_to_8U(img_angle));
        cv::imshow("color_with_grad", color_with_grad(img_grad_norm, img_angle));
    }
}

static void non_maximum_suppress() {
    img_suppressed.setTo(0);
    int rows = img_grad_norm.rows, cols = img_grad_norm.cols;
    if (!linear_interpolation) {
        float angle;
        uint8_t grad, grad1, grad2;
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
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
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
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
    if (interactive) cv::imshow(wnd_name[SUPP], img_suppressed);
}

static void double_threshold() {
    img_strong.setTo(0), img_weak.setTo(0), img_filtered.setTo(0);
    int rows = img_suppressed.rows, cols = img_suppressed.cols;
    uint8_t grad, fill;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad = img_suppressed.at<uint8_t>(i, j);
            fill = 255;
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
    if (interactive) {
        cv::imshow(wnd_name[STRONG], img_strong);
        cv::imshow(wnd_name[WEAK], img_weak);
        cv::imshow(wnd_name[FILTERED], img_filtered);
    }
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
            if (weak_pt == 0) continue;
            if (img_strong.at<uint8_t>(i - 1, j - 1) | img_strong.at<uint8_t>(i - 1, j) | img_strong.at<uint8_t>(i - 1, j + 1) |
                img_strong.at<uint8_t>(i, j - 1) | img_strong.at<uint8_t>(i, j + 1) |
                img_strong.at<uint8_t>(i + 1, j - 1) | img_strong.at<uint8_t>(i + 1, j) | img_strong.at<uint8_t>(i + 1, j + 1)) {
                    img_hysteresis.at<uint8_t>(i, j) = weak_pt;
            }
        }
    }
    if (interactive) {
        cv::imshow(wnd_name[HYST], img_hysteresis);
        cv::imshow("hysteresis_color_with_orientation", color_with_grad(img_hysteresis, img_angle));
    }
}

static void create_trackbars() {
    cv::namedWindow(wnd_name[BLUR], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[GRAD], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[SUPP], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[STRONG], cv::WINDOW_AUTOSIZE);
    cv::namedWindow(wnd_name[WEAK], cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("blur_ksize", wnd_name[BLUR], &blur_ksize, blur_ksize_max, on_trackbar_canny);
    cv::createTrackbar("blur_sigma", wnd_name[BLUR], &blur_sigma, blur_sigma_max, on_trackbar_canny);
    cv::createTrackbar("sobel_size", wnd_name[GRAD], &sobel_ksize, sobel_ksize_max, on_trackbar_canny);
    cv::createTrackbar("linear_interpolation", wnd_name[SUPP], &linear_interpolation, linear_interpolation_max, on_trackbar_canny);
    cv::createTrackbar("low_threshold", wnd_name[WEAK], &low_threshold, low_threshold_max, on_trackbar_canny);
    cv::createTrackbar("high_threshold", wnd_name[STRONG], &high_threshold, high_threshold_max, on_trackbar_canny);
    if (edge_linking) cv::createTrackbar("edge_linking", wnd_name[LINK], &edge_linking_threshold, edge_linking_threshold_max, on_trackbar_canny);
}

static void on_trackbar_canny(int, void *) {
    if (blur_ksize % 2 == 0) {
        blur_ksize++;
        cv::setTrackbarPos("blur_ksize", wnd_name[BLUR], blur_ksize);
    }
    if (sobel_ksize % 2 == 0) {
        sobel_ksize++;
        cv::setTrackbarPos("sobel_size", wnd_name[GRAD], sobel_ksize);
    }
    if (high_threshold < low_threshold) {
        std::swap(high_threshold, low_threshold);
        cv::setTrackbarPos("low_threshold", wnd_name[WEAK], low_threshold);
        cv::setTrackbarPos("high_threshold", wnd_name[STRONG], high_threshold);
    }
    blur();
    get_gradient();
    non_maximum_suppress();
    double_threshold();
    hysteresis();
    if (edge_linking) edge_link();
}

static void edge_link() {
    img_edge_link_grad.setTo(0);
    img_edge_link_orien.setTo(cv::Scalar(0, 0, 0));
    img_edge_vote_grad.setTo(0);
    img_edge_vote_orien.setTo(cv::Scalar(0, 0, 0));
    // vote by strong pixels
    int rows = img_hysteresis.rows, cols = img_hysteresis.cols;
    int pixel_grad;
    float pixel_grad_x, pixel_grad_y;
    float pixel_orien, pixel_orien_tan;
    int pixel_1_x, pixel_1_y, pixel_2_x, pixel_2_y;
    for (int i = 0; i < rows; i++) {    
        for (int j = 0; j < cols; j++) {
            if (!img_hysteresis.at<uint8_t>(i, j)) continue;
            pixel_grad = img_grad_norm.at<uint8_t>(i, j);
            pixel_orien = img_angle.at<float>(i, j);
            pixel_orien_tan = wrap(pixel_orien + 90, 0, 180, 180);
            // pixel_orien_tan = (int)(pixel_orien + 90) % 180;
            pixel_grad_x = pixel_grad * cos(pixel_orien * M_PI / 180);
            pixel_grad_y = pixel_grad * sin(pixel_orien * M_PI / 180);
            if (pixel_orien_tan < 22.5 || pixel_orien_tan >= 157.5) {
                pixel_1_x = pixel_2_x = i;
                pixel_1_y = j - 1, pixel_2_y = j + 1;
            } else if (pixel_orien_tan < 67.5) {
                pixel_1_x = i - 1, pixel_2_x = i + 1;
                pixel_1_y = j - 1, pixel_2_y = j + 1;
            } else if (pixel_orien_tan < 112.5) {
                pixel_1_x = i - 1, pixel_2_x = i + 1;
                pixel_1_y = pixel_2_y = j;
            } else {
                pixel_1_x = i - 1, pixel_2_x = i + 1;
                pixel_1_y = j + 1, pixel_2_y = j - 1;
            }
            std::vector<std::pair<int, int>> pixels;
            if (!img_strong.at<uint8_t>(pixel_1_x, pixel_1_y)) pixels.push_back(std::make_pair(pixel_1_x, pixel_1_y));
            if (!img_strong.at<uint8_t>(pixel_2_x, pixel_2_y)) pixels.push_back(std::make_pair(pixel_2_x, pixel_2_y));
            for (auto pixel : pixels) {
                if (vec_pixel_grad.find(pixel) == vec_pixel_grad.end()) {
                    vec_pixel_grad[pixel] = std::vector<std::pair<float, float>>();
                    vec_pixel_grad[pixel].push_back(std::make_pair(pixel_grad_x, pixel_grad_y));
                } else {
                    vec_pixel_grad[pixel].push_back(std::make_pair(pixel_grad_x, pixel_grad_y));
                }
            }
        }
    }
    // vote by weak pixels
    for (auto pixel_grad : vec_pixel_grad) {
        int pixel_x = pixel_grad.first.first, pixel_y = pixel_grad.first.second;
        if (img_weak.at<uint8_t>(pixel_x, pixel_y) == 0) continue;
        pixel_grad.second.push_back(std::make_pair((float)img_grad_norm.at<uint8_t>(pixel_x, pixel_y), (float)img_grad_norm.at<uint8_t>(pixel_x, pixel_y)));
    }
    // calculate vote
    for (auto pixel_grad : vec_pixel_grad) {
        float pixel_grad_x = 0, pixel_grad_y = 0;
        int pixel_x = pixel_grad.first.first, pixel_y = pixel_grad.first.second;
        for (auto grad : pixel_grad.second) {
            pixel_grad_x += grad.first;
            pixel_grad_y += grad.second;
        }
        pixel_grad_x /= pixel_grad.second.size();
        pixel_grad_y /= pixel_grad.second.size();
        float pixel_grad_abs = std::sqrt(pixel_grad_x * pixel_grad_x + pixel_grad_y * pixel_grad_y);
        if (pixel_grad_abs > 255) pixel_grad_abs = 255;
        if (pixel_grad_abs > edge_linking_threshold) {
            img_edge_vote_grad.at<uint8_t>(pixel_x, pixel_y) = pixel_grad_abs;
            img_edge_vote_orien.at<float>(pixel_x, pixel_y) = wrap((std::atan2(pixel_grad_y, pixel_grad_x) * 180 / M_PI), 0, 360, 360);
        }
    }
    // merge hysterisis and vote
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (img_hysteresis.at<uint8_t>(i, j)) {
                img_edge_link_grad.at<uint8_t>(i, j) = img_hysteresis.at<uint8_t>(i, j);
                img_edge_link_orien.at<float>(i, j) = img_angle.at<float>(i, j);
            } else if (img_edge_vote_grad.at<uint8_t>(i, j)) {
                img_edge_link_grad.at<uint8_t>(i, j) = img_edge_vote_grad.at<uint8_t>(i, j) = 255;
                img_edge_link_orien.at<float>(i, j) = img_edge_vote_orien.at<float>(i, j);
            }
        }
    }
    if (interactive) {
        cv::imshow("edge_vote", img_edge_vote_grad);
        cv::imshow("edge_vote_grad_with_orien", color_with_grad(img_edge_vote_grad, img_edge_vote_orien));
        cv::imshow("edge_link", img_edge_link_grad);
        cv::imshow("edge_link_grad_with_orien", color_with_grad(img_edge_link_grad, img_edge_link_orien));
    }
}