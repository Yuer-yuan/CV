#include "util/util.h"

cv::Mat blur(cv::Mat &img) {
    cv::Mat img_blur;
    CV_Assert(!img.empty());
    cv::GaussianBlur(img, img_blur, cv::Size(5, 5), 1.4);
    return img_blur;
}

cv::Mat normalize(cv::Mat &img) {
    cv::Mat img_normalized;
    CV_Assert(!img.empty());
    cv::normalize(img, img_normalized, 0, 255, cv::NORM_MINMAX);
    return img_normalized;
}

cv::Mat normalize_to_8U(cv::Mat &img) {
    cv::Mat img_normalized;
    CV_Assert(!img.empty());
    cv::normalize(img, img_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return img_normalized;
}