#include "util/util.h"

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

std::string get_file_name(const std::string &path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return path;
    size_t pos2 = path.find_last_of('.');
    if (pos2 == std::string::npos) return path.substr(pos + 1);
    return path.substr(pos + 1, pos2 - pos - 1);
}