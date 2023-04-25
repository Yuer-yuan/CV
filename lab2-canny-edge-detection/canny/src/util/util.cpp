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

cv::Mat color_with_grad(cv::Mat &grad, cv::Mat &orien) {
    // color in hsv mode. hue is orien, saturation and value are grad
    cv::Mat color(grad.size(), CV_8UC3);
    color.setTo(cv::Scalar(0, 0, 0));
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            if (grad.at<uint8_t>(i, j)) {
                color.at<cv::Vec3b>(i, j)[0] = (uint8_t)orien.at<float>(i, j) / 2;
                color.at<cv::Vec3b>(i, j)[1] = grad.at<uint8_t>(i, j);
                color.at<cv::Vec3b>(i, j)[2] = grad.at<uint8_t>(i, j);
            }
        }
    }
    cv::cvtColor(color, color, cv::COLOR_HSV2BGR);
    return color;
}

float wrap(float value, float min, float max, float step) {
    if (value < min) {
        while (value < min) value += step;
    } else if (value > max) {
        while (value > max) value -= step;
    }
    if (value < min || value > max) {
        std::cerr << "Error: value " << value << " is out of range [" << min << ", " << max << "]" << std::endl;
        return 0;
    }
    return value;
}