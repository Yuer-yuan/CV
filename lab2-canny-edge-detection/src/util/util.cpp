#include "util/util.h"
#include <opencv2/highgui.hpp>

const char *blur_wnd_name = "blur";
static cv::Mat img_local, img_blur; 
static int ksize = 5, ksize_max = 30;
static int sigma = 1, sigma_max = 10;
static void on_trackbar_blur(int, void*);

cv::Mat blur(cv::Mat &img) {
    CV_Assert(!img.empty());
    img_local = img.clone();
    if (ksize % 2 == 0) {
        ksize += 1;
    }
    cv::createTrackbar("ksize", blur_wnd_name, &ksize, ksize_max, on_trackbar_blur);
    cv::createTrackbar("sigma", blur_wnd_name, &sigma, sigma_max, on_trackbar_blur);
    cv::GaussianBlur(img_local, img_blur, cv::Size(ksize, ksize), (double)sigma);
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

static void on_trackbar_blur(int, void*) {
    if (ksize % 2 == 0) {
        ksize += 1;
        cv::setTrackbarPos("ksize", blur_wnd_name, ksize);
    }
    cv::GaussianBlur(img_local, img_blur, cv::Size(ksize, ksize), (double)sigma);
    cv::imshow(blur_wnd_name, img_blur);
}