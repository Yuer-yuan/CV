#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

const static std::string squirrel = "/home/bill/mypro/CV/lab1-hybrid-image/gui/res/squirrel1.png";
const static std::string bunny = "/home/bill/mypro/CV/lab1-hybrid-image/gui/res/bunny1.png";

const cv::Mat img1 = cv::imread(squirrel);
const cv::Mat img2 = cv::imread(bunny);

const int ksize_max = 99, sigma_max = 99;
int ksize1 = 33, ksize2 = 33;
int sigma1 = 4.0, sigma2 = 4.0;

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2);
static void on_trackbar(int, void *);

int main()
{
    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize1, sigma2);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::namedWindow("settings", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("ksize1", "settings", &ksize1, ksize_max, on_trackbar);
    cv::createTrackbar("sigma1", "settings", &sigma1, sigma_max, on_trackbar);
    cv::createTrackbar("ksize2", "settings", &ksize2, ksize_max, on_trackbar);
    cv::createTrackbar("sigma2", "settings", &sigma2, sigma_max, on_trackbar);
    cv::waitKey(0);
    
    return 0;
}

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2) {
    cv::Mat dst;
    cv::Mat img1_g1, img2_g2;

    CV_Assert(ksize1 > 0 && ksize1 % 2 == 1 && ksize2 > 0 && ksize2 % 2 == 1);

    cv::GaussianBlur(img1, img1_g1, cv::Size(ksize1, ksize1), sigma1);
    cv::GaussianBlur(img2, img2_g2, cv::Size(ksize2, ksize2), sigma2);
    cv::subtract(img2, img2_g2, img2_g2);
    cv::add(img1_g1, img2_g2, dst);

    return dst;
}

static void on_trackbar(int, void *) {
    if (ksize1 % 2 == 0) ksize1++;
    if (ksize2 % 2 == 0) ksize2++;
    if (sigma1 == 0) sigma1++;
    if (sigma1 == 0) sigma1++;

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize2, sigma2);
    cv::imshow("dst", dst);
}