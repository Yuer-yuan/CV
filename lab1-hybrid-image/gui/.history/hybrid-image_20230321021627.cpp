#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

const int ksize_max = 100, sigma_max = 100;
int ksize1 = 0, ksize2 = 0, sigma1 = 0, sigma2 = 0;

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, int sigma1, int ksize2, int sigma2);
static void on_trackbar

int main()
{
    const cv::Mat img1 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/gui/res/squirrel1.png");
    const cv::Mat img2 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/gui/res/bunny1.png");

    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat hybrid = hybrid_img(img1, img2, 33, 4, 33, 4);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::namedWindow("hybrid", cv::WINDOW_AUTOSIZE);
    // cv::imshow("hybrid", hybrid);
    // cv::waitKey(0);
    cv::createTrackbar("ksize1", "hybrid", ksize1, ksize_max, )
    
    return 0;
}

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, int sigma1, int ksize2, int sigma2) {
    cv::Mat dst;
    cv::Mat img1_g1, img2_g2;

    cv::GaussianBlur(img1, img1_g1, cv::Size(ksize1, ksize1), sigma1);
    cv::GaussianBlur(img2, img2_g2, cv::Size(ksize2, ksize2), sigma2);
    cv::subtract(img2, img2_g2, img2_g2);
    cv::add(img1_g1, img2_g2, dst);

    return dst;
}