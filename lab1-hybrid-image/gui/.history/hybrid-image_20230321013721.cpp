#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat hybrid_img(cv::Mat &img1, cv::Mat &img2, int ksize1, int sigma1, int ksize2, int sigma2);

int main()
{
    const cv::Mat img1 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/gui/res/squirrel1.png");
    const cv::Mat img2 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/gui/res/bunny1.png");

    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat hybrid = hybrid_img(img1, img2, 33, 4, 33, 4);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::imshow("hybrid", hybrid);
    cv::waitKey(0);
    
    return 0;
}

cv::Mat hybrid_img(cv::Mat &img1, cv::Mat &img2, int ksize1, int sigma1, int ksize2, int sigma2) {

}