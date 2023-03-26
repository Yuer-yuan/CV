#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define PAIR 2 // three pairs: 0 for squirrel-bunny, 1 for dog-cat, 2 for donkey-zebra
#define GRAY1 1 // whether read image1 in gray scale
#define GRAY2 0 // whether read image2 in gray scale

const static std::string resource_dir = "../res/";
const static std::vector<std::pair<std::string, std::string>> pairs {
    {resource_dir + "squirrel1.png", resource_dir + "bunny1.png"},
    {resource_dir + "dog.jpg", resource_dir + "cat.jpg"},
    {resource_dir + "donkey1.jpg", resource_dir + "zebra1.jpg"},
};

const cv::Mat img1 = cv::imread(pairs[PAIR].first, GRAY1);
const cv::Mat img2 = cv::imread(pairs[PAIR].second, GRAY2);

const int ksize_max = 99, sigma_max = 99, weight_max = 10;
int ksize1 = 33, ksize2 = 33;
int sigma1 = 4.0, sigma2 = 4.0;
int weight1 = 10, weight2 = 10;

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2, double weight1, double weight2);
void create_trackbars();
static void on_trackbar(int, void *);

int main()
{
    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize1, sigma2, weight1, weight2);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    create_trackbars();
    cv::waitKey(0);
    return 0;
}

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2, double weight1, double weight2) {
    cv::Mat dst;
    cv::Mat img1_g1, img2_g2;

    CV_Assert(img1.data && img2.data);
    CV_Assert(ksize1 > 0 && ksize1 % 2 == 1 && ksize2 > 0 && ksize2 % 2 == 1);

    double duration = static_cast<double>(cv::getTickCount());

    cv::GaussianBlur(img1, img1_g1, cv::Size(ksize1, ksize1), sigma1);
    cv::GaussianBlur(img2, img2_g2, cv::Size(ksize2, ksize2), sigma2);
    cv::subtract(img2, img2_g2, img2_g2);
    cv::addWeighted(img1_g1, weight1 / weight_max, img2_g2, weight2 / weight_max, 0.0, dst);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::imshow("low_passed", img1_g1);
    cv::imshow("high_passed", img2_g2);
    cv::imshow("hybrid_image", dst);

    return dst;
}

static void on_trackbar(int, void *) {
    if (ksize1 % 2 == 0) ksize1++;
    if (ksize2 % 2 == 0) ksize2++;
    if (sigma1 == 0) sigma1++;
    if (sigma2 == 0) sigma1++;
    if (weight1 == 0) weight1++;
    if (weight2 == 0) weight2++;

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize2, sigma2, weight1, weight2);
}

void create_trackbars() {
    cv::createTrackbar("ksize1", "low_passed", &ksize1, ksize_max, on_trackbar);
    cv::createTrackbar("sigma1", "low_passed", &sigma1, sigma_max, on_trackbar);
    cv::createTrackbar("ksize2", "high_passed", &ksize2, ksize_max, on_trackbar);
    cv::createTrackbar("sigma2", "high_passed", &sigma2, sigma_max, on_trackbar);
    cv::createTrackbar("weight1", "hybrid_image", &weight1, weight_max, on_trackbar);
    cv::createTrackbar("weight2", "hybrid_image", &weight2, weight_max, on_trackbar);
}