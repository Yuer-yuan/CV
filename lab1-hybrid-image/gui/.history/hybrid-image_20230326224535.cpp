#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// #define CMP // used for compare color and gray hybrid images

#define PAIR 1  // three pairs: 0 for squirrel-bunny, 1 for dog-cat, 2 for donkey-zebra
#define GRAY1 1 // whether read image1 in gray scale, 0 for color, 1 for grayscale
#define GRAY2 1 // whether read image2 in gray scale, 0 for color, 1 for grayscale

const static std::string resource_dir = "../res/";
const static std::vector<std::pair<std::string, std::string>> pairs {
    {resource_dir + "squirrel1.png", resource_dir + "bunny1.png"},
    {resource_dir + "dog.jpg", resource_dir + "cat.jpg"},
    {resource_dir + "donkey1.jpg", resource_dir + "zebra1.jpg"},
};

cv::Mat img1 = cv::imread(pairs[PAIR].first, GRAY1);
cv::Mat img2 = cv::imread(pairs[PAIR].second, GRAY2);

const int ksize_max = 99, sigma_max = 99, weight_max = 10;
int ksize1 = 33, ksize2 = 19;
int sigma1 = 14, sigma2 = 6;
int weight1 = 10, weight2 = 10;

const int pyr_level = 5;        // number of pyramid levels
const double pyr_ratio = 0.5;   // ratio of pyramid levels

// hybrid image1 (low pass) and image2 (high pass)
cv::Mat hybrid_img(cv::Mat &img1, cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2, double weight1, double weight2);
// create trackbars needed
void create_trackbars();
// callback function for trackbars
static void on_trackbar(int, void *);
// show pyramid of an image
void show_pyramid(cv::Mat &img, int levels, double ratio, std::string wnd_name = "pyramid");

int main()
{
#ifndef CMP
#if !GRAY1
    cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);   // make sure mats are 3-channel
#endif
#if !GRAY2
    cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
#endif
    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize1, sigma2, weight1, weight2);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    create_trackbars();
    cv::waitKey(0);
#else   // compare color and gray hybrid images
    cv::Mat img1 = cv::imread(pairs[PAIR].first, 1);
    cv::Mat img2 = cv::imread(pairs[PAIR].second, 1);
    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize2, sigma2, weight1, weight2);
    show_pyramid(dst, pyr_level, pyr_ratio, "pyramid_color");

    img1 = cv::imread(pairs[PAIR].first, 0);
    img2 = cv::imread(pairs[PAIR].second, 0);
    cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
    dst = hybrid_img(img1, img2, ksize1, sigma1, ksize2, sigma2, weight1, weight2);
    show_pyramid(dst, pyr_level, pyr_ratio, "pyramid_gray");

    cv::waitKey(0);
#endif
    return 0;
}

cv::Mat hybrid_img(cv::Mat &img1, cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2, double weight1, double weight2) {
    cv::Mat dst;
    cv::Mat img1_g1, img2_g2;

    CV_Assert(img1.data && img2.data);
    CV_Assert(ksize1 > 0 && ksize1 % 2 == 1 && ksize2 > 0 && ksize2 % 2 == 1);
#ifndef CMP
    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
#endif
    double duration = static_cast<double>(cv::getTickCount());

    cv::GaussianBlur(img1, img1_g1, cv::Size(ksize1, ksize1), sigma1);  // low-passed image
    cv::GaussianBlur(img2, img2_g2, cv::Size(ksize2, ksize2), sigma2);
    cv::subtract(img2, img2_g2, img2_g2);   // high-passed image
    cv::addWeighted(img1_g1, weight1 / weight_max, img2_g2, weight2 / weight_max, 0.0, dst);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time of hybrid: " << duration << std::endl;
#ifndef CMP
    cv::imshow("low_passed", img1_g1);
    cv::imshow("high_passed", img2_g2);
    cv::imshow("hybrid_image", dst);
    show_pyramid(dst, pyr_level, pyr_ratio, "pyramid");
#endif
    return dst;
}

static void on_trackbar(int, void *) {
    if (ksize1 % 2 == 0) ksize1++;  // make sure ksize is odd
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

void show_pyramid(cv::Mat &img, int levels, double ratio, std::string wnd_name) {
    std::vector<cv::Mat> pyr;
    cv::Mat tmp = img;
    pyr.push_back(tmp);
    for (int i = 0; i < levels - 1; i++) {
        cv::resize(tmp, tmp, cv::Size(), ratio, ratio);
        pyr.push_back(tmp);
    }
    
    int height = pyr[0].rows;
    int width = pyr[0].cols;
    for (int i = 1; i < levels; i++) width += pyr[i].cols;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    int x = 0;
    for (int i = 0; i < levels; i++) {
        cv::Mat tmp = canvas(cv::Rect(x, 0, pyr[i].cols, pyr[i].rows));
        pyr[i].copyTo(tmp);
        x += pyr[i].cols;
    }

    cv::imshow(wnd_name, canvas);
}