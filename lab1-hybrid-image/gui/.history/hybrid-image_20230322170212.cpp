#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

const static std::string squirrel = "../res/squirrel1.png";
const static std::string bunny = "../res/bunny1.png";
const static std::string dog = "../res/dog.jpg";
const static std::string cat = "../res/cat.jpg";
const static std::string donkey = "../res/donkey1.jpg";
const static std::string zebra = "../res/zebra1.jpg";


const static std::string wnd_name = "hybrid";

const cv::Mat img1 = cv::imread(squirrel);   // low frequency needed
const cv::Mat img2 = cv::imread(bunny);   // high frequency needed

// const cv::Mat img1 = cv::imread(dog);   // low frequency needed
// const cv::Mat img2 = cv::imread(cat);   // high frequency needed

// const cv::Mat img1 = cv::imread(donkey);   // low frequency needed
// const cv::Mat img2 = cv::imread(zebra);   // high frequency needed

const int ksize_max = 99, sigma_max = 99, weight_max = 10;
int ksize1 = 33, ksize2 = 33;
int sigma1 = 4.0, sigma2 = 4.0;
int weight1 = 10, weight2 = 10;

cv::Mat hybrid_img(const cv::Mat &img1, const cv::Mat &img2, int ksize1, double sigma1, int ksize2, double sigma2, double weight1, double weight2);
static void on_trackbar(int, void *);

int main()
{
    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat dst = hybrid_img(img1, img2, ksize1, sigma1, ksize1, sigma2, weight1, weight2);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::namedWindow(wnd_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("ksize1", wnd_name, &ksize1, ksize_max, on_trackbar);
    cv::createTrackbar("sigma1", wnd_name, &sigma1, sigma_max, on_trackbar);
    cv::createTrackbar("ksize2", wnd_name, &ksize2, ksize_max, on_trackbar);
    cv::createTrackbar("sigma2", wnd_name, &sigma2, sigma_max, on_trackbar);
    cv::createTrackbar("weight1", wnd_name, &weight1, 10, on_trackbar);
    cv::createTrackbar("weight2", wnd_name, &weight2, 10, on_trackbar);
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
    cv::imshow(wnd_name, dst);

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