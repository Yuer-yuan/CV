#include "sift.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "util.h"

#define USE_MYSIFT

const std::string WORK_DIR = "/home/guo/mypro/CV/lab4-feature-extraction/MySift/";
int ratio_units = 80, max_ratio_units = 100;

cv::Mat img1, img2, img1_gray, img2_gray, img_match;
std::vector<cv::KeyPoint> kps1, kps2;
cv::Mat desc1, desc2;
std::vector<std::vector<cv::DMatch>> matches;
std::vector<cv::DMatch> good_matches;

#ifdef USE_MYSIFT
MySift* sift = new MySift();
#else // !USE_MYSIFT
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
#endif  // USE_MYSIFT

#ifdef DISPLAY

static void show_kpts(const cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {
    cv::Mat img_kpts = img.clone();
    float fct = 2.0;
    for (auto& kpt : kpts) {
        cv::Point2f pt = kpt.pt;
        float angle = kpt.angle;
        float response = kpt.response;
        cv::Point2f pt2 = cv::Point2f(pt.x + fct * response * cos(angle / 180 * CV_PI),
                                      pt.y + fct * response * sin(angle / 180 * CV_PI));
        cv::arrowedLine(img_kpts, pt, pt2, cv::Scalar(0, 255, 255), 1);
    }
    show_image(img_kpts, "kpts");
}

#endif  // DISPLAY

void ratio_test_call_back(int, void *);
void test_detect_and_compute();
void test_match();
void test_with_opencv_sift();
void test_asift();

int main() {
//    test_detect_and_compute();
//    test_match();
//    test_with_opencv_sift();
    test_asift();
    return 0;
}

void ratio_test_call_back(int, void *) {
    if (!ratio_units) {
        ratio_units = 1;
        cv::setTrackbarPos("ratio", "ratio_test", ratio_units);
    }
    matches.clear();
    good_matches.clear();
#ifdef USE_MYSIFT
    sift->detect_and_compute(img1_gray, kps1, desc1);
    sift->detect_and_compute(img2_gray, kps2, desc2);
#else // !USE_MYSIFT
    sift->detectAndCompute(img1_gray, cv::Mat(), kps1, desc1);
    sift->detectAndCompute(img2_gray, cv::Mat(), kps2, desc2);
#endif
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher();
    matcher->knnMatch(desc1, desc2, matches, 2);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ((float)ratio_units / (float)max_ratio_units) * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_match);
    cv::imshow("ratio_test", img_match);
    std::cout << "num of kpts: " << kps1.size() << ' ' << kps2.size() << ", num of matches: " << matches.size() << ", good matches: " << good_matches.size() << std::endl;
}

void test_detect_and_compute() {
    cv::Mat img, img_gray, img_kpts;
    MySift sift;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

    img = cv::imread(WORK_DIR + "assets/Lenna.png", cv::ImreadModes::IMREAD_UNCHANGED);
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    sift.detect_and_compute(img_gray, kpts, desc);
    cv::drawKeypoints(img, kpts, img_kpts, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("img_kpts", img_kpts);
    cv::waitKey(0);

#ifdef DISPLAY
//    show_kpts(img, kpts);
#endif  // DISPLAY
}

void test_match() {
//    img1 = cv::imread(WORK_DIR + "assets/Lenna.png", cv::ImreadModes::IMREAD_UNCHANGED);
//    img2 = cv::imread(WORK_DIR + "assets/Lenna.png", cv::ImreadModes::IMREAD_UNCHANGED);
    img1 = cv::imread(WORK_DIR + "assets/ucsb1.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    img2 = cv::imread(WORK_DIR + "assets/ucsb2.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    cv::namedWindow("ratio_test", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("ratio", "ratio_test", &ratio_units, max_ratio_units, ratio_test_call_back);
    cv::waitKey(0);
}

void test_with_opencv_sift() {
    cv::Mat img1, img2, img1_gray, img2_gray, img_match_m, img_match_o;
    std::vector<cv::KeyPoint> kps1_m, kps2_m, kps1_o, kps2_o;
    cv::Mat desc1_m, desc2_m, desc1_o, desc2_o;
    std::vector<std::vector<cv::DMatch>> matches_m, matches_o;
    std::vector<cv::DMatch> good_matches_m, good_matches_o;
    double tm, to, t;
    double ratio = 0.65;

    img1 = cv::imread(WORK_DIR + "assets/test1.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    img2 = cv::imread(WORK_DIR + "assets/test2.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
//    cv::resize(img1, img1, cv::Size(), 0.25, 0.25);
//    cv::resize(img2, img2, cv::Size(), 0.25, 0.25);
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    t = (double)cv::getTickCount();
    sift->detectAndCompute(img1_gray, cv::Mat(), kps1_o, desc1_o);
    sift->detectAndCompute(img2_gray, cv::Mat(), kps2_o, desc2_o);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher();
    matcher->knnMatch(desc1_o, desc2_o, matches_o, 2);
    for (int i = 0; i < matches_o.size(); i++) {
        if (matches_o[i][0].distance < 0.65 * matches_o[i][1].distance) {
            good_matches_o.push_back(matches_o[i][0]);
        }
    }
    cv::drawMatches(img1, kps1_o, img2, kps2_o, good_matches_o, img_match_o);
    std::cout << "opencv sift: " << t << "s, num of kpts: " << kps1_o.size() << ' ' << kps2_o.size() << ", num of matches: " << matches_o.size() << ", good matches: " << good_matches_o.size() << std::endl;

    MySift mysift;
    t = (double)cv::getTickCount();
    mysift.detect_and_compute(img1_gray, kps1_m, desc1_m);
    mysift.detect_and_compute(img2_gray, kps2_m, desc2_m);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    matcher->knnMatch(desc1_m, desc2_m, matches_m, 2);
    for (int i = 0; i < matches_m.size(); i++) {
        if (matches_m[i][0].distance < 0.65 * matches_m[i][1].distance) {
            good_matches_m.push_back(matches_m[i][0]);
        }
    }
    cv::drawMatches(img1, kps1_m, img2, kps2_m, good_matches_m, img_match_m);
    std::cout << "my sift: " << t << "s, num of kpts: " << kps1_m.size() << ' ' << kps2_m.size() << ", num of matches: " << matches_m.size() << ", good matches: " << good_matches_m.size() << std::endl;

    cv::imshow("opencv sift", img_match_o);
    cv::imshow("my sift", img_match_m);
    cv::waitKey(0);
}

void test_asift() {
    ASift *asift;
    cv::Mat img1, img2, img1_gray, img2_gray, img_match;
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat desc1, desc2;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    double t;

    asift = new ASift();
    img1 = cv::imread(WORK_DIR + "assets/test1.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    img2 = cv::imread(WORK_DIR + "assets/test2.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    cv::resize(img1, img1, cv::Size(), 0.25, 0.25);
    cv::resize(img2, img2, cv::Size(), 0.25, 0.25);

    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    t = (double)cv::getTickCount();
    asift->detect_and_compute(img1_gray, kps1, desc1);
    asift->detect_and_compute(img2_gray, kps2, desc2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "asift: " << t << "s, num of kpts: " << kps1.size() << ' ' << kps2.size() << std::endl;

    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher();
    matcher->knnMatch(desc1, desc2, matches, 2);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.65 * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_match);
    std::cout << "num of matches: " << matches.size() << ", good matches: " << good_matches.size() << std::endl;

    cv::imshow("asift", img_match);
    cv::waitKey(0);
}