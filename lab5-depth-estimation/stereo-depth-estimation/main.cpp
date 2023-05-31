//
// Created by guo on 23-5-29.
//

#include <opencv2/opencv.hpp>
#include <string>

#include "stereo_vision.h"
#include "data/config.h"

int main() {
//    std::string dataset("curule");
    std::string img1_path, img2_path;
//    get_image_path(dataset, img1_path, img2_path);
    img1_path = "/home/guo/mypro/CV/lab5-depth-estimation/stereo-depth-estimation/data/cones/im2.png";
    img2_path = "/home/guo/mypro/CV/lab5-depth-estimation/stereo-depth-estimation/data/cones/im6.png";
    cv::Mat img1 = cv::imread(img1_path);
    cv::Mat img2 = cv::imread(img2_path);
    CV_Assert(!img1.empty() && !img2.empty());
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

//    std::vector<cv::Point2f> pts1, pts2;
//    get_kpts_match(img1_gray, img2_gray, pts1, pts2, MATCH_SIFT);
//
//    cv::Mat F;
//    std::vector<cv::Point2f> pts1_inlier, pts2_inlier;
//    get_fundamental_matrix(pts1, pts2, F, pts1_inlier, pts2_inlier);
//
//    cv::Mat img1_rectified, img2_rectified;
//    std::vector<cv::Point2f> pts1_rectifed, pts2_rectified;
//    rectify(F, img1_gray, img2_gray, pts1_inlier, pts2_inlier, img1_rectified, img2_rectified, pts1_rectifed, pts2_rectified);
//    cv::imshow("img1_rectified", img1_rectified);
//    cv::imshow("img2_rectified", img2_rectified);
//    cv::waitKey(0);
//
//    cv::Mat img1_epilines, img2_epilines;
//    draw_epilines(img1_rectified, img2_rectified, pts1_rectifed, pts2_rectified, F, img1_epilines, img2_epilines);
//    cv::imshow("img1_epilines", img1_epilines);
//    cv::imshow("img2_epilines", img2_epilines);
//    cv::waitKey(0);

    cv::Mat disparity;
//    get_disparity(img1_rectified, img2_rectified, disparity, DISP_SSD);
    get_disparity(img1_gray, img2_gray, disparity, DISP_SSD);
    cv::imshow("disparity", disparity);
    cv::waitKey(0);
    return 0;
}