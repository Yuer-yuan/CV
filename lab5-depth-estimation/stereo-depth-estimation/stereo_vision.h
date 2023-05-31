//
// Created by guo on 23-5-29.
//

#ifndef STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
#define STEREO_DEPTH_ESTIMATION_STEREO_VISION_H

#include <opencv2/opencv.hpp>

#define MATCH_SIFT 0

#define DISP_SSD 0

void sift_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2);
void get_kpts_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, int method = MATCH_SIFT);
void get_fundamental_matrix(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &F, std::vector<cv::Point2f> &pts1_inlier, std::vector<cv::Point2f> &pts2_inlier);
void rectify(const cv::Mat &F, const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &img1_rectified, cv::Mat &img2_rectified, std::vector<cv::Point2f> &pts1_rectified, std::vector<cv::Point2f> &pts2_rectified);
void draw_epilines(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, const cv::Mat &F, cv::Mat &img1_epilines, cv::Mat &img2_epilines);
void get_disparity_ssd(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity);
void get_disparity(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, int method = DISP_SSD);
#endif //STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
