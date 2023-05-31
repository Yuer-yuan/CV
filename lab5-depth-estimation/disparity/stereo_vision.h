//
// Created by guo on 23-5-29.
//

#ifndef STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
#define STEREO_DEPTH_ESTIMATION_STEREO_VISION_H

#include <opencv2/opencv.hpp>

#define MATCH_SIFT 0

#define DISP_LOCAL 0
#define DISP_SEMI 1
#define DISP_GLOBAL 2
#define DISP_SSD 0
#define DISP_SAD 1
#define DISP_C 2
#define DISP_NC 3

void sift_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2);
void get_kpts_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, int method = MATCH_SIFT);
void get_fundamental_matrix(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &F, std::vector<cv::Point2f> &pts1_inlier, std::vector<cv::Point2f> &pts2_inlier);
void rectify(const cv::Mat &F, const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &img1_rectified, cv::Mat &img2_rectified, std::vector<cv::Point2f> &pts1_rectified, std::vector<cv::Point2f> &pts2_rectified);
void draw_epilines(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, const cv::Mat &F, cv::Mat &img1_epilines, cv::Mat &img2_epilines);

float corr_ssd(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size);
float corr_sad(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size);
float corr_c(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size);
float corr_nc(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size);
void get_disparity_local(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm);

uint8_t get_hamming_distance(const uint32_t a, const uint32_t b);
void census_transform(const cv::Mat &img, uint32_t *census_vals);
void get_corr_cost(const uint32_t *census_vals1, const uint32_t *census_vals2, uint8_t *corr_cost, const int min_disp, const int max_disp, const int rows, const int cols);
void cost_aggr_hori(const cv::Mat &img, const uint8_t *corr_cost, uint8_t *h, const int disp_len, const int p1, const int p2, int direction);
void cost_aggr_hori(const cv::Mat &img, const uint8_t *corr_cost, uint8_t *h, const int disp_len, const int p1, const int p2, int direction);
void cost_aggregation(const cv::Mat &img, const uint8_t *corr_cost, uint32_t * aggr_cost, const int rows, const int cols, const int min_disp, const int max_disp);
void get_disp_from_cost(const cv::Mat &img, const uint32_t *aggr_cost, cv::Mat &disp, const int min_disp, const int max_disp);
void get_right_disp(const cv::Mat &disp_left, cv::Mat &disp_right, const uint32_t *aggr_cost, const int min_disp, const int max_disp);
void lr_check(const cv::Mat &disp_left, const cv::Mat &disp_right, cv::Mat &disp_checked);
void get_disparity_semi_global(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm);

void get_disparity(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm, int method = DISP_LOCAL, int corr = DISP_SSD);

#endif //STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
