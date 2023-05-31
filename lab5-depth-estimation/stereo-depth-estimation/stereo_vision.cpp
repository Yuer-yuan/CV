//
// Created by guo on 23-5-29.
//
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

#include "stereo_vision.h"

void sift_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2) {
    double ratio = 0.30f;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(img1, cv::Mat(), kpts1, desc1);
    sift->detectAndCompute(img2, cv::Mat(), kpts2, desc2);
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good;
    matcher->knnMatch(desc1, desc2, matches, 2);
    pts1.clear();
    pts2.clear();
    for (auto & m : matches) {
        if (m[0].distance < ratio * m[1].distance) {
            good.push_back(m[0]);
            pts1.push_back(kpts1[m[0].queryIdx].pt);
            pts2.push_back(kpts2[m[0].trainIdx].pt);
        }
    }

    cv::Mat img_match;
    cv::drawMatches(img1, kpts1, img2, kpts2, good, img_match, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("match", img_match);
    cv::waitKey(0);
}

void get_kpts_match(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, int method) {
    if (method == MATCH_SIFT) {
        sift_match(img1, img2, pts1, pts2);
    } else {
        CV_Assert(false);
    }
}

void get_fundamental_matrix(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &F, std::vector<cv::Point2f> &pts1_inlier, std::vector<cv::Point2f> &pts2_inlier) {
    std::vector<uint8_t> mask;
    F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1, 0.9999, 10000, mask);
    pts1_inlier.clear();
    pts2_inlier.clear();
    for (int i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            pts1_inlier.push_back(pts1[i]);
            pts2_inlier.push_back(pts2[i]);
        }
    }
}

void rectify(const cv::Mat &F, const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &img1_rectified, cv::Mat &img2_rectified, std::vector<cv::Point2f> &pts1_rectified, std::vector<cv::Point2f> &pts2_rectified) {
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(pts1, pts2, F, img1.size(), H1, H2);
    cv::warpPerspective(img1, img1_rectified, H1, img1.size());
    cv::warpPerspective(img2, img2_rectified, H2, img2.size());
    cv::perspectiveTransform(pts1, pts1_rectified, H1);
    cv::perspectiveTransform(pts2, pts2_rectified, H2);
}

void draw_epilines(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, const cv::Mat &F, cv::Mat &img1_epilines, cv::Mat &img2_epilines) {
    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(pts1, 1, F, lines1);
    cv::computeCorrespondEpilines(pts2, 2, F, lines2);
    cv::cvtColor(img1, img1_epilines, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_epilines, cv::COLOR_GRAY2BGR);
    for (auto & l : lines1) {
        cv::line(img1_epilines, cv::Point(0, -l[2] / l[1]), cv::Point(img1.cols, -(l[2] + l[0] * img1.cols) / l[1]), cv::Scalar(0, 255, 0));
    }
    for (auto & l : lines2) {
        cv::line(img2_epilines, cv::Point(0, -l[2] / l[1]), cv::Point(img2.cols, -(l[2] + l[0] * img2.cols) / l[1]), cv::Scalar(0, 255, 0));
    }
    for (auto & p : pts1) {
        cv::circle(img1_epilines, p, 3, cv::Scalar(255, 255, 0), -1);
    }
    for (auto & p : pts2) {
        cv::circle(img2_epilines, p, 3, cv::Scalar(255, 255, 0), -1);
    }
}

void get_disparity_ssd(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity) {
    int wnd_sz = 15;
    int max_disp = 30;
    int rows = img1.rows;
    int cols = img1.cols;
    int half_wnd_sz = wnd_sz >> 2;
    disparity = cv::Mat(img1.size(), CV_32FC1, cv::Scalar(0.f));
    for (int r = half_wnd_sz; r < rows - half_wnd_sz; ++r) {
        for (int c = half_wnd_sz; c < cols - half_wnd_sz; ++c) {
            float min_ssd = FLT_MAX;
            int best_disp = 0;
            for (int d = 0; d < max_disp; ++d) {
                float ssd = 0.f;
                for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
                    for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
                        float diff = img1.at<uchar>(r + i, c + j) - img2.at<uchar>(r + i, c + j - d);
                        ssd += diff * diff;
                    }
                }
                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    best_disp = d;
                }
            }
            disparity.at<float>(r, c) = best_disp;

            std::cout << "(" << r << ", " << c << ") matches (" << r << ", " << c - best_disp << "), disparity = " << best_disp << std::endl;
        }
    }

}

void get_disparity(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, int method) {
    if (method == DISP_SSD) {
        get_disparity_ssd(img1, img2, disparity);
    } else {
        CV_Assert(false);
    }
}

