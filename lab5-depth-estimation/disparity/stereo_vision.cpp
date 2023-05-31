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

float corr_ssd(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_sz) {
    float cost = 0.f;
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
        for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
            float diff = img1.at<uchar>(r + i, c + j) - img2.at<uchar>(r + i, c + j - d);
            cost += diff * diff;
        }
    }
    return cost;
}

float corr_sad(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size) {
    float cost = 0.f;
    for (int i = -half_wnd_size; i <= half_wnd_size; ++i) {
        for (int j = -half_wnd_size; j <= half_wnd_size; ++j) {
            cost += std::abs(img1.at<uchar>(r + i, c + j) - img2.at<uchar>(r + i, c + j - d));
        }
    }
    return cost;
}

float corr_c(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_size) {
// non-normalized correlation
    CV_Assert(false);
    return 0.f;
}

float corr_nc(const cv::Mat &img1, const cv::Mat &img2, int r, int c, int d, int half_wnd_sz) {
    float cost = 0.f;
    float mean1 = 0.f;
    float mean2 = 0.f;
    int wnd_sz = (half_wnd_sz * 2 + 1) * (half_wnd_sz * 2 + 1);
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
        for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
            mean1 += img1.at<uchar>(r + i, c + j);
            mean2 += img2.at<uchar>(r + i, c + j - d);
        }
    }
    mean1 /= wnd_sz * wnd_sz;
    mean2 /= wnd_sz * wnd_sz;
    float std1 = 0.f;
    float std2 = 0.f;
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
        for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
            float diff1 = img1.at<uchar>(r + i, c + j) - mean1;
            float diff2 = img2.at<uchar>(r + i, c + j - d) - mean2;
            std1 += diff1 * diff1;
            std2 += diff2 * diff2;
            cost += diff1 * diff2;
        }
    }
    std1 = sqrt(std1 / (wnd_sz * wnd_sz));
    std2 = sqrt(std2 / (wnd_sz * wnd_sz));
    cost /= std1 * std2;
    return cost;
}

void get_disparity_local(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm, int corr) {
    int wnd_sz = 10;
    int min_disp = 0;
    int max_disp = 64;
    int rows = img1.rows;
    int cols = img1.cols;
    int half_wnd_sz = wnd_sz >> 1;
    disparity = cv::Mat(img1.size(), CV_32FC1, cv::Scalar(0.f));
    for (int r = half_wnd_sz; r < rows - half_wnd_sz; ++r) {
        for (int c = half_wnd_sz; c < cols - half_wnd_sz; ++c) {
            float min_cost = FLT_MAX;
            int best_disp = 0;
            for (int d = min_disp; d <= max_disp; ++d) {
                float cost = 0.f;
                if (c - d < 0 || c - d >= cols) continue;
                if (corr == DISP_SSD) {
                    cost = corr_ssd(img1, img2, r, c, d, half_wnd_sz);
                } else if (corr == DISP_SAD) {
                    cost = corr_sad(img1, img2, r, c, d, half_wnd_sz);
                } else if (corr == DISP_C) {
                    cost = -corr_c(img1, img2, r, c, d, half_wnd_sz);   // negative correlation
                } else if (corr == DISP_NC) {
                    cost = -corr_nc(img1, img2, r, c, d, half_wnd_sz);
                } else {
                    CV_Assert(false);
                }
                if (cost < min_cost) {
                    min_cost = cost;
                    best_disp = d;
                }
            }
            disparity.at<float>(r, c) = best_disp;

//            std::cout << "(" << r << ", " << c << ") matches (" << r << ", " << c - best_disp << "), disparity = " << best_disp << std::endl;
        }
    }
    cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}

void census_transform(const cv::Mat &img, uint32_t *census_vals) {
    int rows = img.rows;
    int cols = img.cols;
    int i, j, m, n;
    uint8_t center, neighbor;
    uint32_t val;
    for (i = 2; i < rows - 2; i++) {
        for (j = 2; j < cols -2; j++) {
            val = 0;
            center = img.at<uint8_t>(i, j);
            for (m = -2; m <= 2; m++) {
                for (n = -2; n <= 2; n++) {
                    val <<= 1;
                    neighbor = img.at<uint8_t>(i + m, j + n);
                    if (neighbor > center) {
                        val += 1;
                    }
                }
            }
            census_vals[i * cols + j] = val;
        }
    }
}

uint8_t get_hamming_distance(const uint32_t a, const uint32_t b) {
    uint32_t val = a ^ b;
    uint8_t dist = 0;
    while (val) {
        ++dist;
        val &= val - 1;
    }
    return dist;
}

void get_corr_cost(const uint32_t *census_vals1, const uint32_t *census_vals2, uint8_t *corr_cost, const int min_disp, const int max_disp, const int rows, const int cols) {
    int disp_len = max_disp - min_disp + 1;
    int i, j, d;
    uint32_t val1, val2;
    uint8_t cost;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            val1 = census_vals1[i * cols + j];
            for (d = min_disp; d <= max_disp; d++) {
                if (j - d < 0) {
                    cost = 255;
                } else {
                    val2 = census_vals2[i * cols + j - d];
                    cost = get_hamming_distance(val1, val2);
                }
                corr_cost[i * cols * disp_len + j * disp_len + d] = cost;
            }
        }
    }
}

void cost_aggr_hori(const cv::Mat &img, const uint8_t *corr_cost, uint8_t *h, const int disp_len, const int p1, const int p2, int direction) {
    CV_Assert(direction == 1 || direction == -1);
    int rows = img.rows;
    int cols = img.cols;
    for (int i = 0; i < rows; i++) {
        const uint8_t *img_ptr = (direction == 1) ? (img.ptr<uint8_t>(i)) : (img.ptr<uint8_t>(i) + cols - 1);
		const uint8_t *corr_cost_ptr = (direction == 1) ? (corr_cost + i * cols * disp_len) : (corr_cost + i * cols * disp_len + (cols - 1) * disp_len);
		uint8_t *aggr_cost_ptr = (direction == 1) ? (h + i * cols * disp_len) : (h + i * cols * disp_len + (cols - 1) * disp_len);

        // init
		uint8_t curr = *img_ptr;
		uint8_t last = *img_ptr;
		std::vector<uint8_t> cost_last_path(disp_len + 2, UINT8_MAX);
		memcpy(aggr_cost_ptr, corr_cost_ptr, disp_len * sizeof(uint8_t));
		memcpy(&cost_last_path[1], aggr_cost_ptr, disp_len * sizeof(uint8_t));
		corr_cost_ptr += direction * disp_len;
		aggr_cost_ptr += direction * disp_len;
		img_ptr += direction;
		uint8_t min_cost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			min_cost_last_path = std::min(min_cost_last_path, cost);
		}

        // cost aggregation from the second column
		for (int j = 0; j < cols - 1; j++) {
			curr = *img_ptr;
			uint8_t min_cost = UINT8_MAX;
			for (int d = 0; d < disp_len; d++){
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + p1, Lr(p-r,d+1) + p1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  corr = corr_cost_ptr[d];
				const uint32_t l1 = cost_last_path[d + 1];
				const uint32_t l2 = cost_last_path[d] + p1;
				const uint32_t l3 = cost_last_path[d + 2] + p1;
				const uint32_t l4 = min_cost_last_path + std::max(p1, p2 / (abs(curr - last) + 1));
				const uint8_t cost = corr + std::min(std::min(l1, l2), std::min(l3, l4)) - min_cost_last_path;
				aggr_cost_ptr[d] = cost;
				min_cost = std::min(min_cost, cost);
			}
			min_cost_last_path = min_cost;
			memcpy(&cost_last_path[1], aggr_cost_ptr, disp_len * sizeof(uint8_t));
			corr_cost_ptr += direction * disp_len;
			aggr_cost_ptr += direction * disp_len;
			img_ptr += direction;
			last = curr;
		}
	}
}

void cost_aggr_vert(const cv::Mat &img, const uint8_t *corr_cost, uint8_t *v, const int disp_len, const int p1, const int p2, int direction) {
    CV_Assert(direction == 1 || direction == -1);
    int rows = img.rows;
    int cols = img.cols;
    for (int j = 0; j < cols; j++) {
        const uint8_t *img_ptr = (direction == 1) ? (img.ptr<uint8_t>(0) + j) : (img.ptr<uint8_t>(rows - 1) + j);
        const uint8_t *corr_cost_ptr = (direction == 1) ? (corr_cost + j * disp_len) : (corr_cost +
                                                                                        (rows - 1) * cols * disp_len +
                                                                                        j * disp_len);
        uint8_t *aggr_cost_ptr = (direction == 1) ? (v + j * disp_len) : (v + (rows - 1) * cols * disp_len +
                                                                          j * disp_len);

        // init
        uint8_t curr = *img_ptr;
        uint8_t last = *img_ptr;
        std::vector<uint8_t> cost_last_path(disp_len + 2, UINT8_MAX);
        memcpy(aggr_cost_ptr, corr_cost_ptr, disp_len * sizeof(uint8_t));
        memcpy(&cost_last_path[1], aggr_cost_ptr, disp_len * sizeof(uint8_t));
        corr_cost_ptr += direction * cols * disp_len;
        aggr_cost_ptr += direction * cols * disp_len;
        img_ptr += direction * img.step;

        uint8_t min_cost_last_path = UINT8_MAX;
        for (auto cost: cost_last_path) {
            min_cost_last_path = std::min(min_cost_last_path, cost);
        }

        // cost aggregation from the second row
        for (int i = 0; i < rows - 1; i++) {
            curr = *img_ptr;
            uint8_t min_cost = UINT8_MAX;
            for (int d = 0; d < disp_len; d++) {
                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + p1, Lr(p-r,d+1) + p1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
                const uint8_t corr = corr_cost_ptr[d];
                const uint32_t l1 = cost_last_path[d + 1];
                const uint32_t l2 = cost_last_path[d] + p1;
                const uint32_t l3 = cost_last_path[d + 2] + p1;
                const uint32_t l4 = min_cost_last_path + std::max(p1, p2 / (abs(curr - last) + 1));
                const uint8_t cost = corr + std::min(std::min(l1, l2), std::min(l3, l4)) - min_cost_last_path;
                aggr_cost_ptr[d] = cost;
                min_cost = std::min(min_cost, cost);
            }
            min_cost_last_path = min_cost;
            memcpy(&cost_last_path[1], aggr_cost_ptr, disp_len * sizeof(uint8_t));
            corr_cost_ptr += direction * cols * disp_len;
            aggr_cost_ptr += direction * cols * disp_len;
            img_ptr += direction * img.step;
            last = curr;
        }
    }
}

void cost_aggregation(const cv::Mat &img, const uint8_t *corr_cost, uint32_t *aggr_cost, const int min_disp, const int max_disp) {
    int rows = img.rows;
    int cols = img.cols;
    int disp_len = max_disp - min_disp + 1;
    int p1 = 10;
    int p2 = 150;

    uint8_t *h1 = new uint8_t[rows * cols * disp_len]();
    uint8_t *h2 = new uint8_t[rows * cols * disp_len]();
    uint8_t *v1 = new uint8_t[rows * cols * disp_len]();
    uint8_t *v2 = new uint8_t[rows * cols * disp_len]();

    cost_aggr_hori(img, corr_cost, h1, disp_len, p1, p2, 1);
    cost_aggr_hori(img, corr_cost, h2, disp_len, p1, p2, -1);
    cost_aggr_vert(img, corr_cost, v1, disp_len, p1, p2, 1);
    cost_aggr_vert(img, corr_cost, v2, disp_len, p1, p2, -1);

    for (int i = 0; i < rows * cols * disp_len; i++) {
        aggr_cost[i] = h1[i] + h2[i] + v1[i] + v2[i];
    }

    delete[] h1;
    delete[] h2;
    delete[] v1;
    delete[] v2;
}

void get_disp_from_cost(const cv::Mat &img, const uint32_t *aggr_cost, cv::Mat &disparity, const int min_disp, const int max_disp) {
    int rows = img.rows;
    int cols = img.cols;
    int disp_len = max_disp - min_disp + 1;
    disparity = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int min_cost = INT_MAX;
            int min_cost_2nd = INT_MAX;
            int best_disp = 0;
            int cost = 0;
            for (int d = min_disp; d <= max_disp; d++) {
                cost = aggr_cost[i * cols * disp_len + j * disp_len + d];
                if (cost < min_cost) {
                    min_cost = cost;
                    best_disp = d;
                } else if (cost < min_cost_2nd) {
                    min_cost_2nd = cost;
                }
            }

            // unique
            if (min_cost_2nd - min_cost < min_cost * 0.05) {
                disparity.at<float>(i, j) = 0;
            } else {
                disparity.at<float>(i, j) = best_disp;
            }

            // subpixel parabola interpolation
            if (best_disp > min_disp && best_disp < max_disp) {
                float d1 = aggr_cost[i * cols * disp_len + j * disp_len + best_disp - 1];
                float d2 = aggr_cost[i * cols * disp_len + j * disp_len + best_disp];
                float d3 = aggr_cost[i * cols * disp_len + j * disp_len + best_disp + 1];
                float dnorm = std::max(1.f, d1 + d3 - 2 * d2);
                float delta_d = (d1 - d3) / (2 * dnorm);
                disparity.at<float>(i, j) += delta_d;
            } else {
                disparity.at<float>(i, j) = 0;
            }
        }
    }
}

void get_right_disp(const cv::Mat &disp_left, cv::Mat &disp_right, const uint32_t *aggr_cost, const int min_disp, const int max_disp) {
    int rows = disp_left.rows;
    int cols = disp_left.cols;
    int disp_len = max_disp - min_disp + 1;
    disp_right = cv::Mat(disp_left.size(), CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int min_cost = INT_MAX;
            int min_cost_2nd = INT_MAX;
            int best_disp = 0;
            int cost = 0;
            for (int d = min_disp; d <= max_disp; d++) {
                int left_idx = j + d;
                if (left_idx < 0 || left_idx >= cols) {
                    continue;
                }
                cost = aggr_cost[i * cols * disp_len + left_idx * disp_len + d - min_disp];
                if (cost < min_cost) {
                    min_cost = cost;
                    best_disp = d;
                } else if (cost < min_cost_2nd) {
                    min_cost_2nd = cost;
                }
            }

            // unique
            if (min_cost_2nd - min_cost < min_cost * 0.05) {
                disp_right.at<float>(i, j) = 0;
            } else {
                disp_right.at<float>(i, j) = best_disp;
            }

            // subpixel parabola interpolation
            if (best_disp > min_disp && best_disp < max_disp) {
                float d1 = aggr_cost[i * cols * disp_len + (j + best_disp) * disp_len + best_disp - 1];
                float d2 = aggr_cost[i * cols * disp_len + (j + best_disp) * disp_len + best_disp];
                float d3 = aggr_cost[i * cols * disp_len + (j + best_disp) * disp_len + best_disp + 1];
                float dnorm = std::max(1.f, d1 + d3 - 2 * d2);
                float delta_d = (d1 - d3) / (2 * dnorm);
                disp_right.at<float>(i, j) += delta_d;
            } else {
                disp_right.at<float>(i, j) = 0;
            }
        }
    }
}

void lr_check(const cv::Mat &disp_left, const cv::Mat &disp_right, cv::Mat &disp_checked) {
    int rows = disp_left.rows;
    int cols = disp_left.cols;
    float thresh = 1.f;

    disp_checked = cv::Mat(disp_left.size(), CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float left = disp_left.at<float>(i, j);
            int right_idx = j - left + 0.5;
            if (right_idx >= 0 && right_idx < cols) {
                float right = disp_right.at<float>(i, right_idx);
                if (std::abs(left - right) > thresh) {
                    disp_checked.at<float>(i, j) = 0;
                } else {
                    disp_checked.at<float>(i, j) = left;
                }
            } else {
                disp_checked.at<float>(i, j) = 0;
            }
        }
    }
}

void get_disparity_semi_global(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm) {
    int rows = img1.rows;
    int cols = img1.cols;
    int min_disp = 0;
    int max_disp = 60;
    int disp_len = max_disp - min_disp + 1;
    uint32_t *census_vals1 = new uint32_t[rows * cols]();
    uint32_t *census_vals2 = new uint32_t[rows * cols]();
    census_transform(img1, census_vals1);
    census_transform(img2, census_vals2);

    uint8_t *corr_cost = new uint8_t[rows * cols * disp_len]();
    get_corr_cost(census_vals1, census_vals2, corr_cost, min_disp, max_disp, rows, cols);

    delete[] census_vals1;  // not used anymore
    delete[] census_vals2;

    uint32_t *aggr_cost = new uint32_t[rows * cols * disp_len]();
    cost_aggregation(img1, corr_cost, aggr_cost, min_disp, max_disp);

    delete[] corr_cost;

    get_disp_from_cost(img1, aggr_cost, disparity, min_disp, max_disp);
    cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("disparity from aggr_cost", disparity_norm);
    cv::waitKey(0);

    cv::Mat disp_right;
    get_right_disp(disparity, disp_right, aggr_cost, min_disp, max_disp);
    cv::normalize(disp_right, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("disparity from right", disparity_norm);
    cv::waitKey(0);

    delete[] aggr_cost;

    cv::Mat disp_checked;
    lr_check(disparity, disp_right, disp_checked);
    cv::normalize(disp_checked, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("disparity after lr_check", disparity_norm);
    cv::waitKey(0);

    // 中值滤波
    cv::medianBlur(disp_checked, disp_checked, 3);
    cv::normalize(disp_checked, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("disparity after medianBlur", disparity_norm);
    cv::waitKey(0);

}

void get_disparity(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &disparity, cv::Mat &disparity_norm, int method, int corr) {
    if (method == DISP_LOCAL) {
        get_disparity_local(img1, img2, disparity, disparity_norm, corr);
    } else if (method == DISP_SEMI) {
        get_disparity_semi_global(img1, img2, disparity, disparity_norm);
    } else {
        CV_Assert(false);
    }
}

