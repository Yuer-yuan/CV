#ifndef STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
#define STEREO_DEPTH_ESTIMATION_STEREO_VISION_H

#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/highgui.hpp"
#include <cstdint>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#define MATCH_SIFT 0

class SemiGlobalDisparity {
  static constexpr size_t MIN_DISP = 0;
  static constexpr size_t MAX_DISP = 60;
  static constexpr size_t DISP_LEN = MAX_DISP - MIN_DISP + 1;
  static constexpr size_t CENSUS_KNL_SZ = 5; // kernel size of census transform
  static constexpr uint8_t P1 = 10;
  static constexpr uint8_t P2 = 150;

  size_t rows_, cols_;
  cv::Mat lhs_, rhs_;
  cv::Mat lhs_gray_, rhs_gray_;
  cv::Mat disp_, disp_right_, disp_checked_, disp_blurred_;

public:
  SemiGlobalDisparity(cv::Mat const &lhs, cv::Mat const &rhs) {
    if (lhs.data == nullptr || rhs.data == nullptr) {
      throw std::runtime_error("given images are empty");
    }
    if (lhs.cols != rhs.cols || lhs.rows != rhs.rows) {
      throw std::runtime_error("Sizes of pictures must match");
    }
    lhs_ = lhs.clone(), rhs_ = rhs.clone();
    rows_ = lhs_.rows, cols_ = lhs_.cols;
    cv::cvtColor(lhs_, lhs_gray_, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rhs_, rhs_gray_, cv::COLOR_BGR2GRAY);
    calc_disp();
  }

private:
  void calc_disp() {
    auto const census_vals1 = census_transform(lhs_gray_),
               census_vals2 = census_transform(rhs_gray_);

    auto const corr_cost = get_corr_cost(census_vals1, census_vals2);

    auto const aggr_cost = cost_aggregation(lhs_gray_, corr_cost);

    auto show_norm_img = [&](cv::Mat const &img, std::string const &wnd_name) {
      cv::Mat norm;
      cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      cv::imshow(wnd_name, norm);
      cv::waitKey(0);
    };

    disp_ = get_disp_from_cost(lhs_gray_, aggr_cost);
    show_norm_img(disp_, "disparity from aggr_cost");

    disp_right_ = get_right_disp(aggr_cost);
    show_norm_img(disp_right_, "disparity from right");

    disp_checked_ = lr_check(disp_, disp_right_);
    show_norm_img(disp_right_, "disparity after lr_check");

    cv::medianBlur(disp_checked_, disp_blurred_, 3);
    show_norm_img(disp_right_, "disparity after medianBlur");

    auto show_heat_img = [&](cv::Mat const &img, std::string const &wnd_name) {
      cv::Mat norm, heat;
      cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      cv::applyColorMap(norm, heat, cv::COLORMAP_JET);
      cv::imshow(wnd_name, heat);
      cv::waitKey(0);
    };

    show_heat_img(disp_blurred_, "heat image");
  };

  std::vector<uint32_t> census_transform(cv::Mat const &img) const {
    size_t const rows = img.rows, cols = img.cols;
    if (rows < CENSUS_KNL_SZ || cols < CENSUS_KNL_SZ) {
      throw std::runtime_error("image is too small to be census-transformed");
    }
    // we adopt `uint32_t` to store census value
    // for kernel size 5, 32 > 25, ok
    if (8 * sizeof(uint32_t) < CENSUS_KNL_SZ * CENSUS_KNL_SZ) {
      throw std::runtime_error("census kernel size is too large");
    }
    std::vector<uint32_t> census_vals(rows * cols);
    for (size_t i = CENSUS_KNL_SZ; i < rows - CENSUS_KNL_SZ / 2; i++) {
      for (size_t j = CENSUS_KNL_SZ; j < cols - CENSUS_KNL_SZ / 2; j++) {
        uint32_t val = 0;
        uint8_t const center = img.at<uint8_t>(i, j);
        for (size_t m = 0; m <= 5; m++) {
          for (size_t n = 0; n <= 5; n++) {
            val *= 2;
            uint8_t const neighbor = img.at<uint8_t>(i + m - 2, j + n - 2);
            if (neighbor > center) {
              val += 1;
            }
          }
        }
        census_vals[i * cols + j] = val;
      }
    }
    return census_vals;
  }

  std::vector<uint8_t>
  get_corr_cost(std::vector<uint32_t> const &census_vals1,
                std::vector<uint32_t> const &census_vals2) const {
    std::vector<uint8_t> corr_cost(rows_ * cols_ * DISP_LEN);
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        uint32_t const val1 = census_vals1[i * cols_ + j];
        for (auto d = MIN_DISP; d <= MAX_DISP; d++) {
          uint8_t cost = 0;
          if (j < d) {
            cost = std::numeric_limits<uint8_t>::max();
          } else {
            auto const val2 = census_vals2[i * cols_ + j - d];
            cost = get_hamming_distance(val1, val2);
          }
          corr_cost[i * cols_ * DISP_LEN + j * DISP_LEN + d] = cost;
        }
      }
    }
    return corr_cost;
  }

  static uint8_t get_hamming_distance(uint32_t const a, uint32_t const b) {
    uint32_t val = a ^ b;
    uint8_t dist = 0;
    while (val) {
      ++dist;
      val &= val - 1;
    }
    return dist;
  }

  std::vector<uint32_t>
  cost_aggregation(cv::Mat const &img,
                   std::vector<uint8_t> const &corr_cost) const {
    auto const size = rows_ * cols_ * DISP_LEN;
    std::vector<uint32_t> aggr_cost(size);
    std::vector<uint8_t> h1(size), h2(size), v1(size), v2(size);

    cost_aggr_hori(img, &corr_cost[0], &h1[0], 1);
    cost_aggr_hori(img, &corr_cost[0], &h2[0], -1);
    cost_aggr_vert(img, &corr_cost[0], &v1[0], 1);
    cost_aggr_vert(img, &corr_cost[0], &v2[0], -1);

    for (size_t i = 0; i < rows_ * cols_ * DISP_LEN; i++) {
      aggr_cost[i] = h1[i] + h2[i] + v1[i] + v2[i];
    }
    return aggr_cost;
  }

  void cost_aggr_hori(cv::Mat const &img, uint8_t const *corr_cost, uint8_t *h,
                      int const direction) const {
    CV_Assert(direction == 1 || direction == -1);
    for (size_t i = 0; i < rows_; i++) {
      uint8_t const *img_ptr = (direction == 1)
                                   ? (img.ptr<uint8_t>(i))
                                   : (img.ptr<uint8_t>(i) + cols_ - 1);
      uint8_t const *corr_cost_ptr =
          (direction == 1)
              ? (corr_cost + i * cols_ * DISP_LEN)
              : (corr_cost + i * cols_ * DISP_LEN + (cols_ - 1) * DISP_LEN);
      uint8_t *aggr_cost_ptr =
          (direction == 1)
              ? (h + i * cols_ * DISP_LEN)
              : (h + i * cols_ * DISP_LEN + (cols_ - 1) * DISP_LEN);

      // init
      uint8_t curr = *img_ptr;
      uint8_t last = *img_ptr;
      std::vector<uint8_t> cost_last_path(DISP_LEN + 2,
                                          std::numeric_limits<uint8_t>::max());
      memcpy(aggr_cost_ptr, corr_cost_ptr, DISP_LEN * sizeof(uint8_t));
      memcpy(&cost_last_path[1], aggr_cost_ptr, DISP_LEN * sizeof(uint8_t));
      corr_cost_ptr += direction * DISP_LEN;
      aggr_cost_ptr += direction * DISP_LEN;
      img_ptr += direction;
      uint8_t min_cost_last_path = std::numeric_limits<uint8_t>::max();
      for (auto cost : cost_last_path) {
        min_cost_last_path = std::min(min_cost_last_path, cost);
      }

      // cost aggregation from the second column
      for (size_t j = 0; j < cols_ - 1; j++) {
        curr = *img_ptr;
        uint8_t min_cost = std::numeric_limits<uint8_t>::max();
        for (size_t d = 0; d < DISP_LEN; d++) {
          // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + p1, Lr(p-r,d+1) +
          // p1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
          const uint8_t corr = corr_cost_ptr[d];
          const uint32_t l1 = cost_last_path[d + 1];
          const uint32_t l2 = cost_last_path[d] + P1;
          const uint32_t l3 = cost_last_path[d + 2] + P1;
          const uint32_t l4 =
              min_cost_last_path +
              std::max((uint32_t)P1, (uint32_t)P2 / (abs(curr - last) + 1));
          const uint8_t cost = corr +
                               std::min(std::min(l1, l2), std::min(l3, l4)) -
                               min_cost_last_path;
          aggr_cost_ptr[d] = cost;
          min_cost = std::min(min_cost, cost);
        }
        min_cost_last_path = min_cost;
        memcpy(&cost_last_path[1], aggr_cost_ptr, DISP_LEN * sizeof(uint8_t));
        corr_cost_ptr += direction * DISP_LEN;
        aggr_cost_ptr += direction * DISP_LEN;
        img_ptr += direction;
        last = curr;
      }
    }
  }

  void cost_aggr_vert(cv::Mat const &img, uint8_t const *corr_cost, uint8_t *v,
                      int const direction) const {
    CV_Assert(direction == 1 || direction == -1);
    for (size_t j = 0; j < cols_; j++) {
      const uint8_t *img_ptr = (direction == 1)
                                   ? (img.ptr<uint8_t>(0) + j)
                                   : (img.ptr<uint8_t>(rows_ - 1) + j);
      const uint8_t *corr_cost_ptr =
          (direction == 1)
              ? (corr_cost + j * DISP_LEN)
              : (corr_cost + (rows_ - 1) * cols_ * DISP_LEN + j * DISP_LEN);
      uint8_t *aggr_cost_ptr =
          (direction == 1)
              ? (v + j * DISP_LEN)
              : (v + (rows_ - 1) * cols_ * DISP_LEN + j * DISP_LEN);

      // init
      uint8_t curr = *img_ptr;
      uint8_t last = *img_ptr;
      std::vector<uint8_t> cost_last_path(DISP_LEN + 2,
                                          std::numeric_limits<uint8_t>::max());
      memcpy(aggr_cost_ptr, corr_cost_ptr, DISP_LEN * sizeof(uint8_t));
      memcpy(&cost_last_path[1], aggr_cost_ptr, DISP_LEN * sizeof(uint8_t));
      corr_cost_ptr += direction * cols_ * DISP_LEN;
      aggr_cost_ptr += direction * cols_ * DISP_LEN;
      img_ptr += direction * img.step;

      uint8_t min_cost_last_path = std::numeric_limits<uint8_t>::max();
      for (auto cost : cost_last_path) {
        min_cost_last_path = std::min(min_cost_last_path, cost);
      }

      // cost aggregation from the second row
      for (size_t i = 0; i < rows_ - 1; i++) {
        curr = *img_ptr;
        uint8_t min_cost = std::numeric_limits<uint8_t>::max();
        for (size_t d = 0; d < DISP_LEN; d++) {
          // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + p1, Lr(p-r,d+1) +
          // p1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
          const uint8_t corr = corr_cost_ptr[d];
          const uint32_t l1 = cost_last_path[d + 1];
          const uint32_t l2 = cost_last_path[d] + P1;
          const uint32_t l3 = cost_last_path[d + 2] + P1;
          const uint32_t l4 =
              min_cost_last_path +
              std::max((size_t)P1, (size_t)P2 / (abs(curr - last) + 1));
          const uint8_t cost = corr +
                               std::min(std::min(l1, l2), std::min(l3, l4)) -
                               min_cost_last_path;
          aggr_cost_ptr[d] = cost;
          min_cost = std::min(min_cost, cost);
        }
        min_cost_last_path = min_cost;
        memcpy(&cost_last_path[1], aggr_cost_ptr, DISP_LEN * sizeof(uint8_t));
        corr_cost_ptr += direction * cols_ * DISP_LEN;
        aggr_cost_ptr += direction * cols_ * DISP_LEN;
        img_ptr += direction * img.step;
        last = curr;
      }
    }
  }

  cv::Mat get_disp_from_cost(cv::Mat const &img,
                             std::vector<uint32_t> const &aggr_cost) {
    cv::Mat disparity = cv::Mat(img.size(), CV_32FC1, cv::Scalar(0));
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        auto min_cost = std::numeric_limits<uint32_t>::max(),
             min_cost_2nd = std::numeric_limits<uint32_t>::max();
        uint32_t best_disp = 0;
        uint32_t cost = 0;
        for (auto d = MIN_DISP; d <= MAX_DISP; d++) {
          cost = aggr_cost[i * cols_ * DISP_LEN + j * DISP_LEN + d];
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
        if (best_disp > MIN_DISP && best_disp < MAX_DISP) {
          float const d1 =
              aggr_cost[i * cols_ * DISP_LEN + j * DISP_LEN + best_disp - 1];
          float const d2 =
              aggr_cost[i * cols_ * DISP_LEN + j * DISP_LEN + best_disp];
          float const d3 =
              aggr_cost[i * cols_ * DISP_LEN + j * DISP_LEN + best_disp + 1];
          float const dnorm = std::max(1.f, d1 + d3 - 2 * d2);
          float const delta_d = (d1 - d3) / (2 * dnorm);
          disparity.at<float>(i, j) += delta_d;
        } else {
          disparity.at<float>(i, j) = 0;
        }
      }
    }
    return disparity;
  }

  cv::Mat get_right_disp(std::vector<uint32_t> const &aggr_cost) {
    cv::Mat const disp_left = disp_;
    cv::Mat disp_right = cv::Mat(disp_left.size(), CV_32FC1, cv::Scalar(0));
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        auto min_cost = std::numeric_limits<uint32_t>::max(),
             min_cost_2nd = std::numeric_limits<uint32_t>::max();
        uint32_t best_disp = 0;
        uint32_t cost = 0;
        for (auto d = MIN_DISP; d <= MAX_DISP; d++) {
          auto const left_idx = j + d;
          if (left_idx >= cols_) {
            continue;
          }
          cost = aggr_cost[i * cols_ * DISP_LEN + left_idx * DISP_LEN + d -
                           MIN_DISP];
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
        if (best_disp > MIN_DISP && best_disp < MAX_DISP) {
          float const d1 =
              aggr_cost[i * cols_ * DISP_LEN + (j + best_disp) * DISP_LEN +
                        best_disp - 1];
          float const d2 = aggr_cost[i * cols_ * DISP_LEN +
                                     (j + best_disp) * DISP_LEN + best_disp];
          float const d3 =
              aggr_cost[i * cols_ * DISP_LEN + (j + best_disp) * DISP_LEN +
                        best_disp + 1];
          float const dnorm = std::max(1.f, d1 + d3 - 2 * d2);
          float const delta_d = (d1 - d3) / (2 * dnorm);
          disp_right.at<float>(i, j) += delta_d;
        } else {
          disp_right.at<float>(i, j) = 0;
        }
      }
    }
    return disp_right;
  }

  cv::Mat lr_check(cv::Mat const &disp_left, cv::Mat const &disp_right) {
    constexpr float THRESH = 1.f;
    cv::Mat disp_checked = cv::Mat(disp_left.size(), CV_32FC1, cv::Scalar(0));
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        float const left = disp_left.at<float>(i, j);
        int const right_idx = j - left + 0.5;
        if (right_idx >= 0 && right_idx < (int)cols_) {
          float const right = disp_right.at<float>(i, right_idx);
          if (std::abs(left - right) > THRESH) {
            disp_checked.at<float>(i, j) = 0;
          } else {
            disp_checked.at<float>(i, j) = left;
          }
        } else {
          disp_checked.at<float>(i, j) = 0;
        }
      }
    }
    return disp_checked;
  }
};

class LocalDisparity {
public:
  enum Corr : uint8_t { SSD = 0, SAD, NC }; // only for local methods

private:
  static constexpr size_t WND_SZ = 10;
  static constexpr size_t HALF_WND_SZ = WND_SZ / 2;
  static constexpr size_t MIN_DISP = 0;
  static constexpr size_t MAX_DISP = 64;

  size_t rows_, cols_;
  Corr corr_;

  cv::Mat lhs_, rhs_;
  cv::Mat lhs_gray_, rhs_gray_;
  cv::Mat disp_;

public:
  LocalDisparity(cv::Mat const &lhs, cv::Mat const &rhs, Corr corr) {
    if (lhs.data == nullptr || rhs.data == nullptr) {
      throw std::runtime_error("given images are empty");
    }
    if (lhs.cols != rhs.cols || lhs.rows != rhs.rows) {
      throw std::runtime_error("Sizes of pictures must match");
    }
    lhs_ = lhs.clone(), rhs_ = rhs.clone();
    rows_ = lhs_.rows, cols_ = lhs_.cols;
    if (rows_ < HALF_WND_SZ || cols_ < HALF_WND_SZ) {
      throw std::runtime_error("image is too small");
    }

    cv::cvtColor(lhs_, lhs_gray_, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rhs_, rhs_gray_, cv::COLOR_BGR2GRAY);
    corr_ = corr;
    calc_disp();
  };

private:
  void calc_disp() {
    cv::Mat disparity = cv::Mat(lhs_gray_.size(), CV_32FC1, cv::Scalar(0.f));
    for (size_t r = HALF_WND_SZ; r + HALF_WND_SZ < rows_; r++) {
      for (size_t c = HALF_WND_SZ; c + HALF_WND_SZ < cols_; c++) {
        float min_cost = std::numeric_limits<float>::max();
        uint32_t best_disp = 0;
        for (auto d = MIN_DISP; d <= MAX_DISP; d++) {
          float cost = 0.f;
          if (c < d || c >= cols_ + d) {
            continue;
          }
          if (corr_ == Corr::SSD) {
            cost = corr_ssd(r, c, d);
          }
          if (corr_ == Corr::SAD) {
            cost = corr_sad(r, c, d);
          }
          if (corr_ == Corr::NC) {
            cost = -corr_nc(r, c, d);
          }
          if (cost < min_cost) {
            min_cost = cost;
            best_disp = d;
          }
        }
        disparity.at<float>(r, c) = best_disp;
      }
    }
    disp_ = disparity;

    auto show_norm_img = [&](cv::Mat const &img, std::string const &wnd_name) {
      cv::Mat norm;
      cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      cv::imshow(wnd_name, norm);
      cv::waitKey(0);
    };
    show_norm_img(disp_, "disparity");
  }

  float corr_ssd(int r, int c, int d) const {
    int const half_wnd_sz = HALF_WND_SZ;
    float cost = 0.f;
    for (int i = half_wnd_sz; i <= half_wnd_sz; ++i) {
      for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
        float const diff = lhs_gray_.at<uchar>(r + i, c + j) -
                           rhs_gray_.at<uchar>(r + i, c + j - d);
        cost += diff * diff;
      }
    }
    return cost;
  }

  float corr_sad(int r, int c, int d) const {
    int const half_wnd_sz = HALF_WND_SZ;
    float cost = 0.f;
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
      for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
        cost += std::abs(lhs_gray_.at<uchar>(r + i, c + j) -
                         rhs_gray_.at<uchar>(r + i, c + j - d));
      }
    }
    return cost;
  }

  float corr_nc(int r, int c, int d) const {
    float cost = 0.f;
    float mean1 = 0.f;
    float mean2 = 0.f;
    int const half_wnd_sz = HALF_WND_SZ;
    int wnd_sz = (half_wnd_sz * 2 + 1) * (half_wnd_sz * 2 + 1);
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
      for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
        mean1 += lhs_gray_.at<uchar>(r + i, c + j);
        mean2 += rhs_gray_.at<uchar>(r + i, c + j - d);
      }
    }
    mean1 /= wnd_sz * wnd_sz;
    mean2 /= wnd_sz * wnd_sz;
    float std1 = 0.f;
    float std2 = 0.f;
    for (int i = -half_wnd_sz; i <= half_wnd_sz; ++i) {
      for (int j = -half_wnd_sz; j <= half_wnd_sz; ++j) {
        float diff1 = lhs_gray_.at<uchar>(r + i, c + j) - mean1;
        float diff2 = rhs_gray_.at<uchar>(r + i, c + j - d) - mean2;
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
};

#endif // STEREO_DEPTH_ESTIMATION_STEREO_VISION_H
