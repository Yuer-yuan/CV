#include "sift.h"
#include <iostream>
#include <opencv2/core/hal/hal.hpp>
#include "util.h"

#ifdef DISPLAY

static void show_gpyr(const std::vector<std::vector<cv::Mat>>& gpyr) {
    for (int i = 0; i < gpyr.size(); i++) {
        for (int j = 0; j < gpyr[i].size(); j++) {
            std::string title = "gpyr[" + std::to_string(i) + "][" + std::to_string(j) + "]";
            show_image(gpyr[i][j], title, false);
        }
    }
}

static void show_dogpyr(const std::vector<std::vector<cv::Mat>>& dogpyr) {
    for (int i = 0; i < dogpyr.size(); i++) {
        for (int j = 0; j < dogpyr[i].size(); j++) {
            std::string title = "dogpyr[" + std::to_string(i) + "][" + std::to_string(j) + "]";
            show_image(dogpyr[i][j], title, true);
        }
    }
}

static void show_adjusted_ptfs(const cv::Mat &dogimg, const std::vector<cv::Point2f> &ptfs_pre, const std::vector<cv::Point2f> &ptfs_aft) {
    cv::Mat img = dogimg.clone();
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < ptfs_pre.size(); i++) {
        cv::Point2f pt_pre = ptfs_pre[i];
        cv::Point2f pt_aft = ptfs_aft[i];
        cv::circle(img, pt_pre, 1, cv::Scalar(0, 255, 255), 5);
        cv::circle(img, pt_aft, 1, cv::Scalar(0, 255, 0), 5);
//        cv::line(img, pt_pre, pt_aft, cv::Scalar(255, 0, 0), 1);
    }
//    show_image(img, "adjusted_ptfs", true);
    img.convertTo(img, CV_8UC3);
    cv::imshow("adjusted_ptfs", img);
    cv::waitKey(0);
}

static void show_local_extrema(const cv::Mat &dogimg, const std::vector<cv::KeyPoint> &kpts, float *hist, int bins) {
    cv::Mat img = dogimg.clone();
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    float fct = 2000.f;
    for (int i = 0; i < kpts.size(); i++) {
        cv::Point2f pt = kpts[i].pt;
        float angle = kpts[i].angle;
        float mag = kpts[i].response;
        cv::arrowedLine(img, pt, pt + fct * cv::Point2f(mag * std::cos(angle), mag * std::sin(angle)), cv::Scalar(0, 255, 0), 1);
    }
    img.convertTo(img, CV_8UC3);
    // plot hist
    std::vector<float> hist_vec(hist, hist + bins);
    std::vector<float> xticks(bins);    // show angle in x axis (degree)
    for (int i = 0; i < bins; i++) {
        xticks[i] = i * 360.f / bins;
    }
    plt::xticks(xticks);
    plt::bar(hist_vec);
    plt::show();
    cv::imshow("local_extrema", img);
    cv::waitKey(0);
}

#endif  // DISPLAY

int MySift::num_octaves(const cv::Mat& img) const {
    int octaves, min_edge;
    min_edge = std::min(img.rows, img.cols);
    octaves = cvRound(std::log((float)min_edge) / std::log(2.f) - 2);
    if (double_size) {
        octaves++;
    }
    octaves = std::min(octaves, SIFT_MAX_OCTAVES);
    return octaves;
}

void MySift::create_initial_image(const cv::Mat &img, cv::Mat &init_img) const {
    cv::Mat gray, gray_fpt;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    gray.convertTo(gray_fpt, CV_32FC1, 1.0 / 255.0, 0);
    if (double_size) {
        double sig_diff = std::sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
        int ksize = 2 * cvRound(SIFT_GAUSSIAN_KERNEL_RATIO * sig_diff) + 1;
        cv::Mat dbl;
        cv::resize(gray_fpt, dbl, cv::Size(gray.cols * 2, gray.rows * 2), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(dbl, init_img, cv::Size(ksize, ksize), sig_diff, sig_diff);
    } else {
        double sig_diff = std::sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 1);
        int ksize = 2 * cvRound(SIFT_GAUSSIAN_KERNEL_RATIO * sig_diff) + 1;
        cv::GaussianBlur(gray_fpt, init_img, cv::Size(ksize, ksize), sig_diff, sig_diff);
    }
}

void MySift::build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr, int nOctaves) const {
    std::vector<double> sig(nOctaveLayers + 3); // relative sigmas in one octave
    gpyr.resize(nOctaves);
    for (int o = 0; o < nOctaves; o++) {
        gpyr[o].resize(nOctaveLayers + 3);
    }
    double k = std::pow(2.0, 1.0 / nOctaveLayers);  // scale factor between layers: 2^(1/3)
    sig[0] = sigma;
    for (int i = 1; i < nOctaveLayers + 3; i++) {
        double sig_prev = std::pow(k, i - 1) * sigma;
        double sig_curr = sig_prev * k;
        sig[i] = std::sqrt(sig_curr * sig_curr - sig_prev * sig_prev);
    }
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers + 3; i++) {
            cv::Mat& dst = gpyr[o][i];
            if (o == 0 && i == 0) {
                dst = init_img;
            } else if (i == 0) {
                const cv::Mat& src = gpyr[o - 1][nOctaveLayers];
                cv::resize(src, dst, cv::Size(src.cols / 2, src.rows / 2), 0, 0, cv::INTER_LINEAR);
            } else {
                const cv::Mat& src = gpyr[o][i - 1];
                int ksize = 2 * cvRound(SIFT_GAUSSIAN_KERNEL_RATIO * sig[i]) + 1;
                cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sig[i], sig[i]);
            }
        }
    }

#ifdef DISPLAY
//    show_gpyr(gpyr);
#endif
}

void MySift::build_dog_pyramid(const std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr) const {
    int nOctaves = (int)gpyr.size();
    dogpyr.resize(nOctaves);
    for (int o = 0; o < nOctaves; o++) {
        dogpyr[o].resize(nOctaveLayers + 2);
    }
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers + 2; i++) {
            dogpyr[o][i] = gpyr[o][i + 1] - gpyr[o][i];
        }
    }

#ifdef DISPLAY
//    show_dogpyr(dogpyr);
#endif
}

bool MySift::adjust_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, cv::KeyPoint &kpt, int octave, int layer, int r, int c) const {
    float xr, xc, xi;
    int iter = 0;
    for (; iter < SIFT_MAX_INTERP_STEPS; iter++) {
        const cv::Mat &prev = dogpyr[octave][layer - 1];
        const cv::Mat &curr = dogpyr[octave][layer];
        const cv::Mat &next = dogpyr[octave][layer + 1];
        // 3D quadratic interpolation
        float dx = (curr.at<float>(r, c + 1) - curr.at<float>(r, c - 1)) * 0.5f;
        float dy = (curr.at<float>(r + 1, c) - curr.at<float>(r - 1, c)) * 0.5f;
        float dz = (next.at<float>(r, c) - prev.at<float>(r, c)) * 0.5f;
        float v2 = curr.at<float>(r, c) * 2;
        float dxx = curr.at<float>(r, c + 1) + curr.at<float>(r, c - 1) - v2;
        float dyy = curr.at<float>(r + 1, c) + curr.at<float>(r - 1, c) - v2;
        float dzz = next.at<float>(r, c) + prev.at<float>(r, c) - v2;
        float dxy = (curr.at<float>(r + 1, c + 1) - curr.at<float>(r + 1, c - 1) - curr.at<float>(r - 1, c + 1) + curr.at<float>(r - 1, c - 1)) * 0.25f;
        float dxz = (next.at<float>(r, c + 1) - next.at<float>(r, c - 1) - prev.at<float>(r, c + 1) + prev.at<float>(r, c - 1)) * 0.25f;
        float dyz = (next.at<float>(r + 1, c) - next.at<float>(r - 1, c) - prev.at<float>(r + 1, c) + prev.at<float>(r - 1, c)) * 0.25f;
        cv::Matx33f H(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz);
        cv::Vec3f dD = cv::Vec3f(dx, dy, dz);
        cv::Vec3f X = H.solve(dD, cv::DECOMP_SVD);  // H * X = dD
        xc = -X[0];
        xr = -X[1];
        xi = -X[2];
        if (std::abs(xc) < 0.5f && std::abs(xr) < 0.5f && std::abs(xi) < 0.5f) {    // convergence
            break;
        }
        if (std::abs(xc) > (float)(INT_MAX / 3) || std::abs(xr) > (float)(INT_MAX / 3) || std::abs(xi) > (float)(INT_MAX / 3)) {
            return false;
        }
        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);
        if (layer < 1 || layer > nOctaveLayers || c < SIFT_IMG_BORDER || c > curr.cols - SIFT_IMG_BORDER || r < SIFT_IMG_BORDER || r > curr.rows - SIFT_IMG_BORDER) {
            return false;
        }
    }
    if (iter >= SIFT_MAX_INTERP_STEPS) {
        return false;
    }
    {   // eliminate low contrast points
        const cv::Mat &prev = dogpyr[octave][layer - 1];
        const cv::Mat &curr = dogpyr[octave][layer];
        const cv::Mat &next = dogpyr[octave][layer + 1];
        float dx = (curr.at<float>(r, c + 1) - curr.at<float>(r, c - 1)) * 0.5f;
        float dy = (curr.at<float>(r + 1, c) - curr.at<float>(r - 1, c)) * 0.5f;
        float dz = (next.at<float>(r, c) - prev.at<float>(r, c)) * 0.5f;
        cv::Matx31f dD(dx, dy, dz);
        float contr = curr.at<float>(r, c) + dD.dot(cv::Matx31f(xc, xr, xi)) * 0.5f;    // contrast
        if (std::abs(contr) * (float)nOctaveLayers < contrastThreshold) {
            return false;
        }
        // eliminate high edge response points
        float v2 = curr.at<float>(r, c) * 2;
        float dxx = curr.at<float>(r, c + 1) + curr.at<float>(r, c - 1) - v2;
        float dyy = curr.at<float>(r + 1, c) + curr.at<float>(r - 1, c) - v2;
        float dxy = (curr.at<float>(r + 1, c + 1) - curr.at<float>(r + 1, c - 1) - curr.at<float>(r - 1, c + 1) + curr.at<float>(r - 1, c - 1)) * 0.25f;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (det < 0 || tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det) {
            return false;
        }

        kpt.pt.x = (c + xc) * (1 << octave);
        kpt.pt.y = (r + xr) * (1 << octave);
        kpt.octave = octave + (layer << 8);
        kpt.size = sigma * powf(2.f, (layer + xi) / (float)nOctaveLayers) * (1 << octave);
        kpt.response = std::abs(contr);
        return true;
    }
}

float MySift::calc_orientation_hist(const cv::Mat &img, cv::Point pt, float scale, float *ori_hist) const {
    int radius = cvRound(SIFT_ORI_RADIUS * scale);
    int len = (radius * 2 + 1) * (radius * 2 + 1);
    float sig = SIFT_ORI_SIG_FCTR * scale;
    float expf_scale = -1.f / (2.f * sig * sig);
    int n = SIFT_ORI_HIST_BINS;
    cv::AutoBuffer<float> buf(len * 5 + n + 4);    // assign `nbins + 4` to `temphist` to soft hist
    //  X, Y, Mag,Ori, W, temphist - 2
    //0, l, 2l, 3l, 4l, 5l
    float *X = buf, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;
    float *temphist = W + len + 2;
    for (int i = 0; i < n; i++) {
        temphist[i] = 0.f;
    }
    int k = 0;
    for (int i = -radius; i <= radius; i++) {
        int y = pt.y + i;
        if (y <= 0 || y >= img.rows - 1) {
            continue;
        }
        for (int j = -radius; j <= radius; j++) {
            int x = pt.x + j;
            if (x <= 0 || x >= img.cols - 1) {
                continue;
            }
            float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            float dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
            X[k] = dx;
            Y[k] = dy;
            W[k] = (i * i + j * j) * expf_scale;
            k++;
        }
    }
    len = k;
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);
    for (int i = 0; i < len; i++) {
        int bin = cvRound(((float)n / 360.f) * Ori[i]);
        if (bin >= n) {
            bin -= n;
        }
        if (bin < 0) {
            bin += n;
        }
        temphist[bin] += W[i] * Mag[i];
    }
    temphist[-1] = temphist[n - 1];
    temphist[-2] = temphist[n - 2];
    temphist[n] = temphist[0];
    temphist[n + 1] = temphist[1];
    for (int i = 0; i < n; i++) {
        ori_hist[i] = (temphist[i - 2] + temphist[i + 2]) * 0.0625f + (temphist[i - 1] + temphist[i + 1]) * 0.25f + temphist[i] * 0.375f;
    }
    float ori_max = ori_hist[0];
    for (int i = 1; i < n; i++) {
        if (ori_max < ori_hist[i]) {
            ori_max = ori_hist[i];
        }
    }
    return ori_max;
}

void MySift::find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &keypoints) const {
    keypoints.clear();
    int nOctaves = (int)gpyr.size();
    float threshold = contrastThreshold / (float)nOctaveLayers; // 0.04 / nOctaveLayers
    float ori_hist[SIFT_ORI_HIST_BINS];
    cv::KeyPoint kpt;
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= nOctaveLayers; i++) {
#ifdef DISPLAY
//            std::vector<cv::Point2f> ptfs_pre, ptfs_aft;
#endif
            const cv::Mat& prev = dogpyr[o][i - 1];
            const cv::Mat& curr = dogpyr[o][i];
            const cv::Mat& next = dogpyr[o][i + 1];
            int step = (int)curr.step1();
            int rows = curr.rows;
            int cols = curr.cols;
            for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++) {
                const float* prev_row = prev.ptr<float>(r);
                const float* curr_row = curr.ptr<float>(r);
                const float* next_row = next.ptr<float>(r);
                for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++) {
                    int pr = c - step, cr = c, nr = c + step;
                    float val = curr_row[cr];
                    if (std::abs(val) > threshold &&
                        ((val > 0 && val >= curr_row[cr - 1] && val >= curr_row[cr + 1] &&
                        val >= curr_row[pr - 1] && val >= curr_row[pr] && val >= curr_row[pr + 1] &&
                        val >= curr_row[nr - 1] && val >= curr_row[nr] && val >= curr_row[nr + 1] &&
                        val >= prev_row[cr] && val >= prev_row[c - 1] && val >= prev_row[c + 1] &&
                        val >= prev_row[pr - 1] && val >= prev_row[pr] && val >= prev_row[pr + 1] &&
                        val >= prev_row[nr - 1] && val >= prev_row[nr] && val >= prev_row[nr + 1] &&
                        val >= next_row[cr] && val >= next_row[cr - 1] && val >= next_row[cr + 1] &&
                        val >= next_row[pr - 1] && val >= next_row[pr] && val >= next_row[pr + 1] &&
                        val >= next_row[nr - 1] && val >= next_row[nr] && val >= next_row[nr + 1]) ||
                        (val < 0 && val <= curr_row[cr - 1] && val <= curr_row[cr + 1] &&
                        val <= curr_row[pr - 1] && val <= curr_row[pr] && val <= curr_row[pr + 1] &&
                        val <= curr_row[nr - 1] && val <= curr_row[nr] && val <= curr_row[nr + 1] &&
                        val <= prev_row[cr] && val <= prev_row[cr - 1] && val <= prev_row[cr + 1] &&
                        val <= prev_row[pr - 1] && val <= prev_row[pr] && val <= prev_row[pr + 1] &&
                        val <= prev_row[nr - 1] && val <= prev_row[nr] && val <= prev_row[nr + 1] &&
                        val <= next_row[cr] && val <= next_row[cr - 1] && val <= next_row[cr + 1] &&
                        val <= next_row[pr - 1] && val <= next_row[pr] && val <= next_row[pr + 1] &&
                        val <= next_row[nr - 1] && val <= next_row[nr] && val <= next_row[nr + 1])))
                    {
#ifdef DISPLAY
//                        ptfs_pre.push_back(cv::Point2f(c, r));
#endif // DISPLAY
                        if (!adjust_local_extrema(dogpyr, kpt, o, i, r, c)) {
                            continue;
                        }
#ifdef DISPLAY
//                        ptfs_aft.push_back(cv::Point2f(kpt.pt.x, kpt.pt.y));
//                        std::vector<cv::KeyPoint> kpts;
#endif // DISPLAY
                        float scl_octv = kpt.size / (float)(1 << o);    // scale in current octave
                        float ori_max = calc_orientation_hist(gpyr[o][i], cv::Point(c, r), scl_octv, ori_hist);
                        float mag_thr = (float)(ori_max * SIFT_ORI_PEAK_RATIO);
                        for (int j = 0; j < SIFT_ORI_HIST_BINS; j++) {
                            int left = (j == 0) ? (SIFT_ORI_HIST_BINS - 1) : (j - 1);
                            int right = (j == SIFT_ORI_HIST_BINS - 1) ? 0 : (j + 1);
                            if (ori_hist[j] > ori_hist[left] && ori_hist[j] > ori_hist[right] && ori_hist[j] >= mag_thr) {
                                float bin = j + 0.5f * (ori_hist[left] - ori_hist[right]) / (ori_hist[left] - 2 * ori_hist[j] + ori_hist[right]);
                                if (bin < 0) {
                                    bin += (float)SIFT_ORI_HIST_BINS;
                                } else if (bin >= (float)SIFT_ORI_HIST_BINS) {
                                    bin -= (float)SIFT_ORI_HIST_BINS;
                                }
                                kpt.angle = (360.f / (float)SIFT_ORI_HIST_BINS) * bin;
#ifdef DISPLAY
//                                kpts.push_back(kpt);
#endif
                                keypoints.push_back(kpt);
                            }
                        }
#ifdef DISPLAY
//                        show_local_extrema(dogpyr[o][i], kpts, ori_hist, SIFT_ORI_HIST_BINS);
#endif
                    }
                }
            }
#ifdef DISPLAY
//            show_adjusted_ptfs(dogpyr[o][i], ptfs_pre, ptfs_aft);
#endif // DISPLAY
        }
    }
}

void MySift::detect(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr) const {
    int nOctaves = num_octaves(img);
    cv::Mat init_img;
    create_initial_image(img, init_img);
    build_gaussian_pyramid(init_img, gpyr, nOctaves);
    build_dog_pyramid(gpyr, dogpyr);
    find_scale_space_extrema(gpyr, dogpyr, keypoints);
    if (nfeatures && nfeatures < (int)keypoints.size()) {
        sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        keypoints.erase(keypoints.begin() + nfeatures, keypoints.end());
    }
}

void MySift::calc_sift_descriptor(const cv::Mat &img, cv::Point2f ptf, float ori, float scl, float *dst) const {
    cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
    int d = SIFT_DESCR_WIDTH;
    int n = SIFT_DESCR_HIST_BINS;
    float cos_t = cosf(-ori * (float)(CV_PI / 180));
    float sin_t = sinf(-ori * (float)(CV_PI / 180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f / (d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * (float)(d + 1) * sqrt(2) * 0.5f);
    int rows = img.rows;
    int cols = img.cols;
    radius = std::min(radius, (int)sqrt((double)rows * rows + (double)cols * cols));
    cos_t /= hist_width;
    sin_t /= hist_width;
    int len = (radius * 2 + 1) * (radius * 2 + 1);
    int histlen = (d + 2) * (d + 2) * (n + 2);
    cv::AutoBuffer<float> buf(len * 7 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;
    for (int i = 0; i < d + 2; i++) {
        for (int j = 0; j < d + 2; j++) {
            for (int k = 0; k < n + 2; k++) {
                hist[(i * (d + 2) + j) * (n + 2) + k] = 0.;
            }
        }
    }
    int k = 0;
    for (int i = -radius; i < radius; i++) {
        for (int j = -radius; j < radius; j++) {
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d / 2 - 0.5f;
            float cbin = c_rot + d / 2 - 0.5f;
            int r = pt.y + i;
            int c = pt.x + j;
            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
                float dx = img.at<float>(r, c + 1) - img.at<float>(r, c - 1);
                float dy = img.at<float>(r + 1, c) - img.at<float>(r - 1, c);
                X[k] = dx;
                Y[k] = dy;
                RBin[k] = rbin;
                CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
                k++;
            }
        }
    }
    len = k;
    cv::hal::fastAtan32f(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);
    cv::hal::exp32f(W, W, len);
    for (k = 0; k < len; k++) {
        float rbin = RBin[k];
        float cbin = CBin[k];
        float obin = (Ori[k] - ori) * bins_per_rad;
        float mag = Mag[k] * W[k];
        int r0 = cvFloor(rbin);
        int c0 = cvFloor(cbin);
        int o0 = cvFloor(obin);
        rbin -= r0;
        cbin -= c0;
        obin -= o0;
        if (o0 < 0) {
            o0 += n;
        }
        if (o0 >= n) {
            o0 -= n;
        }
        // histogram update using tri-linear interpolation
        float v_r1 = mag * rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;
        int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
        hist[idx] += v_rco000;
        hist[idx + 1] += v_rco001;
        hist[idx + (n + 2)] += v_rco010;
        hist[idx + (n + 3)] += v_rco011;
        hist[idx + (d + 2) * (n + 2)] += v_rco100;
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
        hist[idx + (d + 3) * (n + 2)] += v_rco110;
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
    }
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];
            for (k = 0; k < n; k++) {
                dst[(i * d + j) * n + k] = hist[idx + k];
            }
        }
    }
    std::vector<float> dst_vec(dst, dst + d * d * n);
    cv::normalize(dst_vec, dst_vec, 1, 0, cv::NORM_L2);
    for (int i = 0; i < dst_vec.size(); i++) {
        dst_vec[i] = std::min(dst_vec[i], SIFT_DESCR_MAG_THR);
    }
    cv::normalize(dst_vec, dst_vec, 1, 0, cv::NORM_L2);
    for (int i = 0; i < dst_vec.size(); i++) {
        dst[i] = dst_vec[i];
    }
}

void MySift::compute(const std::vector<std::vector<cv::Mat>> &gpyr, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const {
    descriptors = cv::Mat::zeros((int)keypoints.size(), SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS, CV_32F);
    for (int i = 0; i < (int)keypoints.size(); i++) {
        cv::KeyPoint &kpt = keypoints[i];
        int octave = kpt.octave & 255;
        int layer = (kpt.octave >> 8) & 255;
        float scale = kpt.size / (float)(1 << octave);
        float ori = kpt.angle;
        cv::Point2f ptf(kpt.pt.x / (float) (1 << octave), kpt.pt.y / (float) (1 << octave));
        calc_sift_descriptor(gpyr[octave][layer], ptf, ori, scale, descriptors.ptr<float>(i));
        if (double_size) {
            kpt.pt /= 2.f;
        }
    }
}

void MySift::detect_and_compute(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const {
    CV_Assert(!img.empty() && img.depth() == CV_8U);
    std::vector<std::vector<cv::Mat>> gpyr, dogpyr;
    detect(img, keypoints, gpyr, dogpyr);
    compute(gpyr, keypoints, descriptors);
}

void ASift::affine_skew(float tilt, float phi, cv::Mat &img, cv::Mat &mask, cv::Mat &ai) {
    int h = img.rows;
    int w = img.cols;
    mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(255));
    cv::Mat A = cv::Mat::eye(2, 3, CV_32F);
    if (phi != 0.0) {
        phi *= M_PI / 180.;
        double s = sin(phi);
        double c = cos(phi);
        A = (cv::Mat_<float>(2, 2) << c, -s, s, c);
        cv::Mat corners = (cv::Mat_<float>(4, 2) << 0, 0, w, 0, w, h, 0, h);
        cv::Mat tcorners = corners * A.t();
        cv::Mat tcorners_x, tcorners_y;
        tcorners.col(0).copyTo(tcorners_x);
        tcorners.col(1).copyTo(tcorners_y);
        std::vector<cv::Mat> channels;
        channels.push_back(tcorners_x);
        channels.push_back(tcorners_y);
        cv::merge(channels, tcorners);
        cv::Rect rect = cv::boundingRect(tcorners);
        A = (cv::Mat_<float>(2, 3) << c, -s, -rect.x, s, c, -rect.y);
        cv::warpAffine(img, img, A, cv::Size(rect.width, rect.height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    if (tilt != 1.0) {
        double s = 0.8 * sqrt(tilt * tilt - 1);
        GaussianBlur(img, img, cv::Size(0, 0), s, 0.01);
        resize(img, img, cv::Size(0, 0), 1.0 / tilt, 1.0, cv::INTER_NEAREST);
        A.row(0) = A.row(0) / tilt;
    }
    if (tilt != 1.0 || phi != 0.0) {
        h = img.rows;
        w = img.cols;
        cv::warpAffine(mask, mask, A, cv::Size(w, h), cv::INTER_NEAREST);
    }
    cv::invertAffineTransform(A, ai);
}

void ASift::detect_and_compute(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    keypoints.clear();
    descriptors = cv::Mat(0, 128, CV_32F);
    for (int tl = 1; tl < 6; tl++) {
        double t = pow(2, 0.5 * tl);
        for (int phi = 0; phi < 180; phi += (int)(72.0 / t)) {
            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            cv::Mat img_tilt, mask, ai;
            img.copyTo(img_tilt);
            affine_skew(t, phi, img_tilt, mask, ai);
#ifdef DISPLAY
//            cv::Mat img_disp;
//            bitwise_and(mask, img_tilt, img_disp);
//            cv::namedWindow( "Skew", cv::WINDOW_AUTOSIZE );// Create a window for display.
//            cv::imshow( "Skew", img_disp );
//            cv::waitKey(0);
#endif // DISPLAY
            MySift sift;
            sift.detect_and_compute(img_tilt, kps, desc);
            for (auto & kp : kps) {
                cv::Point3f kpt(kp.pt.x, kp.pt.y, 1);
                cv::Mat kpt_t = ai * cv::Mat(kpt);
                kp.pt.x = kpt_t.at<float>(0, 0);
                kp.pt.y = kpt_t.at<float>(1, 0);
            }
            keypoints.insert(keypoints.end(), kps.begin(), kps.end());
            descriptors.push_back(desc);
        }
    }
}
