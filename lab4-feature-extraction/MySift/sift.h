#ifndef __SIFT_H__
#define __SIFT_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

class MySift {
private:
    int nfeatures;
    int nOctaveLayers;
    float contrastThreshold;
    float edgeThreshold;
    float sigma;
    bool double_size;

    int SIFT_MAX_OCTAVES = 8;
    float SIFT_INIT_SIGMA = 0.5f;
    float SIFT_GAUSSIAN_KERNEL_RATIO = 3.0f;
    int SIFT_ORI_HIST_BINS = 36;
    int SIFT_IMG_BORDER = 2;
    float SIFT_ORI_RADIUS = 4.5f;
    float SIFT_ORI_SIG_FCTR = 1.5f;
    float SIFT_ORI_PEAK_RATIO = 0.8f;
    int SIFT_MAX_INTERP_STEPS = 5;
    int SIFT_DESCR_WIDTH = 4;
    int SIFT_DESCR_HIST_BINS = 8;
    float SIFT_DESCR_SCL_FCTR = 3.0f;
    float SIFT_DESCR_MAG_THR = 0.2f;
    
public:
    MySift(
        int nfeatures = 0,
        int nOctaveLayers = 3,
        float contrastThreshold = 0.04f,
        float edgeThreshold = 10.0f,
        float sigma = 1.6f,
        bool double_size = true
    ) : nfeatures(nfeatures),
        nOctaveLayers(nOctaveLayers),
        contrastThreshold(contrastThreshold),
        edgeThreshold(edgeThreshold),
        sigma(sigma),
        double_size(double_size) {}

    int num_octaves(const cv::Mat& img) const;

    void create_initial_image(
        const cv::Mat& img, 
        cv::Mat& init_img
    ) const;

    void build_gaussian_pyramid(
        const cv::Mat& init_img, 
        std::vector<std::vector<cv::Mat>> &gpyr,
        int nOctaves
    ) const;

    void build_dog_pyramid(
        const std::vector<std::vector<cv::Mat>>& gpyr,
        std::vector<std::vector<cv::Mat>>& dogpyr
    ) const;

    bool adjust_local_extrema(
        const std::vector<std::vector<cv::Mat>>& dogpyr,
        cv::KeyPoint& kpt,
        int octave,
        int layer,
        int r,
        int c
    ) const;

    float calc_orientation_hist(
        const cv::Mat& img,
        cv::Point pt,
        float scale,
        float *hist
    ) const;

    void find_scale_space_extrema(
        const std::vector<std::vector<cv::Mat>>& gpyr,
        const std::vector<std::vector<cv::Mat>>& dogpyr,
        std::vector<cv::KeyPoint>& keypoints
    ) const;

    void calc_sift_descriptor(
        const cv::Mat& img,
        cv::Point2f pt,
        float ori,
        float scl,
        float *dst
    ) const;

    void detect(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>& keypoints,
        std::vector<std::vector<cv::Mat>> &gpyr,
        std::vector<std::vector<cv::Mat>> &dogpyr
    ) const;

    void compute(
        const std::vector<std::vector<cv::Mat>>& gpyr,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors
    ) const;

    void detect_and_compute(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors
    ) const;
};

class ASift {
public:
    ASift() {}

    void affine_skew(
        float tilt,
        float phi,
        cv::Mat& img,
        cv::Mat& mask,
        cv::Mat& ai
    );

    void detect_and_compute(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors
    );
};

#define DISPLAY // display key steps

#endif //__SIFT_H__
