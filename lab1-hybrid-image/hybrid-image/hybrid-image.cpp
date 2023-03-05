#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat gen_gaussian_kernel(int size, double sigma = -1.0, int dim = 2);
cv::Mat pad(const cv::Mat& src, int size, int border_type = 0);
cv::Mat conv2D(const cv::Mat& src, const cv::Mat& kernel, int padding, int stride, int border_type = 0);
cv::Mat filter2D(const cv::Mat& src, const cv::Mat& kernel, int border_type = 0);
cv::Mat add(const cv::Mat& src1, const cv::Mat& src2);
cv::Mat subtract(const cv::Mat& src1, const cv::Mat& src2);
cv::Mat hybrid_img(const cv::Mat& img1, const cv::Mat& img2, int ksize, double sigma = -1.0);

int main(int argc, char** argv)
{
    const cv::Mat img1 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/hybrid-image/res/dog.jpg");
    const cv::Mat img2 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/hybrid-image/res/cat.jpg");

    double duration = static_cast<double>(cv::getTickCount());

    cv::Mat hybrid = hybrid_img(img1, img2, 33, 4);

    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    cv::imshow("hybrid", hybrid);
    cv::waitKey(0);
	return 0;
}

/**
 * @brief Generate Gaussian Kernel
 * 
 * @param size kernel size which should be possitive and odd
 * @param sigma standard deviation which can be set manually or calculated by size
 * @param dim dimension of kernel, 1D or 2D
 * @return cv::Mat 
 */
cv::Mat gen_gaussian_kernel(int size, double sigma, int dim) {
    cv::Mat kernel;

    CV_Assert(size % 2 == 1 && size > 0);
    CV_Assert(dim == 1 || dim == 2);

    if (sigma < 0) {
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8;
    }
    switch (dim)
    {
        case 1: {
            kernel = cv::Mat::zeros(1, size, CV_64FC1);
            double *p = kernel.ptr<double>(0);
            for (int i = 0; i < size; i++) {
                p[i] = exp(-0.5 * pow((i - (size - 1) / 2) / sigma, 2)) / (sigma * sqrt(2 * CV_PI));
            }
            break;
        }
        case 2: {
            kernel = cv::Mat::zeros(size, size, CV_64FC1);
            double *p;
            for (int i = 0; i < size; i++) {
                p = kernel.ptr<double>(i);
                for (int j = 0; j < size; j++) {
                    p[j] = exp(-0.5 * (pow((i - (size - 1) / 2) / sigma, 2) + pow((j - (size - 1) / 2) / sigma, 2))) / (2 * CV_PI * sigma * sigma);
                }
            }
            break;
        }
    }

    return kernel;
}

/**
 * @brief Convolution of 2D image with 2D kernel
 * 
 * @param src source image
 * @param kernel kernel
 * @param border_type `0`(default) for zero padding, `1` for replicate padding
 * @return cv::Mat 
 */
cv::Mat filter2D(const cv::Mat& src, const cv::Mat& kernel, int border_type) {
    cv::Mat dst;
    
    CV_Assert(src.data);
    CV_Assert(kernel.data);
    CV_Assert(border_type == 0 || border_type == 1);

    return conv2D(src, kernel, (kernel.cols - 1) >> 1, 1, border_type);
}

/**
 * @brief Pad image with zeros or replicate border. 
 *      V1.0 only support 8UC3 image
 * 
 * @param src 
 * @param padding 
 * @param border_type 
 * @return cv::Mat 
 */
cv::Mat pad(const cv::Mat& src, int padding, int border_type) {
    cv::Mat dst;

    CV_Assert(src.data);
    CV_Assert(padding >= 0);
    CV_Assert(border_type == 0 || border_type == 1);

    const cv::Vec3b *p_src;
    cv::Vec3b *p_dst;   // only deal with 8UC3
    int src_rows, src_cols;
    if (!padding) {
        return src;
    }
    dst = cv::Mat::zeros(src.rows + (padding << 1), src.cols + (padding << 1), src.type());
    for (int i = 0; i < src.rows; i++) {
        p_src = src.ptr<cv::Vec3b>(i);
        p_dst = dst.ptr<cv::Vec3b>(i + padding);
        for (int j = 0; j < src.cols; j++) {
            p_dst[j + padding] = p_src[j];
        }
    }
    if (border_type == 1) {
        // TODO
    }
    return dst;
}

/**
 * @brief Convolution of 2D image with 2D kernel (unoptimized version)
 *    V1.0 only support 8UC3 image and symmetric convolution
 * 
 * @param src source image
 * @param kernel kernel
 * @param padding padding size
 * @param stride stride size
 * @param border_type 0 for zero padding, 1 for replicate padding
 * @return cv::Mat 
 */
cv::Mat conv2D(const cv::Mat& src, const cv::Mat& kernel, int padding, int stride, int border_type) {
    cv::Mat dst;

    CV_Assert(src.data);
    CV_Assert(kernel.data);
    CV_Assert(padding >= 0);
    CV_Assert(stride > 0);
    CV_Assert(border_type == 0 || border_type == 1);

    int dst_rows, dst_cols;
    cv::Mat padded;
    const cv::Vec3b *p_padded;
    cv::Vec3b *p_dst;
    const double *p_kernel;
    int kernel_size = kernel.rows;

    dst_rows = (src.rows - kernel_size + (padding << 1)) / stride + 1;
    dst_cols = (src.cols - kernel_size + (padding << 1)) / stride + 1;
    dst = cv::Mat::zeros(dst_rows, dst_cols, src.type());
    padded = pad(src, padding, border_type);
    for (int i = 0; i < dst_rows; i++) {
        p_dst = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst_cols; j++) {
            for (int k = 0; k < kernel_size; k++) {
                p_padded = padded.ptr<cv::Vec3b>(i * stride + k);
                p_kernel = kernel.ptr<double>(k);
                for (int l = 0; l < kernel_size; l++) {
                    p_dst[j] += p_padded[j * stride + l] * p_kernel[l];
                }
            }
        }
    }
    return dst;
}

/**
 * @brief Subtract two images (src1 - src2)
 *        V1.0 Size of src1 must equal that of src2
 * 
 * @param src1 
 * @param src2 
 * @return cv::Mat 
 */
cv::Mat subtract(const cv::Mat& src1, const cv::Mat& src2) {
    cv::Mat dst;

    CV_Assert(src1.data && src2.data);

    dst = cv::Mat::zeros(src1.rows, src1.cols, src1.type());
    for (int i = 0; i < src1.rows; i++) {
        const cv::Vec3b *p_src1 = src1.ptr<cv::Vec3b>(i);
        const cv::Vec3b *p_src2 = src2.ptr<cv::Vec3b>(i);
        cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src1.cols; j++) {
            p_dst[j] = p_src1[j] - p_src2[j];
        }
    }
    return dst;
}

/**
 * @brief Add two images (src1 + src2)
 * 
 * @param src1 
 * @param src2 
 * @return cv::Mat 
 */
cv::Mat add(const cv::Mat& src1, const cv::Mat& src2) {
    cv::Mat dst;

    CV_Assert(src1.data && src2.data);

    dst = cv::Mat::zeros(src1.rows, src1.cols, src1.type());
    for (int i = 0; i < src1.rows; i++) {
        const cv::Vec3b *p_src1 = src1.ptr<cv::Vec3b>(i);
        const cv::Vec3b *p_src2 = src2.ptr<cv::Vec3b>(i);
        cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src1.cols; j++) {
            p_dst[j] = p_src1[j] + p_src2[j];
        }
    }
    return dst;
}

/**
 * @brief Hybrid image
 * 
 * @param img1 img to be low-pass filtered
 * @param img2 img to be high-pass filtered
 * @param ksize kernel size
 * @param sigma sigma of Gaussian kernel
 * @return cv::Mat 
 */
cv::Mat hybrid_img(const cv::Mat& img1, const cv::Mat& img2, int ksize, double sigma) {
    cv::Mat dst;

    CV_Assert(img1.data && img2.data);
    CV_Assert(ksize > 0 && ksize % 2 == 1);

    cv::Mat gaussian_kernel = gen_gaussian_kernel(ksize, sigma);
    cv::Mat img1_low_pass = filter2D(img1, gaussian_kernel);
    cv::Mat img2_low_pass = filter2D(img2, gaussian_kernel);
    cv::Mat img2_high_pass = subtract(img2, img2_low_pass);
    dst = add(img1_low_pass, img2_high_pass);

    cv::imwrite("img1_low_pass.jpg", img1_low_pass);
    cv::imwrite("img2_low_pass.jpg", img2_low_pass);
    cv::imwrite("img2_high_pass.jpg", img2_high_pass);
    cv::imwrite("hybrid_img.jpg", dst);
    return dst;
}
