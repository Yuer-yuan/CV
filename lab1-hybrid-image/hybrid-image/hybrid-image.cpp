#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define RESOLVE false

enum BorderType {
    BLACK = 0,
    REPLICATE,
    REFLECT,
};

cv::Mat gen_gaussian_kernel(int ksize, double sigma = -1.0, int dim = 2);
cv::Mat flip_kernel(const cv::Mat& kernel, int dim = 2);
cv::Mat transform(const cv::Mat& src);
cv::Mat pad(const cv::Mat& src, int size, int border_type = 0);
cv::Mat conv1D(const cv::Mat& src, const cv::Mat& kernel, int padding = 0, int stride = 1, int border_type = 0);
cv::Mat conv2D(const cv::Mat& src, const cv::Mat& kernel, int padding = 0, int stride = 1, int border_type = 0);
cv::Mat gaussian_blur(const cv::Mat& src, int ksize, double sigma = -1.0, bool resolve = false);
cv::Mat add(const cv::Mat& src1, const cv::Mat& src2);
cv::Mat subtract(const cv::Mat& src1, const cv::Mat& src2);
cv::Mat hybrid_img(const cv::Mat& img1, const cv::Mat& img2, int ksize, double sigma = -1.0);

int main(int argc, char** argv)
{
    const cv::Mat img1 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/hybrid-image/res/squirrel1.png");
    const cv::Mat img2 = cv::imread("/home/bill/mypro/CV/lab1-hybrid-image/hybrid-image/res/bunny1.png");

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
 * @param dim dimension of kernel, 1D (1*ksize) or 2D (ksize*ksize)
 * @return cv::Mat 
 */
cv::Mat gen_gaussian_kernel(int ksize, double sigma, int dim) {
    cv::Mat kernel;

    CV_Assert(ksize % 2 == 1 && ksize > 0);
    CV_Assert(dim == 1 || dim == 2);

    if (sigma < 0) {
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;  // ref: https://docs.opencv.org/4.7.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    }
    switch (dim)
    {
        case 1: {
            kernel = cv::Mat::zeros(1, ksize, CV_64FC1);
            double *p = kernel.ptr<double>(0);
            for (int i = 0; i < ksize; i++) {
                p[i] = exp(-0.5 * pow((i - (ksize - 1) / 2) / sigma, 2)) / (sigma * sqrt(2 * CV_PI));
            }
            break;
        }
        case 2: {
            kernel = cv::Mat::zeros(ksize, ksize, CV_64FC1);
            double *p;
            for (int i = 0; i < ksize; i++) {
                p = kernel.ptr<double>(i);
                for (int j = 0; j < ksize; j++) {
                    p[j] = exp(-0.5 * (pow((i - (ksize - 1) / 2) / sigma, 2) + pow((j - (ksize - 1) / 2) / sigma, 2))) / (2 * CV_PI * sigma * sigma);
                }
            }
            break;
        }
    }
    return kernel;
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
    CV_Assert(border_type == BorderType::BLACK || border_type == BorderType::REPLICATE);

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
    if (border_type == BorderType::REPLICATE) {
        p_src = src.ptr<cv::Vec3b>(0);
        for (int i = 0; i < padding; i++) {
            p_dst = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < padding; j++) {
                p_dst[j] = p_src[0];
            }
            for (int j = padding, j1 = 0; j1 < src.cols; j++, j1++) {
                p_dst[j] = p_src[j1];
            }
            for (int j = dst.cols - padding; j < dst.cols; j++) {
                p_dst[j] = p_src[src.cols - 1];
            }
        }
        for (int i = padding, i1 = 0; i1 < src.rows; i++, i1++) {
            p_src = src.ptr<cv::Vec3b>(i1);
            p_dst = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < padding; j++) {
                p_dst[j] = p_src[0];
            }
            for (int j = dst.cols - padding; j < dst.cols; j++) {
                p_dst[j] = p_src[src.cols - 1];
            }
        }
        p_src = src.ptr<cv::Vec3b>(src.rows - 1);
        for (int i = dst.rows - padding; i < dst.rows; i++) {
            p_dst = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < padding; j++) {
                p_dst[j] = p_src[0];
            }
            for (int j = padding, j1 = 0; j1 < src.cols; j++, j1++) {
                p_dst[j] = p_src[j1];
            }
            for (int j = dst.cols - padding; j < dst.cols; j++) {
                p_dst[j] = p_src[src.cols - 1];
            }
        }
    }
    if (border_type == BorderType::REFLECT) {
        // TODO
    }
    return dst;
}

/**
 * @brief Convolution of 2D image with 2D kernel (unoptimized version)
 *    V1.0 only support 8UC3 image and centrosymmetric kernel. extremly slow...
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
    CV_Assert(kernel.rows == kernel.cols);
    CV_Assert(padding >= 0);
    CV_Assert(stride > 0);
    CV_Assert(border_type == BorderType::BLACK || border_type == BorderType::REPLICATE);

    cv::Mat padded;
    int dst_rows, dst_cols;
    const cv::Vec3b *p_padded;
    cv::Vec3b *p_dst;
    const double *p_kernel;
    int kernel_size;

    kernel_size = kernel.rows;
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
 * @brief conv with 1D kernel
 * @version 1.0 only support the kernel of double type
 * @param src
 * @param kernel
 * @param padding
 * @param stride
 * @param border_type
 * @return
 */
cv::Mat conv1D(const cv::Mat& src, const cv::Mat& kernel, int padding, int stride, int border_type) {
    cv::Mat dst;

    CV_Assert(src.data);
    CV_Assert(kernel.data);
    CV_Assert(kernel.rows == 1);
    CV_Assert(padding >= 0);
    CV_Assert(stride >= 0);
    CV_Assert(border_type == BorderType::BLACK || border_type == BorderType::REPLICATE);

    cv::Mat padded, temp;
    int dst_rows, dst_cols, temp_rows, temp_cols;
    int kernel_size;
    const cv::Vec3b *p_padded;
    cv::Vec3b *p_dst, *p_temp;
    const double *p_kernel;

    kernel_size = kernel.cols;
    padded = pad(src, padding, border_type);
    p_kernel = kernel.ptr<double>(0);
    dst_rows = (padded.rows - kernel_size) / stride + 1;
    dst_cols = (padded.cols - kernel_size) / stride + 1;
    dst = cv::Mat::zeros(dst_cols, dst_rows, src.type());
    temp_rows = padded.rows;
    temp_cols = dst_cols;
    temp = cv::Mat::zeros(temp_rows, temp_cols, src.type());

    for (int i = 0; i < temp_rows; i++) {
        p_temp = temp.ptr<cv::Vec3b>(i);
        p_padded = padded.ptr<cv::Vec3b>(i);
        for (int j = 0; j < temp_cols; j++) {
            for (int k = 0; k < kernel_size; k++) {
                p_temp[j] += p_padded[j * stride + k] * p_kernel[k];
            }
        }
    }
    cv::Mat another = transform(temp);
    for (int i = 0; i < dst.rows; i++) {    // BUG FIXED: dst.rows (right), but dst_rows (wrong) ... don't know why...
        p_dst = dst.ptr<cv::Vec3b>(i);
        p_temp = another.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int k = 0; k < kernel_size; k++) {
                p_dst[j] += p_temp[j * stride + k] * p_kernel[k];
            }
        }
    }
    dst = transform(dst);
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

    cv::Mat img1_low_pass = gaussian_blur(img1, ksize, sigma, RESOLVE);
    cv::Mat img2_low_pass = gaussian_blur(img2, ksize, sigma, RESOLVE);
    cv::Mat img2_high_pass = subtract(img2, img2_low_pass);
    dst = add(img1_low_pass, img2_high_pass);

    cv::imwrite("img1_low_pass.jpg", img1_low_pass);
    cv::imwrite("img2_low_pass.jpg", img2_low_pass);
    cv::imwrite("img2_high_pass.jpg", img2_high_pass);
    cv::imwrite("hybrid_img.jpg", dst);
    return dst;
}

/**
 * @brief Gaussian blur
 * 
 * @param src 
 * @param ksize positive odd number
 * @param sigma 
 * @return cv::Mat 
 */
cv::Mat gaussian_blur(const cv::Mat &src, int ksize, double sigma, bool resolve) {
    cv::Mat dst;

    CV_Assert(src.data);
    CV_Assert(ksize > 0 && ksize % 2 == 1);

    cv::Mat kernel;
    if (resolve) {
        kernel = gen_gaussian_kernel(ksize, sigma, 1);
        dst = conv1D(src, kernel, (kernel.cols - 1) >> 1, 1, 1);
    } else {
        kernel = gen_gaussian_kernel(ksize, sigma, 2);
        dst = conv2D(src, kernel, (kernel.cols - 1) >> 1, 1, 1);
    }
    return dst;
}

/**
 * @brief flip kernel (for convolution may not be used)
 *        V1.0 only for double type
 * 
 * @param kernel 
 */
cv::Mat flip_kernel(const cv::Mat &kernel, int dim) {
    cv::Mat dst;

    CV_Assert(kernel.data);
    CV_Assert(kernel.type() == CV_64FC1);
    CV_Assert(dim == 1 || dim == 2);

    dst = kernel.clone();
    int rows = dst.rows;
    int cols = dst.cols;
    for (int i = 0; i < rows; i++) {    // flip horizontally
        for (int j = 0; j < cols >> 1; j++) {
            std::swap(dst.at<double>(i, j), dst.at<double>(i, cols - j - 1));
        }
    }
    for (int i = 0; i < rows >> 1; i++) {   // flip vertically
        for (int j = 0; j < cols; j++) {
            std::swap(dst.at<double>(i, j), dst.at<double>(rows - i - 1, j));
        }
    }
    return dst;
}

/**
 * @brief transform mat from size[m, n] to size[n, m]
 * @param src
 * @return
 */
cv::Mat transform(const cv::Mat& src) {
    cv::Mat dst;

    CV_Assert(src.data);

    int rows = src.cols;
    int cols = src.rows;
    dst = cv::Mat::zeros(rows, cols, src.type());
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (src.type() == CV_8UC3) {
                dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(j, i);
            }
            if (src.type() == CV_64F) {
                dst.at<double>(i, j) = src.at<double>(j, i);
            }
        }
    }
    return dst;
}