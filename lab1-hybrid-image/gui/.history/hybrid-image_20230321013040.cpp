#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
int main()
{
    cv::Mat R = cv::Mat(200, 200, CV_8UC3);
    cv::randu(R, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::imshow("R", R);
    cv::waitKey(0);
	return 0;
}