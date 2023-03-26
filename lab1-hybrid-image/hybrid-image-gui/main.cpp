#include "mainwindow.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QApplication>

const std::string workspace = "/home/bill/mypro/CV/lab1-hybrid-image/hybrid-image-gui";

int main(int argc, char *argv[])
{
    cv::Mat image = cv::imread(workspace + "/images/cat.jpg");
    cv::imshow("image", image);
    cv::waitKey(0);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
