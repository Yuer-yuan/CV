#include "canny/canny.h"
#include "util/util.h"
#include <iostream>
#include <getopt.h> // get option. Ref: https://blog.csdn.net/qq_34719188/article/details/83788199
#include <cstring>
#include <opencv2/opencv.hpp>

// global variables
char *img_path = nullptr;
char *save_dir = nullptr;
double low_threshold = 0.3 * 255;
double high_threshold = 0.6 * 255;

// options
const char *optstring = "i:s:Hl:h:";
static struct option long_options[] = {
    {"img", required_argument, nullptr, 'i'},
    {"low_threshold", required_argument, nullptr, 'l'},
    {"high_threshold", required_argument, nullptr, 'r'},
    {"save", required_argument, nullptr, 's'},
    {"help", no_argument, nullptr, 'H'},
    {nullptr, 0, nullptr, 0}
};

const char *usage[] = {
    "Usage: <exec> -option1 <value1> -option2 <value2> ...",
    "Options:",
    "  -i, --img <img_path>    Path of image to be processed",
    "  -l, --low_threshold     Low threshold of double threshold",
    "  -h, --high_threshold    High threshold of double threshold",
    "  -s, --save <save_dir>   Directory of save file",
    "  -H, --help              Show this help message and exit",
    nullptr
};

int main(int argc, char *argv[]) {
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                img_path = optarg;
                break;
            case 's':
                save_dir = optarg;
                break;
            case 'l':
                low_threshold = atof(optarg);
                break;
            case 'h':
                high_threshold = atof(optarg);
                break;
            case 'H': default:
                for (int i = 0; usage[i] != nullptr; ++i) {
                    std::cout << usage[i] << std::endl;
                }
                return 0;
        }
    }

    if (img_path == nullptr) {
        for (int i = 0; usage[i] != nullptr; ++i) {
            std::cout << usage[i] << std::endl;
        }
        return 0;
    }

    cv::Mat img_gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    img_gray = blur(img_gray);
    canny(img_gray, low_threshold, high_threshold, save_dir);

    return 0;
}