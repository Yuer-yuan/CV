#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "stereo_vision.hpp"

int main(int argc, char *argv[]) {
  std::string const usage = "usage:\n"
                            "./build/stereo [dataset_dir] local "
                            "{methods to get correlation: SSD, SAD, NC}"
                            "\nor\n"
                            "./build/stereo [dataset_dir] semi";
  if (argc < 2) {
    throw std::runtime_error("please input [dataset_dir]\n" + usage);
  }
  std::string data_dir = argv[1];
  if (data_dir[data_dir.size() - 1] != '/') {
    data_dir += '/';
  }
  std::string const img1_path = data_dir + "im0.png",
                    img2_path = data_dir + "im1.png";
  cv::Mat const img1 = cv::imread(img1_path);
  cv::Mat const img2 = cv::imread(img2_path);

  // convert to gray color

  cv::Mat disparity, disparity_norm;
  if (argc == 3) {
    if (strcmp(argv[2], "semi") != 0) {
      throw std::runtime_error(usage);
    }
    SemiGlobalDisparity sgd(img1, img2);

  } else if (argc == 4) {
    if (strcmp(argv[2], "local") != 0) {
      throw std::runtime_error(usage);
    }

    using Corr = LocalDisparity::Corr;
    std::map<std::string, Corr> const corr_tb{
        {"SSD", Corr::SSD},
        {"SAD", Corr::SAD},
        {"NC", Corr::NC},
    };
    std::string const key = argv[3];
    auto it = corr_tb.find(key);
    if (it == corr_tb.end()) {
      throw std::runtime_error("cannot find methods for correlation\n" + usage);
    }
    LocalDisparity ld(img1, img2, it->second);
  } else {
    throw std::runtime_error(usage);
  }

  return 0;
}
