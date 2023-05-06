#include <iostream>
#include <getopt.h>
#include <cstring>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <filesystem>

// global variables

cv::Mat img_1, img_2;
cv::Mat img_1_gray, img_2_gray;
cv::Mat img_match;

const float scale = 0.2;

enum Wnd_Type { 
    IMG_1, 
    IMG_2, 
    DISTANCE_RATIO,
};
std::map<Wnd_Type, std::string> wnd_names = {
    {IMG_1, "img_1"},
    {IMG_2, "img_2"},
    {DISTANCE_RATIO, "distance ratio"},
};

enum Track_Bar_Type { 
    RATIO,
    PRECISION,
};
std::map<Track_Bar_Type, std::string> track_bar_names = {
    {RATIO, "ratio"},
    {PRECISION, "precision"},
};

cv::Ptr<cv::SiftDescriptorExtractor> sift_descriptor_extractor;
std::vector<cv::KeyPoint> key_points_1, key_points_2;
cv::Mat descriptors_1, descriptors_2;
std::vector<std::vector<cv::DMatch>> nearest_matches;
std::vector<cv::DMatch> matches_fillered_by_distance_ratio;

// command line options

char *img_1_path = nullptr, *img_2_path = nullptr, *save_dir = nullptr;
int max_num_units = 10, ratio_num_units = 0.6 * max_num_units;

const char *optstring = "H1:2:r:s:p:";

static struct option long_options[] = {
    {"img_1", required_argument, nullptr, '1'},
    {"img_2", required_argument, nullptr, '2'},
    {"ratio", required_argument, nullptr, 'r'},
    {"precision", required_argument, nullptr, 'p'},
    {"save", required_argument, nullptr, 's'},
    {"help", no_argument, nullptr, 'H'},
    {nullptr, 0, nullptr, 0}
};

const char *usage[] = {
    "Usage: <exec> -option1 <value1> -option2 <value2> ...",
    "Options:",
    "  -1, --img_1 <path>        Path to the first image",
    "  -2, --img_2 <path>        Path to the second image",
    "  -r, --ratio <value>      Number of ratio units",
    "  -p  --precision <value>  Max ratio units. Precision = ratio_num_units / max_num_units"
    "  -s, --save <path>        Path to the directory to save the blended image",
    nullptr
};

// function declarations

int main(int argc, char *argv[]);
void compute_key_points_and_descriptors(const cv::Mat &img_gray, std::vector<cv::KeyPoint> &key_points, cv::Mat &descriptors);
void match_key_points(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2, std::vector<std::vector<cv::DMatch>> &nearest_matches);
void filter_by_nearest_distance_ratio(const std::vector<std::vector<cv::DMatch>> &nearest_matches, const double &ratio, std::vector<cv::DMatch> &matches_filtered);
void ratio_test_call_back(int, void *);

int main(int argc, char *argv[]) {
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        switch (opt) {
            case '1': { 
                img_1_path = optarg;
                img_1 = cv::imread(img_1_path);
                CV_Assert(!img_1.empty());
                cv::resize(img_1, img_1, cv::Size(), scale, scale);   // in case the image is too large (optional)
                img_1_gray = img_1.clone();
                cv::cvtColor(img_1, img_1_gray, cv::COLOR_BGR2GRAY);
                break;
            }
            case '2': { 
                img_2_path = optarg;
                img_2 = cv::imread(img_2_path);
                CV_Assert(!img_2.empty());
                cv::resize(img_2, img_2, cv::Size(), scale, scale);
                img_2_gray = img_2.clone();
                cv::cvtColor(img_2, img_2_gray, cv::COLOR_BGR2GRAY);
                break; 
            }
            case 'r': { 
                ratio_num_units = atoi(optarg);
                CV_Assert(ratio_num_units > 0);
                break; 
            }
            case 'p': {
                max_num_units = atoi(optarg);
                CV_Assert(max_num_units > 0);
                break;
            }
            case 's': { 
                save_dir = optarg;
                if (!std::filesystem::exists(save_dir)) {
                    std::filesystem::create_directory(save_dir);
                }
                else {
                    CV_Assert(std::filesystem::is_directory(save_dir));
                }
                break;
            }
            case 'H': default: { 
                for (auto u : usage) {
                    std::cout << u << std::endl; 
                    return 0; 
                }
            }
        }
    }

    // start timer
    double duration = static_cast<double>(cv::getTickCount());

    // compute key points and descriptors
    compute_key_points_and_descriptors(img_1_gray, key_points_1, descriptors_1);
    compute_key_points_and_descriptors(img_2_gray, key_points_2, descriptors_2);

    // match key points and get the nearest matches
    match_key_points(descriptors_1, descriptors_2, nearest_matches);

    // if (ratio = nearest_distance / second_nearest_distance < ratio) then keep the match
    double ratio = (double)ratio_num_units / max_num_units;
    filter_by_nearest_distance_ratio(nearest_matches, ratio, matches_fillered_by_distance_ratio);

    // end timer
    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << "Time: " << duration << std::endl;

    // draw the matches
    cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_fillered_by_distance_ratio, img_match, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
#ifdef INTERACTIVE
    cv::imshow(wnd_names[DISTANCE_RATIO], img_match);
    cv::createTrackbar(track_bar_names[RATIO], wnd_names[DISTANCE_RATIO], &ratio_num_units, max_num_units, ratio_test_call_back);
    cv::createTrackbar(track_bar_names[PRECISION], wnd_names[DISTANCE_RATIO], &max_num_units, 1000, ratio_test_call_back);
    cv::waitKey(0);
#else
    cv::imwrite(std::string(save_dir) + "/" + wnd_names[DISTANCE_RATIO] + ".png", img_match);
#endif
    return 0;
}

void compute_key_points_and_descriptors(const cv::Mat &img_gray, std::vector<cv::KeyPoint> &key_points, cv::Mat &descriptors) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detect(img_gray, key_points);
    sift->compute(img_gray, key_points, descriptors);
}

void match_key_points(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2, std::vector<std::vector<cv::DMatch>> &nearest_matches) {
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descriptors_1, descriptors_2, nearest_matches, 2);
}

void filter_by_nearest_distance_ratio(const std::vector<std::vector<cv::DMatch>> &nearest_matches, const double &ratio, std::vector<cv::DMatch> &matches_filtered) {
    for (int i = 0; i < nearest_matches.size(); i++) {
        if (nearest_matches[i][0].distance < ratio * nearest_matches[i][1].distance) {
            matches_filtered.push_back(nearest_matches[i][0]);
        }
    }
}

void ratio_test_call_back(int, void *) {
    // set track bars
    if (!ratio_num_units) {
        ratio_num_units = 1;
        cv::setTrackbarPos(track_bar_names[RATIO], wnd_names[DISTANCE_RATIO], ratio_num_units);
    }
    if (max_num_units < ratio_num_units) {
        max_num_units = ratio_num_units;
        cv::setTrackbarPos(track_bar_names[PRECISION], wnd_names[DISTANCE_RATIO], max_num_units);
    }
    cv::setTrackbarMax(track_bar_names[RATIO], wnd_names[DISTANCE_RATIO], max_num_units);

    // clear filtered
    matches_fillered_by_distance_ratio.clear();

    // start timer
    double duration = static_cast<double>(cv::getTickCount());

    // compute key points and descriptors
    compute_key_points_and_descriptors(img_1_gray, key_points_1, descriptors_1);
    compute_key_points_and_descriptors(img_2_gray, key_points_2, descriptors_2);

    // match key points and get the nearest matches
    match_key_points(descriptors_1, descriptors_2, nearest_matches);

    // if (ratio = nearest_distance / second_nearest_distance < ratio) then keep the match
    double ratio = (double)ratio_num_units / max_num_units;
    filter_by_nearest_distance_ratio(nearest_matches, ratio, matches_fillered_by_distance_ratio);

    // end timer
    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();

    // log
    std::cout << "Time: " << duration << std::endl << "Ratio: " << ratio << std::endl << "Num of key points: " << key_points_1.size() << ' ' << key_points_2.size() << std::endl << "Num of good matches: " << matches_fillered_by_distance_ratio.size() << std::endl;

    // draw the matches
    // cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_fillered_by_distance_ratio, img_match, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_fillered_by_distance_ratio, img_match);
    cv::imshow(wnd_names[DISTANCE_RATIO], img_match);
}
