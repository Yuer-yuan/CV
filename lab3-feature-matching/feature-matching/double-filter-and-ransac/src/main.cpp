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
cv::Mat img_ratio;
cv::Mat img_ransac;

const float scale = 0.2;

enum Wnd_Type { 
    IMG_1, 
    IMG_2, 
    IMG_RATIO,
	IMG_RANSAC,
};
std::map<Wnd_Type, std::string> wnd_names = {
    {IMG_1, "img_1"},
    {IMG_2, "img_2"},
    {IMG_RATIO, "distance ratio"},
	{IMG_RANSAC, "ransac"},
};

enum Track_Bar_Type { 
    RATIO,
    PRECISION,
	RANSAC,
	CONFIDENCE,
};
std::map<Track_Bar_Type, std::string> track_bar_names = {
    {RATIO, "ratio"},
    {PRECISION, "precision"},
	{RANSAC, "ransac"},
	{CONFIDENCE, "confidence"},
};

cv::Ptr<cv::SiftDescriptorExtractor> sift_descriptor_extractor;
std::vector<cv::KeyPoint> key_points_1, key_points_2;
cv::Mat descriptors_1, descriptors_2;
std::vector<std::vector<cv::DMatch>> nearest_matches_1, nearest_matches_2;
std::vector<cv::DMatch> matches_filtered_by_distance_ratio_1, matches_filtered_by_distance_ratio_2, matches_double_filtered, matches_ransac_filtered;

// command line options

char *img_1_path = nullptr, *img_2_path = nullptr, *save_dir = nullptr;
int max_num_units = 100, ratio_num_units = 0.6 * max_num_units;
int ransac_reproj_thresh = 1, max_ransac_thresh = 10;
int confidence_units = 99, max_confidence_units = 100;

const char *optstring = "H1:2:r:s:p:a:c:";

static struct option long_options[] = {
    {"img_1", required_argument, nullptr, '1'},
    {"img_2", required_argument, nullptr, '2'},
    {"ratio", required_argument, nullptr, 'r'},
    {"precision", required_argument, nullptr, 'p'},
	{"ransac", required_argument, nullptr, 'a'},
	{"confidence", required_argument, nullptr, 'c'},
    {"save", required_argument, nullptr, 's'},
    {"help", no_argument, nullptr, 'H'},
    {nullptr, 0, nullptr, 0}
};

const char *usage[] = {
    "Usage: <exec> -option1 <value1> -option2 <value2> ...",
    "Options:",
    "  -1, --img_1 <path>        Path to the first image",
    "  -2, --img_2 <path>        Path to the second image",
    "  -r, --ratio <value>       Number of ratio units",
    "  -p, --precision <value>   Max ratio units. Precision = ratio_num_units / max_num_units",
	"  -a, --ransac <value>      Ransac reproj thresh.",
	"  -c, --confidence <value>  Ransac confidence.",
    "  -s, --save <path>         Path to the directory to save the blended image",
    nullptr
};

// function declarations

int main(int argc, char *argv[]);
void compute_key_points_and_descriptors(
    const cv::Mat &img_gray, 
    std::vector<cv::KeyPoint> &key_points, 
    cv::Mat &descriptors
);
void match_key_points(
    const cv::Mat &descriptors_1, 
    const cv::Mat &descriptors_2, 
    std::vector<std::vector<cv::DMatch>> &nearest_matches
);
void filter_by_nearest_distance_ratio(
    const std::vector<std::vector<cv::DMatch>> &nearest_matches, 
    const double &ratio, 
    std::vector<cv::DMatch> &matches_filtered
);
void double_filter(
	const std::vector<cv::DMatch> &matches_1, 
	const std::vector<cv::DMatch> &matches_2, 
	std::vector<cv::DMatch> &matches_double_filtered
);
void ransac(
	const std::vector<cv::KeyPoint> &key_points_1, 
	const std::vector<cv::KeyPoint> &key_points_2, 
	const std::vector<cv::DMatch> &matches_double_filtered, 
	const double ransac_reproj_thresh,
	const double confidence,
	std::vector<cv::DMatch> &matches_ransac_filtered);
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
			case 'a': {
				ransac_reproj_thresh = atoi(optarg);
				break;
			}
			case 'c': {
				confidence_units = atoi(optarg);
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
    cv::namedWindow(wnd_names[IMG_RATIO]);
    cv::createTrackbar(track_bar_names[RATIO], wnd_names[IMG_RATIO], &ratio_num_units, max_num_units, ratio_test_call_back);
    /* cv::createTrackbar(track_bar_names[PRECISION], wnd_names[IMG_RATIO], &max_num_units, 1000, ratio_test_call_back); */
	cv::createTrackbar(track_bar_names[RANSAC], wnd_names[IMG_RANSAC], &ransac_reproj_thresh, max_ransac_thresh, ratio_test_call_back);
	cv::createTrackbar(track_bar_names[CONFIDENCE], wnd_names[IMG_RANSAC], &confidence_units, max_confidence_units, ratio_test_call_back);
#ifdef INTERACTIVE
    cv::waitKey(0);
#else
    cv::imwrite(std::string(save_dir) + "/" + wnd_names[IMG_RATIO] + ".png", img_ratio);
	cv::imwrite(std::string(save_dir) + "/" + wnd_names[IMG_RANSAC] + ".png", img_ransac);
#endif
    return 0;
}

void compute_key_points_and_descriptors(
		const cv::Mat &img_gray, 
		std::vector<cv::KeyPoint> &key_points, 
		cv::Mat &descriptors
		) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detect(img_gray, key_points);
    sift->compute(img_gray, key_points, descriptors);
}

void match_key_points(
		const cv::Mat &descriptors_1, 
		const cv::Mat &descriptors_2, 
		std::vector<std::vector<cv::DMatch>> &nearest_matches
		) {
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descriptors_1, descriptors_2, nearest_matches, 2);
}

void filter_by_nearest_distance_ratio(
		const std::vector<std::vector<cv::DMatch>> &nearest_matches, 
		const double &ratio, 
		std::vector<cv::DMatch> &matches_filtered
		) {
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
        cv::setTrackbarPos(track_bar_names[RATIO], wnd_names[IMG_RATIO], ratio_num_units);
    }
    /* if (max_num_units < ratio_num_units) { */
        /* max_num_units = ratio_num_units; */
        /* cv::setTrackbarPos(track_bar_names[PRECISION], wnd_names[IMG_RATIO], max_num_units); */
    /* } */

    // clear filtered
    matches_filtered_by_distance_ratio_1.clear();
    matches_filtered_by_distance_ratio_2.clear();
    matches_double_filtered.clear();
	matches_ransac_filtered.clear();

    // start timer
    double duration = static_cast<double>(cv::getTickCount());

    // compute key points and descriptors
    compute_key_points_and_descriptors(img_1_gray, key_points_1, descriptors_1);
    compute_key_points_and_descriptors(img_2_gray, key_points_2, descriptors_2);

    // match key points and get the nearest matches
    match_key_points(descriptors_1, descriptors_2, nearest_matches_1);
    match_key_points(descriptors_2, descriptors_1, nearest_matches_2);

    // if (ratio = nearest_distance / second_nearest_distance < ratio) then keep the match
    double ratio = (double)ratio_num_units / max_num_units;
	double confidence = (double)confidence_units / max_confidence_units;
    filter_by_nearest_distance_ratio(nearest_matches_1, ratio, matches_filtered_by_distance_ratio_1);
    filter_by_nearest_distance_ratio(nearest_matches_2, ratio, matches_filtered_by_distance_ratio_2);
	double_filter(matches_filtered_by_distance_ratio_1, matches_filtered_by_distance_ratio_2, matches_double_filtered);
	ransac(key_points_1, key_points_2, matches_double_filtered, ransac_reproj_thresh, confidence, matches_ransac_filtered);

    // end timer
    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();

    // log
    std::cout 
		<< std::endl
		<< "Time: " << duration << std::endl << "Ratio: " << ratio << std::endl << "confidence: " << confidence << std::endl
        << "Num of key points:\n" << "[kp1]: " << key_points_1.size() << " [kp2]: " << key_points_2.size() << std::endl 
        << "Num of good matches:\n" << "[m1]: " << matches_filtered_by_distance_ratio_1.size() << " [m2]: " << matches_filtered_by_distance_ratio_2.size() << std::endl 
		<< "Num after double filtered: " << matches_double_filtered.size() << std::endl
		<< "Num after ransac filtered: " << matches_ransac_filtered.size() << std::endl;

    // draw the matches
    // cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_fillered_by_distance_ratio, img_ratio, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_double_filtered, img_ratio);
	cv::drawMatches(img_1, key_points_1, img_2, key_points_2, matches_ransac_filtered, img_ransac);
    cv::imshow(wnd_names[IMG_RATIO], img_ratio);
	cv::imshow(wnd_names[IMG_RANSAC], img_ransac);
}

void double_filter(
		const std::vector<cv::DMatch> &matches_1, 
		const std::vector<cv::DMatch> &matches_2, 
		std::vector<cv::DMatch> &matches_double_filtered
		) {
	std::set<std::pair<int, int>> double_match_pairs;
	for (auto match : matches_2) {
		if (double_match_pairs.find(std::make_pair(match.queryIdx, match.trainIdx)) == double_match_pairs.end()) {
			double_match_pairs.insert(std::make_pair(match.queryIdx, match.trainIdx));			
		}
	}
	for (auto match : matches_1) {
		if (double_match_pairs.find(std::make_pair(match.trainIdx, match.queryIdx)) != double_match_pairs.end()) {
			matches_double_filtered.push_back(match);
		}
	}
}

void ransac(
		const std::vector<cv::KeyPoint> &key_points_1, 
		const std::vector<cv::KeyPoint> &key_points_2, 
		const std::vector<cv::DMatch> &matches_double_filtered, 
		const double ransac_reproj_thresh,
		const double confidence,
		std::vector<cv::DMatch> &matches_ransac_filtered
		) {
    // ref: https://github.com/amin-abouee/robust-feature-matching/blob/master/src/RobustFeatureMatching.cpp#L18
	std::vector<cv::Point2f> points_1, points_2;
	for (auto match : matches_double_filtered) {
		int query_idx = match.queryIdx, train_idx = match.trainIdx;
		points_1.push_back(cv::Point2f(key_points_1[query_idx].pt.x, key_points_1[query_idx].pt.y));
		points_2.push_back(cv::Point2f(key_points_2[train_idx].pt.x, key_points_2[train_idx].pt.y));
	}
	std::vector<uchar> inliers(points_1.size(), 0);
	cv::Mat fundamental_mat = cv::findFundamentalMat(
        cv::Mat(points_1), 
        cv::Mat(points_2), 
        cv::FM_RANSAC, 
        ransac_reproj_thresh, 
        confidence, 
        inliers
    );
	int tmp_idx = 0;
	for (auto inlier : inliers) {
		if (inlier) {
			matches_ransac_filtered.push_back(cv::DMatch(matches_double_filtered[tmp_idx].queryIdx, matches_double_filtered[tmp_idx].trainIdx, 0));
		}
		tmp_idx++;
	}	
}
