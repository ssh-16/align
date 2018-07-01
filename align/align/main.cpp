#include <cmath>
#include <io.h>
#include <string>
#include <algorithm>

#include <cv.h>
#include <highgui.h>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#define DEBUG

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static const int MAX_FEATURES = 5000;
static const float GOOD_MATCH_PERCENT = 0.2f;

Mat process(Mat &color_ref, Mat &color_dst)
{
	Mat ref, dst;
	cvtColor(color_ref, ref, CV_BGR2GRAY);
	cvtColor(color_dst, dst, CV_BGR2GRAY);
	
	double min_ref, max_ref, min_dst, max_dst;
	minMaxLoc(ref, &min_ref, &max_ref);
	minMaxLoc(dst, &min_dst, &max_dst);
	subtract(ref, min_ref, ref);
	subtract(dst, min_dst, dst);

	printf("min_ref=%lf, min_dst=%lf\n", min_ref, min_dst);

	Ptr<Feature2D> feature = ORB::create(MAX_FEATURES);	// SIFT might not be suitable for this case according to the results.
	vector<KeyPoint> kp_ref, kp_dst;
	feature->detect(ref, kp_ref);
	feature->detect(dst, kp_dst);

	Mat desc_ref, desc_dst;
	feature->compute(ref, kp_ref, desc_ref);
	feature->compute(dst, kp_dst, desc_dst);

	vector<DMatch> matches;
	//BFMatcher matcher;	// FlannBasedMatcher doesn't work well, too.
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(desc_ref, desc_dst, matches, Mat());

	sort(matches.begin(), matches.end());

	const int num_good_matches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + num_good_matches, matches.end());

	vector<Point2f> p_ref;
	vector<Point2f> p_dst;

	for (int i = 0; i < matches.size(); i++) {
		p_ref.emplace_back(kp_ref[matches[i].queryIdx].pt);
		p_dst.emplace_back(kp_dst[matches[i].trainIdx].pt);
	}
	vector<uchar> good_mask(p_ref.size());
	Mat homo = findHomography(p_dst, p_ref, CV_FM_RANSAC, 3.0, good_mask);

	printf("before: matched point = %d\n", matches.size());
	vector<DMatch> good;
	for (int i = 0; i < good_mask.size(); i++) {
		if (good_mask[i]) {
			good.emplace_back(matches[i]);
		}
	}
	matches.swap(good);
	printf("after:  matched point = %d\n", matches.size());

	cout << "homo=\n" << homo << "\n";

	Mat n_dst(ref.size(), CV_8UC3);
	warpPerspective(color_dst, n_dst, homo, ref.size());

#if defined(DEBUG)
	imshow("new dst image", n_dst);
	cvWaitKey(0);

	Mat matched;
	drawMatches(ref, kp_ref, dst, kp_dst, matches, matched);

	double scale = 0.4;
	Size sz = Size(scale * matched.cols, scale * matched.rows);
	Mat scaled_matched = Mat(sz, CV_32S);
	resize(matched, scaled_matched, sz);

	imshow("comparison", scaled_matched);
	cvWaitKey(0);
#endif

	return n_dst;
}

void test(string ref_file, string dst_file)
{
	Mat ref = imread(ref_file);
	Mat dst = imread(dst_file);

	Mat n_dst = process(ref, dst);

	string n_dst_file = dst_file + ".new.jpg";
	imwrite(n_dst_file, n_dst);
}

int main()
{
	const string work_dir = "D:\\Documents\\Git4VS\\repo\\align\\align\\align\\";
	string ref_file = work_dir + "images\\_MG_5706.jpg";
	string dst_file[] = {
		work_dir + "images\\_MG_5746.jpg",
		work_dir + "images\\_MG_5821.jpg",
		work_dir + "images\\_MG_5835.jpg",
		work_dir + "images\\_MG_5706_m.jpg",
		work_dir + "images\\_MG_5918.jpg",
		work_dir + "images\\_MG_5918_m.jpg",
		work_dir + "images\\_MG_6232.jpg",
		work_dir + "images\\_MG_6248.jpg",
		work_dir + "images\\_MG_6246.jpg",
		work_dir + "images\\_MG_6378.jpg",
		work_dir + "images\\_MG_6464.jpg",
		work_dir + "images\\_MG_6512.jpg",
		work_dir + "images\\_MG_6557.jpg",
	};

	
	for (int i = 1; i < 11; i++) {
		test(ref_file, dst_file[i]);
	}
	

	return 0;
}
