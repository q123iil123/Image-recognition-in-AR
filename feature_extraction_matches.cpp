#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<time.h>
using namespace std;
using namespace cv;

clock_t start, stop;
double duration;

int find_feature_matches(Mat img_1, Mat img_2);
int feature_extraction_matches(Mat img1, Mat img2);


int main()
{
	Mat img1 = imread("D:/c++_exercise/Feture/Feture/121/2.jpg");
	Mat img2 = imread("D:/c++_exercise/Feture/Feture/target/x6.jpg");

	feature_extraction_matches(img1, img2);

	waitKey(0);
}

int feature_extraction_matches(Mat img1, Mat img2)
{
	Ptr<FeatureDetector> detector = AKAZE::create();       //用的是AKAZE，想用ORB的话吧AKAZE改成ORB就可以了
	Ptr<DescriptorExtractor> descriptor = AKAZE::create();

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(img1, keypoints_1);
	detector->detect(img2, keypoints_2);
	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img1, keypoints_1, descriptors_1);
	descriptor->compute(img2, keypoints_2, descriptors_2);

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> matches;
	matcher->match(descriptors_1, descriptors_2, matches);

	//-- 第四步:匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- 第五步:绘制匹配结果
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img_match);
	drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, img_goodmatch);
	//imshow("所有匹配点对", img_match);
	imshow("优化后匹配点对", img_goodmatch);

	//cout << matches.size() << endl;
	cout << "匹配成功的点的数量：" << good_matches.size() << endl;


	//对图2进行kmeans聚类，找一个最好的类与图1匹配
	Mat labels;
	kmeans(keypoints_2, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),3, KMEANS_RANDOM_CENTERS);

	std::vector<KeyPoint>K1, K2, K3;
	for (int i = 0; i < keypoints_2.size(); i++)
	{
		if (labels.at<int>(i) == 0)
			K1.push_back(keypoints_2[i]);
		if (labels.at<int>(i) == 1)
			K2.push_back(keypoints_2[i]);
		if (labels.at<int>(i) == 2)
			K3.push_back(keypoints_2[i]);
	}
	Mat img_n1;
	drawKeypoints(img2, K1, img_n1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("聚类1", img_n1);
	cout << "第一类的关键点数量为：" << K1.size() << endl;

	Mat img_n2;
	drawKeypoints(img2, K2, img_n2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("聚类2", img_n2);
	cout << "第二类的关键点数量为：" << K2.size() << endl;

	Mat img_n3;
	drawKeypoints(img2, K3, img_n3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("聚类3", img_n3);
	cout << "第三类的关键点数量为：" << K3.size() << endl;

	vector<DMatch> matchesk2;
	Mat descriptors_k2;
	descriptor->compute(img2, K2, descriptors_k2);
	matcher->match(descriptors_1, descriptors_k2, matchesk2);

	vector<DMatch> good_matchesk2;
	for (int i = 0; i < descriptors_k2.rows; i++)
	{
		double dist = matchesk2[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	for (int i = 0; i < descriptors_k2.rows; i++)
	{
		if (matchesk2[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matchesk2.push_back(matchesk2[i]);
		}
	}

	Mat img_match_k2;
	drawMatches(img1, keypoints_1, img_n2, K2, good_matchesk2, img_match_k2);
	imshow("优化后匹配点对222", img_match_k2);
	return 0;
}
