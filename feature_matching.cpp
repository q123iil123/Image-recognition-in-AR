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

int main()
{

	//string pattern = "D:/C++excercise/Feture/Feture/images/*.png";
	//vector<Mat> images;
	//vector<String> fn;
	//glob(pattern, fn, false);
	//size_t count = fn.size();
	//cout <<"总共"<< count<<"张图像" <<endl<<endl;

	////-- 读取图像
	//Mat img_1 = imread("D:/C++excercise/Feture/Feture/121/2.jpg");	//target

	//int k = 0;
	//for (int i = 0; i < count; i++)
	//{
	//	Mat img_2 = imread(fn[i]);
	//	//Mat img_2 = imread("D:/C++excercise/Feture/Feture/target/x2.jpg");
	//	if (find_feature_matches(img_1, img_2) == 1)
	//	{
	//		k++;
	//		cout << "第" << i + 1 << "张图像" <<"匹配成功！"<< endl;
	//	}
	//}
	//cout << "总共" << k << "张匹配成功" << endl;

	Mat img1 = imread("D:/c++_exercise/Feture/Feture/target/x7.jpg");
	Mat img2 = imread("D:/c++_exercise/Feture/Feture/target/x1.jpg");
    
	find_feature_matches(img1, img2);
	

	waitKey(0);
}



int find_feature_matches(Mat img_1, Mat img_2)
{
	start = clock();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = AKAZE::create();       //用的是AKAZE，想用ORB的话吧AKAZE改成ORB就可以了
	Ptr<DescriptorExtractor> descriptor = AKAZE::create();
	// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//cout << keypoints_2.size() << endl;

	Mat labels;
	kmeans(keypoints_2, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_RANDOM_CENTERS);
	//for(int i=0;i<labels.rows;i++)
		//cout << labels.at<int>(i);
	//cout <<endl<<labels.rows<< endl;

	//当前思路：提取ORB或者AKAZE特征点，即检测角点、根据角点计算描述子（也叫特征向量或特征描述），我们现在就有了关键点和关键点对应的描述子了
	//图1为target模板图片，图2是我们需要匹配的图片
	//我们要对图2中的特征点进行聚类，使用k-means算法，分成三类，分别创建三对不同的关键点和描述子来存储聚类的结果
	//让图1分别和这三个类里面的特征点进行匹配，匹配的结果是否使用有一定的约束
	//1.三个类别至少有一个类别的匹配成功的特征点的数量应该在一个阈值以内 2.若1成功，则取特征点匹配成功数量最多的一个类别的特征点为我们要的特征点

	//std::vector<KeyPoint>K1, K2, K3;
	//Mat dk1, dk2, dk3;
	//detector->detect(img_2, K1);

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB特征点", outimg1);


	Mat outimg12;
	drawKeypoints(img_2, keypoints_2, outimg12, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB特征点2", outimg12);
	//cout << keypoints_2.size() << endl;

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> matches;
	//BFMatcher matcher ( NORM_HAMMING );
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

	// 仅供娱乐的写法
	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		//if (matches[i].distance <= max(2*min_dist, 30.0))
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- 第五步:绘制匹配结果
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	//imshow("所有匹配点对", img_match);
	imshow("优化后匹配点对", img_goodmatch);

	//cout << matches.size() << endl;
	cout << "匹配成功的点的数量：" << good_matches.size() << endl;

	vector<DMatch> m_Matches;
	m_Matches = good_matches;
	int ptCount = good_matches.size();
	if (ptCount < 100)
	{
		cout << "Don't find enough match points for RANSAC" << endl;
		return 0;
	}

	//坐标转换为float类型
	vector <KeyPoint> RAN_KP1, RAN_KP2;
	//size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		RAN_KP1.push_back(keypoints_1[good_matches[i].queryIdx]);
		RAN_KP2.push_back(keypoints_2[good_matches[i].trainIdx]);
		//RAN_KP1是要存储img01中能与img02匹配的点
		//goodMatches存储了这些匹配点对的img01和img02的索引值
	}
	//坐标变换
	vector <Point2f> p01, p02;
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		p01.push_back(RAN_KP1[i].pt);
		p02.push_back(RAN_KP2[i].pt);
	}



	//MatH = findHomography(scene, obj, CV_FM_RANSAC, 3.0, inliersMask, 100);

	vector<uchar> RansacStatus;
	Mat Fundamental = findHomography(p01, p02, RansacStatus, RANSAC,3.0);
	//重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
	vector <KeyPoint> RR_KP1, RR_KP2;
	vector <DMatch> RR_matches;
	int index = 0;
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
	
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
			m_Matches[i].queryIdx = index;
			m_Matches[i].trainIdx = index;
			RR_matches.push_back(m_Matches[i]);
			index++;
		}
	}
	cout << "RANSAC后匹配点数" << RR_matches.size() << endl;
	Mat img_RR_matches;
	drawMatches(img_1, RR_KP1, img_2, RR_KP2, RR_matches, img_RR_matches);
	imshow("After RANSAC", img_RR_matches);

	



	stop = clock();
	duration = (double)(stop - start) / CLK_TCK; //CLK_TCK为clock()函数的时间单位，即时钟打点
												 //cout << "算法耗时：　" << duration << "s" << endl;

	if (good_matches.size() > 67 && good_matches.size() < 118)
	{
		//cout << "找到目标图像！！！！！" << endl << endl;
		return 1;
	}

	else
	{
		//cout << "没有找到目标图像" << endl << endl;
		return -1;
	}
}