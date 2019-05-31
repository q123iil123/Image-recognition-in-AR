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
	//cout <<"�ܹ�"<< count<<"��ͼ��" <<endl<<endl;

	////-- ��ȡͼ��
	//Mat img_1 = imread("D:/C++excercise/Feture/Feture/121/2.jpg");	//target

	//int k = 0;
	//for (int i = 0; i < count; i++)
	//{
	//	Mat img_2 = imread(fn[i]);
	//	//Mat img_2 = imread("D:/C++excercise/Feture/Feture/target/x2.jpg");
	//	if (find_feature_matches(img_1, img_2) == 1)
	//	{
	//		k++;
	//		cout << "��" << i + 1 << "��ͼ��" <<"ƥ��ɹ���"<< endl;
	//	}
	//}
	//cout << "�ܹ�" << k << "��ƥ��ɹ�" << endl;

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
	Ptr<FeatureDetector> detector = AKAZE::create();       //�õ���AKAZE������ORB�Ļ���AKAZE�ĳ�ORB�Ϳ�����
	Ptr<DescriptorExtractor> descriptor = AKAZE::create();
	// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//-- ��һ��:��� Oriented FAST �ǵ�λ��
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//cout << keypoints_2.size() << endl;

	Mat labels;
	kmeans(keypoints_2, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_RANDOM_CENTERS);
	//for(int i=0;i<labels.rows;i++)
		//cout << labels.at<int>(i);
	//cout <<endl<<labels.rows<< endl;

	//��ǰ˼·����ȡORB����AKAZE�����㣬�����ǵ㡢���ݽǵ���������ӣ�Ҳ�������������������������������ھ����˹ؼ���͹ؼ����Ӧ����������
	//ͼ1Ϊtargetģ��ͼƬ��ͼ2��������Ҫƥ���ͼƬ
	//����Ҫ��ͼ2�е���������о��࣬ʹ��k-means�㷨���ֳ����࣬�ֱ𴴽����Բ�ͬ�Ĺؼ�������������洢����Ľ��
	//��ͼ1�ֱ������������������������ƥ�䣬ƥ��Ľ���Ƿ�ʹ����һ����Լ��
	//1.�������������һ������ƥ��ɹ��������������Ӧ����һ����ֵ���� 2.��1�ɹ�����ȡ������ƥ��ɹ���������һ������������Ϊ����Ҫ��������

	//std::vector<KeyPoint>K1, K2, K3;
	//Mat dk1, dk2, dk3;
	//detector->detect(img_2, K1);

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB������", outimg1);


	Mat outimg12;
	drawKeypoints(img_2, keypoints_2, outimg12, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB������2", outimg12);
	//cout << keypoints_2.size() << endl;

	//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
	vector<DMatch> matches;
	//BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, matches);

	//-- ���Ĳ�:ƥ����ɸѡ
	double min_dist = 10000, max_dist = 0;

	//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	// �������ֵ�д��
	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		//if (matches[i].distance <= max(2*min_dist, 30.0))
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- ���岽:����ƥ����
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	//imshow("����ƥ����", img_match);
	imshow("�Ż���ƥ����", img_goodmatch);

	//cout << matches.size() << endl;
	cout << "ƥ��ɹ��ĵ��������" << good_matches.size() << endl;

	vector<DMatch> m_Matches;
	m_Matches = good_matches;
	int ptCount = good_matches.size();
	if (ptCount < 100)
	{
		cout << "Don't find enough match points for RANSAC" << endl;
		return 0;
	}

	//����ת��Ϊfloat����
	vector <KeyPoint> RAN_KP1, RAN_KP2;
	//size_t�Ǳ�׼C���ж���ģ�ӦΪunsigned int����64λϵͳ��Ϊlong unsigned int,��C++��Ϊ����Ӧ��ͬ��ƽ̨�����ӿ���ֲ�ԡ�
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		RAN_KP1.push_back(keypoints_1[good_matches[i].queryIdx]);
		RAN_KP2.push_back(keypoints_2[good_matches[i].trainIdx]);
		//RAN_KP1��Ҫ�洢img01������img02ƥ��ĵ�
		//goodMatches�洢����Щƥ���Ե�img01��img02������ֵ
	}
	//����任
	vector <Point2f> p01, p02;
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		p01.push_back(RAN_KP1[i].pt);
		p02.push_back(RAN_KP2[i].pt);
	}



	//MatH = findHomography(scene, obj, CV_FM_RANSAC, 3.0, inliersMask, 100);

	vector<uchar> RansacStatus;
	Mat Fundamental = findHomography(p01, p02, RansacStatus, RANSAC,3.0);
	//���¶���ؼ���RR_KP��RR_matches���洢�µĹؼ���ͻ�������ͨ��RansacStatus��ɾ����ƥ���
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
	cout << "RANSAC��ƥ�����" << RR_matches.size() << endl;
	Mat img_RR_matches;
	drawMatches(img_1, RR_KP1, img_2, RR_KP2, RR_matches, img_RR_matches);
	imshow("After RANSAC", img_RR_matches);

	



	stop = clock();
	duration = (double)(stop - start) / CLK_TCK; //CLK_TCKΪclock()������ʱ�䵥λ����ʱ�Ӵ��
												 //cout << "�㷨��ʱ����" << duration << "s" << endl;

	if (good_matches.size() > 67 && good_matches.size() < 118)
	{
		//cout << "�ҵ�Ŀ��ͼ�񣡣�������" << endl << endl;
		return 1;
	}

	else
	{
		//cout << "û���ҵ�Ŀ��ͼ��" << endl << endl;
		return -1;
	}
}