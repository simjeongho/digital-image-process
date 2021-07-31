#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

double gaussian2D(float c, float r, double sigma);
void myGaussian(const Mat& src_img, Mat& dst_img, Size size);
void myKernelConv(const Mat& src_img, Mat dst_img, const Mat& kn);
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx();
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void doBilateralEx();
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
float distance(int x, int y, int i, int j);
double gaussian(float x, double sigma);

int main()
{
	Mat saltpepper = imread("salt_peper2.jpg", 1);
	imshow("salt peper",saltpepper);

	waitKey(0);
}

double gaussian2D(float c, float r, double sigma)
{
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}
void myGaussian(const Mat& src_img, Mat& dst_img, Size size)
{
	//Ŀ�� ����
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 1.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++)
	{
		for (int r = 0; r < kn.rows; r++)
		{
			kn_data[r * kn.cols + c] = (float)gaussian2D((float)(c - kn.cols / 2), (float)(r - kn.rows / 2), sigma);
		}
	}
	//Ŀ�� ������� ����
	myKernelConv(src_img, dst_img, kn);
}

void myKernelConv(const Mat& src_img, Mat dst_img, const Mat& kn)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; 
	int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2;
	int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	//�ʼ� �ε���(�����ڸ� ����)
	for (int c = rad_w + 1; c < wd - rad_w; c++)
	{
		for (int r = rad_h + 1; r < hg - rad_h; r++)
		{
			tmp = 0.f;
			sum = 0.f;
			//<Ŀ�� �ε���>
			for (int kc = -rad_w; kc <= rad_w; kc++)
			{
				for (int kr = -rad_h; kr <= rad_h; kr++)
				{
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum; //����ȭ �� overflow ����
			else tmp = abs(tmp);

			if (tmp > 255.f)tmp = 255.f; // overflow ����

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd * khg](); // Ŀ�� ���̺� �����Ҵ�
	float tmp;

	//�ȼ� �ε���(�����ڸ� ����)
	for (int c = rad_w + 1; c < wd - rad_w; c++)
	{
		for (int r = rad_h + 1; r < hg - rad_h; r++)
		{

		}
	}

	delete table; // �����Ҵ� ���� 
}

void doMedianEx()
{
	cout << " ---doMediaEx() --- \n" << endl;

	//<�Է�>
	Mat src_img = imread("salt_pepper.png", 0);
	if (!src_img.data) printf("No image data \n");

	//<Median ���͸� ����>
	Mat dst_img;
#if USE_OPENCV
	medianBlur(src_img, dst_img, 3);
#else
	myMedian(src_img, dst_img, Size(3, 3));
#endif

	//���
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEx()", result_img);
	waitKey(0);
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;

	//�ȼ� �ε���(�����ڸ� ����)>
	for (int c = radius + 1; c < hg - radius; c++)
	{
		for (int r = radius + 1; r < wh - radius; r++)
		{
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			//ȭ�Һ� Bilateral ��� ����
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1); // Mat type ��ȯ
}

void doBilateralEx()
{
	cout << " --- do BilateralEx()  --- \n" << endl;

	// < �Է� >
	Mat src_img = imread("rock.png", 0);
	Mat dst_img;
	if (!src_img.data) printf("No image data \n");

	// < Bilateral ���͸� ����>
	myBilateral(src_img, dst_img, 5, 25.0, 50.0);

	//<���>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx()", result_img);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s)
{
	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;

	//<Ŀ�� �ε���>
	for (int kc = -radius; kc <= radius; kc++)
	{
		for (int kr = -radius; kr <= radius; kr++)
		{
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			//range calc
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			//spatial calc
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; // ����ȭ
}

double gaussian(float x, double sigma)
{
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI + pow(sigma, 2));
}
//double gaussian2D(float c, float r, double sigma)
//{
//	return exp(-(pow(c, 2) + pow(r, 2) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
//}
float distance(int x, int y, int i, int j)
{
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

