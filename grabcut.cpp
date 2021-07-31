#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;
void mygrabCut();

void main()
{
	mygrabCut();

	waitKey(0);
}

void mygrabCut()
{
	Mat img = imread("tiger.jpg");
	Rect rect;
	rect = Rect(Point(3, 27), Point(234, 358));

	Mat result, bg_model, fg_model;
	grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);

	compare(result, GC_PR_FGD, result, CMP_EQ);
	//GC_PR_FGD: GrabCut class foreground ÇÈ¼¿
	// CMP_EQ: compare ¿É¼Ç(equal)

	Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img.copyTo(mask, result);
	imshow("img", img);
	imshow("result", result);
	imshow("mask", mask);
}