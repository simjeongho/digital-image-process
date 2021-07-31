#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
Mat doDft(Mat srcImg); // discrete fourier transform
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat myNormalize(Mat src);
Mat centralize(Mat complex);
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);
Mat doLPF(Mat srcImg);
Mat doHPF(Mat srcImg);

using namespace cv;
using namespace std;
int main()
{
	Mat einstein = imread("img1_5.jpg", 0);
	imshow("einstein", einstein);

	waitKey(0);
}

Mat doDft(Mat srcImg) // discrete fourier transform
{
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg)
{
	Mat planes[2];
	split(complexImg, planes);
	//�Ǽ���, ����� �и�

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//magnitude ���
	//log(1+sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src)
{
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
} // normalize

Mat getPhase(Mat complexImg)
{
	Mat planes[2];
	split(complexImg, planes);
	//�Ǽ��� , ����� �и�

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);

	return phaImg;
}

Mat myNormalize(Mat src)
{
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
};

Mat centralize(Mat complex)
{
	Mat planes[2];

	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;// �ӽ� ����

	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);
	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;

} // ��ǥ�� �߾� �̵�;

Mat setComplex(Mat magImg, Mat phaImg)
{
	exp(magImg, magImg);

	magImg -= Scalar::all(1);
	//magnitude ����� �ݴ�� ����

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	//�� ��ǥ�� -> ���� ��ǥ��(������ ũ��κ��� 2���� ��ǥ)

	Mat complexImg;
	merge(planes, 2, complexImg);
	//�Ǽ���, ����� ��ü
	return complexImg; // �ٽ� complexImg ���
}

Mat doIdft(Mat complexImg)
{
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT�� �̿��� ���� ���� ���

	Mat planes[2];
	split(idftcvt, planes);
	
	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);

	return dstImg;
}

Mat doLPF(Mat srcImg)
{
	Mat padImg;
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal,&maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}
Mat doBPF(Mat srcImg)
{
	Mat padImg;
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<BPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(1), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0);
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}
Mat doHPF(Mat srcImg)
{
	Mat padImg;
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}