#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;

void exCvMeanShift();
void exMyMeanShift();

void main()
{
	exCvMeanShift();
	exMyMeanShift();
}

void exCvMeanShift()
{
	Mat img = imread("fruit1.jpeg");
	if (img.empty()) exit(-1);
	cout << "------exCvMeanshift() -------" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exCvMeanShift_src.jpg", img);

	pyrMeanShiftFiltering(img, img, 8, 16);

	imshow("DstexcvMeanShift", img);
	//waitKey();
	//destroyAllWindows();
	imwrite("exCvMeanShift_dst.jpg", img);
}

class Point5D
{
	//Mean shift 구현을 위한 전용 포인트 클래스

public:
	float x, y, l, u, v;

	Point5D();
	~Point5D();

	void accumPt(Point5D); //포인트 축적
	void copyPt(Point5D); // 포인트 복사
	float getColorDist(Point5D); // 색상 거리 계산
	float getSpatialDsit(Point5D); // 좌표 거리 계산
	void scalePt(float); // 포인트 스케일링 함수(평균용)
	void setPt(float, float, float, float, float); // 포인트 값 설정 함수
	void printPt(); // 포인트 값 출려함수
};

Point5D::Point5D() {
	x = -1;
	y = -1;
}
Point5D::~Point5D()
{

}

void Point5D::accumPt(Point5D Pt)
{
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	u += Pt.u;
	v += Pt.v;
}

void Point5D::copyPt(Point5D Pt)
{
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	u = Pt.u;
	v = Pt.v;
}

float Point5D::getColorDist(Point5D Pt)
{
	return sqrt(pow(l - Pt.l, 2) +
		pow(u - Pt.u, 2) +
		pow(v - Pt.v, 2));
}

float Point5D::getSpatialDsit(Point5D Pt)
{
	return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}

void Point5D::scalePt(float scale)
{
	x *= scale;
	y *= scale;
	l *= scale;
	u *= scale;
	v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb)
{
	x = px;
	y = py;
	l = pl;
	u = pa;
	v = pb;
}

void Point5D::printPt()
{
	cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

class MeanShift
{
	//Mean shift 클래스

public:
	float bw_spatial = 8;
	float bw_color = 16;
	float min_shift_color = 0.1;
	float min_shift_spatial = 0.1;
	int max_steps = 10;
	vector <Mat> img_split;
	MeanShift(float, float, float, float, int);
	void doFiltering(Mat&);
};

MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms)
{
	//생성자

	bw_spatial = bs;
	bw_color = bc;
	max_steps = ms;
	min_shift_color = msc;
	min_shift_spatial = mss;
}

void MeanShift::doFiltering(Mat& img)
{
	int height = img.rows;
	int width = img.cols;
	split(img, img_split);
	Point5D pt, pt_prev, pt_cur, pt_sum;

	int pad_left, pad_right, pad_top, pad_bottom;
	size_t n_pt, step;

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{

			pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
			pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
			pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
			pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;

			//<현재 픽셀 세팅>

			pt_cur.setPt(row, col, (float)img_split[0].at<uchar>(row, col), (float)img_split[1].at<uchar>(row, col), (float)img_split[2].at<uchar>(row, col));

			//<주변 픽셀 탐색>
			step = 0;
			do {
				pt_prev.copyPt(pt_cur);
				pt_sum.setPt(0, 0, 0, 0, 0);
				n_pt = 0;
				for (int hx = pad_top; hx < pad_bottom; hx++)
				{
					for (int hy = pad_left; hy < pad_right; hy++)
					{
						pt.setPt(hx, hy, (float)img_split[0].at<uchar>(hx, hy), (float)img_split[1].at<uchar>(hx, hy), (float)img_split[2].at<uchar>(hx, hy));

						//<color bandwidth 안에서 축적>
						if (pt.getColorDist(pt_cur) < bw_color)
						{
							pt_sum.accumPt(pt);
							n_pt++;
						}
					}
				}

				// <축적 결과를 기반으로 현재픽셀 갱신>
				pt_sum.scalePt(1.0 / n_pt); // 축적 결과 평균
				pt_cur.copyPt(pt_sum);
				step++;
			}
			while ((pt_cur.getColorDist(pt_prev) > min_shift_color) && (pt_cur.getSpatialDsit(pt_prev) > min_shift_spatial) && (step < max_steps));
			 // 변화량 최소조건을 만족할 때까지 반복
			//최대 반복횟수 조건도 포함

			//<결과 픽셀 갱신>
			img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
		}
	}
}

void exMyMeanShift()
{
	Mat img = imread("fruit1.jpeg");
	if (img.empty()) exit(-1);
	cout << "------exMyMeanshift() -------" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exMyMeanShift_src.jpg", img);
	cvtColor(img, img, CV_RGB2Luv);
	
	MeanShift MSProc(8, 16, 0.1, 0.1, 10);
	MSProc.doFiltering(img);

	cvtColor(img, img, CV_Luv2RGB);

	imshow("Dst1exMyMeanShift", img);
	waitKey();
	destroyAllWindows();
	imwrite("exCvMeanShift_dst.jpg", img);
}
