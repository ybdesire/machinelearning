// image_pre_process.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	Mat img = imread("lena.bmp");
	int x;
	if(img.empty())
	{
		cout<<"error";
		cin>>x;
		return -1;
	}
	imshow("photo",img);
	waitKey();
	return 0;
}

