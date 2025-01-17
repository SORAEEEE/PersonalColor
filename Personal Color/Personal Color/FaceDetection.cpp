#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

#define PI 3.14159265

using namespace std;
using namespace cv;


void setFaceLab();
void setFaceHSV();
void setLipLab();
void setLipHSV();
void setPalletteLab();
void setPalletteHSV();
void sampleExtraction(Mat frame);
void rgbToLab();
void rgbToHSV();
void classification();
Point3d binarySplit(Point3d sample[]);
double getDistanceLab(Point3d a, Point3d b);
double getDistanceHSV(Point3d a, Point3d b);
int findSkin(Point3d a[]);
void findLip(Point3d a[], int personalColor);
int findMinIdx(double arr[], int len);
int findMaxIdx(int arr[], int len);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String nose_cascade_name = "Nariz.xml";
String img_name = "크리스탈2.png";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;

Mat sample[5]; //forehead, temple1, temple2, cheek1, cheek2
static Point3d forehead[100];
static Point3d temple1[100];
static Point3d temple2[100];
static Point3d cheek1[100];
static Point3d cheek2[100];

static Point3d skinColor[4][4];
static Point3d lipColor[4][6];
static Point3d palletteColor[16][8];

int main(int argc, const char** argv) {
	Mat img = imread(img_name);
	Point3d mainVal[5]; //sample main value
	int personalColor;

	/*
	if (img.data == NULL) {
		printf("이미지 열기 실패");
		return -1;
	}

	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error Face cascade loading\n"); return -1;};
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error Eyes cascade loading\n"); return -1;};
	if (!nose_cascade.load(nose_cascade_name)) { printf("--(!)Error Nose cascade loading\n"); return -1;};
	*/
	
	//setFaceHSV(); //personal color setting
	//setLipHSV();
	//setPalletteHSV();
	//sampleExtraction(img); //Sample Extraction
	//rgbToHSV(); //RGB to HSV
	

	
	//setFaceLab(); 
	setLipLab();
	setPalletteLab();
	//sampleExtraction(img); 
	//rgbToLab(); 
	
	classification();

	/* Select main value */
	//mainVal[0] = binarySplit(forehead);
	//mainVal[1] = binarySplit(temple1);
	//mainVal[2] = binarySplit(temple2);
	//mainVal[3] = binarySplit(cheek1);
	//mainVal[4] = binarySplit(cheek2);
	
	/* Find personal color */
	//personalColor = findSkin(mainVal);
	//findLip(mainVal, personalColor);
	
	waitKey(0);
	return 0;
}

void setFaceLab () {
	Mat temp(4, 4, CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(168, 211, 251); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(149, 202, 255);
	temp.at<Vec3b>(0, 2) = Vec3b(161, 197, 253);
	temp.at<Vec3b>(0, 3) = Vec3b(130, 204, 252);
	temp.at<Vec3b>(1, 0) = Vec3b(174, 231, 253); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(192, 219, 255);
	temp.at<Vec3b>(1, 2) = Vec3b(170, 217, 254);
	temp.at<Vec3b>(1, 3) = Vec3b(122, 210, 254);
	temp.at<Vec3b>(2, 0) = Vec3b(150, 221, 255); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(152, 206, 247);
	temp.at<Vec3b>(2, 2) = Vec3b(128, 201, 249);
	temp.at<Vec3b>(2, 3) = Vec3b(101, 169, 212);
	temp.at<Vec3b>(3, 0) = Vec3b(147, 220, 255); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(148, 206, 242);
	temp.at<Vec3b>(3, 2) = Vec3b(121, 207, 247);
	temp.at<Vec3b>(3, 3) = Vec3b(102, 173, 216);

	Mat tempLab;
	temp.convertTo(tempLab, CV_32FC3, (double)1.f / 255.f);

	cvtColor(tempLab, tempLab, CV_BGR2Lab);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			skinColor[i][j].x = tempLab.at<Vec3f>(i, j)[0];
			skinColor[i][j].y = tempLab.at<Vec3f>(i, j)[1];
			skinColor[i][j].z = tempLab .at<Vec3f>(i, j)[2];
		}
	}

	//imshow("personal color", temp);
}

void setFaceHSV() {
	Mat temp(4,4,CV_8UC3);
	
	temp.at<Vec3b>(0, 0) = Vec3b(168, 211, 251); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(149, 202, 255);
	temp.at<Vec3b>(0, 2) = Vec3b(161, 197, 253);
	temp.at<Vec3b>(0, 3) = Vec3b(130, 204, 252);
	temp.at<Vec3b>(1, 0) = Vec3b(174, 231, 253); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(192, 219, 255);
	temp.at<Vec3b>(1, 2) = Vec3b(170, 217, 254);
	temp.at<Vec3b>(1, 3) = Vec3b(122, 210, 254);
	temp.at<Vec3b>(2, 0) = Vec3b(150, 221, 255); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(152, 206, 247);
	temp.at<Vec3b>(2, 2) = Vec3b(128, 201, 249);
	temp.at<Vec3b>(2, 3) = Vec3b(101, 169, 212);
	temp.at<Vec3b>(3, 0) = Vec3b(147, 220, 255); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(148, 206, 242);
	temp.at<Vec3b>(3, 2) = Vec3b(121, 207, 247);
	temp.at<Vec3b>(3, 3) = Vec3b(102, 173, 216);

	Mat tempHSV;
	temp.convertTo(tempHSV, CV_32FC3, (double)1.f/255.f);

	cvtColor(tempHSV, tempHSV, CV_BGR2HSV);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			skinColor[i][j].x = tempHSV.at<Vec3f>(i, j)[0];
			skinColor[i][j].y = 100*tempHSV.at<Vec3f>(i, j)[1];
			skinColor[i][j].z = 100*tempHSV.at<Vec3f>(i, j)[2];
		}
	}
	
	//imshow("personal color", temp);
}

void setLipLab() {
	Mat temp(4, 6, CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(79, 79, 234); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(73, 29, 195);
	temp.at<Vec3b>(0, 2) = Vec3b(130, 125, 254);
	temp.at<Vec3b>(0, 3) = Vec3b(111, 127, 247);
	temp.at<Vec3b>(0, 4) = Vec3b(47, 43, 223);
	temp.at<Vec3b>(0, 5) = Vec3b(84, 110, 250);
	temp.at<Vec3b>(1, 0) = Vec3b(147, 63, 243); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(73, 29, 195);
	temp.at<Vec3b>(1, 2) = Vec3b(120, 39, 170);
	temp.at<Vec3b>(1, 3) = Vec3b(43, 18, 203);
	temp.at<Vec3b>(1, 4) = Vec3b(144, 116, 232);
	temp.at<Vec3b>(1, 5) = Vec3b(106, 50, 228);
	temp.at<Vec3b>(2, 0) = Vec3b(104, 107, 227); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(41, 48, 175);
	temp.at<Vec3b>(2, 2) = Vec3b(45, 32, 186);
	temp.at<Vec3b>(2, 3) = Vec3b(43, 18, 203);
	temp.at<Vec3b>(2, 4) = Vec3b(47, 43, 223);
	temp.at<Vec3b>(2, 5) = Vec3b(34, 45, 184);
	temp.at<Vec3b>(3, 0) = Vec3b(63, 26, 160); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(103, 37, 207);
	temp.at<Vec3b>(3, 2) = Vec3b(120, 39, 170);
	temp.at<Vec3b>(3, 3) = Vec3b(81, 33, 198);
	temp.at<Vec3b>(3, 4) = Vec3b(40, 5, 164);
	temp.at<Vec3b>(3, 5) = Vec3b(19, 2, 208);

	Mat tempLab;
	temp.convertTo(tempLab, CV_32FC3, (double)1.f / 255.f);

	cvtColor(tempLab, tempLab, CV_BGR2Lab);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++) {
			lipColor[i][j].x = tempLab.at<Vec3f>(i, j)[0];
			lipColor[i][j].y = tempLab.at<Vec3f>(i, j)[1];
			lipColor[i][j].z = tempLab.at<Vec3f>(i, j)[2];
		}
	}

	//imshow("LIP", tempLab);
}

void setLipHSV() {
	Mat temp(4, 6, CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(79, 79, 234); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(73, 29, 195);
	temp.at<Vec3b>(0, 2) = Vec3b(130, 125, 254);
	temp.at<Vec3b>(0, 3) = Vec3b(111, 127, 247);
	temp.at<Vec3b>(0, 4) = Vec3b(47, 43, 223);
	temp.at<Vec3b>(0, 5) = Vec3b(84, 110, 250);
	temp.at<Vec3b>(1, 0) = Vec3b(147, 63, 243); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(73, 29, 195);
	temp.at<Vec3b>(1, 2) = Vec3b(120, 39, 170);
	temp.at<Vec3b>(1, 3) = Vec3b(43, 18, 203);
	temp.at<Vec3b>(1, 4) = Vec3b(144, 116, 232);
	temp.at<Vec3b>(1, 5) = Vec3b(106, 50, 228);
	temp.at<Vec3b>(2, 0) = Vec3b(104, 107, 227); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(41, 48, 175);
	temp.at<Vec3b>(2, 2) = Vec3b(45, 32, 186);
	temp.at<Vec3b>(2, 3) = Vec3b(43, 18, 203);
	temp.at<Vec3b>(2, 4) = Vec3b(47, 43, 223);
	temp.at<Vec3b>(2, 5) = Vec3b(34, 45, 184);
	temp.at<Vec3b>(3, 0) = Vec3b(63, 26, 160); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(103, 37, 207);
	temp.at<Vec3b>(3, 2) = Vec3b(120, 39, 170);
	temp.at<Vec3b>(3, 3) = Vec3b(81, 33, 198);
	temp.at<Vec3b>(3, 4) = Vec3b(40, 5, 164);
	temp.at<Vec3b>(3, 5) = Vec3b(19, 2, 208);

	Mat tempHSV;
	temp.convertTo(tempHSV, CV_32FC3, (double)1.f / 255.f);

	cvtColor(tempHSV, tempHSV, CV_BGR2HSV);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++) {
			lipColor[i][j].x = tempHSV.at<Vec3f>(i, j)[0];
			tempHSV.at<Vec3f>(i, j)[1] *= 100;
			tempHSV.at<Vec3f>(i, j)[2] *= 100;
			lipColor[i][j].y = tempHSV.at<Vec3f>(i, j)[1];
			lipColor[i][j].z = tempHSV.at<Vec3f>(i, j)[2];
		}
	}

	//imshow("LIP", tempHSV);
}

void setPalletteLab() {
	Mat temp(16, 8, CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(45, 103, 182); 	temp.at<Vec3b>(0, 1) = Vec3b(117, 162, 198);	temp.at<Vec3b>(0, 2) = Vec3b(159, 205, 234);	temp.at<Vec3b>(0, 3) = Vec3b(1210, 253, 254);	temp.at<Vec3b>(0, 4) = Vec3b(141, 224, 242);	temp.at<Vec3b>(0, 5) = Vec3b(65, 182, 215);		temp.at<Vec3b>(0, 6) = Vec3b(41, 184, 237);		temp.at<Vec3b>(0, 7) = Vec3b(56, 223, 250);
	temp.at<Vec3b>(1, 0) = Vec3b(217, 134, 14);		temp.at<Vec3b>(1, 1) = Vec3b(226, 228, 62);		temp.at<Vec3b>(1, 2) = Vec3b(208, 213, 84);		temp.at<Vec3b>(1, 3) = Vec3b(185, 185, 0);		temp.at<Vec3b>(1, 4) = Vec3b(27, 210, 74);		temp.at<Vec3b>(1, 5) = Vec3b(60, 210, 163);		temp.at<Vec3b>(1, 6) = Vec3b(112, 235, 225);	temp.at<Vec3b>(1, 7) = Vec3b(35, 240, 252);
	temp.at<Vec3b>(2, 0) = Vec3b(138, 0, 2); 		temp.at<Vec3b>(2, 1) = Vec3b(197, 59, 57);		temp.at<Vec3b>(2, 2) = Vec3b(207, 113, 85);		temp.at<Vec3b>(2, 3) = Vec3b(203, 157, 193);	temp.at<Vec3b>(2, 4) = Vec3b(228, 127, 219);	temp.at<Vec3b>(2, 5) = Vec3b(152, 34, 168);		temp.at<Vec3b>(2, 6) = Vec3b(108, 0, 254);		temp.at<Vec3b>(2, 7) = Vec3b(129, 140, 247);
	temp.at<Vec3b>(3, 0) = Vec3b(29, 123, 248); 	temp.at<Vec3b>(3, 1) = Vec3b(170, 207, 254);	temp.at<Vec3b>(3, 2) = Vec3b(184, 211, 251);	temp.at<Vec3b>(3, 3) = Vec3b(6, 59, 247);		temp.at<Vec3b>(3, 4) = Vec3b(11, 138, 253);		temp.at<Vec3b>(3, 5) = Vec3b(88, 89, 252);		temp.at<Vec3b>(3, 6) = Vec3b(70, 0, 243);		temp.at<Vec3b>(3, 7) = Vec3b(215, 219, 251);

	temp.at<Vec3b>(4, 0) = Vec3b(151, 168, 183);	temp.at<Vec3b>(4, 1) = Vec3b(222, 222, 222);	temp.at<Vec3b>(4, 2) = Vec3b(198, 198, 198);	temp.at<Vec3b>(4, 3) = Vec3b(187, 178, 169);	temp.at<Vec3b>(4, 4) = Vec3b(78, 61, 44);		temp.at<Vec3b>(4, 5) = Vec3b(80, 25, 25);		temp.at<Vec3b>(4, 6) = Vec3b(180, 132, 113);	temp.at<Vec3b>(4, 7) = Vec3b(233, 215, 186);
	temp.at<Vec3b>(5, 0) = Vec3b(219, 147, 191);	temp.at<Vec3b>(5, 1) = Vec3b(170, 68, 122);		temp.at<Vec3b>(5, 2) = Vec3b(252, 61, 164);		temp.at<Vec3b>(5, 3) = Vec3b(223, 154, 158);	temp.at<Vec3b>(5, 4) = Vec3b(208, 113, 85);		temp.at<Vec3b>(5, 5) = Vec3b(209, 12, 12);		temp.at<Vec3b>(5, 6) = Vec3b(211, 126, 0);		temp.at<Vec3b>(5, 7) = Vec3b(236, 211, 145);
	temp.at<Vec3b>(6, 0) = Vec3b(254, 182, 227);	temp.at<Vec3b>(6, 1) = Vec3b(217, 123, 222);	temp.at<Vec3b>(6, 2) = Vec3b(143, 80, 151);		temp.at<Vec3b>(6, 3) = Vec3b(169, 80, 213);		temp.at<Vec3b>(6, 4) = Vec3b(182, 113, 255);	temp.at<Vec3b>(6, 5) = Vec3b(130, 49, 234);		temp.at<Vec3b>(6, 6) = Vec3b(105, 16, 230);		temp.at<Vec3b>(6, 7) = Vec3b(185, 163, 250);
	temp.at<Vec3b>(7, 0) = Vec3b(139, 138, 0);		temp.at<Vec3b>(7, 1) = Vec3b(147, 149, 5);		temp.at<Vec3b>(7, 2) = Vec3b(152, 187, 6);		temp.at<Vec3b>(7, 3) = Vec3b(195, 229, 167);	temp.at<Vec3b>(7, 4) = Vec3b(210, 251, 210);	temp.at<Vec3b>(7, 5) = Vec3b(152, 249, 251);	temp.at<Vec3b>(7, 6) = Vec3b(44, 0, 231);		temp.at<Vec3b>(7, 7) = Vec3b(76, 35, 253);

	temp.at<Vec3b>(8, 0) = Vec3b(212, 252, 255);	temp.at<Vec3b>(8, 1) = Vec3b(187, 233, 253);	temp.at<Vec3b>(8, 2) = Vec3b(172, 208, 255);	temp.at<Vec3b>(8, 3) = Vec3b(90, 137, 255);		temp.at<Vec3b>(8, 4) = Vec3b(115, 150, 254);	temp.at<Vec3b>(8, 5) = Vec3b(37, 73, 248);		temp.at<Vec3b>(8, 6) = Vec3b(12, 39, 215);		temp.at<Vec3b>(8, 7) = Vec3b(29, 64, 194);
	temp.at<Vec3b>(9, 0) = Vec3b(28, 112, 161);		temp.at<Vec3b>(9, 1) = Vec3b(67, 182, 217);		temp.at<Vec3b>(9, 2) = Vec3b(44, 205, 252);		temp.at<Vec3b>(9, 3) = Vec3b(17, 185, 246);		temp.at<Vec3b>(9, 4) = Vec3b(5, 181, 251);		temp.at<Vec3b>(9, 5) = Vec3b(31, 128, 255);		temp.at<Vec3b>(9, 6) = Vec3b(74, 101, 225);		temp.at<Vec3b>(9, 7) = Vec3b(19, 75, 190);
	temp.at<Vec3b>(10, 0) = Vec3b(60, 138, 210);	temp.at<Vec3b>(10, 1) = Vec3b(69, 114, 175);	temp.at<Vec3b>(10, 2) = Vec3b(40, 78, 112);		temp.at<Vec3b>(10, 3) = Vec3b(118, 163, 198);	temp.at<Vec3b>(10, 4) = Vec3b(159, 205, 234);	temp.at<Vec3b>(10, 5) = Vec3b(153, 183, 201);	temp.at<Vec3b>(10, 6) = Vec3b(77, 101, 77);		temp.at<Vec3b>(10, 7) = Vec3b(68, 116, 100);
	temp.at<Vec3b>(11, 0) = Vec3b(92, 75, 108);		temp.at<Vec3b>(11, 1) = Vec3b(207, 113, 85);	temp.at<Vec3b>(11, 2) = Vec3b(139, 138, 0);		temp.at<Vec3b>(11, 3) = Vec3b(181, 202, 29);	temp.at<Vec3b>(11, 4) = Vec3b(91, 137, 2);		temp.at<Vec3b>(11, 5) = Vec3b(41, 79, 2);		temp.at<Vec3b>(11, 6) = Vec3b(167, 205, 168);	temp.at<Vec3b>(11, 7) = Vec3b(106, 236, 174);

	temp.at<Vec3b>(12, 0) = Vec3b(225, 229, 230);	temp.at<Vec3b>(12, 1) = Vec3b(215, 215, 215);	temp.at<Vec3b>(12, 2) = Vec3b(196, 196, 196);	temp.at<Vec3b>(12, 3) = Vec3b(91, 80, 65);		temp.at<Vec3b>(12, 4) = Vec3b(0, 0, 0);			temp.at<Vec3b>(12, 5) = Vec3b(112, 44, 0);		temp.at<Vec3b>(12, 6) = Vec3b(137, 2, 2);		temp.at<Vec3b>(12, 7) = Vec3b(197, 60, 57);
	temp.at<Vec3b>(13, 0) = Vec3b(3, 137, 2);		temp.at<Vec3b>(13, 1) = Vec3b(84, 131, 2);		temp.at<Vec3b>(13, 2) = Vec3b(139, 138, 0);		temp.at<Vec3b>(13, 3) = Vec3b(202, 189, 74);	temp.at<Vec3b>(13, 4) = Vec3b(174, 152, 0);		temp.at<Vec3b>(13, 5) = Vec3b(243, 226, 194);	temp.at<Vec3b>(13, 6) = Vec3b(211, 126, 0);		temp.at<Vec3b>(13, 7) = Vec3b(207, 113, 85);
	temp.at<Vec3b>(14, 0) = Vec3b(105, 194, 30);	temp.at<Vec3b>(14, 1) = Vec3b(186, 249, 251);	temp.at<Vec3b>(14, 2) = Vec3b(0, 248, 255);		temp.at<Vec3b>(14, 3) = Vec3b(0, 0, 254);		temp.at<Vec3b>(14, 4) = Vec3b(44, 0, 231);		temp.at<Vec3b>(14, 5) = Vec3b(39, 2, 138);		temp.at<Vec3b>(14, 6) = Vec3b(105, 16, 231);	temp.at<Vec3b>(14, 7) = Vec3b(112, 39, 247);
	temp.at<Vec3b>(15, 0) = Vec3b(177, 93, 130);	temp.at<Vec3b>(15, 1) = Vec3b(248, 207, 231);	temp.at<Vec3b>(15, 2) = Vec3b(225, 58, 211);	temp.at<Vec3b>(15, 3) = Vec3b(254, 0, 255);		temp.at<Vec3b>(15, 4) = Vec3b(199, 21, 252);	temp.at<Vec3b>(15, 5) = Vec3b(208, 113, 254);	temp.at<Vec3b>(15, 6) = Vec3b(153, 2, 254);		temp.at<Vec3b>(15, 7) = Vec3b(235, 222, 251);

	Mat tempLab;
	temp.convertTo(tempLab, CV_32FC3, (double)1.f / 255.f);

	cvtColor(tempLab, tempLab, CV_BGR2Lab);

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 8; j++) {
			palletteColor[i][j].x = tempLab.at<Vec3f>(i, j)[0];
			palletteColor[i][j].y = tempLab.at<Vec3f>(i, j)[1];
			palletteColor[i][j].z = tempLab.at<Vec3f>(i, j)[2];
		}
	}

 	//imshow("Pallette", tempLab);
}

void setPalletteHSV() {
	Mat temp(16, 8, CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(45, 103, 182); 	temp.at<Vec3b>(0, 1) = Vec3b(117, 162, 198);	temp.at<Vec3b>(0, 2) = Vec3b(159, 205, 234);	temp.at<Vec3b>(0, 3) = Vec3b(1210, 253, 254);	temp.at<Vec3b>(0, 4) = Vec3b(141, 224, 242);	temp.at<Vec3b>(0, 5) = Vec3b(65, 182, 215);		temp.at<Vec3b>(0, 6) = Vec3b(41, 184, 237);		temp.at<Vec3b>(0, 7) = Vec3b(56, 223, 250);
	temp.at<Vec3b>(1, 0) = Vec3b(217, 134, 14);		temp.at<Vec3b>(1, 1) = Vec3b(226, 228, 62);		temp.at<Vec3b>(1, 2) = Vec3b(208, 213, 84);		temp.at<Vec3b>(1, 3) = Vec3b(185, 185, 0);		temp.at<Vec3b>(1, 4) = Vec3b(27, 210, 74);		temp.at<Vec3b>(1, 5) = Vec3b(60, 210, 163);		temp.at<Vec3b>(1, 6) = Vec3b(112, 235, 225);	temp.at<Vec3b>(1, 7) = Vec3b(35, 240, 252);
	temp.at<Vec3b>(2, 0) = Vec3b(138, 0, 2); 		temp.at<Vec3b>(2, 1) = Vec3b(197, 59, 57);		temp.at<Vec3b>(2, 2) = Vec3b(207, 113, 85);		temp.at<Vec3b>(2, 3) = Vec3b(203, 157, 193);	temp.at<Vec3b>(2, 4) = Vec3b(228, 127, 219);	temp.at<Vec3b>(2, 5) = Vec3b(152, 34, 168);		temp.at<Vec3b>(2, 6) = Vec3b(108, 0, 254);		temp.at<Vec3b>(2, 7) = Vec3b(129, 140, 247);
	temp.at<Vec3b>(3, 0) = Vec3b(29, 123, 248); 	temp.at<Vec3b>(3, 1) = Vec3b(170, 207, 254);	temp.at<Vec3b>(3, 2) = Vec3b(184, 211, 251);	temp.at<Vec3b>(3, 3) = Vec3b(6, 59, 247);		temp.at<Vec3b>(3, 4) = Vec3b(11, 138, 253);		temp.at<Vec3b>(3, 5) = Vec3b(88, 89, 252);		temp.at<Vec3b>(3, 6) = Vec3b(70, 0, 243);		temp.at<Vec3b>(3, 7) = Vec3b(215, 219, 251);

	temp.at<Vec3b>(4, 0) = Vec3b(151, 168, 183);	temp.at<Vec3b>(4, 1) = Vec3b(222, 222, 222);	temp.at<Vec3b>(4, 2) = Vec3b(198, 198, 198);	temp.at<Vec3b>(4, 3) = Vec3b(187, 178, 169);	temp.at<Vec3b>(4, 4) = Vec3b(78, 61, 44);		temp.at<Vec3b>(4, 5) = Vec3b(80, 25, 25);		temp.at<Vec3b>(4, 6) = Vec3b(180, 132, 113);	temp.at<Vec3b>(4, 7) = Vec3b(233, 215, 186);
	temp.at<Vec3b>(5, 0) = Vec3b(219, 147, 191);	temp.at<Vec3b>(5, 1) = Vec3b(170, 68, 122);		temp.at<Vec3b>(5, 2) = Vec3b(252, 61, 164);		temp.at<Vec3b>(5, 3) = Vec3b(223, 154, 158);	temp.at<Vec3b>(5, 4) = Vec3b(208, 113, 85);		temp.at<Vec3b>(5, 5) = Vec3b(209, 12, 12);		temp.at<Vec3b>(5, 6) = Vec3b(211, 126, 0);		temp.at<Vec3b>(5, 7) = Vec3b(236, 211, 145);
	temp.at<Vec3b>(6, 0) = Vec3b(254, 182, 227);	temp.at<Vec3b>(6, 1) = Vec3b(217, 123, 222);	temp.at<Vec3b>(6, 2) = Vec3b(143, 80, 151);		temp.at<Vec3b>(6, 3) = Vec3b(169, 80, 213);		temp.at<Vec3b>(6, 4) = Vec3b(182, 113, 255);	temp.at<Vec3b>(6, 5) = Vec3b(130, 49, 234);		temp.at<Vec3b>(6, 6) = Vec3b(105, 16, 230);		temp.at<Vec3b>(6, 7) = Vec3b(185, 163, 250);
	temp.at<Vec3b>(7, 0) = Vec3b(139, 138, 0);		temp.at<Vec3b>(7, 1) = Vec3b(147, 149, 5);		temp.at<Vec3b>(7, 2) = Vec3b(152, 187, 6);		temp.at<Vec3b>(7, 3) = Vec3b(195, 229, 167);	temp.at<Vec3b>(7, 4) = Vec3b(210, 251, 210);	temp.at<Vec3b>(7, 5) = Vec3b(152, 249, 251);	temp.at<Vec3b>(7, 6) = Vec3b(44, 0, 231);		temp.at<Vec3b>(7, 7) = Vec3b(76, 35, 253);

	temp.at<Vec3b>(8, 0) = Vec3b(212, 252, 255);	temp.at<Vec3b>(8, 1) = Vec3b(187, 233, 253);	temp.at<Vec3b>(8, 2) = Vec3b(172, 208, 255);	temp.at<Vec3b>(8, 3) = Vec3b(90, 137, 255);		temp.at<Vec3b>(8, 4) = Vec3b(115, 150, 254);	temp.at<Vec3b>(8, 5) = Vec3b(37, 73, 248);		temp.at<Vec3b>(8, 6) = Vec3b(12, 39, 215);		temp.at<Vec3b>(8, 7) = Vec3b(29, 64, 194);
	temp.at<Vec3b>(9, 0) = Vec3b(28, 112, 161);		temp.at<Vec3b>(9, 1) = Vec3b(67, 182, 217);		temp.at<Vec3b>(9, 2) = Vec3b(44, 205, 252);		temp.at<Vec3b>(9, 3) = Vec3b(17, 185, 246);		temp.at<Vec3b>(9, 4) = Vec3b(5, 181, 251);		temp.at<Vec3b>(9, 5) = Vec3b(31, 128, 255);		temp.at<Vec3b>(9, 6) = Vec3b(74, 101, 225);		temp.at<Vec3b>(9, 7) = Vec3b(19, 75, 190);
	temp.at<Vec3b>(10, 0) = Vec3b(60, 138, 210);	temp.at<Vec3b>(10, 1) = Vec3b(69, 114, 175);	temp.at<Vec3b>(10, 2) = Vec3b(40, 78, 112);		temp.at<Vec3b>(10, 3) = Vec3b(118, 163, 198);	temp.at<Vec3b>(10, 4) = Vec3b(159, 205, 234);	temp.at<Vec3b>(10, 5) = Vec3b(153, 183, 201);	temp.at<Vec3b>(10, 6) = Vec3b(77, 101, 77);		temp.at<Vec3b>(10, 7) = Vec3b(68, 116, 100);
	temp.at<Vec3b>(11, 0) = Vec3b(92, 75, 108);		temp.at<Vec3b>(11, 1) = Vec3b(207, 113, 85);	temp.at<Vec3b>(11, 2) = Vec3b(139, 138, 0);		temp.at<Vec3b>(11, 3) = Vec3b(181, 202, 29);	temp.at<Vec3b>(11, 4) = Vec3b(91, 137, 2);		temp.at<Vec3b>(11, 5) = Vec3b(41, 79, 2);		temp.at<Vec3b>(11, 6) = Vec3b(167, 205, 168);	temp.at<Vec3b>(11, 7) = Vec3b(106, 236, 174);

	temp.at<Vec3b>(12, 0) = Vec3b(225, 229, 230);	temp.at<Vec3b>(12, 1) = Vec3b(215, 215, 215);	temp.at<Vec3b>(12, 2) = Vec3b(196, 196, 196);	temp.at<Vec3b>(12, 3) = Vec3b(91, 80, 65);		temp.at<Vec3b>(12, 4) = Vec3b(0, 0, 0);			temp.at<Vec3b>(12, 5) = Vec3b(112, 44, 0);		temp.at<Vec3b>(12, 6) = Vec3b(137, 2, 2);		temp.at<Vec3b>(12, 7) = Vec3b(197, 60, 57);
	temp.at<Vec3b>(13, 0) = Vec3b(3, 137, 2);		temp.at<Vec3b>(13, 1) = Vec3b(84, 131, 2);		temp.at<Vec3b>(13, 2) = Vec3b(139, 138, 0);		temp.at<Vec3b>(13, 3) = Vec3b(202, 189, 74);	temp.at<Vec3b>(13, 4) = Vec3b(174, 152, 0);		temp.at<Vec3b>(13, 5) = Vec3b(243, 226, 194);	temp.at<Vec3b>(13, 6) = Vec3b(211, 126, 0);		temp.at<Vec3b>(13, 7) = Vec3b(207, 113, 85);
	temp.at<Vec3b>(14, 0) = Vec3b(105, 194, 30);	temp.at<Vec3b>(14, 1) = Vec3b(186, 249, 251);	temp.at<Vec3b>(14, 2) = Vec3b(0, 248, 255);		temp.at<Vec3b>(14, 3) = Vec3b(0, 0, 254);		temp.at<Vec3b>(14, 4) = Vec3b(44, 0, 231);		temp.at<Vec3b>(14, 5) = Vec3b(39, 2, 138);		temp.at<Vec3b>(14, 6) = Vec3b(105, 16, 231);	temp.at<Vec3b>(14, 7) = Vec3b(112, 39, 247);
	temp.at<Vec3b>(15, 0) = Vec3b(177, 93, 130);	temp.at<Vec3b>(15, 1) = Vec3b(248, 207, 231);	temp.at<Vec3b>(15, 2) = Vec3b(225, 58, 211);	temp.at<Vec3b>(15, 3) = Vec3b(254, 0, 255);		temp.at<Vec3b>(15, 4) = Vec3b(199, 21, 252);	temp.at<Vec3b>(15, 5) = Vec3b(208, 113, 254);	temp.at<Vec3b>(15, 6) = Vec3b(153, 2, 254);		temp.at<Vec3b>(15, 7) = Vec3b(235, 222, 251);

	Mat tempHSV;
	temp.convertTo(tempHSV, CV_32FC3, (double)1.f / 255.f);

	cvtColor(tempHSV, tempHSV, CV_BGR2HSV);

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 8; j++) {
			palletteColor[i][j].x = tempHSV.at<Vec3f>(i, j)[0];
			tempHSV.at<Vec3f>(i, j)[1] *= 100;
			tempHSV.at<Vec3f>(i, j)[2] *= 100;
			palletteColor[i][j].y = tempHSV.at<Vec3f>(i, j)[1];
			palletteColor[i][j].z = tempHSV.at<Vec3f>(i, j)[2];
		}
	}

	//imshow("Pallette", tempHSV);
}

void sampleExtraction(Mat frame) {
	vector<Rect> faces; 
	vector<Rect> eyes;
	vector<Rect> nose;
	Mat frame_gray;
	Mat faceROI;
	Point face_center;
	Point eyes_center[2];
	Point nose_center;

	Point eyes_middle;
	Point forehead_center;
	Point cheek_center[2];
	Point temple_center[2];
	const int offset = 5; //sample size = 10x10


	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	/**************** Detect Face *****************/
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	faceROI = frame_gray(faces[0]);
	face_center = Point(faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5);
	//rectangle(frame, Point(faces[0].x, faces[0].y), Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height), Scalar(255, 0, 0), 1, 8, 0);
	//rectangle(frame, face_center, face_center, Scalar(255, 0, 0), 3, 8, 0);

	/***************** Detect Eye *****************/
	eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t j = 0; j < eyes.size(); j++){
		eyes_center[j] = Point(faces[0].x + eyes[j].x + eyes[j].width*0.5, faces[0].y + eyes[j].y + eyes[j].height*0.5);
		//rectangle(frame, eyes_center[j] - Point(eyes[j].width*0.5, eyes[j].height*0.5), eyes_center[j] + Point(eyes[j].width*0.5, eyes[j].height*0.5), Scalar(0, 0, 255), 1, 8, 0);
		//rectangle(frame, eyes_center[j], eyes_center[j], Scalar(255, 0, 255), 3, 8, 0);
	}

	/****************** Detect Nose ****************/	
	nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	nose_center = Point(faces[0].x + nose[0].x + nose[0].width*0.5, faces[0].y + nose[0].y + nose[0].height*0.5);
	//rectangle(frame, nose_center - Point(nose[0].width*0.5, nose[0].height*0.5), nose_center + Point(nose[0].width*0.5, nose[0].height*0.5), Scalar(0, 0, 255), 1, 8, 0);
	//rectangle(frame, nose_center, nose_center, Scalar(255, 0, 255), 3, 8, 0);
		
	/***************** Sample Extraction ****************/
	eyes_middle = (eyes_center[0] + eyes_center[1]) / 2;
	//rectangle(frame, eyes_middle, eyes_middle, Scalar(0, 255, 255), 3, 8, 0);
	
	forehead_center = 2 * eyes_middle - nose_center; // 이마랑 코의 중점이 middle of eyes
	//rectangle(frame, forehead_center, forehead_center, Scalar(0, 255, 255), 3, 8, 0);
	
	temple_center[0] = Point(eyes_center[0].x + eyes[0].width*0.5, eyes_center[0].y);
	temple_center[1] = Point(eyes_center[1].x - eyes[1].width*0.5, eyes_center[1].y);
	
	for (int i = 0; i < 2; i++) {
		cheek_center[i] = Point(eyes_center[i].x, nose_center.y);
	//	rectangle(frame, cheek_center[i], cheek_center[i], Scalar(0, 255, 255), 3, 8, 0);
	
		temple_center[i] = 2 * eyes_center[i] - eyes_middle;
		//rectangle(frame, temple_center[i], temple_center[i], Scalar(0, 255, 255), 3, 8, 0);
	}

	sample[0] = frame(Rect(forehead_center.x - offset, forehead_center.y - offset, offset*2, offset*2)); // forehead
	sample[1] = frame(Rect(temple_center[0].x - offset, temple_center[0].y - offset, offset * 2, offset * 2)); // temple
	sample[2] = frame(Rect(temple_center[1].x - offset, temple_center[1].y - offset, offset * 2, offset * 2));
	sample[3] = frame(Rect(cheek_center[0].x - offset, cheek_center[0].y - offset, offset * 2, offset * 2)); // cheek
	sample[4] = frame(Rect(cheek_center[1].x - offset, cheek_center[1].y - offset, offset * 2, offset * 2));
	
	/*
	rectangle(frame, Point(temple_center[0].x - 5, temple_center[0].y - 5), Point(temple_center[0].x + 5, temple_center[0].y + 5), Scalar(255, 255, 255), 1, 8, 0);
	rectangle(frame, Point(temple_center[1].x - 5, temple_center[1].y - 5), Point(temple_center[1].x + 5, temple_center[1].y + 5), Scalar(255, 255, 255), 1, 8, 0);
	rectangle(frame, Point(cheek_center[0].x - 5, cheek_center[0].y - 5), Point(cheek_center[0].x + 5, cheek_center[0].y + 5), Scalar(255, 255, 255), 1, 8, 0);
	rectangle(frame, Point(cheek_center[1].x - 5, cheek_center[1].y - 5), Point(cheek_center[1].x + 5, cheek_center[1].y + 5), Scalar(255, 255, 255), 1, 8, 0);
	rectangle(frame, Point(forehead_center.x - 5, forehead_center.y - 5), Point(forehead_center.x + 5, forehead_center.y + 5), Scalar(255, 255, 255), 1, 8, 0);
	*/

	//imwrite("CenterDetection2.png", frame);
	//imshow("Frame", frame);
	//imshow("GRAY", faceROI);

}

void rgbToLab() {
	int i;
	Mat tempLab[5];

	//Lab로 변환
	for (int i = 0; i < 5; i++) {
		sample[i].convertTo(tempLab[i], CV_32FC3, (double)1.f / 255.f);
		cvtColor(tempLab[i], tempLab[i], CV_BGR2Lab);
	}


	i = 0;
	for (int x = 0; x < 10; x++) {  //10 = sample[0].cols
		for (int y = 0; y < 10; y++) { //10 = sample[0].rows
			forehead[i].x = tempLab[0].at<Vec3f>(y, x)[0];
			forehead[i].y = tempLab[0].at<Vec3f>(y, x)[1];
			forehead[i].z = tempLab[0].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			temple1[i].x = tempLab[1].at<Vec3f>(y, x)[0];
			temple1[i].y = tempLab[1].at<Vec3f>(y, x)[1];
			temple1[i].z = tempLab[1].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			temple2[i].x = tempLab[2].at<Vec3f>(y, x)[0];
			temple2[i].y = tempLab[2].at<Vec3f>(y, x)[1];
			temple2[i].z = tempLab[2].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			cheek1[i].x = tempLab[3].at<Vec3f>(y, x)[0];
			cheek1[i].y = tempLab[3].at<Vec3f>(y, x)[1];
			cheek1[i].z = tempLab[3].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			cheek2[i].x = tempLab[4].at<Vec3f>(y, x)[0];
			cheek2[i].y = tempLab[4].at<Vec3f>(y, x)[1];
			cheek2[i].z = tempLab[4].at<Vec3f>(y, x)[2];
			i++;
		}
	}

}

void rgbToHSV() {
	int i;
	Mat tempHSV[5];

	//HSV로 변환
	for (int i = 0; i < 5; i++) {
		sample[i].convertTo(tempHSV[i], CV_32FC3, (double)1.f / 255.f);
		cvtColor(tempHSV[i], tempHSV[i], CV_BGR2HSV);
	}


	i = 0;
	for (int x = 0; x < 10; x++) {  //10 = sample[0].cols
		for (int y = 0; y < 10; y++) { //10 = sample[0].rows
			forehead[i].x = tempHSV[0].at<Vec3f>(y, x)[0];
			forehead[i].y = 100 * tempHSV[0].at<Vec3f>(y, x)[1];
			forehead[i].z = 100 * tempHSV[0].at<Vec3f>(y, x)[2];
			i++;
		}	
	}

	i = 0;
	for (int x = 0; x < 10; x++) {  
		for (int y = 0; y < 10; y++) { 
			temple1[i].x = tempHSV[1].at<Vec3f>(y, x)[0];
			temple1[i].y = 100 * tempHSV[1].at<Vec3f>(y, x)[1];
			temple1[i].z = 100 * tempHSV[1].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {  
		for (int y = 0; y < 10; y++) { 
			temple2[i].x = tempHSV[2].at<Vec3f>(y, x)[0];
			temple2[i].y = 100 * tempHSV[2].at<Vec3f>(y, x)[1];
			temple2[i].z = 100 * tempHSV[2].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) { 
		for (int y = 0; y < 10; y++) { 
			cheek1[i].x = tempHSV[3].at<Vec3f>(y, x)[0];
			cheek1[i].y = 100 * tempHSV[3].at<Vec3f>(y, x)[1];
			cheek1[i].z = 100 * tempHSV[3].at<Vec3f>(y, x)[2];
			i++;
		}
	}

	i = 0;
	for (int x = 0; x < 10; x++) {  
		for (int y = 0; y < 10; y++) { 
			cheek2[i].x = tempHSV[4].at<Vec3f>(y, x)[0];
			cheek2[i].y = 100 * tempHSV[4].at<Vec3f>(y, x)[1];
			cheek2[i].z = 100 * tempHSV[4].at<Vec3f>(y, x)[2];
			i++;
		}
	}

}

Point3d binarySplit(Point3d sample[]) {
	Point3d center[2] = { 0.0, };
	double distance[2][100];
	int flag[100];
	double maxDst = 0.0;
	int maxIdx = 0;
	int flagNum[2] = { 0, };

	for (int i = 0; i < 100; i++) {
		center[0] += sample[i];
	}
	center[0] /= 100;

	//printf("%lf %lf %lf \n", center[0].x, center[0].y, center[0].z);
	
	for (int i = 0; i < 100; i++) {
		distance[0][i] = getDistanceLab(sample[i], center[0]);

		//printf("%lf %lf %lf %lf\n", sample[i].x, sample[i].y, sample[i].z, distance[0][i]);
		if (maxDst < distance[0][i]) {
			maxDst = distance[0][i];
			maxIdx = i;
		}
	}
	center[1] = sample[maxIdx];
	//printf("%lf %d\n", maxDst, maxIdx);
	
	for (int i = 0; i < 100; i++) {
		distance[1][i] = getDistanceLab(sample[i], center[1]);
		//printf("%lf %lf ", distance[0][i], distance[1][i]);

		if (distance[0][i] > distance[1][i]) {
			flag[i] = 1;
			flagNum[1] += 1;
		}
		else {
			flag[i] = 0;
			flagNum[0] += 1;
		}
		//printf("%d\n", flag[i]);
	}
	
	center[0] = Point3d(0.0, 0.0, 0.0); center[1] = Point3d(0.0, 0.0, 0.0);

	for (int i = 0; i < 100; i++) {
		if (flag[i] == 1)
			center[1] += sample[i];
		else if (flag[i] == 0)
			center[0] += sample[i];
	}

	center[0] /= flagNum[0];
	center[1] /= flagNum[1];
	
	/*
	printf("%d %d\n", flagNum[0], flagNum[1]);
	printf("%lf %lf %lf \n", center[0].x, center[0].y, center[0].z);
	printf("%lf %lf %lf \n", center[1].x, center[1].y, center[1].z);
	printf("\n");
	*/

	if (flagNum[0] > flagNum[1]) return center[0];
	else return center[1];
}

double getDistanceLab(Point3d a, Point3d b) {
	double distance;
	
	distance = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
	
	return distance;
}

double getDistanceHSV(Point3d a, Point3d b) {
	double distance;
	float h1, h2, s1, s2;
	float theta;
	
	if (a.x > b.x) {
		h1 = a.x; h2 = b.x;
		s1 = a.y; s2 = b.y;
	}
	else {
		h1 = b.x; h2 = a.x;
		s1 = b.y; s2 = a.y;
	}
	theta = (h1 - h2)*PI / 180;
	
	distance = (theta*sqrt(pow(s1, 2) + pow(s2, 2) - 2 * s1*s2*cos(theta))) / (2 * sin(theta / 2));
	
	return distance;
}

int findSkin(Point3d a[]) {
	double distance[16];
	int personalColor[4] = { 0, }; //봄,여름,가을,겨울 = 0,1,2,3
	
	
	for (int sampleN = 0; sampleN < 5; sampleN++) {
		int idx = 0;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				distance[idx] = getDistanceLab(a[sampleN], skinColor[i][j]);
				printf("%lf\t", distance[idx]);
				idx++;
			}
			printf("\n");
		}
		personalColor[findMinIdx(distance, 16)] ++; //거리가 가장 짧은 계절(행index)을 반환한다	
		for (int i = 0; i < 16; i++) { distance[i] = 0; } //초기화
		
		printf("\n\n");
	}
	
	
	for (int i = 0; i < 4; i++) {
		printf("%d  ", personalColor[i]);
	}
	

	return findMaxIdx(personalColor, 4);

 }

void findLip(Point3d a[], int personalColor) {
	float distance[5][6]; 
    
	//LIP color 확인용
	

	//printf("%12lf\n%12lf\n%12lf\n\n", a[0].x, a[0].y, a[0].z);

	/*
	for (int i = 0; i < 6; i++) {
		printf("%12lf", lipColor[personalColor][i].x);
	}
	printf("\n");

	for (int i = 0; i < 6; i++) {
		printf("%12lf", lipColor[personalColor][i].y);
	}
	printf("\n");

	for (int i = 0; i < 6; i++) {
		printf("%12lf", lipColor[personalColor][i].z);
	}
	printf("\n\n");
	*/



	/*
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 6; j++) {
			distance[i][j] = getDistanceLab(a[i], lipColor[personalColor][j]);
			printf("%12lf", distance[i][j]);
		}
		printf("\n");
	}
	*/

}

int findMinIdx(double arr[], int len) {
	double min = arr[0];
	int minIdx = 0;
	
	for (int i = 0; i < len; i++) {
		if (min > arr[i]) {
			min = arr[i];
			minIdx = i;
		}
	}
	return minIdx;
}

int findMaxIdx(int arr[], int len) {
	int max = arr[0];
	int maxIdx = 0;

	for (int i = 0; i < len; i++) {
		if (max < arr[i]) {
			max = arr[i];
			maxIdx = i;
		}
	}

	return maxIdx;
}

void classification() {
	double distance[32];
	int idx;
	int minIdx = 0;
	
	//for (int i = 0; i < 6; i++) {
		idx = 0;
		//printf("%12lf %12lf %12lf\n", lipColor[1][0].x, lipColor[1][0].y, lipColor[1][0].z);

		for (int a = 0; a < 4; a++) {
			for (int b = 0; b < 8; b++) {
				distance[idx] = getDistanceLab(lipColor[1][0], palletteColor[a][b]);
				//printf("%12lf", distance[idx]);
				idx++;
			}
			//printf("\n");
		}
		//printf("\n");
		minIdx = findMinIdx(distance, 32);
		printf("%d %d\n", (minIdx / 8)+1, (minIdx % 8)+1);
	//}
	

}