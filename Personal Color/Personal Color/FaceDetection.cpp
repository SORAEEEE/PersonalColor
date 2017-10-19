#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void setFaceLab();
void setLipLab();
void sampleExtraction(Mat frame);
void rgbToLab();
Point3d binarySplit(Point3d sample[]);
double getDistance(Point3d a, Point3d b);
void findSkin(Point3d a);
void findLip(Point3d a, int personalColor);
int findMinIdx(float arr[], int len);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String nose_cascade_name = "Nariz.xml";
String img_name = "face.png";
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

int main(int argc, const char** argv) {
	Mat img = imread(img_name);
	Point3d mainVal[5]; //sample main value

	if (img.data == NULL) {
		printf("이미지 열기 실패");
		return -1;
	}

	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error Face cascade loading\n"); return -1;};
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error Eyes cascade loading\n"); return -1;};
	if (!nose_cascade.load(nose_cascade_name)) { printf("--(!)Error Nose cascade loading\n"); return -1;};

	setFaceLab(); //personal color setting
	setLipLab();
	sampleExtraction(img); //Sample Extraction
	rgbToLab(); //RGB to Lab
	
	//imshow("sampleLab.png", sample[2]);

	/* Select main value */
	mainVal[0] = binarySplit(forehead);
	mainVal[1] = binarySplit(temple1);
	mainVal[2] = binarySplit(temple2);
	mainVal[3] = binarySplit(cheek1);
	mainVal[4] = binarySplit(cheek2);
	
	/* Find personal color */
	for (int i = 0; i < 5; i++) {
		findSkin(mainVal[i]);
	}
	
	waitKey(0);
	return 0;
}

void setFaceLab() {
	Mat temp(4,4,CV_8UC3);

	temp.at<Vec3b>(0, 0) = Vec3b(175, 215, 250); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(158, 207, 255);
	temp.at<Vec3b>(0, 2) = Vec3b(170, 203, 253);
	temp.at<Vec3b>(0, 3) = Vec3b(140, 209, 252);
	temp.at<Vec3b>(1, 0) = Vec3b(183, 234, 254); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(198, 223, 255);
	temp.at<Vec3b>(1, 2) = Vec3b(178, 221, 254);
	temp.at<Vec3b>(1, 3) = Vec3b(132, 215, 253);
	temp.at<Vec3b>(2, 0) = Vec3b(159, 225, 254); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(161, 210, 248);
	temp.at<Vec3b>(2, 2) = Vec3b(138, 205, 250);
	temp.at<Vec3b>(2, 3) = Vec3b(112, 177, 216);
	temp.at<Vec3b>(3, 0) = Vec3b(157, 224, 255); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(157, 211, 242);
	temp.at<Vec3b>(3, 2) = Vec3b(131, 212, 249);
	temp.at<Vec3b>(3, 3) = Vec3b(112, 181, 220);

	Mat tempLab;
	temp.convertTo(tempLab, CV_32FC3, (double)1.f/255.f);

	cvtColor(tempLab, tempLab, CV_BGR2Lab);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			skinColor[i][j].x = tempLab.at<Vec3f>(i, j)[0];
			skinColor[i][j].y = tempLab.at<Vec3f>(i, j)[1];
			skinColor[i][j].z = tempLab.at<Vec3f>(i, j)[2];
		}
	}
	
	//imshow("personal color", temp);
}

void setLipLab() {
	Mat temp(4, 6, CV_8UC3);
	
	temp.at<Vec3b>(0, 0) = Vec3b(79,79,234); //Spring
	temp.at<Vec3b>(0, 1) = Vec3b(73,29,195);
	temp.at<Vec3b>(0, 2) = Vec3b(130,125,254);
	temp.at<Vec3b>(0, 3) = Vec3b(111,127,247);
	temp.at<Vec3b>(0, 4) = Vec3b(47,43,223);
	temp.at<Vec3b>(0, 5) = Vec3b(84,110,250);
	temp.at<Vec3b>(1, 0) = Vec3b(147,63,243); //Summer
	temp.at<Vec3b>(1, 1) = Vec3b(73,29,195);
	temp.at<Vec3b>(1, 2) = Vec3b(120,39,170);
	temp.at<Vec3b>(1, 3) = Vec3b(43,18,203);
	temp.at<Vec3b>(1, 4) = Vec3b(144,116,232);
	temp.at<Vec3b>(1, 5) = Vec3b(106,50,228);
	temp.at<Vec3b>(2, 0) = Vec3b(104,107,227); //Autumn
	temp.at<Vec3b>(2, 1) = Vec3b(41,48,175);
	temp.at<Vec3b>(2, 2) = Vec3b(45,32,186);
	temp.at<Vec3b>(2, 3) = Vec3b(43,18,203);
	temp.at<Vec3b>(2, 4) = Vec3b(47,43,223);
	temp.at<Vec3b>(2, 5) = Vec3b(34,45,184);
	temp.at<Vec3b>(3, 0) = Vec3b(63,26,160); //Winter
	temp.at<Vec3b>(3, 1) = Vec3b(103,37,207);
	temp.at<Vec3b>(3, 2) = Vec3b(120,39,170);
	temp.at<Vec3b>(3, 3) = Vec3b(81,33,198);
	temp.at<Vec3b>(3, 4) = Vec3b(40,5,164);
	temp.at<Vec3b>(3, 5) = Vec3b(19,2,208);

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

	//imshow("LIP", temp);
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
		distance[0][i] = getDistance(sample[i], center[0]);

		//printf("%lf %lf %lf %lf\n", sample[i].x, sample[i].y, sample[i].z, distance[0][i]);
		if (maxDst < distance[0][i]) {
			maxDst = distance[0][i];
			maxIdx = i;
		}
	}
	center[1] = sample[maxIdx];
	//printf("%lf %d\n", maxDst, maxIdx);
	
	for (int i = 0; i < 100; i++) {
		distance[1][i] = getDistance(sample[i], center[1]);
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

double getDistance(Point3d a, Point3d b) {
	double distance;

	distance = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
	
	return distance;
}

void findSkin(Point3d a) {
	float distance[16];
	int idx = 0;
	int personalColor; //봄,여름,가을,겨울 = 0,1,2,3

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			distance[idx] = getDistance(a, skinColor[i][j]);
			idx++;
		}
	}

	personalColor = findMinIdx(distance, 16); //거리가 가장 짧은 계절(행index)을 반환한다	
	findLip(a, personalColor);

	/*
	for (int i = 0; i < 16; i++) {
		printf("%lf\t", distance[i]);
		if (i % 4 == 3) printf("\n");
	}

	printf("%lf %lf %lf\n", a.x, a.y, a.z);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%lf\t", getDistance(a, skinColor[i][j]));
		}
		printf("\n");
	}
	printf("\n\n");
	*/
}

void findLip(Point3d a, int personalColor) {
	float distance[6]; 
    
	/*
	printf("%12lf\t%12lf\t%12lf\n\n", a.x, a.y, a.z);
	
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

	for (int i = 0; i < 6; i++) {
		distance[i] = getDistance(a, lipColor[personalColor][i]);
		printf("%12lf",distance[i]);
	}
	printf("\n");


}

int findMinIdx(float arr[], int len) {
	float min = arr[0];
	int minIdx = 0;
	
	for (int i = 0; i < len; i++) {
		if (min > arr[i]) {
			min = arr[i];
			minIdx = i;
		}
	}

	//printf("%d %lf", minIdx, min);

	return minIdx / 4;
}