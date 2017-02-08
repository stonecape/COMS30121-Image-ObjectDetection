#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>
#include <fstream>



using namespace std;
using namespace cv;

class BoundBox
{
public:
	int x;   // Length of a box
	int y;  // Breadth of a box
	int width;   // Width of a box
	int height;  // Height of a box

};

/*dart.cpp*/
void detectAndDisplay(Mat & frame, String outputFileName, 
	string groundTruthStr[], int groundSize, Mat centers);
vector<string> split(const string &s, const string &seperator);
float bbOverlap(const BoundBox& box1, const BoundBox& box2);
Mat doCircleDetect(string imagePath);


//const String cascade_name = "D:\\course_folder\\image\\frontalface.xml";
//const String cascade_name = "/Users/Theabo/Documents/code/ImageProcessing/Final submission/cascade.xml";
const String cascade_name = "cascade.xml";

const  float ground_truth_threshold = 0.5;
const  int center_point_count_threshold = 80;
const int inter_point_threshold = 4;

/*houghCircle.cpp*/
void doSobel(Mat &source, Mat &i_x, Mat &i_y, Mat &i_magnitude, Mat &i_direction,
	Mat &i_magnitude_t, int threshold);
int calXGradient(Mat image, int x, int y);
int calYGradient(Mat image, int x, int y);
int normalizeGrandient(int gradient);
int normalizeAngle(float atan_value);
float denormalizeAngle(uchar angle_value);
int calTresholdVal(int originalValue, int threshold);
void doHough(Mat &circle_image, Mat &i_mag_treshold, Mat &i_direction, Mat &i_hough_2d,
	int min_radius, int max_radius, float min_dist, int acc_threshold);
void drawCircle(Mat &image, int x, int y, int r);
void hough2DdetectCenter(Mat &center, Mat &i_hough2d);

const int center_threshold = 50;

/*surf.cpp*/
bool doSurfDetect(Mat img_scene, Rect dartRect);

/*line*/
bool doHoughLine(Mat frame_gray, Rect dartRect);
