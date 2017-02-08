#include "common.h"

bool doHoughLine (Mat frame_gray, Rect dartRect) {
    
	Rect rect(dartRect.x, dartRect.y, dartRect.width, dartRect.height);
	Mat i_canny;
	Mat edgImg;
	vector<Vec2f> lines;
   // Mat frame_gray;
    
    //Mat src = imread("/Users/Theabo/Documents/code/ImageProcessing/Final submission/originalimages/dart14.jpg", 1);
    //cvtColor(src, frame_gray, CV_BGR2GRAY);
    
	Canny(frame_gray, i_canny, 50, 200, 3);
	i_canny(rect).copyTo(edgImg);
	Mat intersectPointsVote = Mat(edgImg.rows, edgImg.cols, CV_32SC1, Scalar::all(0));


	HoughLines(edgImg, lines, 1, CV_PI / 180, 50, 0, 0);
    cout<<"line size()"<<lines.size()<<endl;

    imwrite("edgImg.jpg", edgImg);
	for (size_t j = 0; j < lines.size(); j++)
	{
		float rho = lines[j][0], theta = lines[j][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double m = -a / b, c = rho / b;

		for (int x = 0; x < intersectPointsVote.cols; x++) {
			for (int y = 0; y < intersectPointsVote.rows; y++) {
				if (y == cvRound(m * x + c)) {
					intersectPointsVote.at<float>(x, y) = intersectPointsVote.at<float>(x, y) + 1;
				}
			}
		}
	}
    imwrite("intersect.jpg", intersectPointsVote);

	float maxInterValue = 0;
	for (int x = 0; x < intersectPointsVote.cols; x++) {
		for (int y = 0; y < intersectPointsVote.rows; y++) {
			if (intersectPointsVote.at<float>(x, y) > maxInterValue) {
				maxInterValue = intersectPointsVote.at<float>(x, y);
			}
		}
	}

	return maxInterValue > inter_point_threshold;
}
