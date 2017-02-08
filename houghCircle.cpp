#include "common.h"

Mat doCircleDetect(string imagePath) {
	Mat image = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
	/*namedWindow("Original_Image", CV_WINDOW_AUTOSIZE);
	imshow("Original_Image", image);*/
	Mat i_x = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat i_y = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat i_magnitude = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat i_direction = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat i_magnitude_t = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat i_hough_2d = Mat(image.rows, image.cols, CV_8UC1, Scalar::all(0));
	Mat center = Mat(image.rows, image.cols, CV_32SC1, Scalar::all(0));
	Mat circle_image = image.clone();
	int min_radius = 1;
	// int max_radius = (sqrt(pow(image.rows, 2) + pow(image.cols, 2))) / 2;
	int max_radius = 120; // manual setting, theoretically we should use the above formula

	doSobel(image, i_x, i_y, i_magnitude, i_direction, i_magnitude_t, 20);
	doHough(circle_image, i_magnitude_t, i_direction, i_hough_2d, min_radius, max_radius, 0.0, 8);
	hough2DdetectCenter(center, i_hough_2d);

	imwrite("i_magnitude.jpg", i_magnitude);
	
	imwrite("i_hough_2d.jpg", i_hough_2d);
	
	imwrite("center.jpg", center);
	
	
	return center;
}


void doSobel(Mat &source, Mat &i_x, Mat &i_y, Mat &i_magnitude,
	Mat &i_direction, Mat &i_magnitude_t, int threshold) {
	// define some local variables
	int gx, gy, sum;

	// blur image
	Mat blurredSource;
	GaussianBlur(source, blurredSource, Size(7, 7), 0, 0);

	// set the borders
	Mat padSource;
	copyMakeBorder(blurredSource, padSource, 1, 1, 1, 1, BORDER_REPLICATE);

	for (int x = 1; x < source.rows - 1; x++) {
		for (int y = 1; y < source.cols - 1; y++) {
			gx = calXGradient(source, x, y);
			gy = calYGradient(source, x, y);

			int normal_gx = normalizeGrandient(gx);
			int normal_gy = normalizeGrandient(gy);

			sum = sqrt(pow(normal_gx - 128, 2) + pow(normal_gy - 128, 2));

			i_x.at<uchar>(x, y) = normal_gx;
			i_y.at<uchar>(x, y) = normal_gy;
			i_magnitude.at<uchar>(x, y) = sum;
			i_magnitude_t.at<uchar>(x, y) = calTresholdVal(sum, threshold);

			if (gx != 0) {
				i_direction.at<uchar>(x, y) = normalizeAngle(atan(gy / gx));
				//std::cout << normalizeAngle(atan(gy / gx))<< std::endl;

			}
			else {
				//std::cout << gx << "  " << gy << std::endl;
				i_direction.at<uchar>(x, y) = normalizeAngle(CV_PI / 2);
			}

		}
	}

}

// normalization functions
int normalizeGrandient(int gradient) {
	return (gradient + 4 * 255) / 8;
}

int normalizeAngle(float atan_value) {
	return (atan_value * 256) / CV_PI + 128;
}

float denormalizeAngle(uchar angle_value) {
	return (angle_value - 128) * CV_PI / 256;
}

int calXGradient(Mat image, int x, int y) {
	return image.at<uchar>(x - 1, y - 1) +
		2 * image.at<uchar>(x, y - 1) +
		image.at<uchar>(x + 1, y - 1) -
		image.at<uchar>(x - 1, y + 1) -
		2 * image.at<uchar>(x, y + 1) -
		image.at<uchar>(x + 1, y + 1);
}

int calYGradient(Mat image, int x, int y) {
	return image.at<uchar>(x - 1, y - 1) +
		2 * image.at<uchar>(x - 1, y) +
		image.at<uchar>(x - 1, y + 1) -
		image.at<uchar>(x + 1, y - 1) -
		2 * image.at<uchar>(x + 1, y) -
		image.at<uchar>(x + 1, y + 1);
}

int calTresholdVal(int originalValue, int threshold) {
	if (originalValue >= threshold)
	{
		return 255;
	}
	else
	{
		return 0;
	}

}

void doHough(Mat &circle_image, Mat &i_mag_treshold, Mat &i_direction, Mat &i_hough_2d,
	int min_radius, int max_radius, float min_dist, int acc_threshold) {

	int sizes3D[] = { i_mag_treshold.rows, i_mag_treshold.cols, max_radius };
	Mat hough_space = Mat(3, sizes3D, CV_32SC1, cv::Scalar(0));//x0,y0,r
	Mat val2D = Mat(i_hough_2d.rows, i_hough_2d.cols, CV_32SC1, Scalar::all(0));
	

	double max_hough = 0.0;
	
	for (int x = 0; x < i_mag_treshold.rows; x++) {
		for (int y = 0; y < i_mag_treshold.cols; y++) {
			if (i_mag_treshold.at<uchar>(x, y) == (uchar)255) {
				float theta = denormalizeAngle(i_direction.at<uchar>(x, y));
				for (int r = min_radius; r <= max_radius; r++) {
					int x0_1 = x + r * sin(theta);
					int y0_1 = y + r * cos(theta);
					int x0_2 = x - r * sin(theta);
					int y0_2 = y - r * cos(theta);

					if (x0_1 < i_mag_treshold.rows && x0_1 >= 0 &&
						y0_1 < i_mag_treshold.cols && y0_1 >= 0) {
						//hough_space.at<int>(x0_1, y0_1, r)++;
						val2D.at<float>(x0_1, y0_1)++;
						if (val2D.at<float>(x0_1, y0_1) > max_hough) {
							max_hough = val2D.at<float>(x0_1, y0_1);
						}
					}


					if (x0_2 < i_mag_treshold.rows && x0_2 >= 0 &&
						y0_2 < i_mag_treshold.cols && y0_2 >= 0) {
						//hough_space.at<float>(x0_2, y0_2, r)++;
						val2D.at<float>(x0_2, y0_2)++;
						if (val2D.at<float>(x0_2, y0_2) > max_hough) {
							max_hough = val2D.at<float>(x0_2, y0_2);
						}

					}

				}
			}

		}
	}
	
	for (int x = 0; x < i_hough_2d.rows; x++) {
		for (int y = 0; y < i_hough_2d.cols; y++) {
			i_hough_2d.at<uchar>(x, y) = (uchar)((val2D.at<float>(x, y) / max_hough) * 255);

		}
	}

	for (int x = 0; x < i_mag_treshold.rows; x++) {
		for (int y = 0; y < i_mag_treshold.cols; y++) {
			for (int r = 0; r < max_radius; r++) {
				int houghVal = hough_space.at<float>(x, y, r);

				if (houghVal > acc_threshold) {
					// highlight points by threshold
					//std::cout << x << " " << y << " " << r << std::endl;
					
					//drawCircle(circle_image, x, y, r);
					Point center(x, y);
					int radius = r;
					// draw the circle center
					circle(circle_image, center, 3, Scalar(0, 255, 0), -1, 8, 0);
					// draw the circle outline
					circle(circle_image, center, radius, Scalar(0, 0, 255), 3, 8, 0);
				}
			}
		}
	}
	
}

void hough2DdetectCenter(Mat &center, Mat &i_hough2d) {
	for (int x = 0; x < i_hough2d.rows; x++) {
		for (int y = 0; y < i_hough2d.cols; y++) {
			if (i_hough2d.at<uchar>(x, y) > center_threshold) {
				
				center.at<float>(x, y)++;
				//std::cout << "centers---" << x << " " << y << " "<< center.at<float>(x, y) << std::endl;
			}

		}
	}
}

void drawCircle(Mat &image, int x, int y, int r) {
	for (int i = x - r; i <= x + r; i++)
	{
		for (int j = y - r; j <= y + r; j++)
		{
			int distance = (int)sqrt((i - x)*(i - x) + (j - y)*(j - y));
			if (distance == r) {
				image.at<int>(i, j) = 50;
			}
		}

	}
}

