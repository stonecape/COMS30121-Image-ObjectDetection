#include "common.h"
CascadeClassifier cascade;

/** @function main */

int main(int argc, char** argv)
{
		// 1. Read Input Image
		//String inputFilePath = argv[1];
        char* inputFilePath=argv[1];
		//String inputFileName = "dart" + str + ".jpg";
		String groundTruthFileName = argv[1];
		String outputFileName = "detected.jpg";

		//String inputFilePath = "/Users/Theabo/Documents/code/ImageProcessing/Final submission/originalimages/" + inputFileName;
		String groundTruthPath = "/groundtruth/" + groundTruthFileName;
		Mat frame = imread(inputFilePath, CV_LOAD_IMAGE_COLOR);
		Mat img_scene = imread(inputFilePath, CV_LOAD_IMAGE_GRAYSCALE);
		Mat centers = doCircleDetect(inputFilePath);

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

		// 3. Load ground truth data
		fstream   groundFile;
		string groundTruthStr[10];
		int index = 0;
		groundFile.open(groundTruthPath, ios::in);
		if (groundFile.fail())
		{
			//cout << "groundFile doesn't exist." << endl;
			groundFile.close();
		}
		else {
			while (!groundFile.eof())
			{
				char buffer[256];
				groundFile.getline(buffer, 256, '\n');
				groundTruthStr[index] = buffer;
				index++;
			}
		}


		// 4. Detect Faces and Display Result
		detectAndDisplay(frame, outputFileName, groundTruthStr, index, centers);

		// 5. Save Result Image
		//String outputFilePath = "/Users/Theabo/Documents/code/ImageProcessing/Final submission/detectedimage/" + outputFileName;
		//std::cout << outputFilePath << std::endl;
		cv::imwrite(outputFileName, frame);

	return 0;
}

void detectAndDisplay(Mat & frame, String outputFileName, string groundTruthStr[], int groundSize, Mat centers)
{
	/*namedWindow("center", CV_WINDOW_AUTOSIZE);
	imshow("center", centers);

	waitKey(0);*/
	std::vector<Rect> darts;
	Mat frame_gray;
    Mat frame_gray_line;


	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cvtColor(frame, frame_gray_line, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, darts, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Print number of Faces found
	std::cout << "darts.size->" << darts.size() << std::endl;

	// 4. sobel circle detection
	
	int true_count = 0;
	int filterdDartsCount = 0;
	int centerPointsCount[100];
	int hasInterPoints[100];
	

	if (darts.size() > 0) {
		// 5.Draw box around dart found, hough circle and lines
		for (int i = 0; i < darts.size(); i++)
		{
			// houghcircle, count the number of center points in possible darts central area 
			centerPointsCount[i] = 0;
			hasInterPoints[i] = 0;
			int middleSpace = darts[i].width / 3;
			for (int x = darts[i].x + middleSpace; x < darts[i].x + darts[i].width - middleSpace; x++) {
				for (int y = darts[i].y + middleSpace; y < darts[i].y + darts[i].height - middleSpace; y++) {
					if (x >= 0 && x < centers.cols && y >= 0 && y < centers.rows) {
						if (centers.at<float>(y, x) > 0) {
							centerPointsCount[i]++;
						}
					}
				}
			}	

		}

		// row-> the #row th ground truth
		// col-> the #col th dart
		// value-> bbOverlap
		float groundDarts[10][50];
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 50; j++) {
				groundDarts[i][j] = 0.0;
			}
		}

		for (int j = 0; j < darts.size(); j++) {
			//&& hasInterPoints[j] == 1
			int vote = 0;
			if (centerPointsCount[j] > center_point_count_threshold) {
                cout<<"circle done"<<endl;
				vote++;
			}
			if (doSurfDetect(frame, darts[j])) {
                cout<<"surf"<<endl;
				vote++;
			}
			if (doHoughLine(frame_gray_line, darts[j])) {
                cout<<"line done"<<endl;
				vote++;
			}

			if (vote >= 2) {
				filterdDartsCount++;
				doSurfDetect(frame, darts[j]);

				cv::rectangle(frame, Point(darts[j].x, darts[j].y),
					Point(darts[j].x + darts[j].width, darts[j].y + darts[j].height),
					Scalar(0, 255, 0), 2);


				// calculate F1 depend on ground truth
				for (int z = 0; z < groundSize; z++) {
					vector<string> v = split(groundTruthStr[z], ",");
					BoundBox g_box, box;
					g_box.x = atoi(v[0].c_str());
					g_box.y = atoi(v[1].c_str());
					g_box.width = atoi(v[2].c_str());
					g_box.height = atoi(v[3].c_str());
					box.x = darts[j].x;
					box.y = darts[j].y;
					box.width = darts[j].width;
					box.height = darts[j].height;

					//cout << bbOverlap(g_box, box) << endl;
					if (bbOverlap(g_box, box) > ground_truth_threshold) {
						groundDarts[z][j] = bbOverlap(g_box, box);
					}

				}
			}

		}
	
		// 6. ignore different bouding boxes detect the same dart and draw ground truth
		for (int i = 0; i < groundSize; i++)
		{
			int darts_count = 0;
			for (int j = 0; j < darts.size(); j++) {
				if (groundDarts[i][j] != 0.0) {
					darts_count++;
				}
			}

			if (darts_count > 1) {
				// over a bouding box detecting the same dart
				filterdDartsCount -= (darts_count - 1);
				true_count++;
			}
			else if (darts_count == 1) {
				true_count++;
			}

			vector<string> v = split(groundTruthStr[i], ",");
			int x = atoi(v[0].c_str());
			int y = atoi(v[1].c_str());
			int width = atoi(v[2].c_str());
			int height = atoi(v[3].c_str());
			//cout << x << " " << y << " " << width << " " << height << endl;
			rectangle(frame, Point(x, y),
				Point(x + width, y + height), Scalar(0, 0, 255), 2);
		}
	}

	float precision = (float)true_count / filterdDartsCount;
	float recall = (float)true_count / groundSize;
	float f1 = 2 * (precision * recall) / (precision + recall);
	std::cout << "filterdDartsCount->" << filterdDartsCount << std::endl;
	std::cout << "true_count->" << true_count << std::endl;
	std::cout << "precision->" << precision << std::endl;
	std::cout << "recall->" << recall << std::endl;
	std::cout << "f1->" << f1 << std::endl;

	/*String outputFilePath = "D:\\course_folder\\image\\COMS30121_2017\\facedetected\\detected.jpg";
	std::cout << outputFilePath << std::endl;
	cv::imwrite(outputFilePath, frame);*/

}

float bbOverlap(const BoundBox& box1, const BoundBox& box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width * box1.height;
	float area2 = box2.width * box2.height;
	float result = intersection / (area1 + area2 - intersection);
	return result;
}

vector<string> split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}
