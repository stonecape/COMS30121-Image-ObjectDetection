# ObjectDetection
## Subtask 1: The Viola-Jones Object Detector
Introduction. Being able to detect and locate instances of an object class in images is important for many computer vision applications as well as an ongoing area of research. This assignment requires we 1) to experiment with the Viola-Jones object detection framework, and 2) combine it with other detection approaches to improve its efficacy. 

## Subtask 2: Building & Testing your own Detector
The second subtask requires we to build an object detector that recognises dartboards. The initial steps of this subtask introduce we to OpenCV’s boosting tool, which we can use to construct an object detector that utilises Haar-like features. 

## Subtask 3: Integration with Shape Detectors
For the third subtask, use the Hough Transform on the query images in order to detect important shape configurations of dartboards.

## Subtask 4: Improving your Detector
The main mission in this task is to reduce the number of FP darts. To achieve this reduction, we decided to apply SURF (SpeededUp Robust Features) to our detector. SURF is a local feature detector and descriptor, which can be used for tasks such as object recognition, image registration etc. 
