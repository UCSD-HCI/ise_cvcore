#include <stdio.h>
#include <stdlib.h>

#include <highgui.h>
#include "DataTypes.h"
#include "KinectSimulator.h"
#include "Detector.h"

int main()
{
	const char windowName[] = "Window";

	IseCommonSettings settings;
	settings.rgbWidth = 640;
	settings.rgbHeight = 480;
	settings.depthWidth = 640;
	settings.depthHeight = 480;
	settings.maxDepthValue = 65535;

	iseKinectInitWithSettings(&settings, "C:\\Users\\cuda\\kinect\\record\\rec130406-1714");
	iseDetectorInitWithSettings(&settings);

	cvNamedWindow(windowName);

	IseRgbFrame rgbFrame;
	IseDepthFrame depthFrame;
	IseRgbFrame debugFrame;

	//allocate debug frame
	debugFrame.width = settings.depthWidth;
	debugFrame.height = settings.depthHeight;
	debugFrame.isDataOwner = 1;
	debugFrame.dataBytes = settings.depthWidth * settings.depthHeight * 3;
	debugFrame.data = (uchar*)malloc(debugFrame.dataBytes);

	IplImage* rgbFrameIpl = cvCreateImageHeader(cvSize(settings.rgbWidth, settings.rgbHeight), IPL_DEPTH_8U, 3);
	IplImage* debugFrameIpl = cvCreateImageHeader(cvSize(settings.depthWidth, settings.depthHeight), IPL_DEPTH_8U, 3);

	while(iseKinectCapture(&rgbFrame, &depthFrame) != ERROR_KINECT_EOF)
	{
		//rgbFrameIpl->imageData = (char*)rgbFrame.data;
		//cvShowImage(windowName, rgbFrameIpl);

		iseDetectorDetect(&rgbFrame, &depthFrame, &debugFrame);
		debugFrameIpl->imageData = (char*)debugFrame.data;
		cvShowImage(windowName, debugFrameIpl);

		if (cvWaitKey(33) == 27)
		{
			break;
		}
	}

	iseDetectorRelease();
	iseKinectRelease();

	free(debugFrame.data);
	debugFrame.isDataOwner = 0;

	return 0;
}

