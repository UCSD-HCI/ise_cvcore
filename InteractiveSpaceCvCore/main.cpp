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

	cvNamedWindow(windowName);

	IseRgbFrame rgbFrame;
	IseDepthFrame depthFrame;
	IseRgbFrame debugFrame;

	//init rgb/depth frame
	rgbFrame.header = iseCreateImageHeader(settings.rgbWidth, settings.rgbHeight, 3);
	depthFrame.header = iseCreateImageHeader(settings.depthWidth, settings.depthHeight, sizeof(ushort));

	//allocate debug frame
	debugFrame.header = iseCreateImageHeader(settings.depthWidth, settings.depthHeight, 3, 1);
	debugFrame.data = (uchar*)malloc(debugFrame.header.dataBytes);

	//init simulator and detector
	iseKinectInitWithSettings(&settings, "C:\\Users\\cuda\\kinect\\record\\rec130408-1700", &rgbFrame, &depthFrame);
	iseDetectorInitWithSettings(&settings);

	//for output
	IplImage* debugFrameIpl = cvCreateImageHeader(cvSize(settings.depthWidth, settings.depthHeight), IPL_DEPTH_8U, 3);

	while(iseKinectCapture() != ERROR_KINECT_EOF)
	{ 
		//rgbFrameIpl->imageData = (char*)rgbFrame.data;
		//cvShowImage(windowName, rgbFrameIpl);

		iseDetectorDetect(&rgbFrame, &depthFrame, &debugFrame);
		debugFrameIpl->imageData = (char*)debugFrame.data;
		cvShowImage(windowName, debugFrameIpl);

		if (cvWaitKey(1) == 27)
		{
			break; 
		}
	}

	iseDetectorRelease();
	iseKinectRelease();

	free(debugFrame.data);
	debugFrame.data = NULL;
	debugFrame.header.isDataOwner = 0;

	return 0;
}

