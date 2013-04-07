#include <stdio.h>
#include <stdlib.h>

#include <highgui.h>
#include "DataTypes.h"
#include "KinectSimulator.h"

int main()
{
	const char windowName[] = "Window";

	IseCommonSettings settings;
	settings.rgbWidth = 640;
	settings.rgbHeight = 480;
	settings.depthWidth = 640;
	settings.depthHeight = 480;

	iseKinectInitWithSettings(&settings, "C:\\Users\\cuda\\kinect\\record\\rec130406-1714");

	cvNamedWindow(windowName);

	IseRgbFrame rgbFrame;
	IseDepthFrame depthFrame;

	IplImage* rgbFrameIpl = cvCreateImageHeader(cvSize(settings.rgbWidth, settings.rgbHeight), IPL_DEPTH_8U, 3);

	while(iseKinectCapture(&rgbFrame, &depthFrame) != ERROR_KINECT_EOF)
	{
		rgbFrameIpl->imageData = (char*)rgbFrame.data;
		cvShowImage(windowName, rgbFrameIpl);

		if (cvWaitKey(33) == 27)
		{
			iseKinectRelease();
			break;
		}
	}

	return 0;
}

