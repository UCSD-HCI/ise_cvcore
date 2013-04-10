#include "KinectSimulator.h"

#include <cv.h>
#include <highgui.h>
#include <stdio.h>

static IplImage* _rgbFrameIpl;
static IplImage* _depthFrameIpl;

//passed by from the caller. KinectSimulator won't release them. 
static IseRgbFrame* _rgbFrame;
static IseDepthFrame* _depthFrame;
static int _dataCopy;

static FILE* _depthFp;
static CvCapture* _rgbCapture;

static int _currentFrame;
static int _frameCount;

int iseKinectInitWithSettings(const IseCommonSettings* settings, const char* recFilePrefix, IseRgbFrame* rgbFrameBuffer, IseDepthFrame* depthFrameBuffer, int dataCopy)
{
	_rgbFrameIpl = cvCreateImage(cvSize(settings->rgbWidth, settings->rgbHeight), IPL_DEPTH_8U, 3);
	_depthFrameIpl = cvCreateImage(cvSize(settings->depthWidth, settings->depthHeight), IPL_DEPTH_16U, 1);
	
	_rgbFrame = rgbFrameBuffer;
	_depthFrame = depthFrameBuffer;
	_dataCopy = dataCopy;
	
	if (!_dataCopy)
	{
		//just point the data pointer in rgb/depth frame to the data in rgb/depth ipl frame.
		_rgbFrame->data = (uchar*)_rgbFrameIpl->imageData;
		_depthFrame->data = (ushort*)_depthFrameIpl->imageData;
	}

	//open rgb capture
	char path[255];
	sprintf(path, "%s.rgb.avi", recFilePrefix);
	_rgbCapture = cvCaptureFromAVI(path);
	_frameCount = cvGetCaptureProperty(_rgbCapture, CV_CAP_PROP_FRAME_COUNT);

	sprintf(path, "%s.depth.bin", recFilePrefix);
	_depthFp = fopen(path, "rb");

	//find the file size
	fseek(_depthFp, 0L, SEEK_END);
	int sz = ftell(_depthFp);
	fseek(_depthFp, 0L, SEEK_SET);

	int depthFrameCount = sz / (_depthFrameIpl->imageSize);
	if (depthFrameCount < _frameCount)
	{
		_frameCount = depthFrameCount;
	}

	_currentFrame = 0;

	return 0;
}

int iseKinectCapture()
{
	if (_dataCopy)
	{
		//TODO: implement
		assert(0);
	}

	if (_currentFrame >= _frameCount)
	{
		return ERROR_KINECT_EOF;
	}

	IplImage* frame = cvQueryFrame(_rgbCapture);
	cvCopy(frame, _rgbFrameIpl);

	fread(_depthFrameIpl->imageData, _depthFrameIpl->imageSize, 1, _depthFp);

	_currentFrame++;

	return 0;
}

int iseKinectRelease()
{
	cvReleaseCapture(&_rgbCapture);
	fclose(_depthFp);

	cvReleaseImage(&_rgbFrameIpl);
	cvReleaseImage(&_depthFrameIpl);

	_rgbFrame->data = NULL;
	_depthFrame->data = NULL;

	return 0;
}