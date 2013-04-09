#include "KinectSimulator.h"

#include <cv.h>
#include <highgui.h>
#include <stdio.h>

static IplImage* _rgbFrameIpl;
static IplImage* _depthFrameIpl;

static FILE* _depthFp;
static CvCapture* _rgbCapture;

static int _currentFrame;
static int _frameCount;

int iseKinectInitWithSettings(const IseCommonSettings* settings, const char* recFilePrefix)
{
	_rgbFrameIpl = cvCreateImage(cvSize(settings->rgbWidth, settings->rgbHeight), IPL_DEPTH_8U, 3);
	_depthFrameIpl = cvCreateImage(cvSize(settings->depthWidth, settings->depthHeight), IPL_DEPTH_16U, 1);
	

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

int iseKinectCapture(IseRgbFrame* rgbFrame, IseDepthFrame* depthFrame, int dataCopy)
{
	if (dataCopy)
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

	rgbFrame->header.width = _rgbFrameIpl->width;
	rgbFrame->header.height = _rgbFrameIpl->height;
	rgbFrame->header.dataBytes = _rgbFrameIpl->imageSize;
	rgbFrame->header.isDataOwner = 0;
	rgbFrame->data = (uchar*)_rgbFrameIpl->imageData;

	depthFrame->header.width = _depthFrameIpl->width;
	depthFrame->header.height = _depthFrameIpl->height;
	depthFrame->header.dataBytes = _depthFrameIpl->imageSize;
	depthFrame->header.isDataOwner = 0;
	depthFrame->data = (ushort*)_depthFrameIpl->imageData;

	return 0;
}

int iseKinectRelease()
{
	cvReleaseCapture(&_rgbCapture);
	fclose(_depthFp);

	cvReleaseImage(&_rgbFrameIpl);
	cvReleaseImage(&_depthFrameIpl);

	return 0;
}