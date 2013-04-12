#include "KinectSimulator.h"

#include <opencv2\opencv.hpp>
#include <highgui.h>
#include <stdio.h>
using namespace cv;
using namespace ise;

KinectSimulator::KinectSimulator(const CommonSettings& settings, const char* recFilePrefix, Mat& rgbFrameBuffer, Mat& depthFrameBuffer)
    : _settings(settings), _rgbFrame(rgbFrameBuffer), _depthFrame(depthFrameBuffer), _currentFrame(0)
{
    assert(_depthFrame.isContinuous());

	//open rgb capture
	char path[255];
	sprintf(path, "%s.rgb.avi", recFilePrefix);
    _rgbCapture.open(path);
    _frameCount = (int)_rgbCapture.get(CV_CAP_PROP_FRAME_COUNT);
	
	sprintf(path, "%s.depth.bin", recFilePrefix);
	_depthFp = fopen(path, "rb");

	//find the file size
	fseek(_depthFp, 0L, SEEK_END);
	int sz = ftell(_depthFp);
	fseek(_depthFp, 0L, SEEK_SET);

    int depthFrameCount = sz / (_settings.depthWidth * _settings.depthHeight * sizeof(ushort));
	if (depthFrameCount < _frameCount)
	{
		_frameCount = depthFrameCount;
	}
}

int KinectSimulator::capture()
{
	if (_currentFrame >= _frameCount)
	{
		return ERROR_KINECT_EOF;
	}

    _rgbCapture >> _rgbFrame;
    
    //We can do this on GPU
    //cvtColor(_rgbFrame, _rgbFrame, CV_BGR2RGB);
    
    //_depthFrame must be continuous. Checked in constructor
    fread(_depthFrame.data, sizeof(ushort), _settings.depthWidth * _settings.depthHeight, _depthFp);

	_currentFrame++;

	return 0;
}

KinectSimulator::~KinectSimulator()
{
	fclose(_depthFp);

    //_rgbCapture released by its destructor
}