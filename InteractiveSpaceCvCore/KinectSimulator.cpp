#include "KinectSimulator.h"

#include <opencv2\opencv.hpp>
#include <highgui.h>
#include <stdio.h>

#include <Windows.h>
#include <NuiApi.h>

using namespace cv;
using namespace ise;

KinectSimulator::KinectSimulator(const CommonSettings& settings, const char* recFilePrefix, Mat& rgbFrame, Mat& depthFrame, Mat& depthToRgbCoordFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _depthToRgbCoordFrame(depthToRgbCoordFrame), _currentFrame(0)
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

    HRESULT hr = NuiCreateSensorByIndex(0, &_sensor);
    assert(SUCCEEDED(hr));

    hr = _sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH); 
    assert(SUCCEEDED(hr));

}

KinectSimulator::~KinectSimulator()
{
	fclose(_depthFp);
    _sensor->Release();
    //_rgbCapture released by its destructor
}

int KinectSimulator::capture()
{
	if (_currentFrame >= _frameCount)
	{
		return ERROR_KINECT_EOF;
	}

    _rgbCapture >> _rgbFrameBuffer;
    
    //copy data
    assert(_rgbFrame.isContinuous() && _rgbFrameBuffer.isContinuous());
    //memcpy(_rgbFrame.data, _rgbFrameBuffer.data, _rgbFrame.rows * _rgbFrame.step);

    //We can do this on GPU
    cvtColor(_rgbFrameBuffer, _rgbFrame, CV_BGR2RGB);
    
    //_depthFrame must be continuous. Checked in constructor
    fread(_depthFrame.data, sizeof(ushort), _settings.depthWidth * _settings.depthHeight, _depthFp);

    //compute depth to color coordinate frame
    HRESULT hr = _sensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
        NUI_IMAGE_RESOLUTION_640x480,
        NUI_IMAGE_RESOLUTION_640x480,
        _settings.depthWidth * _settings.depthHeight,
        (ushort*)_depthFrame.data,
        _settings.depthWidth * _settings.depthHeight * 2,
        (long*)_depthToRgbCoordFrame.data
    );
    assert(SUCCEEDED(hr));
    

	_currentFrame++;

	return 0;
}

