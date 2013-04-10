#pragma once
#ifndef ISE_KINECT_SIMULATOR_H
#define ISE_KINECT_SIMULATOR_H

#include "DataTypes.h"

#define ERROR_KINECT_EOF -10

//The caller is responsible to create proper header for rgbFrameBuffer and depthFrameBuffer. If dataCopy == 1, the caller should also allocate memory for the data field. 
//dataCopy: if set true, then caller should allocate memory for rgbFrame->data and depthFrame->data, and the frames would be copied.
//if set false, then the data pointer will be pointed to an internal memory area. 
int iseKinectInitWithSettings(const IseCommonSettings* settings, const char* recFilePrefix, IseRgbFrame* rgbFrameBuffer, IseDepthFrame* depthFrameBuffer, int dataCopy = 0);

//read the next rgb/depth frames and store them in the buffers specified in initWithSettings
int iseKinectCapture();

int iseKinectRelease();

#endif