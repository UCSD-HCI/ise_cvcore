#pragma once
#ifndef ISE_KINECT_SIMULATOR_H
#define ISE_KINECT_SIMULATOR_H

#include "DataTypes.h"

#define ERROR_KINECT_EOF -10

int iseKinectInitWithSettings(const IseCommonSettings* settings, const char* recFilePrefix);

//dataCopy: if set true, then caller should allocate memory for rgbFrame->data and depthFrame->data, and the frames would be copied.
//if set false, then the data pointer will be pointed to an internal memory area. 
int iseKinectCapture(IseRgbFrame* rgbFrame, IseDepthFrame* depthFrame, int dataCopy = 0);

int iseKinectRelease();

#endif