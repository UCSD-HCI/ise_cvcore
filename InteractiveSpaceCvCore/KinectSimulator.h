#pragma once
#ifndef ISE_KINECT_SIMULATOR_H
#define ISE_KINECT_SIMULATOR_H

#include "DataTypes.h"

#define ERROR_KINECT_EOF -10

int iseKinectInitWithSettings(const IseCommonSettings* settings, const char* recFilePrefix);

int iseKinectCapture(IseRgbFrame* rgbFrame, IseDepthFrame* depthFrame);

int iseKinectRelease();

#endif