#pragma once
#ifndef ISE_PROCESSOR_INTERFACE_H
#define ISE_PROCESSOR_INTERFACE_H

#include "DataTypes.h"

int iseInitWithSettings(const IseCommonSettings* settings);

int iseUpdateDynamicParameters(const IseDynamicParameters* parameters);

int iseUploadFrames(const IseRgbFrame* rgbFrame, const IseDepthFrame* depthFrame);

int iseDetect();

int iseDownloadResults(IseDetectionResults* results);

int iseDownloadDebugFrames(IseRgbFrame* rgbFrame);

int iseRelease();

//TODO: for CUDA, provide interface for OpenGL interop

#endif

