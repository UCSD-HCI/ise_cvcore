#pragma once
#ifndef ISE_DETECTOR_H
#define ISE_DETECTOR_H

#include "DataTypes.h"

int iseDetectorInitWithSettings(const IseCommonSettings* settings);

int iseDetectorUpdateDynamicParameters(const IseDynamicParameters* parameters);

//debugFrame: caller should allocate it as RGB888 with size specified in common settings
IseFingerDetectionResults iseDetectorDetect(const IseRgbFrame* rgbFrame, const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame = NULL);

int iseDetectorRelease();


#endif
