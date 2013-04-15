#pragma once
#ifndef ISE_DATA_TYPES_H
#define ISE_DATA_TYPES_H

#ifndef NULL
#define NULL 0
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;

namespace ise
{
    const static int ISE_MAX_FINGER_NUM = 64;

    //sub-structures
    typedef struct _OmniTouchParameters
    {
	    int stripMaxBlankPixel;
	    int fingerMinPixelLength;
	    int fingerToHandOffset;	//in millimeters
	    int clickFloodArea;

	    double fingerWidthMin;
	    double fingerWidthMax;
	    double fingerLengthMin;
	    double fingerLengthMax;
	    double fingerRisingThreshold;
	    double fingerFallingThreshold;
	    double clickFloodMaxGrad;
    } OmniTouchParameters;

    typedef struct _KinectIntrinsicParameters
    {
	    float realWorldXToZ;
	    float realWorldYToZ;
	    float depthSlope;
	    float depthIntercept;
    } KinectIntrinsicParameters;

    typedef struct _Finger
    {
	    int tipX, tipY, tipZ;
	    int endX, endY, endZ;
	    int isOnSurface;
    } Finger;

    //root level structures
    typedef struct _CommonSettings
    {
	    int rgbWidth;
	    int rgbHeight;
	    int depthWidth;
	    int depthHeight;
	    int maxDepthValue;
	    KinectIntrinsicParameters kinectIntrinsicParameters;
    } CommonSettings;

    typedef struct _DynamicParameters
    {
	    OmniTouchParameters omniTouchParam;
    } DynamicParameters;

    typedef struct _FingerDetectionResults
    {
	    int error;	//0 for no error
	    int fingerCount;
	    Finger fingers[ISE_MAX_FINGER_NUM];

    } FingerDetectionResults;
}

#endif
