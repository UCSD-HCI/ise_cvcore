#pragma once
#ifndef ISE_DATA_TYPES_H
#define ISE_DATA_TYPES_H

#ifndef NULL
#define NULL 0
#endif

#define ISE_MAX_FINGER_NUM 20

typedef unsigned char uchar;
typedef unsigned short ushort;


//sub-structures
typedef struct _IseOmniTouchParameters
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
} IseOmniTouchParameters;

typedef struct _IseKinectIntrinsicParameters
{
	float realWorldXToZ;
	float realWorldYToZ;
	float depthSlope;
	float depthIntercept;
} IseKinectIntrinsicParameters;

typedef struct _IseImageHeader
{
	int width;
	int height;
	int bytesPerPixel; //in bytes
	int dataBytes; //in bytes

	// data owner means the data is only accessible by this frame struct; 
	// isDataOnwer == 0 means the data is created and managed by some other objects, and thus you should not delete it 
	// Warning: Consider carefully when you assign an IseRgbFrame to another. i.e., either copy pointer only and set isDataOnwer = 0, or explictly copy data and set isDataOwner = 1, 
	int isDataOwner; 
} IseImageHeader;

typedef struct _IseFinger
{
	int tipX, tipY, tipZ;
	int endX, endY, endZ;
	int isOnSurface;
} IseFinger;

//root level structures
typedef struct _IseCommonSettings
{
	int rgbWidth;
	int rgbHeight;
	int depthWidth;
	int depthHeight;
	int maxDepthValue;
	IseKinectIntrinsicParameters kinectIntrinsicParameters;
} IseCommonSettings;

typedef struct _IseDynamicParameters
{
	IseOmniTouchParameters omniTouchParam;
} IseDynamicParameters;

typedef struct _IseRgbFrame
{
	IseImageHeader header;

	uchar* data;	//RGB888
} IseRgbFrame;

typedef struct _IseDepthFrame
{
	IseImageHeader header;

	ushort* data;	//16-bit depth per pixel
} IseDepthFrame;

typedef struct _IseSobelFrame
{
	IseImageHeader header;
	int* data;
} IseSobelFrame;

typedef struct _IseFingerDetectionResults
{
	int error;	//0 for no error
	int fingerCount;
	IseFinger fingers[ISE_MAX_FINGER_NUM];

} IseFingerDetectionResults;


IseImageHeader iseCreateImageHeader(int width, int height, int bytesPerPixel, int isDataOwner = 0);

#endif
