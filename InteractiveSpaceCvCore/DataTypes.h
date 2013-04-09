#pragma once
#ifndef ISE_DATA_TYPES_H
#define ISE_DATA_TYPES_H

#ifndef NULL
#define NULL 0
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;

//sub-structures
typedef struct _IseOmniTouchParameters
{
} IseOmniTouchParameters;

typedef struct _IseKinectIntrinsicParameters
{
	
} IseKinectIntrinsicParameters;

typedef struct _IseImageHeader
{
	int width;
	int height;
	int dataBytes; //in bytes

	// data owner means the data is only accessible by this frame struct; 
	// isDataOnwer == 0 means the data is created and managed by some other objects, and thus you should not delete it 
	// Warning: Consider carefully when you assign an IseRgbFrame to another. i.e., either copy pointer only and set isDataOnwer = 0, or explictly copy data and set isDataOwner = 1, 
	int isDataOwner; 
} IseImageHeader;

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

} IseFingerDetectionResults;


#endif
