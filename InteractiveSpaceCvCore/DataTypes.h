#pragma once
#ifndef ISE_DATA_TYPES_H
#define ISE_DATA_TYPES_H

typedef unsigned char uchar;
typedef unsigned short ushort;

//sub-structures
struct IseOmniTouchParameters;

typedef struct _IseKinectIntrinsicParameters
{
	
} IseKinectIntrinsicParameters;

//root level structures
typedef struct _IseCommonSettings
{
	int rgbWidth;
	int rgbHeight;
	int depthWidth;
	int depthHeight;
	IseKinectIntrinsicParameters kinectIntrinsicParameters;
} IseCommonSettings;

struct IseDynamicParameters;

typedef struct _IseRgbFrame
{
	int width;
	int height;
	int dataBytes;	//in bytes

	// data owner means the data is only accessible by this frame struct; 
	// isDataOnwer == 0 means the data is created and managed by some other objects, and thus you should not delete it 
	// Warning: Consider carefully when you assign an IseRgbFrame to another. i.e., either copy pointer only and set isDataOnwer = 0, or explictly copy data and set isDataOwner = 1, 
	int isDataOwner; 

	uchar* data;	//RGB888
} IseRgbFrame;

typedef struct _IseDepthFrame
{
	int width;
	int height;
	int dataBytes;	//in bytes

	// data owner means the data is only accessible by this frame struct; 
	// isDataOnwer == 0 means the data is created and managed by some other objects, and thus you should not delete it 
	// Warning: Consider carefully when you assign an IseDepthFrame to another. i.e., either copy pointer only and set isDataOnwer = 0, or explictly copy data and set isDataOwner = 1, 
	int isDataOwner;

	ushort* data;	//16-bit depth per pixel
} IseDepthFrame;

struct IseDetectionResults;

#endif
