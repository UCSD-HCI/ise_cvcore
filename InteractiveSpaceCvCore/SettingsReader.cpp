#include "SettingsReader.h"
#include <stdio.h>
#include <cxcore.h>

int loadCommonSettings(const char* pathPrefix, ise::CommonSettings* settings)
{
	settings->rgbWidth = 640;
	settings->rgbHeight = 480;
	settings->depthWidth = 640;
	settings->depthHeight = 480;
	settings->maxDepthValue = 65535;

	char path[255];
	sprintf(path, "%s.yml", pathPrefix);

	CvFileStorage* fs = cvOpenFileStorage(path, 0, CV_STORAGE_READ);
	CvFileNode* kinectNode = cvGetFileNodeByName(fs, NULL, "KinectIntrinsicParameters");

	settings->kinectIntrinsicParameters.realWorldXToZ = cvReadRealByName(fs, kinectNode, "realWorldXToZ");
	settings->kinectIntrinsicParameters.realWorldYToZ = cvReadRealByName(fs, kinectNode, "realWorldYToZ");
	settings->kinectIntrinsicParameters.depthSlope = cvReadRealByName(fs, kinectNode, "depthSlope");
	settings->kinectIntrinsicParameters.depthIntercept = cvReadRealByName(fs, kinectNode, "depthIntercept");

	cvReleaseFileStorage(&fs);

	return 0;	//TOOD: report error
}

int loadDynamicParameters(const char* pathPrefix, ise::DynamicParameters* params)
{
	char path[255];
	sprintf(path, "%s.yml", pathPrefix);

	CvFileStorage* fs = cvOpenFileStorage(path, 0, CV_STORAGE_READ);
	CvFileNode* omniNode = cvGetFileNodeByName(fs, NULL, "OmniTouchParameters");

	params->omniTouchParam.stripMaxBlankPixel = cvReadIntByName(fs, omniNode, "stripMaxBlankPixel");
	params->omniTouchParam.fingerMinPixelLength = cvReadIntByName(fs, omniNode, "fingerMinPixelLength");
	params->omniTouchParam.fingerToHandOffset = cvReadIntByName(fs, omniNode, "fingerToHandOffset");
	params->omniTouchParam.clickFloodArea = cvReadIntByName(fs, omniNode, "clickFloodArea");
	params->omniTouchParam.fingerWidthMin = cvReadRealByName(fs, omniNode, "fingerWidthMin");
	params->omniTouchParam.fingerWidthMax = cvReadRealByName(fs, omniNode, "fingerWidthMax");
	params->omniTouchParam.fingerLengthMin = cvReadRealByName(fs, omniNode, "fingerLengthMin");
	params->omniTouchParam.fingerLengthMax = cvReadRealByName(fs, omniNode, "fingerLengthMax");
	params->omniTouchParam.fingerRisingThreshold = cvReadRealByName(fs, omniNode, "fingerRisingThreshold");
	params->omniTouchParam.fingerFallingThreshold = cvReadRealByName(fs, omniNode, "fingerFallingThreshold");
	params->omniTouchParam.clickFloodMaxGrad = cvReadRealByName(fs, omniNode, "clickFloodMaxGrad");

	cvReleaseFileStorage(&fs);

	return 0;	//TODO: report error
}