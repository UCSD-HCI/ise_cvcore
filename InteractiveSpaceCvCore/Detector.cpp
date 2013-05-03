#include "Detector.h"
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#include <vector>
#include <deque>
#include <algorithm>

#include <cv.h>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2\gpu\stream_accessor.hpp>

using namespace std;
using namespace cv;
using namespace ise;

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, const cv::Mat& depthToColorCoordFrame, cv::Mat& debugFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _depthToColorCoordFrame(depthToColorCoordFrame), _debugFrame(debugFrame),
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),
    _rgbLabFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32FC3),
    _rgbPdfFrame(settings.rgbHeight, settings.rgbWidth, CV_32F),
    _rgbPdfFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32F),
    _depthFrameGpu(settings.depthHeight, settings.depthWidth, CV_16U),
    _sobelFrameGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _sobelFrameBufferGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _debugFrameGpu(settings.depthHeight, settings.depthWidth, CV_8UC3),
    _debugSobelEqFrameGpu(settings.depthHeight, settings.depthWidth, CV_8U),
    _debugSobelEqHistGpu(1, 256, CV_32SC1),
    _debugSobelEqBufferGpu(settings.depthHeight, settings.depthWidth, CV_8U)
{
	
    //page lock
    gpu::registerPageLocked(_rgbPdfFrame);

     //init memory for storing fingers
    _stripVisitedFlags = new uchar[(MAX_STRIPS_PER_ROW + 1) * settings.depthHeight];

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];
    
	//init vectors
	_fingers.reserve(ISE_MAX_FINGER_NUM);

    cudaInit();
}

Detector::~Detector()
{
    cudaRelease();
    gpu::unregisterPageLocked(_rgbPdfFrame);
    
    delete [] _stripVisitedFlags;
    delete [] _floodHitTestVisitedFlag;
}


//the algorithm goes here. The detection algorithm runs per frame. The input is rgbFrame and depthFrame. The output is the return value, and also the debug frame.
//have a look at main() to learn how to use this.
FingerDetectionResults Detector::detect()
{
    gpuProcess();

    findFingers();
    floodHitTest();
    
	FingerDetectionResults r;

	r.error = 0;
	r.fingerCount = _fingers.size() < ISE_MAX_FINGER_NUM ? (int)_fingers.size() : ISE_MAX_FINGER_NUM;
	for (int i = 0; i < r.fingerCount; i++)
	{
		r.fingers[i].tipX = _fingers[i].tipX;
		r.fingers[i].tipY = _fingers[i].tipY;
		r.fingers[i].tipZ = _fingers[i].tipZ;
		r.fingers[i].endX = _fingers[i].endX;
		r.fingers[i].endY = _fingers[i].endY;
		r.fingers[i].endZ = _fingers[i].endZ;
		r.fingers[i].isOnSurface = _fingers[i].isOnSurface ? 1 : 0;
	}

	return r;
}

void Detector::convertProjectiveToRealWorld(int x, int y, int depth, double& rx, double& ry, double& rz)
{
	rx = (x / (double)_settings.depthWidth - 0.5) * depth * _settings.kinectIntrinsicParameters.realWorldXToZ;
	ry = (0.5 - y / (double)_settings.depthHeight) * depth * _settings.kinectIntrinsicParameters.realWorldYToZ;
	rz = depth / 100.0 * _settings.kinectIntrinsicParameters.depthSlope + _settings.kinectIntrinsicParameters.depthIntercept;
}

double Detector::getSquaredDistanceInRealWorld(int x1, int y1, int depth1, int x2, int y2, int depth2)
{
	double rx1, ry1, rz1, rx2, ry2, rz2;

	convertProjectiveToRealWorld(x1, y1, depth1, rx1, ry1, rz1);
	convertProjectiveToRealWorld(x2, y2, depth2, rx2, ry2, rz2);

	return ((rx1 - rx2) * (rx1 - rx2) + (ry1 - ry2) * (ry1 - ry2) + (rz1 - rz2) * (rz1 - rz2));
}

void Detector::findFingers()
{
    //init visited flags; 
    memset(_stripVisitedFlags, 0, _settings.depthHeight * _maxStripRowCount);

    //init global finger count
    _fingers.clear();
	
	for (int row = 0; row < _settings.depthHeight; row++)
	{
        for (int col = 0; col < _stripsHost[row].end - 1; col++)
        {
            int stripOffset = (col + 1) * _settings.depthHeight + row;

			if (_stripVisitedFlags[stripOffset] > 0)
			{
				continue;
			}

            _stripBuffer.clear();
            _stripBuffer.push_back(_stripsHost + stripOffset);
            _stripVisitedFlags[stripOffset] = 1;

			//search down
			int blankCounter = 0;
			for (int si = row; si < _settings.depthHeight; si++)   
			{
                _OmniTouchStripDev* currTop = _stripBuffer[_stripBuffer.size() - 1];

				//search strip
				bool stripFound = false;
                
                int searchDownOffset = _settings.depthHeight + si;

                for (int sj = 0; sj < _stripsHost[si].end - 1; ++sj, searchDownOffset += _settings.depthHeight)
				{
					if (_stripVisitedFlags[searchDownOffset])
					{
						continue;
					}

                    _OmniTouchStripDev* candidate = _stripsHost + searchDownOffset;

                    if (candidate->end > currTop->start && candidate->start < currTop->end)	//overlap!
					{
                        _stripBuffer.push_back(_stripsHost + searchDownOffset);
                        
                        //Note: race condition happens here. But won't generate incorrect results.
                        _stripVisitedFlags[searchDownOffset] = 1;
						
                        stripFound = true;
						break;
					}
				}

				if (!stripFound) //blank
				{
					blankCounter++;
					if (blankCounter > _parameters.omniTouchParam.stripMaxBlankPixel)
					{
						//Too much blank, give up
						break;
					}
				}
			}

			//check length
			_OmniTouchStripDev* first = _stripBuffer[0];
            _OmniTouchStripDev* last = _stripBuffer[_stripBuffer.size() - 1];
            
            OmniTouchFinger finger;

            //int firstMidCol = (first->start + first->end) / 2;
            finger.tipX = (first->start + first->end) / 2;
            finger.tipY = first->row;
			//int lastMidCol = (last->start + last->end) / 2;
            finger.endX = (last->start + last->end) / 2;
            finger.endY = last->row;

            finger.tipZ = *(ushort*)(_depthFrame.ptr((first->row + last->row) / 2) + (finger.tipX + finger.endX) / 2 * sizeof(ushort));
            finger.endZ = finger.tipZ;
			
            double lengthSquared = getSquaredDistanceInRealWorld(finger.tipX, finger.tipY, finger.tipZ, finger.endX, finger.endY, finger.endZ);
			int pixelLength = finger.endY - finger.tipY + 1;
			
            if (pixelLength >= _parameters.omniTouchParam.fingerMinPixelLength 
				&& lengthSquared >= _parameters.omniTouchParam.fingerLengthMin * _parameters.omniTouchParam.fingerLengthMin 
				&& lengthSquared <= _parameters.omniTouchParam.fingerLengthMax * _parameters.omniTouchParam.fingerLengthMax)	//finger!
			{
				//fill back
				int bufferPos = -1;
                float colorPdfScore = 0;
                int pixelCount = 0;

				for (int rowFill = first->row; rowFill <= last->row; rowFill++)
				{
					int leftCol, rightCol;
                    _OmniTouchStripDev* nextBufferItem = _stripBuffer[bufferPos + 1];

					if (rowFill == nextBufferItem->row)	//find next detected row
					{
                        leftCol = nextBufferItem->start;
                        rightCol = nextBufferItem->end;
                        bufferPos++;
					}
					else	//in blank area, interpolate
					{
                        _OmniTouchStripDev* thisBufferItem = _stripBuffer[bufferPos];

						float ratio = (float)(rowFill - thisBufferItem->row) / (float)(nextBufferItem->row - thisBufferItem->row);
                        leftCol = (int)(thisBufferItem->start + (nextBufferItem->start - thisBufferItem->start) * ratio + 0.5f);
                        rightCol = (int)(thisBufferItem->end + (nextBufferItem->end - thisBufferItem->end) * ratio + 0.5f);
					}

					for (int colFill = leftCol; colFill <= rightCol; colFill++)
					{
                        uchar* dstPixel = _debugFrame.ptr(rowFill) + colFill * 3;
                        
						//dstPixel[0] = 255;
						//dstPixel[2] = 255;

                        //read color
                        const int* mapCoord = (int*)_depthToColorCoordFrame.ptr(rowFill) + colFill * 2;
                        int cx = mapCoord[0];
                        int cy = mapCoord[1];
                        const float* pdfPixel = (float*)_rgbPdfFrame.ptr(cy) + cx;

                        dstPixel[0] = (uchar)(*pdfPixel * 1000.0f * 255.0f + 0.5f);
                        dstPixel[1] = dstPixel[0];
                        dstPixel[2] = dstPixel[0];

                        colorPdfScore += *pdfPixel;
                        pixelCount++;

                        const uchar* rgbPixel = _rgbFrame.ptr(cy) + cx * 3;
                        memcpy(dstPixel, rgbPixel, 3);
					}
				}

                colorPdfScore /= pixelCount;

                //printf("%f ", colorPdfScore);
                if (colorPdfScore >= 1e-4)  //TODO: avoid hard coding
                {
                    _fingers.push_back(finger);
                }     
                else
                {
                    circle(_debugFrame, Point(finger.tipX, finger.tipY), 5, Scalar(224,80,1), -1);
                }
                
			} // check length
		
        }   // for each col
	} //for each row

    //printf("\n");
    sort(_fingers.begin(), _fingers.end());
}


void Detector::floodHitTest()
{

	static const int neighborOffset[3][2] =
	{
		{-1, 0},
		{1, 0},
		{0, -1}
	};

	for (vector<OmniTouchFinger>::iterator it = _fingers.begin(); it != _fingers.end(); ++it)
	{
		deque<_IntPoint3D> dfsQueue;
		int area = 0;
		memset(_floodHitTestVisitedFlag, 0, _settings.depthWidth * _settings.depthHeight);

		ushort tipDepth = *ushortValAt(_depthFrame, it->tipY, it->tipX);
		_IntPoint3D p;
		p.x = it->tipX;
		p.y = it->tipY;
		p.z = it->tipZ;
		dfsQueue.push_back(p);

		while(!dfsQueue.empty())
		{
			_IntPoint3D centerPoint = dfsQueue.front();
			dfsQueue.pop_front();

			for (int i = 0; i < 3; i++)
			{
				int row = centerPoint.y + neighborOffset[i][1];
				int col = centerPoint.x + neighborOffset[i][0];

				if (row < 0 || row >= _settings.depthHeight || col < 0 || col >= _settings.depthWidth
					|| _floodHitTestVisitedFlag[row * _settings.depthWidth + col] > 0)
				{
					continue;
				}

				ushort neiborDepth = *ushortValAt(_depthFrame, row, col);
				if (abs(neiborDepth - centerPoint.z) > _parameters.omniTouchParam.clickFloodMaxGrad)
				{
					continue;					
				}

				p.x = col;
				p.y = row;
				p.z = neiborDepth;
				dfsQueue.push_back(p);
				area++;
				_floodHitTestVisitedFlag[row * _settings.depthWidth + col] = 255;

				uchar* dstPixel = rgb888ValAt(_debugFrame, row, col);
				dstPixel[0] = 255;
				dstPixel[1] = 255;
				dstPixel[2] = 0;
			}

			if (area >= _parameters.omniTouchParam.clickFloodArea)
			{
				it->isOnSurface = true;
				break;
			}
		}


        circle(_debugFrame, Point(it->tipX, it->tipY), 5, Scalar(0, 148, 42), -1);
	}

}

