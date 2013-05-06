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

const double Detector::MIN_FINGER_COLOR_PDF = 1e-4;

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, const cv::Mat& depthToColorCoordFrame, cv::Mat& debugFrame, cv::Mat& debugFrame2) :
    //initialization list
    //settings
    _settings(settings), 
    
    //external images
    _rgbFrame(rgbFrame), 
    _depthFrame(depthFrame), 
    _depthToColorCoordFrame(depthToColorCoordFrame), 
    _debugFrame(debugFrame),
    _debugFrame2(debugFrame2),

    //host images
    _rgbPdfFrame(settings.rgbHeight, settings.rgbWidth, CV_32F),

    //host images, transposed
    _transposedDepthFrame(settings.depthWidth, settings.depthHeight, CV_16U),
    _transposedDebugFrame(settings.depthWidth, settings.depthHeight, CV_8UC3),

    //gpu images
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),                               
    _rgbLabFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32FC3),
    _rgbPdfFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32F),
    _depthFrameGpu(settings.depthHeight, settings.depthWidth, CV_16U),
    _sobelFrameGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _sobelFrameBufferGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _debugFrameGpu(settings.depthHeight, settings.depthWidth, CV_8UC3),
    _debugSobelEqFrameGpu(settings.depthHeight, settings.depthWidth, CV_8U),
    _debugSobelEqHistGpu(1, 256, CV_32SC1),
    _debugSobelEqBufferGpu(settings.depthHeight, settings.depthWidth, CV_8U),

    //gpu images, transposed
    _transposedDepthFrameGpu(settings.depthWidth, settings.depthHeight, CV_16U),
    _transposedSobelFrameGpu(settings.depthWidth, settings.depthHeight, CV_32F),
    _transposedSobelFrameBufferGpu(settings.depthWidth, settings.depthHeight, CV_32F),
    _transposedDebugFrameGpu(settings.depthWidth, settings.depthHeight, CV_8UC3),
    _transposedDebugSobelEqFrameGpu(settings.depthWidth, settings.depthHeight, CV_8U),
    _transposedDebugSobelEqHistGpu(1, 256, CV_32SC1),
    _transposedDebugSobelEqBufferGpu(settings.depthWidth, settings.depthHeight, CV_8U)
{
	
    //page lock
    gpu::registerPageLocked(_rgbPdfFrame);
    gpu::registerPageLocked(_transposedDepthFrame);

     //init memory for storing fingers
    _stripVisitedFlags = new uchar[(MAX_STRIPS_PER_ROW + 1) * settings.depthHeight];
    _transposedStripVisitedFlags = new uchar[(MAX_STRIPS_PER_ROW + 1) * settings.depthWidth];

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];
    _transposedFloodHitTestVisitedFlag = new uchar[_settings.depthHeight * _settings.depthWidth];
    
	//init vectors
	_fingers.reserve(ISE_MAX_FINGER_NUM);
    _transposedFingers.reserve(ISE_MAX_FINGER_NUM);

    cudaInit();
}

Detector::~Detector()
{
    cudaRelease();
    gpu::unregisterPageLocked(_transposedDepthFrame);
    gpu::unregisterPageLocked(_rgbPdfFrame);
    
    delete [] _stripVisitedFlags;
    delete [] _floodHitTestVisitedFlag;
}


//the algorithm goes here. The detection algorithm runs per frame. The input is rgbFrame and depthFrame. The output is the return value, and also the debug frame.
//have a look at main() to learn how to use this.
FingerDetectionResults Detector::detect()
{
    transpose(_depthFrame, _transposedDepthFrame);

    gpuProcess();

    findFingers<DirDefault>();
    findFingers<DirTransposed>();
    floodHitTest<DirDefault>();
    floodHitTest<DirTransposed>();
    
    transpose(_transposedDebugFrame, _debugFrame2);

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

template <_ImageDirection dir> 
void Detector::findFingers()
{
    std::vector<_OmniTouchFinger>* fingers;
    int width, height, maxStripRowCount;
    uchar* stripVisitedFlags;
    _OmniTouchStripDev* stripsHost;
    Mat debugFrame, depthFrame;

    if (dir == DirTransposed)
    {
        fingers = &_transposedFingers;
        width = _settings.depthHeight;
        height = _settings.depthWidth;
        maxStripRowCount = _transposedMaxStripRowCount;
        stripVisitedFlags = _transposedStripVisitedFlags;
        stripsHost = _transposedStripsHost;
        debugFrame = _transposedDebugFrame;
        depthFrame = _transposedDepthFrame;
    }
    else
    {
        fingers = &_fingers;
        width = _settings.depthWidth;
        height = _settings.depthHeight;
        maxStripRowCount = _maxStripRowCount;
        stripVisitedFlags = _stripVisitedFlags;
        stripsHost = _stripsHost;
        debugFrame = _debugFrame;
        depthFrame = _depthFrame;
    }

    //init visited flags; 
    memset(stripVisitedFlags, 0, height * maxStripRowCount);  

    //init global finger count
    fingers->clear();
	
	for (int row = 0; row < height; row++)
	{
        for (int col = 0; col < stripsHost[row].end - 1; col++)
        {
            int stripOffset = (col + 1) * height + row;

			if (stripVisitedFlags[stripOffset] > 0)
			{
				continue;
			}

            _stripBuffer.clear();
            _stripBuffer.push_back(stripsHost + stripOffset);
            stripVisitedFlags[stripOffset] = 1;

			//search down
			int blankCounter = 0;
			for (int si = row; si < height; si++)   
			{
                _OmniTouchStripDev* currTop = _stripBuffer[_stripBuffer.size() - 1];

				//search strip
				bool stripFound = false;
                
                int searchDownOffset = height + si;

                for (int sj = 0; sj < stripsHost[si].end - 1; ++sj, searchDownOffset += height)
				{
					if (stripVisitedFlags[searchDownOffset])
					{
						continue;
					}

                    _OmniTouchStripDev* candidate = stripsHost + searchDownOffset;

                    if (candidate->end > currTop->start && candidate->start < currTop->end)	//overlap!
					{
                        _stripBuffer.push_back(stripsHost + searchDownOffset);
                        
                        stripVisitedFlags[searchDownOffset] = 1;
						
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
            finger.direction = (dir == DirTransposed ? FingerDirHorizontal : FingerDirVertical);

            //int firstMidCol = (first->start + first->end) / 2;
            finger.tipX = (first->start + first->end) / 2;
            finger.tipY = first->row;
			//int lastMidCol = (last->start + last->end) / 2;
            finger.endX = (last->start + last->end) / 2;
            finger.endY = last->row;

            finger.tipZ = *(ushort*)(depthFrame.ptr((first->row + last->row) / 2) + (finger.tipX + finger.endX) / 2 * sizeof(ushort));
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

                vector<int> widthList;  //TODO: speed optimize? 
                vector<Point> centerPoints;

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
                        uchar* dstPixel = debugFrame.ptr(rowFill) + colFill * 3;
                        
						dstPixel[0] = 255;
						dstPixel[2] = 255;

                        //read color
                        int dx = (dir == DirTransposed ? rowFill : colFill);
                        int dy = (dir == DirTransposed ? colFill : rowFill);

                        const int* mapCoord = (int*)_depthToColorCoordFrame.ptr(dy) + dx * 2;
                        int cx = mapCoord[0];
                        int cy = mapCoord[1];
                        const float* pdfPixel = (float*)_rgbPdfFrame.ptr(cy) + cx;
                        
                        /* //draw pdf values
                        dstPixel[0] = (uchar)(*pdfPixel * 1000.0f * 255.0f + 0.5f);
                        dstPixel[1] = dstPixel[0];
                        dstPixel[2] = dstPixel[0];
                        */

                        colorPdfScore += *pdfPixel;
                        pixelCount++;

                        //draw rgb values
                        /*const uchar* rgbPixel = _rgbFrame.ptr(cy) + cx * 3;
                        memcpy(dstPixel, rgbPixel, 3);*/
					}

                    centerPoints.push_back(Point((rightCol + leftCol) / 2, rowFill)); 
                    widthList.push_back(rightCol - leftCol + 1);
				}

                colorPdfScore /= pixelCount;

                //printf("%f ", colorPdfScore);
                if (colorPdfScore >= MIN_FINGER_COLOR_PDF)  //TODO: avoid hard coding
                {
                    //use median as the width of the finger
                    size_t mid = widthList.size() / 2;
                    nth_element(widthList.begin(), widthList.begin() + mid, widthList.end());
                    finger.width = widthList[mid];

                    //line-fitting to find the angle; TODO: use RANSAC
                    Vec4f line;
                    fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01); //TODO: test performance
                    finger.cosTheta = line[0] / line[1];

                    //filter by angle
                    if (abs(finger.cosTheta) <= 0.8660f)   // -60 ~ +60 //TODO: avoid hard code
                    {
                        //adjust tipX and endX
                        finger.tipX = line[2] + (finger.tipY - line[3]) * line[0] / line[1];
                        finger.endX = line[2] + (finger.endY - line[3]) * line[0] / line[1];

                        fingers->push_back(finger);
                        drawFingerBoundingBox(finger);
                    }
                }     
                else
                {
                    circle(debugFrame, Point(finger.tipX, finger.tipY), 5, Scalar(224,80,1), -1);
                }
                
			} // check length
		
        }   // for each col
	} //for each row

    //printf("\n");
    sort(fingers->begin(), fingers->end());
}

template <_ImageDirection dir> 
void Detector::floodHitTest()
{
    uchar* floodHitTestVisitedFlag;
    std::vector<_OmniTouchFinger>* fingers;
    int width, height;
    Mat debugFrame, depthFrame;

    if (dir == DirTransposed)
    {
        fingers = &_transposedFingers;
        width = _settings.depthHeight;
        height = _settings.depthWidth;
        debugFrame = _transposedDebugFrame;
        depthFrame = _transposedDepthFrame;
        floodHitTestVisitedFlag = _transposedFloodHitTestVisitedFlag;
    }
    else
    {
        fingers = &_fingers;
        width = _settings.depthWidth;
        height = _settings.depthHeight;
        debugFrame = _debugFrame;
        depthFrame = _depthFrame;
        floodHitTestVisitedFlag = _floodHitTestVisitedFlag;
    }


	static const int neighborOffset[3][2] =
	{
		{-1, 0},
		{1, 0},
		{0, -1}
	};

	for (vector<OmniTouchFinger>::iterator it = fingers->begin(); it != fingers->end(); ++it)
	{
		deque<_IntPoint3D> dfsQueue;
		int area = 0;
		memset(floodHitTestVisitedFlag, 0, width * height);

		ushort tipDepth = *ushortValAt(depthFrame, it->tipY, it->tipX);
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

				if (row < 0 || row >= height || col < 0 || col >= width
					|| floodHitTestVisitedFlag[row * width + col] > 0)
				{
					continue;
				}

				ushort neiborDepth = *ushortValAt(depthFrame, row, col);
				if (abs(neiborDepth - centerPoint.z) > _parameters.omniTouchParam.clickFloodMaxGrad)
				{
					continue;					
				}

				p.x = col;
				p.y = row;
				p.z = neiborDepth;
				dfsQueue.push_back(p);
				area++;
				floodHitTestVisitedFlag[row * width + col] = 255;

				uchar* dstPixel = rgb888ValAt(debugFrame, row, col);
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


        circle(debugFrame, Point(it->tipX, it->tipY), 5, Scalar(0, 148, 42), -1);
	}

}

void Detector::drawFingerBoundingBox(const _OmniTouchFinger& finger)
{
    //float dist = sqrt(powf(finger.tipX - finger.endX, 2) + powf(finger.tipY - finger.endY, 2));
    //float sinA = (finger.endY - finger.tipY) / dist;
    //float cosA = (finger.endX - finger.tipX) / dist;
    float cosTheta = finger.cosTheta;
    float sinTheta = sqrt(1 - powf(finger.cosTheta, 2));

    float recWidth = finger.width * sinTheta;
    int dx = (int)(recWidth * sinTheta / 2.f + 0.5f);
    int dy = (int)(recWidth * cosTheta / 2.f + 0.5f);

    //float adjEndX = finger.tipX + (finger.endY - finger.tipY) * cosTheta / sinTheta;

    Point p[4] =
    {
        Point(finger.tipX - dx, finger.tipY + dy),
        Point(finger.tipX + dx, finger.tipY - dy),
        Point(finger.endX + dx, finger.endY - dy),
        Point(finger.endX - dx, finger.endY + dy)
    };
    const Point* pArr[1] = {p};
    int nptsArr[1] = {4};
    
    Mat& debugFrame = (finger.direction == FingerDirHorizontal ? _transposedDebugFrame : _debugFrame);
    cv::polylines(debugFrame, pArr, nptsArr, 1, true, Scalar(255, 128, 0), 3, 8, 0);
}