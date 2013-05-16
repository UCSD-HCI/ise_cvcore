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
#include <omp.h>

using namespace std;
using namespace cv;
using namespace ise;

const double Detector::MIN_FINGER_COLOR_PDF = 1e-4;
const float Detector::MIN_STRIP_OVERLAP = 0.3f;
const float Detector::MIN_FINGER_DIR_OVERLAP = 0.3f;

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

#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0)
        {
            findFingers<DirDefault>();
        }
        else
        {
            findFingers<DirTransposed>();
        }
    }

    combineFingers();

#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0)
        {
            floodHitTest<FloodTestNormal>();
        }
        else
        {
            floodHitTest<FloodTestInversed>();
        }
    }

    decideFingerDirections();

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
		r.fingers[i].isOnSurface = _fingers[i].isTipOnSurface ? 1 : 0;
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

inline float Detector::getSegOverlapPercentage(float min1, float max1, float min2, float max2)
{
    if (min1 >= max2 || min2 >= max1)
    {
        return 0;
    }
    
    float p[4] = {min1, max1, min2, max2};
    sort(p, p + 4);

    return (p[2] - p[1]) / (p[3] - p[0]);
}

template <_ImageDirection dir> 
void Detector::findFingers()
{
    std::vector<_OmniTouchFinger>* fingers;
    int width, height, maxStripRowCount;
    uchar* stripVisitedFlags;
    _OmniTouchStripDev* stripsHost;
    Mat debugFrame, depthFrame;
    std::vector<_OmniTouchStripDev*>* stripBuffer;

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
        stripBuffer = &_transposedStripBuffer;
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
        stripBuffer = &_stripBuffer;
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

            stripBuffer->clear();
            stripBuffer->push_back(stripsHost + stripOffset);
            stripVisitedFlags[stripOffset] = 1;

			//search down
			int blankCounter = 0;
			for (int si = row; si < height; si++)   
			{
                _OmniTouchStripDev* currTop = stripBuffer->at(stripBuffer->size() - 1);

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

                    float overlap = getSegOverlapPercentage(candidate->start, candidate->end, currTop->start, currTop->end);
                    //if (candidate->end > currTop->start && candidate->start < currTop->end)	//overlap!
                    if (overlap > MIN_STRIP_OVERLAP)
					{
                        stripBuffer->push_back(stripsHost + searchDownOffset);
                        
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
            _OmniTouchStripDev* first = stripBuffer->at(0);
            _OmniTouchStripDev* last = stripBuffer->at(stripBuffer->size() - 1);
            
            OmniTouchFinger finger;
            
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
                vector<Point2i> centerPoints;
                vector<Point2i> depthList;

				for (int rowFill = first->row; rowFill <= last->row; rowFill++)
				{
					int leftCol, rightCol;
                    _OmniTouchStripDev* nextBufferItem = stripBuffer->at(bufferPos + 1);

					if (rowFill == nextBufferItem->row)	//find next detected row
					{
                        leftCol = nextBufferItem->start;
                        rightCol = nextBufferItem->end;
                        bufferPos++;
					}
					else	//in blank area, interpolate
					{
                        _OmniTouchStripDev* thisBufferItem = stripBuffer->at(bufferPos);

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
                        const uchar* rgbPixel = _rgbFrame.ptr(cy) + cx * 3;
                        memcpy(dstPixel, rgbPixel, 3);
                        
					}

                    int centerCol = (rightCol + leftCol) / 2;
                    ushort centerDepth = *((ushort*)depthFrame.ptr(rowFill) + centerCol);
                    depthList.push_back(Point2i(rowFill, centerDepth));
                    centerPoints.push_back(Point2i(centerCol, rowFill)); 
                    widthList.push_back(rightCol - leftCol + 1);
				}

                colorPdfScore /= pixelCount;

                //printf("%f ", colorPdfScore);
                if (colorPdfScore >= MIN_FINGER_COLOR_PDF)  //TODO: avoid hard coding
                {
                    //line-fitting to find the angle; TODO: use RANSAC
                    Vec4f line;
                    fitLine(centerPoints, line, CV_DIST_HUBER, 0, 0.01, 0.01); //TODO: test performance
                    
                    if (line[1] >= 0)
                    {
                        finger.dx = line[0];
                        finger.dy = line[1];
                    }
                    else
                    {
                        finger.dx = -line[0];
                        finger.dy = -line[1];
                    }

                    //filter by angle
                    if (abs(finger.dx) <= 0.8660f)   // -60 ~ +60 //TODO: avoid hard code
                    {
                        //use median as the width of the finger
                        size_t mid = widthList.size() / 2;
                        nth_element(widthList.begin(), widthList.begin() + mid, widthList.end());
                        finger.width = widthList[mid] * finger.dy;

                        //adjust tipX and endX
                        finger.tipX = line[2] + (finger.tipY - line[3]) * finger.dx / finger.dy;
                        finger.endX = line[2] + (finger.endY - line[3]) * finger.dx / finger.dy;
                        
                        //adjust depth
                        Vec4f depthLine;
                        fitLine(depthList, depthLine, CV_DIST_HUBER, 0, 0.01, 0.01);
                        finger.tipZ = depthLine[3] + (finger.tipY - depthLine[2]) * depthLine[1] / depthLine[0];
                        finger.endZ = depthLine[3] + (finger.endY - depthLine[2]) * depthLine[1] / depthLine[0];

                        if (dir == DirTransposed)
                        {
                            swap(finger.tipX, finger.tipY);
                            swap(finger.endX, finger.endY);
                            swap(finger.dx, finger.dy);
                            finger.direction = FingerDirLeft;
                        }
                        else
                        {
                            finger.direction = FingerDirUp;
                        }

                        fingers->push_back(finger);
                        drawFingerBoundingBox<dir>(finger);
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

template <_FloodTestDirection dir>
void Detector::floodHitTest()
{
    uchar* floodHitTestVisitedFlag;
    std::vector<_OmniTouchFinger>* fingers;
    int width, height;
    Mat debugFrame, depthFrame;

    fingers = &_fingers;
    width = _settings.depthWidth;
    height = _settings.depthHeight;
    debugFrame = _debugFrame;
    depthFrame = _depthFrame;
    floodHitTestVisitedFlag = _floodHitTestVisitedFlag;

	static const int neighborOffset[4][3][2] =
	{
        { {-1, 0}, {1, 0}, {0, -1} },    //up
        { {0, -1}, {0, 1}, {-1, 0} },    //left
        { {-1, 0}, {1, 0}, {0, 1} },     //down
        { {0, -1}, {0, 1}, {1, 0} }      //right
	};

	for (vector<OmniTouchFinger>::iterator it = fingers->begin(); it != fingers->end(); ++it)
	{
		deque<_IntPoint3D> dfsQueue;
		int area = 0;
		memset(floodHitTestVisitedFlag, 0, width * height);

		_IntPoint3D p;

        if (dir == FloodTestInversed)
        {
            p.x = it->endX;
		    p.y = it->endY;
		    p.z = it->endZ;
        }
        else
        {
		    p.x = it->tipX;
		    p.y = it->tipY;
		    p.z = it->tipZ;
        }
		dfsQueue.push_back(p);

		while(!dfsQueue.empty())
		{
			_IntPoint3D centerPoint = dfsQueue.front();
			dfsQueue.pop_front();

			for (int i = 0; i < 3; i++)
			{
                FingerDirection testDir;
                
                if (dir == FloodTestInversed)
                {
                    testDir = (it->direction == FingerDirUp ? FingerDirDown : FingerDirRight);
                }
                else
                {
                    testDir = it->direction;
                }

                int row = centerPoint.y + neighborOffset[testDir][i][1];
                int col = centerPoint.x + neighborOffset[testDir][i][0];

				if (row < 0 || row >= height || col < 0 || col >= width
					|| floodHitTestVisitedFlag[row * width + col] > 0)
				{
					continue;
				}

                ushort neiborDepth = *(depthFrame.ptr<ushort>(row) + col);
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

				uchar* dstPixel = debugFrame.ptr(row) + col * 3;
				dstPixel[0] = 255;
				dstPixel[1] = 255;
				dstPixel[2] = 0;
			}

			if (area >= _parameters.omniTouchParam.clickFloodArea)
			{
                if (dir == FloodTestNormal)
                {
                    it->isTipOnSurface = true;
                }
                else
                {
                    it->isEndOnSurface = true;
                }

				break;
			}
		}


        //circle(debugFrame, Point(it->tipX, it->tipY), 5, Scalar(0, 148, 42), -1);
	}

}

template <_ImageDirection dir>
void Detector::drawFingerBoundingBox(const _OmniTouchFinger& finger)
{
    //float dist = sqrt(powf(finger.tipX - finger.endX, 2) + powf(finger.tipY - finger.endY, 2));
    //float sinA = (finger.endY - finger.tipY) / dist;
    //float cosA = (finger.endX - finger.tipX) / dist;

    float cosTheta = finger.dx;
    float sinTheta = finger.dy; //sqrt(1 - powf(finger.cosTheta, 2));
    //(dx, dy) is normalized

    int dx = (int)(finger.width * sinTheta / 2.f + 0.5f);
    int dy = (int)(finger.width * cosTheta / 2.f + 0.5f);

    Point p[4] =
    {
        Point(finger.tipX - dx, finger.tipY + dy),
        Point(finger.tipX + dx, finger.tipY - dy),
        Point(finger.endX + dx, finger.endY - dy),
        Point(finger.endX - dx, finger.endY + dy)
    };
    Point pTr[4] = 
    {
        Point(finger.tipY + dy, finger.tipX - dx),
        Point(finger.tipY - dy, finger.tipX + dx),
        Point(finger.endY - dy, finger.endX + dx),
        Point(finger.endY + dy, finger.endX - dx)
    };
    const Point* pArr[1] = {p};
    const Point* pTrArr[1] = {pTr};
    int nptsArr[1] = {4};
    
    if (dir == DirTransposed)
    {
        //draw on transposed
        cv::polylines(_transposedDebugFrame, pTrArr, nptsArr, 1, true, Scalar(8, 111, 161), 2);

        //draw on origin
        cv::polylines(_debugFrame, pArr, nptsArr, 1, true, Scalar(8, 111, 161), 2);
    }
    else
    {
        //draw on origin
        cv::polylines(_debugFrame, pArr, nptsArr, 1, true, Scalar(255, 137, 0), 2);

        //draw on transposed
        cv::polylines(_transposedDebugFrame, pTrArr, nptsArr, 1, true, Scalar(255, 137, 0), 2);
    }
}

float Detector::pointToLineDistance(float x0, float y0, float dx, float dy, float x, float y)
{
    return abs(dy * (x - x0) - dx * (y - y0));
}

float Detector::fingerOverlapPercentage(const _OmniTouchFinger& f1, const _OmniTouchFinger& f2, Mat& debugFrame)
{
    float height1 = sqrt(powf(f1.endY - f1.tipY, 2) + powf(f1.endX - f1.tipX, 2));
    float height2 = sqrt(powf(f2.endY - f2.tipY, 2) + powf(f2.endX - f2.tipX, 2));

    float area1 = height1 * f1.width;
    float area2 = height2 * f2.width;

    const _OmniTouchFinger* pf1;
    const _OmniTouchFinger* pf2;

    if (area1 > area2)
    {
        pf1 = &f2;
        pf2 = &f1;
        swap(height1, height2);
        swap(area1, area2);
    }
    else
    {
        pf1 = &f1;
        pf2 = &f2;
    }
    //now 1 is smaller, 2 is bigger
    
    //rotate to rectify the bigger one
    float cosTheta = pf2->dy;
    float sinTheta = pf2->dx;
    float tipX2 = pf2->tipX * cosTheta - pf2->tipY * sinTheta;
    float tipY2 = pf2->tipX * sinTheta + pf2->tipY * cosTheta;
    float tipX1 = pf1->tipX * cosTheta - pf1->tipY * sinTheta;
    float tipY1 = pf1->tipX * sinTheta + pf1->tipY * cosTheta;
    float dx1 = pf1->dx * cosTheta - pf1->dy * sinTheta;
    float dy1 = pf1->dx * sinTheta + pf1->dy * cosTheta;
    //now dx1, dy1 still normalized
    
    float xMin = tipX2 - pf2->width / 2;
    float xMax = tipX2 + pf2->width / 2;
    float yMin = tipY2;
    float yMax = tipY2 + height2;

    int nOverlap = 0;
    //try sample points TODO: better method     //TODO: omp here?
    for (int hi = 0; hi < height1; hi++)
    {
        for (int wi = -pf1->width / 2; wi < pf1->width / 2; wi++)
        {
            float x = tipX1 + dx1 * hi + dy1 * wi;
            float y = tipY1 + dy1 * hi - dx1 * wi;
            //float x = tipX1 + wi * dy1 - hi * dx1;
            //float y = tipY1 + wi * dx1 + hi * dy1;
            if (x >= xMin && x <= xMax && y >= yMin && y <= yMax)
            {
                nOverlap++;

                //debug
                int xOrigin = (int)(x * cosTheta + y * sinTheta + 0.5f);
                int yOrigin = (int)(-x * sinTheta + y * cosTheta + 0.5f);

                if (xOrigin >= 0 && xOrigin < 640 && yOrigin >= 0 && yOrigin < 480)
                {
                    uchar* px = debugFrame.ptr(yOrigin) + xOrigin * 3;
                    px[0] = 255;
                    px[1] = 255;
                    px[2] = 0;
                }
            }
        }
    }

    return nOverlap / (area1 + area2 - nOverlap);
}

void Detector::combineFingers()
{
    //find overlaps
    for (int i = 0; i < _fingers.size(); i++)
    {
        for (int j = 0; j < _transposedFingers.size(); j++)
        {
            float r = fingerOverlapPercentage(_fingers[i], _transposedFingers[j], _debugFrame);
            if (r >= MIN_FINGER_DIR_OVERLAP)
            {
                if (abs(_fingers[i].dx) <= abs(_transposedFingers[j].dx))
                {
                    //choose i, erase j
                    _transposedFingers.erase(_transposedFingers.begin() + j);
                    j--;
                }
                else
                {
                    //choose j, replace i
                    _fingers[i] = _transposedFingers[j];
                }
            }
        }
    }

    _fingers.insert(_fingers.end(), _transposedFingers.begin(), _transposedFingers.end());
}

//make (dx, dy) points to the direction from finger end to tip.
void Detector::amendFingerDirection(_OmniTouchFinger& f, bool flip)
{
    if (flip)
    {
        f.direction = (f.direction == FingerDirLeft ? FingerDirRight : FingerDirDown);
        swap(f.tipX, f.endX);
        swap(f.tipY, f.endY);
        swap(f.tipZ, f.endZ);
        swap(f.isTipOnSurface, f.isEndOnSurface);
    }

    if (!flip)
    {
        //in findFingers we made dy >= 0, which is from tip to end
        f.dx = -f.dx;   
        f.dy = -f.dy; 
    }
}

bool isDanglingFinger(const _OmniTouchFinger& f)
{
    return (!f.isTipOnSurface && !f.isEndOnSurface);
}

void Detector::decideFingerDirections()
{
    //remove cases that neigher tip nor end is on surface, which is very possibly false positive
   _fingers.erase(remove_if(_fingers.begin(), _fingers.end(), isDanglingFinger), _fingers.end());

    for (int i = 0; i < _fingers.size(); i++)
    {
        OmniTouchFinger& f = _fingers[i];

        if (f.isTipOnSurface && f.isEndOnSurface)
        {
            //on surface
            //TODO important: should compare on tabletop coordinate
            amendFingerDirection(f, f.direction == FingerDirLeft && f.tipZ < f.endZ);   
        }
        else if (!f.isTipOnSurface && !f.isEndOnSurface)
        {
            //theoretically not possible. happens in real case. regard it as non-finger so as to reduce false positive
            assert(0);  //should be removed above
        }
        else
        {
            amendFingerDirection(f, f.isTipOnSurface);
        }
        
        circle(_debugFrame, Point(f.tipX, f.tipY), 5, Scalar(0, 148, 42), -1);
        line(_debugFrame, Point(f.tipX, f.tipY), Point(f.tipX + f.dx * 20, f.tipY + f.dy * 20), Scalar(0, 148, 42));
    }
}
