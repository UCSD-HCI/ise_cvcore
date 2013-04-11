#include "Detector.h"
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#include <vector>
#include <deque>
#include <algorithm>

//for debug
//#include "DebugUtils.h"
//#include <math.h>

using namespace std;

//#define byteValAt(imgPtr, row, col) ((byte*)((imgPtr)->imageData + (row) * (imgPtr)->widthStep + col))
#define ushortValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
#define intValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
#define rgb888ValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width * 3 + (col) * 3)

typedef enum
{
	StripSmooth,
	StripRising,
	StripMidSmooth,
	StripFalling
} StripState;

typedef struct _OmniTouchStrip
{
	int row;
	int leftCol, rightCol;
	bool visited;
	struct _OmniTouchStrip(int row, int leftCol, int rightCol) : row(row), leftCol(leftCol), rightCol(rightCol), visited(false) { }
} OmniTouchStrip;

typedef struct _OmniTouchFinger
{
	int tipX, tipY, tipZ;
	int endX, endY, endZ;
	struct _OmniTouchFinger(int tipX, int tipY, int tipZ, int endX, int endY, int endZ) : tipX(tipX), tipY(tipY), tipZ(tipZ), endX(endX), endY(endY), endZ(endZ), isOnSurface(false) { }
	bool operator<(const _OmniTouchFinger& ref) const { return endY - tipY > ref.endY - ref.tipY; }	//sort more to less
	bool isOnSurface;
} OmniTouchFinger;

typedef struct __IseIntPoint3D
{
	int x, y, z;
} _IseIntPoint3D;	//for hit test queue

static IseCommonSettings _settings;
static IseDynamicParameters _parameters;
static IseSobelFrame _sobelFrame;
static int _maxHistogramSize;
static int* _histogram;
static uchar* _floodHitTestVisitedFlag;

void _iseHistEqualize(const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame)
{
	if (!debugFrame)
	{
		return;
	}

	int nDepth = _settings.maxDepthValue + 1;
	int* depthHistogram = (int*)malloc(nDepth * sizeof(int));
	memset(depthHistogram, 0, nDepth * sizeof(int));

    int points = 0;
	for (int y = 0; y < _settings.depthHeight; ++y)
	{
		ushort* ptr = (ushort*)(depthFrame->data + y * _settings.depthWidth);
		for (int x = 0; x < _settings.depthWidth; ++x, ++ptr)
		{
			if (*ptr != 0)
			{
				depthHistogram[*ptr]++;
				points++;
			}
		}
	}
	
	//inclusive scan
	for (int i = 1; i < nDepth; i++)
    {
        depthHistogram[i] += depthHistogram[i - 1];
    }

    if (points > 0)
    {
        for (int i = 1; i < nDepth; i++)
        {
            depthHistogram[i] = (int)(256 * (1.0f - (depthHistogram[i] / (double)points)));
		}
	}

	for (int y = 0; y < _settings.depthHeight; ++y)
	{
		ushort* srcPtr = (ushort*)(depthFrame->data + y * _settings.depthWidth);
		uchar* dstPtr = (uchar*)(debugFrame->data + y * _settings.depthWidth * 3);
		for (int x = 0; x < _settings.depthWidth; ++x, ++srcPtr, dstPtr += 3)
		{
			dstPtr[0] = depthHistogram[*srcPtr];
			dstPtr[1] = depthHistogram[*srcPtr];
			dstPtr[2] = depthHistogram[*srcPtr];
		}
	}

	free(depthHistogram);
}

void _iseDetectorSobel(const IseDepthFrame* src, IseSobelFrame* dst)
{
	static const double tpl[5][5] =	
	{
		{1, 2, 0, -2, -1},
		{4, 8, 0, -8, -4},
		{6, 12, 0, -12, -6},
		{4, 8, 0, -8, -4},
		{1, 2, 0, -2, -1}
	};

	const int tpl_offset = 2;

	for (int i = 0; i < src->header.height; i++)
	{
		for (int j = 0; j < src->header.width; j++)
		{
			double depthH = 0;
			for (int ti = 0; ti < 5; ti++)
			{
				int neighbor_row = i + ti - tpl_offset;
				if(neighbor_row < 0 || neighbor_row >= src->header.height)
					continue;

				for (int tj = 0; tj < 5; tj++)
				{
					int neighbor_col = j + tj - tpl_offset;
					if(neighbor_col < 0 || neighbor_col >= src->header.width)
						continue;

					ushort srcDepthVal = *ushortValAt(src, neighbor_row, neighbor_col);
					//depthH += tpl[ti][tj] * srcDepthVal;
					depthH += tpl[ti][tj] * (srcDepthVal == 0 ? _settings.maxDepthValue : srcDepthVal);
				}
			}

			*intValAt(dst, i, j) = (int)(depthH + 0.5);
		}
	}
}

void _iseDetectorConvertProjectiveToRealWorld(int x, int y, int depth, double* rx, double* ry, double* rz)
{
	*rx = (x / (double)_settings.depthWidth - 0.5) * depth * _settings.kinectIntrinsicParameters.realWorldXToZ;
	*ry = (0.5 - y / (double)_settings.depthHeight) * depth * _settings.kinectIntrinsicParameters.realWorldYToZ;
	*rz = depth / 100.0 * _settings.kinectIntrinsicParameters.depthSlope + _settings.kinectIntrinsicParameters.depthIntercept;
}

double _iseDetectorGetSquaredDistanceInRealWorld(int x1, int y1, int depth1, int x2, int y2, int depth2)
{
	double rx1, ry1, rz1, rx2, ry2, rz2;

	_iseDetectorConvertProjectiveToRealWorld(x1, y1, depth1, &rx1, &ry1, &rz1);
	_iseDetectorConvertProjectiveToRealWorld(x2, y2, depth2, &rx2, &ry2, &rz2);

	return ((rx1 - rx2) * (rx1 - rx2) + (ry1 - ry2) * (ry1 - ry2) + (rz1 - rz2) * (rz1 - rz2));
}

void _iseDetectorFindStrips(const IseDepthFrame* depthPtr, const IseSobelFrame* sobelPtr, IseRgbFrame* debugPtr, vector<vector<OmniTouchStrip> >* strips)
{
	for (int i = 0; i < _settings.depthHeight; i++)
	{
		strips->push_back(vector<OmniTouchStrip>());

		StripState state = StripSmooth;
		int partialMin, partialMax;
		int partialMinPos, partialMaxPos;
		for (int j = 0; j < _settings.depthWidth; j++)
		{
			int currVal = *intValAt(sobelPtr, i, j);

			switch(state)
			{
			case StripSmooth:	//TODO: smooth
				if (currVal > _parameters.omniTouchParam.fingerRisingThreshold)
				{
					partialMax = currVal;
					partialMaxPos = j;
					state = StripRising;
				}
				break;

			case StripRising:
				if (currVal > _parameters.omniTouchParam.fingerRisingThreshold)
				{
					if (currVal > partialMax)
					{
						partialMax = currVal;
						partialMaxPos = j;
					}
				}
				else 
				{
					state = StripMidSmooth;
				}
				break;

			case StripMidSmooth:
				if (currVal < -_parameters.omniTouchParam.fingerFallingThreshold)
				{
					partialMin = currVal;
					partialMinPos = j;
					state = StripFalling;
				}
				else if (currVal > _parameters.omniTouchParam.fingerRisingThreshold)
				{
					//previous trial faied, start over
					partialMax = currVal;
					partialMaxPos = j;
					state = StripRising;
				}
				break;

			case StripFalling:
				if (currVal < -_parameters.omniTouchParam.fingerFallingThreshold)
				{
					if (currVal < partialMin)
					{
						partialMin = currVal;
						partialMinPos = j;
					}
				}
				else
				{
					ushort depth = *ushortValAt(depthPtr, i, (partialMaxPos + partialMinPos) / 2);	//use the middle point of the strip to measure depth, assuming it is the center of the finger

					double distSquared = _iseDetectorGetSquaredDistanceInRealWorld(
						partialMaxPos, i, depth,
						partialMinPos, i, depth);

					if (distSquared >= _parameters.omniTouchParam.fingerWidthMin * _parameters.omniTouchParam.fingerWidthMin 
						&& distSquared <= _parameters.omniTouchParam.fingerWidthMax * _parameters.omniTouchParam.fingerWidthMax)
					{
						//DEBUG("dist (" << (partialMaxPos + partialMinPos) / 2 << ", " << i << ", " << depth << "): " << sqrt(distSquared));
						for (int tj = partialMaxPos; tj <= partialMinPos; tj++)
						{
							//bufferPixel(tmpPixelBuffer, i, tj)[0] = 0;
							rgb888ValAt(debugPtr, i, tj)[1] = 255;
							//bufferPixel(tmpPixelBuffer, i, tj)[2] = 0;
						}
						(*strips)[i].push_back(OmniTouchStrip(i, partialMaxPos, partialMinPos));
						
						partialMax = currVal;
						partialMaxPos = j;
					}

					state = StripSmooth;
				}
				break;
			}
		}
	}
}

void _iseDetectorFindFingers(const IseDepthFrame* depthPtr, IseRgbFrame* debugPtr, 
	/*in*/ vector<vector<OmniTouchStrip> >* strips, 
	/*out*/ vector<OmniTouchFinger>* fingers)
{
	vector<OmniTouchStrip*> stripBuffer;	//used to fill back

	for (int i = 0; i < _settings.depthHeight; i++)
	{
		for (vector<OmniTouchStrip>::iterator it = (*strips)[i].begin(); it != (*strips)[i].end(); ++it)
		{
			if (it->visited)
			{
				continue;
			}

			stripBuffer.clear();
			stripBuffer.push_back(&(*it));
			it->visited = true;

			//search down
			int blankCounter = 0;
			for (int si = i; si < _settings.depthHeight; si++)
			{
				OmniTouchStrip* currTop = stripBuffer[stripBuffer.size() - 1];

				//search strip
				bool stripFound = false;
				for (vector<OmniTouchStrip>::iterator sIt = (*strips)[si].begin(); sIt != (*strips)[si].end(); ++sIt)
				{
					if (sIt->visited)
					{
						continue;
					}

					if (sIt->rightCol > currTop->leftCol && sIt->leftCol < currTop->rightCol)	//overlap!
					{
						stripBuffer.push_back(&(*sIt));
						sIt->visited = true;
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
			OmniTouchStrip* first = stripBuffer[0];
			int firstMidCol = (first->leftCol + first->rightCol) / 2;
			OmniTouchStrip* last = stripBuffer[stripBuffer.size() - 1];
			int lastMidCol = (last->leftCol + last->rightCol) / 2;

			ushort depth = *ushortValAt(depthPtr, (first->row + last->row) / 2, (firstMidCol + lastMidCol) / 2);	//just a try
						
			double lengthSquared = _iseDetectorGetSquaredDistanceInRealWorld(
				firstMidCol, first->row, depth, // *srcDepth(first->row, firstMidCol),
				lastMidCol, last->row, depth //*srcDepth(last->row, lastMidCol),
				);
			int pixelLength = last->row - first->row +1;
			
			if (pixelLength >= _parameters.omniTouchParam.fingerMinPixelLength 
				&& lengthSquared >= _parameters.omniTouchParam.fingerLengthMin * _parameters.omniTouchParam.fingerLengthMin 
				&& lengthSquared <= _parameters.omniTouchParam.fingerLengthMax * _parameters.omniTouchParam.fingerLengthMax)	//finger!
			{
				//fill back
				int bufferPos = -1;
				for (int row = first->row; row <= last->row; row++)
				{
					int leftCol, rightCol;
					if (row == stripBuffer[bufferPos + 1]->row)	//find next detected row
					{
						bufferPos++;
						leftCol = stripBuffer[bufferPos]->leftCol;
						rightCol = stripBuffer[bufferPos]->rightCol;
					}
					else	//in blank area, interpolate
					{
						double ratio = (double)(row - stripBuffer[bufferPos]->row) / (double)(stripBuffer[bufferPos + 1]->row - stripBuffer[bufferPos]->row);
						leftCol = (int)(stripBuffer[bufferPos]->leftCol + (stripBuffer[bufferPos + 1]->leftCol - stripBuffer[bufferPos]->leftCol) * ratio + 0.5);
						rightCol = (int)(stripBuffer[bufferPos]->rightCol + (stripBuffer[bufferPos + 1]->rightCol - stripBuffer[bufferPos]->rightCol) * ratio + 0.5);
					}

					for (int col = leftCol; col <= rightCol; col++)
					{
						uchar* dstPixel = rgb888ValAt(debugPtr, row, col);
						dstPixel[0] = 255;
						//bufferPixel(tmpPixelBuffer, row, col)[1] = 255;
						dstPixel[2] = 255;
					}
				}

				fingers->push_back(OmniTouchFinger(firstMidCol, first->row, depth, lastMidCol, last->row, depth));	//TODO: depth?
			}
		}
	}

	sort(fingers->begin(), fingers->end());
}

void _iseDetectorFloodHitTest(const IseDepthFrame* depthPtr, IseRgbFrame* debugPtr, /*in & out*/ vector<OmniTouchFinger>* fingers)
{
	int neighborOffset[3][2] =
	{
		{-1, 0},
		{1, 0},
		{0, -1}
	};

	for (vector<OmniTouchFinger>::iterator it = fingers->begin(); it != fingers->end(); ++it)
	{
		deque<_IseIntPoint3D> dfsQueue;
		int area = 0;
		memset(_floodHitTestVisitedFlag, 0, _settings.depthWidth * _settings.depthHeight);

		ushort tipDepth = *ushortValAt(depthPtr, it->tipY, it->tipX);
		_IseIntPoint3D p;
		p.x = it->tipX;
		p.y = it->tipY;
		p.z = it->tipZ;
		dfsQueue.push_back(p);

		while(!dfsQueue.empty())
		{
			_IseIntPoint3D centerPoint = dfsQueue.front();
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

				ushort neiborDepth = *ushortValAt(depthPtr, row, col);
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

				uchar* dstPixel = rgb888ValAt(debugPtr, row, col);
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
	}

}

void _iseDetectorRefineDebugImage(const IseSobelFrame* sobelPtr, IseRgbFrame* dstPtr)
{
	
	//generate histogram
	int min = 65535, max = 0;
	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		for (int j = 0; j < sobelPtr->header.width; j++)
		{
			int h = (int)abs(*intValAt(sobelPtr, i, j));
			if (h > max) max = h;
			if (h < min) min = h;
		}
	}
	
	int histogramSize = max - min + 1;
	assert(histogramSize < _maxHistogramSize);
	int histogramOffset = min;

	memset(_histogram, 0, histogramSize * sizeof(int));

	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		for (int j = 0; j < sobelPtr->header.width; j++)
		{
			int h = (int)abs(*intValAt(sobelPtr, i, j));
			_histogram[h - histogramOffset]++;
		}
	}

	for (int i = 1; i < histogramSize; i++)
	{
		_histogram[i] += _histogram[i-1];
	}

	int points = sobelPtr->header.width * sobelPtr->header.height;
	for (int i = 0; i < histogramSize; i++)
	{
		_histogram[i] = (int)(256 * ((double)_histogram[i] / (double)points) + 0.5);
	}

	//draw the image
	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		uchar* dstPixel = rgb888ValAt(dstPtr, i, 0);
		for (int j = 0; j < sobelPtr->header.width; j++, dstPixel += 3)
		{
			if (dstPixel[0] == 255 || dstPixel[1] == 255 || dstPixel[2] == 255)
			{
				//leave as is
			} 
			else
			{
				int depth = *intValAt(sobelPtr, i, j);
				if (depth >= 0)
				{
					dstPixel[0] = 0;
					dstPixel[2] = _histogram[depth - histogramOffset];
				}
				else
				{
					dstPixel[0] = _histogram[-depth - histogramOffset];
					dstPixel[2] = 0;
				}
				dstPixel[1] = 0;
			}
		}
	}

}

int iseDetectorInitWithSettings(const IseCommonSettings* settings)
{
	_settings = *settings;
	//on device: upload settings to device memory

	//init sobel
	_sobelFrame.header = iseCreateImageHeader(settings->depthWidth, settings->depthHeight, sizeof(int), 1);
	_sobelFrame.data = (int*)malloc(_sobelFrame.header.dataBytes);

	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
	_histogram = (int*)malloc(_maxHistogramSize * sizeof(int));

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = (uchar*)malloc(settings->depthWidth * settings->depthHeight);

	return 0;
}

int iseDetectorUpdateDynamicParameters(const IseDynamicParameters* parameters)
{
	_parameters = *parameters;
	//on device: upload parameters to device memory

	return 0;
}

IseFingerDetectionResults iseDetectorDetect(const IseRgbFrame* rgbFrame, const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame)
{
	//_iseHistEqualize(depthFrame, debugFrame);

	memset(debugFrame->data, 0, debugFrame->header.dataBytes);	//set debug frame to black

	_iseDetectorSobel(depthFrame, &_sobelFrame);

	vector<vector<OmniTouchStrip> > strips;
	_iseDetectorFindStrips(depthFrame, &_sobelFrame, debugFrame, &strips);

	vector<OmniTouchFinger> fingers;
	_iseDetectorFindFingers(depthFrame, debugFrame, &strips, &fingers);
	_iseDetectorFloodHitTest(depthFrame, debugFrame, &fingers);

	_iseDetectorRefineDebugImage(&_sobelFrame, debugFrame);

	IseFingerDetectionResults r;

	r.error = 0;
	return r;
}

int iseDetectorRelease()
{
	free(_sobelFrame.data);
	_sobelFrame.data = NULL;
	_sobelFrame.header.isDataOwner = 0;

	free(_histogram);
	free(_floodHitTestVisitedFlag);
	return 0;
}