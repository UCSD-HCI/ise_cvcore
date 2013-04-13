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
#include <cuda.h>
#include <cuda_runtime.h>


//for debug
//#include "DebugUtils.h"
//#include <math.h>

using namespace std;
using namespace cv;
using namespace ise;

//#define byteValAt(imgPtr, row, col) ((byte*)((imgPtr)->imageData + (row) * (imgPtr)->widthStep + col))
//#define ushortValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
//#define intValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
//#define rgb888ValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width * 3 + (col) * 3)
//#define floatValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))

//declare textures
texture<ushort, 2> texDepth;
//texture<float, 2> texSobel;

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, cv::Mat& debugFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _debugFrame(debugFrame),
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),
    _depthFrameGpu(settings.depthHeight, settings.depthWidth, CV_16U),
    _sobelFrame(settings.depthHeight, settings.depthWidth, CV_32F),
    _sobelFrameGpu(settings.depthHeight, settings.depthWidth, CV_32F)
{
	//on device: upload settings to device memory

	//init sobel
    gpu::registerPageLocked(_sobelFrame);

    //bind texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrStepSz(_depthFrameGpu);

    cudaSafeCall(cudaBindTexture2D(NULL, texDepth, ptrStepSz.data, desc, ptrStepSz.cols, ptrStepSz.rows, ptrStepSz.step));
	
	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
	_histogram = new int[_maxHistogramSize];

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];

	//init vectors
	_strips.reserve(_settings.depthHeight);
	_fingers.reserve(ISE_MAX_FINGER_NUM);
}

//update the parameters used by the algorithm
void Detector::updateDynamicParameters(const DynamicParameters& parameters)
{
	_parameters = parameters;
	//on device: upload parameters to device memory

}

//the algorithm goes here. The detection algorithm runs per frame. The input is rgbFrame and depthFrame. The output is the return value, and also the debug frame.
//have a look at main() to learn how to use this.
FingerDetectionResults Detector::detect()
{
	//_iseHistEqualize(depthFrame, debugFrame);

    _debugFrame.setTo(Scalar(0,0,0));	//set debug frame to black, can also done at GPU
    memset(_floodHitTestVisitedFlag, 0, _settings.depthWidth * _settings.depthHeight);

	sobel();
    findStrips();
    findFingers();
    floodHitTest();
    refineDebugImage();
	
	FingerDetectionResults r;

	r.error = 0;
	r.fingerCount = _fingers.size() < ISE_MAX_FINGER_NUM ? _fingers.size() : ISE_MAX_FINGER_NUM;
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

Detector::~Detector()
{
    cudaSafeCall(cudaUnbindTexture(texDepth));
    gpu::unregisterPageLocked(_sobelFrame);

    delete [] _histogram;
    delete [] _floodHitTestVisitedFlag;
}


const ushort* Detector::ushortValAt(const cv::Mat& mat, int row, int col)
{
    assert(mat.type() == CV_16U);
    return (ushort*)(mat.data + row * mat.step + col * sizeof(ushort));
}

float* Detector::floatValAt(cv::Mat& mat, int row, int col)
{
    assert(mat.type() == CV_32F);
    return (float*)(mat.data + row * mat.step + col * sizeof(float));
}

uchar* Detector::rgb888ValAt(cv::Mat& mat, int row, int col)
{
    assert(mat.type() == CV_8UC3);
    return (uchar*)(mat.data + row * mat.step + col * 3);
}

int Detector::divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

void Detector::cudaSafeCall(cudaError_t err)
{
    //TODO: better handler
    if (err != 0)
    {
        printf(cudaGetErrorString(err));
        assert(0); 
    }
}

/*
void _iseHistEqualize(const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame)
{
	assert(0);	//deprecated

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
*/

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

/*__global__ void adjustDepthKernel(gpu::PtrStepSzb ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < ptr.cols && y < ptr.rows)
    {
        ushort v = tex2D(texDepth, x, y);
        if (v == 0)
        {
            ushort* p = (ushort*)(ptr.data + y * ptr.step + x * sizeof(ushort));
            *p = 65535; //TODO: avoid hard coding
        }
    }
}*/

void Detector::sobel()
{
    _depthFrameGpu.upload(_depthFrame);

    /*dim3 grid(1, 1);
    dim3 threads(32, 16);

    grid.x = divUp(_settings.depthWidth, threads.x);
    grid.y = divUp(_settings.depthHeight, threads.y);
    adjustDepthKernel<<<grid, threads>>>(_depthFrameGpu);*/
    

    cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, 5, -1);
    _sobelFrameGpu.download(_sobelFrame);
}

void Detector::findStrips()
{
	_strips.clear();
	for (int i = 0; i < _settings.depthHeight; i++)
	{
		_strips.push_back(vector<OmniTouchStrip>());

		StripState state = StripSmooth;
		int partialMin, partialMax;
		int partialMinPos, partialMaxPos;
		for (int j = 0; j < _settings.depthWidth; j++)
		{
			int currVal = *floatValAt(_sobelFrame, i, j);

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
					ushort depth = *ushortValAt(_depthFrame, i, (partialMaxPos + partialMinPos) / 2);	//use the middle point of the strip to measure depth, assuming it is the center of the finger

					double distSquared = getSquaredDistanceInRealWorld(
						partialMaxPos, i, depth,
						partialMinPos, i, depth);

					if (distSquared >= _parameters.omniTouchParam.fingerWidthMin * _parameters.omniTouchParam.fingerWidthMin 
						&& distSquared <= _parameters.omniTouchParam.fingerWidthMax * _parameters.omniTouchParam.fingerWidthMax)
					{
						//DEBUG("dist (" << (partialMaxPos + partialMinPos) / 2 << ", " << i << ", " << depth << "): " << sqrt(distSquared));
						for (int tj = partialMaxPos; tj <= partialMinPos; tj++)
						{
							//bufferPixel(tmpPixelBuffer, i, tj)[0] = 0;
							rgb888ValAt(_debugFrame, i, tj)[1] = 255;
							//bufferPixel(tmpPixelBuffer, i, tj)[2] = 0;
						}
						_strips[i].push_back(OmniTouchStrip(i, partialMaxPos, partialMinPos));
						
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

void Detector::findFingers()
{
	_fingers.clear();
	vector<OmniTouchStrip*> stripBuffer;	//used to fill back

	for (int i = 0; i < _settings.depthHeight; i++)
	{
		for (vector<OmniTouchStrip>::iterator it = _strips[i].begin(); it != _strips[i].end(); ++it)
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
				for (vector<OmniTouchStrip>::iterator sIt = _strips[si].begin(); sIt != _strips[si].end(); ++sIt)
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

			ushort depth = *ushortValAt(_depthFrame, (first->row + last->row) / 2, (firstMidCol + lastMidCol) / 2);	//just a try
						
			double lengthSquared = getSquaredDistanceInRealWorld(
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
						uchar* dstPixel = rgb888ValAt(_debugFrame, row, col);
						dstPixel[0] = 255;
						//bufferPixel(tmpPixelBuffer, row, col)[1] = 255;
						dstPixel[2] = 255;
					}
				}

				_fingers.push_back(OmniTouchFinger(firstMidCol, first->row, depth, lastMidCol, last->row, depth));	//TODO: depth?
			}
		}
	}

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
	}

}

void Detector::refineDebugImage()
{
	
	//generate histogram
	int min = 65535, max = 0;
    
    assert(_sobelFrame.isContinuous());
    float* p = (float*)_sobelFrame.datastart;
    float* pEnd = (float*)_sobelFrame.dataend;

    for (; p < pEnd; ++p)
    {
        int h = (int)abs(*p);
        if (h > 0 && h < _maxHistogramSize)
        {
            if (h > max) max = h;
		    if (h < min) min = h;
        } //else out-of-range data
    }

	int histogramSize = max - min + 1;
	assert(histogramSize < _maxHistogramSize);
	int histogramOffset = min;

	memset(_histogram, 0, histogramSize * sizeof(int));

    p = (float*)_sobelFrame.datastart;
    pEnd = (float*)_sobelFrame.dataend;

	for (; p < pEnd; ++p)
	{
		int h = (int)abs(*p);
        if (h > 0 && h < _maxHistogramSize)
        {
		    _histogram[h - histogramOffset]++;
        }
	}

	for (int i = 1; i < histogramSize; i++)
	{
		_histogram[i] += _histogram[i-1];
	}

    int points = _sobelFrame.size().area();
	for (int i = 0; i < histogramSize; i++)
	{
		_histogram[i] = (int)(256 * ((double)_histogram[i] / (double)points) + 0.5);
	}
    

    //truncate and eq histogram on sobel
    /*convertScaleAbs(_sobelFrame, _sobelFrame);
    threshold(_sobelFrame, _sobelFrame, _maxHistogramSize, _maxHistogramSize, THRESH_TRUNC);
    equalizeHist(_sobelFrame, _sobelFrame);*/

	//draw the image
    assert(_debugFrame.isContinuous());
    float* sobelPixel = (float*)_sobelFrame.datastart;
    uchar* dstPixel = _debugFrame.datastart;
    uchar* dstEnd = _debugFrame.dataend;

	for (; dstPixel < dstEnd; dstPixel += 3, ++sobelPixel)
    {
        if (dstPixel[0] == 255 || dstPixel[1] == 255 || dstPixel[2] == 255)
		{
			//leave as is
		} 
		else
		{
            //dstPixel[1] = (int)*sobelPixel;
			int depth = (int)*sobelPixel;
            if (depth == 0 || abs(depth) >= _maxHistogramSize)
            {
                continue;
            }

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

