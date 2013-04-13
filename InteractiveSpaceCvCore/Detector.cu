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

//declare textures
texture<ushort, 2> texDepth;
texture<float, 2> texSobel;

__constant__ CommonSettings _settingsDev[1];
__constant__ DynamicParameters _dynamicParametersDev[1];

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, cv::Mat& debugFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _debugFrame(debugFrame),
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),
    _depthFrameGpu(settings.depthHeight, settings.depthWidth, CV_16U),
    _sobelFrame(settings.depthHeight, settings.depthWidth, CV_32F),
    _sobelFrameGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _debugSobelEqualizedFrame(settings.depthHeight, settings.depthWidth, CV_8U),
    _debugFrameGpu(settings.depthHeight, settings.depthWidth, CV_8UC3)
{
	//on device: upload settings to device memory
    cudaSafeCall(cudaMemcpyToSymbol(_settingsDev, &settings, sizeof(CommonSettings)));

	//init sobel
    gpu::registerPageLocked(_sobelFrame);

    //init gpu memory for storing strips
    //trips for each row of the depth image are stored in each column of _stripsDev. 
    //The tranpose is to minimize the downloading. 
    //TODO: might destroy coalesced access. What's the tradeoff?
    cudaSafeCall(cudaMallocHost(&_stripsHost, (MAX_STRIPS_PER_ROW + 1) * settings.depthHeight * sizeof(_OmniTouchStripDev)));
    cudaSafeCall(cudaMalloc(&_stripsDev, (MAX_STRIPS_PER_ROW + 1) * settings.depthHeight * sizeof(_OmniTouchStripDev)));

	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
	_histogram = new int[_maxHistogramSize];

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];

	//init vectors
	_strips.reserve(_settings.depthHeight);
	_fingers.reserve(ISE_MAX_FINGER_NUM);
}

Detector::~Detector()
{
    cudaSafeCall(cudaFree(_stripsDev));
    cudaSafeCall(cudaFreeHost(_stripsHost));
    gpu::unregisterPageLocked(_sobelFrame);

    delete [] _histogram;
    delete [] _floodHitTestVisitedFlag;
}

//update the parameters used by the algorithm
void Detector::updateDynamicParameters(const DynamicParameters& parameters)
{
	_parameters = parameters;
	
    //on device: upload parameters to device memory
    cudaSafeCall(cudaMemcpyToSymbol(_dynamicParametersDev, &parameters, sizeof(DynamicParameters)));
}

//the algorithm goes here. The detection algorithm runs per frame. The input is rgbFrame and depthFrame. The output is the return value, and also the debug frame.
//have a look at main() to learn how to use this.
FingerDetectionResults Detector::detect()
{
	//_iseHistEqualize(depthFrame, debugFrame);

    //_debugFrame.setTo(Scalar(0,0,0));	//set debug frame to black, can also done at GPU
    _debugFrameGpu.setTo(Scalar(0,0,0));

    memset(_floodHitTestVisitedFlag, 0, _settings.depthWidth * _settings.depthHeight);

    _depthFrameGpu.upload(_depthFrame);
	sobel();
    _sobelFrameGpu.download(_sobelFrame);

    //bind sobel for following usage
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    gpu::PtrStepSzb ptrStepSz(_sobelFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texSobel, ptrStepSz.data, desc, ptrStepSz.cols, ptrStepSz.rows, ptrStepSz.step));

    //bind depth
    cudaChannelFormatDesc descDepth = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrStepSzDepth(_depthFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texDepth, ptrStepSzDepth.data, descDepth, ptrStepSzDepth.cols, ptrStepSzDepth.rows, ptrStepSzDepth.step));

    findStrips();

    //unbind textures
    cudaSafeCall(cudaUnbindTexture(texSobel));
    cudaSafeCall(cudaUnbindTexture(texDepth));

    _debugFrameGpu.download(_debugFrame);

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
        printf("%s\n", cudaGetErrorString(err));
        assert(0); 
    }
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
    /*dim3 grid(1, 1);
    dim3 threads(32, 16);

    grid.x = divUp(_settings.depthWidth, threads.x);
    grid.y = divUp(_settings.depthHeight, threads.y);
    adjustDepthKernel<<<grid, threads>>>(_depthFrameGpu);*/
    cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, 5, -1);
}

__device__ _FloatPoint3D convertProjectiveToRealWorld(_IntPoint3D p)
{
    _FloatPoint3D r;
    r.x = (p.x / (float)_settingsDev[0].depthWidth - 0.5f) * p.z * _settingsDev[0].kinectIntrinsicParameters.realWorldXToZ;
    r.y = (0.5f - p.y / (float)_settingsDev[0].depthHeight) * p.z * _settingsDev[0].kinectIntrinsicParameters.realWorldYToZ;
    r.z = p.z / 100.0f * _settingsDev[0].kinectIntrinsicParameters.depthSlope + _settingsDev[0].kinectIntrinsicParameters.depthIntercept;

    return r;
}

__device__ float getSquaredDistanceInRealWorld(_IntPoint3D p1, _IntPoint3D p2)
{
    _FloatPoint3D rp1, rp2;

    rp1 = convertProjectiveToRealWorld(p1);
	rp2 = convertProjectiveToRealWorld(p2);

    return ((rp1.x - rp2.x) * (rp1.x - rp2.x) + (rp1.y - rp2.y) * (rp1.y - rp2.y) + (rp1.z - rp2.z) * (rp1.z - rp2.z));
}

__device__ int maxStripRowCount;

__global__ void findStripsKernel(gpu::PtrStepb debugPtr, _OmniTouchStripDev* resultPtr)
{
    extern __shared__ int stripCount[];
    int row = threadIdx.x;

    stripCount[row] = 1;
	StripState state = StripSmooth;
	int partialMin, partialMax;
	int partialMinPos, partialMaxPos;

	for (int col = 0; col < _settingsDev[0].depthWidth; col++)
	{
		float currVal = tex2D(texSobel, col, row);
        
        
		switch(state)
		{
		case StripSmooth:	//TODO: smooth
			if (currVal > _dynamicParametersDev[0].omniTouchParam.fingerRisingThreshold)
			{
				partialMax = currVal;
				partialMaxPos = col;
				state = StripRising;
			}
			break;

		case StripRising:
			if (currVal > _dynamicParametersDev[0].omniTouchParam.fingerRisingThreshold)
			{
				if (currVal > partialMax)
				{
					partialMax = currVal;
					partialMaxPos = col;
				}
			}
			else 
			{
				state = StripMidSmooth;
			}
			break;

		case StripMidSmooth:
			if (currVal < -_dynamicParametersDev[0].omniTouchParam.fingerFallingThreshold)
			{
				partialMin = currVal;
				partialMinPos = col;
				state = StripFalling;
			}
			else if (currVal > _dynamicParametersDev[0].omniTouchParam.fingerRisingThreshold)
			{
				//previous trial faied, start over
				partialMax = currVal;
				partialMaxPos = col;
				state = StripRising;
			}
			break;

		case StripFalling:
			if (currVal < -_dynamicParametersDev[0].omniTouchParam.fingerFallingThreshold)
			{
				if (currVal < partialMin)
				{
					partialMin = currVal;
					partialMinPos = col;
				}
			}
			else
			{
                ushort depth = tex2D(texDepth, (partialMaxPos + partialMinPos) / 2, row);
				
                _IntPoint3D p1, p2;
                p1.x = partialMaxPos;
                p1.y = row;
                p1.z = depth;
                p2.x = partialMinPos;
                p2.y = row;
                p2.z = depth;

				float distSquared = getSquaredDistanceInRealWorld(p1, p2);

				if (distSquared >= _dynamicParametersDev[0].omniTouchParam.fingerWidthMin * _dynamicParametersDev[0].omniTouchParam.fingerWidthMin 
					&& distSquared <= _dynamicParametersDev[0].omniTouchParam.fingerWidthMax * _dynamicParametersDev[0].omniTouchParam.fingerWidthMax)
				{
					for (int tj = partialMaxPos; tj <= partialMinPos; tj++)
					{
                        uchar* pixel = debugPtr.data + row * debugPtr.step + tj * 3;
						pixel[1] = 255;
					}

                    int resultOffset = stripCount[row] * _settingsDev[0].depthHeight + row;
                    resultPtr[resultOffset].start = partialMaxPos;
                    resultPtr[resultOffset].end = partialMinPos;
                    stripCount[row]++;

					partialMax = currVal;
					partialMaxPos = col;
				}

				state = StripSmooth;
			}
			break;
		} //switch 
	} //for 

    //the first row stores count for each column
    //resultPtr[row].start = 1;   //this field unused
    resultPtr[row].end = stripCount[row];

    __syncthreads();
    //map-recude to find the maximum strip count
    int total = blockDim.x;
    //int mid = (blockDim.x + 1) / 2;    //div up
    while (total > 1) 
    {
        int mid = (total + 1) / 2;
        if (row < mid)
        {
            if ( (row + mid < total) && stripCount[row + mid] > stripCount[row] ) 
            {
                stripCount[row] = stripCount[row + mid];
            }
        }
        __syncthreads();
        total = mid;
    } 

    if (row == 0)
    {
        maxStripRowCount = stripCount[0];
    }
}

void Detector::findStrips()
{
    //TODO: what if maximum thread < depthHeight? 
    //the third params: shared memory size in BYTES
    findStripsKernel<<<1, _settings.depthHeight, _settings.depthHeight * sizeof(int)>>>(_debugFrameGpu, _stripsDev);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaMemcpyFromSymbol(&_maxStripRowCount, maxStripRowCount, sizeof(int)));

    //download effective data, there are maxStripCount + 1 rows. The extra row stores count of strips for each column
    cudaSafeCall(cudaMemcpy(_stripsHost, _stripsDev, _maxStripRowCount * _settings.depthHeight * sizeof(_OmniTouchStripDev), cudaMemcpyDeviceToHost));

    //TODO: according to profiler, this trick seems not necessary. consider optimize for coelesence? 
}

void Detector::findFingers()
{
	_fingers.clear();
	vector<OmniTouchStrip*> stripBuffer;	//used to fill back

    //convert data
    _strips.clear();
    for (int i = 0; i < _settings.depthHeight; i++)
    {
        _strips.push_back(vector<OmniTouchStrip>());
        int stripCount = _stripsHost[i].end;
        _OmniTouchStripDev* p = _stripsHost + _settings.depthHeight + i;
        for (int j = 1; j < stripCount; ++j, p += _settings.depthHeight)
        {
            _strips[i].push_back(OmniTouchStrip(i, p->start, p->end));
        }
    }

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
    //truncate and eq histogram on sobel

    //convertScaleAbs uses saturation_cast, which truncates the value
    convertScaleAbs(_sobelFrame, _debugSobelEqualizedFrame, 256.0 / _maxHistogramSize);
    equalizeHist(_debugSobelEqualizedFrame, _debugSobelEqualizedFrame);

	//draw the image
    assert(_debugFrame.isContinuous());
    float* sobelPixel = (float*)_sobelFrame.datastart;
    uchar* debugSobelPixel = (uchar*)_debugSobelEqualizedFrame.datastart;
    uchar* dstPixel = _debugFrame.datastart;
    uchar* dstEnd = _debugFrame.dataend;

    for (; dstPixel < dstEnd; dstPixel += 3, ++sobelPixel, ++debugSobelPixel)
    {
        if (dstPixel[0] == 255 || dstPixel[1] == 255 || dstPixel[2] == 255)
		{
			//leave as is
		} 
		else
		{
            uchar sobelEq = *debugSobelPixel;
            float sobelVal = (float)*sobelPixel;

            if (sobelVal >= 0)
            {
                dstPixel[0] = 0;
				dstPixel[2] = sobelEq;
            } else 
            {
                dstPixel[0] = sobelEq;
				dstPixel[2] = 0;
            }
            dstPixel[1] = 0;
		}
    }

}

