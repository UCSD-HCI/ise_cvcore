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
//#include <thrust/sort.h>


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
//__constant__ int _floodFillNeighborOffset[6];
__constant__ int _maxHistogramSizeDev[1];

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, cv::Mat& debugFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _debugFrame(debugFrame),
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),
    _depthFrameGpu(settings.depthHeight, settings.depthWidth, CV_16U),
     _sobelFrameGpu(settings.depthHeight, settings.depthWidth, CV_32F),
    _debugSobelEqFrameGpu(settings.depthHeight, settings.depthWidth, CV_8U),
    _debugFrameGpu(settings.depthHeight, settings.depthWidth, CV_8UC3)
{
	//on device: upload settings to device memory
    cudaSafeCall(cudaMemcpyToSymbol(_settingsDev, &settings, sizeof(CommonSettings)));

    //init gpu memory for storing strips
    //trips for each row of the depth image are stored in each column of _stripsDev. 
    //The tranpose is to minimize the downloading. 
    //TODO: might destroy coalesced access. What's the tradeoff?
    cudaSafeCall(cudaMalloc(&_stripsDev, (MAX_STRIPS_PER_ROW + 1) * settings.depthHeight * sizeof(_OmniTouchStripDev)));

    //init gpu memory for storing fingers
    cudaSafeCall(cudaMallocHost(&_fingersHost, ISE_MAX_FINGER_NUM * sizeof(_OmniTouchFingerDev)));
    cudaSafeCall(cudaMalloc(&_fingersDev, ISE_MAX_FINGER_NUM * sizeof(_OmniTouchFingerDev)));
 
	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
    cudaSafeCall(cudaMemcpyToSymbol(_maxHistogramSizeDev, &_maxHistogramSize, sizeof(int)));
	
    //init flood fill constant
    /*const int neighborOffset[3][2] =
	{
		{-1, 0},
		{1, 0},
		{0, -1}
	};
    cudaSafeCall(cudaMemcpyToSymbol(_floodFillNeighborOffset, neighborOffset, 6 * sizeof(int)));*/

	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];
    
	//init vectors
	//_strips.reserve(_settings.depthHeight);
	_fingers.reserve(ISE_MAX_FINGER_NUM);
}

Detector::~Detector()
{
    cudaSafeCall(cudaFree(_stripsDev));
    
    cudaSafeCall(cudaFree(_fingersDev));
    cudaSafeCall(cudaFreeHost(_fingersHost));

    //delete [] _histogram;
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

    _depthFrameGpu.upload(_depthFrame);
	sobel();
    
    //bind sobel for following usage
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    gpu::PtrStepSzb ptrStepSz(_sobelFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texSobel, ptrStepSz.data, desc, ptrStepSz.cols, ptrStepSz.rows, ptrStepSz.step));

    //bind depth
    cudaChannelFormatDesc descDepth = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrStepSzDepth(_depthFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texDepth, ptrStepSzDepth.data, descDepth, ptrStepSzDepth.cols, ptrStepSzDepth.rows, ptrStepSzDepth.step));

    findStrips();
    findFingers();
    refineDebugImage();
    
    //unbind textures
    cudaSafeCall(cudaUnbindTexture(texSobel));
    cudaSafeCall(cudaUnbindTexture(texDepth));
    
    _debugFrameGpu.download(_debugFrame);
    floodHitTest();

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

void Detector::sobel()
{
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

__device__ int maxStripRowCountDev;

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
                        //uchar* pixel = debugPtr.data + row * debugPtr.step + tj * 3;
                        uchar* pixel = debugPtr.ptr(row) + tj * 3;
						pixel[1] = 255;
					}

                    int resultOffset = stripCount[row] * _settingsDev[0].depthHeight + row;
                    resultPtr[resultOffset].start = partialMaxPos;
                    resultPtr[resultOffset].end = partialMinPos;
                    resultPtr[resultOffset].row = row;
                    stripCount[row]++;

					partialMax = currVal;
					partialMaxPos = col;
				}

				state = StripSmooth;
			}
			break;
		} //switch 

        if (stripCount[row] > Detector::MAX_STRIPS_PER_ROW)
        {
            break;
        }
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
        maxStripRowCountDev = stripCount[0];
    }
}

void Detector::findStrips()
{
    //TODO: what if maximum thread < depthHeight? 
    //the third params: shared memory size in BYTES
    findStripsKernel<<<1, _settings.depthHeight, _settings.depthHeight * sizeof(int)>>>(_debugFrameGpu, _stripsDev);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaMemcpyFromSymbol(&_maxStripRowCount, maxStripRowCountDev, sizeof(int)));

}

__device__ int fingerCountDev;

__global__ void findFingersKernel(gpu::PtrStepb debugPtr, _OmniTouchStripDev* strips, _OmniTouchFingerDev* fingers)
{
    extern __shared__ uchar sharedBuffer[];

    int col = threadIdx.x;
    uchar* visitedFlags = sharedBuffer; //mapping one by one to strips; 1st row unused. 
    ushort* stripBuffer = (ushort*)(sharedBuffer + _settingsDev[0].depthHeight * (blockDim.x + 1)) 
        + col * Detector::MAX_FINGER_PIXEL_LENGTH; //offset to the current thread
    
    //init visited flags; 
    uchar* visitedFlagsForCol = visitedFlags + (col + 1) * _settingsDev[0].depthHeight;
    for (int row = 0; row < _settingsDev[0].depthHeight; row++, visitedFlagsForCol++)
    {
        *visitedFlagsForCol = 0;
    }

    //init global finger count
    if (col == 0)
    {
        fingerCountDev = 0;
    }
	
	for (int row = 0; row < _settingsDev[0].depthHeight; row++)
	{
        __syncthreads();

        if (col < strips[row].end - 1)
        {
            ushort stripOffset = (col + 1) * _settingsDev[0].depthHeight + row;

			if (visitedFlags[stripOffset] > 0)
			{
				continue;
			}

            stripBuffer[0] = stripOffset;
            int stripBufferCount = 1;
            visitedFlags[stripOffset] = 1;

			//search down
			int blankCounter = 0;
			for (int si = row; si < _settingsDev[0].depthHeight; si++)    //algorithm changed. seems to find a bug in original version
			{
				_OmniTouchStripDev* currTop = strips + stripBuffer[stripBufferCount - 1];

				//search strip
				bool stripFound = false;
                
                int searchDownOffset = _settingsDev[0].depthHeight + si;

                for (int sj = 0; sj < strips[si].end - 1; ++sj, searchDownOffset += _settingsDev[0].depthHeight)
				{
					if (visitedFlags[searchDownOffset])
					{
						continue;
					}

                    _OmniTouchStripDev* candidate = strips + searchDownOffset;

                    if (candidate->end > currTop->start && candidate->start < currTop->end)	//overlap!
					{
                        stripBuffer[stripBufferCount] = searchDownOffset;
                        stripBufferCount++;

                        //Note: race condition happens here. But won't generate incorrect results.
                        visitedFlags[searchDownOffset] = 1;
						
                        stripFound = true;
						break;
					}
				}

				if (!stripFound) //blank
				{
					blankCounter++;
					if (blankCounter > _dynamicParametersDev[0].omniTouchParam.stripMaxBlankPixel)
					{
						//Too much blank, give up
						break;
					}
				}
			}

			//check length
			_OmniTouchStripDev* first = strips + stripBuffer[0];
            _OmniTouchStripDev* last = strips + stripBuffer[stripBufferCount - 1];
            
            _OmniTouchFingerDev finger;

            //int firstMidCol = (first->start + first->end) / 2;
            finger.tip.x = (first->start + first->end) / 2;
            finger.tip.y = first->row;
			//int lastMidCol = (last->start + last->end) / 2;
            finger.end.x = (last->start + last->end) / 2;
            finger.end.y = last->row;

            finger.tip.z = tex2D(texDepth, (finger.tip.x + finger.end.x) / 2, (first->row + last->row) / 2);	//just a try
            finger.end.z = finger.tip.z;
			
			float lengthSquared = getSquaredDistanceInRealWorld(finger.tip, finger.end);
			int pixelLength = finger.end.y - finger.tip.y + 1;
			
			if (pixelLength >= _dynamicParametersDev[0].omniTouchParam.fingerMinPixelLength 
				&& lengthSquared >= _dynamicParametersDev[0].omniTouchParam.fingerLengthMin * _dynamicParametersDev[0].omniTouchParam.fingerLengthMin 
				&& lengthSquared <= _dynamicParametersDev[0].omniTouchParam.fingerLengthMax * _dynamicParametersDev[0].omniTouchParam.fingerLengthMax)	//finger!
			{
				//fill back
				int bufferPos = -1;
				for (int rowFill = first->row; rowFill <= last->row; rowFill++)
				{
					int leftCol, rightCol;
                    _OmniTouchStripDev* nextBufferItem = strips + stripBuffer[bufferPos + 1];

					if (rowFill == nextBufferItem->row)	//find next detected row
					{
                        leftCol = nextBufferItem->start;
                        rightCol = nextBufferItem->end;
                        bufferPos++;
					}
					else	//in blank area, interpolate
					{
                        _OmniTouchStripDev* thisBufferItem = strips + stripBuffer[bufferPos];

						float ratio = (float)(rowFill - thisBufferItem->row) / (float)(nextBufferItem->row - thisBufferItem->row);
                        leftCol = (int)(thisBufferItem->start + (nextBufferItem->start - thisBufferItem->start) * ratio + 0.5f);
                        rightCol = (int)(thisBufferItem->end + (nextBufferItem->end - thisBufferItem->end) * ratio + 0.5f);
					}

					for (int colFill = leftCol; colFill <= rightCol; colFill++)
					{
                        uchar* dstPixel = debugPtr.ptr(rowFill) + colFill * 3;
                        //uchar* dstPixel = debugPtr.data + rowFill * debugPtr.step + colFill * 3;
						dstPixel[0] = 255;
						dstPixel[2] = 255;
					}
				}

                int currentFingerCount = atomicAdd(&fingerCountDev, 1);
                fingers[currentFingerCount] = finger;
			} // check length
		
        }   // if threadIdx.x < strip count
	} //for each row

}


void Detector::findFingers()
{
    int maxStripCount = _maxStripRowCount - 1;  //do not include the first row storing strip count
    int sharedMemSize = _settings.depthHeight * _maxStripRowCount // visited flags
                        + MAX_FINGER_PIXEL_LENGTH * maxStripCount * sizeof(ushort); //strip buffer

    assert(sharedMemSize < 49152);  //TODO: read shared memory size from device query; if not large enough, use global memory instead

    findFingersKernel<<<1, maxStripCount, sharedMemSize>>>(_debugFrameGpu, _stripsDev, _fingersDev);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaMemcpyFromSymbol(&_fingerCount, fingerCountDev, sizeof(int)));

    //convert data
    cudaSafeCall(cudaMemcpy(_fingersHost, _fingersDev, _fingerCount * sizeof(_OmniTouchFingerDev), cudaMemcpyDeviceToHost));
    

    _fingers.clear();
    for (int i = 0; i < _fingerCount; i++)
    {
        _OmniTouchFingerDev* f = _fingersHost + i;
        OmniTouchFinger finger(f->tip.x, f->tip.y, f->tip.z, f->end.x, f->end.y, f->end.z);
        _fingers.push_back(finger);
    }

    sort(_fingers.begin(), _fingers.end());

}

/*  //this algorithm is wrong. have to use DFS
__global__ void floodHitTestKernel(gpu::PtrStepb debugPtr, _OmniTouchFingerDev* fingers)
{
    extern __shared__ uchar prevState[];
    __shared__ int area;
    __shared__ int currentLevelHits;

    int currentLevelHitsCopy;
    int level = 0;
        
    if (threadIdx.x == 0)
    {
        area = 0;
        currentLevelHits = 0;
        prevState[0] = 1;
    }
    else
    {
        prevState[threadIdx.x] = 0;
    }

    __syncthreads();
    
    do
    {
        level++;

        bool continuous = false;
        bool prevFilled = false;
        //check neighbor's previous state
                
        if (threadIdx.x <= level * 2)
        {

            int x = fingers[blockIdx.x].tip.x;
            int y = fingers[blockIdx.x].tip.y;
            int x1, y2;

            if (threadIdx.x % 2 == 0)
            {
                x += threadIdx.x / 2;
                y -= level - (threadIdx.x / 2);
                x1 = x - 1;
            }
            else
            {
                x -= (threadIdx.x + 1) / 2;
                y -= level - (threadIdx.x + 1) / 2;
                x1 = x + 1;
            }
            y2 = y + 1;

            
            if (x >= 0 && y >= 0 && x < _settingsDev[0].depthWidth && y < _settingsDev[0].depthHeight)
            {
                int depth = tex2D(texDepth, x, y);
                int prevDepth;
            
                if (y2 < _settingsDev[0].depthHeight && prevState[threadIdx.x])
                {
                    prevDepth = tex2D(texDepth, x, y2);
                    continuous = (fabsf(depth - prevDepth) < _dynamicParametersDev[0].omniTouchParam.clickFloodMaxGrad);
                }
                
                if (!continuous && threadIdx.x > 0 && x1 >= 0 && x1 < _settingsDev[0].depthWidth &&
                        (threadIdx.x == 1 ? prevState[0] : prevState[threadIdx.x - 2])
                    )
                {
                    prevDepth = tex2D(texDepth, x1, y);
                    continuous = (fabsf(depth - prevDepth) < _dynamicParametersDev[0].omniTouchParam.clickFloodMaxGrad);
                }
            
                if (continuous)
                {
                    uchar* pixel = debugPtr.ptr(y) + x * 3;
                    //uchar* pixel = debugPtr.data + y * debugPtr.step + x * 3;
                    pixel[0] = 255;
				    pixel[1] = 255;
				    pixel[2] = 0;
                    atomicAdd(&currentLevelHits, 1);
                }
            }

        } //if prevFilled

        __syncthreads();

        currentLevelHitsCopy = currentLevelHits;
        prevState[threadIdx.x] = continuous;
        if (threadIdx.x == 0)
        {
            area += currentLevelHits;
            currentLevelHits = 0;
        }

        __syncthreads();

    } while (currentLevelHitsCopy && area < _dynamicParametersDev[0].omniTouchParam.clickFloodArea);

    if (threadIdx.x == 0)
    {
        fingers[blockIdx.x].isOnSurface = (area >= _dynamicParametersDev[0].omniTouchParam.clickFloodArea);
    }
}*/

/*
__global__ void floodHitTestKernel(gpu::PtrStepb debugPtr, _OmniTouchFingerDev* fingers)
{

    extern __shared__ _ShortPoint2D dfsQueue[];
    __shared__ int visitedFlags[Detector::FLOOD_FILL_RADIUS * Detector::FLOOD_FILL_RADIUS * 2 / 32];
    __shared__ int dfsHead;
    __shared__ int dfsEnd;
    __shared__ int area;

    //init data
    int fingerId = blockIdx.x;

    for (int i = threadIdx.x; i < Detector::FLOOD_FILL_RADIUS * Detector::FLOOD_FILL_RADIUS * 2 / 32; i += blockDim.x)
    {
        visitedFlags[i] = 0;
    }

    if (threadIdx.x == ((Detector::FLOOD_FILL_RADIUS / 32) % blockDim.x))  //use this thread that initialized visitedFlag[FLOOD_FILL_RADIUS / 32] to avoid extra sync
    {
        //enque the center
        dfsQueue[0].x = 0;
        dfsQueue[0].y = 0;

        dfsHead = 0;
        dfsEnd = 1;
        area = 1;
        visitedFlags[Detector::FLOOD_FILL_RADIUS / 32] = 1;
    }

    int prevArea = 0;

    while(area > prevArea 
        && area < _dynamicParametersDev[0].omniTouchParam.clickFloodArea
        && dfsEnd < 512 //queue is not full TODO: avoid hard coding; use circular buffer
        && dfsEnd > dfsHead) //queue is not empty
	{
        //this sync and the sync at the end of the loop guarantees that all threads have the same evaluation of dfsEnd > dfsHead

        prevArea = area;

        //dispatch
        int elemIdx = dfsHead + threadIdx.x;
        if (threadIdx.x == 0)
        {
            dfsHead = (dfsEnd < dfsHead + blockDim.x ) ? dfsEnd : (dfsHead + blockDim.x);
        }

        int prevEnd = dfsEnd;

        __syncthreads();    

        if (elemIdx < prevEnd)
        {

            int centerRltX = dfsQueue[elemIdx].x;
            int centerRltY = dfsQueue[elemIdx].y;

            int centerAbsX = fingers[fingerId].tip.x + centerRltX;
            int centerAbsY = fingers[fingerId].tip.y + centerRltY;

            ushort centerDepth = tex2D(texDepth, centerAbsX, centerAbsY);

		    for (int i = 0; i < 3; i++)
		    {
                int dx = _floodFillNeighborOffset[i * 2];
                int dy = _floodFillNeighborOffset[i * 2 + 1];

                int neighborRltX = centerRltX + dx;
                int neighborRltY = centerRltY + dy;

                if ( (neighborRltX > 0 ? (neighborRltX >= Detector::FLOOD_FILL_RADIUS) : (neighborRltX <= -Detector::FLOOD_FILL_RADIUS))
                    || (neighborRltY < -Detector::FLOOD_FILL_RADIUS) )
                {
                    //exceed the flood radius
                    continue;
                }

                int neighborAbsX = centerAbsX + dx;
                int neighborAbsY = centerAbsY + dy;

                if (neighborAbsX < 0 || neighborAbsX >= _settingsDev[0].depthWidth || neighborAbsY < 0)
                {
                    //exceed the image border
                    continue;
                }

                //the bitwise trick
                int visitedFlagPos = -neighborRltY * (Detector::FLOOD_FILL_RADIUS * 2) + neighborRltX + Detector::FLOOD_FILL_RADIUS;
                int visitedFlagIdx = visitedFlagPos / 32;
                int visitedFlagTestBit = 1 << (visitedFlagPos % 32);

                //this check is thread safe because a visited flag will never be cleared once set. 
                if (visitedFlags[visitedFlagIdx] & visitedFlagTestBit)
                {
                    continue;
                }

			    ushort neiborDepth = tex2D(texDepth, neighborAbsX, neighborAbsY);
			    if (fabsf((float)neiborDepth - (float)centerDepth) > _dynamicParametersDev[0].omniTouchParam.clickFloodMaxGrad)
			    {
				    continue;					
			    }


                //Check visited again. This time we set the flag. If it is a newly set flag, then enque.
                int prevVisited = atomicOr(&(visitedFlags[visitedFlagIdx]), visitedFlagTestBit);

                if (prevVisited & visitedFlagTestBit)
                {
                    continue;
                }

                uchar* dstPixel = debugPtr.ptr(neighborAbsY) + neighborAbsX * 3;
			    dstPixel[0] = 255;
			    dstPixel[1] = 255;
			    dstPixel[2] = 0;

                //enqueue
                atomicAdd(&area, 1);
                int currEnd = atomicAdd(&dfsEnd, 1);

                if (currEnd < 512)  //queue is not full TODO: avoid hard coding
                {
                    dfsQueue[currEnd].x = neighborRltX;
                    dfsQueue[currEnd].y = neighborRltY;
                }
		    }   //end for
        }   //if (currHead < prevEnd)

        __syncthreads();
	}   //end while
	
    if (threadIdx.x == 0)
    { 
		if (area >= _dynamicParametersDev[0].omniTouchParam.clickFloodArea)
		{
			fingers[fingerId].isOnSurface = 1;
		}
    }
}*/

void Detector::floodHitTest()
{
    /*if (_fingerCount > 0)
    {
        //TODO: bad scalability (when image goes large) and too many syncthreads
        //floodHitTestKernel<<<_fingerCount, 512, 512>>>(_debugFrameGpu, _fingersDev);
        floodHitTestKernel<<<_fingerCount, 512, 512 * sizeof(_ShortPoint2D)>>>(_debugFrameGpu, _fingersDev);
        cudaSafeCall(cudaGetLastError());
    
        //download result
        cudaSafeCall(cudaMemcpy(_fingersHost, _fingersDev, _fingerCount * sizeof(_OmniTouchFingerDev), cudaMemcpyDeviceToHost));
    }*/

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

__global__ void convertScaleAbsKernel(gpu::PtrStepb debugSobelEqPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < _settingsDev[0].depthWidth && y < _settingsDev[0].depthHeight)
    {
        float sobel = tex2D(texSobel, x, y);
        uchar res = (uchar)(fabsf(sobel) / (float)(_maxHistogramSizeDev[0]) * 256.0f + 0.5f);
        *(debugSobelEqPtr.ptr(y) + x) = res;
    }
}

__global__ void refineDebugImageKernel(gpu::PtrStepb debugPtr, gpu::PtrStepb sobelEqPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < _settingsDev[0].depthWidth && y < _settingsDev[0].depthHeight)
    {
        uchar* dstPixel = debugPtr.ptr(y) + x * 3;

        if (dstPixel[0] == 255 || dstPixel[1] == 255 || dstPixel[2] == 255)
		{
			//leave as is
		} 
		else
		{
            uchar sobelEq = *(sobelEqPtr.ptr(y) + x);
            float sobelVal = tex2D(texSobel, x, y);

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

void Detector::refineDebugImage()
{
    //truncate and eq histogram on sobel
    dim3 threads(16, 32);
    dim3 grid(divUp(_settings.depthWidth, threads.x), divUp(_settings.depthHeight, threads.y));
    convertScaleAbsKernel<<<grid, threads>>>(_debugSobelEqFrameGpu);
    cudaSafeCall(cudaGetLastError());

    gpu::equalizeHist(_debugSobelEqFrameGpu, _debugSobelEqFrameGpu);
    
	//draw the image
    refineDebugImageKernel<<<grid, threads>>>(_debugFrameGpu, _debugSobelEqFrameGpu);
    cudaSafeCall(cudaGetLastError());
    
}

