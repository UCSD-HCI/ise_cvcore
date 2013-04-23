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
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
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

Detector::Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, const cv::Mat& depthToColorCoordFrame, cv::Mat& debugFrame)
    : _settings(settings), _rgbFrame(rgbFrame), _depthFrame(depthFrame), _depthToColorCoordFrame(depthToColorCoordFrame), _debugFrame(debugFrame),
    _rgbFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_8UC3),
    _rgbFloatFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32FC3),
    _rgbLuvFrameGpu(settings.rgbHeight, settings.rgbWidth, CV_32FC3),
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
	//on device: upload settings to device memory
    cudaSafeCall(cudaMemcpyToSymbol(_settingsDev, &settings, sizeof(CommonSettings)));

    //page lock
    gpu::registerPageLocked(_rgbPdfFrame);

    //init gpu memory for storing strips
    //trips for each row of the depth image are stored in each column of _stripsDev. 
    //The tranpose is to minimize the downloading. 
    //TODO: might destroy coalesced access. What's the tradeoff?
    cudaSafeCall(cudaMallocHost(&_stripsHost, (MAX_STRIPS_PER_ROW + 1) * settings.depthHeight * sizeof(_OmniTouchStripDev)));
    cudaSafeCall(cudaMalloc(&_stripsDev, (MAX_STRIPS_PER_ROW + 1) * settings.depthHeight * sizeof(_OmniTouchStripDev)));

    //init memory for storing fingers
    _stripVisitedFlags = new uchar[(MAX_STRIPS_PER_ROW + 1) * settings.depthHeight];

	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
    cudaSafeCall(cudaMemcpyToSymbol(_maxHistogramSizeDev, &_maxHistogramSize, sizeof(int)));
	
	//allocate memory for flood test visited flag
	_floodHitTestVisitedFlag = new uchar[_settings.depthWidth * _settings.depthHeight];
    
	//init vectors
	_fingers.reserve(ISE_MAX_FINGER_NUM);
}

Detector::~Detector()
{
    gpu::unregisterPageLocked(_rgbPdfFrame);

    cudaSafeCall(cudaFree(_stripsDev));
    cudaSafeCall(cudaFreeHost(_stripsHost));
    
    delete [] _stripVisitedFlags;
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


    gpuProcess();
    
    /*
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

    refineDebugImage();
    findStrips();

    //unbind textures
    cudaSafeCall(cudaUnbindTexture(texSobel));
    cudaSafeCall(cudaUnbindTexture(texDepth));
    
    _debugFrameGpu.download(_debugFrame);
    */

    findFingers();
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    stripCount[threadIdx.x] = 1;

    if (row < _settingsDev[0].depthHeight)
    {
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

                        int resultOffset = stripCount[threadIdx.x] * _settingsDev[0].depthHeight + row;
                        resultPtr[resultOffset].start = partialMaxPos;
                        resultPtr[resultOffset].end = partialMinPos;
                        resultPtr[resultOffset].row = row;
                        stripCount[threadIdx.x]++;

					    partialMax = currVal;
					    partialMaxPos = col;
				    }

				    state = StripSmooth;
			    }
			    break;
		    } //switch 

            if (stripCount[threadIdx.x] > Detector::MAX_STRIPS_PER_ROW)
            {
                break;
            }
	    } //for 

        //the first row stores count for each column
        //resultPtr[row].start = 1;   //this field unused
        resultPtr[row].end = stripCount[threadIdx.x];
    }   //if row < 0

    __syncthreads();
    //map-recude to find the local maximum strip count
    int total = blockDim.x;
    //int mid = (blockDim.x + 1) / 2;    //div up
    while (total > 1) 
    {
        int mid = (total + 1) / 2;
        if (threadIdx.x < mid)
        {
            if ( (threadIdx.x + mid < total) && stripCount[threadIdx.x + mid] > stripCount[threadIdx.x] ) 
            {
                stripCount[threadIdx.x] = stripCount[threadIdx.x + mid];
            }
        }
        __syncthreads();
        total = mid;
    } 

    if (threadIdx.x == 0)
    {
        atomicMax(&maxStripRowCountDev, stripCount[0]);
    }
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
                        
						dstPixel[0] = 255;
						dstPixel[2] = 255;

                        //read color
                        /*const int* mapCoord = (int*)_depthToColorCoordFrame.ptr(rowFill) + colFill * 2;
                        int cx = mapCoord[0];
                        int cy = mapCoord[1];
                        const uchar* rgbPixel = _rgbFrame.ptr(cy) + cx * 3;
                        //const uchar* rgbPixel = _rgbFrame.ptr(rowFill) + colFill * 3;

                        memcpy(dstPixel, rgbPixel, 3);*/
					}
				}

                _fingers.push_back(finger);
			} // check length
		
        }   // for each col
	} //for each row

    sort(_fingers.begin(), _fingers.end());
}


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
        uchar res = (uchar)(fabsf(sobel) / (float)(_maxHistogramSizeDev[0]) * 255.0f + 0.5f);
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


__global__ void applySkinColorModel(gpu::PtrStepf luvPtr, gpu::PtrStepf pdfPtr)
{
    const int nComp = 3;
    const float mu[nComp][2] = { {11.2025f, 8.2296f},
                         {35.5613f, 30.1054f},
                         {22.7229f, 19.4680f} };
    const float conv[nComp][2] = { {37.691f, 43.929f},
                            {225.23f, 242.08f},
                            {54.738f, 61.146f} };
    const float prop[nComp] = {0.28847f, 0.24641f, 0.46512f};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < _settingsDev[0].rgbWidth && y < _settingsDev[0].rgbHeight)
    {
        float* luv = luvPtr.ptr(y) + x * 3;
        float u = luv[1];
        float v = luv[2];

        float p = 0;
        
        #pragma unroll
        for (int i = 0; i < nComp; i++)
        {
            float d = 1.f / (2 * CUDART_PI_F * sqrt(conv[i][0] * conv[i][1]));
            float e = expf(-0.5f * (powf(u - mu[i][0], 2) / conv[i][0] + powf(v - mu[i][1], 2) / conv[i][1]));
            p += prop[i] * d * e;
        }

        float* dst = pdfPtr.ptr(y) + x;
        //*dst = p * 500.f;
        //*dst = luv[0] / 100.0f;
        *dst = p * 1000.f;
    }
}


/*__global__ void applySkinColorModel(gpu::PtrStepf luvPtr, gpu::PtrStepf pdfPtr)
{
    const int nComp = 3;
    const double mu[nComp][2] = {{11.2024686283253,          8.22956679034156},
                                {35.5612918973632,          30.1054062096261},
                                {22.7229203550896,          19.4680134168419}};
    const double conv[nComp][2] = { { 37.6914561020612,          43.9294235115161},
                                    { 225.22898713039,          242.081315811438},
                                    { 54.7381847499349,          61.1464611751676} };
    const double prop[nComp] = {0.28846824006064,         0.246407551490279,          0.46512420844908};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < _settingsDev[0].rgbWidth && y < _settingsDev[0].rgbHeight)
    {
        float* luv = luvPtr.ptr(y) + x * 3;
        double u = luv[1];
        double v = luv[2];

        double p = 0;
        
        #pragma unroll
        for (int i = 0; i < nComp; i++)
        {
            double d = 1.0 / (2.0 * CUDART_PI_F * sqrt(conv[i][0] * conv[i][1]));
            double e = exp(-0.5 * (pow(u - mu[i][0], 2) / conv[i][0] + pow(v - mu[i][1], 2) / conv[i][1]));
            p += prop[i] * d * e;
        }

        float* dst = pdfPtr.ptr(y) + x;
        //*dst = p * 500.f;
        //*dst = luv[0] / 100.0f;
        *dst = (float)(p * 1000.0);
    }
}*/

void Detector::gpuProcess()
{
    cudaStream_t cudaStreamDepthDebug = gpu::StreamAccessor::getStream(_gpuStreamDepthDebug);
    cudaStream_t cudaStreamDepthWorking = gpu::StreamAccessor::getStream(_gpuStreamDepthWorking);
    cudaStream_t cudaStreamRgbWorking = gpu::StreamAccessor::getStream(_gpuStreamRgbWorking);
    
    _depthFrameGpu.upload(_depthFrame);
    //_gpuStreamDepthWorking.enqueueUpload(_depthFrame, _depthFrameGpu);
    
    //Looks like when running Sobel async, visual profiler won't generate any timeline.
    cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, _sobelFrameBufferGpu, 5, -1);
    //cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, _sobelFrameBufferGpu, 5, -1.0f, BORDER_DEFAULT, -1, _gpuStreamDepthWorking);
    //_gpuStreamDepthWorking.waitForCompletion();

    _gpuStreamDepthWorking.enqueueMemSet(_debugFrameGpu, Scalar(0,0,0));
    
    //bind sobel for following usage
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    gpu::PtrStepSzb ptrStepSz(_sobelFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texSobel, ptrStepSz.data, desc, ptrStepSz.cols, ptrStepSz.rows, ptrStepSz.step));

    //bind depth
    cudaChannelFormatDesc descDepth = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrStepSzDepth(_depthFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texDepth, ptrStepSzDepth.data, descDepth, ptrStepSzDepth.cols, ptrStepSzDepth.rows, ptrStepSzDepth.step));
    
    //find strips on stream2: upload data
    //TODO: what if maximum thread < depthHeight? 
    //the third params: shared memory size in BYTES
    int* maxStripRowCountDevPtr;
    cudaSafeCall(cudaGetSymbolAddress((void**)&maxStripRowCountDevPtr, maxStripRowCountDev));
    cudaSafeCall(cudaMemsetAsync(maxStripRowCountDevPtr, 0, sizeof(int), cudaStreamDepthWorking));

    //find strips on stream 2: kernel call
    //turns out 1 block is the best even though profiler suggests more blocks
    int nThread = _settings.depthHeight;    
    int nBlock = 1; //divUp(_settings.depthHeight, nThread);
    findStripsKernel<<<nBlock, nThread, nThread * sizeof(int), cudaStreamDepthWorking>>>(_debugFrameGpu, _stripsDev);
    //cudaSafeCall(cudaGetLastError());

    //rgb manipulation
    _gpuStreamRgbWorking.enqueueUpload(_rgbFrame, _rgbFrameGpu);
    _gpuStreamRgbWorking.enqueueConvert(_rgbFrameGpu, _rgbFloatFrameGpu, CV_32FC3, 1.f / 255.f);
    gpu::cvtColor(_rgbFloatFrameGpu, _rgbLuvFrameGpu, CV_RGB2Luv, 0, _gpuStreamRgbWorking);

    //rgb skin color model
    dim3 rgbThreads(16, 32);
    dim3 rgbGrid(divUp(_settings.rgbWidth, rgbThreads.x), divUp(_settings.rgbHeight, rgbThreads.y));
    applySkinColorModel<<<rgbGrid, rgbThreads, 0, cudaStreamRgbWorking>>>(_rgbLuvFrameGpu, _rgbPdfFrameGpu);
    cudaSafeCall(cudaGetLastError());

    //refine debug image on stream1: kernel call
    dim3 threads(16, 32);
    dim3 grid(divUp(_settings.depthWidth, threads.x), divUp(_settings.depthHeight, threads.y));
    convertScaleAbsKernel<<<grid, threads, 0, cudaStreamDepthDebug>>>(_debugSobelEqFrameGpu);
    //cudaSafeCall(cudaGetLastError());

    gpu::equalizeHist(_debugSobelEqFrameGpu, _debugSobelEqFrameGpu, _debugSobelEqHistGpu, _debugSobelEqBufferGpu, _gpuStreamDepthDebug);
    
    //rgb download
    _gpuStreamRgbWorking.enqueueDownload(_rgbPdfFrameGpu, _rgbPdfFrame);

    //find strips on stream 2: download data
    cudaSafeCall(cudaMemcpyFromSymbolAsync(&_maxStripRowCount, maxStripRowCountDev, sizeof(int), 0, cudaMemcpyDeviceToHost, cudaStreamDepthWorking));

    //download strips
    //download effective data, there are maxStripCount + 1 rows. The extra row stores count of strips for each column
    cudaSafeCall(cudaMemcpyAsync(_stripsHost, _stripsDev, _maxStripRowCount * _settings.depthHeight * sizeof(_OmniTouchStripDev), 
        cudaMemcpyDeviceToHost, cudaStreamDepthWorking));
    //TODO: according to profiler, this trick seems not necessary. consider optimize for coelesence? 
  
    _gpuStreamDepthDebug.waitForCompletion();
    _gpuStreamDepthWorking.waitForCompletion();  
    cudaSafeCall(cudaGetLastError());

    //draw the debug image
    refineDebugImageKernel<<<grid, threads, 0, cudaStreamDepthDebug>>>(_debugFrameGpu, _debugSobelEqFrameGpu);
    cudaSafeCall(cudaGetLastError());
    
    //_debugFrameGpu.download(_debugFrame);
    _gpuStreamDepthDebug.enqueueDownload(_debugFrameGpu, _debugFrame);
    _gpuStreamDepthDebug.waitForCompletion();


    //unbind textures
    cudaSafeCall(cudaUnbindTexture(texSobel));
    cudaSafeCall(cudaUnbindTexture(texDepth));
    

    _gpuStreamRgbWorking.waitForCompletion();
    
}