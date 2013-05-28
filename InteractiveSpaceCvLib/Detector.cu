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

using namespace std;
using namespace cv;
using namespace ise;

//declare textures
texture<ushort, 2> texDepth;
texture<float, 2> texSobel;
texture<ushort, 2> texTrDepth;
texture<float, 2> texTrSobel;

__constant__ CommonSettings _settingsDev[1];
__constant__ DynamicParameters _dynamicParametersDev[1];
//__constant__ int _floodFillNeighborOffset[6];
__constant__ int _maxHistogramSizeDev[1];

template <_ImageDirection dir>
__device__ int depthWidth()
{
    return (dir == DirTransposed ? _settingsDev[0].depthHeight : _settingsDev[0].depthWidth);
}

template <_ImageDirection dir>
__device__ int depthHeight()
{
    return (dir == DirTransposed ? _settingsDev[0].depthWidth : _settingsDev[0].depthHeight);
}

//choose texture according to direction
#define dirTex2D(dir, tex, texTr, x, y) ((dir) == DirTransposed ? tex2D((texTr), (x), (y)) : tex2D((tex), (x), (y)))

void Detector::cudaSafeCall(cudaError_t err)
{
    //TODO: better handler
    if (err != 0)
    {
        const char* errStr = cudaGetErrorString(err);
        printf("%s\n", errStr);
        assert(0); 
    }
}

void Detector::cudaInit()
{
    cudaSafeCall(cudaMemcpyToSymbol(_settingsDev, &_settings, sizeof(CommonSettings)));

    //init gpu memory for storing strips
    //trips for each row of the depth image are stored in each column of _stripsDev. 
    //The tranpose is to minimize the downloading. 
    //TODO: might destroy coalesced access. What's the tradeoff?
    cudaSafeCall(cudaMallocHost(&_stripsHost, (MAX_STRIPS_PER_ROW + 1) * _settings.depthHeight * sizeof(_OmniTouchStripDev)));
    cudaSafeCall(cudaMalloc(&_stripsDev, (MAX_STRIPS_PER_ROW + 1) * _settings.depthHeight * sizeof(_OmniTouchStripDev)));

    //strips transposed
    cudaSafeCall(cudaMallocHost(&_transposedStripsHost, (MAX_STRIPS_PER_ROW + 1) * _settings.depthWidth * sizeof(_OmniTouchStripDev)));
    cudaSafeCall(cudaMalloc(&_transposedStripsDev, (MAX_STRIPS_PER_ROW + 1) * _settings.depthWidth * sizeof(_OmniTouchStripDev)));


    //init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
    cudaSafeCall(cudaMemcpyToSymbol(_maxHistogramSizeDev, &_maxHistogramSize, sizeof(int)));
	
}

void Detector::cudaRelease()
{
    cudaSafeCall(cudaFree(_stripsDev));
    cudaSafeCall(cudaFreeHost(_stripsHost));

    cudaSafeCall(cudaFree(_transposedStripsDev));
    cudaSafeCall(cudaFreeHost(_transposedStripsHost));
}

//update the parameters used by the algorithm
void Detector::updateDynamicParameters(const DynamicParameters& parameters)
{
	_parameters = parameters;
	
    //on device: upload parameters to device memory
    cudaSafeCall(cudaMemcpyToSymbol(_dynamicParametersDev, &parameters, sizeof(DynamicParameters)));
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
__device__ int trMaxStripRowCountDev;

template <_ImageDirection dir>
__global__ void findStripsKernel(gpu::PtrStepb debugPtr, _OmniTouchStripDev* resultPtr)
{
    extern __shared__ int stripCount[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    stripCount[threadIdx.x] = 1;

    int width = depthWidth<dir>();
    int height = depthHeight<dir>();

    if (row < height)
    {
	    StripState state = StripSmooth;
	    int partialMin, partialMax;
	    int partialMinPos, partialMaxPos;

	    for (int col = 0; col < width; col++)
	    {
		    float currVal = dirTex2D(dir, texSobel, texTrSobel, col, row);
            ushort depthVal = dirTex2D(dir, texDepth, texTrDepth, col, row);
        
		    switch(state)
		    {
		    case StripSmooth:	//TODO: smooth
                if (depthVal == 0 || depthVal == Detector::DEPTH_UNKNOWN_VALUE)
                {
                    //same state
                }
			    else if (currVal > _dynamicParametersDev[0].omniTouchParam.fingerRisingThreshold)
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
			    if (depthVal != 0 && depthVal != Detector::DEPTH_UNKNOWN_VALUE 
                    && currVal < -_dynamicParametersDev[0].omniTouchParam.fingerFallingThreshold)
			    {
				    if (currVal < partialMin)
				    {
					    partialMin = currVal;
					    partialMinPos = col;
				    }
			    }
			    else
			    {
                    ushort depth = dirTex2D(dir, texDepth, texTrDepth, (partialMaxPos + partialMinPos) / 2, row);
				
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

                        int resultOffset = stripCount[threadIdx.x] * height + row;
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
        atomicMax( (dir == DirTransposed ? &trMaxStripRowCountDev : &maxStripRowCountDev), stripCount[0]);
    }
}

template <_ImageDirection dir>
__global__ void convertScaleAbsKernel(gpu::PtrStepb debugSobelEqPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = depthWidth<dir>();
    int height = depthHeight<dir>();

    if (x < width && y < height)
    {
        float sobel = dirTex2D(dir, texSobel, texTrSobel, x, y);

        uchar res = (uchar)(fabsf(sobel) / (float)(_maxHistogramSizeDev[0]) * 255.0f + 0.5f);
        *(debugSobelEqPtr.ptr(y) + x) = res;
    }
}

template <_ImageDirection dir>
__global__ void refineDebugImageKernel(gpu::PtrStepSzb debugPtr, gpu::PtrStepb sobelEqPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = depthWidth<dir>();
    int height = depthHeight<dir>();

    if (x < width && y < height)
    {
        uchar* dstPixel = debugPtr.ptr(y) + x * 3;

        if (dstPixel[0] == 255 || dstPixel[1] == 255 || dstPixel[2] == 255)
		{
			//leave as is
		} 
		else
		{
            uchar sobelEq = *(sobelEqPtr.ptr(y) + x);
            float sobelVal = dirTex2D(dir, texSobel, texTrSobel, x, y);

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
    const float mu[nComp][2] = {{ 9.57907488860893f,          12.7703451268183f},
                                { 25.0879013587047f,          35.3988094238412f},
                                { 19.8826767803543f,          23.1472246974151f}};
    const float conv[nComp][2] = { { 47.8492848747446f,          79.3673906277784f},
                                    {  170.235884002246f,          156.732288245996f},
                                    {  56.1710526374039f,          73.2031869267223f} };
    const float prop[nComp] = { 0.328943172607604f,         0.271925127240458f,         0.399131700151939f};

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
        //*dst = p * 1000.f;
        *dst = p;
    }
}

void Detector::gpuProcess()
{
    cudaStream_t cudaStreamDepthDebug = gpu::StreamAccessor::getStream(_gpuStreamDepthDebug);
    cudaStream_t cudaStreamDepthWorking = gpu::StreamAccessor::getStream(_gpuStreamDepthWorking);
    cudaStream_t cudaStreamTransposedDepthDebug = gpu::StreamAccessor::getStream(_gpuStreamTransposedDepthDebug);
    cudaStream_t cudaStreamTransposedDepthWorking = gpu::StreamAccessor::getStream(_gpuStreamTransposedDepthWorking);
    cudaStream_t cudaStreamRgbWorking = gpu::StreamAccessor::getStream(_gpuStreamRgbWorking);
    
    _depthFrameGpu.upload(_depthFrame);
    _transposedDepthFrameGpu.upload(_transposedDepthFrame);
    //_gpuStreamDepthWorking.enqueueUpload(_depthFrame, _depthFrameGpu);
    
    //Looks like when running Sobel async, visual profiler won't generate any timeline.
    cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, _sobelFrameBufferGpu, 5, -1);
    //cv::gpu::Sobel(_depthFrameGpu, _sobelFrameGpu, CV_32F, 1, 0, _sobelFrameBufferGpu, 5, -1.0f, BORDER_DEFAULT, -1, _gpuStreamDepthWorking);
    //_gpuStreamDepthWorking.waitForCompletion();

    cv::gpu::Sobel(_transposedDepthFrameGpu, _transposedSobelFrameGpu, CV_32F, 1, 0, _transposedSobelFrameBufferGpu, 5, -1);

    _gpuStreamDepthWorking.enqueueMemSet(_debugFrameGpu, Scalar(0,0,0));
    _gpuStreamTransposedDepthWorking.enqueueMemSet(_transposedDebugFrameGpu, Scalar(0,0,0));
    
    //bind sobel for future usage
    cudaChannelFormatDesc descSobel = cudaCreateChannelDesc<float>();
    gpu::PtrStepSzb ptrSobel(_sobelFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texSobel, ptrSobel.data, descSobel, ptrSobel.cols, ptrSobel.rows, ptrSobel.step));

    cudaChannelFormatDesc descTrSobel = cudaCreateChannelDesc<float>();
    gpu::PtrStepSzb ptrTrSobel(_transposedSobelFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texTrSobel, ptrTrSobel.data, descTrSobel, ptrTrSobel.cols, ptrTrSobel.rows, ptrTrSobel.step));

    //bind depth
    cudaChannelFormatDesc descDepth = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrDepth(_depthFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texDepth, ptrDepth.data, descDepth, ptrDepth.cols, ptrDepth.rows, ptrDepth.step));
    
    cudaChannelFormatDesc descTrDepth = cudaCreateChannelDesc<ushort>();
    gpu::PtrStepSzb ptrTrDepth(_transposedDepthFrameGpu);
    cudaSafeCall(cudaBindTexture2D(NULL, texTrDepth, ptrTrDepth.data, descTrDepth, ptrTrDepth.cols, ptrTrDepth.rows, ptrTrDepth.step));


    //find strips on stream2: upload data
    //TODO: what if maximum thread < depthHeight? 
    //the third params: shared memory size in BYTES
    int* maxStripRowCountDevPtr;
    cudaSafeCall(cudaGetSymbolAddress((void**)&maxStripRowCountDevPtr, maxStripRowCountDev));
    cudaSafeCall(cudaMemsetAsync(maxStripRowCountDevPtr, 0, sizeof(int), cudaStreamDepthWorking));

    int* trMaxStripRowCountDevPtr;
    cudaSafeCall(cudaGetSymbolAddress((void**)&trMaxStripRowCountDevPtr, trMaxStripRowCountDev));
    cudaSafeCall(cudaMemsetAsync(trMaxStripRowCountDevPtr, 0, sizeof(int), cudaStreamTransposedDepthWorking));

    //find strips on stream 2: kernel call
    //turns out 1 block is the best even though profiler suggests more blocks
    int stripThread = _settings.depthHeight;    
    int stripBlock = 1; //divUp(_settings.depthHeight, nThread);
    findStripsKernel<DirDefault><<<stripBlock, stripThread, stripThread * sizeof(int), cudaStreamDepthWorking>>>(_debugFrameGpu, _stripsDev);
    
    int trStripThread = _settings.depthWidth;    
    int trStripBlock = 1; //divUp(_settings.depthHeight, nThread);
    findStripsKernel<DirTransposed><<<trStripBlock, trStripThread, trStripThread * sizeof(int), cudaStreamTransposedDepthWorking>>>
        (_transposedDebugFrameGpu, _transposedStripsDev);
    

    //rgb manipulation
    _gpuStreamRgbWorking.enqueueUpload(_rgbFrame, _rgbFrameGpu);
    _gpuStreamRgbWorking.enqueueConvert(_rgbFrameGpu, _rgbLabFrameGpu, CV_32FC3, 1.f / 255.f);
    gpu::cvtColor(_rgbLabFrameGpu, _rgbLabFrameGpu, CV_RGB2Luv, 0, _gpuStreamRgbWorking);

    //rgb skin color model
    dim3 rgbThreads(16, 32);
    dim3 rgbGrid(divUp(_settings.rgbWidth, rgbThreads.x), divUp(_settings.rgbHeight, rgbThreads.y));
    applySkinColorModel<<<rgbGrid, rgbThreads, 0, cudaStreamRgbWorking>>>(_rgbLabFrameGpu, _rgbPdfFrameGpu);
    cudaSafeCall(cudaGetLastError());

    //refine debug image
    dim3 threads(16, 32);
    dim3 grid(divUp(_settings.depthWidth, threads.x), divUp(_settings.depthHeight, threads.y));
    if (DRAW_DEBUG_IMAGE)
    {
        convertScaleAbsKernel<DirDefault><<<grid, threads, 0, cudaStreamDepthDebug>>>(_debugSobelEqFrameGpu);
        gpu::equalizeHist(_debugSobelEqFrameGpu, _debugSobelEqFrameGpu, _debugSobelEqHistGpu, _debugSobelEqBufferGpu, _gpuStreamDepthDebug);
    }

    //refine transposed debug image
    dim3 trThreads(16, 32);
    dim3 trGrid(divUp(_settings.depthHeight, threads.x), divUp(_settings.depthWidth, threads.y));
    
    if (DRAW_DEBUG_IMAGE)
    {
        convertScaleAbsKernel<DirTransposed><<<trGrid, trThreads, 0, cudaStreamTransposedDepthDebug>>>(_transposedDebugSobelEqFrameGpu);
        gpu::equalizeHist(_transposedDebugSobelEqFrameGpu, _transposedDebugSobelEqFrameGpu, _transposedDebugSobelEqHistGpu, 
            _transposedDebugSobelEqBufferGpu, _gpuStreamTransposedDepthDebug);
    }

    //rgb download
    _gpuStreamRgbWorking.enqueueDownload(_rgbPdfFrameGpu, _rgbPdfFrame);

    //find strips: download data
    cudaSafeCall(cudaMemcpyFromSymbolAsync(&_maxStripRowCount, maxStripRowCountDev, sizeof(int), 0, cudaMemcpyDeviceToHost, cudaStreamDepthWorking));
    //download strips
    //download effective data, there are maxStripCount + 1 rows. The extra row stores count of strips for each column
    cudaSafeCall(cudaMemcpyAsync(_stripsHost, _stripsDev, _maxStripRowCount * _settings.depthHeight * sizeof(_OmniTouchStripDev), 
        cudaMemcpyDeviceToHost, cudaStreamDepthWorking));
    //TODO: according to profiler, this trick seems not necessary. consider optimize for coelesence? 
  
    cudaSafeCall(cudaMemcpyFromSymbolAsync(&_transposedMaxStripRowCount, trMaxStripRowCountDev, sizeof(int), 0, 
        cudaMemcpyDeviceToHost, cudaStreamTransposedDepthWorking));
    cudaSafeCall(cudaMemcpyAsync(_transposedStripsHost, _transposedStripsDev, _transposedMaxStripRowCount * _settings.depthWidth * sizeof(_OmniTouchStripDev), 
        cudaMemcpyDeviceToHost, cudaStreamDepthWorking));
    
    if (DRAW_DEBUG_IMAGE)
    {
        _gpuStreamDepthDebug.waitForCompletion();
        _gpuStreamTransposedDepthDebug.waitForCompletion();
    }
    
    _gpuStreamDepthWorking.waitForCompletion();  
    _gpuStreamTransposedDepthWorking.waitForCompletion();
    cudaSafeCall(cudaGetLastError());

    //draw the debug image
    if (DRAW_DEBUG_IMAGE)
    {
        refineDebugImageKernel<DirDefault><<<grid, threads, 0, cudaStreamDepthDebug>>>(_debugFrameGpu, _debugSobelEqFrameGpu);
        refineDebugImageKernel<DirTransposed><<<trGrid, trThreads, 0, cudaStreamTransposedDepthDebug>>>(_transposedDebugFrameGpu, _transposedDebugSobelEqFrameGpu);
    }

    _gpuStreamDepthDebug.enqueueDownload(_debugFrameGpu, _debugFrame);
    _gpuStreamTransposedDepthDebug.enqueueDownload(_transposedDebugFrameGpu, _transposedDebugFrame);
    _gpuStreamDepthDebug.waitForCompletion();
    _gpuStreamTransposedDepthDebug.waitForCompletion();
    cudaSafeCall(cudaGetLastError());    
               
    //unbind textures
    cudaSafeCall(cudaUnbindTexture(texSobel));
    cudaSafeCall(cudaUnbindTexture(texDepth));
    cudaSafeCall(cudaUnbindTexture(texTrSobel));
    cudaSafeCall(cudaUnbindTexture(texTrDepth));
    
    _gpuStreamRgbWorking.waitForCompletion();
    
}

