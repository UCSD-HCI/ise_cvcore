#pragma once
#ifndef ISE_DETECTOR_H
#define ISE_DETECTOR_H

#include "DataTypes.h"
#include <cv.h>
#include <opencv2\gpu\gpu.hpp>
#include <vector>
#include <driver_types.h>

namespace ise
{
    typedef enum
    {
	    StripSmooth,
	    StripRising,
	    StripMidSmooth,
	    StripFalling
    } StripState;

    typedef enum __ImageDirection
    {
        DirDefault,
        DirTransposed
    } _ImageDirection;

    typedef enum __FloodTestDirection
    {
        FloodTestNormal,
        FloodTestInversed
    } _FloodTestDirection;

    typedef enum _FingerDirection
    {
        FingerDirUp,    //default for vertical
        FingerDirLeft,  //default for horizontal (tranposed)
        FingerDirDown,
        FingerDirRight
    } FingerDirection;

    typedef struct _OmniTouchFinger
    {
	    int tipX, tipY, tipZ;
	    int endX, endY, endZ;
        float width;
        float dx, dy;
        FingerDirection direction;
        bool isTipOnSurface;
        bool isEndOnSurface;

        struct _OmniTouchFinger() : isTipOnSurface(false), isEndOnSurface(false), direction(FingerDirUp) { }
        struct _OmniTouchFinger(int tipX, int tipY, int tipZ, int endX, int endY, int endZ) : tipX(tipX), tipY(tipY), tipZ(tipZ), endX(endX), endY(endY), endZ(endZ), isTipOnSurface(false), isEndOnSurface(false), direction(FingerDirUp) { }
        bool operator<(const _OmniTouchFinger& ref) const { return endY - tipY > ref.endY - ref.tipY; }	//sort more to less     
    } OmniTouchFinger;

    typedef struct __IntPoint3D
    {
	    int x, y, z;
    } _IntPoint3D;	//for hit test queue

    typedef struct __ShortPoint2D
    {
        short x,y;
    } _ShortPoint2D; //for dfs queue of flood fill

    typedef struct __FloatPoint3D
    {
        float x, y, z;
    } _FloatPoint3D;

    typedef struct __OmniTouchStripDev
    {
        int start;
        int end;
        int row;
    } _OmniTouchStripDev;

    class Detector
    {
    public: 
        static const int MAX_STRIPS_PER_ROW = 128;
        static const int MAX_FINGER_PIXEL_LENGTH = 128; //in pixels. TODO: compute this from max finger length (in real) 
        static const int FLOOD_FILL_RADIUS = 128;
        static const double MIN_FINGER_COLOR_PDF;
        static const float MIN_STRIP_OVERLAP;
        static const float MIN_FINGER_DIR_OVERLAP;
        static const bool DRAW_DEBUG_IMAGE = true; //TODO: move to dynamic parameters
        static const ushort DEPTH_UNKNOWN_VALUE = 0xfff8;   //65528

    private:
        //settings
        CommonSettings _settings;
        DynamicParameters _parameters;
        int _maxHistogramSize;

        //data for find strips
        _OmniTouchStripDev* _stripsHost;
        _OmniTouchStripDev* _stripsDev;
        int _maxStripRowCount; //maximum strip count (+1 for count of each column) of a row in the current frame
        
        //data for find strips, transposed
        _OmniTouchStripDev* _transposedStripsHost;
        _OmniTouchStripDev* _transposedStripsDev;
        int _transposedMaxStripRowCount;
        
        //data for find fingers and flood hit
        std::vector<_OmniTouchStripDev*> _stripBuffer;      //warning: shared, must change for multi-threading
        std::vector<OmniTouchFinger> _fingers;
        uchar* _stripVisitedFlags;                          //warning: shared, must change for multi-threading
        uchar* _floodHitTestVisitedFlag;                    //warning: shared, must change for multi-threading

        //data for find fingers and flood hit, transposed
        std::vector<_OmniTouchStripDev*> _transposedStripBuffer;
        std::vector<OmniTouchFinger> _transposedFingers;
        uchar* _transposedStripVisitedFlags;
        uchar* _transposedFloodHitTestVisitedFlag;

        //external images
        const cv::Mat& _rgbFrame;
        const cv::Mat& _depthFrame;
        const cv::Mat& _depthToColorCoordFrame;
        cv::Mat& _debugFrame;
        cv::Mat& _debugFrame2;  //currently for transposed detection

        //host images
        cv::Mat _rgbPdfFrame;
        
        //host images, transposed
        cv::Mat _transposedDebugFrame;
        cv::Mat _transposedDepthFrame;
            
        //gpu images
        cv::gpu::GpuMat _rgbFrameGpu;
        cv::gpu::GpuMat _rgbLabFrameGpu;
        cv::gpu::GpuMat _rgbPdfFrameGpu;
        cv::gpu::GpuMat _depthFrameGpu;  
        cv::gpu::GpuMat _sobelFrameGpu;
        cv::gpu::GpuMat _sobelFrameBufferGpu;
        cv::gpu::GpuMat _debugFrameGpu; 
        cv::gpu::GpuMat _debugSobelEqFrameGpu;
        cv::gpu::GpuMat _debugSobelEqHistGpu;
        cv::gpu::GpuMat _debugSobelEqBufferGpu;
        
        //gpu images, transposed
        cv::gpu::GpuMat _transposedDepthFrameGpu;
        cv::gpu::GpuMat _transposedSobelFrameGpu;
        cv::gpu::GpuMat _transposedSobelFrameBufferGpu;
        cv::gpu::GpuMat _transposedDebugFrameGpu; 
        cv::gpu::GpuMat _transposedDebugSobelEqFrameGpu;
        cv::gpu::GpuMat _transposedDebugSobelEqHistGpu;
        cv::gpu::GpuMat _transposedDebugSobelEqBufferGpu;

        //gpu streams
        cv::gpu::Stream _gpuStreamDepthDebug;
        cv::gpu::Stream _gpuStreamDepthWorking;
        cv::gpu::Stream _gpuStreamTransposedDepthDebug;
        cv::gpu::Stream _gpuStreamTransposedDepthWorking;
        cv::gpu::Stream _gpuStreamRgbWorking;

        //TODO: move these to a common h file
        inline static int divUp(int total, int grain) { return (total + grain - 1) / grain; }
        inline static void cudaSafeCall(cudaError_t err);
        
        void cudaInit();
        void cudaRelease();

        void gpuProcess();

        template <_ImageDirection dir> 
        void findFingers();

        template <_FloodTestDirection dir>
        void floodHitTest();
        
        void combineFingers();
        void decideFingerDirections();

        //these inlines only used in Detector.cpp
        inline void convertProjectiveToRealWorld(int x, int y, int depth, double& rx, double& ry, double& rz);
        inline double getSquaredDistanceInRealWorld(int x1, int y1, int depth1, int x2, int y2, int depth2);
        static inline float getSegOverlapPercentage(float min1, float max1, float min2, float max2);
        static inline float pointToLineDistance(float x0, float y0, float dx, float dy, float x, float y);
        static float fingerOverlapPercentage(const _OmniTouchFinger& f1, const _OmniTouchFinger& f2, cv::Mat& debugFrame);
        static inline void amendFingerDirection(_OmniTouchFinger& f, bool flip);  

        template <_ImageDirection dir>
        void drawFingerBoundingBox(const _OmniTouchFinger& finger);

    public:
        Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, const cv::Mat& depthToColorCoordFrame, cv::Mat& debugFrame, cv::Mat& debugFrame2);

        void updateDynamicParameters(const DynamicParameters& parameters);

        //debugFrame: caller should allocate it as RGB888 with size specified in common settings
        FingerDetectionResults detect();

        ~Detector();

        //temp
        inline const cv::Mat& getPdfFrame() { return _rgbPdfFrame; }
    };
}


#endif
