#pragma once
#ifndef ISE_DETECTOR_H
#define ISE_DETECTOR_H

#include "DataTypes.h"
#include <opencv2\opencv.hpp>
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

    typedef struct __IntPoint3D
    {
	    int x, y, z;
    } _IntPoint3D;	//for hit test queue

    class Detector
    {
    private:

        CommonSettings _settings;
        DynamicParameters _parameters;
        int _maxHistogramSize;
        int* _histogram;
        uchar* _floodHitTestVisitedFlag;
        std::vector<std::vector<OmniTouchStrip> > _strips;
        std::vector<OmniTouchFinger> _fingers;

        const cv::Mat& _rgbFrame;
        const cv::Mat& _depthFrame;
        cv::Mat& _debugFrame;

        cv::gpu::GpuMat _rgbFrameGpu;
        cv::gpu::GpuMat _depthFrameGpu;
        //cv::gpu::GpuMat _debugFrameGpu;   //TODO

        cv::Mat _sobelFrame;
        cv::gpu::GpuMat _sobelFrameGpu;

        inline ushort* ushortValAt(cv::Mat& mat, int row, int col);
        inline const ushort* ushortValAt(const cv::Mat& mat, int row, int col);
        inline float* floatValAt(cv::Mat& mat, int row, int col);
        inline uchar* rgb888ValAt(cv::Mat& mat, int row, int col);

        //TODO: move these to a common h file
        inline static int divUp(int total, int grain);
        inline static void cudaSafeCall(cudaError_t err);
        
        void sobel();
        void findStrips();
        void findFingers();
        void floodHitTest();
        void refineDebugImage();
        
        inline void convertProjectiveToRealWorld(int x, int y, int depth, double& rx, double& ry, double& rz);
        inline double getSquaredDistanceInRealWorld(int x1, int y1, int depth1, int x2, int y2, int depth2);

    public:
        Detector(const CommonSettings& settings, const cv::Mat& rgbFrame, const cv::Mat& depthFrame, cv::Mat& debugFrame);

        void updateDynamicParameters(const DynamicParameters& parameters);

        //debugFrame: caller should allocate it as RGB888 with size specified in common settings
        FingerDetectionResults detect();

        ~Detector();
    };
}


#endif
