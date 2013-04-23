#pragma once
#ifndef ISE_KINECT_SIMULATOR_H
#define ISE_KINECT_SIMULATOR_H

#include "DataTypes.h"
#include <opencv2\opencv.hpp>

class INuiSensor;

namespace ise
{

    class KinectSimulator 
    {
    private:
        
        //passed by from the caller. KinectSimulator won't release them. 
        cv::Mat& _rgbFrame;
        cv::Mat& _depthFrame;
        cv::Mat& _depthToRgbCoordFrame;
        
        cv::Mat _rgbFrameBuffer;

        const CommonSettings& _settings;

        FILE* _depthFp;
        cv::VideoCapture _rgbCapture;

        int _currentFrame;
        int _frameCount;

        INuiSensor* _sensor; //temporary for mapping coordinates

    public:
        static const int ERROR_KINECT_EOF = -10;

        KinectSimulator(const CommonSettings& settings, const char* recFilePrefix, cv::Mat& rgbFrame, cv::Mat& depthFrame, cv::Mat& depthToRgbCoordFrame);

        //read the next rgb/depth frames and store them in the buffers specified in initWithSettings
        int capture();
        inline int getCurrentFrame() const { return _currentFrame; }

        ~KinectSimulator();
    };

}

#endif