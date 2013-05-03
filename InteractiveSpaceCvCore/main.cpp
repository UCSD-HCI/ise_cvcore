#include <stdio.h>
#include <stdlib.h>

#include "DataTypes.h"
#include "KinectSimulator.h"
#include "Detector.h"
#include "SettingsReader.h"

#include <GL/glew.h>
#include <gl/GL.h>
#include <GL/freeglut.h>

#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <cuda_runtime.h>
#include <fstream>


using namespace cv;
using namespace ise;

static CommonSettings _settings;
static cv::Mat _rgbFrame, _depthFrame, _debugFrame, _depthToRgbCoordFrame;
static KinectSimulator* _kinectSimulator;
static Detector* _detector;

void glutDisplay()
{
	static int fpsFrameCount = 0;
	static int fpsStartTime = 0;
	static int fpsCurrTime = 0;

    if (_kinectSimulator->capture() == KinectSimulator::ERROR_KINECT_EOF)
	{
		glutLeaveMainLoop();
		return;
	}

	//rgbFrameIpl->imageData = (char*)rgbFrame.data;
	//cvShowImage(windowName, rgbFrameIpl);

    _detector->detect();

	//opengl draw
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(_settings.depthWidth, _settings.depthHeight, GL_RGB, GL_UNSIGNED_BYTE, _debugFrame.data);
    //glDrawPixels(_settings.depthWidth, _settings.depthHeight, GL_LUMINANCE, GL_FLOAT, _detector->getPdfFrame().data);
	glutSwapBuffers();

    /*
    if (_kinectSimulator->getCurrentFrame() == 30)
    {
        std::ofstream of("test.csv");
        of << format(_detector->getPdfFrame(), "csv") << std::endl;
        of.close();
    }*/

	//compute fps
	fpsCurrTime = glutGet(GLUT_ELAPSED_TIME);
	fpsFrameCount++;
	if (fpsCurrTime - fpsStartTime > 500)
	{
		double fps = fpsFrameCount * 1000.0 / (fpsCurrTime - fpsStartTime);
		fpsStartTime = fpsCurrTime;
		fpsFrameCount = 0;

		printf("FPS = %6.2f\r", fps);
	}

    Sleep(100);
}

int main(int argc, char** argv)
{
    const char pathPrefix[] = "C:\\Users\\cuda\\kinect\\record\\rec130421-2139";      //test color
	//const char pathPrefix[] = "C:\\Users\\cuda\\kinect\\record\\rec130412-2036";    //normal
    //const char pathPrefix[] = "C:\\Users\\cuda\\kinect\\record\\rec130417-1429";    //crazy

	//load settings
	loadCommonSettings(pathPrefix, &_settings);
	DynamicParameters dynamicParams;
	loadDynamicParameters(pathPrefix, &dynamicParams);

	//init rgb/depth frame  
	_rgbFrame.create(_settings.rgbHeight, _settings.rgbWidth, CV_8UC3);
	_depthFrame.create(_settings.depthHeight, _settings.depthWidth, CV_16U);
	_debugFrame.create(_settings.depthHeight, _settings.depthWidth, CV_8UC3);

    //init coord frame
    _depthToRgbCoordFrame.create(_settings.depthHeight, _settings.depthWidth, CV_32SC2);

    try
    {
        gpu::registerPageLocked(_rgbFrame);
        gpu::registerPageLocked(_depthFrame);
        gpu::registerPageLocked(_debugFrame);
    }
    catch (cv::Exception e)
    {
        printf(e.what());
    }

	//init simulator and detector
    _kinectSimulator = new KinectSimulator(_settings, pathPrefix, _rgbFrame, _depthFrame, _depthToRgbCoordFrame);
    _detector = new Detector(_settings, _rgbFrame, _depthFrame, _depthToRgbCoordFrame, _debugFrame);
    _detector->updateDynamicParameters(dynamicParams);

	//init glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(_settings.depthWidth, _settings.depthHeight);
	glutInitWindowPosition(500, 500);
	glutCreateWindow("Window");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	//set opengl parameters
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glRasterPos2i(-1, 1);
	glPixelZoom(1.0f, -1.0f);

	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutMainLoop();	

	//release resources
	delete _detector;
    _detector = NULL;

	delete _kinectSimulator;
    _kinectSimulator = NULL;
    
    gpu::unregisterPageLocked(_depthFrame);
    gpu::unregisterPageLocked(_rgbFrame);
    gpu::unregisterPageLocked(_debugFrame);

	_rgbFrame.release();
	_depthFrame.release();
	_debugFrame.release();
    _depthToRgbCoordFrame.release();
    
    //system("PAUSE");

	return 0;
}

