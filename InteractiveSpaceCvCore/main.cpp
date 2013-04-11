#include <stdio.h>
#include <stdlib.h>

#include <Windows.h> //for timer

#include "DataTypes.h"
#include "KinectSimulator.h"
#include "Detector.h"
#include "SettingsReader.h"

#include <GL/glew.h>
#include <gl/GL.h>
#include <GL/freeglut.h>

static IseCommonSettings _settings;
static IseRgbFrame _rgbFrame;
static IseDepthFrame _depthFrame;
static IseRgbFrame _debugFrame;
static GLuint _bufferObj;

void glutDisplay()
{
	static int fpsFrameCount = 0;
	static int fpsStartTime = 0;
	static int fpsCurrTime = 0;

	if (iseKinectCapture() == ERROR_KINECT_EOF)
	{
		glutLeaveMainLoop();
		return;
	}

	//rgbFrameIpl->imageData = (char*)rgbFrame.data;
	//cvShowImage(windowName, rgbFrameIpl);

	iseDetectorDetect(&_rgbFrame, &_depthFrame, &_debugFrame);

	//opengl draw
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(_settings.depthWidth, _settings.depthHeight, GL_RGB, GL_UNSIGNED_BYTE, _debugFrame.data);
	glutSwapBuffers();

	//compute fps
	fpsCurrTime = glutGet(GLUT_ELAPSED_TIME);
	fpsFrameCount++;
	if (fpsCurrTime - fpsStartTime > 1000)
	{
		double fps = fpsFrameCount * 1000.0 / (fpsCurrTime - fpsStartTime);
		fpsStartTime = fpsCurrTime;
		fpsFrameCount = 0;

		printf("FPS = %6.2f\r", fps);
	}
}

int main(int argc, char** argv)
{
	const char pathPrefix[] = "C:\\Users\\cuda\\kinect\\record\\rec130408-1700";

	//load settings
	loadCommonSettings(pathPrefix, &_settings);
	IseDynamicParameters dynamicParams;
	loadDynamicParameters(pathPrefix, &dynamicParams);

	//init rgb/depth frame
	_rgbFrame.header = iseCreateImageHeader(_settings.rgbWidth, _settings.rgbHeight, 3);
	_depthFrame.header = iseCreateImageHeader(_settings.depthWidth, _settings.depthHeight, sizeof(ushort));

	//allocate debug frame
	_debugFrame.header = iseCreateImageHeader(_settings.depthWidth, _settings.depthHeight, 3, 1);
	_debugFrame.data = (uchar*)malloc(_debugFrame.header.dataBytes);

	//init simulator and detector
	iseKinectInitWithSettings(&_settings, pathPrefix, &_rgbFrame, &_depthFrame);
	iseDetectorInitWithSettings(&_settings);
	iseDetectorUpdateDynamicParameters(&dynamicParams);

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
	iseDetectorRelease();
	iseKinectRelease();

	free(_debugFrame.data);
	_debugFrame.data = NULL;
	_debugFrame.header.isDataOwner = 0;

	return 0;
}

