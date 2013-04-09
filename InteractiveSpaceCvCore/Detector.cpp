#include "Detector.h"
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

//#define byteValAt(imgPtr, row, col) ((byte*)((imgPtr)->imageData + (row) * (imgPtr)->widthStep + col))
#define ushortValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
#define intValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width + (col))
#define rgb888ValAt(imgPtr, row, col) ((imgPtr)->data + (row) * (imgPtr)->header.width * 3 + (col) * 3)


static IseCommonSettings _settings;
static IseDynamicParameters _parameters;
static IseSobelFrame _sobelFrame;
static int _maxHistogramSize;
static int* _histogram;

void _iseHistEqualize(const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame)
{
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

void _iseDetectorSobel(const IseDepthFrame* src, IseSobelFrame* dst)
{
	static const double tpl[5][5] =	
	{
		{1, 2, 0, -2, -1},
		{4, 8, 0, -8, -4},
		{6, 12, 0, -12, -6},
		{4, 8, 0, -8, -4},
		{1, 2, 0, -2, -1}
	};

	const int tpl_offset = 2;

	for (int i = 0; i < src->header.height; i++)
	{
		for (int j = 0; j < src->header.width; j++)
		{

			double depthH = 0;
			for (int ti = 0; ti < 5; ti++)
			{
				int neighbor_row = i + ti - tpl_offset;
				if(neighbor_row < 0 || neighbor_row >= src->header.height)
					continue;

				for (int tj = 0; tj < 5; tj++)
				{
					int neighbor_col = j + tj - tpl_offset;
					if(neighbor_col < 0 || neighbor_col >= src->header.width)
						continue;

					ushort srcDepthVal = *ushortValAt(src, neighbor_row, neighbor_col);
					//depthH += tpl[ti][tj] * srcDepthVal;
					depthH += tpl[ti][tj] * (srcDepthVal == 0 ? _settings.maxDepthValue : srcDepthVal);
				}
			}

			*intValAt(dst, i, j) = (int)(depthH + 0.5);
		}
	}
}

void _iseDetectorRefineDebugImage(const IseSobelFrame* sobelPtr, IseRgbFrame* dstPtr)
{
	
	//generate histogram
	int min = 65535, max = 0;
	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		for (int j = 0; j < sobelPtr->header.width; j++)
		{
			int h = (int)abs(*intValAt(sobelPtr, i, j));
			if (h > max) max = h;
			if (h < min) min = h;
		}
	}
	
	int histogramSize = max - min + 1;
	assert(histogramSize < _maxHistogramSize);
	int histogramOffset = min;

	memset(_histogram, 0, histogramSize * sizeof(int));

	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		for (int j = 0; j < sobelPtr->header.width; j++)
		{
			int h = (int)abs(*intValAt(sobelPtr, i, j));
			_histogram[h - histogramOffset]++;
		}
	}

	for (int i = 1; i < histogramSize; i++)
	{
		_histogram[i] += _histogram[i-1];
	}

	int points = sobelPtr->header.width * sobelPtr->header.height;
	for (int i = 0; i < histogramSize; i++)
	{
		_histogram[i] = (int)(256 * ((double)_histogram[i] / (double)points) + 0.5);
	}

	//draw the image
	for (int i = 0; i < sobelPtr->header.height; i++)
	{
		uchar* dstPixel = rgb888ValAt(dstPtr, i, 0);
		for (int j = 0; j < sobelPtr->header.width; j++, dstPixel += 3)
		{
			int depth = *intValAt(sobelPtr, i, j);
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

int iseDetectorInitWithSettings(const IseCommonSettings* settings)
{
	_settings = *settings;
	//on device: upload settings to device memory

	//init sobel
	_sobelFrame.header.width = settings->depthWidth;
	_sobelFrame.header.height = settings->depthHeight;
	_sobelFrame.header.dataBytes = _sobelFrame.header.width * _sobelFrame.header.height * 4;
	_sobelFrame.header.isDataOwner = 1;
	_sobelFrame.data = (int*)malloc(_sobelFrame.header.dataBytes);

	//init histogram for debug
	_maxHistogramSize = _settings.maxDepthValue * 48 * 2;
	_histogram = (int*)malloc(_maxHistogramSize * sizeof(int));

	return 0;
}

int iseDetectorUpdateDynamicParameters(const IseDynamicParameters* parameters)
{
	_parameters = *parameters;
	//on device: upload parameters to device memory

	return 0;
}

IseFingerDetectionResults iseDetectorDetect(const IseRgbFrame* rgbFrame, const IseDepthFrame* depthFrame, IseRgbFrame* debugFrame)
{
	//_iseHistEqualize(depthFrame, debugFrame);

	_iseDetectorSobel(depthFrame, &_sobelFrame);

	_iseDetectorRefineDebugImage(&_sobelFrame, debugFrame);

	IseFingerDetectionResults r;

	r.error = 0;
	return r;
}

int iseDetectorRelease()
{
	free(_sobelFrame.data);
	_sobelFrame.header.isDataOwner = 0;

	free(_histogram);

	return 0;
}