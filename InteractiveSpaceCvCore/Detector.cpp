#include "Detector.h"
#include <stdlib.h>
#include <memory.h>

static IseCommonSettings _settings;
static IseDynamicParameters _parameters;

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

int iseDetectorInitWithSettings(const IseCommonSettings* settings)
{
	_settings = *settings;
	//on device: upload settings to device memory

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
	_iseHistEqualize(depthFrame, debugFrame);

	IseFingerDetectionResults r;

	r.error = 0;
	return r;
}

int iseDetectorRelease()
{
	return 0;
}