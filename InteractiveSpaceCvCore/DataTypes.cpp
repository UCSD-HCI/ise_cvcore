#include "DataTypes.h"

IseImageHeader iseCreateImageHeader(int width, int height, int bytesPerPixel, int isDataOwner)
{
	IseImageHeader r;
	r.width = width;
	r.height = height;
	r.bytesPerPixel = bytesPerPixel;
	r.dataBytes = width * height * bytesPerPixel;
	r.isDataOwner = isDataOwner;

	return r;
}