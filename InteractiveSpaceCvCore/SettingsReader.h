#pragma once
#ifndef ISE_SETTINGS_H
#define ISE_SETTINGS_H

#include "DataTypes.h"

int loadCommonSettings(const char* pathPrefix, IseCommonSettings* settings);
int loadDynamicParameters(const char* pathPrefix, IseDynamicParameters* params);

#endif