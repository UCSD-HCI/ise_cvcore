#pragma once
#ifndef ISE_SETTINGS_H
#define ISE_SETTINGS_H

#include "DataTypes.h"

int loadCommonSettings(const char* pathPrefix, ise::CommonSettings* settings);
int loadDynamicParameters(const char* pathPrefix, ise::DynamicParameters* params);

#endif