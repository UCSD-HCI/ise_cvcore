#pragma once
#ifndef ISE_DEBUG_UTILS_H
#define ISE_DEBUG_UTILS_H

#include <Windows.h>
#include <iostream>
#include <sstream>

//FIXME: not work at all
#define DEBUG( s )            \
{                             \
   std::ostringstream os;    \
   os << s << std::endl;                   \
   OutputDebugString( os.str().c_str() );  \
}

#endif
