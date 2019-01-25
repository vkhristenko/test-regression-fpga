#ifndef utils_h
#define utils_h

#include "cl2wrap.hpp"

namespace clapi {

std::string get_device_name(cl::Device const& d);

int get_device_type(cl::Device const& d);

}

#endif // utils_h
