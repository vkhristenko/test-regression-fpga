#include "include/utils.hpp"

namespace clapi {

std::string get_device_name(cl::Device const& d) {
    return d.getInfo<CL_DEVICE_NAME>();
}

int get_device_type(cl::Device const& d) {
    return d.getInfo<CL_DEVICE_TYPE>();
}

}
