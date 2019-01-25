#include <iostream>

#include "include/cl_pretty_print.hpp"

#define print_info_option(os, device, option, mod) \
    os << mod << #option " = " << device.getInfo<option>() << "\n"

#define print_pretty(expr, mod) \
    std::cout << mod #expr " = " << expr << "\n"

namespace clapi {
    
void pretty_print_all(std::vector<cl::Platform> const& platforms, 
                      std::ostream &os, std::string const& indent) {
    for (auto const& p : platforms) {
        pretty_print_cl_platform(p, os);
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        std::cout << "Number of devices: " << devices.size() << std::endl;
        os << "----------------------" << "\n";
        for (auto const& d : devices) {
            pretty_print_cl_device(d, os, indent);
            os << indent << "----------------------" << "\n";
        }
    }
}

void pretty_print_cl_platform(cl::Platform const& platform, 
                              std::ostream& os) {
    os << "Platform:" << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    os << "Profile:" << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
    os << "Version:" << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
    os << "Extensions:" << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
}

void pretty_print_cl_device(cl::Device const& device,
                            std::ostream &os, std::string const& indent) {
    print_info_option(os, device, CL_DEVICE_NAME, indent);
    print_info_option(os, device, CL_DEVICE_TYPE, indent);
    print_info_option(os, device, CL_DEVICE_MAX_CLOCK_FREQUENCY, indent);
    print_info_option(os, device, CL_DEVICE_AVAILABLE, indent);
    print_info_option(os, device, CL_DEVICE_COMPILER_AVAILABLE, indent);
#if __APPLE__
    print_info_option(os, device, CL_DEVICE_OPENCL_C_VERSION, indent);
#endif
    print_info_option(os, device, CL_DEVICE_MAX_COMPUTE_UNITS, indent);
    print_info_option(os, device, CL_DEVICE_LOCAL_MEM_SIZE, indent);
    print_info_option(os, device, CL_DEVICE_GLOBAL_MEM_SIZE, indent);
    print_info_option(os, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, indent);
    print_info_option(os, device, CL_DEVICE_MAX_WORK_GROUP_SIZE, indent);

    auto work_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    std::cout << "\t\tmax work group sizes:\n";
    for (auto const& ws : work_item_sizes)
        print_pretty(ws, indent + "\t");
}

}
