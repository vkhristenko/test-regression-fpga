#ifndef cl_pretty_print_h
#define cl_pretty_print_h

#include <string>
#include <ostream>
#include <iostream>

#include "cl2wrap.hpp"

namespace clapi {

void pretty_print_all(std::vector<cl::Platform> const&, std::ostream&, std::string const&);
void pretty_print_cl_platform(cl::Platform const&, std::ostream& os = std::cout);
void pretty_print_cl_device(cl::Device const&, 
    std::ostream& os = std::cout, std::string const& indent = "\t\t");

}

#endif // cl_pretty_print_h
