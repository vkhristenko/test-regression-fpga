#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <boost/program_options.hpp>

#include "multifit_ocl/include/cl_pretty_print.hpp"
#include "multifit_ocl/include/utils.hpp"
#include "multifit_ocl/include/inplace_fnnls.h"

using data_type = float;
using my_matrix = matrix_t<data_type>;
using my_vector = vector_t<data_type>;

template<typename T>
struct duration_measurer {
    duration_measurer(std::string const& msg)
        : msg{msg}, start{std::chrono::high_resolution_clock::now()}
    {}
    ~duration_measurer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<T>(end - start).count();
        std::cout << msg << " duration = " << duration << std::endl;
    }

    std::string msg;
    std::chrono::high_resolution_clock::time_point start;
};

std::vector<my_matrix> generate_As(unsigned int n) {
    std::vector<my_matrix> result(n);
    for (int i=0; i<n; ++i)
        result[i] = my_matrix::Random();

    return result;
}

std::vector<my_vector> generate_bs(unsigned int n) {
    std::vector<my_vector> result(n);
    for (int i=0; i<n; i++) {
        result[i] = my_vector::Random();
    }

    return result;
}

#define NUM_SAMPLES 10

template<typename T>
void print_matrix(T *pM, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << pM[i*n + j] << "   ";
        }   
        std::cout << "\n";
    }   
}

template<typename T>
void print_vector(T *pv, int n) {
    for (int i=0; i<n; ++i)
        std::cout << pv[i] << "   ";
    std::cout << std::endl;
}

template<typename T, typename Vector>
void print_vectors_side_by_side(T *pv, Vector &evec, int size) {
    for (int i=0; i<size; ++i)
        std::cout << pv[i] << "  " << evec(i) << std::endl;

}

std::vector<unsigned char> get_binary(std::string const& binary_file_name) {
    auto *fp = fopen(binary_file_name.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    std::size_t size = ftell(fp);
    std::vector<unsigned char> buffer; buffer.resize(size);
    rewind(fp);
    fread(buffer.data(), size, 1, fp);
    fclose(fp);

    return buffer;
}

std::string get_source(std::string const& source_file_name) {   
    std::ifstream is {source_file_name.c_str()};
    if (!is.is_open()) {
        std::cout << "can not open a source file" << std::endl;
        exit(1);
    }

    return std::string{std::istreambuf_iterator<char>(is),
                       std::istreambuf_iterator<char>()};
}

int main(int argc, char **argv ) {
    //
    // use boost to parse the cli args
    //
    namespace po = boost::program_options;
    po::options_description desc{"allowed program options"};
    std::string intel {"Intel"};
    desc.add_options()
        ("help", "produce help msgs")
        ("device-type", po::value<std::string>(), "a device type: ['cpu' | 'gpu' | 'fpga']")
        ("manufacturer", po::value<std::string>(&intel)->default_value("Intel"), "manufacturer of the device: ['Intel', 'Nvidia']")
        ("compile-only", po::value<bool>()->default_value(true), "if true (default) should just compile and print the compilation log")
        ("dump-source", po::value<bool>()->default_value(false), "if should dump the opencl source code to standard output")
        ("num-channels", po::value<int>()->default_value(10), "number of channels to test")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc<2) {
        std::cout << desc << std::endl;
        return 0;
    }

    auto device_type_name = vm["device-type"].as<std::string>();
    auto manufac_name = vm["manufacturer"].as<std::string>();
    auto compile_only = vm["compile-only"].as<bool>();
    auto dump_source = vm["dump-source"].as<bool>();
    auto num_channels = vm["num-channels"].as<int>();


    /*
    std::vector<std::string> args;
    try {
        args = parse_args(argc, argv);
    } catch (bad_args &e) {
        std::cout << "bad cli rguments" << std::endl;
        exit(1);
    }
    */

    // predefine several conversions
    std::map<std::string, int> typename_to_type {
        {"fpga", CL_DEVICE_TYPE_ACCELERATOR},
        {"gpu", CL_DEVICE_TYPE_GPU},
        {"cpu", CL_DEVICE_TYPE_CPU}
    };
    std::map<std::string, std::vector<std::string>> typename_to_producer {
        {"fpga", {"intel"}},
        {"gpu", {"intel", "nvidia"}}
    };

    // get all platforms and print out debug info
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    clapi::pretty_print_all(platforms, std::cout, "\t\t");
    std::cout << std::endl;
    
    // for now just use hte only platform
    auto &p = platforms[0];

    // select the right device
    auto dtype_to_use = typename_to_type[device_type_name];
    std::vector<cl::Device> devices;
    p.getDevices(dtype_to_use, &devices);
    cl::Device const* ptr_d = nullptr;
    if (devices.size() > 1) {
        for (auto const& device : devices ) {
            for (auto const& p : typename_to_producer[manufac_name]) {
                if (clapi::get_device_name(device).find(p) != std::string::npos) 
                    ptr_d = &device;
            }
        }
    } else if (devices.size() == 1){
        ptr_d = &devices[0];
    } else {
        std::cout << "no devices of type " << device_type_name << std::endl;
        exit(1);
    }
    auto &d = *ptr_d;
    
    // create a context
    cl::Context ctx {devices};

    // prepare the input data
    auto As = generate_As(num_channels);
    auto bs = generate_bs(num_channels);
    std::vector<data_type> h_vA; 
    h_vA.resize(NUM_SAMPLES * NUM_SAMPLES * num_channels);
    std::vector<data_type> h_vb; 
    h_vb.resize(NUM_SAMPLES * num_channels);
    std::vector<data_type> h_vx; 
    h_vx.resize(NUM_SAMPLES * num_channels);
    for (unsigned int i=0; i<num_channels; i++) {
        auto const& A = As[i];
        auto const& b = bs[i];

        unsigned int matrix_offset = i*NUM_SAMPLES*NUM_SAMPLES;
        unsigned int vector_offset = i*NUM_SAMPLES;
        for (int row=0; row<NUM_SAMPLES; ++row) {
            auto row_offset = row * NUM_SAMPLES;
            for (int col=0; col<NUM_SAMPLES; ++col) {
                h_vA[matrix_offset + row_offset + col] = A(row, col);
            }
            h_vb[vector_offset + row] = b(row);
        }

        // validate input
        if (i%100 == 0) {
            std::cout << "********************" << std::endl;
            std::cout << "cl input: matrix A = \n";
            print_matrix(&h_vA[matrix_offset], NUM_SAMPLES);
            std::cout << "eigen input: matrix A = \n" << A << std::endl;

            std::cout << "********************" << std::endl;
            std::cout << "cl input: vector b = \n";
            print_vector(&h_vb[vector_offset], NUM_SAMPLES);
            std::cout << "eigen input: vector b = \n" << b << std::endl;
        }
    }
    double const epsilon = 1e-11;
    unsigned int const max_iterations = 1000;
    /*
    for (std::size_t i=0; i<NUM_CHANNELS; ++i) {
        for (std::size_t j=0; j<NUM_SAMPLES; ++j) {
            for (std::size_t k=0; k<NUM_SAMPLES; ++k) {
                h_vA.push_back(j+k);
            }
            h_vb.push_back(k);
        }
    }*/

    // need to compile the device side or load an image
    int error = 0;
    cl::Program program;
    if (dtype_to_use == CL_DEVICE_TYPE_ACCELERATOR) {
        duration_measurer<std::chrono::milliseconds> raid{"program creation"};
        std::string binary_file {"inplace_fnnls_original.aocx"};
        std::cout << "trying to get binary " << binary_file << std::endl;
        auto bin = get_binary(binary_file);
        std::cout << "got a binary image of size " << bin.size() << std::endl;
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        program = cl::Program{ctx, devices, {bin}};
#else
        program = cl::Program{ctx, devices, {bin.data(), bin.size()}};
#endif
    } else {
        // for now hardcode this guy
        // should be obtained somehow, env?
        std::string path_to_source_file = "/Users/vk/software/test-regression/multifit_ocl/device";
        std::string source_file {path_to_source_file + "/" + "inplace_fnnls_original.cl"};
        auto source = get_source(source_file);
        std::cout << "got a source file: " << source.size() << " Bytes in total"<< std::endl;
        if (dump_source) {
            std::cout << "--- source start ---" << std::endl;
            std::cout << source << std::endl;
            std::cout << "--- source end ---" << std::endl;
        }
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        program = cl::Program{ctx, source};
#else
        program = cl::Program{ctx, source};
#endif
    }

    int status;
    try {
        duration_measurer<std::chrono::milliseconds> raid{"program building"};
        status = program.build(devices);
    } catch (cl::BuildError &err) {
        std::cout << "build error: " << err.what() << std::endl;
        auto log = err.getBuildLog();
        for (auto const& p : log) 
            std::cout << "log:\n"
                      << "-----------------------------------\n"
                      << p.second << "\n"
                      << "-----------------------------------\n"
                      << std::endl;
        exit(1);
    }
    if (status != CL_SUCCESS) {
        std::cout << "error building" << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d) << std::endl;
        exit(1);
    }
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d) << std::endl;
    std::cout << "a program has been built" << std::endl;

    if (compile_only) {
        std::cout << "compile only mode. set --compile-only=false to actually run" << std::endl;
        return 0;
    }

    {
        duration_measurer<std::chrono::milliseconds> raid{"fpga execution"};

        // a queue
        std::cout << "initialize a command queue" << std::endl;
        cl::CommandQueue queue{ctx, d};

        // create a kernel functor
    //    auto k_vector_add = cl::Kernel{program, "vector_add"};
        std::cout << "create a kernel wrapper" << std::endl;
        auto status_kernel = 0;
        auto k_fnnls = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, double, unsigned int>(
            program, "inplace_fnnls", &status_kernel);
        if (status_kernel != CL_SUCCESS) {
            std::cout << "failed creating a 'make_kernel' wrapper " << std::endl;
            exit(1);
        }

        // link host's memory to device buffers
        std::cout << "create buffers" << std::endl;
        cl::Buffer d_vA, d_vb, d_vx;
        d_vA = cl::Buffer{ctx, CL_MEM_READ_ONLY, h_vA.size() * sizeof(data_type)};
        d_vb = cl::Buffer{ctx, CL_MEM_READ_ONLY, h_vb.size() * sizeof(data_type)};
        d_vx = cl::Buffer{ctx, CL_MEM_WRITE_ONLY, sizeof(data_type) * h_vb.size()};

        // explicit transfer
        queue.enqueueWriteBuffer(d_vA, CL_TRUE, 0, h_vA.size() * sizeof(data_type), 
            h_vA.data());
        queue.enqueueWriteBuffer(d_vb, CL_TRUE, 0, h_vb.size() * sizeof(data_type),
            h_vb.data());

        std::cout << "launch the kernel" << std::endl;
        int const count = num_channels;
        auto event = k_fnnls(cl::EnqueueArgs{queue, 1},
             d_vA, d_vb, d_vx, static_cast<unsigned int>(num_channels), epsilon, max_iterations);
        if (status_kernel != CL_SUCCESS) {
            std::cout << "problem with launching a kernel" << std::endl;
            exit(1);
        }
        event.wait();

        std::cout << "copy the data back to the host" << std::endl;
        cl::copy(queue, d_vx, std::begin(h_vx), std::end(h_vx));
    //    queue.enqueueReadBuffer(d_c, CL_TRUE, 0, h_a.size() * sizeof(float), h_c.data());
        queue.finish();
    }

    {
        duration_measurer<std::chrono::milliseconds> raid{"cpu execution + comparison"};

        std::cout << "validate against the reference" << std::endl;
        float precision = 1e-4; // 1e-5 starts having issues
        bool all_good = true;
        for (unsigned int i=0; i<num_channels; i++) {
            auto const& A = As[i];
            auto const& b = bs[i];
            my_vector x = my_vector::Zero();
            cpu_inplace_fnnls(A, b, x);

            for (int ts=0; ts<NUM_SAMPLES; ++ts) {
                all_good &= std::abs(x(ts) - h_vx[i*NUM_SAMPLES + ts]) < precision;
            }
            
            if (i%100 == 0) {
                std::cout << "**************** " << i << " ****************" << std::endl;
                print_vectors_side_by_side(&h_vx[i*NUM_SAMPLES], x, NUM_SAMPLES);
            }
        }
        if (all_good)
            std::cout << "PASS TEST" << std::endl;
        else 
            std::cout << "FAILED TEST" << std::endl;
    }

    return 0;
}
