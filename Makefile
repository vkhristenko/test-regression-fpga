OS=$(shell uname)
EXECS=fastnnls_nofpga
LIBS=-lboost_program_options -lboost_system 
ifeq ($(OS), Darwin)
	LIBS+=-framework OpenCL
else
	LIBS+=-lOpenCL
endif
CXX=clang++
#EIGEN_HOME=/Users/vk/software/eigen_from_bitbucket/eigen
ifeq ($(FPGA_TEST), ON)
	INTEL_FPGA_LINK_FLAGS=$(shell aocl ldflags)
$(warning compiling with intel altera linking flags)
$(warning INTEL_FPGA_LINK_FLAGS=$(INTEL_FPGA_LINK_FLAGS))
	EXECS+=fastnnls_fpga
endif
CXXFLAGS=-I$(shell pwd) --std=c++17 -I$(EIGEN_HOME) -O2
UTILS=src/cl_pretty_print.o src/utils.o

.PHONY: all clean

all: $(EXECS)

fastnnls_nofpga: $(UTILS)
	$(CXX) $(CXXFLAGS) -o $@ driver.cpp $(LIBS) $(UTILS)

fastnnls_fpga: $(UTILS)
	$(CXX) $(CXXFLAGS) -o $@ driver.cpp $(LIBS) $(UTILS) $(INTEL_FPGA_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm src/*.o
	rm $(EXECS)
