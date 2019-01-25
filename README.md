# test-regression-fpga
fast nnls in opencl for fpgas

## prerequisites
- get `eigen`. execute `hg clone https://bitbucket.org/eigen/eigen`
- get `boost`, i used 1.63... not needed in principle...

## to build
- w/ or w/o fpga exec. the difference is only in linking against intel altera libs or not.
- w/o fpga exec: `make EIGEN_HOME=<path to eigen>` and `make clean` 
- w/ fpga exec `FPGA_TEST=ON make EIGEN_HOME=<path to eigen>` and `FPGA_TEST=ON make clean`
- if boost and opencl are not in standard locations, then use:
```
FPGA_TEST=ON make EIGEN_HOME=../eigen OPENCL_LIB_DIR=/data/PAC/inteldevstack/intelFPGA_pro/hld/host/linux64/lib OPENCL_INCLUDE_DIR=/data/PAC/inteldevstack/intelFPGA_pro/hld/host/include BOOST_HOME=/cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/boost/1.63.0
```
which corresponds to 
```
FPGA_TEST=ON make EIGEN_HOME=<path to eigen> OPENCL_LIB_DIR=<path to the folder containing libOpenCL.so> OPENCL_INCLUDE_DIR=<path to opencl folder with headers> BOOST_HOME=<path to the root of boost distr>
```
