# test-regression-fpga
fast nnls in opencl for fpgas

## prerequisites
- get eigen. execute `hg clone https://bitbucket.org/eigen/eigen`
- they use mercury...

## to build
- w/ or w/o fpga exec. the difference is only in linking against intel altera libs or not.
- w/o fpga exec: `make EIGEN_HOME=<path to eigen>` and `make clean` 
- w/ fpga exec `FPGA_TEST=ON make EIGEN_HOME=<path to eigen>` and `FPGA_TEST=ON make clean`
