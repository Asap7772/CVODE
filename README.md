# CUDA ODE Solver
This work was done for the STELLOPT package of PPPL created by Sam Lazerson under Dr. Stephane Ethier. The goals of the project were the following:
- Replacing the current LSODE solver with s CVODE solver which is compatible with CUDA
- Accelerating the Jacobian and Right Hand Side functions for the solver by converting the spline interpolation function to a CUDA kernel

# How to build the project:
- source the environ file (edit if need be for missing *.so)
- run make clean
- run make all

# Dependencies
- CVODE Library installed and linked
- cuda toolkit 9.1 or greater
- gcc 6.1.0
- make
- cmake
- any dependencies required for STELLOPT

