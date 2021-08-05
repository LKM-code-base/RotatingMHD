#!/bin/bash
rm -r CMakeFiles
rm CMakeCache.txt cmake_install.cmake CTestTestfile.cmake librotatingMHD.a Makefile
cd applications
rm -r CMakeFiles
rm cmake_install.cmake Makefile Christensen Couette DFG Guermond GuermondNeumannBC MIT step-35 TGV ThermalTGV
cd ../geometries
rm -r CMakeFiles
rm Makefile cmake_install.cmake CMakeCache.txt
cd ../tests
rm -r CMakeFiles
rm cmake_install.cmake CTestTestfile.cmake Makefile timestepping_coefficients
cd ..
cmake CMakeLists.txt
