#!/bin/bash
cd /mnt/05DF881E2ADAFD37/Documents/Projekte/RotatingMHD
make -j4
cd applications
rm *.vtk *.pvtu *.vtu
mpirun -np 4 ./Couette
cd /mnt/05DF881E2ADAFD37/Documents/Projekte/RotatingMHD
