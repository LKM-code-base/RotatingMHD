#!/bin/bash
nproc=${1:-4}

make -j$nproc
cd applications

dir="CouetteResults"
 
if [ -d "$dir" -a ! -h "$dir" ]
then
   cd $dir 
   rm *.pvtu *.vtu
   cd ..
fi

mpirun -np $nproc ./Couette
cd ..
