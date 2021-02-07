#!/bin/bash
nproc=${1:-4}

make -j$nproc
cd applications

dir="TGVResults"
 
[ $# -eq 0 ] && { echo "Usage: $0 dir-name"; exit 1; }
 
if [ -d "$dir" -a ! -h "$dir" ]
then
   cd $dir 
   rm *.pvtu *.vtu
   cd ..
fi

mpirun -np $nproc ./TGV
cd ..
