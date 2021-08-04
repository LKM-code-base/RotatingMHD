#!/bin/bash

# adjust the compilers according to the system
C_COMPILER=mpicc
CXX_COMPILER=mpicxx
FORTRAN_COMPILER=mpif90

# number of CPUs used by make
N_CPUS=10

# change these paths as you like
BUILD_DIR=$TMPDIR
INSTALL_DIR="$HOME/software"

if [ ! -d $BUILD_DIR ]
then mkdir $BUILD_DIR && echo "Created build directory"
fi

if [ ! -d $INSTALL_DIR ]
then mkdir $INSTALL_DIR && echo "Created install directory"
fi

# Trilinos
echo "Starting with Trilinos..."
TRILINOS_DIR="$BUILD_DIR/Trilinos/"
TRILINOS_BUILD_DIR="$TRILINOS_DIR/build/"
TRILINOS_INSTALL_DIR="$INSTALL_DIR/trilinos-13-0/"
cd $BUILD_DIR
git clone --branch trilinos-release-13-0-branch https://github.com/trilinos/Trilinos.git
mkdir --parents $TRILINOS_BUILD_DIR
cd $TRILINOS_BUILD_DIR
echo "   Starting cmake for Trilinos "
echo $(pwd)
cmake \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_Fortran_COMPILER=$FORTRAN_COMPILER \
    -DCMAKE_CXX_FLAGS="-march=native -O3" \
    -DCMAKE_C_FLAGS="-march=native -O3" \
    -DCMAKE_FORTRAN_FLAGS="-march=native" \
    -DTrilinos_ENABLE_Amesos=ON \
    -DTrilinos_ENABLE_Epetra=ON \
    -DTrilinos_ENABLE_EpetraExt=ON \
    -DTrilinos_ENABLE_Ifpack=ON \
    -DTrilinos_ENABLE_AztecOO=ON \
    -DTrilinos_ENABLE_Sacado=ON \
    -DTrilinos_ENABLE_Teuchos=ON \
    -DTrilinos_ENABLE_MueLu=ON \
    -DTrilinos_ENABLE_ML=ON \
    -DTrilinos_ENABLE_ROL=ON \
    -DTrilinos_ENABLE_Tpetra=ON \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
    -DTrilinos_ENABLE_COMPLEX_FLOAT=ON \
    -DTrilinos_ENABLE_Zoltan=ON \
    -DTrilinos_VERBOSE_CONFIGURE=OFF \
    -DTPL_ENABLE_MPI=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX:PATH=$TRILINOS_INSTALL_DIR \
    $TRILINOS_DIR
echo "   Completed cmake for Trilinos "
make --jobs=$N_CPUS --max-load
make --jobs=$N_CPUS --max-load install
cd $BUILD_DIR
rm -rf $TRILINOS_BUILD_DIR
echo "Completed Trilinos"

# PETSc
echo "Starting with PETSc..."
PETSC_DIR="$BUILD_DIR/petsc/"
PETSC_INSTALL_DIR="$INSTALL_DIR/petsc-13-0/"
cd $BUILD_DIR
git clone -b v3.12.2-branch https://gitlab.com/petsc/petsc.git petsc
cd $PETSC_DIR
./configure \
    --with-cc=$C_COMPILER \
    --with-fc=$FORTRAN_COMPILER \
    --with-cxx=$CXX_COMPILER \
    --with-batch \
    --with-shared-libraries=1 \
    --with-64-bit-indices = 1 \
    --with-debugging=0 \
    --with-mpi=1 \
    --download-hypre=1 \
    CXXOPTFLAGS="-O3 -march=native" \
    COPTFLAGS="-O3 -march=native -funroll-all-loops" \
    FOPTFLAGS="-O3 -march=native -funroll-all-loops -malign-double"
make --jobs=$N_CPUS --max-load all
echo "Completed PETSc"

# p4est
echo "Starting with p4est..."
cd $BUILD_DIR
P4EST_BUILD_DIR="$BUILD_DIR/p4est-build"
P4EST_INSTALL_DIR="$INSTALL_DIR/p4est-2.2"
wget https://p4est.github.io/release/p4est-2.2.tar.gz
wget https://www.dealii.org/current/external-libs/p4est-setup.sh
chmod u+x ./p4est-setup.sh
./p4est-setup.sh ./p4est-2.2.tar.gz $P4EST_INSTALL_DIR
rm -rf p4est-2.2/
rm -f  p4est-2.2.tar.gz
rm -rf $P4EST_BUILD_DIR
rm -f ./p4est-setup.sh
echo "Completed p4est"
p4est
# deal.ii
echo "Starting with deal.ii..."
cd $BUILD_DIR
DEAL_II_DIR="$BUILD_DIR/dealii-9.2.0"
DEAL_II_INSTALL_DIR="$INSTALL_DIR/dealii-9.2.0"
DEAL_II_BUILD_DIR="$BUILD_DIR/dealii-build"
wget https://dealii.43-1.org/downloads/dealii-9.2.0.tar.gz
tar xfz dealii-9.2.0.tar.gz
mkdir --parents $DEAL_II_BUILD_DIR
cd $DEAL_II_BUILD_DIR
echo "   Starting cmake for deal.ii "
cmake \
    -DCMAKE_BUILD_TYPE="DebugRelease" \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_Fortran_COMPILER=$FORTRAN_COMPILER \
    -DCMAKE_INSTALL_PREFIX=$DEAL_II_INSTALL_DIR \
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_P4EST=ON \
    -DEAL_II_WITH_HDF5=ON \
    -DP4EST_DIR=$P4EST_INSTALL_DIR \
    -DDEAL_II_WITH_64BIT_INDICES=ON \
    -DDEAL_II_WITH_TRILINOS=ON \
    -DTRILINOS_DIR=$TRILINOS_INSTALL_DIR \
    -DCMAKE_C_FLAGS="-march=native -Wno-array-bounds" \
    -DCMAKE_CXX_FLAGS="-std=c++17 -march=native -Wno-array-bounds -Wno-literal-suffix -pthread" \
    -DDEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -DDEAL_II_CXX_FLAGS_DEBUG="-Og" \
    -DDEAL_II_LINKER_FLAGS="-lpthread" \
    $DEAL_II_DIR
echo "   Completed cmake for deal.ii "
make --jobs=$N_CPUS --max-load
make test
make --jobs=$N_CPUS --max-load install
cd $BUILD_DIR
rm -rf $DEAL_II_BUILD_DIR
echo "Completed deal.ii"
