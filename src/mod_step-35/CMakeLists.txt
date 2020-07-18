##
#  CMake script for the step-1 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "step-35_barebones")
SET(TARGET_2 "step-35_constraints")
SET(TARGET_3 "step-35_sebastian")
SET(TARGET_4 "step-35_original")
SET(TARGET_5 "step-35_mod")
SET(TARGET_6 "step-35_wip")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cc
  )

SET(DEAL_II_DIR "/home/j507/deal.ii-candi/deal.II-v9.1.1")

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.1.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

IF(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_2}.cc")
  ADD_EXECUTABLE(${TARGET_2} ${TARGET_2}.cc)
  DEAL_II_SETUP_TARGET(${TARGET_2})
ENDIF()

IF(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_3}.cc")
  ADD_EXECUTABLE(${TARGET_3} ${TARGET_3}.cc)
  DEAL_II_SETUP_TARGET(${TARGET_3})
ENDIF()

IF(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_4}.cc")
  ADD_EXECUTABLE(${TARGET_4} ${TARGET_4}.cc)
  DEAL_II_SETUP_TARGET(${TARGET_4})
ENDIF()

IF(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_5}.cc")
  ADD_EXECUTABLE(${TARGET_5} ${TARGET_5}.cc)
  DEAL_II_SETUP_TARGET(${TARGET_5})
ENDIF()

IF(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_6}.cc")
  ADD_EXECUTABLE(${TARGET_6} ${TARGET_6}.cc)
  DEAL_II_SETUP_TARGET(${TARGET_6})
ENDIF()