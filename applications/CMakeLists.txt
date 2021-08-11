INCLUDE_DIRECTORIES(
    include
    ${CMAKE_SOURCE_DIR}/include
    )

SET(SOURCE_FILES
    DFG.cc
    step-35.cc
    TGV.cc
    Guermond.cc
    GuermondNeumannBC.cc
    Couette.cc
    ThermalTGV.cc
    MIT.cc
    Christensen.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
    PROJECT(${executablename})
    ADD_EXECUTABLE(${executablename} ${sourcefile})
    DEAL_II_SETUP_TARGET(${executablename})
    TARGET_LINK_LIBRARIES(${executablename} rotatingMHD stdc++fs)
ENDFOREACH(sourcefile ${APP_SOURCES})
