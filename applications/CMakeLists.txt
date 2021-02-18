PROJECT(applications)

INCLUDE_DIRECTORIES(
    include
    ${CMAKE_SOURCE_DIR}/include
    )

SET(SOURCE_FILES
    DFG.cc
    step-35.cc
#    TGV.cc
#    Guermond.cc
#    GuermondNeumannBC.cc
#    Couette.cc
#    ThermalTGV.cc
#    MIT.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" testname ${sourcefile} )
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    DEAL_II_SETUP_TARGET(${testname})
    TARGET_LINK_LIBRARIES(${testname} rotatingMHD stdc++fs)
ENDFOREACH( sourcefile ${APP_SOURCES} )
