cmake_minimum_required(VERSION 2.8)
project(test_sp)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(OpenCV 3.0)
if(NOT OpenCV_FOUND)
 message(FATAL_ERROR "OpenCV > 3.0 not found.")
endif()

#set(Torch_DIR "/home/yutong/Downloads/pytorch/torch/share/cmake/Torch")
#set(Torch_DIR "/home/yutong/Downloads/libtorch/share/cmake/Torch")
#set(Torch_DIR "/home/yutong/.local/lib/python3.6/site-packages/torch/share/cmake/Torch")
#set(CMAKE_PREFIX_PATH "/home/yutong/.local/lib/python2.7/site-packages/torch/")
#set(Torch required PATHS /home/yutong/Downloads/libtorch/share/cmake/Torch)
#message("Torch_DIR set to: ${Torch_DIR}")

set(Torch_DIR "/home/yutong/Downloads/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Torch version is: ${Torch_VERSION}")


#set( DBoW3_LIBS "/home/yutong/slambook2/test_sp/Thirdparty/DBow3/build/src/libDBoW3.so.0.0.1" )

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(sp src/sp.cpp src/SuperPoint.cpp
	 Thirdparty/DUtils/Random.cpp
     Thirdparty/DUtils/Timestamp.cpp
     Thirdparty/DVision/BRIEF.cpp
     Thirdparty/VocabularyBinary.cpp
     Thirdparty/DVision/BRIEF_Extractor.cpp)
target_link_libraries(sp ${OpenCV_LIBS} ${TORCH_LIBRARIES})

#add_executable(test_glue src/test_glue.cpp src/SuperGlue.cpp)
#target_link_libraries(test_glue ${OpenCV_LIBS} ${TORCH_LIBRARIES})

add_executable(genVoc src/gen_voc.cpp src/SuperPoint.cpp 
		Thirdparty/DBow3_src/src/Vocabulary.cpp
		Thirdparty/DBow3_src/src/BowVector.cpp
		Thirdparty/DBow3_src/src/Database.cpp
		Thirdparty/DBow3_src/src/DescManip.cpp
		Thirdparty/DBow3_src/src/FeatureVector.cpp
		Thirdparty/DBow3_src/src/QueryResults.cpp
		Thirdparty/DBow3_src/src/ScoringObject.cpp
		Thirdparty/DBow3_src/src/quicklz.c
		)
target_link_libraries(genVoc ${OpenCV_LIBS} ${TORCH_LIBRARIES})

