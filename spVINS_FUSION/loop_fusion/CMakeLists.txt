cmake_minimum_required(VERSION 2.8.3)
project(loop_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    vins
    )

find_package(OpenCV 3.0)
if(NOT OpenCV_FOUND)
 message(FATAL_ERROR "OpenCV > 3.0 not found.")
endif()


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(loop_fusion_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

target_link_libraries(loop_fusion_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

add_executable(sp_loop_fusion_node
    src/sp_pose_graph_node.cpp
    src/sp_pose_graph.cpp
    src/keyframe_sp.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DBoW3/Vocabulary.cpp
	src/ThirdParty/DBoW3/BowVector.cpp
	src/ThirdParty/DBoW3/Database.cpp
	src/ThirdParty/DBoW3/DescManip.cpp
	src/ThirdParty/DBoW3/FeatureVector.cpp
	src/ThirdParty/DBoW3/QueryResults.cpp
	src/ThirdParty/DBoW3/ScoringObject.cpp
	src/ThirdParty/DBoW3/quicklz.c
)

target_link_libraries(sp_loop_fusion_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}) 