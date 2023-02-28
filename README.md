# SP-Loop
## A Robust and Efficient Loop Closure Detection Approach for Hybrid Ground/Aerial Vehicles

Our system is developed on the basis of the state-of-the-art [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion), which contains visual-inertial odometry (vins_estimator), pose graph optimization, and loop closure detection (loop_fusion). In this work, our proposed loop closure detection approach is used to replace VINS-Loop to improve robustness against viewpoint changes. The flow diagram, illustrating three main stages and the pipeline, is shown as below:.

<img src="https://github.com/yutongwangBIT/SP-Loop/blob/master/support_files/image/online.png" width = 80% height = 80% />

Firstly, keyframes from vins_estimator are processed to extract the required SuperPoint descriptors. Secondly, a loop candidate is retrieved from the database based on the offline-trained visual vocabulary. Thirdly, the SuperGlue model is applied to find feature correspondences, which are then used in relative pose computation. Loop detection is finally predicted by examining the number of correspondences and the relative pose. 

**Features:**
- visual loop closure detection approach robust against viewpoint differences
- combining deep learning models with state-of-the-art SLAM framework

**Authors:** Yutong Wang, Bin Xu, Wei Fan, Changle Xiang from the School of Mechanical Engineering, Beijing Institute of Technology.


**Related Paper:**

* **Wang, Yutong, et al. "A Robust and Efficient Loop Closure Detection Approach for Hybrid Ground/Aerial Vehicles." Drones 7.2 (2023): 135.**


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04.
ROS Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)


### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **Libtorch**
Libtorch can be compiled from source code, or you can simply download the precompiled versions. The latter is recommended. There are both CPU and GPU versions:
* CPU version: we have tested with versions Libtorch1.1, 1.4 and 1.7. 1.7 may conflict with OPENCV during compilation. Libtorch 1.4 is recommended, which can be downloaded [here](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip). 
* GPU version: we have tested with CUDA 10.1, CUDNN 7.6, [Libtorch1.4](https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.4.0.zip). The versions are determined according to Libtorch version. You can also check the compatibility of your GPU driver, CUDA and Libtorch online. 

**Note**: after downloading, you should set your **own** path in the Cmakelists.txt of loop-fusion as follows:
```
set(Torch_DIR "$your own path$/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Torch version is: ${Torch_VERSION}")
```

## 2. Build SP-Loop
Clone the repository and catkin_make:
```
    mkdir ~/catkin_ws/src/SP-Loop
    cd ~/catkin_ws/src/SP-Loop
    git clone https://github.com/yutongwangBIT/SP-Loop.git
    cd ../
    catkin build
    source ~/catkin_ws/devel/setup.bash
```
(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

## 3. EuRoC Example
Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to YOUR_DATASET_FOLDER. Take MH_01 for example, you can run VINS-Fusion with three sensor types (monocular camera + IMU, stereo cameras + IMU and stereo cameras). 

### 3.1 Single Session: Monocualr camera + IMU
- Change pathes of weights (sp_path and spglue_path) to your own in ~/catkin_ws/src/SP-Loop/config/euroc/euroc_sp_loop.yaml
- Change path of bag_file to your own in ~/catkin_ws/src/SP-Loop/vins_estimator/launch/run_euroc_bag.launch
```
    roslaunch vins_estimator run_euroc_bag.launch
```
### 3.2 Multisession:
- Change pathes of weights (sp_path and spglue_path) to your own in ~/catkin_ws/src/SP-Loop/config/euroc/euroc_sp_loop.yaml
```
    roslaunch vins_estimator run_euroc_bag_multi.launch
```
play rosbags in sequence:
```
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
    rosbag play YOUR_DATASET_FOLDER/MH_02_easy.bag
    rosbag play YOUR_DATASET_FOLDER/MH_03_medium.bag
    rosbag play YOUR_DATASET_FOLDER/MH_04_difficult.bag
    rosbag play YOUR_DATASET_FOLDER/MH_05_difficult.bag
```

## 4. Real-world Data collected by a quad ducted-fan HyTAV

For security reason, we will upload our real-world data later.


## 5. Acknowledgements
We use [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) as the framework, [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/). The pre-trained weights we adopt are from [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

## 6. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

