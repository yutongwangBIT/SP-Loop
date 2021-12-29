# SP-Loop
## A robust and effienct loop closure detection approach for hybrid terrestrial and aerial vehicles (HyTAVs)

Our system is developed on the basis of the state-of-the-art [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion), which contains visual-inertial odometry (vins_estimator), pose graph optimization, and loop closure detection (loop_fusion). In this work, our proposed loop closure detection approach is used to replace VINS-Loop to improve robustness against viewpoint changes. The original VINS-Loop also can be run and tested in our code. The flow diagram, illustrating three main stages and the pipeline, is shown as below:.

<img src="https://github.com/yutongwangBIT/SP-Loop/blob/master/support_files/image/pipeline.jpg" width = 80% height = 80% />

Firstly, keyframes from vins_estimator are processed to extract the required SuperPoint descriptors. Secondly, a loop candidate is retrieved from the database based on the offline-trained visual vocabulary. Thirdly, the SuperGlue model is applied to find feature correspondences, which are then used in relative pose computation. Loop detection is finally predicted by examining the number of correspondences and the relative pose. 

**Features:**
- visual loop closure detection approach robust against viewpoint differences
- combining deep learning models with state-of-the-art SLAM framework

**Authors:** Yutong Wang, Bin Xu, Wei Fan, Changle Xiang from the School of Mechanical Engineering, Beijing Institute of Technology.


**Related Paper:**

* **A Robust and Efficient Loop Closure Detection Approach for Hybrid Terrestrial and Aerial Vehicles**, Yutong Wang, Bin Xu, Wei Fan, Changle Xiang (submitted to IEEE RAL)


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
Download  to YOUR_DATASET_FOLDER.
```
    roslaunch vins vins_rviz.launch
```


## 5. Docker Support
To further facilitate the building process, we add docker in our code. Docker environment is like a sandbox, thus makes our code environment-independent. To run with docker, first make sure [ros](http://wiki.ros.org/ROS/Installation) and [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) are installed on your machine. Then add your account to `docker` group by `sudo usermod -aG docker $YOUR_USER_NAME`. **Relaunch the terminal or logout and re-login if you get `Permission denied` error**, type:
```
cd ~/catkin_ws/src/VINS-Fusion/docker
make build
```
Note that the docker building process may take a while depends on your network and machine. After VINS-Fusion successfully built, you can run vins estimator with script `run.sh`.
Script `run.sh` can take several flags and arguments. Flag `-k` means KITTI, `-l` represents loop fusion, and `-g` stands for global fusion. You can get the usage details by `./run.sh -h`. Here are some examples with this script:
```
# Euroc Monocualr camera + IMU
./run.sh ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml

# Euroc Stereo cameras + IMU with loop fusion
./run.sh -l ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml

# KITTI Odometry (Stereo)
./run.sh -k ~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/

# KITTI Odometry (Stereo) with loop fusion
./run.sh -kl ~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/

#  KITTI GPS Fusion (Stereo + GPS)
./run.sh -kg ~/catkin_ws/src/VINS-Fusion/config/kitti_raw/kitti_10_03_config.yaml YOUR_DATASET_FOLDER/2011_10_03_drive_0027_sync/

```
In Euroc cases, you need open another terminal and play your bag file. If you need modify the code, simply re-run `./run.sh` with proper auguments after your changes.


## 6. Acknowledgements
We use [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) as the framework, [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/). The pre-trained weights we adopt are from [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

## 7. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

We are still working on improving the code reliability. For any technical issues, please contact Tong Qin <qintonguavATgmail.com>.

For commercial inquiries, please contact Shaojie Shen <eeshaojieATust.hk>.
