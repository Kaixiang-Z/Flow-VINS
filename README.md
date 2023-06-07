<!--
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-06-01 22:23:18
 * @Description: 
-->

# FLOW-VINS

## Introduction
This project is developed based on VINS-Fusion to solve the visual SLAM problem in dynamic scenes. This is also my graduation project

The work that has been completed so far is：

* add a instance segmentation model ( Yolov8n-seg ), to detect dynamic objects in scene, after acceleration by TensorRT, segment speed is reach 10ms each image(GTX 1070).
* make a hierarchical management of feature points: Limit the scale of back-end optimization by classifying feature points and improve the speed of back-end optimization， by test, the optimization speed is up to 100Hz each iterations(11th i5)
* Break away from the ROS framework as much as possible (but not completely), and merge multiple nodes into a single node.


Work to be done is：
* Adding a magnetometer to overcome the heading drift of the VIO system.
* Increase the health detection mechanism and adjust the weight of sensor data during backend optimization online.

## Environment

* ROS Noetic
* CUDA 11.2 (for Segment)
* TensorRT 8.2.5.1 (for Segment)
* OpenCV 4.7.0
* Ceres Solver
* Eigen

## Example

Clone the repository and catkin_make:

```
    cd ~/catkin_ws/src
    git clone https://github.com/ErenJaeger-01/Flow-VINS.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

### EuRoC Example
（Loop detection is currently enabled by default）
#### Monocular camera + IMU

```
    roslaunch vio_system vins_euroc_mono_imu.launch
    
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```
#### Stereo camera + IMU

```
    roslaunch vio_system vins_euroc_stereo_imu.launch
    
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

#### Stereo camera 

```
    roslaunch vio_system vins_euroc_stereo.launch
    
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

#### TUM Dynamic Example

```
    roslaunch vio_system vins_tum_rgbd.launch
    
    rosbag play YOUR_DATASET_FOLDER/rgbd_dataset_freiburg3_walking_xyz.bag
```

