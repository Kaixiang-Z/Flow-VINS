'''
Author: Kx Zhang
Mailbox: kxzhang@buaa.edu.cn
Date: 2023-10-28 19:20:36
Description: 
'''
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    rviz_config_file = '/home/zkx/Work/Slam/HyperVins/src/HyperVins/config/vio_system_config.rviz'
    yaml_config_file = '/home/zkx/Work/Slam/HyperVins/src/HyperVins/config/tum/tum_mono_config.yaml'

    return LaunchDescription([
        Node(
            package='estimator',
            executable='estimator_node',
            name='estimator_node',
            arguments=[yaml_config_file],
            output='screen'
         ),
        Node(
            package='loopfusion',
            executable='loopfusion_node',
            name='loopfusion_node',
            arguments=[yaml_config_file],
        ),
        Node(
            package='segment',
            executable='segment_node',
            name='segment_node',
            arguments=[yaml_config_file],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d',rviz_config_file],
            output='screen'
        )
    ])