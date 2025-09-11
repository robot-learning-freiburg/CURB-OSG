#!/bin/bash
dataset1="/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-11-13-24-51-radar-oxford-10k"
dataset2="/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-14-14-15-12-radar-oxford-10k"
dataset3="/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-15-13-06-37-radar-oxford-10k"

host=$(hostname)

if [ "${host}" == 'ovomaltine' ] ; then
    echo "I am ${host}, I will run on ${dataset3}:"
    echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=stereo/left;
    echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_right;
    echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_left;
    echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_rear;
    echo "ok?";
    sleep 10 &&
    roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=stereo/left;
    roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_right;
    roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_left;
    roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset3} camera_name:=mono_rear;
fi

# if [ "${host}" == 'ovomaltine' ] ; then
#     echo "I am ${host}, I will run on ${dataset1}:"
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=stereo/left;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_right;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_left;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_rear;
#     echo "ok?";
#     sleep 10 &&
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=stereo/left;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_right;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_left;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset1} camera_name:=mono_rear;
# elif [ "${host}" == 'kaegi' ] ; then
#     echo "I am ${host}, I will run on ${dataset2}:"
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=stereo/left;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_right;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_left;
#     echo roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_rear;
#     echo "ok?";
#     sleep 10 &&
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=stereo/left;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_right;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_left;
#     roslaunch curb_projection offline_masa_processor.launch dataset_root:=${dataset2} camera_name:=mono_rear;
# fi