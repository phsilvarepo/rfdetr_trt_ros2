## RF-DETR ROS2 Package

This repository contains a package to run inference on ROS2 Jazzy using the RFDETR architecture and using the TensorRT SDK to allow for faster inference. This package subscribes to the /camera/image_raw (sensor_msgs/Image) topic and published to the /rfdetr/image_annotated topic a (sensor_msgs/Image) message with the detected bounding boxes.
