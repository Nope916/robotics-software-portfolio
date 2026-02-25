# Robotics Software Portfolio

This repository contains my robotics-related software projects, including SLAM, multi-sensor fusion, humanoid robot control, and deep learning-based vision systems.

I am currently an M.S. student in Electrical Engineering focusing on robotics software, state estimation, and perception systems.

---

# 🔹 Project Overview

## 1️⃣ Soccer Field SLAM Simulation

Particle filter localization and occupancy grid mapping in a simulated soccer field environment.

### Key Features
- Particle Filter Localization
- Low Variance Resampling
- LiDAR Ray Casting
- Occupancy Grid Mapping
- Motion & Sensor Noise Modeling

### Technologies
Python, NumPy, Matplotlib

→ Folder: `slam_soccer_field/`

---

## 2️⃣ Multi-Sensor Fusion (IMU + Vision)

State estimation using a Kalman filter integrating IMU and vision measurements.

### Key Features
- Gyroscope smoothing
- IMU-based orientation integration
- Kalman filter prediction & update
- Vision measurement correction

### Technologies
Python, NumPy

→ Folder: `sensor_fusion_kalman/`

---

## 3️⃣ OP3 Humanoid Robot Control (ROS)

State-machine-based humanoid robot control using color detection and head tracking.

### Key Features
- ROS-based control
- HSV color detection
- State machine behavior design
- Head pan/tilt tracking

### Technologies
Python, ROS, OpenCV

→ Folder: `op3_robot_control/`

---

## 4️⃣ Garbage Classification CNN

Custom CNN trained with 10-fold cross validation and compared against VGG16 baseline.

### Key Features
- CNN built from scratch
- Data augmentation pipeline
- Stratified 10-fold cross validation
- Precision / Recall / F1 evaluation
- Confusion matrix analysis
- Comparison with VGG16

### Technologies
PyTorch, NumPy, Matplotlib

→ Folder: `garbage_classification_cnn/`

---

## 5️⃣ Climbing Route Detection & Path Planning

Color-based hold detection and A* path planning.

### Key Features
- HSV segmentation
- Contour detection
- Centroid extraction
- A* shortest path search

### Technologies
Python, OpenCV

→ Folder: `climbing_route_detection/`

---

# 🧠 Technical Skills

- Programming: Python, C++
- Robotics: SLAM, Particle Filter, Kalman Filter
- Vision: OpenCV, HSV segmentation
- Deep Learning: PyTorch, CNN, Cross-validation
- Systems: ROS

---

# 📌 Notes

This repository is intended to showcase structured robotics software projects rather than small coding exercises.  
Each folder represents an independent project module.
