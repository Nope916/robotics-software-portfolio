# OP3 Humanoid Robot Control (FIRA Competition)

## Overview

This project was developed for the FIRA International Robot Competition (Weightlifting Challenge).

The system controls a humanoid robot (OP3) to detect and track objects using vision and execute task-specific state-machine-based behavior.

---

## Competition Context

- Competition: FIRA International Robot Competition
- Task: Humanoid Weightlifting Challenge
- Robot Platform: ROBOTIS OP3
- Control Framework: ROS

---

## System Architecture

1. Vision Module (HSV color detection)
2. Object localization
3. Head tracking control
4. State machine behavior control
5. Motion command execution

---

## Key Features

- HSV-based object detection
- Contour filtering and centroid extraction
- Real-time head pan/tilt tracking
- State-machine-based task logic
- ROS-based robot control

---

## State Machine Design

States include:

- Searching
- Tracking
- Aligning
- Lifting
- Reset

Transitions are triggered based on vision feedback and position thresholds.

---

## Technologies

- Python
- ROS
- OpenCV
- ROBOTIS OP3

---

## Notes

This project demonstrates real-world robotics integration including perception, state control, and hardware interaction under competition constraints.
