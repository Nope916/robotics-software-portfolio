# Multi-Sensor Fusion using Kalman Filter

## Overview

This project implements a Kalman filter-based state estimation system integrating IMU and vision measurements.

The objective is to improve pose estimation accuracy by combining high-frequency IMU prediction with lower-frequency vision correction.

---

## Problem Statement

IMU integration alone suffers from drift over time.

Vision measurements provide position correction but are noisy and lower frequency.

This project fuses both sensors to achieve stable and accurate state estimation.

---

## State Model

State vector:

x = [x, y, theta, vx, vy, omega]

Where:
- x, y: position
- theta: orientation
- vx, vy: linear velocity
- omega: angular velocity

---

## System Design

### Prediction Step
- IMU acceleration input
- Gyroscope angular velocity
- Motion model propagation

### Update Step
- Vision-based position measurement
- Kalman gain computation
- Covariance update

---

## Key Features

- Low-pass filtering for gyroscope smoothing
- IMU-based orientation integration
- Full Kalman prediction & correction cycle
- Covariance propagation and update
- Modular filter implementation

---

## Technologies

- Python
- NumPy
- Matplotlib

---

## Example Results

(Add trajectory plot here)

Example observations:
- IMU-only estimation shows drift over time
- Kalman fusion reduces accumulated error
- State estimate remains stable under sensor noise

---

## Notes

This project demonstrates fundamental robotics state estimation techniques commonly used in SLAM, autonomous navigation, and humanoid control systems.
