# Soccer Field SLAM Simulation

## Overview

This project implements a 2D SLAM system in a simulated soccer field environment using particle filter localization and occupancy grid mapping.

The goal is to estimate the robot pose while simultaneously building a map under motion and sensor noise.

---

## Problem Setting

- Environment: Simulated soccer field
- Sensor: 2D LiDAR (ray casting simulation)
- Motion model: Noisy velocity input
- Objective: Robust localization + consistent mapping

---

## System Architecture

1. Motion Model (with noise)
2. Particle Filter Localization
3. Low Variance Resampling
4. LiDAR Ray Casting
5. Occupancy Grid Mapping

---

## Particle Filter Design

Each particle represents a hypothesis of robot pose:

x = [x, y, theta]

Algorithm steps:
1. Sample motion update
2. Compute sensor likelihood
3. Normalize weights
4. Low variance resampling

---

## Occupancy Grid Mapping

- Binary occupancy grid
- Log-odds update
- Ray tracing for free space marking
- Obstacle cell update

---

## Key Features

- Custom particle filter implementation
- Noise modeling for motion & measurement
- LiDAR ray simulation
- Map consistency under uncertainty
- Stable localization convergence

---

## Technologies

- Python
- NumPy
- Matplotlib

---

## Example Results

(Add visualization images here)

Example observations:
- Particle convergence over time
- Accurate pose estimation under noise
- Consistent occupancy map generation

---

## Notes

This project demonstrates core SLAM principles including probabilistic localization, Bayesian filtering, and grid-based mapping, without relying on external SLAM libraries.
