# Climbing Route Detection & Path Planning

## Overview

This project implements a computer vision-based climbing hold detection system combined with A* path planning.

The system detects colored climbing holds using HSV segmentation and computes an optimal route between selected start and goal holds.

---

## System Pipeline

1. Image input (camera or static image)
2. HSV color segmentation
3. Morphological mask cleaning
4. Contour detection
5. Hold centroid extraction
6. Graph construction
7. A* shortest path planning

---

## Key Features

- HSV-based multi-color hold detection
- Contour filtering based on area threshold
- Centroid extraction for hold localization
- Graph-based path modeling
- A* search for optimal climbing route
- Visualization overlay of detected holds and route

---

## A* Search Design

Each detected hold is treated as a node.

Edges are formed between nodes within a maximum distance threshold.

Cost function:
- Euclidean distance between holds

Heuristic:
- Straight-line distance to goal

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Example Output

- Detected climbing holds (contours + centroids)
- Computed optimal path overlay

(Add visualization image here if available)

---

## Notes

This project demonstrates integration of perception (vision) and graph-based planning in a simplified climbing route scenario.
