# ir-person-detector


This repository contains code for training and evaluating a person detection model using infrared video data from multiple streams. It uses Hydra to format the configs and easily navigate between different models. It is optimized for use on the Orin GPU.

- Training pipeline for person detection using PyTorch and Ultralytics YOLOv8
- Utilities for:
  - Frame extraction from video
  - Visualization of bounding boxes
  - Dataset loading and augmentation
- Modular scripts for training, evaluation, and inference
