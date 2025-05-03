# Face Detection Utilities

A collection of Python scripts for face detection and cropping using OpenCV and MTCNN (Facenet-PyTorch).

## Table of Contents

* [Overview](#overview)
* [Scripts](#scripts)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Examples](#examples)
* [License](#license)

## Overview

This repository provides three standalone scripts:

1. **Haar Cascade Detector** (`detect_faces.py` / `6.py`)
   Uses OpenCV’s built-in Haar Cascade classifier to detect faces in an image, draws bounding boxes labeled with incremental IDs, and displays the annotated image in a window.

2. **MTCNN Counter** (`count_faces_mtcnn.py` / `7.py`)
   Utilizes the MTCNN face detector from `facenet-pytorch` to count faces in an image, optionally draw bounding boxes, and save an annotated output image.

3. **MTCNN Crop & Annotate** (`count_faces_mtcnn_crops.py` / `8.py`)
   Extends the counter script to:

   * Save individual face crops with timestamped filenames.
   * Optionally annotate and save the original image with bounding boxes.

## Scripts

| Filename                            | Description                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- |
| `detect_faces.py` (or `6.py`)       | Haar Cascade-based face detection & ID labeling                        |
| `count_faces_mtcnn.py` (7.py)       | MTCNN-based face counting & optional annotation                        |
| `count_faces_mtcnn_crops.py` (8.py) | MTCNN-based face counting, timestamped face crops, optional annotation |

## Requirements

* Python 3.6+
* `opencv-python` or `opencv-python-headless`
* `torch`, `torchvision`, `facenet-pytorch` (for MTCNN scripts)
* `pillow` (PIL)
* `dlib` (if using the benchmark script)

## Installation

Install dependencies via pip:

```bash
pip install opencv-python torch torchvision facenet-pytorch pillow dlib
```

> On Windows, ensure you have CMake and Visual Studio Build Tools to compile `dlib`.

## Usage

### 1. Haar Cascade Detector

```bash
python detect_faces.py --image path/to/image.jpg
```

* Opens a window showing detected faces labeled `ID 1`, `ID 2`, etc.

### 2. MTCNN Face Counter

```bash
python count_faces_mtcnn.py --image path/to/image.jpg [--output path/to/annotated.jpg]
```

* Prints the detected face count.
* Use `--output` to save an annotated image with green bounding boxes.

### 3. MTCNN Crop & Annotate

```bash
python count_faces_mtcnn_crops.py --image path/to/image.jpg \
    [--crop-dir path/to/crops] [--output path/to/annotated.jpg]
```

* Saves individual face crops into `--crop-dir` (defaults to `<image>_crops` folder) with filenames `stem_YYYYMMDD_HHMMSS_idx.jpg`.
* Use `--output` to save annotated original image.

## Examples

```bash
# Detect & label with Haar
python detect_faces.py --image group.jpg

# Count faces with MTCNN
python count_faces_mtcnn.py -i portrait.jpg

# Save cropped faces and annotated image
python count_faces_mtcnn_crops.py -i team.jpg -c face_crops -o team_annotated.jpg
```

## License

This project is released under the MIT License. Feel free to copy, modify, and distribute as you see fit.
