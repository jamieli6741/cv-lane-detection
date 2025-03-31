# Simple Lane Detection Using Classical Computer Vision

This project implements a lightweight lane detection pipeline using classical computer vision techniques such as Hough Transform and perspective transformation. It is designed to be simple, modular, and educational — with no reliance on deep learning frameworks.

## 🚗 Overview

The lane detection system processes either video or image input to identify lane lines using traditional image processing techniques. It supports two modes:
- **Hough Transform-based Detection**
- **Bird's Eye View (Perspective Transform)-based Detection**

Both pipelines include color filtering, edge detection, region-of-interest masking, and optional visualizations.

## 🔧 Features

- 🧠 Classical CV methods only (no deep learning)
- 🛣️ Handles both straight and curved lane lines
- 🎞️ Supports video and image input
- ⚙️ Easily configurable through YAML
- 🖼️ Visualization of fitted lines and processing steps

## 🗂️ Project Structure

```
📁 lane-detection/
├── lane_detection.py                 # Main detection logic (Hough and perspective)
├── perspective_trans_lane_detect.py # Perspective transformation-based detection
├── utils.py                          # Utility functions: filtering, warping, plotting
├── cfg/
│   └── my_config.yaml                # Configuration file
├── sample_images/                    # Test images
└── final_videos/                     # Output videos
```

## 🔄 Usage

### 1. Clone the repo
```bash
git clone https://github.com/your_username/lane-detection.git
cd lane-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your config
Edit `cfg/my_config.yaml` to specify:
- input video/image path
- lane detection mode (`hough` or `perspective`)
- region of interest
- transformation points

### 4. Run lane detection
```bash
python lane_detection.py
```

Output video will be saved to `final_videos/`.

## 📌 Techniques Used

- Color filtering in HSV space (yellow & white)
- Canny edge detection
- ROI masking
- Hough Line Transform
- Perspective warping
- Sliding window + polyfit

## 📁 Configuration Example (`cfg/my_config.yaml`)

```yaml
lane_detect_mode: hough
show_y: 50
roi_coords:
  - [200, 360]
  - [440, 360]
  - [550, 480]
  - [100, 480]
pixel_points:
  - [100, 480]
  - [550, 480]
  - [440, 360]
  - [200, 360]
actual_points:
  - [0, 500]
  - [300, 500]
  - [300, 0]
  - [0, 0]
```

## 📘 License

MIT License
