# Real-Time Color Detection using OpenCV

A computer vision project that detects red, green, and blue regions in real time using a webcam. It uses HSV color segmentation, morphological dilation, and contour detection.

## Features

- Real-time webcam-based color detection
- Detection of red, green, and blue regions
- HSV color thresholding
- Morphological dilation for mask processing
- Contour-based object localization
- Bounding boxes and color labels
- Returns detection information, including bounding boxes and contour areas

## Technologies Used

- Python
- OpenCV
- NumPy

## How It Works

1. Captures frames from the webcam.
2. Converts each frame from BGR to HSV color space.
3. Creates a binary mask for each target color.
4. Applies morphological dilation.
5. Finds contours in each mask.
6. Filters contours by minimum area.
7. Draws bounding boxes and labels around detected regions.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the application:

```bash
python ColorDetectionModule.py
```

Show red, green, or blue objects to the webcam. Press `q` to exit.

## Using the Module

You can also import the detector into another Python project:

```python
import cv2
from ColorDetectionModule import ColorDetector

detector = ColorDetector()

img = cv2.imread("image.jpg")

if img is not None:
    result, detections = detector.findColors(img)

    print(detections)
    cv2.imshow("Color Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

The `findColors()` method returns the processed image and a list of detections containing each color name, bounding box, and contour area.

## Limitations

- Detection depends on lighting and the selected HSV thresholds.
- Only the configured color ranges are detected.
- Small regions with contour areas at or below the minimum threshold are ignored.

## Author

Developed as a computer vision learning project.
