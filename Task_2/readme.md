# Satellite Image Matching with Sentinel-2 Data

This project implements an algorithm for matching satellite images using Sentinel-2 data. The goal is to develop a model that can accurately identify and match images taken during different seasons. This functionality is crucial for applications in environmental monitoring, agriculture, and urban planning.

## Solution Explanation

The task is approached by preprocessing Sentinel-2 satellite images and employing a feature matching algorithm. The processed images are matched based on keypoints, allowing for the identification of similar regions across different seasons. 

## How the Model Works

### Dataset
A collection of Sentinel-2 images is utilized, stored in a structured format. Each image is preprocessed to ensure consistency in size and format. The project uses images with `.jp2` extensions, which are converted to `.jpg` format for easier handling.

### Preprocessing
The preprocessing involves the following steps:
- **Loading Images**: The `.jp2` images are loaded from their respective directories.
- **Resizing**: Each image is resized to a maximum dimension of 512 pixels while maintaining the aspect ratio to ensure uniformity and reduce computational load.
- **Grayscale Conversion**: The images are converted to grayscale, simplifying the feature extraction process.

### Feature Matching
The project employs a feature matching technique based on keypoints:
- **Keypoint Detection**: Keypoints are detected from the images using algorithms like ORB (Oriented FAST and Rotated BRIEF).
- **Matching Algorithm**: The Brute-Force Matcher compares keypoints between pairs of images to find matches. A ratio test is applied to filter good matches based on a specified threshold.
- **Parallel Processing**: The comparisons are performed in parallel to optimize runtime efficiency, leveraging multiple CPU cores for faster execution.





