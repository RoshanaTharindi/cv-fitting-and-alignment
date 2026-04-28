# cv-fitting-and-alignment
This repository contains Python implementations developed for a computer vision coursework project. The repository covers fundamental computer vision and data fitting techniques, including line fitting with noise, pinhole camera geometry, and image alignment using homography.

Table of Contents:

Overview
Dependencies
Topics Covered

1. Line Fitting (TLS & RANSAC)
2. Camera Model & Physical Sizing
3. Homography and Image Alignment

Overview:

This project tackles three main computer vision problems:

- Extracting mathematical lines from noisy 2D point scatters.
- Calculating physical dimensions of real-world objects from images using internal and external camera parameters.
- Aligning circuit board images and finding physical differences using perspective warping (Homography).

Dependencies:

To run the scripts in this repository, you will need the following Python libraries:

numpy
pandas
matplotlib
scikit-learn
opencv-python (cv2)

You can install these via pip:

pip install numpy pandas matplotlib scikit-learn opencv-python


Topics Covered:

1. Line Fitting (TLS & RANSAC)

The dataset lines.csv contains scattered 2D points belonging to three distinct lines.
Total Least Squares (TLS): Also known as Orthogonal Distance Regression, this approach is used to fit a single line by minimizing the perpendicular distance from points to the line. It is implemented using Singular Value Decomposition (SVD) on the centered data matrix.
Sequential RANSAC: To extract three different lines from a noisy flattened dataset, a sequential RANSAC (RANdom SAmple Consensus) approach is used. The algorithm iteratively finds the most dominant line, extracts the inlier consensus, and repeats the process on the remaining outliers until all three lines are modeled.

2. Camera Model & Physical Sizing

Calculates the physical size of objects (e.g., earrings) captured by a camera using the pinhole camera model. By utilizing similar triangles, the magnification factor $m$ is calculated as the ratio of the focal length $f$ to the object distance $Z$ ($m = f/Z$). Combined with the sensor pixel size, this gives a direct conversion from image pixels to real-world millimeters.

3. Homography and Image Alignment

Two images of circuit boards taken from different perspectives are aligned to find physical differences.
Manual Alignment: A custom OpenCV UI allows the user to click corresponding quadrangles in both images. A homography matrix is computed to warp the first image to the perspective of the second, and an absolute difference mask is generated.
Automated Alignment (SIFT): To improve accuracy and eliminate human error, the Scale-Invariant Feature Transform (SIFT) algorithm automatically detects keypoints and descriptors. Descriptors are matched using a Brute-Force Matcher with Lowe's ratio test, and RANSAC is used to compute a robust, sub-pixel accurate homography matrix.
