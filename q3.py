import cv2 as cv
import numpy as np
import os

# Number of points to click (changed to 4 for the four corners)
N = 4
global n
n = 0
p1 = np.empty((N, 2))
p2 = np.empty((N, 2))

im1 = cv.imread('images/c1.jpg', cv.IMREAD_REDUCED_COLOR_4)
im2 = cv.imread('images/c2.jpg', cv.IMREAD_REDUCED_COLOR_4)

im1copy = im1.copy()
im2copy = im2.copy()

# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global n
    p = param[0]
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[1], (x, y), 5, (255, 0, 0), -1)
        p[n] = (x, y)
        n += 1

print("--- Part A & B: Manual Homography ---")
print(f"Please click {N} corresponding points on Image 1, then press any key to proceed.")
cv.namedWindow('Image 1', cv.WINDOW_AUTOSIZE)
param = [p1, im1copy]
cv.setMouseCallback('Image 1', draw_circle, param)

while (1):
    cv.imshow("Image 1", im1copy)
    if n == N:
        break
    if cv.waitKey(20) & 0xFF == 27:
        break

n = 0
print(f"Please click {N} corresponding points on Image 2, then press any key to proceed.")
cv.namedWindow('Image 2', cv.WINDOW_AUTOSIZE)
param = [p2, im2copy]
cv.setMouseCallback('Image 2', draw_circle, param)

while (1):
    cv.imshow("Image 2", im2copy)
    if n == N:
        break
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()

# --- Compute Manual Homography ---
H_manual, status = cv.findHomography(p1, p2)
height, width = im2.shape[:2]

# Warp im1 to im2's perspective
im1_warped_manual = cv.warpPerspective(im1, H_manual, (width, height))
diff_manual = cv.absdiff(im2, im1_warped_manual)

cv.imwrite('results/q3_manual_warp.png', im1_warped_manual)
cv.imwrite('results/q3_manual_diff.png', diff_manual)


# --- Part C & D: Automated Homography using SIFT ---
print("\n--- Part C & D: Automated SIFT Homography ---")
gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match descriptors
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n_match in matches:
    if m.distance < 0.75 * n_match.distance:
        good_matches.append(m)

# Draw and save matches
img_matches = cv.drawMatches(im1, kp1, im2, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('results/q3_sift_matches.png', img_matches)

if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography with RANSAC
    H_auto, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # Warp and compute difference
    im1_warped_auto = cv.warpPerspective(im1, H_auto, (width, height))
    diff_auto = cv.absdiff(im2, im1_warped_auto)

    cv.imwrite('results/q3_auto_warp.png', im1_warped_auto)
    cv.imwrite('results/q3_auto_diff.png', diff_auto)
    print("Processing complete! Images saved to the 'results' folder.")
else:
    print("Not enough SIFT matches found to compute homography.")