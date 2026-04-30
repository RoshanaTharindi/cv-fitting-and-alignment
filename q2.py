import cv2
import numpy as np
import os

img = cv2.imread('images/earrings.jpg')

if img is None:
    print("Error: 'earrings.jpg' not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image (earrings are dark/golden on a white background)
# Binary inverse to make the earrings white (255) and background black (0)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours (outlines) of the objects
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area to filter out any small noise, keeping the two largest
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

# --- Camera Parameters ---
f_mm = 8.0                 # Focal length
Z_mm = 720.0               # Distance from lens to object
pixel_size_mm = 0.0022     # 2.2 um converted to mm

# Magnification m = f / Z
# Real Size = Sensor Size / m = (Pixel Count * pixel_size) * (Z / f)
conversion_factor = pixel_size_mm * (Z_mm / f_mm)

print(f"Magnification (m): 1/{Z_mm/f_mm:.0f}")
print(f"Real-world size per pixel: {conversion_factor:.4f} mm/pixel\n")

for i, contour in enumerate(contours):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate physical dimensions
    real_w = w * conversion_factor
    real_h = h * conversion_factor
    
    print(f"Earring {i+1}:")
    print(f"  Pixel Dimensions:    {w}px wide x {h}px high")
    print(f"  Physical Dimensions: {real_w:.2f}mm wide x {real_h:.2f}mm high\n")
    
    # Draw bounding box and dimensions on the image for visual verification
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{real_w:.1f}x{real_h:.1f}mm", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

# Save the final image to the results folder
output_path = os.path.join('results', 'q2_earrings_measured.png')
cv2.imwrite(output_path, img)
print(f"Result image saved to: {output_path}")

# Show the image with bounding boxes
cv2.imshow("Earrings Measurement", img)
cv2.waitKey(0)
cv2.destroyAllWindows()