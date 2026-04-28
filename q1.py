import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# Load the dataset
try:
    D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
except OSError:
    print("Error: 'lines.csv' not found. Please ensure the file is in the same folder.")
    exit()

# ==========================================
# Part (a): Total Least Squares (TLS)
# ==========================================
print("--- Part (a): Total Least Squares ---")
# The first line's data corresponds to x1 (col 0) and y1 (col 3)
x1 = D[:, 0]
y1 = D[:, 3]

# 1. Center the data by subtracting the mean
x_mean = np.mean(x1)
y_mean = np.mean(y1)

# 2. Construct the data matrix A
A = np.vstack((x1 - x_mean, y1 - y_mean)).T

# 3. Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)

# 4. The normal vector [a, b] to the line is the last row of V transpose
a, b = Vt[-1, :]

# 5. Calculate the slope (m) and intercept (c)
# The line equation is a*(x - x_mean) + b*(y - y_mean) = 0
# Rearranging for y: y = (-a/b)*x + (a/b)*x_mean + y_mean
m_tls = -a / b
c_tls = (a / b) * x_mean + y_mean

print(f"Parameters for Line 1 (TLS):")
print(f"Slope (m):     {m_tls:.4f}")
print(f"Intercept (c): {c_tls:.4f}")
print(f"Equation:      y = {m_tls:.4f}x + {c_tls:.4f}\n")


# ==========================================
# Part (b): Sequential RANSAC
# ==========================================
print("--- Part (b): Sequential RANSAC ---")

# Using the snippet provided in the assignment
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

# Scikit-learn expects X to be a 2D array, reshape it to (-1, 1)
X_rem = X_all.reshape(-1, 1).copy()
Y_rem = Y_all.copy()

# Store found lines and inliers for plotting later
found_lines = []
all_inliers_x = []
all_inliers_y = []

# Iterate 3 times to find 3 lines
for i in range(3):
    # Initialize RANSAC. 
    # Note: residual_threshold might need tweaking depending on the exact noise in lines.csv
    ransac = RANSACRegressor(min_samples=2, residual_threshold=0.5, random_state=42)
    
    # Fit the remaining data
    ransac.fit(X_rem, Y_rem)

    # Extract the line parameters
    m_ransac = ransac.estimator_.coef_[0]
    c_ransac = ransac.estimator_.intercept_
    found_lines.append((m_ransac, c_ransac))

    print(f"Line {i+1} found by RANSAC:")
    print(f"Slope (m):     {m_ransac:.4f}")
    print(f"Intercept (c): {c_ransac:.4f}")
    print(f"Equation:      y = {m_ransac:.4f}x + {c_ransac:.4f}\n")

    # Mask the consensus (find which points belong to this line)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Save inliers for plotting
    all_inliers_x.append(X_rem[inlier_mask].flatten())
    all_inliers_y.append(Y_rem[inlier_mask])

    # Keep ONLY the outliers for the next loop iteration
    X_rem = X_rem[outlier_mask]
    Y_rem = Y_rem[outlier_mask]


# ==========================================
# Optional: Plot the results to verify
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the three RANSAC lines and their inliers
colors = ['red', 'green', 'blue']
for i in range(3):
    # Plot points
    plt.scatter(all_inliers_x[i], all_inliers_y[i], color=colors[i], label=f'Line {i+1} Inliers', alpha=0.6)
    
    # Plot mathematical line
    x_line = np.array([min(all_inliers_x[i]), max(all_inliers_x[i])])
    y_line = found_lines[i][0] * x_line + found_lines[i][1]
    plt.plot(x_line, y_line, color=colors[i], linewidth=2)

# Plot the TLS line from Part A as a dashed black line for comparison
x_tls_line = np.array([min(x1), max(x1)])
y_tls_line = m_tls * x_tls_line + c_tls
plt.plot(x_tls_line, y_tls_line, color='black', linestyle='--', linewidth=2, label='TLS Line (Part A)')

plt.title('Line Fitting: TLS vs Sequential RANSAC')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()