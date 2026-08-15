# ============================================================
# PCC-03: DIGITAL IMAGE AND VIDEO PROCESSING
# Lab: Spatial Filtering
# Problem: AGV Camera Feed Restoration and Enhancement
#
# Operations:
# 1. 2D Correlation
# 2. 2D Convolution
# 3. Gaussian Noise
# 4. Motion Blur
# 5. Averaging Filter
# 6. Laplacian Sharpening
# 7. Unsharp Masking / High-Boost Filtering
# 8. PSNR
# 9. Sharpness using Variance of Laplacian
# ============================================================


# ============================================================
# READ IMAGE
# ============================================================


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images in grayscale
source = cv2.imread("source.png", 0)
reference = cv2.imread("reference.png", 0)

# Check images
if source is None or reference is None:
    print("Image not found!")
    exit()

# Find histogram of source and reference
source_hist = np.bincount(source.ravel(), minlength=256)
reference_hist = np.bincount(reference.ravel(), minlength=256)

# Find CDF
source_cdf = np.cumsum(source_hist)
reference_cdf = np.cumsum(reference_hist)

# Normalize CDF
source_cdf = source_cdf / source_cdf[-1]
reference_cdf = reference_cdf / reference_cdf[-1]

# Create mapping table
mapping = np.zeros(256, dtype=np.uint8)

for i in range(256):
    difference = abs(reference_cdf - source_cdf[i])
    mapping[i] = np.argmin(difference)

# Apply mapping to source image
matched = mapping[source]

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(source, cmap="gray")
plt.title("Source Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reference, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(matched, cmap="gray")
plt.title("Matched Image")
plt.axis("off")

plt.show()
    