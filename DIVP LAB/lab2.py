import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# 1. Read the image
# ---------------------------------------

img = cv2.imread("road.png", 0)

if img is None:
    print("Image not found!")
    exit()


# ---------------------------------------
# 2. Add Gaussian Noise
# ---------------------------------------

noise = np.random.normal(0, 10, img.shape)

noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)


# ---------------------------------------
# 3. Averaging Filter from Scratch
# ---------------------------------------

def averaging_filter(image, size):

    pad = size // 2

    # Add border
    padded = np.pad(image, pad, mode="edge")

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # Take small window
            window = padded[i:i+size, j:j+size]

            # Calculate average
            output[i, j] = np.mean(window)

    return output


# Apply different filters

filter3 = averaging_filter(noisy_img, 3)
filter5 = averaging_filter(noisy_img, 5)
filter9 = averaging_filter(noisy_img, 9)


# ---------------------------------------
# 4. Laplacian Sharpening
# ---------------------------------------

def laplacian_sharpen(image):

    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    padded = np.pad(image, 1, mode="edge")

    output = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            window = padded[i:i+3, j:j+3]

            output[i, j] = np.sum(window * kernel)

    sharpened = image.astype(float) + output

    sharpened = np.clip(sharpened, 0, 255)

    return sharpened.astype(np.uint8)


sharpened = laplacian_sharpen(filter5)


# ---------------------------------------
# 5. Display Results
# ---------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(noisy_img, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(filter3, cmap="gray")
plt.title("3x3 Average")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(filter5, cmap="gray")
plt.title("5x5 Average")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(filter9, cmap="gray")
plt.title("9x9 Average")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sharpened, cmap="gray")
plt.title("Laplacian Sharpened")
plt.axis("off")

plt.tight_layout()
plt.show()