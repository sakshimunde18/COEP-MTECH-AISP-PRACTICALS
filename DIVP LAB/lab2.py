# ----------------------------------------------------------
# DIGITAL IMAGE PROCESSING
# Manual Implementation of Image Enhancement Techniques
# 1. Negative Transformation
# 2. Gamma (Power Law) Transformation
# 3. Log Transformation
#
# NOTE:
# Mathematical operations are implemented manually.
# OpenCV is used ONLY for reading and displaying images.
# ---------------------------------------------------------import cv2
import numpy as np
# ---------------- Image Loading ----------------
path = "DATASET/LAB_2/lab_2_image.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Unable to load image.")
    exit()

print("Image Loaded Successfully")


# ---------------- Negative ----------------
def negative(image):
    rows, cols = image.shape

    result = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = 255 - image[i, j]

    return result


# ---------------- Gamma ----------------
def gamma_transform(image, gamma):

    rows, cols = image.shape

    result = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):

            r = image[i, j] / 255.0

            s = pow(r, gamma)

            result[i, j] = np.uint8(s * 255)

    return result

# // C:\Users\Acer\OneDrive\Desktop\divp\myenv\lab2.py
# ---------------- Log ----------------
def log_transform(image):

    rows, cols = image.shape

    result = np.zeros((rows, cols), dtype=np.uint8)

    c = 255 / np.log(256)

    for i in range(rows):
        for j in range(cols):

            result[i, j] = np.uint8(c * np.log(1 + image[i, j]))

    return result


# ---------------- Menu ----------------
while True:

    print("\n========== MENU ==========")
    print("1. Negative")
    print("2. Gamma (Power Law)")
    print("3. Log Transformation")
    print("4. Exit")

    choice = int(input("Enter Choice: "))

    if choice == 1:

        output = negative(img)

        cv2.imshow("Original", img)
        cv2.imshow("Negative", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == 2:

        gamma = float(input("Enter Gamma Value: "))

        output = gamma_transform(img, gamma)

        cv2.imshow("Original", img)
        cv2.imshow("Gamma", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == 3:

        output = log_transform(img)

        cv2.imshow("Original", img)
        cv2.imshow("Log", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == 4:
        break

    else:
        print("Invalid Choice")
