# ================================================================
# PCC-03: DIGITAL IMAGE AND VIDEO PROCESSING
# Lab: Spatial Filtering
# Problem: AGV Camera Feed Restoration and Enhancement
#
# Operations implemented from scratch:
# 1. 2D Correlation
# 2. 2D Convolution
# 3. Gaussian Noise
# 4. Motion Blur
# 5. Averaging Filter
# 6. Laplacian Sharpening
# 7. Unsharp Masking / High-Boost Filtering
# 8. PSNR
# 9. Sharpness using Variance of Laplacian
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os


# ================================================================
# 1. CREATE OUTPUT FOLDER
# ================================================================

output_folder = "AGV_Output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# ================================================================
# 2. CREATE / LOAD CLEAN AGV IMAGE
# ================================================================

def create_agv_image():

    height = 360
    width = 520

    y, x = np.mgrid[0:height, 0:width]

    # Background / sky
    image = 145 + 10 * np.sin(x / 35)

    # Ground
    horizon = int(height * 0.43)

    ground = (
        92
        + 15 * np.sin(x / 22)
        + 8 * np.sin(y / 17)
    )

    image[horizon:] = ground[horizon:]

    # Road area
    road = (
        (y > horizon)
        &
        (x > width / 2 - (y - horizon) * 0.52)
        &
        (x < width / 2 + (y - horizon) * 0.52)
    )

    road_value = (
        72
        + 8 * np.sin(x / 13)
        + 5 * np.sin(y / 9)
    )

    image[road] = road_value[road]

    # Road boundaries
    for side in [-1, 1]:

        line_x = width / 2 + side * (y - horizon) * 0.52

        boundary = (
            (y > horizon)
            &
            (np.abs(x - line_x) < 3)
        )

        image[boundary] = 215

    # Center road marking
    center_x = width / 2 + 0.03 * (y - horizon)

    center_line = (
        (y > horizon + 20)
        &
        (np.abs(x - center_x) < 2.5)
        &
        (((y // 35) % 2) == 0)
    )

    image[center_line] = 220

    # Obstacles
    obstacles = [
        (105, 205, 42, 55, 45),
        (365, 192, 50, 68, 38),
        (235, 245, 45, 38, 170),
        (72, 275, 55, 30, 125),
        (405, 285, 65, 34, 150)
    ]

    for cx, cy, bw, bh, value in obstacles:

        x0 = int(cx - bw / 2)
        x1 = int(cx + bw / 2)

        y0 = int(cy - bh / 2)
        y1 = int(cy + bh / 2)

        image[y0:y1, x0:x1] = value

        # Highlight edges
        image[y0:y0 + 3, x0:x1] = min(245, value + 65)

        image[y0:y1, x0:x0 + 3] = min(245, value + 65)

    # Small ground texture
    np.random.seed(7)

    noise = np.random.normal(
        0,
        3,
        (height, width)
    )

    image[horizon:] += noise[horizon:]

    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


# Create clean image
clean_image = create_agv_image()

Image.fromarray(clean_image).save(
    output_folder + "/AGV_Clean.png"
)


# ================================================================
# 3. DISPLAY CLEAN IMAGE
# ================================================================

plt.figure(figsize=(8, 5))

plt.imshow(
    clean_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Clean AGV Ground Truth Image")
plt.axis("off")

plt.savefig(
    output_folder + "/Clean_Image.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 4. 2D CORRELATION FROM SCRATCH
# ================================================================

def correlate2d(image, kernel):

    image = image.astype(float)

    kernel = np.asarray(
        kernel,
        dtype=float
    )

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = np.pad(
        image,
        (
            (pad_h, pad_h),
            (pad_w, pad_w)
        ),
        mode="edge"
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            region = padded_image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


# ================================================================
# 5. 2D CONVOLUTION FROM SCRATCH
# ================================================================

def convolve2d(image, kernel):

    flipped_kernel = np.flipud(
        np.fliplr(kernel)
    )

    return correlate2d(
        image,
        flipped_kernel
    )


# ================================================================
# 6. CONVERT IMAGE TO 0-255
# ================================================================

def convert_to_uint8(image):

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(np.uint8)


# ================================================================
# 7. ADD GAUSSIAN NOISE
# ================================================================

def add_gaussian_noise(
        image,
        sigma,
        seed):

    np.random.seed(seed)

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    noisy_image = (
        image.astype(float)
        + noise
    )

    return convert_to_uint8(
        noisy_image
    )


# ================================================================
# 8. MOTION BLUR
# ================================================================

def motion_blur(
        image,
        length=9):

    kernel = np.zeros(
        (length, length)
    )

    # Horizontal motion
    kernel[length // 2, :] = 1 / length

    blurred_image = correlate2d(
        image,
        kernel
    )

    return convert_to_uint8(
        blurred_image
    )


# ================================================================
# 9. AVERAGING FILTER
# ================================================================

def averaging_filter(
        image,
        kernel_size):

    kernel = np.ones(
        (kernel_size, kernel_size)
    )

    kernel = kernel / (
        kernel_size * kernel_size
    )

    output = correlate2d(
        image,
        kernel
    )

    return convert_to_uint8(
        output
    )


# ================================================================
# 10. LAPLACIAN FILTERS
# ================================================================

# 4-neighbor Laplacian
laplacian_4 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])


# 8-neighbor Laplacian
laplacian_8 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])


# ================================================================
# 11. LAPLACIAN SHARPENING
# ================================================================

def laplacian_sharpen(
        image,
        kernel):

    laplacian_response = correlate2d(
        image,
        kernel
    )

    sharpened = (
        image.astype(float)
        + laplacian_response
    )

    sharpened = convert_to_uint8(
        sharpened
    )

    return sharpened, laplacian_response


# ================================================================
# 12. UNSHARP MASKING / HIGH BOOST
# ================================================================

def high_boost(
        image,
        k):

    # 3x3 averaging filter
    blur_kernel = np.ones(
        (3, 3)
    ) / 9

    blurred = correlate2d(
        image,
        blur_kernel
    )

    # Mask
    mask = (
        image.astype(float)
        - blurred
    )

    # High boost
    output = (
        image.astype(float)
        + k * mask
    )

    return convert_to_uint8(
        output
    )


# ================================================================
# 13. PSNR
# ================================================================

def calculate_psnr(
        original,
        processed):

    original = original.astype(float)

    processed = processed.astype(float)

    mse = np.mean(
        (original - processed) ** 2
    )

    if mse == 0:

        return float("inf")

    psnr_value = (
        10
        * np.log10(
            (255 ** 2) / mse
        )
    )

    return psnr_value


# ================================================================
# 14. SHARPNESS METRIC
# ================================================================

def calculate_sharpness(image):

    response = correlate2d(
        image,
        laplacian_4
    )

    # Variance of Laplacian
    sharpness = np.var(
        response
    )

    return sharpness


# ================================================================
# 15. GENERATE DEGRADED IMAGES
# ================================================================

print("\nGenerating degraded images...")

# Gaussian noise sigma = 10
noisy_sigma10 = add_gaussian_noise(
    clean_image,
    10,
    10
)

# Gaussian noise sigma = 25
noisy_sigma25 = add_gaussian_noise(
    clean_image,
    25,
    25
)

# Motion blur
motion_blurred = motion_blur(
    clean_image,
    9
)


Image.fromarray(
    noisy_sigma10
).save(
    output_folder + "/Noisy_Sigma10.png"
)


Image.fromarray(
    noisy_sigma25
).save(
    output_folder + "/Noisy_Sigma25.png"
)


Image.fromarray(
    motion_blurred
).save(
    output_folder + "/Motion_Blurred.png"
)


# ================================================================
# 16. DISPLAY DEGRADED IMAGES
# ================================================================

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)

plt.imshow(
    noisy_sigma10,
    cmap="gray"
)

plt.title("Gaussian Noise σ=10")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    noisy_sigma25,
    cmap="gray"
)

plt.title("Gaussian Noise σ=25")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    motion_blurred,
    cmap="gray"
)

plt.title("Motion Blur")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    output_folder + "/Degraded_Images.png",
    dpi=150
)

plt.show()


# ================================================================
# TASK 1
# NOISE SUPPRESSION USING AVERAGING FILTER
# ================================================================

print("\n================ TASK 1 ================")

kernel_sizes = [3, 5, 9]

task1_results = []


plt.figure(figsize=(14, 7))


# ----------------------------
# Sigma = 10
# ----------------------------

for index, size in enumerate(
        kernel_sizes):

    output = averaging_filter(
        noisy_sigma10,
        size
    )

    psnr_value = calculate_psnr(
        clean_image,
        output
    )

    sharpness_value = calculate_sharpness(
        output
    )

    task1_results.append([
        "Task 1",
        "Sigma=10",
        str(size) + "x" + str(size),
        psnr_value,
        sharpness_value
    ])

    Image.fromarray(output).save(
        output_folder
        + f"/Task1_Sigma10_{size}x{size}.png"
    )

    plt.subplot(2, 3, index + 1)

    plt.imshow(
        output,
        cmap="gray"
    )

    plt.title(
        f"Sigma=10, {size}x{size}"
    )

    plt.axis("off")


# ----------------------------
# Sigma = 25
# ----------------------------

for index, size in enumerate(
        kernel_sizes):

    output = averaging_filter(
        noisy_sigma25,
        size
    )

    psnr_value = calculate_psnr(
        clean_image,
        output
    )

    sharpness_value = calculate_sharpness(
        output
    )

    task1_results.append([
        "Task 1",
        "Sigma=25",
        str(size) + "x" + str(size),
        psnr_value,
        sharpness_value
    ])

    Image.fromarray(output).save(
        output_folder
        + f"/Task1_Sigma25_{size}x{size}.png"
    )

    plt.subplot(2, 3, index + 4)

    plt.imshow(
        output,
        cmap="gray"
    )

    plt.title(
        f"Sigma=25, {size}x{size}"
    )

    plt.axis("off")


plt.suptitle(
    "Task 1 - Averaging Filter"
)

plt.tight_layout()

plt.savefig(
    output_folder
    + "/Task1_Averaging_Grid.png",
    dpi=150
)

plt.show()


# ================================================================
# PRINT TASK 1 RESULTS
# ================================================================

print("\nTask 1 Results")

for row in task1_results:

    print(
        row[1],
        row[2],
        "PSNR =",
        round(row[3], 3),
        "dB",
        "Sharpness =",
        round(row[4], 3)
    )


# ================================================================
# TASK 2
# LAPLACIAN SHARPENING
# ================================================================

print("\n================ TASK 2 ================")


sharp_4, response_4 = laplacian_sharpen(
    motion_blurred,
    laplacian_4
)


sharp_8, response_8 = laplacian_sharpen(
    motion_blurred,
    laplacian_8
)


Image.fromarray(
    sharp_4
).save(
    output_folder
    + "/Laplacian_4_Neighbor.png"
)


Image.fromarray(
    sharp_8
).save(
    output_folder
    + "/Laplacian_8_Neighbor.png"
)


psnr_4 = calculate_psnr(
    clean_image,
    sharp_4
)

psnr_8 = calculate_psnr(
    clean_image,
    sharp_8
)


sharpness_4 = calculate_sharpness(
    sharp_4
)

sharpness_8 = calculate_sharpness(
    sharp_8
)


print(
    "4-neighbor Laplacian:"
)

print(
    "PSNR =",
    round(psnr_4, 3),
    "dB"
)

print(
    "Sharpness =",
    round(sharpness_4, 3)
)


print(
    "\n8-neighbor Laplacian:"
)

print(
    "PSNR =",
    round(psnr_8, 3),
    "dB"
)

print(
    "Sharpness =",
    round(sharpness_8, 3)
)


task2_results = [

    [
        "Task 2",
        "Motion Blur",
        "4-neighbor",
        psnr_4,
        sharpness_4
    ],

    [
        "Task 2",
        "Motion Blur",
        "8-neighbor",
        psnr_8,
        sharpness_8
    ]

]


# Display Task 2
plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)

plt.imshow(
    motion_blurred,
    cmap="gray"
)

plt.title("Motion Blurred")

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(
    response_4,
    cmap="gray"
)

plt.title("4-Neighbour Laplacian")

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(
    response_8,
    cmap="gray"
)

plt.title("8-Neighbour Laplacian")

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(
    sharp_4,
    cmap="gray"
)

plt.title("4-Neighbour Sharpened")

plt.axis("off")


plt.tight_layout()

plt.savefig(
    output_folder
    + "/Task2_Laplacian_Comparison.png",
    dpi=150
)

plt.show()


# ================================================================
# TASK 3
# UNSHARP MASKING / HIGH BOOST
# ================================================================

print("\n================ TASK 3 ================")

k_values = [
    1,
    1.5,
    2,
    3
]

task3_results = []

sharpness_values = []
psnr_values = []


plt.figure(figsize=(18, 4))


for index, k in enumerate(k_values):

    output = high_boost(
        noisy_sigma10,
        k
    )

    psnr_value = calculate_psnr(
        clean_image,
        output
    )

    sharpness_value = calculate_sharpness(
        output
    )

    task3_results.append([
        "Task 3",
        "Sigma=10",
        "k=" + str(k),
        psnr_value,
        sharpness_value
    ])

    psnr_values.append(
        psnr_value
    )

    sharpness_values.append(
        sharpness_value
    )

    Image.fromarray(
        output
    ).save(
        output_folder
        + f"/HighBoost_k{k}.png"
    )

    plt.subplot(
        1,
        4,
        index + 1
    )

    plt.imshow(
        output,
        cmap="gray"
    )

    plt.title(
        f"k = {k}"
    )

    plt.axis("off")


plt.suptitle(
    "Task 3 - Unsharp Masking / High Boost"
)

plt.tight_layout()

plt.savefig(
    output_folder
    + "/Task3_HighBoost_Grid.png",
    dpi=150
)

plt.show()


# ================================================================
# TASK 3 METRICS GRAPH
# ================================================================

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.plot(
    k_values,
    psnr_values,
    marker="o"
)

plt.xlabel(
    "Boost Factor k"
)

plt.ylabel(
    "PSNR (dB)"
)

plt.title(
    "PSNR vs k"
)

plt.grid()


plt.subplot(1, 2, 2)

plt.plot(
    k_values,
    sharpness_values,
    marker="o"
)

plt.xlabel(
    "Boost Factor k"
)

plt.ylabel(
    "Sharpness"
)

plt.title(
    "Sharpness vs k"
)

plt.grid()


plt.tight_layout()

plt.savefig(
    output_folder
    + "/Task3_Metrics_vs_k.png",
    dpi=150
)

plt.show()


# ================================================================
# PRINT TASK 3 RESULTS
# ================================================================

print("\nTask 3 Results")

for row in task3_results:

    print(
        row[2],
        "PSNR =",
        round(row[3], 3),
        "dB",
        "Sharpness =",
        round(row[4], 3)
    )


# ================================================================
# TASK 4
# CREATE COMPLETE METRICS TABLE
# ================================================================

all_results = (
    task1_results
    + task2_results
    + task3_results
)


results_dataframe = pd.DataFrame(
    all_results,
    columns=[
        "Task",
        "Input",
        "Parameter",
        "PSNR_dB",
        "Sharpness"
    ]
)


print(
    "\n================ TASK 4 ================"
)

print(
    results_dataframe.to_string(
        index=False
    )
)


results_dataframe.to_csv(
    output_folder
    + "/Task4_Metrics.csv",
    index=False
)


# ================================================================
# FIND BEST DENOISING FILTER
# ================================================================

sigma25_results = results_dataframe[
    results_dataframe["Input"] == "Sigma=25"
]


best_denoising = sigma25_results.loc[
    sigma25_results["PSNR_dB"].idxmax()
]


best_parameter = best_denoising[
    "Parameter"
]


best_psnr = best_denoising[
    "PSNR_dB"
]


print(
    "\nBest denoising filter for Sigma=25:"
)

print(
    best_parameter
)

print(
    "PSNR =",
    round(best_psnr, 3),
    "dB"
)


# ================================================================
# TASK 5
# COMPLETE PIPELINE
# ================================================================

print(
    "\n================ TASK 5 ================"
)


# Extract best kernel size
best_kernel_size = int(
    best_parameter.split("x")[0]
)


# Apply best denoising
denoised_image = averaging_filter(
    noisy_sigma10,
    best_kernel_size
)


# Test high boost values after denoising
pipeline_results = []


for k in k_values:

    final_image = high_boost(
        denoised_image,
        k
    )

    p = calculate_psnr(
        clean_image,
        final_image
    )

    s = calculate_sharpness(
        final_image
    )

    pipeline_results.append([
        k,
        p,
        s
    ])


pipeline_dataframe = pd.DataFrame(
    pipeline_results,
    columns=[
        "k",
        "PSNR_dB",
        "Sharpness"
    ]
)


# Best k according to PSNR
best_k_row = pipeline_dataframe.loc[
    pipeline_dataframe["PSNR_dB"].idxmax()
]


best_k = best_k_row["k"]


# Final recommended output
final_image = high_boost(
    denoised_image,
    best_k
)


Image.fromarray(
    denoised_image
).save(
    output_folder
    + "/Recommended_Denoised.png"
)


Image.fromarray(
    final_image
).save(
    output_folder
    + "/Recommended_Final_Image.png"
)


print(
    "\nRecommended Pipeline:"
)

print(
    "1. Gaussian noise reduction"
)

print(
    "2. Averaging filter =",
    best_kernel_size,
    "x",
    best_kernel_size
)

print(
    "3. High-Boost sharpening k =",
    best_k
)


print(
    "\nFinal PSNR =",
    round(
        best_k_row["PSNR_dB"],
        3
    ),
    "dB"
)


print(
    "Final Sharpness =",
    round(
        best_k_row["Sharpness"],
        3
    )
)


# ================================================================
# REFLECTION QUESTION 1
# SHARPEN FIRST VS DENOISE FIRST
# ================================================================

print(
    "\n================ REFLECTION Q1 ================"
)


# Sharpen first
sharpen_first = high_boost(
    noisy_sigma10,
    1
)

sharpen_then_denoise = averaging_filter(
    sharpen_first,
    5
)


# Denoise first
denoise_first = averaging_filter(
    noisy_sigma10,
    5
)

denoise_then_sharpen = high_boost(
    denoise_first,
    1
)


psnr_sharpen_first = calculate_psnr(
    clean_image,
    sharpen_then_denoise
)

psnr_denoise_first = calculate_psnr(
    clean_image,
    denoise_then_sharpen
)


print(
    "Sharpen -> Denoise PSNR:",
    round(
        psnr_sharpen_first,
        3
    ),
    "dB"
)


print(
    "Denoise -> Sharpen PSNR:",
    round(
        psnr_denoise_first,
        3
    ),
    "dB"
)


# ================================================================
# REFLECTION QUESTION 2
# CORRELATION VS CONVOLUTION
# ================================================================

print(
    "\n================ REFLECTION Q2 ================"
)


# Convolution with 4-neighbor Laplacian
conv4_response = convolve2d(
    motion_blurred,
    laplacian_4
)

conv4_sharp = convert_to_uint8(
    motion_blurred.astype(float)
    + conv4_response
)


difference_4 = np.max(
    np.abs(
        sharp_4.astype(int)
        - conv4_sharp.astype(int)
    )
)


# Convolution with 8-neighbor Laplacian
conv8_response = convolve2d(
    motion_blurred,
    laplacian_8
)

conv8_sharp = convert_to_uint8(
    motion_blurred.astype(float)
    + conv8_response
)


difference_8 = np.max(
    np.abs(
        sharp_8.astype(int)
        - conv8_sharp.astype(int)
    )
)


print(
    "Maximum difference for 4-neighbor:",
    difference_4
)

print(
    "Maximum difference for 8-neighbor:",
    difference_8
)


# ================================================================
# REFLECTION QUESTION 4
# TEST HIGHER NOISE LEVELS
# ================================================================

print(
    "\n================ REFLECTION Q4 ================"
)


noise_levels = [
    25,
    40,
    60,
    80
]


high_noise_results = []


for sigma in noise_levels:

    noisy = add_gaussian_noise(
        clean_image,
        sigma,
        sigma
    )

    sharpened = high_boost(
        noisy,
        2
    )

    noisy_psnr = calculate_psnr(
        clean_image,
        noisy
    )

    sharpened_psnr = calculate_psnr(
        clean_image,
        sharpened
    )

    high_noise_results.append([
        sigma,
        noisy_psnr,
        sharpened_psnr
    ])


high_noise_dataframe = pd.DataFrame(
    high_noise_results,
    columns=[
        "Noise Sigma",
        "Noisy PSNR",
        "After Sharpening PSNR"
    ]
)


print(
    high_noise_dataframe.to_string(
        index=False
    )
)


high_noise_dataframe.to_csv(
    output_folder
    + "/High_Noise_Experiment.csv",
    index=False
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 60)

print(
    "FINAL AGV PREPROCESSING PIPELINE"
)

print("=" * 60)

print(
    "Denoising Filter :",
    best_kernel_size,
    "x",
    best_kernel_size,
    "Averaging Filter"
)

print(
    "Sharpening Method: High-Boost Filtering"
)

print(
    "Boost Factor k   :",
    best_k
)

print(
    "Final PSNR       :",
    round(
        best_k_row["PSNR_dB"],
        3
    ),
    "dB"
)

print(
    "Final Sharpness  :",
    round(
        best_k_row["Sharpness"],
        3
    )
)

print("=" * 60)

print(
    "\nAll output files are saved inside:"
)

print(
    output_folder
)

print(
    "\nProgram completed successfully!"
)