import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    """

    # 1. Hitung histogram
    hist = np.zeros(256)

    for pixel in image.flatten():
        hist[pixel] += 1

    # 2. Hitung cumulative histogram
    cdf = hist.cumsum()

    # Normalisasi CDF
    cdf_normalized = cdf / cdf[-1]

    # 3. Hitung transformation function
    transform_function = np.floor(255 * cdf_normalized).astype('uint8')

    # 4. Apply transformation
    equalized_image = transform_function[image]

    return equalized_image, transform_function


# =============================
# MAIN PROGRAM
# =============================

# Baca gambar grayscale
img = cv2.imread("lukisan.jpeg", cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan. Periksa path file.")
    exit()

# Terapkan histogram equalization manual
equalized_img, tf = manual_histogram_equalization(img)

# Tampilkan hasil
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Manual Histogram Equalization")
plt.axis("off")

plt.show()