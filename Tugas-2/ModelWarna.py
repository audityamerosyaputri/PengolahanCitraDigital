# ==========================================================
# PROYEK MINI FINAL
# Implementasi dan Analisis Konversi Model Warna
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# LOAD CITRA
# ==========================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print("ERROR: File tidak ditemukan ->", path)
        exit()
    return img

images = {
    "Terang": load_image("buahTerang.png"),
    "Normal": load_image("buahNormal.png"),
    "Gelap": load_image("buahGelap.png")
}

# ==========================================================
# KONVERSI WARNA
# ==========================================================

def convert_color(image):
    start = time.time()

    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    end = time.time()
    return rgb, gray, hsv, lab, (end-start)

# ==========================================================
# KUANTISASI
# ==========================================================

def quant_uniform(image, level=16):
    start = time.time()
    step = 256 // level
    result = ((image // step) * step).astype(np.uint8)
    end = time.time()
    return result, (end-start)

def quant_kmeans(image, k=16):
    start = time.time()

    Z = image.reshape((-1,1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, label, center = cv2.kmeans(Z, k, None,
                                  criteria, 10,
                                  cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result = res.reshape(image.shape)

    end = time.time()
    return result, (end-start)

# ==========================================================
# SEGMENTASI HSV PALING STABIL
# ==========================================================

def segmentasi_hsv_stabil(hsv):

    h = hsv[:,:,0]
    s = hsv[:,:,1]

    # Multi warna buah
    mask1 = cv2.inRange(h, 0, 10)      # merah
    mask2 = cv2.inRange(h, 15, 40)     # kuning/oranye
    mask3 = cv2.inRange(h, 125, 155)   # ungu

    mask_warna = mask1 | mask2 | mask3
    mask_s = cv2.inRange(s, 50, 255)

    return mask_warna & mask_s

# ==========================================================
# ANALISIS MEMORI
# ==========================================================

def memory(image):
    return image.nbytes

# ==========================================================
# PROSES UTAMA
# ==========================================================

for nama, img in images.items():

    print("\n=====================================")
    print("ANALISIS:", nama)
    print("=====================================")

    print("Resolusi:", img.shape)

    rgb, gray, hsv, lab, t_conv = convert_color(img)

    gray_u, t_uni = quant_uniform(gray)
    gray_k, t_km = quant_kmeans(gray)

    mask = segmentasi_hsv_stabil(hsv)

    # ======================================================
    # INFORMASI TEKNIS
    # ======================================================

    mem_rgb = memory(img)
    mem_gray = memory(gray)
    mem_16 = gray.size * 0.5

    print("Waktu Konversi:", round(t_conv*1000,3), "ms")
    print("Waktu Uniform:", round(t_uni*1000,3), "ms")
    print("Waktu KMeans:", round(t_km*1000,3), "ms")

    print("Memori RGB:", mem_rgb, "byte")
    print("Memori Gray:", mem_gray, "byte")
    print("Estimasi 16 level:", int(mem_16), "byte")

    print("Rasio RGB:Gray =", round(mem_rgb/mem_gray,2))
    print("Rasio RGB:16-level =", round(mem_rgb/mem_16,2))

    print("Jumlah piksel terdeteksi:",
          np.sum(mask>0))

    # ======================================================
    # VISUALISASI UTAMA
    # ======================================================

    plt.figure(figsize=(12,8))
    plt.suptitle("Konversi Warna - " + nama)

    plt.subplot(2,3,1)
    plt.imshow(rgb)
    plt.title("RGB")

    plt.subplot(2,3,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")

    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title("HSV")

    plt.subplot(2,3,4)
    plt.imshow(gray_u, cmap='gray')
    plt.title("Uniform 16-Level")

    plt.subplot(2,3,5)
    plt.imshow(gray_k, cmap='gray')
    plt.title("K-Means 16 Cluster")

    plt.subplot(2,3,6)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentasi HSV")

    plt.tight_layout()
    plt.show()

# ======================================================
# REPRESENTASI MATRIKS DAN VEKTOR SEDERHANA
# ======================================================

    print("\n--- REPRESENTASI MATRIKS & VEKTOR ---")

    # Matriks RGB 5x5
    print("\nMatriks RGB 5x5 (pojok kiri atas):")
    print(img[:5, :5])

    # Matriks Grayscale 5x5
    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("\nMatriks Grayscale 5x5:")
    print(gray_temp[:5, :5])

    # Vektor flatten (25 nilai pertama)
    print("\n25 Nilai Pertama (Flatten RGB):")
    print(img.flatten()[:25])

    # Contoh reshape jadi vektor 1D
    vector_reshape = img.reshape(-1,3)
    print("\n5 Vektor RGB Pertama (hasil reshape Nx3):")
    print(vector_reshape[:5])

    # ======================================================
    # HISTOGRAM
    # ======================================================

    plt.figure(figsize=(10,4))
    plt.suptitle("Histogram - " + nama)

    plt.subplot(1,3,1)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.hist(gray_u.ravel(), bins=256)
    plt.title("Uniform")

    plt.subplot(1,3,3)
    plt.hist(gray_k.ravel(), bins=256)
    plt.title("KMeans")

    plt.tight_layout()
    plt.show()

print("\nSELESAI - PROGRAM BERHASIL DIJALANKAN")