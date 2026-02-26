
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. ANALISIS MODEL WARNA UNTUK APLIKASI SPESIFIK
# ==========================================================

def analyze_color_model_suitability(image, application):

    results = {}

    # Konversi berbagai model warna
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if application == 'skin_detection':
        # HSV cocok untuk deteksi warna kulit
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        results['best_model'] = 'HSV'
        results['output'] = mask
        results['reason'] = 'HSV memisahkan Hue dari intensitas cahaya.'

    elif application == 'shadow_removal':
        # YCrCb lebih stabil terhadap pencahayaan
        y, cr, cb = cv2.split(ycrcb)
        results['best_model'] = 'YCrCb'
        results['output'] = y
        results['reason'] = 'Channel Y memisahkan luminance dari warna.'

    elif application == 'text_extraction':
        # Grayscale cocok untuk thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        results['best_model'] = 'Grayscale'
        results['output'] = thresh
        results['reason'] = 'Grayscale menyederhanakan citra untuk segmentasi teks.'

    elif application == 'object_detection':
        # Edge detection dari grayscale
        edges = cv2.Canny(gray, 100, 200)
        results['best_model'] = 'Grayscale + Edge Detection'
        results['output'] = edges
        results['reason'] = 'Tepi objek lebih mudah dikenali.'

    else:
        results['best_model'] = 'Unknown'
        results['output'] = gray
        results['reason'] = 'Aplikasi tidak dikenali.'

    return results


# ==========================================================
# 2. SIMULASI EFEK ALIASING PADA CITRA
# ==========================================================

def simulate_image_aliasing(image, downsampling_factors):
    

    results = {}

    height, width = image.shape[:2]

    for factor in downsampling_factors:
        # Downsampling tanpa anti-aliasing (nearest neighbor)
        small = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_NEAREST)

        # Upscale kembali ke ukuran asli
        restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

        results[factor] = restored

    return results


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    # Ganti dengan nama file citra Anda
    image_path = "lukisan.jpeg"
    img = cv2.imread(image_path)

    if img is None:
        print("Gambar tidak ditemukan. Periksa path file.")
        exit()

    print("=== ANALISIS MODEL WARNA ===")
    result = analyze_color_model_suitability(img, 'skin_detection')
    print("Model terbaik :", result['best_model'])
    print("Alasan        :", result['reason'])

    plt.figure()
    plt.title("Hasil Analisis Warna")
    plt.imshow(result['output'], cmap='gray')
    plt.axis('off')
    plt.show()


    print("\n=== SIMULASI ALIASING ===")
    factors = [2, 4, 8]
    aliasing_results = simulate_image_aliasing(img, factors)

    for factor, aliased_img in aliasing_results.items():
        plt.figure()
        plt.title(f"Downsampling Faktor {factor}")
        plt.imshow(cv2.cvtColor(aliased_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    print("\n=== PROGRAM SELESAI ===")
