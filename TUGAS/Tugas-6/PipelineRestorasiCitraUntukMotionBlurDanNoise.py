# =====================================================
# PIPELINE RESTORASI CITRA UNTUK MOTION BLUR DAN NOISE
# =====================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy

# ==========================================================
# CITRA ASLI
# ==========================================================

image = cv2.imread("Bunga.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Gambar tidak ditemukan")
    exit()

image = cv2.resize(image, (256,256))


# ==========================================================
# MEMBUAT PSF MOTION BLUR
# ==========================================================

def motion_psf(length=15, angle=30):

    psf = np.zeros((length, length))
    center = length // 2

    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))

        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1

    psf /= psf.sum()
    return psf


psf = motion_psf(15,30)


# ==========================================================
# FUNGSI DEGRADASI
# ==========================================================

def apply_motion_blur(img, psf):
    return fftconvolve(img, psf, mode='same')


def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy,0,255)


def add_sp_noise(img, prob=0.05):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)

    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255

    return noisy


# ==========================================================
# MEMBUAT VARIASI DEGRADASI
# ==========================================================

blur_only = apply_motion_blur(image, psf)

gauss_blur = apply_motion_blur(add_gaussian_noise(image), psf)

sp_blur = apply_motion_blur(add_sp_noise(image), psf)


# ==========================================================
# RESTORASI
# ==========================================================

def inverse_filter(img, psf, eps=1e-3):

    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)

    F_hat = G / (H + eps)

    f = np.fft.ifft2(F_hat)
    return np.abs(f)


def wiener_filter(img, psf, K=0.01):

    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)

    H_conj = np.conj(H)

    F_hat = (H_conj / (np.abs(H)**2 + K)) * G

    f = np.fft.ifft2(F_hat)
    return np.abs(f)


def lucy_restoration(img, psf, iter=10):
    img = img / 255.0
    result = richardson_lucy(img, psf, num_iter=iter)
    return np.clip(result*255,0,255)


# ==========================================================
# METRIK EVALUASI
# ==========================================================

def mse(a,b):
    return np.mean((a-b)**2)


def psnr(a,b):
    m = mse(a,b)
    if m == 0:
        return 100
    return 20*np.log10(255.0/np.sqrt(m))


def ssim_val(a,b):
    return ssim(a.astype(np.uint8), b.astype(np.uint8))


# ==========================================================
# PROSES RESTORASI DAN EVALUASI
# ==========================================================

datasets = {
    "Motion Blur": blur_only,
    "Gaussian + Blur": gauss_blur,
    "SP + Blur": sp_blur
}

methods = {
    "Inverse": inverse_filter,
    "Wiener": wiener_filter,
    "Lucy": lucy_restoration
}

results = []

for name, img in datasets.items():

    for mname, method in methods.items():

        start = time.time()

        if mname == "Lucy":
            restored = method(img, psf)
        else:
            restored = method(img, psf)

        end = time.time()

        results.append([
            name,
            mname,
            mse(image, restored),
            psnr(image, restored),
            ssim_val(image, restored),
            end-start
        ])


# ==========================================================
# TABEL HASIL
# ==========================================================

print("\nTABEL HASIL RESTORASI")
print("="*90)
print(f"{'Data':20} {'Metode':15} {'MSE':10} {'PSNR':10} {'SSIM':10} {'Time':10}")
print("-"*90)

for r in results:
    print(f"{r[0]:20} {r[1]:15} {r[2]:10.2f} {r[3]:10.2f} {r[4]:10.3f} {r[5]:10.5f}")


# ==========================================================
# VISUALISASI CITRA
# ==========================================================

# ----------------------------------------------------------
# CITRA ASLI DAN DEGRADASI
# ----------------------------------------------------------

plt.figure(figsize=(12,5))
plt.suptitle("Citra Asli dan Degradasi Motion Blur dan Noise")

plt.subplot(1,4,1)
plt.title("Citra Asli")
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Motion Blur")
plt.imshow(blur_only,cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Gaussian + Blur")
plt.imshow(gauss_blur,cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("SP + Blur")
plt.imshow(sp_blur,cmap='gray')
plt.axis('off')

plt.show()


# ----------------------------------------------------------
# HASIL RESTORASI
# ----------------------------------------------------------

for name, img in datasets.items():

    plt.figure(figsize=(10,4))
    plt.suptitle("Hasil Restorasi - " + name)

    for i,(mname,method) in enumerate(methods.items()):

        if mname == "Lucy":
            res = method(img, psf)
        else:
            res = method(img, psf)

        plt.subplot(1,3,i+1)
        plt.title(mname)
        plt.imshow(res,cmap='gray')
        plt.axis('off')

    plt.show()


# ----------------------------------------------------------
# SPEKTRUM FREKUENSI
# ----------------------------------------------------------

def show_spectrum(img, title):

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift)+1)

    plt.imshow(magnitude,cmap='gray')
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(10,4))
plt.suptitle("Spektrum Frekuensi")

plt.subplot(1,2,1)
show_spectrum(image,"Asli")

plt.subplot(1,2,2)
show_spectrum(blur_only,"Blur")

plt.show()


# ==========================================================
# ANALISIS OTOMATIS
# ==========================================================

print("\nANALISIS METODE TERBAIK (PSNR)")
print("="*60)

for name in datasets.keys():

    subset = [r for r in results if r[0]==name]

    best = max(subset, key=lambda x: x[3])

    print(f"{name} -> {best[1]} (PSNR = {best[3]:.2f})")
