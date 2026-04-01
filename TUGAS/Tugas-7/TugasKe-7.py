import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# LOAD IMAGE
# ==========================================
img_nat = cv2.imread('natural.png', 0)
img_pat = cv2.imread('noise.png', 0)

if img_nat is None or img_pat is None:
    raise Exception("Gambar tidak ditemukan!")

img_nat = cv2.resize(img_nat, (512, 512)).astype(np.float32)
img_pat = cv2.resize(img_pat, (512, 512)).astype(np.float32)

# ==========================================
# NORMALISASI
# ==========================================
def normalize(img):
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    return img

# ==========================================
# FFT ANALYSIS + DOMINANT FREQUENCY
# ==========================================
def fft_analysis(img, title):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)
    phase = np.angle(fshift)

    # DETEKSI FREKUENSI DOMINAN
    mag_copy = magnitude.copy()
    center = (img.shape[0]//2, img.shape[1]//2)
    mag_copy[center[0]-10:center[0]+10, center[1]-10:center[1]+10] = 0

    idx = np.unravel_index(np.argmax(mag_copy), mag_copy.shape)
    print(f"[{title}] Frekuensi dominan di: {idx}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title(title)
    plt.subplot(1,3,2); plt.imshow(magnitude, cmap='gray'); plt.title("Magnitude")
    plt.subplot(1,3,3); plt.imshow(phase, cmap='gray'); plt.title("Phase")
    plt.tight_layout()
    plt.show()

    return fshift

# ==========================================
# REKONSTRUKSI
# ==========================================
def reconstruct(fshift):
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    mag_only = magnitude * np.exp(1j * 0)
    img_mag = np.abs(np.fft.ifft2(np.fft.ifftshift(mag_only)))
    img_mag = normalize(img_mag)

    phase_only = np.exp(1j * phase)
    img_phase = np.real(np.fft.ifft2(np.fft.ifftshift(phase_only)))
    img_phase = normalize(img_phase)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img_mag, cmap='gray'); plt.title("Magnitude Only")
    plt.subplot(1,2,2); plt.imshow(img_phase, cmap='gray'); plt.title("Phase Only")
    plt.tight_layout()
    plt.show()

# ==========================================
# FILTER
# ==========================================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y-crow)**2 + (X-ccol)**2)
    return (dist <= cutoff).astype(np.float32)

def ideal_highpass(shape, cutoff):
    return 1 - ideal_lowpass(shape, cutoff)

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    d2 = (Y-crow)**2 + (X-ccol)**2
    return np.exp(-d2/(2*(cutoff**2)))

def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)

def bandpass(shape, low, high):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y-crow)**2 + (X-ccol)**2)
    return np.logical_and(dist > low, dist < high).astype(np.float32)

def apply_filter(img, mask):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return normalize(np.real(img_back)) * 255

# ==========================================
# NOTCH FILTER
# ==========================================
def notch_filter(shape, centers, radius=10):
    mask = np.ones(shape, dtype=np.float32)
    Y, X = np.ogrid[:shape[0], :shape[1]]

    for (cx, cy) in centers:
        dist = np.sqrt((Y-cx)**2 + (X-cy)**2)
        mask[dist < radius] = 0
    return mask

# ==========================================
# WAVELET
# ==========================================
def wavelet_process(img, wavelet='haar'):
    coeffs = pywt.wavedec2(img, wavelet, level=2)
    cA, (cH, cV, cD), *_ = coeffs

    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1); plt.imshow(normalize(cA), cmap='gray'); plt.title("cA")
    plt.subplot(2,2,2); plt.imshow(normalize(cH), cmap='gray'); plt.title("cH")
    plt.subplot(2,2,3); plt.imshow(normalize(cV), cmap='gray'); plt.title("cV")
    plt.subplot(2,2,4); plt.imshow(normalize(cD), cmap='gray'); plt.title("cD")
    plt.suptitle(wavelet)
    plt.tight_layout()
    plt.show()

    # Rekonstruksi hanya dari aproksimasi
    coeffs_mod = [cA] + [(np.zeros_like(h), np.zeros_like(v), np.zeros_like(d))
                         for (h,v,d) in coeffs[1:]]

    rec = pywt.waverec2(coeffs_mod, wavelet)
    rec = rec[:img.shape[0], :img.shape[1]]
    return normalize(rec)*255

# ==========================================
# MAIN PROCESS
# ==========================================
# FFT
f_nat = fft_analysis(img_nat, "Natural")
f_pat = fft_analysis(img_pat, "Pattern")

# Rekonstruksi
reconstruct(f_nat)
reconstruct(f_pat)

# VARIASI CUTOFF
cutoffs = [10, 30, 60]

for c in cutoffs:
    mask = gaussian_lowpass(img_nat.shape, c)
    result = apply_filter(img_nat, mask)

    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title(f"Gaussian LPF cutoff={c}")
    plt.show()

# HIGH PASS + BANDPASS
hp = apply_filter(img_nat, gaussian_highpass(img_nat.shape, 30))
bp = apply_filter(img_nat, bandpass(img_nat.shape, 20, 60))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(hp, cmap='gray'); plt.title("High Pass")
plt.subplot(1,2,2); plt.imshow(bp, cmap='gray'); plt.title("Band Pass")
plt.show()

# NOTCH
notch = notch_filter(img_pat.shape, [(256,200),(256,300)],10)
notch_img = apply_filter(img_pat, notch)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img_pat, cmap='gray'); plt.title("Noisy")
plt.subplot(1,2,2); plt.imshow(notch_img, cmap='gray'); plt.title("Notch")
plt.show()

# WAVELET
rec_haar = wavelet_process(img_nat, 'haar')
rec_db4  = wavelet_process(img_nat, 'db4')

# SPATIAL FILTER
spatial = cv2.GaussianBlur(img_nat, (15,15), 5)

# EVALUASI + WAKTU
print("\n=== EVALUASI ===")

start = time.time()
_ = np.fft.fft2(img_nat)
fft_time = time.time() - start

start = time.time()
_ = pywt.wavedec2(img_nat, 'haar')
wave_time = time.time() - start

start = time.time()
_ = cv2.GaussianBlur(img_nat, (15,15), 5)
spatial_time = time.time() - start

print(f"FFT Time     : {fft_time:.6f}s")
print(f"Wavelet Time : {wave_time:.6f}s")
print(f"Spatial Time : {spatial_time:.6f}s")

print(f"PSNR Notch   : {psnr(img_nat, notch_img, data_range=255):.2f}")
print(f"SSIM Notch   : {ssim(img_nat, notch_img, data_range=255):.4f}")
print(f"PSNR Wavelet : {psnr(img_nat, rec_haar, data_range=255):.2f}")
print(f"SSIM Wavelet : {ssim(img_nat, rec_haar, data_range=255):.4f}")