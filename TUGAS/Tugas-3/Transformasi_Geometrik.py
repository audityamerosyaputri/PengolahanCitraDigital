import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

print("=== PIPELINE TRANSFORMASI GEOMETRIK & REGISTRASI CITRA ===")

# =========================================================
# 1. LOAD CITRA
# =========================================================

img1 = cv2.imread('g.lurus.jpeg', 0)  # Reference
img2 = cv2.imread('g.miring.jpeg', 0)        # Moving

if img1 is None or img2 is None:
    print("Error: Pastikan file tersedia.")
    exit()

h, w = img1.shape

# Samakan ukuran jika berbeda
img2 = cv2.resize(img2, (w, h))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img1, cmap='gray')
plt.title("Reference")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img2, cmap='gray')
plt.title("Moving")
plt.axis('off')
plt.tight_layout()
plt.show()

# =========================================================
# 2. TRANSFORMASI BERBASIS MATRIKS 3×3 HOMOGEN
# =========================================================

def apply_transformation(image, H):
    return cv2.warpPerspective(image, H, (w, h))

def translation(tx, ty):
    return np.array([[1,0,tx],
                     [0,1,ty],
                     [0,0,1]], dtype=np.float32)

def rotation(angle):
    theta = np.deg2rad(angle)
    cx, cy = w/2, h/2
    T1 = translation(-cx, -cy)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,0,1]], dtype=np.float32)
    T2 = translation(cx, cy)
    return T2 @ R @ T1

def scaling(sx, sy):
    return np.array([[sx,0,0],
                     [0,sy,0],
                     [0,0,1]], dtype=np.float32)

def affine_example():
    pts1 = np.float32([[50,50],[w-50,50],[50,h-50]])
    pts2 = np.float32([[30,70],[w-80,40],[70,h-30]])
    A = cv2.getAffineTransform(pts1,pts2)
    H = np.vstack([A,[0,0,1]])
    return H.astype(np.float32)

def perspective_example():
    pts1 = np.float32([[50,50],[w-50,50],[w-50,h-50],[50,h-50]])
    pts2 = np.float32([[0,0],[w,0],[w-100,h],[100,h]])
    return cv2.getPerspectiveTransform(pts1,pts2)

transformations = [
    ("Original", np.eye(3,dtype=np.float32)),
    ("Translation", translation(40,30)),
    ("Rotation 30°", rotation(30)),
    ("Scaling 1.2x", scaling(1.2,1.2)),
    ("Affine", affine_example()),
    ("Perspective", perspective_example())
]

plt.figure(figsize=(15,8))

for i,(title,H) in enumerate(transformations):
    plt.subplot(2,3,i+1)
    result = apply_transformation(img1, H)
    plt.imshow(result,cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# 3. INTERPOLASI & EVALUASI
# =========================================================

def evaluate_interpolation(image):

    methods = [
        ('Nearest',cv2.INTER_NEAREST),
        ('Bilinear',cv2.INTER_LINEAR),
        ('Bicubic',cv2.INTER_CUBIC)
    ]

    plt.figure(figsize=(16,4))

    # Tampilkan gambar asli
    plt.subplot(1,4,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i,(name,flag) in enumerate(methods):

        start=time.time()

        down=cv2.resize(image,(w//2,h//2),interpolation=flag)
        up=cv2.resize(down,(w,h),interpolation=flag)

        mse=np.mean((image.astype(float)-up.astype(float))**2)
        psnr=10*np.log10(255**2/mse) if mse>0 else float('inf')
        ssim_val=ssim(image,up)

        print(f"\nMetode: {name}")
        print(f"MSE  : {mse:.4f}")
        print(f"PSNR : {psnr:.4f} dB")
        print(f"SSIM : {ssim_val:.4f}")
        print(f"Waktu: {time.time()-start:.6f} detik")

        # Tampilkan hasil interpolasi
        plt.subplot(1,4,i+2)
        plt.imshow(up, cmap='gray')
        plt.title(f"{name}")
        plt.axis('off')

    plt.suptitle("Perbandingan Metode Interpolasi", fontsize=14)
    plt.tight_layout()
    plt.show()

print("\n=== EVALUASI INTERPOLASI ===")
rot_img = apply_transformation(img1, rotation(30))
evaluate_interpolation(rot_img)

# =========================================================
# 4. REGISTRASI CITRA (ECC - HOMOGRAPHY)
# =========================================================

print("\n=== REGISTRASI CITRA (ECC HOMOGRAPHY) ===")

ref = img1.astype(np.float32)
mov = img2.astype(np.float32)

warp_matrix = np.eye(3,3,dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT, 
            5000, 1e-7)

try:
    cc, warp_matrix = cv2.findTransformECC(
        ref, mov,
        warp_matrix,
        cv2.MOTION_HOMOGRAPHY,
        criteria)

    aligned = cv2.warpPerspective(
        img2, warp_matrix,
        (w,h),
        flags=cv2.INTER_LINEAR +
              cv2.WARP_INVERSE_MAP)

    print("Registrasi berhasil.")
    print("Nilai ECC:", cc)

except:
    print("Registrasi gagal.")
    aligned = img2

# =========================================================
# 5. EVALUASI KUALITAS REGISTRASI
# =========================================================

print("\n=== EVALUASI HASIL REGISTRASI ===")

mse_reg = np.mean((img1.astype(float) - aligned.astype(float))**2)
psnr_reg = 10*np.log10(255**2/mse_reg) if mse_reg>0 else float('inf')
ssim_reg = ssim(img1, aligned)

print(f"MSE   : {mse_reg:.4f}")
print(f"PSNR  : {psnr_reg:.4f} dB")
print(f"SSIM  : {ssim_reg:.4f}")

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(img1,cmap='gray')
plt.title("Reference")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(img2,cmap='gray')
plt.title("Moving")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(aligned,cmap='gray')
plt.title("Aligned")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nMatriks Homography Hasil:")
print(np.round(warp_matrix,4))