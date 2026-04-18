"""
=================================================================
  PIPELINE MORFOLOGI: OCR PREPROCESSING + OBJECT COUNTING
=================================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import time
import os

# =================================================================
#  KONFIGURASI — SESUAIKAN NAMA FILE CITRA DI SINI
# =================================================================

NAMA_CITRA_A = "citraA.png"   # citra teks dengan noise (untuk OCR)
NAMA_CITRA_B = "citraB.png"   # citra objek overlapping (untuk counting)

# -----------------------------------------------------------------
# LOAD / BUAT CITRA
# -----------------------------------------------------------------

def load_or_synthetic_A(nama_file):
    """Muat Citra A dari file, atau buat sintetis jika tidak ada."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nama_file)
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"  [LOAD] Citra A: {nama_file} | {img.shape[1]}x{img.shape[0]} px")
            return img
        print(f"  [ERROR] Gagal membaca '{nama_file}', pakai citra sintetis.")
    else:
        print(f"  [WARN] '{nama_file}' tidak ditemukan, pakai citra sintetis.")

    # Citra sintetis: teks + noise
    img = np.ones((300, 500), dtype=np.uint8) * 255
    texts = [
        ("Pipeline Morfologi",    (40,  50), 0.8),
        ("OCR Preprocessing",    (40, 100), 0.7),
        ("Noise titik & goresan", (40, 150), 0.6),
        ("OpenCV Python 3.x",    (40, 200), 0.7),
        ("Erosi Dilasi Opening",  (40, 250), 0.65),
    ]
    for text, pos, scale in texts:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2)
    rng = np.random.default_rng(42)
    for _ in range(300):  # salt noise
        cv2.circle(img, (int(rng.integers(0,500)), int(rng.integers(0,300))),
                   int(rng.integers(1,3)), 255, -1)
    for _ in range(400):  # pepper noise
        cv2.circle(img, (int(rng.integers(0,500)), int(rng.integers(0,300))),
                   int(rng.integers(1,2)), 0, -1)
    for _ in range(8):    # goresan
        x1, y1 = int(rng.integers(0,500)), int(rng.integers(0,300))
        cv2.line(img, (x1,y1),
                 (x1+int(rng.integers(-60,60)), y1+int(rng.integers(-20,20))), 0, 1)
    print("  [INFO] Menggunakan Citra A sintetis.")
    return img


def load_or_synthetic_B(nama_file):
    """Muat Citra B dari file, atau buat sintetis jika tidak ada."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nama_file)
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"  [LOAD] Citra B: {nama_file} | {img.shape[1]}x{img.shape[0]} px")
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return img_bin, None   # n_manual=None karena tidak diketahui
        print(f"  [ERROR] Gagal membaca '{nama_file}', pakai citra sintetis.")
    else:
        print(f"  [WARN] '{nama_file}' tidak ditemukan, pakai citra sintetis.")

    # Citra sintetis: 15 lingkaran overlapping
    img = np.zeros((400, 500), dtype=np.uint8)
    circles = [
        (100,100,45),(180,130,40),(260,90,50),(350,120,42),(130,220,48),
        (220,240,44),(310,210,46),(80,320,40),(180,330,52),(280,310,45),
        (380,280,43),(420,160,38),(450,340,40),(60,180,35),(340,340,47),
    ]
    for (x, y, r) in circles:
        cv2.circle(img, (x,y), r, 255, -1)
    rng = np.random.default_rng(99)
    noise = rng.integers(0, 25, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print("  [INFO] Menggunakan Citra B sintetis (15 lingkaran).")
    return img, 15


# -----------------------------------------------------------------
# STRUCTURING ELEMENTS
# -----------------------------------------------------------------

def visualize_se():
    """Tampilkan variasi structuring element."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 7))
    fig.suptitle('Variasi Structuring Element (Ukuran x Bentuk)',
                 fontsize=13, fontweight='bold')
    shapes = [('Square', cv2.MORPH_RECT),
              ('Cross',  cv2.MORPH_CROSS),
              ('Ellipse',cv2.MORPH_ELLIPSE)]
    sizes  = [3, 5, 7]
    for i, sz in enumerate(sizes):
        for j, (sh_name, sh_cv) in enumerate(shapes):
            ax  = axes[i][j]
            se  = cv2.getStructuringElement(sh_cv, (sz, sz))
            vis = np.kron(se, np.ones((18, 18))) * 200
            ax.imshow(vis, cmap='Blues', vmin=0, vmax=255)
            ax.set_title(f'{sh_name} {sz}x{sz}', fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            for r in range(sz):
                for c in range(sz):
                    sym = '■' if se[r,c] else '□'
                    ax.text(c*18+9, r*18+9, sym, ha='center', va='center',
                            fontsize=8, color='white' if se[r,c] else '#aaa')
            ax.set_xlim(0, sz*18); ax.set_ylim(sz*18, 0)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# EROSI & DILASI
# -----------------------------------------------------------------

def visualize_erode_dilate(img_A, img_B):
    """Tampilkan erosi (Citra A) dan dilasi (Citra B) dengan variasi iterasi."""
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    iterations = [1, 2, 3, 5]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    fig.suptitle('Erosi & Dilasi - Variasi Iterasi (SE Ellipse 3x3)',
                 fontsize=12, fontweight='bold')

    axes[0][0].imshow(img_A, cmap='gray', vmin=0, vmax=255)
    axes[0][0].set_title('Original\n(Citra A)', fontsize=8)
    axes[0][0].axis('off')
    axes[1][0].imshow(img_B, cmap='gray', vmin=0, vmax=255)
    axes[1][0].set_title('Original\n(Citra B)', fontsize=8)
    axes[1][0].axis('off')

    for col, itr in enumerate(iterations, start=1):
        eroded  = cv2.erode(img_A, kernel, iterations=itr)
        dilated = cv2.dilate(img_B, kernel, iterations=itr)
        axes[0][col].imshow(eroded,  cmap='gray', vmin=0, vmax=255)
        axes[0][col].set_title(f'Erosi {itr}x', fontsize=8)
        axes[0][col].axis('off')
        axes[1][col].imshow(dilated, cmap='gray', vmin=0, vmax=255)
        axes[1][col].set_title(f'Dilasi {itr}x', fontsize=8)
        axes[1][col].axis('off')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# OPERASI MAJEMUK
# -----------------------------------------------------------------

def visualize_compound_ops(img):
    """Tampilkan hasil semua operasi majemuk pada Citra A."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    results = {
        "Opening":  cv2.morphologyEx(img, cv2.MORPH_OPEN,     kernel),
        "Closing":  cv2.morphologyEx(img, cv2.MORPH_CLOSE,    kernel),
        "Gradient": cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
        "Top Hat":  cv2.morphologyEx(img, cv2.MORPH_TOPHAT,   kernel),
        "Black Hat":cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel),
    }
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    fig.suptitle('Operasi Majemuk (SE Ellipse 5x5)', fontsize=12, fontweight='bold')
    for ax, (title, im) in zip(axes, [("Original", img)] + list(results.items())):
        ax.imshow(im, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# APLIKASI 1: OCR PREPROCESSING PIPELINE
# -----------------------------------------------------------------

def count_noise(img, max_area=10):
    """Hitung objek kecil (noise) pada citra biner invers."""
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < max_area)


def ocr_pipeline(img_A):
    """Pipeline 3-tahap OCR preprocessing + tampilkan hasil."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    step1 = cv2.morphologyEx(img_A, cv2.MORPH_OPEN,  k)           # hapus noise
    step2 = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, k)           # sambung gap
    _, step3 = cv2.threshold(step2, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU) # binarisasi

    n_before = count_noise(img_A)
    n_after  = count_noise(step3)
    pct      = (1 - n_after / max(n_before, 1)) * 100

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('OCR Preprocessing Pipeline (Citra A)', fontsize=12, fontweight='bold')
    for ax, im, t in zip(axes,
        [img_A, step1, step2, step3],
        [f'Original\nNoise: {n_before} obj',
         'Step 1: Opening\n(Hapus noise titik)',
         'Step 2: Closing\n(Sambung gap)',
         f'Step 3: Otsu\nNoise sisa: {n_after} obj']):
        ax.imshow(im, cmap='gray')
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\n  === Evaluasi OCR Preprocessing ===")
    print(f"  Noise sebelum : {n_before} objek")
    print(f"  Noise sesudah : {n_after} objek")
    print(f"  Noise terhapus: {pct:.1f}%")
    print(f"  Est. peningkatan character rate: +{min(pct*0.25, 25):.0f}%")


# -----------------------------------------------------------------
# APLIKASI 2: OBJECT COUNTING — WATERSHED
# -----------------------------------------------------------------

def watershed_counting(img_B, n_manual=None):
    """Hitung objek dengan Distance Transform + Watershed."""
    kernel  = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(img_B, kernel, iterations=3)
    dist    = cv2.distanceTransform(img_B, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(img_B, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)
    result  = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]

    n_auto = n_markers - 1

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Object Counting via Watershed - {n_auto} objek terdeteksi',
                 fontsize=11, fontweight='bold')
    for ax, im, t, cmap in zip(axes,
        [img_B, dist, markers, result],
        ['Original (Citra B)', 'Distance Transform',
         f'Watershed Markers\n({n_auto} seed)',
         f'Hasil Segmentasi\nAuto: {n_auto}'],
        ['gray', 'jet', 'nipy_spectral', None]):
        ax.imshow(im if cmap else cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                  cmap=cmap)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\n  === Evaluasi Counting ===")
    if n_manual is not None:
        acc = (1 - abs(n_auto - n_manual) / n_manual) * 100
        print(f"  Ground truth : {n_manual} objek")
        print(f"  Hasil auto   : {n_auto} objek")
        print(f"  Akurasi      : {acc:.1f}%")
    else:
        print(f"  Hasil auto   : {n_auto} objek")


# -----------------------------------------------------------------
# BENCHMARK WAKTU
# -----------------------------------------------------------------

def benchmark(img, n_runs=200):
    """Ukur waktu rata-rata setiap operasi morfologi."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    ops = {
        "Erosi":    lambda: cv2.erode(img, kernel),
        "Dilasi":   lambda: cv2.dilate(img, kernel),
        "Opening":  lambda: cv2.morphologyEx(img, cv2.MORPH_OPEN,     kernel),
        "Closing":  lambda: cv2.morphologyEx(img, cv2.MORPH_CLOSE,    kernel),
        "Gradient": lambda: cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
        "Top Hat":  lambda: cv2.morphologyEx(img, cv2.MORPH_TOPHAT,   kernel),
        "Black Hat":lambda: cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel),
    }
    print(f"\n  === Benchmark Waktu Komputasi ({n_runs} iterasi/operasi) ===")
    print(f"  {'Operasi':<12} | {'Rata-rata':>10} | {'Std Dev':>10} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*60}")
    for name, fn in ops.items():
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"  {name:<12} | {arr.mean():>8.4f} ms | {arr.std():>8.4f} ms |"
              f" {arr.min():>6.4f} ms | {arr.max():>6.4f} ms")


# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 65)
    print("  PIPELINE MORFOLOGI: OCR Preprocessing + Object Counting")
    print("=" * 65)

    # Load citra sesuai nama yang dikonfigurasi di atas
    print(f"\n[INFO] Mencari citra A: '{NAMA_CITRA_A}'")
    img_A = load_or_synthetic_A(NAMA_CITRA_A)

    print(f"\n[INFO] Mencari citra B: '{NAMA_CITRA_B}'")
    img_B, n_manual = load_or_synthetic_B(NAMA_CITRA_B)

    # 1. Structuring Elements
    print("\n[1/5] Visualisasi Structuring Elements...")
    visualize_se()

    # 2. Erosi & Dilasi
    print("[2/5] Erosi & Dilasi...")
    visualize_erode_dilate(img_A, img_B)

    # 3. Operasi Majemuk
    print("[3/5] Operasi Majemuk...")
    visualize_compound_ops(img_A)

    # 4. OCR Pipeline
    print("[4/5] OCR Preprocessing Pipeline...")
    ocr_pipeline(img_A)

    # 5. Object Counting
    print("[5/5] Object Counting...")
    watershed_counting(img_B, n_manual)

    # Benchmark
    benchmark(img_A)

    print("\n" + "=" * 65)
    print("  SELESAI.")
    print("=" * 65)


if __name__ == "__main__":
    main()