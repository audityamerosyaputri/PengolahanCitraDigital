import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path, compare_image_path=None):
    """Analyze your own image"""
    
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Gambar tidak ditemukan. Periksa path file.")
    
    print("\n=== ANALISIS CITRA SENDIRI ===")
    
    # 1️⃣ Dimensi dan Resolusi
    height, width, channels = img.shape
    resolution = width * height
    
    print(f"Dimensi: {width} x {height}")
    print(f"Channels: {channels}")
    print(f"Resolusi: {resolution:,} pixels")
    
    # 2️⃣ Aspect Ratio
    aspect_ratio = width / height
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    # 3️⃣ Konversi ke Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("\nPerbandingan Ukuran Memori:")
    color_memory = img.size * img.dtype.itemsize
    gray_memory = gray.size * gray.dtype.itemsize
    
    print(f"Color Image: {color_memory:,} bytes")
    print(f"Grayscale:   {gray_memory:,} bytes")
    
    # 4️⃣ Statistik
    print("\nStatistik Grayscale:")
    print(f"Mean: {gray.mean():.2f}")
    print(f"Std Dev: {gray.std():.2f}")
    print(f"Min: {gray.min()}")
    print(f"Max: {gray.max()}")
    
    # 5️⃣ Histogram
    plt.figure(figsize=(12,5))
    
    # Histogram grayscale
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(), 256, [0,256], color='gray')
    plt.title("Histogram Grayscale")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    # Histogram warna
    plt.subplot(1,2,2)
    colors = ('b','g','r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color=col)
    plt.title("Histogram RGB")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # 6️⃣ Bandingkan dengan citra lain (optional)
    if compare_image_path:
        img2 = cv2.imread(compare_image_path)
        if img2 is not None:
            print("\n=== PERBANDINGAN DENGAN CITRA LAIN ===")
            h2, w2, _ = img2.shape
            print(f"Citra 1 Resolusi: {resolution:,}")
            print(f"Citra 2 Resolusi: {w2*h2:,}")
            print(f"Perbedaan Resolusi: {abs(resolution - (w2*h2)):,}")
    
    # Simpan hasil dalam dictionary
    analysis_results = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "mean": gray.mean(),
        "std": gray.std(),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "color_memory": color_memory,
        "gray_memory": gray_memory
    }
    
    return analysis_results
results = analyze_my_image("Yaya23.jpg")
print(results)
