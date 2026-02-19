import cv2
import numpy as np

# Membaca citra
img = cv2.imread("lukisan.jpeg")

if img is None:
    print("Error: Gambar tidak ditemukan atau gagal dibaca!")
else:
    print("Analisis Parameter Citra \n")

    # 1. Resolusi Spasial
   
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    print("Resolusi Spasial (Lebar x Tinggi):", width, "x", height, "piksel")
    total_pixels = width * height
    print("Total Jumlah Piksel:", total_pixels, "piksel\n")

    # 2. Bit Depth dan Intensitas
    bit_depth_per_channel = img.dtype.itemsize * 8   # biasanya 8 bit
    total_bit_depth = bit_depth_per_channel * channels

    print("Bit Depth per Channel:", bit_depth_per_channel, "bit")
    print("Total Bit Depth (dengan channel):", total_bit_depth, "bit")

    intensitas = 2 ** bit_depth_per_channel
    print("Jumlah Intensitas per Channel:", intensitas)
    print("Jumlah Kombinasi Warna Maksimum:", intensitas ** channels, "\n")

    # 3. Aspect Ratio
    aspect_ratio = width / height
    print("Aspect Ratio:", round(aspect_ratio, 2), "\n")

    # 4. Ukuran Memori
    memory_bytes = width * height * channels * (bit_depth_per_channel / 8)
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024

    print("Ukuran Memori:")
    print(" - Byte :", int(memory_bytes), "bytes")
    print(" - KB   :", round(memory_kb, 2), "KB")
    print(" - MB   :", round(memory_mb, 2), "MB\n")

    # 5. Simulasi Perubahan
    print("Simulasi Perubahan\n")

    new_width = width * 2
    new_height = height * 2
    new_bit_depth = bit_depth_per_channel / 2

    new_memory_bytes = new_width * new_height * channels * (new_bit_depth / 8)
    new_memory_mb = new_memory_bytes / (1024 * 1024)

    print("Resolusi Baru:", new_width, "x", new_height)
    print("Bit Depth Baru per Channel:", new_bit_depth, "bit")
    print("Perkiraan Ukuran Memori Baru:", round(new_memory_mb, 2), "MB") 