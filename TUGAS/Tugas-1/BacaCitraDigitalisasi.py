import cv2
import numpy as np

# Membaca citra
img = cv2.imread("lukisan.jpeg")

# Cek apakah gambar berhasil dibaca
if img is None:
    print("Error: Gambar tidak ditemukan atau gagal dibaca!")
else:
    # Menampilkan dimensi citra
    print("Dimensi Citra Asli:", img.shape)

    # Konversi ke grayscale agar lebih sederhana
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Dimensi Citra Grayscale:", gray.shape)

    # Representasi citra dalam bentuk matriks
    print("\nMatriks 5x5 Piksel Pertama:")
    print(gray[:5, :5])

    # Representasi dalam bentuk vektor
    vector = gray.flatten()

    print("\n25 Nilai Pertama Vektor:")
    print(vector[:25])

    print("\nPanjang Vektor:", len(vector))

    # Menampilkan gambar asli dan grayscale
    cv2.imshow("Citra Asli", img)
    cv2.imshow("Citra Grayscale", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()