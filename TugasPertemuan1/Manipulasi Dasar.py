import cv2
import numpy as np

# Membaca citra
img = cv2.imread("lukisan.jpeg")

if img is None:
    print("Error: Gambar tidak ditemukan atau gagal dibaca!")

else:

    tinggi, lebar = img.shape[:2]

    # CROPPING

    crop = img[int(tinggi*0.25):int(tinggi*0.75),
               int(lebar*0.25):int(lebar*0.75)]

    before_crop = cv2.resize(img, (300,300))
    after_crop = cv2.resize(crop, (300,300))

    gabung_crop = np.hstack((before_crop, after_crop))

    # RESIZING

    resize = cv2.resize(img, (300,300))

    before_resize = cv2.resize(img, (300,300))
    after_resize = resize

    gabung_resize = np.hstack((before_resize, after_resize))

    # ROTASI

    rotasi = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    before_rotasi = cv2.resize(img, (300,300))
    after_rotasi = cv2.resize(rotasi, (300,300))

    gabung_rotasi = np.hstack((before_rotasi, after_rotasi))

    # FLIP

    flip = cv2.flip(img, 1)

    before_flip = cv2.resize(img, (300,300))
    after_flip = cv2.resize(flip, (300,300))

    gabung_flip = np.hstack((before_flip, after_flip))

    # Tampilan

    cv2.imshow("Crop (Sebelum | Sesudah)", gabung_crop)
    cv2.imshow("Resize (Sebelum | Sesudah)", gabung_resize)
    cv2.imshow("Rotasi (Sebelum | Sesudah)", gabung_rotasi)
    cv2.imshow("Flip (Sebelum | Sesudah)", gabung_flip)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
