import cv2
import numpy as np
import matplotlib.pyplot as plt

def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for images
    
    Parameters:
    medical_image : input image
    modality : 'X-ray', 'MRI', 'CT', 'Ultrasound'
    
    Returns:
    enhanced_image : hasil citra setelah enhancement
    report : laporan enhancement
    """

    # pastikan gambar grayscale
    if len(medical_image.shape) == 3:
        medical_image = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)

    report = {}
    enhanced_image = medical_image.copy()

    # Enhancement untuk X-ray
    if modality == 'X-ray':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(medical_image)

        report['Technique'] = "CLAHE"
        report['Purpose'] = "Improve contrast"

    # Enhancement untuk MRI
    elif modality == 'MRI':
        denoise = cv2.GaussianBlur(medical_image,(5,5),0)

        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])

        enhanced_image = cv2.filter2D(denoise,-1,kernel)

        report['Technique'] = "Gaussian Blur + Sharpening"
        report['Purpose'] = "Enhance tissue boundaries"

    # Enhancement untuk CT
    elif modality == 'CT':

        min_val = np.min(medical_image)
        max_val = np.max(medical_image)

        if max_val - min_val == 0:
            enhanced_image = medical_image
        else:
            enhanced_image = ((medical_image - min_val) /
                              (max_val - min_val) * 255).astype(np.uint8)

        report['Technique'] = "Contrast Stretching"
        report['Purpose'] = "Improve visibility"

    # Enhancement untuk Ultrasound
    elif modality == 'Ultrasound':

        denoise = cv2.medianBlur(medical_image,5)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(denoise)

        report['Technique'] = "Median Filter + CLAHE"
        report['Purpose'] = "Reduce noise"

    else:
        report['Technique'] = "None"
        report['Purpose'] = "Unknown modality"

    # Metrics sederhana
    report['Original Mean'] = float(np.mean(medical_image))
    report['Enhanced Mean'] = float(np.mean(enhanced_image))
    report['Original Std'] = float(np.std(medical_image))
    report['Enhanced Std'] = float(np.std(enhanced_image))

    return enhanced_image, report


# =========================
# MAIN PROGRAM
# =========================

image_path = "pemandangan.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: file memanah.jpg tidak ditemukan.")
    print("Pastikan gambar berada di folder yang sama dengan file python.")
else:

    enhanced_img, report = medical_image_enhancement(img, modality='X-ray')

    # tampilkan gambar
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced_img, cmap='gray')
    plt.axis("off")

    plt.show()

    print("\nEnhancement Report:")
    for key, value in report.items():
        print(f"{key} : {value}")