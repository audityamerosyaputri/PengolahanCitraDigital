import numpy as np
import cv2
import matplotlib.pyplot as plt

def praktikum_9_2():
    print("\nPRAKTIKUM 9.2: EDGE DETECTION DAN REGION-BASED SEGMENTATION")
    print("=" * 70)

    # ===============================
    # CREATE TEST IMAGE
    # ===============================
    def create_edge_test_image():
        img = np.zeros((300, 400), dtype=np.uint8)

        # Step edge
        cv2.rectangle(img, (50, 50), (150, 150), 100, -1)
        cv2.rectangle(img, (151, 50), (250, 150), 200, -1)

        # Ramp edge
        for i in range(50, 150):
            img[160:240, i] = 50 + (i - 50) * 2

        # Triangle
        triangle_cnt = np.array([(300, 160), (350, 240), (250, 240)])
        cv2.drawContours(img, [triangle_cnt], 0, 150, -1)

        # Line
        cv2.line(img, (50, 260), (350, 260), 200, 3)

        # Noise (lebih kecil biar hasil bagus)
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

        return img

    # ===============================
    # EDGE DETECTION
    # ===============================
    def sobel(image):
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        mag = np.sqrt(sx**2 + sy**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        direction = np.arctan2(sy, sx) * 180 / np.pi
        return mag.astype(np.uint8), direction

    def prewitt(image):
        kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        px = cv2.filter2D(image.astype(np.float64), -1, kx)
        py = cv2.filter2D(image.astype(np.float64), -1, ky)
        mag = np.sqrt(px**2 + py**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag.astype(np.uint8)

    def canny(image):
        blur = cv2.GaussianBlur(image, (5,5), 1.4)
        return cv2.Canny(blur, 50, 150)

    def laplacian(image):
        lap = cv2.Laplacian(image, cv2.CV_64F)
        lap = np.abs(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        return lap.astype(np.uint8)

    # ===============================
    # REGION GROWING (STABIL)
    # ===============================
    def region_growing(image, seeds, threshold=20):
        result = np.zeros_like(image)
        visited = np.zeros_like(image, dtype=bool)

        for seed in seeds:
            stack = [seed]
            seed_val = image[seed]

            while stack:
                x, y = stack.pop()

                if visited[x, y]:
                    continue

                visited[x, y] = True
                result[x, y] = 255

                for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        if not visited[nx, ny]:
                            if abs(int(image[nx, ny]) - int(seed_val)) < threshold:
                                stack.append((nx, ny))
        return result

    # ===============================
    # SPLIT & MERGE (SIMPLE)
    # ===============================
    def split_merge(image):
        result = np.zeros_like(image)
        block = 50
        val = 50

        for i in range(0, image.shape[0], block):
            for j in range(0, image.shape[1], block):
                region = image[i:i+block, j:j+block]
                if np.std(region) < 20:
                    result[i:i+block, j:j+block] = val
                    val = min(val+40, 255)
        return result

    # ===============================
    # MAIN PROCESS
    # ===============================
    img = create_edge_test_image()

    sobel_mag, sobel_dir = sobel(img)
    prewitt_img = prewitt(img)
    canny_img = canny(img)
    lap_img = laplacian(img)

    seeds = [(75,100), (200,100), (250,200)]
    region_img = region_growing(img, seeds)
    split_img = split_merge(img)

    # ===============================
    # VISUALISASI EDGE
    # ===============================
    fig, ax = plt.subplots(2,3, figsize=(15,10))

    ax[0,0].imshow(img, cmap='gray'); ax[0,0].set_title("Original")
    ax[0,1].imshow(sobel_mag, cmap='gray'); ax[0,1].set_title("Sobel")
    ax[0,2].imshow(prewitt_img, cmap='gray'); ax[0,2].set_title("Prewitt")

    ax[1,0].imshow(canny_img, cmap='gray'); ax[1,0].set_title("Canny")
    ax[1,1].imshow(lap_img, cmap='gray'); ax[1,1].set_title("Laplacian")
    ax[1,2].imshow(region_img, cmap='gray'); ax[1,2].set_title("Region Growing")

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

    # ===============================
    # VISUALISASI 
    # ===============================
    fig, ax = plt.subplots(2, 3, figsize=(15,10))

    # Row 1
    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title("Original")

    ax[0,1].imshow(region_img, cmap='gray')
    ax[0,1].set_title("Region Growing")

    seed_display = img.copy()
    for (x,y) in seeds:
        cv2.circle(seed_display, (y,x), 5, 255, -1)
    ax[0,2].imshow(seed_display, cmap='gray')
    ax[0,2].set_title("Seed Points")

    # Row 2
    boundary = cv2.Canny(region_img, 50, 150)
    ax[1,0].imshow(boundary, cmap='gray')
    ax[1,0].set_title("Region Boundary")

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[region_img > 0] = [0,255,0]
    ax[1,1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1,1].set_title("Overlay")

    difference = cv2.absdiff(img, region_img)
    ax[1,2].imshow(difference, cmap='gray')
    ax[1,2].set_title("Difference")

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nKESIMPULAN:")
    print("- Sobel & Prewitt: deteksi gradien")
    print("- Canny: paling bersih & tipis")
    print("- Laplacian: sensitif noise")
    print("- Region Growing: tergantung seed")
    print("- Split: segmentasi berbasis blok")

    return img

# RUN
praktikum_9_2()