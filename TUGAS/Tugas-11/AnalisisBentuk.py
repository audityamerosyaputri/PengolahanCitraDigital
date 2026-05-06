import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# LOAD DATASET
# =========================
def load_dataset(path):
    images, labels = [], []
    for label in sorted(os.listdir(path)):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, file), 0)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# =========================
# VISUALISASI DATASET
# =========================
def show_dataset(images, labels):
    cols = 6
    rows = int(np.ceil(len(images)/cols))
    plt.figure(figsize=(15,3*rows))
    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.suptitle("Dataset")
    plt.tight_layout()
    plt.show()

# =========================
# CONTOUR
# =========================
def get_contour(img):
    _,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return None, th
    return max(contours, key=cv2.contourArea), th

def show_contour(images):
    cols = 6
    rows = int(np.ceil(len(images)/cols))
    plt.figure(figsize=(15,3*rows))
    for i,img in enumerate(images):
        c,_ = get_contour(img)
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if c is not None:
            cv2.drawContours(canvas,[c],-1,(0,255,0),2)
        plt.subplot(rows, cols, i+1)
        plt.imshow(canvas[:,:,::-1])
        plt.axis('off')
    plt.suptitle("Contour")
    plt.tight_layout()
    plt.show()

# =========================
# REGION FEATURES
# =========================
def region_features(c):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x,y,w,h = cv2.boundingRect(c)
    aspect = w/h if h!=0 else 0
    extent = area/(w*h) if w*h!=0 else 0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area!=0 else 0

    M = cv2.moments(c)
    cx = M['m10']/M['m00'] if M['m00']!=0 else 0
    cy = M['m01']/M['m00'] if M['m00']!=0 else 0

    return [area, perimeter, cx, cy, x, y, w, h, aspect, extent, solidity]

# =========================
# MOMENTS
# =========================
def moment_features(th):
    M = cv2.moments(th)
    hu = cv2.HuMoments(M).flatten()
    return [M['m00'],M['m10'],M['m01'],M['mu20'],M['mu02'],M['mu11']] + list(hu)

# =========================
# CHAIN CODE
# =========================
def chain_code(c, mode=8):
    dirs4 = [(1,0),(0,1),(-1,0),(0,-1)]
    dirs8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    dirs = dirs4 if mode==4 else dirs8

    pts = c.reshape(-1,2)
    code = []

    for i in range(len(pts)):
        dx = np.sign(pts[(i+1)%len(pts)][0]-pts[i][0])
        dy = np.sign(pts[(i+1)%len(pts)][1]-pts[i][1])
        for idx,(x,y) in enumerate(dirs):
            if (dx,dy)==(x,y):
                code.append(idx)
                break
    return code

def normalize_chain(code):
    if len(code)==0:
        return [0]
    min_idx = np.argmin(code)
    return code[min_idx:] + code[:min_idx]

# =========================
# POLYGON (Douglas-Peucker)
# =========================
def polygon(c):
    approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    return len(approx)

# =========================
# FOURIER DESCRIPTOR
# =========================
def fourier_descriptor(c,n=20):
    pts = c.reshape(-1,2)
    z = pts[:,0] + 1j*pts[:,1]
    fft = np.fft.fft(z)

    mag = np.abs(fft)
    mag = mag/mag[0] if mag[0]!=0 else mag  # normalisasi

    return fft, mag[:n]

def reconstruct(fft,n):
    f = fft.copy()
    f[n:-n] = 0
    rec = np.fft.ifft(f)
    return rec.real, rec.imag

# =========================
# EKSTRAKSI SEMUA FITUR
# =========================
def extract(images):
    feats, table = [], []

    for img in images:
        c,th = get_contour(img)
        if c is None:
            continue

        reg = region_features(c)
        mom = moment_features(th)

        cc4 = len(normalize_chain(chain_code(c,4)))
        cc8 = len(normalize_chain(chain_code(c,8)))
        poly = polygon(c)

        ffft,fd = fourier_descriptor(c)

        features = reg + mom + [cc4,cc8,poly] + list(fd)
        feats.append(features)

        table.append(reg + mom)

    return np.array(feats), pd.DataFrame(table)

# =========================
# KLASIFIKASI (DATASET KECIL)
# =========================
def evaluate(X,y):
    model = KNeighborsClassifier(n_neighbors=1)  # cocok untuk data kecil
    model.fit(X,y)
    pred = model.predict(X)

    acc = accuracy_score(y,pred)
    cm = confusion_matrix(y,pred)

    return acc, cm

# =========================
# MAIN
# =========================
def main():
    path="dataset"
    images, labels = load_dataset(path)

    # VISUALISASI
    show_dataset(images,labels)
    show_contour(images)

    # EKSTRAKSI
    X,table = extract(images)
    y = np.array(labels[:len(X)])

    print("\n=== TABEL REGION & MOMENTS ===")
    print(table)

    # FITUR TERPILIH
    region = X[:,:11]
    moment = X[:,11:24]
    fourier = X[:,-20:]

    acc_all,cm = evaluate(X,y)
    acc_region,_ = evaluate(region,y)
    acc_moment,_ = evaluate(moment,y)
    acc_fourier,_ = evaluate(fourier,y)

    print("\n=== AKURASI ===")
    print("All:",acc_all*100)
    print("Region:",acc_region*100)
    print("Moment:",acc_moment*100)
    print("Fourier:",acc_fourier*100)

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    # FOURIER REKONSTRUKSI
    c,_ = get_contour(images[0])
    fft,_ = fourier_descriptor(c)

    plt.figure(figsize=(10,4))
    for i,n in enumerate([5,10,20]):
        x,y = reconstruct(fft,n)
        plt.subplot(1,3,i+1)
        plt.plot(x,y)
        plt.title(f"{n} coeff")
    plt.suptitle("Rekonstruksi Fourier")
    plt.show()

# RUN
main()