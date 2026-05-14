import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ==========================================
# 0. KONFIGURASI & UTILITY
# ==========================================
DATASET_PATH = 'dataset/'
OUTPUT_PATH = 'output/'
SUB_FOLDERS = ['keypoints', 'matching', 'bovw', 'pca']

for sub in SUB_FOLDERS:
    os.makedirs(os.path.join(OUTPUT_PATH, sub), exist_ok=True)

def load_dataset():
    data = {}
    categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    for cat in categories:
        data[cat] = {}
        for img_name in os.listdir(os.path.join(DATASET_PATH, cat)):
            img_path = os.path.join(DATASET_PATH, cat, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                data[cat][img_name.split('.')[0]] = img
    return data

# ==========================================
# 1. FEATURE DETECTION & DESCRIPTION
# ==========================================
def get_detector(name):
    if name == 'SIFT':
        return cv2.SIFT_create()
    elif name == 'ORB':
        return cv2.ORB_create(nfeatures=2000)
    elif name == 'SURF':
        try:
            return cv2.xfeatures2d.SURF_create()
        except:
            print("[WARNING] SURF tidak tersedia di versi OpenCV ini.")
            return None
    return None

def extract_features(img, method='SIFT'):
    detector = get_detector(method)
    if detector is None: return None, None, 0
    
    start_time = time.time()
    kp, des = detector.detectAndCompute(img, None)
    elapsed = time.time() - start_time
    
    return kp, des, elapsed

# ==========================================
# 2. FEATURE MATCHING (RANSAC & Homography)
# ==========================================
def match_features(des1, des2, method='SIFT', matcher_type='BF'):
    if des1 is None or des2 is None: return [], 0
    
    if method == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        if matcher_type == 'FLANN':
            index_params = dict(algorithm=1, trees=5) # FLANN SIFT/SURF
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = matcher.knnMatch(des1, des2, k=2)
    
    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    return good_matches, len(matches)

def estimate_homography(kp1, kp2, matches):
    if len(matches) < 4: return None, 0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = np.sum(mask) if mask is not None else 0
    return mask, inliers

# ==========================================
# 3. BAG OF VISUAL WORDS (BoVW)
# ==========================================
class BoVW:
    def __init__(self, k=50):
        self.k = k
        self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        self.vocab = None

    def build_vocabulary(self, descriptors_list):
        print(f"[INFO] Building BoVW vocabulary with k={self.k}...")
        all_des = np.vstack(descriptors_list)
        self.kmeans.fit(all_des)
        self.vocab = self.kmeans.cluster_centers_

    def get_histogram(self, descriptors):
        if descriptors is None: return np.zeros(self.k)
        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=range(self.k + 1), density=True)
        return hist

# ==========================================
# 4. PCA ANALYSIS
# ==========================================
def apply_pca(descriptors, n_components):

    if descriptors is None:
        return None

    max_comp = min(
        descriptors.shape[0],
        descriptors.shape[1]
    )

    if n_components > max_comp:
        return descriptors

    pca = PCA(
        n_components=n_components
    )

    return pca.fit_transform(descriptors)

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
def run_pipeline():
    dataset = load_dataset()
    results_list = []
    methods = ['SIFT', 'ORB', 'SURF']
    
    # --- Part 1 & 2: Detection & Matching ---
    for method in methods:
        print(f"[INFO] Processing {method}...")
        for cat, imgs in dataset.items():
            if 'ref' not in imgs: continue
            ref_img = imgs['ref']
            kp_ref, des_ref, t_ref = extract_features(ref_img, method)
            
            if kp_ref is None: continue

            # Simpan Visualisasi Keypoints
            kp_viz = cv2.drawKeypoints(ref_img, kp_ref, None, color=(0, 255, 0))
            cv2.imwrite(f"{OUTPUT_PATH}keypoints/{cat}_{method}.jpg", kp_viz)

            for task in ['rotasi', 'skala', 'iluminasi', 'oklusi']:
                if task not in imgs: continue
                target_img = imgs[task]
                kp_t, des_t, t_t = extract_features(target_img, method)
                
                # Matching
                matches, total_raw = match_features(des_ref, des_t, method)
                mask, inliers = estimate_homography(kp_ref, kp_t, matches)
                
                # Visualisasi Matching
                match_viz = cv2.drawMatches(ref_img, kp_ref, target_img, kp_t, matches[:50], None, flags=2)
                cv2.imwrite(f"{OUTPUT_PATH}matching/{cat}_{method}_{task}.jpg", match_viz)
                
                results_list.append({
                    'Method': method, 'Category': cat, 'Task': task,
                    'Keypoints': len(kp_ref), 'Time': t_ref,
                    'Raw_Matches': total_raw, 'Good_Matches': len(matches), 'Inliers': inliers
                })

    # --- Part 3: BoVW Classification ---
    print("[INFO] Starting BoVW Pipeline...")
    all_descriptors = []
    labels = []
    cat_map = {name: i for i, name in enumerate(dataset.keys())}
    
    for cat, imgs in dataset.items():
        for img in imgs.values():
            _, des, _ = extract_features(img, 'SIFT')
            if des is not None:
                all_descriptors.append(des)
                labels.append(cat_map[cat])

    k_values = [10, 20, 50, 100]
    for kv in k_values:
        bovw_model = BoVW(k=kv)
        bovw_model.build_vocabulary(all_descriptors)
        X = np.array([bovw_model.get_histogram(d) for d in all_descriptors])
        y = np.array(labels)
        
        clf = SVC(kernel='linear')
        clf.fit(X, y)
        y_pred = clf.predict(X)
        
        acc = accuracy_score(y, y_pred)
        print(f"[INFO] BoVW Accuracy (k={kv}): {acc:.2f}")
        
        # Simpan Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, xticklabels=dataset.keys(), yticklabels=dataset.keys())
        plt.title(f"BoVW Confusion Matrix k={kv}")
        plt.savefig(f"{OUTPUT_PATH}bovw/cm_k{kv}.png")
        plt.close()

    # --- Part 4: PCA Evaluation ---
    print("[INFO] Running PCA Analysis...")
    pca_results = []
    for comp in [16, 32, 64]:
        for method in ['SIFT', 'ORB']:
            # Ambil descriptor sample
            _, des, _ = extract_features(dataset['botol']['ref'], method)
            if des is None: continue
            
            start_t = time.time()
            des_pca = apply_pca(des, comp)
            comp_t = time.time() - start_t
            
            pca_results.append({'Method': method, 'Components': comp, 'Time': comp_t})

    # Simpan Grafik PCA
    pca_df = pd.DataFrame(pca_results)
    plt.figure()
    sns.lineplot(data=pca_df, x='Components', y='Time', hue='Method')
    plt.title("PCA Components vs Computation Time")
    plt.savefig(f"{OUTPUT_PATH}pca/pca_analysis.png")
    plt.close()

    # --- Final Report ---
    df_eval = pd.DataFrame(results_list)
    df_eval.to_csv(f"{OUTPUT_PATH}evaluation_results.csv", index=False)
    print("\n[INFO] Pipeline Selesai. Hasil disimpan di folder output/.")
    print(df_eval.groupby('Method')[['Keypoints', 'Time', 'Good_Matches', 'Inliers']].mean())

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")