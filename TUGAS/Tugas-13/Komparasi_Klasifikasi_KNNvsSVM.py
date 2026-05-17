"""
============================================================
PRAKTIKUM: KOMPARASI KLASIFIKASI KNN vs SVM
============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORT LIBRARY
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, time, os
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     GridSearchCV, learning_curve,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             roc_curve, auc, classification_report)
from sklearn.pipeline import Pipeline
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

# ─────────────────────────────────────────────
# 1. LABEL KELAS FASHION-MNIST
# ─────────────────────────────────────────────
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ─────────────────────────────────────────────
# 2. LOAD DATASET FASHION-MNIST
# ─────────────────────────────────────────────
def load_fashion_mnist():
    """
    Cara 1: Download otomatis via Keras (direkomendasikan).
    Cara 2: Jika tidak ada koneksi, gunakan dataset simulasi lokal.
    """
    print("=" * 60)
    print("LOADING FASHION-MNIST DATASET")
    print("=" * 60)

    try:
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype(np.float32)
        X_test  = X_test.reshape(-1, 784).astype(np.float32)
        print(f"[OK] Keras: Train {X_train.shape} | Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    except Exception:
        pass

    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
        X, y = data.data.astype(np.float32), data.target.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=42, stratify=y)
        print(f"[OK] OpenML: Train {X_train.shape} | Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    except Exception:
        pass

    # Fallback: generate synthetic dataset (struktur identik Fashion-MNIST)
    print("[INFO] Menggunakan dataset lokal (simulasi Fashion-MNIST 28x28)...")
    try:
        X_train = np.load('X_train.npy')
        X_test  = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test  = np.load('y_test.npy')
    except FileNotFoundError:
        print("[INFO] Membuat dataset simulasi baru...")
        np.random.seed(42)
        def make_class_pattern(cls, n, sz=28, noise=25):
            imgs = []
            for _ in range(n):
                img = np.zeros((sz, sz))
                if cls == 0:
                    img[5:20, 8:20] = 200; img[5:12, 2:8] = 180; img[5:12, 20:26] = 180
                elif cls == 1:
                    img[5:15, 10:18] = 200; img[15:26, 5:13] = 190; img[15:26, 15:23] = 190
                elif cls == 2:
                    img[4:22, 7:21] = 210; img[4:10, 2:7] = 190; img[4:10, 21:26] = 190
                elif cls == 3:
                    img[4:9, 10:18] = 200; img[9:25, 7:21] = 180
                elif cls == 4:
                    img[4:24, 6:22] = 210; img[4:14, 1:6] = 190; img[4:14, 22:27] = 190
                elif cls == 5:
                    img[18:24, 5:23] = 200; img[10:18, 7:11] = 180; img[10:18, 17:21] = 180
                elif cls == 6:
                    img[5:21, 8:20] = 200; img[5:15, 2:8] = 180; img[5:15, 20:26] = 180
                elif cls == 7:
                    img[16:24, 3:25] = 210; img[8:16, 5:22] = 190
                elif cls == 8:
                    img[6:22, 6:22] = 200; img[4:6, 10:18] = 170
                elif cls == 9:
                    img[14:26, 4:24] = 210; img[5:14, 8:20] = 190
                imgs.append(np.clip(img + np.random.normal(0, noise, (sz, sz)), 0, 255).flatten())
            return np.array(imgs, dtype=np.float32)

        Xtr, ytr, Xte, yte = [], [], [], []
        for c in range(10):
            Xtr.append(make_class_pattern(c, 600)); ytr.extend([c]*600)
            Xte.append(make_class_pattern(c, 100)); yte.extend([c]*100)
        X_train = np.vstack(Xtr); y_train = np.array(ytr)
        X_test  = np.vstack(Xte); y_test  = np.array(yte)
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
        np.save('X_test.npy', X_test);   np.save('y_test.npy', y_test)

    print(f"[OK] Lokal: Train {X_train.shape} | Test {X_test.shape}")
    return X_train, X_test, y_train.astype(int), y_test.astype(int)


# ─────────────────────────────────────────────
# 3. VISUALISASI SAMPEL DATASET
# ─────────────────────────────────────────────
def plot_dataset_samples(X, y):
    print("\n[PLOT] Visualisasi sampel dataset...")
    fig, axes = plt.subplots(2, 10, figsize=(20, 6))
    fig.suptitle('Sampel Dataset Fashion-MNIST (2 sampel per kelas)',
                 fontsize=14, fontweight='bold', y=0.98)
    for cls in range(10):
        idx = np.where(y == cls)[0][:2]
        for row, i in enumerate(idx):
            ax = axes[row, cls]
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            ax.set_title(CLASS_NAMES[cls], fontsize=7, pad=2)
            ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ─────────────────────────────────────────────
# 4. EKSTRAKSI FITUR
# ─────────────────────────────────────────────
def extract_hog_features(X):
    """HOG – Histogram of Oriented Gradients"""
    feats = []
    for img in X:
        im = img.reshape(28, 28)
        f = hog(im, orientations=9, pixels_per_cell=(7, 7),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        feats.append(f)
    return np.array(feats, dtype=np.float32)

def extract_lbp_features(X, radius=1, n_points=8):
    """LBP – Local Binary Pattern (tekstur)"""
    feats = []
    for img in X:
        im = (img.reshape(28, 28) / 255.0)
        lbp = local_binary_pattern(im, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats, dtype=np.float32)

def extract_combined_features(X):
    """Gabungkan HOG + LBP"""
    print("   Mengekstrak fitur HOG...")
    hog_f = extract_hog_features(X)
    print(f"   HOG shape: {hog_f.shape}")
    print("   Mengekstrak fitur LBP...")
    lbp_f = extract_lbp_features(X)
    print(f"   LBP shape: {lbp_f.shape}")
    combined = np.hstack([hog_f, lbp_f])
    print(f"   Combined shape: {combined.shape}")
    return combined, hog_f, lbp_f


# ─────────────────────────────────────────────
# 5. VISUALISASI FITUR HOG
# ─────────────────────────────────────────────
def plot_hog_visualization(X, y):
    print("\n[PLOT] Visualisasi HOG per kelas...")
    fig, axes = plt.subplots(3, 10, figsize=(22, 7))
    fig.suptitle('Visualisasi Ekstraksi Fitur HOG per Kelas Fashion-MNIST',
                 fontsize=13, fontweight='bold')
    for cls in range(10):
        idx = np.where(y == cls)[0][0]
        img = X[idx].reshape(28, 28)
        fd, hog_img = hog(img, orientations=9, pixels_per_cell=(7, 7),
                          cells_per_block=(2, 2), visualize=True, feature_vector=True)
        axes[0, cls].imshow(img, cmap='gray');       axes[0, cls].set_title(CLASS_NAMES[cls], fontsize=7); axes[0, cls].axis('off')
        axes[1, cls].imshow(hog_img, cmap='magma');  axes[1, cls].set_title('HOG', fontsize=7);            axes[1, cls].axis('off')
        lbp = local_binary_pattern(img/255.0, 8, 1, method='uniform')
        axes[2, cls].imshow(lbp, cmap='hot');        axes[2, cls].set_title('LBP', fontsize=7);            axes[2, cls].axis('off')
    for r, lbl in enumerate(['Original', 'HOG', 'LBP']):
        axes[r, 0].set_ylabel(lbl, fontsize=9, labelpad=5)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 6. IMPLEMENTASI KNN – VARIASI K & METRIK
# ─────────────────────────────────────────────
def run_knn_experiments(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("EKSPERIMEN KNN")
    print("=" * 60)

    k_values  = [1, 3, 5, 7, 9, 11]
    metrics   = ['euclidean', 'manhattan', 'minkowski']
    results   = []

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    for metric in metrics:
        print(f"\n  Metrik: {metric.upper()}")
        for k in k_values:
            p = 3 if metric == 'minkowski' else 2
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p, n_jobs=-1)

            t0 = time.time()
            clf.fit(Xtr, y_train)
            t_train = time.time() - t0

            t0 = time.time()
            y_pred = clf.predict(Xte)
            t_infer = time.time() - t0

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({'Algoritma': 'KNN', 'Metrik/Kernel': metric,
                            'k/C': k, 'Accuracy': acc, 'Precision': prec,
                            'Recall': rec, 'F1-Score': f1,
                            'Train Time(s)': t_train, 'Infer Time(s)': t_infer,
                            'y_pred': y_pred})
            print(f"    k={k:2d} | Acc={acc:.4f} | F1={f1:.4f} | "
                  f"Train={t_train:.3f}s | Infer={t_infer:.3f}s")

    return results


# ─────────────────────────────────────────────
# 7. PLOT KNN – PENGARUH K TERHADAP AKURASI
# ─────────────────────────────────────────────
def plot_knn_k_analysis(knn_results):
    print("\n[PLOT] Analisis K vs Akurasi KNN...")
    metrics = ['euclidean', 'manhattan', 'minkowski']
    colors  = ['#2196F3', '#F44336', '#4CAF50']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Analisis Pengaruh K terhadap Akurasi & Overfitting – KNN',
                 fontsize=13, fontweight='bold')

    ax1, ax2 = axes

    for metric, color in zip(metrics, colors):
        rows = [r for r in knn_results if r['Metrik/Kernel'] == metric]
        ks   = [r['k/C'] for r in rows]
        accs = [r['Accuracy'] for r in rows]
        ax1.plot(ks, accs, 'o-', label=metric, color=color, lw=2, ms=7)

    ax1.set_xlabel('Nilai K', fontsize=11); ax1.set_ylabel('Akurasi', fontsize=11)
    ax1.set_title('Akurasi vs K (berbagai metrik jarak)', fontsize=11)
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xticks([1, 3, 5, 7, 9, 11])

    # Bias-Variance trade-off annotation
    rows_eu = [r for r in knn_results if r['Metrik/Kernel'] == 'euclidean']
    ks   = [r['k/C'] for r in rows_eu]
    accs = [r['Accuracy'] for r in rows_eu]
    ax2.plot(ks, accs, 'o-', color='#2196F3', lw=2.5, ms=8)
    ax2.axvspan(1, 2, alpha=0.1, color='red',   label='Overfitting zone (k kecil)')
    ax2.axvspan(9, 12, alpha=0.1, color='blue', label='Underfitting zone (k besar)')
    ax2.set_xlabel('Nilai K', fontsize=11); ax2.set_ylabel('Akurasi', fontsize=11)
    ax2.set_title('Bias-Variance Trade-off (Euclidean)', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xticks([1, 3, 5, 7, 9, 11])

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 8. IMPLEMENTASI SVM – VARIASI KERNEL & PARAM
# ─────────────────────────────────────────────
def run_svm_experiments(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("EKSPERIMEN SVM")
    print("=" * 60)

    configs = [
        # (kernel, C, gamma, label)
        ('linear', 0.1, 'scale', 'Linear C=0.1'),
        ('linear', 1.0, 'scale', 'Linear C=1'),
        ('linear', 10,  'scale', 'Linear C=10'),
        ('linear', 100, 'scale', 'Linear C=100'),
        ('poly',   1,   'scale', 'Poly C=1'),
        ('poly',   10,  'scale', 'Poly C=10'),
        ('rbf',    1,   0.001,   'RBF C=1 γ=0.001'),
        ('rbf',    1,   0.01,    'RBF C=1 γ=0.01'),
        ('rbf',    10,  0.001,   'RBF C=10 γ=0.001'),
        ('rbf',    10,  0.01,    'RBF C=10 γ=0.01'),
        ('rbf',    10,  0.1,     'RBF C=10 γ=0.1'),
        ('rbf',    100, 0.001,   'RBF C=100 γ=0.001'),
    ]

    results = []
    scaler  = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    for kernel, C, gamma, label in configs:
        clf = SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape='ovr',
                  random_state=42, max_iter=2000)
        t0 = time.time()
        clf.fit(Xtr, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(Xte)
        t_infer = time.time() - t0

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({'Algoritma': 'SVM', 'Metrik/Kernel': kernel,
                        'k/C': C, 'Gamma': gamma, 'Label': label,
                        'Accuracy': acc, 'Precision': prec,
                        'Recall': rec, 'F1-Score': f1,
                        'Train Time(s)': t_train, 'Infer Time(s)': t_infer,
                        'y_pred': y_pred})
        print(f"  {label:25s} | Acc={acc:.4f} | F1={f1:.4f} | "
              f"Train={t_train:.3f}s | Infer={t_infer:.3f}s")

    return results


# ─────────────────────────────────────────────
# 9. DECISION BOUNDARY (PCA 2D)
# ─────────────────────────────────────────────
def plot_decision_boundary(X_train, y_train, scaler):
    print("\n[PLOT] Decision Boundary PCA 2D...")

    pca = PCA(n_components=2, random_state=42)
    Xtr_s = scaler.transform(X_train)
    Xtr2d = pca.fit_transform(Xtr_s)

    classifiers = {
        'KNN (k=5, Euclidean)': KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1),
        'SVM Linear (C=1)':     SVC(kernel='linear', C=1, random_state=42, max_iter=1000),
        'SVM RBF (C=10, γ=0.01)': SVC(kernel='rbf', C=10, gamma=0.01, random_state=42, max_iter=1000),
    }

    # Gunakan hanya 5 kelas agar lebih jelas
    mask = y_train < 5
    X2d_5 = Xtr2d[mask]; y_5 = y_train[mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Decision Boundary (PCA 2D, 5 kelas pertama)',
                 fontsize=13, fontweight='bold')

    colors5 = ['#E53935', '#8E24AA', '#1E88E5', '#43A047', '#FB8C00']
    cmap    = plt.cm.get_cmap('RdYlBu', 5)

    x_min, x_max = X2d_5[:, 0].min()-1, X2d_5[:, 0].max()+1
    y_min, y_max = X2d_5[:, 1].min()-1, X2d_5[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    for ax, (name, clf) in zip(axes, classifiers.items()):
        clf.fit(X2d_5, y_5)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        for c, col in enumerate(colors5):
            idx = y_5 == c
            ax.scatter(X2d_5[idx, 0], X2d_5[idx, 1], c=col, s=10,
                       label=CLASS_NAMES[c], alpha=0.7, edgecolors='none')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 10. CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrices(knn_results, svm_results, y_test):
    print("\n[PLOT] Confusion Matrix (model terbaik KNN & SVM)...")

    best_knn = max(knn_results, key=lambda r: r['Accuracy'])
    best_svm = max(svm_results, key=lambda r: r['Accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, result, title in zip(
            axes,
            [best_knn, best_svm],
            [f"KNN – k={best_knn['k/C']}, {best_knn['Metrik/Kernel']} (Acc={best_knn['Accuracy']:.4f})",
             f"SVM – {best_svm.get('Label', '')} (Acc={best_svm['Accuracy']:.4f})"]):
        cm = confusion_matrix(y_test, result['y_pred'])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={'label': '%'})
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.suptitle('Confusion Matrix – Model Terbaik KNN vs SVM',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 11. ROC CURVE (One-vs-Rest)
# ─────────────────────────────────────────────
def plot_roc_curves(X_train, X_test, y_train, y_test):
    print("\n[PLOT] ROC Curve & AUC (One-vs-Rest)...")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    n_classes = 10
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ROC Curve & AUC – KNN vs SVM (One-vs-Rest)',
                 fontsize=13, fontweight='bold')

    classifiers = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM RBF':   SVC(kernel='rbf', C=10, gamma=0.01, probability=True,
                         random_state=42, max_iter=2000),
    }

    cmap = plt.cm.tab10
    for ax, (name, clf) in zip(axes, classifiers.items()):
        clf.fit(Xtr, y_train)
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(Xte)
        else:
            y_score = clf.decision_function(Xte)
            if y_score.ndim == 1:
                y_score = np.column_stack([-y_score, y_score])

        auc_scores = []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            ax.plot(fpr, tpr, lw=1.2, color=cmap(c/n_classes),
                    label=f'{CLASS_NAMES[c]} (AUC={roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}\nMean AUC = {np.mean(auc_scores):.4f}', fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 12. LEARNING CURVE
# ─────────────────────────────────────────────
def plot_learning_curves(X_train, y_train):
    print("\n[PLOT] Learning Curve...")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        'KNN (k=5, Euclidean)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM RBF (C=10, γ=0.01)': SVC(kernel='rbf', C=10, gamma=0.01,
                                       random_state=42, max_iter=2000),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Curve – KNN vs SVM', fontsize=13, fontweight='bold')

    for ax, (name, clf) in zip(axes, classifiers.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            clf, Xtr, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy')

        tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
        va_mean = val_scores.mean(axis=1);   va_std = val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, 'o-', color='#2196F3', lw=2, label='Training Score')
        ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color='#2196F3')
        ax.plot(train_sizes, va_mean, 's-', color='#F44336', lw=2, label='Validation Score')
        ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color='#F44336')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Jumlah Sampel Training'); ax.set_ylabel('Akurasi')
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 13. CROSS-VALIDATION (Stratified K-Fold)
# ─────────────────────────────────────────────
def run_cross_validation(X_train, y_train):
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (Stratified 5-Fold)")
    print("=" * 60)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        'KNN k=3 Euclidean': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        'KNN k=5 Euclidean': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'KNN k=7 Euclidean': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        'SVM Linear C=1':    SVC(kernel='linear', C=1, random_state=42, max_iter=1000),
        'SVM RBF C=10':      SVC(kernel='rbf', C=10, gamma=0.01, random_state=42, max_iter=2000),
        'SVM Poly C=10':     SVC(kernel='poly', C=10, random_state=42, max_iter=1000),
    }

    cv_results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, Xtr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:30s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Plot CV results
    fig, ax = plt.subplots(figsize=(12, 5))
    names   = list(cv_results.keys())
    means   = [cv_results[n].mean() for n in names]
    stds    = [cv_results[n].std()  for n in names]
    colors  = ['#2196F3']*3 + ['#F44336']*3
    bars    = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Akurasi CV (mean ± std)', fontsize=11)
    ax.set_title('Hasil Cross-Validation 5-Fold (Stratified) – KNN vs SVM',
                 fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(color='#2196F3', label='KNN'), Patch(color='#F44336', label='SVM')]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

    return cv_results


# ─────────────────────────────────────────────
# 14. GRIDSEARCHCV – HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def run_gridsearch(X_train, y_train):
    print("\n" + "=" * 60)
    print("GRIDSEARCHCV – HYPERPARAMETER TUNING")
    print("=" * 60)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # KNN GridSearch
    print("\n  GridSearch KNN...")
    param_knn = {'n_neighbors': [3, 5, 7, 9],
                 'metric': ['euclidean', 'manhattan']}
    gs_knn = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_knn,
                          cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    gs_knn.fit(Xtr, y_train)
    print(f"  Best KNN: {gs_knn.best_params_}  Score: {gs_knn.best_score_:.4f}")

    # SVM GridSearch
    print("\n  GridSearch SVM (RBF)...")
    param_svm = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    gs_svm = GridSearchCV(SVC(kernel='rbf', random_state=42, max_iter=2000),
                          param_svm, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    gs_svm.fit(Xtr, y_train)
    print(f"  Best SVM: {gs_svm.best_params_}  Score: {gs_svm.best_score_:.4f}")

    # Heatmap SVM GridSearch
    svm_scores = gs_svm.cv_results_['mean_test_score'].reshape(3, 3)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(svm_scores, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[0.001, 0.01, 0.1],
                yticklabels=[1, 10, 100], ax=ax)
    ax.set_xlabel('Gamma'); ax.set_ylabel('C')
    ax.set_title('GridSearchCV – SVM RBF: Accuracy Heatmap\n(5-Fold Stratified CV)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return gs_knn, gs_svm


# ─────────────────────────────────────────────
# 15. TABEL PERBANDINGAN & PLOT FINAL
# ─────────────────────────────────────────────
def plot_final_comparison(knn_results, svm_results):
    print("\n[PLOT] Tabel & Grafik Perbandingan Final...")

    # Pilih representatif
    knn_rep = [r for r in knn_results if r['Metrik/Kernel'] == 'euclidean']
    svm_rep = [r for r in svm_results]

    # Gabungkan data
    all_names, all_acc, all_f1, all_train_t, all_infer_t = [], [], [], [], []

    for r in knn_rep:
        all_names.append(f"KNN k={r['k/C']}")
        all_acc.append(r['Accuracy']); all_f1.append(r['F1-Score'])
        all_train_t.append(r['Train Time(s)']); all_infer_t.append(r['Infer Time(s)'])

    for r in svm_rep:
        label = r.get('Label', f"SVM {r['Metrik/Kernel']} C={r['k/C']}")
        all_names.append(label)
        all_acc.append(r['Accuracy']); all_f1.append(r['F1-Score'])
        all_train_t.append(r['Train Time(s)']); all_infer_t.append(r['Infer Time(s)'])

    x = np.arange(len(all_names))
    n_knn = len(knn_rep); n_svm = len(svm_rep)

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2196F3']*n_knn + ['#F44336']*n_svm
    bars = ax1.bar(x, all_acc, color=colors, alpha=0.85, edgecolor='black', width=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(all_names, rotation=55, ha='right', fontsize=7)
    ax1.set_ylabel('Accuracy'); ax1.set_title('(a) Akurasi Semua Model', fontweight='bold')
    ax1.set_ylim([0, 1.1]); ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=max(all_acc), color='gold', lw=1.5, linestyle='--', label=f'Max={max(all_acc):.4f}')
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, all_acc):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', fontsize=6, rotation=0)

    # (b) F1-Score
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, all_f1, color=colors, alpha=0.85, edgecolor='black', width=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(all_names, rotation=55, ha='right', fontsize=7)
    ax2.set_ylabel('F1-Score'); ax2.set_title('(b) F1-Score Semua Model', fontweight='bold')
    ax2.set_ylim([0, 1.1]); ax2.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars2, all_f1):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', fontsize=6)

    # (c) Train Time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x, all_train_t, color=colors, alpha=0.85, edgecolor='black', width=0.6)
    ax3.set_xticks(x); ax3.set_xticklabels(all_names, rotation=55, ha='right', fontsize=7)
    ax3.set_ylabel('Waktu (detik)'); ax3.set_title('(c) Waktu Training', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # (d) Inference Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x, all_infer_t, color=colors, alpha=0.85, edgecolor='black', width=0.6)
    ax4.set_xticks(x); ax4.set_xticklabels(all_names, rotation=55, ha='right', fontsize=7)
    ax4.set_ylabel('Waktu (detik)'); ax4.set_title('(d) Waktu Inference', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elems = [Patch(color='#2196F3', label='KNN'), Patch(color='#F44336', label='SVM')]
    fig.legend(handles=legend_elems, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Perbandingan Komprehensif KNN vs SVM – Fashion-MNIST',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.show()


# ─────────────────────────────────────────────
# 16. CLASSIFICATION REPORT (TEKS)
# ─────────────────────────────────────────────
def print_classification_reports(knn_results, svm_results, y_test):
    best_knn = max(knn_results, key=lambda r: r['Accuracy'])
    best_svm = max(svm_results, key=lambda r: r['Accuracy'])

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT – KNN TERBAIK")
    print(f"k={best_knn['k/C']}, Metrik={best_knn['Metrik/Kernel']}")
    print("=" * 60)
    print(classification_report(y_test, best_knn['y_pred'],
                                target_names=CLASS_NAMES, zero_division=0))

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT – SVM TERBAIK")
    print(f"{best_svm.get('Label', '')}")
    print("=" * 60)
    print(classification_report(y_test, best_svm['y_pred'],
                                target_names=CLASS_NAMES, zero_division=0))


# ─────────────────────────────────────────────
# 17. RINGKASAN TABEL AKHIR
# ─────────────────────────────────────────────
def print_summary_table(knn_results, svm_results):
    print("\n" + "=" * 80)
    print("TABEL RINGKASAN PERBANDINGAN PERFORMA")
    print("=" * 80)
    header = f"{'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'Train(s)':>9} {'Infer(s)':>9}"
    print(header)
    print("-" * 80)

    all_res = knn_results + svm_results
    all_res.sort(key=lambda r: r['Accuracy'], reverse=True)

    for r in all_res:
        if r['Algoritma'] == 'KNN':
            name = f"KNN k={r['k/C']} {r['Metrik/Kernel']}"
        else:
            name = r.get('Label', f"SVM {r['Metrik/Kernel']}")
        print(f"{name:<30} {r['Accuracy']:>9.4f} {r['Precision']:>10.4f} "
              f"{r['Recall']:>8.4f} {r['F1-Score']:>9.4f} "
              f"{r['Train Time(s)']:>9.3f} {r['Infer Time(s)']:>9.3f}")

    print("=" * 80)
    best = all_res[0]
    best_label = best.get('Label') or f"KNN k={best['k/C']} {best['Metrik/Kernel']}"
    print(f"\n MODEL TERBAIK: {best_label}")
    print(f"   Accuracy: {best['Accuracy']:.4f} | F1-Score: {best['F1-Score']:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PRAKTIKUM: KOMPARASI KNN vs SVM – FASHION-MNIST         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Load Dataset
    X_train_raw, X_test_raw, y_train, y_test = load_fashion_mnist()

    # Ambil subset untuk kecepatan (gunakan semua data jika ingin akurasi max)
    N_TRAIN = min(3000, len(X_train_raw))  # set lebih besar untuk akurasi maksimal
    N_TEST  = min(500,  len(X_test_raw))
    print(f"\n[INFO] Menggunakan {N_TRAIN} train + {N_TEST} test samples")

    idx_tr = np.random.RandomState(42).permutation(len(X_train_raw))[:N_TRAIN]
    idx_te = np.random.RandomState(42).permutation(len(X_test_raw))[:N_TEST]
    X_train_s = X_train_raw[idx_tr]; y_train_s = y_train[idx_tr]
    X_test_s  = X_test_raw[idx_te];  y_test_s  = y_test[idx_te]

    # 2. Plot sampel
    plot_dataset_samples(X_train_s, y_train_s)

    # 3. Ekstraksi Fitur
    print("\n" + "=" * 60)
    print("EKSTRAKSI FITUR (HOG + LBP)")
    print("=" * 60)
    X_train_feat, hog_tr, lbp_tr = extract_combined_features(X_train_s)
    X_test_feat,  hog_te, lbp_te = extract_combined_features(X_test_s)

    # 4. Visualisasi HOG
    plot_hog_visualization(X_train_s, y_train_s)

    # 5. KNN Experiments
    knn_results = run_knn_experiments(X_train_feat, X_test_feat, y_train_s, y_test_s)
    plot_knn_k_analysis(knn_results)

    # 6. SVM Experiments
    scaler = StandardScaler()
    scaler.fit(X_train_feat)
    svm_results = run_svm_experiments(X_train_feat, X_test_feat, y_train_s, y_test_s)

    # 7. Decision Boundary
    plot_decision_boundary(X_train_feat, y_train_s, scaler)

    # 8. Confusion Matrix
    plot_confusion_matrices(knn_results, svm_results, y_test_s)

    # 9. ROC Curves
    plot_roc_curves(X_train_feat, X_test_feat, y_train_s, y_test_s)

    # 10. Learning Curves
    plot_learning_curves(X_train_feat, y_train_s)

    # 11. Cross-Validation
    cv_results = run_cross_validation(X_train_feat, y_train_s)

    # 12. GridSearchCV
    gs_knn, gs_svm = run_gridsearch(X_train_feat, y_train_s)

    # 13. Final Comparison Plot
    plot_final_comparison(knn_results, svm_results)

    # 14. Classification Reports
    print_classification_reports(knn_results, svm_results, y_test_s)

    # 15. Summary Table
    print_summary_table(knn_results, svm_results)

    print("\n" + "=" * 60)
    print("✅ SEMUA EKSPERIMEN SELESAI!")
    print("=" * 60)