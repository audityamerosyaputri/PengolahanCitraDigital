import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ========== 1. GENERATE SYNTHETIC IMAGES ==========
def create_bimodal_image(size=256):
    img = np.full((size, size), 30, dtype=np.uint8)
    cv2.rectangle(img, (40,40), (110,110), 220, -1)
    cv2.circle(img, (180,80), 45, 200, -1)
    cv2.rectangle(img, (60,160), (130,220), 210, -1)
    cv2.circle(img, (190,190), 35, 195, -1)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gt = np.zeros((size,size), dtype=np.uint8)
    cv2.rectangle(gt, (40,40), (110,110), 255, -1)
    cv2.circle(gt, (180,80), 45, 255, -1)
    cv2.rectangle(gt, (60,160), (130,220), 255, -1)
    cv2.circle(gt, (190,190), 35, 255, -1)
    return img, gt

def create_uneven_illumination_image(size=256):
    img = np.full((size, size), 80, dtype=np.uint8)
    Y,X = np.mgrid[0:size,0:size]
    illum = (180 * np.exp(-((X**2+Y**2)/(2*(size*0.6)**2)))).astype(np.uint8)
    cv2.ellipse(img, (80,80), (40,30), 0,0,360,140,-1)
    cv2.rectangle(img, (160,50), (220,110), 130, -1)
    cv2.ellipse(img, (70,190), (35,25), 30,0,360,145,-1)
    cv2.rectangle(img, (155,165), (225,230), 135, -1)
    result = np.clip(img.astype(np.float32) + illum.astype(np.float32)*0.4, 0,255).astype(np.uint8)
    noise = np.random.normal(0,6,result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16)+noise,0,255).astype(np.uint8)
    gt = np.zeros((size,size), dtype=np.uint8)
    cv2.ellipse(gt, (80,80), (40,30), 0,0,360,255,-1)
    cv2.rectangle(gt, (160,50), (220,110), 255, -1)
    cv2.ellipse(gt, (70,190), (35,25), 30,0,360,255,-1)
    cv2.rectangle(gt, (155,165), (225,230), 255, -1)
    return result, gt

def create_overlapping_objects_image(size=256):
    img = np.full((size, size), 40, dtype=np.uint8)
    circles = [(80,80,38),(130,70,35),(170,110,32),(75,155,36),(135,160,33),
               (190,60,30),(195,190,34),(60,200,31),(155,210,35),(110,115,30)]
    for cx,cy,r in circles:
        cv2.circle(img, (cx,cy), r, 190, -1)
        cv2.circle(img, (cx,cy), r, 160, 2)
    noise = np.random.normal(0,10,img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
    gt = np.zeros((size,size), dtype=np.uint8)
    for cx,cy,r in circles:
        cv2.circle(gt, (cx,cy), r, 255, -1)
    return img, gt

# ========== 2. THRESHOLDING ==========
def global_thresholding(img, T=None):
    t0 = time.perf_counter()
    if T is None: T = img.mean()
    _, res = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return res.astype(np.uint8), time.perf_counter()-t0, T

def otsu_thresholding(img):
    t0 = time.perf_counter()
    T, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return res.astype(np.uint8), time.perf_counter()-t0, T

def adaptive_thresholding(img, method='mean', block=31, C=5):
    t0 = time.perf_counter()
    if method == 'mean':
        res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
    else:
        res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)
    return res.astype(np.uint8), time.perf_counter()-t0

# ========== 3. EDGE DETECTION ==========
def sobel_detection(img):
    t0 = time.perf_counter()
    Gx = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1,0, ksize=3)
    Gy = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0,1, ksize=3)
    mag = np.sqrt(Gx**2+Gy**2)
    mag_norm = cv2.normalize(mag, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bin = cv2.threshold(mag_norm, 50, 255, cv2.THRESH_BINARY)
    return bin.astype(np.uint8), mag, np.arctan2(Gy,Gx)*180/np.pi, time.perf_counter()-t0

def prewitt_detection(img):
    t0 = time.perf_counter()
    Kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
    Ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float64)
    Gx = cv2.filter2D(img.astype(np.float64), -1, Kx)
    Gy = cv2.filter2D(img.astype(np.float64), -1, Ky)
    mag = np.sqrt(Gx**2+Gy**2)
    mag_norm = cv2.normalize(mag, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bin = cv2.threshold(mag_norm, 50, 255, cv2.THRESH_BINARY)
    return bin.astype(np.uint8), mag, time.perf_counter()-t0

def canny_detection(img, low=50, high=150):
    t0 = time.perf_counter()
    edges = cv2.Canny(cv2.GaussianBlur(img, (5,5), 1.4), low, high)
    return edges.astype(np.uint8), time.perf_counter()-t0

def edge_to_mask(edge_img):
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    h,w = closed.shape
    mask = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(closed, mask, (0,0), 255)
    return cv2.bitwise_or(closed, cv2.bitwise_not(closed))

# ========== 4. REGION-BASED ==========
def region_growing(img, seeds=None, thresh=25):
    t0 = time.perf_counter()
    h,w = img.shape
    visited = np.zeros((h,w), bool)
    res = np.zeros((h,w), np.uint8)
    if seeds is None:
        seeds = [(i,j) for i in range(20,h-20,60) for j in range(20,w-20,60) if img[i,j] > img.mean()+20]
    q = deque(seeds)
    for y,x in seeds:
        visited[y,x]=True; res[y,x]=255
    while q:
        y,x = q.popleft()
        seed_val = float(img[y,x])
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny,nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w and not visited[ny,nx] and abs(int(img[ny,nx])-seed_val)<thresh:
                visited[ny,nx]=True; res[ny,nx]=255; q.append((ny,nx))
    return res, time.perf_counter()-t0

def watershed_segmentation(img):
    t0 = time.perf_counter()
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)
    res = np.zeros_like(img); res[markers>1]=255
    return res, time.perf_counter()-t0

def connected_components_analysis(img):
    t0 = time.perf_counter()
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin, connectivity=8)
    res = np.zeros_like(img)
    for i in range(1,n):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            res[labels==i] = 255
    return res, n-1, time.perf_counter()-t0

# ========== 5. EVALUATION METRICS ==========
def compute_metrics(pred, gt):
    p = pred>127; g = gt>127
    TP = (p & g).sum(); FP = (p & ~g).sum(); FN = (~p & g).sum(); TN = (~p & ~g).sum()
    eps = 1e-7
    return {'IoU': round(TP/(TP+FP+FN+eps),4), 'Dice': round(2*TP/(2*TP+FP+FN+eps),4),
            'Accuracy': round((TP+TN)/(TP+FP+FN+TN+eps),4),
            'Precision': round(TP/(TP+FP+eps),4), 'Recall': round(TP/(TP+FN+eps),4)}

# ========== 6. RUN EXPERIMENTS ==========
def run_experiments():
    images = {'Bimodal': create_bimodal_image(),
              'Uneven Illumination': create_uneven_illumination_image(),
              'Overlapping Objects': create_overlapping_objects_image()}
    results = {}
    for name, (img, gt) in images.items():
        print(f"\n{'='*50}\nProcessing: {name}\n{'='*50}")
        res = {}
        # Thresholding
        mask,t,T = global_thresholding(img); res['Global Threshold'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':f'T={T:.1f}'}
        mask,t,T = otsu_thresholding(img); res["Otsu's Method"] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':f'T={T:.1f}'}
        mask,t = adaptive_thresholding(img,'mean'); res['Adaptive Mean'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':'Block=31'}
        mask,t = adaptive_thresholding(img,'gaussian'); res['Adaptive Gaussian'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':'Block=31'}
        # Edge
        bin, mag, _, t = sobel_detection(img); mask = edge_to_mask(bin); res['Sobel'] = {'mask':mask,'edge':bin,'magnitude':mag,'time':t,'metrics':compute_metrics(mask,gt),'info':'ksize=3'}
        bin, mag, t = prewitt_detection(img); mask = edge_to_mask(bin); res['Prewitt'] = {'mask':mask,'edge':bin,'time':t,'metrics':compute_metrics(mask,gt),'info':'3x3 kernel'}
        edges,t = canny_detection(img,40,120); mask = edge_to_mask(edges); res['Canny'] = {'mask':mask,'edge':edges,'time':t,'metrics':compute_metrics(mask,gt),'info':'low=40,high=120'}
        # Region
        mask,t = region_growing(img); res['Region Growing'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':'thresh=25'}
        mask,t = watershed_segmentation(img); res['Watershed'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':'marker-based'}
        mask,n_comp,t = connected_components_analysis(img); res['Connected Components'] = {'mask':mask,'time':t,'metrics':compute_metrics(mask,gt),'info':f'n={n_comp}'}
        results[name] = {'image':img,'gt':gt,'methods':res}
        for method,data in res.items():
            m = data['metrics']
            print(f"  {method:22s} | IoU={m['IoU']:.3f} | Dice={m['Dice']:.3f} | Acc={m['Accuracy']:.3f} | t={data['time']*1000:.2f}ms")
    return results

# ========== 7. VISUALIZATION ==========
def plot_comparison(results):
    import os
    os.makedirs('figures', exist_ok=True)
    method_list = list(next(iter(results.values()))['methods'].keys())
    img_names = list(results.keys())

    # Fig 1: Original + GT
    fig, axes = plt.subplots(2,3,figsize=(15,9)); fig.suptitle('Citra Asli dan Ground Truth', fontsize=16, fontweight='bold')
    for col,name in enumerate(img_names):
        axes[0,col].imshow(results[name]['image'], cmap='gray'); axes[0,col].set_title(f'Original: {name}'); axes[0,col].axis('off')
        axes[1,col].imshow(results[name]['gt'], cmap='gray'); axes[1,col].set_title(f'Ground Truth: {name}'); axes[1,col].axis('off')
    plt.tight_layout(); plt.show()

    # Fig 2: Thresholding
    thresh = ['Global Threshold', "Otsu's Method", 'Adaptive Mean', 'Adaptive Gaussian']
    fig, axes = plt.subplots(3,4,figsize=(18,13)); fig.suptitle('Hasil Metode Thresholding', fontsize=15, fontweight='bold')
    for row,name in enumerate(img_names):
        for col,m in enumerate(thresh):
            mask = results[name]['methods'][m]['mask']; met = results[name]['methods'][m]['metrics']
            axes[row,col].imshow(mask, cmap='gray'); axes[row,col].set_title(f'{m}\n[{name[:12]}]\nIoU={met["IoU"]:.3f} | Dice={met["Dice"]:.3f}', fontsize=8); axes[row,col].axis('off')
    plt.tight_layout(); plt.show()

    # Fig 3: Edge detection
    edge_m = ['Sobel','Prewitt','Canny']
    fig, axes = plt.subplots(3,4,figsize=(18,13)); fig.suptitle('Hasil Metode Edge Detection', fontsize=15, fontweight='bold')
    for row,name in enumerate(img_names):
        axes[row,0].imshow(results[name]['image'], cmap='gray'); axes[row,0].set_title(f'Original\n{name[:16]}'); axes[row,0].axis('off')
        for col,m in enumerate(edge_m,1):
            edge = results[name]['methods'][m].get('edge', results[name]['methods'][m]['mask'])
            met = results[name]['methods'][m]['metrics']
            axes[row,col].imshow(edge, cmap='gray'); axes[row,col].set_title(f'{m} Edges\nIoU={met["IoU"]:.3f} | Dice={met["Dice"]:.3f}', fontsize=8); axes[row,col].axis('off')
    plt.tight_layout(); plt.show()

    # Fig 4: Region-based
    reg_m = ['Region Growing','Watershed','Connected Components']
    fig, axes = plt.subplots(3,4,figsize=(18,13)); fig.suptitle('Hasil Metode Region-Based', fontsize=15, fontweight='bold')
    for row,name in enumerate(img_names):
        axes[row,0].imshow(results[name]['image'], cmap='gray'); axes[row,0].set_title(f'Original\n{name[:16]}'); axes[row,0].axis('off')
        for col,m in enumerate(reg_m,1):
            mask = results[name]['methods'][m]['mask']; met = results[name]['methods'][m]['metrics']
            axes[row,col].imshow(mask, cmap='gray'); axes[row,col].set_title(f'{m}\nIoU={met["IoU"]:.3f} | Dice={met["Dice"]:.3f}', fontsize=8); axes[row,col].axis('off')
    plt.tight_layout(); plt.show()

    # Fig 5: Contour overlay terbaik
    fig, axes = plt.subplots(3,3,figsize=(15,14)); fig.suptitle('Overlay Kontur: Original | Ground Truth | Hasil Terbaik', fontsize=13, fontweight='bold')
    for row,name in enumerate(img_names):
        img = results[name]['image']; gt = results[name]['gt']
        best = max(results[name]['methods'].items(), key=lambda x: x[1]['metrics']['IoU'])
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_gt = img_color.copy(); img_pred = img_color.copy()
        contours_gt,_ = cv2.findContours((gt>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_gt, contours_gt, -1, (0,255,0), 2)
        contours_pred,_ = cv2.findContours((best[1]['mask']>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_pred, contours_pred, -1, (0,0,255), 2)
        axes[row,0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)); axes[row,0].set_title(f'Original\n{name}'); axes[row,0].axis('off')
        axes[row,1].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)); axes[row,1].set_title('Ground Truth Overlay\n(kontur hijau)'); axes[row,1].axis('off')
        axes[row,2].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)); axes[row,2].set_title(f'Best: {best[0]}\nIoU={best[1]["metrics"]["IoU"]:.3f} (kontur merah)'); axes[row,2].axis('off')
    plt.tight_layout(); plt.show()

    # Fig 6: Bar chart IoU
    fig, axes = plt.subplots(1,3,figsize=(20,6)); fig.suptitle('Perbandingan Metrik per Citra', fontsize=14, fontweight='bold')
    for col,name in enumerate(img_names):
        ious = [results[name]['methods'][m]['metrics']['IoU'] for m in method_list]
        bars = axes[col].barh(method_list, ious, color=plt.cm.tab10(np.linspace(0,1,len(method_list))))
        axes[col].set_xlim(0,1.05); axes[col].set_xlabel('IoU Score'); axes[col].set_title(f'IoU Score\n{name}', fontsize=11, fontweight='bold')
        for bar,val in zip(bars, ious): axes[col].text(val+0.01, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout(); plt.show()

    # Fig 7: Waktu komputasi
    fig, ax = plt.subplots(figsize=(14,6)); x = np.arange(len(method_list)); width=0.25
    colors = ['#2196F3','#4CAF50','#FF5722']
    for i,name in enumerate(img_names):
        times = [results[name]['methods'][m]['time']*1000 for m in method_list]
        ax.bar(x+i*width, times, width, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x+width); ax.set_xticklabels(method_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Waktu (ms)'); ax.set_title('Waktu Komputasi Setiap Metode', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

    # Fig 8 & 9: Robustness (noise & illumination)
    img_test, gt_test = create_bimodal_image()
    noise_levels = [0,10,20,30,50]
    methods_test = ["Otsu's Method",'Adaptive Gaussian','Canny','Watershed']
    rob_noise = {m:[] for m in methods_test}
    for sigma in noise_levels:
        noisy = np.clip(img_test.astype(np.int16)+np.random.normal(0,sigma,img_test.shape).astype(np.int16),0,255).astype(np.uint8)
        mask,_,_ = otsu_thresholding(noisy); rob_noise["Otsu's Method"].append(compute_metrics(mask,gt_test)['IoU'])
        mask,_ = adaptive_thresholding(noisy,'gaussian'); rob_noise['Adaptive Gaussian'].append(compute_metrics(mask,gt_test)['IoU'])
        edges,_ = canny_detection(noisy,40,120); rob_noise['Canny'].append(compute_metrics(edge_to_mask(edges),gt_test)['IoU'])
        mask,_ = watershed_segmentation(noisy); rob_noise['Watershed'].append(compute_metrics(mask,gt_test)['IoU'])
    fig, ax = plt.subplots(figsize=(10,6)); styles = ['-o','-s','-^','-D']
    for (m,vals),sty in zip(rob_noise.items(), styles):
        ax.plot(noise_levels, vals, sty, label=m, linewidth=2, markersize=7)
    ax.set_xlabel('Noise Level (σ)'); ax.set_ylabel('IoU Score'); ax.set_title('Robustness terhadap Noise (Citra Bimodal)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0,1.05); plt.tight_layout(); plt.show()

    img_test, gt_test = create_uneven_illumination_image()
    illum_factors = [0.5,0.8,1.0,1.2,1.5]
    rob_illum = {m:[] for m in methods_test}
    for factor in illum_factors:
        adj = np.clip(img_test.astype(np.float32)*factor,0,255).astype(np.uint8)
        mask,_,_ = otsu_thresholding(adj); rob_illum["Otsu's Method"].append(compute_metrics(mask,gt_test)['IoU'])
        mask,_ = adaptive_thresholding(adj,'gaussian'); rob_illum['Adaptive Gaussian'].append(compute_metrics(mask,gt_test)['IoU'])
        edges,_ = canny_detection(adj,40,120); rob_illum['Canny'].append(compute_metrics(edge_to_mask(edges),gt_test)['IoU'])
        mask,_ = watershed_segmentation(adj); rob_illum['Watershed'].append(compute_metrics(mask,gt_test)['IoU'])
    fig, ax = plt.subplots(figsize=(10,6))
    for (m,vals),sty in zip(rob_illum.items(), styles):
        ax.plot(illum_factors, vals, sty, label=m, linewidth=2, markersize=7)
    ax.set_xlabel('Faktor Iluminasi'); ax.set_ylabel('IoU Score'); ax.set_title('Robustness terhadap Variasi Iluminasi', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0,1.05); plt.tight_layout(); plt.show()

# ========== MAIN ==========
if __name__ == '__main__':
    results = run_experiments()
    plot_comparison(results)
    print("\nSemua berhasil ditampilkan!")