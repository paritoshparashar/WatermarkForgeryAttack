import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
import lpips
import torch

DATASET_DIR = 'Dataset'
CLEAN_DIR = os.path.join(DATASET_DIR, 'clean_targets')
SOURCE_DIR = os.path.join(DATASET_DIR, 'watermarked_sources')
OUTPUT_DIR = 'submission_temp'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# mapping of watermark to clean
WM_MAP = {
    1: (1, 25),
    2: (26, 50),
    3: (51, 75),
    4: (76, 100),
    5: (101, 125),
    6: (126, 150),
    7: (151, 175),
    8: (176, 200)
}

# load and save
def load_img(path):
    return np.array(Image.open(path).convert('RGB'))

def save_img(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

# initialize lpips for s_qlt
print("loading lpips model...")
loss_fn = lpips.LPIPS(net='alex')

#process each watermark
for wm_id in range(1, 9):
    print(f"\n=== Processing WM_{wm_id} ===")
    
    # get all source images for this watermark
    src_folder = os.path.join(SOURCE_DIR, f'WM_{wm_id}')
    src_files = [f for f in os.listdir(src_folder) if f.endswith('.png')]
    
    # load them -> resize -> average
    first = load_img(os.path.join(src_folder, src_files[0]))
    h, w = first.shape[:2]
    
    imgs = []
    for f in src_files:
        img = load_img(os.path.join(src_folder, f))
        # resize if diff dimensions
        if img.shape[0] != h or img.shape[1] != w:
            pil = Image.fromarray(img)
            pil = pil.resize((w, h), Image.BILINEAR)
            img = np.array(pil)
        imgs.append(img)
    
    # average all 25 images
    avg = np.mean(imgs, axis=0)
    
    # subtract a blurred version
    blur = gaussian_filter(avg, sigma=15)
    wm_signal = avg - blur
    
    # inject into the clean target images
    start, end = WM_MAP[wm_id]
    
    
    alpha = 0.3
    
    lpips_vals = []
    
    for i in range(start - 1, end):
        # load clean target
        clean_path = os.path.join(CLEAN_DIR, f"{i + 1}.png")
        clean = load_img(clean_path).astype(float)
        ch, cw = clean.shape[:2]
        
        # resize the watermark signal to match this image's size
        zh = ch / wm_signal.shape[0]
        zw = cw / wm_signal.shape[1]
        resized = zoom(wm_signal, (zh, zw, 1), order=1)
        
        # add the watermark to the clean image
        forged = clean + (alpha * resized)
        forged = np.clip(forged, 0, 255).astype(np.uint8)
        
        # save it
        out_path = os.path.join(OUTPUT_DIR, f"{i + 1}.png")
        save_img(forged, out_path)
        
        # check lpips
        clean_t = torch.tensor(clean.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        forged_t = torch.tensor(forged).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        dist = loss_fn(clean_t, forged_t).item()
        lpips_vals.append(dist)
    
    # print lpips and s_qlt
    avg_lpips = np.mean(lpips_vals)
    est_qlt = np.exp(-8 * avg_lpips)
    print(f"  Avg LPIPS: {avg_lpips:.4f} | Est S_qlt: {est_qlt:.4f}")
