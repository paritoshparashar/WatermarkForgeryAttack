import os, json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
import torch

# Sweep alpha for each WM to find a good LPIPS band before wasting a leaderboard submission.
# Pick alpha per WM that gives LPIPS around 0.02-0.05 (S_qlt ~ 0.67-0.85) and decent signal.
# This script only measures LPIPS locally -- S_det (bit acc) still needs the leaderboard.

DATASET_DIR = 'Dataset'
CLEAN_DIR = os.path.join(DATASET_DIR, 'clean_targets')
SOURCE_DIR = os.path.join(DATASET_DIR, 'watermarked_sources')

WM_MAP = {
    1: (1, 25), 2: (26, 50), 3: (51, 75), 4: (76, 100),
    5: (101, 125), 6: (126, 150), 7: (151, 175), 8: (176, 200),
}

ALPHAS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
METHODS = [
    ('highpass', 15),
    ('highpass', 30),
    ('highpass', 50),
    ('mean_subtract', 0),
    ('direct', 0),
]
N_SAMPLES = 5  # clean images to test per WM


def load_img(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32)


def resize_sig(sig, h, w):
    if sig.shape[0] == h and sig.shape[1] == w:
        return sig
    return zoom(sig, (h / sig.shape[0], w / sig.shape[1], 1), order=3).astype(np.float32)


def extract(imgs, method, sigma):
    avg = np.mean(imgs, axis=0)
    if method == 'highpass':
        lp = np.zeros_like(avg)
        for c in range(avg.shape[2]):
            lp[:, :, c] = gaussian_filter(avg[:, :, c], sigma=sigma)
        return (avg - lp).astype(np.float32)
    elif method == 'mean_subtract':
        return (avg - avg.mean(axis=(0, 1), keepdims=True)).astype(np.float32)
    elif method == 'direct':
        # WMCopier-style: just use (avg - 128) as the watermark
        return (avg - 128.0).astype(np.float32)
    raise ValueError(method)


def texture_mask(img, sigma=5, gamma=0.5, floor=0.3):
    # boost alpha in textured regions, LPIPS cares less there
    g = img.mean(axis=2)
    lp = gaussian_filter(g, sigma=sigma)
    hp = np.abs(g - lp)
    p99 = np.percentile(hp, 99) + 1e-6
    m = np.power(np.clip(hp / p99, 0, 1), gamma)
    m = floor + (1.0 - floor) * m
    return m[:, :, np.newaxis].astype(np.float32)


def main():
    import lpips
    print("loading lpips...")
    loss_fn = lpips.LPIPS(net='alex')

    results = {}
    rng = np.random.default_rng(42)

    for wm_id in range(1, 9):
        print(f"\n=== WM_{wm_id} ===")
        src_folder = os.path.join(SOURCE_DIR, f'WM_{wm_id}')
        src_files = sorted(f for f in os.listdir(src_folder) if f.endswith('.png'))

        first = load_img(os.path.join(src_folder, src_files[0]))
        h, w = first.shape[:2]
        imgs = []
        for f in src_files:
            im = load_img(os.path.join(src_folder, f))
            if im.shape[0] != h or im.shape[1] != w:
                im = np.array(Image.fromarray(im.astype(np.uint8)).resize((w, h), Image.LANCZOS)).astype(np.float32)
            imgs.append(im)
        imgs = np.stack(imgs)

        # pick a few clean images to test on
        start, end = WM_MAP[wm_id]
        sample_ids = list(range(start, end + 1))
        if len(sample_ids) > N_SAMPLES:
            sample_ids = list(rng.choice(sample_ids, size=N_SAMPLES, replace=False))
        cleans = [load_img(os.path.join(CLEAN_DIR, f"{i}.png")) for i in sample_ids]

        wm_res = {}
        for method, sigma in METHODS:
            sig = extract(imgs, method, sigma)
            std = float(np.std(sig))
            key = f"{method}_{sigma}" if method == 'highpass' else method
            print(f"\n  {key}  (std={std:.2f})")
            print(f"    alpha   LPIPS   S_qlt   S_final*")
            sweep = []
            for a in ALPHAS:
                lps = []
                for clean in cleans:
                    ch, cw = clean.shape[:2]
                    r = resize_sig(sig, ch, cw)
                    m = texture_mask(clean)
                    forged = np.clip(clean + a * r * m, 0, 255)
                    ct = torch.from_numpy(clean / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
                    ft = torch.from_numpy(forged / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
                    lps.append(loss_fn(ct, ft).item())
                avg_lp = float(np.mean(lps))
                sq = float(np.exp(-8 * avg_lp))
                # * assumes S_det = 1.0, real score will be lower
                print(f"    {a:5.2f}  {avg_lp:7.4f}  {sq:6.4f}  {sq:7.4f}")
                sweep.append({'alpha': a, 'lpips': avg_lp, 's_qlt': sq})
            wm_res[key] = {'signal_std': std, 'sweep': sweep}

        results[wm_id] = wm_res

    json.dump(results, open('alpha_sweep_results.json', 'w'), indent=2)
    print("\nsaved -> alpha_sweep_results.json")
    # pick alpha that lands LPIPS in [0.02, 0.05] per WM, then put in forge_watermark.py


if __name__ == '__main__':
    main()
