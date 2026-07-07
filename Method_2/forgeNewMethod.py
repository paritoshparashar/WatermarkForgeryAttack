import os, json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom, median_filter
import torch

# Watermark forgery. Averaging attack (Yang et al. NeurIPS 24) + WMCopier-style direct (Dong et al. NeurIPS 25).
# Per-WM config below -- tweak alpha/method/sigma after checking leaderboard.

DATASET_DIR = 'Dataset'
CLEAN_DIR = os.path.join(DATASET_DIR, 'clean_targets')
SOURCE_DIR = os.path.join(DATASET_DIR, 'watermarked_sources')
OUTPUT_DIR = 'submission_temp'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WM_MAP = {
    1: (1, 25),    2: (26, 50),   3: (51, 75),   4: (76, 100),
    5: (101, 125), 6: (126, 150), 7: (151, 175), 8: (176, 200),
}

# alpha = injection strength. method = extraction. sigma = blur for highpass.
# adaptive = modulate with texture mask (LPIPS much lower at same alpha).
PER_WM_CONFIG = {
    1: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    2: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    3: dict(alpha=0.75, method='highpass', sigma=30, adaptive=True),
    4: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    5: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    6: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    7: dict(alpha=1.0,  method='highpass', sigma=30, adaptive=True),
    8: dict(alpha=0.75, method='highpass', sigma=30, adaptive=True),
}


def load_img(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32)


def save_img(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def resize_sig(sig, h, w):
    if sig.shape[0] == h and sig.shape[1] == w:
        return sig
    return zoom(sig, (h / sig.shape[0], w / sig.shape[1], 1), order=3).astype(np.float32)


def extract(imgs, method, sigma):
    # start from pixel-wise avg -- backgrounds cancel, watermark survives
    avg = np.mean(imgs, axis=0)
    if method == 'highpass':
        # avg - gauss_blur -- removes low-freq background ghost
        lp = np.zeros_like(avg)
        for c in range(avg.shape[2]):
            lp[:, :, c] = gaussian_filter(avg[:, :, c], sigma=sigma)
        return (avg - lp).astype(np.float32)
    elif method == 'mean_subtract':
        return (avg - avg.mean(axis=(0, 1), keepdims=True)).astype(np.float32)
    elif method == 'direct':
        # WMCopier approach: just (avg - 128)
        return (avg - 128.0).astype(np.float32)
    elif method == 'median_subtract':
        lp = np.zeros_like(avg)
        for c in range(avg.shape[2]):
            lp[:, :, c] = median_filter(avg[:, :, c], size=15)
        return (avg - lp).astype(np.float32)
    raise ValueError(method)


def texture_mask(img, sigma=5, gamma=0.5, floor=0.3):
    # LPIPS cares way more about flat regions than textured ones, so push watermark
    # harder into textured areas where it's invisible
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

    per_wm = {}
    all_lpips = []

    for wm_id in range(1, 9):
        cfg = PER_WM_CONFIG[wm_id]
        print(f"\n=== WM_{wm_id} | alpha={cfg['alpha']} | {cfg['method']} | sigma={cfg['sigma']} | adaptive={cfg['adaptive']} ===")

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

        sig = extract(imgs, cfg['method'], cfg['sigma'])
        print(f"  signal std={np.std(sig):.3f} max={np.max(np.abs(sig)):.2f}")

        start, end = WM_MAP[wm_id]
        alpha = cfg['alpha']
        lpips_vals = []

        for i in range(start - 1, end):
            clean = load_img(os.path.join(CLEAN_DIR, f"{i + 1}.png"))
            ch, cw = clean.shape[:2]
            r = resize_sig(sig, ch, cw)

            if cfg['adaptive']:
                m = texture_mask(clean)
                forged = clean + alpha * r * m
            else:
                forged = clean + alpha * r

            forged = np.clip(forged, 0, 255)
            save_img(forged, os.path.join(OUTPUT_DIR, f"{i + 1}.png"))

            ct = torch.from_numpy(clean / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
            ft = torch.from_numpy(forged / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
            lpips_vals.append(loss_fn(ct, ft).item())

        avg_lp = float(np.mean(lpips_vals))
        sq = float(np.exp(-8 * avg_lp))
        print(f"  avg LPIPS={avg_lp:.4f}  est S_qlt={sq:.4f}  (if S_det=1)")

        per_wm[wm_id] = dict(alpha=alpha, method=cfg['method'], sigma=cfg['sigma'],
                             adaptive=cfg['adaptive'], signal_std=float(np.std(sig)),
                             avg_lpips=avg_lp, est_qlt=sq)
        all_lpips.append(avg_lp)

    overall_lp = float(np.mean(all_lpips))
    overall_sq = float(np.exp(-8 * overall_lp))
    print(f"\n{'='*60}")
    print(f"overall LPIPS={overall_lp:.4f}  overall S_qlt={overall_sq:.4f}  (if all S_det=1)")
    print(f"{'='*60}")
    for wm_id, s in per_wm.items():
        print(f"  WM_{wm_id}: alpha={s['alpha']:.2f}  {s['method']:<16}  std={s['signal_std']:.2f}  LPIPS={s['avg_lpips']:.4f}  qlt={s['est_qlt']:.4f}")

    json.dump(per_wm, open('wm_stats.json', 'w'), indent=2)
    print(f"\nforged -> {OUTPUT_DIR}/  stats -> wm_stats.json")


if __name__ == '__main__':
    main()
