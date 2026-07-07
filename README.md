# Assignment 4 — Watermark Forgery Attack

Reproducing our leaderboard submission.

## Setup

```bash
pip install numpy scipy pillow torch torchvision lpips
```

Dataset layout (place under the parent directory of the repo):

```
../Dataset/
├── watermarked_sources/
│   ├── WM_1/   (25 PNGs, all carrying WM_1)
│   ├── WM_2/
│   └── ... WM_8/
└── clean_targets/
    └── 1.png ... 200.png
```

## Files

- `forge_watermark.py` — main script. Extracts each watermark from its 25 sources, injects into the 25 corresponding clean targets, writes PNGs to `../submission_temp/`.
- `alpha_sweep.py` — helper. Sweeps alpha × extraction methods per WM and reports local LPIPS. Use this to pick a config without burning leaderboard submissions.

## Running

```bash
# 1. (optional) sweep to find good alpha per WM
python alpha_sweep.py
# -> writes alpha_sweep_results.json, prints a table per WM

# 2. edit PER_WM_CONFIG in forge_watermark.py based on sweep output

# 3. forge the submission
python forge_watermark.py
# -> writes ../submission_temp/1.png ... 200.png
# -> writes wm_stats.json with per-WM LPIPS

# 4. zip submission_temp and upload
```

## Method

Two-pass averaging attack:

1. **Extract** — average the 25 source images for each WM (backgrounds cancel, watermark survives), then apply one of: `highpass` (Gaussian blur subtract), `direct` (avg − 128, WMCopier-style), `mean_subtract`, or `median_subtract`.
2. **Inject** — `forged = clean + alpha * signal * texture_mask`, clip to [0, 255], save as PNG.

The texture mask scales the watermark strength with local image texture (LPIPS is far more sensitive in flat regions), so we can push alpha much higher at the same LPIPS cost.

## Tuning

Per-WM config lives in `PER_WM_CONFIG` at the top of `forge_watermark.py`. Loop:

1. Run `alpha_sweep.py`, find (method, alpha) landing LPIPS in [0.02, 0.05] per WM.
2. Update `PER_WM_CONFIG`, run `forge_watermark.py`.
3. Submit to leaderboard.
4. If score < local S_qlt → S_det is weak → raise alpha or switch method.
5. If score = local S_qlt → S_det saturated → lower alpha to gain S_qlt.

## Results

| Method | Score |
|---|---|
| Method 1 (Gaussian highpass, alpha=0.5, uniform) | 0.316 |
| Method 2 (multi-filter per-WM: Gaussian / Median / DWT) | 0.349 |

## References

- Yang et al., "Can Simple Averaging Defeat Modern Watermarks?", NeurIPS 2024.
- Dong et al., "WMCopier: Forging Invisible Image Watermarks on Arbitrary Images", NeurIPS 2025.
- Kutter, Voloshynovskiy, Herrigel, "The Watermark Copy Attack", SPIE 2000.
