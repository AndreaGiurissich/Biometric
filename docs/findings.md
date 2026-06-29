# SOCOFing robustness study — Tier-A findings

**Status:** Tier-A complete (all 3 models). Tier-B (full-probe best estimate) pending.
**Date:** 2026-06-28
**Protocol:** closed-set 1:N identification, 6000-identity gallery, 2000-probe
stratified manifest (SIFT on a nested 500). Conditions: `baseline` (raw) vs
`preprocessed` (contrast-mask on gallery + probes). Minutiae model dropped (see
`minutiae_investigation.md`). Numbers below are from `results/summary.csv`;
figures in `results/figures/` (regenerate with `scripts/synthesize.py`).

---

## TL;DR — three findings

1. **The ranking is inverted: SIFT ≫ Gabor ≫ DINOv2**, at every level, and the
   gap widens with difficulty.
2. **Only SIFT is usable at a strict operating point** (FAR@FRR=1%); the texture
   and deep models are not.
3. **The contrast-mask preprocessing helps are model-dependent, not universal**:
   it clearly helps Gabor (more as difficulty rises), is marginal for DINOv2, and
   is neutral-to-harmful for SIFT.

## Headline numbers — Rank-1 (baseline → preprocessed)

| level | SIFT | Gabor | DINOv2 |
|-------|------|-------|--------|
| Easy   | 0.994 → 0.994 | 0.961 → 0.967 | 0.857 → 0.864 |
| Medium | 0.986 → 0.990 | 0.864 → 0.895 | 0.459 → 0.463 |
| Hard   | 0.974 → 0.960 | 0.812 → 0.846 | 0.283 → 0.290 |
| **Easy→Hard drop (base)** | **−2.0 pp** | −14.9 pp | **−57.4 pp** |

EER (baseline): SIFT 0.002 / 0.008 / 0.013 · Gabor 0.055 / 0.114 / 0.131 ·
DINOv2 0.071 / 0.169 / 0.198 (Easy / Medium / Hard).

## Finding 1 — inverted ranking, and robustness to difficulty

SIFT (keypoints + RANSAC homography, the oldest method) is the strongest and
barely degrades (−2 pp Easy→Hard). DINOv2 (frozen ViT-B/14 CLS embedding, the
newest) is the weakest and **collapses** (−57 pp). Gabor sits in between.

Mechanism: SIFT's RANSAC verification matches the *surviving* keypoints and
ignores the rest, so it tolerates local damage; DINOv2's single global CLS
embedding shifts whenever the image changes; the model is also out-of-distribution
(natural-image self-supervision) on low-resolution fingerprint texture.

## Finding 2 — only SIFT works at a strict threshold

**FAR@FRR=1% (baseline):** SIFT 0.0007 / 0.002 / 0.016 vs Gabor 0.33 / 0.66 / 0.77
vs DINOv2 0.34 / 0.66 / 0.75 (Easy / Medium / Hard). SIFT's score is a RANSAC
inlier count with enormous genuine/impostor separation (~37 vs ~0.5 inliers), so
a low-false-reject operating point still keeps impostors out. Gabor and DINOv2
have heavily overlapping score tails and are unusable at FRR=1% on hard
alterations, despite decent EER/AUC.

## Finding 3 — preprocessing is contingent on the representation

Δ Rank-1 from preprocessing (percentage points):

| model | Easy | Medium | Hard |
|-------|------|--------|------|
| Gabor  | +0.7 | +3.0 | **+3.4** |
| DINOv2 | +0.7 | +0.5 | +0.7 |
| SIFT   |  0.0 | +0.4 | **−1.4** |

The contrast-mask (sharpen ridge regions via a soft mask) **helps the texture
descriptor (Gabor) and increasingly so with difficulty**, is roughly flat for the
deep embedding, and **hurts the keypoint matcher at Hard** — the sharpening
introduces spurious keypoints and degrades geometric consistency. A preprocessing
operator borrowed from a face-detection robustness paper is therefore not a
universal good: its value depends on the downstream representation.

## Per-alteration mechanism — Rank-1 at Hard (baseline)

| alteration | SIFT | Gabor | DINOv2 |
|------------|------|-------|--------|
| CR (rotation)        | 0.958 | 0.802 | 0.234 |
| Obl (obliteration)   | 0.982 | 0.654 | 0.310 |
| Zcut (cut)           | 0.982 | 0.980 | 0.305 |

- **DINOv2 fails on all three** — the global embedding can't absorb local damage,
  and even a clean *cut* (Zcut) wrecks it (0.305) while SIFT/Gabor shrug it off.
- **Gabor's weakness is obliteration** (0.654): texture statistics degrade when
  ridges are destroyed, though it stays robust to cutting (0.980).
- **SIFT is robust to all** (≥0.958) — geometric keypoint consistency survives
  rotation, cutting, and partial obliteration.

## Statistical significance (paired)

Both conditions are evaluated on the same probes, so the preprocessing effect is
tested paired: **McNemar** on Rank-1 hits (identification) and a **paired
bootstrap** CI/p for ΔEER and ΔAUC (verification), resampling probes. Run
`scripts/significance.py` -> `results/significance.csv`. Report each ΔRank-1/ΔEER
with its p-value: the Gabor gains on Medium/Hard are expected to clear
significance while the small Easy gain (+0.65 pp Rank-1) is likely within
sampling noise -- state that explicitly rather than over-claiming.

## Caveats (for honest reporting)

- **SIFT runs on a nested 500-probe subset** (it is O(N × gallery)); its numbers
  carry slightly wider confidence than the 2000-probe Gabor/DINOv2 results.
- **DINOv2 is used exactly as the frozen spec dictates** (ViT-B/14, CLS, 224 px,
  cosine). The result reflects off-the-shelf frozen features, not a fingerprint-
  tuned or higher-resolution (518) variant. A 518 run is a sanctioned optional
  follow-up if a reviewer questions the resolution.
- Single seeded run per cell; Tier B (Gabor + DINOv2 on the full probe set) will
  tighten the best estimates but is not expected to change the ordering.

## Report-ready paragraph

> On SOCOFing we compared three frozen recognition models under increasing
> alteration severity. Contrary to the expectation that a large self-supervised
> model dominates, a classical SIFT + RANSAC matcher was by far the most accurate
> and robust (Rank-1 0.994→0.974 from Easy to Hard), a Gabor texture descriptor
> was intermediate (0.961→0.812), and frozen DINOv2 CLS embeddings were the
> weakest and collapsed on hard alterations (0.857→0.283). Only SIFT was usable at
> a strict FAR@FRR=1% operating point. A contrast-mask preprocessing pipeline
> adapted from face-detection robustness improved the texture model (increasingly
> with difficulty) but was neutral-to-detrimental for the keypoint matcher,
> showing that preprocessing benefit is contingent on the underlying
> representation rather than universal.

## Reproduction

```bash
# per model: identification + verification, both conditions, all levels
python scripts/run_model.py --model gabor  --level {Easy,Medium,Hard} --condition both
python scripts/run_model.py --model dinov2 --level {Easy,Medium,Hard} --condition both
python scripts/run_model.py --model sift   --level {Easy,Medium,Hard} --condition both --workers 4
# digest + figures
python scripts/analyze_scores.py
python scripts/synthesize.py
```
