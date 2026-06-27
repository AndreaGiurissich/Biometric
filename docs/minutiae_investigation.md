# Minutiae model investigation — outcome and decision

**Status:** closed — **minutiae paradigm dropped for SOCOFing.**
**Date:** 2026-06-27
**Scope:** the minutiae slot of the frozen 4-model set (NBIS / Gabor / SIFT / DINOv2).

---

## TL;DR

SOCOFing's native image size (**~96×103 px**) is **below the working range of every
minutiae extractor we tested** — both the classical NIST NBIS (MINDTCT) and the
pretrained deep MinutiaeNet. At native resolution both produce **degenerate
templates** (≤6 minutiae, zero or meaningless match scores). MinutiaeNet only
recovers usable minutiae after **~4× upsampling**, and even then the yield is
**modest, scale-sensitive, and validated on a single fingerprint triple**.

**Decision:** exclude the minutiae paradigm from this study and proceed with the
remaining **three models — Gabor, SIFT, DINOv2**. The negative result is itself a
reportable finding (it characterizes a real limitation of the dataset for
minutiae-based recognition).

---

## Background

The protocol originally specified four frozen, inference-only models, including a
**minutiae** matcher. SourceAFIS was the intended minutiae model but has **no
Python package** (official project is Java/.NET), so it was substituted with
**NIST NBIS** (MINDTCT extraction + BOZORTH3 matching), built from source on
Kaggle. The spike step (`scripts/spike_nbis.py`) existed precisely to decide
**go/no-go** on minutiae before investing in a full wrapper.

## Method

A "spike" extracts templates for one fingerprint **triple** and checks the basic
sanity inequality *genuine > impostor*:

- **probe:** `100__M_Left_index_finger_CR.BMP` (identity `100_Left_index`, level Easy)
- **genuine (true mate):** `100__M_Left_index_finger.BMP`
- **impostor:** `100__M_Left_little_finger.BMP`

The headline number is the **minutiae count** per image: a usable fingerprint
yields dozens; single digits means the extractor has effectively failed.

## Results

### 1. NBIS (MINDTCT + BOZORTH3) — `scripts/spike_nbis.py`

| metric | value |
|---|---|
| working input format | WSQ (BMP → WSQ via Pillow plugin) |
| minutiae (probe/gen/imp) | **6 / 5 / 4** |
| genuine score (BOZORTH3) | **0** |
| impostor score | **0** |
| sanity (gen > imp) | **False** |

Plumbing correct, result degenerate. MINDTCT blocks the image into ~8 px cells
for ridge-flow/frequency estimation; at ~96 px there are too few valid blocks, so
almost no minutiae survive. With 4–6 minutiae BOZORTH3 cannot reach its minimum
matching-pair count → score 0.

### 2. MinutiaeNet (deep, pretrained) — `scripts/spike_minutiaenet.py`

Run via the **`fingerflow`** package (MinutiaeNet = CoarseNet + FineNet +
ClassifyNet + CoreNet for extraction; VerifyNet-10 for matching), frozen
inference. Upscaling sweep (cubic, applied identically to all three images):

| upscale | image size | minutiae (probe/gen/imp) | usable (≥10)? |
|---|---|---|---|
| **x1 (native)** | 96×103 | **0 / 0 / 0** | no |
| x2 | 192×206 | 11 / 9 / 5 | borderline |
| **x4** | 384×412 | **21 / 14 / 16** | **yes** |
| x6 | 576×618 | 4 / 8 / 5 | no |
| x8 | 768×824 | 12 / 19 / 16 | yes |

Matcher (VerifyNet, on the best factor x4): **genuine 0.758 > impostor 0.722**,
sanity **True**.

**Reading:**
- At **native resolution the deep extractor is as dead as NBIS** (0 minutiae) —
  confirming the bottleneck is **input resolution**, not any one algorithm.
- **~4× upsampling rescues** minutiae into double digits and yields the correct
  genuine > impostor ordering.
- But the count vs. scale curve is **non-monotonic** (x4 strong, x6 collapses to
  4, x8 recovers) → the extractor is sensitive to interpolation artifacts; the
  "right" factor is not clean.

## Honest caveats on the positive (upscaled) result

1. **Single triple.** The 0.758 vs 0.722 margin confirms *direction*, not
   performance; no statistical claim is possible from one pair.
2. **Scale-sensitive / noisy** minutiae yield (non-monotonic across factors).
3. **Upsampling is a model-input preprocessing step** that would have to be added
   to the (frozen) pipeline and applied identically to gallery and probes — a
   change requiring sign-off.

These three together are why the modest, fragile upside did not justify keeping
the minutiae model.

## Decision and protocol impact

- **Minutiae paradigm dropped.** The study proceeds with **3 models: Gabor
  (texture), SIFT + RANSAC (keypoints), DINOv2 (deep embedding).**
- Tier A (fair cross-model) becomes a **3-model** comparison; Tier B is
  unchanged (Gabor + DINOv2 on the full probe set).
- The minutiae result is retained as a **dataset-limitation finding** for the
  report: *SOCOFing's ~96×103 px images fall below the operating resolution of
  both classical (NBIS/MINDTCT) and deep (MinutiaeNet) minutiae extractors;
  usable minutiae are only recoverable via ~4× upsampling, with unstable yield.*

## For the report (adaptable paragraph)

> We evaluated a minutiae-based matcher as the classical baseline. Because no
> Python SourceAFIS exists, we used NIST NBIS (MINDTCT + BOZORTH3); on SOCOFing's
> native ~96×103 px images MINDTCT extracted only 4–6 minutiae per print and
> BOZORTH3 returned zero scores. A pretrained deep extractor (MinutiaeNet, via
> fingerflow) confirmed the limitation: it extracted 0 minutiae at native
> resolution and required ~4× upsampling to reach ~15–20 minutiae, with a
> non-monotonic, interpolation-sensitive yield. We therefore excluded
> minutiae-based recognition from the comparison and report results for three
> models — Gabor texture, SIFT+RANSAC, and DINOv2 — noting that SOCOFing's
> resolution is the binding constraint for minutiae methods.

## Reproduction

```bash
# NBIS spike
python scripts/spike_nbis.py --level Easy

# MinutiaeNet spike + upscaling sweep (weights attached as a Kaggle dataset)
python scripts/spike_minutiaenet.py \
    --models-dir /kaggle/input/minutiaenet-weights --precision 10 \
    --upscale 1,2,4,6,8
```

### Environment note (reproducibility)

`fingerflow` vendored MinutiaeNet code (circa 2018) does not run as-is on a 2026
Kaggle image. `scripts/spike_minutiaenet.py` applies compatibility shims at
startup, with **no library downgrades**:
- **Keras 2** routing via `tf-keras` + `TF_USE_LEGACY_KERAS=1` (the code passes
  `weights=` to `Conv2D`, removed in Keras 3);
- restored NumPy aliases removed in 1.24 (`np.bool/int/float`) and 2.0
  (`np.product → np.prod`, `np.lib.pad → np.pad`, …);
- a wrapper translating scikit-image's removed `gaussian(multichannel=)` kwarg to
  `channel_axis`.

Raw spike outputs: `logs/nbis_spike.json`, `logs/minutiaenet_spike.json`
(ephemeral on Kaggle — persist via `scripts/save_results.py`).
