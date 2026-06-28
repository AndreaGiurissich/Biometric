"""Model-agnostic identification + verification runner.

Any model exposing the unified interface plugs in here:
    .extract(image_gray_u8) -> 1D float32 descriptor
    .descriptor_length      -> int
Scoring is cosine (descriptors are compared by dot product after L2 norm).

The expensive gallery descriptors are cached per (model, condition). Preprocessing
(baseline vs contrast-mask) is applied to the grayscale image UPSTREAM, before the
model sees it, so every model shares the same condition definition.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src import dataset as ds  # noqa: F401  (re-exported convenience)
from src import evaluation as ev
from src import manifest as mf
from src.preprocessing import contrast_mask


def build_model(name: str, cfg: Dict):
    if name == "gabor":
        from src.models.gabor import GaborModel
        return GaborModel(cfg)
    if name == "dinov2":
        from src.models.dinov2 import Dinov2Model
        return Dinov2Model(cfg)
    raise SystemExit(f"unknown model: {name!r}")


def read_image(path: str, condition: str, cfg) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cv2 could not read {path}")
    if condition == "preprocessed":
        img = contrast_mask(img, cfg)
    return img


def gallery_descriptors(name, gallery, ids, model, cfg, condition, cache_dir):
    """(N, D) float32 gallery-descriptor matrix, cached per (model, condition)."""
    cache = Path(cache_dir) / condition / f"{name}_gallery.npz"
    key = [mf._id_key(i) for i in ids]
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == key:
            print(f"  gallery descriptors: cache hit ({cache})")
            return data["G"].astype(np.float32)
    print(f"  extracting {len(ids)} gallery descriptors ({name}/{condition})...")
    G = np.zeros((len(ids), model.descriptor_length), dtype=np.float32)
    t0 = time.time()
    for i, idt in enumerate(ids):
        G[i] = model.extract(read_image(gallery[idt].path, condition, cfg))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(key), G=G)
    return G


def _load_done(jsonl: Path):
    records, done = [], set()
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                done.add(rec["stem"])
    return records, done


def run_condition(name, level, condition, gallery, ids, probes, model, cfg, paths):
    print(f"\n== {name} | level={level} | condition={condition} ==")
    row_of = {idt: i for i, idt in enumerate(ids)}
    G = gallery_descriptors(name, gallery, ids, model, cfg, condition, paths["cache_dir"])

    exp_dir = Path(paths["results_dir"]) / "raw" / f"{name}_{level}_{condition}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    jsonl = exp_dir / "scores.jsonl"
    records, done = _load_done(jsonl)
    if done:
        print(f"  resuming: {len(done)} probes already scored")

    flush_every = int(cfg["checkpointing"].get("flush_every_probes", 50))
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh:
        for probe in probes:
            if probe.stem in done:
                continue
            scores = G @ model.extract(read_image(probe.path, condition, cfg))
            true_row = row_of[probe.identity]
            is_true = np.zeros(len(ids), dtype=bool)
            is_true[true_row] = True
            rank = ev.rank_of_match(scores, is_true, higher_is_better=True)
            imp_ids = mf.sample_impostor_ids(probe.identity, ids, cfg)
            rec = {
                "stem": probe.stem, "identity": probe.identity_str, "alt": probe.alt,
                "rank": rank, "genuine": float(scores[true_row]),
                "impostors": [float(scores[row_of[i]]) for i in imp_ids],
            }
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if len(records) % flush_every == 0:
                fh.flush()
                print(f"    scored {len(records)}/{len(probes)}  ({time.time() - t0:.0f}s)")

    return summarize(name, records, len(ids), cfg, level, condition, exp_dir)


def summarize(name, records, n_gallery, cfg, level, condition, exp_dir) -> dict:
    ranks = [r["rank"] for r in records]
    genuine = np.array([r["genuine"] for r in records], dtype=float)
    impostor = np.array([s for r in records for s in r["impostors"]], dtype=float)

    ec = cfg["evaluation"]
    ranks_to_report = ec.get("ranks_to_report", [1, 5, 10])
    idm = ev.identification_metrics(ranks, n_gallery, ranks_to_report)
    far_targets = ec.get("far_at_frr", [0.01])
    vm = ev.verification_metrics(genuine, impostor, higher_is_better=True,
                                 far_at_frr_targets=far_targets,
                                 frr_at_far_targets=far_targets)
    cmc = idm.pop("cmc")

    per_alt = {}
    for a in sorted({r["alt"] for r in records}):
        sub = [r for r in records if r["alt"] == a]
        sr = np.array([r["rank"] for r in sub])
        sg = np.array([r["genuine"] for r in sub], dtype=float)
        si = np.array([s for r in sub for s in r["impostors"]], dtype=float)
        per_alt[a] = {"n": len(sub),
                      **{f"rank_{k}": float(np.mean(sr <= k)) for k in ranks_to_report},
                      "eer": ev.eer(sg, si, higher_is_better=True),
                      "auc": ev.roc_auc(sg, si, higher_is_better=True)}

    summary = {
        "model": name, "level": level, "condition": condition,
        "n_probes": len(records), "n_gallery": n_gallery,
        "n_impostor_scores": int(impostor.size),
        "identification": idm, "verification": vm, "per_alteration": per_alt,
    }
    with (exp_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.save(exp_dir / "cmc.npy", cmc)

    print(f"  -- results ({name}/{condition}) --")
    print(f"    Rank-1/5/10 : {idm.get('rank_1'):.4f} / {idm.get('rank_5'):.4f} "
          f"/ {idm.get('rank_10'):.4f}   MRR {idm['mrr']:.4f}")
    print(f"    EER {vm['eer']:.4f}   AUC {vm['auc']:.4f}   "
          f"FAR@FRR={far_targets[0]} {vm[f'far_at_frr_{far_targets[0]}']:.4f}")
    for a, pa in per_alt.items():
        print(f"      by-alt {a:<5} n={pa['n']:<5} "
              f"Rank-1 {pa['rank_1']:.4f}  EER {pa['eer']:.4f}")
    print(f"    summary -> {exp_dir / 'summary.json'}")
    return summary


def run(name, level, conditions, gallery, ids, probes, cfg, paths) -> List[dict]:
    model = build_model(name, cfg)
    return [run_condition(name, level, c, gallery, ids, probes, model, cfg, paths)
            for c in conditions]
