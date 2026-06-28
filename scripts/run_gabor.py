"""Gabor end-to-end: closed-set identification + verification on SOCOFing.

Ties together dataset -> (optional) preprocessing -> Gabor descriptor -> scoring
-> metrics, for one level and one or both conditions (baseline / preprocessed).

For a condition it:
  1. extracts the 6000 gallery descriptors once (cached to <cache>/<condition>/),
  2. for each probe (Tier-A subsample, or --full): extracts its descriptor,
     scores it against the whole gallery (cosine), records the true-mate rank
     (identification) and the genuine + seeded-impostor scores (verification),
  3. appends each probe's result to scores.jsonl (resume skips done probes),
  4. aggregates Rank-1/5/10 + CMC + MRR and EER/AUC/FAR@FRR/FRR@FAR, writes
     summary.json and prints it.

Usage:
    python scripts/run_gabor.py [--level Easy] [--condition both]
                                [--full] [--limit N] [--input-root /kaggle/input]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import manifest as mf  # noqa: E402
from src.models.gabor import GaborModel  # noqa: E402
from src.preprocessing import contrast_mask  # noqa: E402


def read_image(path: str, condition: str, cfg) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cv2 could not read {path}")
    if condition == "preprocessed":
        img = contrast_mask(img, cfg)
    return img


def gallery_descriptors(gallery, ids, model, cfg, condition, cache_dir):
    """Return an (N, D) float32 matrix of gallery descriptors, cached per condition."""
    cache = Path(cache_dir) / condition / "gabor_gallery.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == [mf._id_key(i) for i in ids]:
            print(f"  gallery descriptors: cache hit ({cache})")
            return data["G"].astype(np.float32)
    print(f"  extracting {len(ids)} gallery descriptors ({condition})...")
    G = np.zeros((len(ids), model.descriptor_length), dtype=np.float32)
    t0 = time.time()
    for i, idt in enumerate(ids):
        G[i] = model.extract(read_image(gallery[idt].path, condition, cfg))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array([mf._id_key(i) for i in ids]), G=G)
    return G


def _load_done(jsonl: Path):
    """Resume support: read already-scored probes from scores.jsonl."""
    records, done = [], set()
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            done.add(rec["stem"])
    return records, done


def run_condition(level, condition, gallery, ids, probes, model, cfg, paths) -> dict:
    print(f"\n== Gabor | level={level} | condition={condition} ==")
    row_of = {idt: i for i, idt in enumerate(ids)}
    G = gallery_descriptors(gallery, ids, model, cfg, condition, paths["cache_dir"])

    exp_dir = Path(paths["results_dir"]) / "raw" / f"gabor_{level}_{condition}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    jsonl = exp_dir / "scores.jsonl"
    records, done = _load_done(jsonl)
    if done:
        print(f"  resuming: {len(done)} probes already scored")

    flush_every = int(cfg["checkpointing"].get("flush_every_probes", 50))
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh:
        for n, probe in enumerate(probes):
            if probe.stem in done:
                continue
            scores = G @ model.extract(read_image(probe.path, condition, cfg))
            true_row = row_of[probe.identity]
            is_true = np.zeros(len(ids), dtype=bool)
            is_true[true_row] = True
            rank = ev.rank_of_match(scores, is_true, higher_is_better=True)
            imp_ids = mf.sample_impostor_ids(probe.identity, ids, cfg)
            rec = {
                "stem": probe.stem,
                "identity": probe.identity_str,
                "alt": probe.alt,
                "rank": rank,
                "genuine": float(scores[true_row]),
                "impostors": [float(scores[row_of[i]]) for i in imp_ids],
            }
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if (len(records)) % flush_every == 0:
                fh.flush()
                print(f"    scored {len(records)}/{len(probes)}  ({time.time() - t0:.0f}s)")

    return summarize(records, len(ids), cfg, level, condition, exp_dir)


def summarize(records, n_gallery, cfg, level, condition, exp_dir) -> dict:
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
    cmc = idm.pop("cmc")  # array -> store separately, keep summary JSON-clean

    # Per-alteration breakdown (protocol evaluates per level AND per alteration).
    per_alt = {}
    for a in sorted({r["alt"] for r in records}):
        sub = [r for r in records if r["alt"] == a]
        sr = np.array([r["rank"] for r in sub])
        sg = np.array([r["genuine"] for r in sub], dtype=float)
        si = np.array([s for r in sub for s in r["impostors"]], dtype=float)
        per_alt[a] = {
            "n": len(sub),
            **{f"rank_{k}": float(np.mean(sr <= k)) for k in ranks_to_report},
            "eer": ev.eer(sg, si, higher_is_better=True),
            "auc": ev.roc_auc(sg, si, higher_is_better=True),
        }

    summary = {
        "model": "gabor", "level": level, "condition": condition,
        "n_probes": len(records), "n_gallery": n_gallery,
        "n_impostor_scores": int(impostor.size),
        "identification": idm, "verification": vm,
        "per_alteration": per_alt,
    }
    with (exp_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.save(exp_dir / "cmc.npy", cmc)

    print(f"  -- results ({condition}) --")
    print(f"    Rank-1/5/10 : {idm.get('rank_1'):.4f} / {idm.get('rank_5'):.4f} "
          f"/ {idm.get('rank_10'):.4f}   MRR {idm['mrr']:.4f}")
    print(f"    EER {vm['eer']:.4f}   AUC {vm['auc']:.4f}   "
          f"FAR@FRR={far_targets[0]} {vm[f'far_at_frr_{far_targets[0]}']:.4f}")
    for a, pa in per_alt.items():
        print(f"      by-alt {a:<5} n={pa['n']:<5} "
              f"Rank-1 {pa['rank_1']:.4f}  EER {pa['eer']:.4f}")
    print(f"    summary -> {exp_dir / 'summary.json'}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Gabor end-to-end identification + verification.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", default="Easy")
    ap.add_argument("--condition", default="both",
                    choices=["baseline", "preprocessed", "both"])
    ap.add_argument("--full", action="store_true",
                    help="Tier B: all probes (default: Tier-A seeded subsample)")
    ap.add_argument("--limit", type=int, default=None, help="cap probes (smoke test)")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)
    model = GaborModel(cfg)

    gallery, *_ = ds.build_gallery(paths["real_dir"])
    ids = sorted(gallery.keys())
    probes, *_ = ds.build_probes(paths["level_dirs"][args.level], gallery)
    if not probes:
        raise SystemExit(f"No probes for level {args.level}; run verify_dataset first.")
    probes = probes if args.full else mf.build_manifest(probes, cfg)
    if args.limit:
        probes = probes[:args.limit]
    print(f"gallery={len(ids)}  probes={len(probes)}  "
          f"({'full' if args.full else 'Tier-A subsample'})")

    conditions = ["baseline", "preprocessed"] if args.condition == "both" else [args.condition]
    summaries = [run_condition(args.level, c, gallery, ids, probes, model, cfg, paths)
                 for c in conditions]

    if len(summaries) == 2:
        b, p = summaries
        print("\n== baseline vs preprocessed ==")
        print(f"  Rank-1 : {b['identification']['rank_1']:.4f} -> "
              f"{p['identification']['rank_1']:.4f}")
        print(f"  EER    : {b['verification']['eer']:.4f} -> "
              f"{p['verification']['eer']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
