"""Evaluate a detector's bird boxes against the multi-box hand-labelled ground truth.

Ground truth : one row per box (review_state == "boxed"), or one row with review_state == "no_bird".
Predictions  : one row per box, columns  source_id, x, y, w, h, score   (any extra columns ignored).
               For the old one-box-per-image manifest, pass --legacy and it is converted automatically.

Matching is COCO-style: per image, predictions sorted by score are greedily matched to the unmatched GT box
with the highest IoU >= --iou. GT boxes whose shorter side is < --min-side px are "ignore" boxes: missing
them is not a miss, and a prediction that lands on one is not counted as a false positive.

Run from the Notebooks folder, e.g.
  python "iNaturalist Samples/bbox_eval_multi.py" --pred "Outputs/.../ssd_boxes.csv"
  python "iNaturalist Samples/bbox_eval_multi.py" --pred "Outputs/iNaturalist Outputs/Bounding Box Outputs/inaturalist_bird_bbox_manifest.csv" --legacy
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--gt", default="iNaturalist Samples/sampled_manifest_ground_truth.csv")
ap.add_argument("--pred", required=True)
ap.add_argument("--legacy", action="store_true", help="pred is the old one-box-per-image manifest")
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--min-side", type=float, default=16, help="GT boxes smaller than this (px) are ignored")
ap.add_argument("--out", default="iNaturalist Samples/bbox_eval_matches.csv")
a = ap.parse_args()

gt = pd.read_csv(a.gt)
images = gt.groupby("source_id").agg(review_state=("review_state", "first"),
                                     relative_path=("relative_path", "first")).reset_index()
gtb = gt[gt.review_state == "boxed"].rename(columns={"bbox_x": "x", "bbox_y": "y", "bbox_width": "w", "bbox_height": "h"})
gtb = gtb[["source_id", "x", "y", "w", "h"]].reset_index(drop=True)
gtb["ignore"] = np.minimum(gtb.w, gtb.h) < a.min_side

if a.legacy:
    p = pd.read_csv(a.pred, encoding="utf-8-sig", low_memory=False)
    p = p[p.bbox_x.notna()].rename(columns={"bridge_image_id": "source_id", "bbox_x": "x", "bbox_y": "y",
                                            "bbox_width": "w", "bbox_height": "h", "bbox_confidence": "score"})
else:
    p = pd.read_csv(a.pred, encoding="utf-8-sig")
# images marked "unusable" (bird present but not labelable) are excluded from scoring entirely
skip = set(images.source_id[images.review_state == "unusable"])
p = p[p.source_id.isin(images.source_id) & ~p.source_id.isin(skip)][["source_id", "x", "y", "w", "h", "score"]].reset_index(drop=True)


def iou_matrix(P, G):
    px1, py1, px2, py2 = P[:, 0], P[:, 1], P[:, 0] + P[:, 2], P[:, 1] + P[:, 3]
    gx1, gy1, gx2, gy2 = G[:, 0], G[:, 1], G[:, 0] + G[:, 2], G[:, 1] + G[:, 3]
    iw = (np.minimum(px2[:, None], gx2) - np.maximum(px1[:, None], gx1)).clip(0)
    ih = (np.minimum(py2[:, None], gy2) - np.maximum(py1[:, None], gy1)).clip(0)
    inter = iw * ih
    return inter / (P[:, 2:3] * P[:, 3:4] + (G[:, 2] * G[:, 3])[None] - inter)


rows = []  # one row per prediction: score, outcome (tp / fp / ignored), iou, matched gt index
gt_hit = np.zeros(len(gtb), bool)
for sid, pp in p.groupby("source_id"):
    pp = pp.sort_values("score", ascending=False)
    gg = gtb[gtb.source_id == sid]
    if len(gg) == 0:
        rows += [dict(source_id=sid, score=s, outcome="fp", iou=0.0, gt_idx=-1) for s in pp.score]
        continue
    M = iou_matrix(pp[["x", "y", "w", "h"]].values, gg[["x", "y", "w", "h"]].values)
    used = np.zeros(len(gg), bool)
    for k, s in enumerate(pp.score.values):
        cand = np.where(~used, M[k], -1)
        j = int(cand.argmax())
        if cand[j] >= a.iou:
            used[j] = True
            gi = gg.index[j]
            gt_hit[gi] = True
            rows.append(dict(source_id=sid, score=s, outcome="ignored" if gtb.ignore[gi] else "tp", iou=M[k, j], gt_idx=gi))
        else:
            rows.append(dict(source_id=sid, score=s, outcome="fp", iou=float(M[k].max()), gt_idx=-1))
r = pd.DataFrame(rows)

n_gt = int((~gtb.ignore).sum())
print(f"GT: {images.source_id.nunique()} images, {len(gtb)} boxes ({n_gt} scored, {int(gtb.ignore.sum())} ignored <{a.min_side:g}px), "
      f"{(images.review_state == 'no_bird').sum()} no_bird, {(images.review_state == 'unusable').sum()} unusable (skipped)")
print(f"Predictions: {len(p)} boxes on {p.source_id.nunique()} images\n")

# precision / recall sweep and AP
s = r[r.outcome != "ignored"].sort_values("score", ascending=False)
tp = (s.outcome == "tp").cumsum().values
fp = (s.outcome == "fp").cumsum().values
rec, prec = tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1)
ap50 = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * np.maximum.accumulate(prec[::-1])[::-1])) if len(s) else 0.0
print(f"AP@{a.iou:g} = {ap50:.3f}   (max recall {rec[-1] if len(rec) else 0:.1%})")
print(f"{'thr':>5} {'prec':>6} {'recall':>7} {'meanIoU(tp)':>12} {'FP on no_bird imgs':>19}")
nob = set(images.source_id[images.review_state == "no_bird"])
for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    k = s[s.score >= t]
    ntp = (k.outcome == "tp").sum()
    print(f"{t:5.2f} {ntp / max(len(k), 1):6.1%} {ntp / max(n_gt, 1):7.1%} {k[k.outcome == 'tp'].iou.mean():12.3f} "
          f"{(k.outcome.eq('fp') & k.source_id.isin(nob)).sum():19d}")

# recall split by image type
gtb["hit"] = gt_hit
nbox = gtb.groupby("source_id").size()
gtb["image_type"] = np.where(gtb.source_id.map(nbox) > 1, "multi-bird", "single-bird")
print("\nRecall (any score) by image type, scored boxes only:")
print(gtb[~gtb.ignore].groupby("image_type").hit.agg(["size", "mean"]).rename(columns={"size": "gt_boxes", "mean": "recall"}).to_string())

r.merge(images, on="source_id").to_csv(a.out, index=False)
print("\nwrote", a.out)
