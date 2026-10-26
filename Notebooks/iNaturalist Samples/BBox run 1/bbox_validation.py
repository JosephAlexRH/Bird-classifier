"""Score machine bboxes (inaturalist_bird_bbox_manifest.csv) against the 200-image hand-labelled ground truth.
Run from the Notebooks folder:  python "iNaturalist Samples/bbox_validation.py"
"""
import numpy as np, pandas as pd
GT = "iNaturalist Samples/sampled_manifest_ground_truth.csv"
MACHINE = "Outputs/iNaturalist Outputs/Bounding Box Outputs/inaturalist_bird_bbox_manifest.csv"
OUT = "iNaturalist Samples/bbox_validation_scored.csv"

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = (np.minimum(ax+aw, bx+bw) - np.maximum(ax, bx)).clip(0)
    iy = (np.minimum(ay+ah, by+bh) - np.maximum(ay, by)).clip(0)
    inter = ix*iy
    return inter/(aw*ah + bw*bh - inter), inter

gt = pd.read_csv(GT)
m = pd.read_csv(MACHINE, encoding="utf-8-sig", low_memory=False)
j = gt.merge(m, left_on="source_id", right_on="bridge_image_id", how="left", suffixes=("_gt", "_m"))
assert j.bridge_image_id.notna().all(), "some GT rows missing from machine manifest"
assert (j.source_width == j.bbox_image_width_px).all() and (j.source_height == j.bbox_image_height_px).all()

g = [j[f"bbox_{c}_gt"] for c in ("x", "y", "width", "height")]
mb = [j[f"bbox_{c}_m"] for c in ("x", "y", "width", "height")]
j["iou"], inter = iou(g, mb)
j["area_ratio"] = mb[2]*mb[3]/(g[2]*g[3])
j["gt_covered_frac"] = inter/(g[2]*g[3])
j["machine_on_gt_frac"] = inter/(mb[2]*mb[3])
j["gt_area_frac"] = g[2]*g[3]/(j.source_width*j.source_height)

print(pd.crosstab(j.review_state, j.bbox_detection_status, margins=True))
d = j[(j.review_state == "boxed") & (j.bbox_detection_status == "detected")]
print(f"\nDetected & boxed: n={len(d)} meanIoU={d.iou.mean():.3f} median={d.iou.median():.3f} "
      + " ".join(f"IoU>={t}:{(d.iou>=t).mean():.1%}" for t in (.5, .75, .9)))

# Box-expansion check: scale machine boxes about their centre, clipped to image
for s in (1.0, 1.05, 1.10, 1.15, 1.20):
    cx, cy = mb[0]+mb[2]/2, mb[1]+mb[3]/2
    x0 = (cx - mb[2]*s/2).clip(lower=0); y0 = (cy - mb[3]*s/2).clip(lower=0)
    x1 = np.minimum(cx + mb[2]*s/2, j.source_width); y1 = np.minimum(cy + mb[3]*s/2, j.source_height)
    v, _ = iou(g, (x0, y0, x1-x0, y1-y0)); v = v[d.index]
    print(f"scale {s:.2f}: meanIoU={v.mean():.3f} median={v.median():.3f} IoU>=.75:{(v>=.75).mean():.1%}")

cols = ["source_id", "relative_path_gt", "source_width", "source_height", "review_state", "bbox_detection_status",
        "bbox_confidence", "bird_detection_count", "bbox_x_gt", "bbox_y_gt", "bbox_width_gt", "bbox_height_gt",
        "bbox_x_m", "bbox_y_m", "bbox_width_m", "bbox_height_m", "iou", "area_ratio", "gt_covered_frac",
        "machine_on_gt_frac", "gt_area_frac"]
j[cols].sort_values("iou", na_position="first").to_csv(OUT, index=False)
print("wrote", OUT)
