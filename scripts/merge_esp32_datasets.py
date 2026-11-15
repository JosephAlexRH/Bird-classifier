#!/usr/bin/env python3
"""
Merge the legacy ESP32 annotations (160x120, 25%-context crops) into the new
combined_label_bbox_editor schema (640x480, tight boxes, no context).

Canonical rule for the merged table:
  * every row stores the TIGHT bird box, in native pixel coords of its own image
  * context is a crop/training-time choice, not an annotation -> legacy crop_*
    columns are verified, then dropped; context_* just mirrors the box
  * derived columns (area, context, split_group) are recomputed identically
    for both sources so nothing depends on which tool wrote the row

Usage:
  python merge_esp32_datasets.py --old old.csv --new new.csv \
      --old-image-dir "C:\\path\\to\\legacy_captures" --out combined.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_SOURCE = "esp32_legacy_160x120"
SESSION_GAP_MIN = 10  # captures closer together than this share a split_group
TS_RE = r"capture_(\d{8}T\d{6})_(\d{6})Z"
BOX = ["xmin", "ymin", "xmax", "ymax"]


def read_table(path):
    # sniffs comma vs tab automatically
    return pd.read_csv(path, sep=None, engine="python")


def check_legacy_boxes_are_tight(old):
    """Rebuild the legacy crops from xmin..ymax + context fraction. If they match
    the stored crop_* columns, xmin..ymax are the tight boxes and can be used as-is."""
    b = old[old["crop_xmin"].notna()].copy()
    for c in BOX + ["crop_" + c for c in BOX] + ["source_width", "source_height", "crop_context_fraction"]:
        b[c] = b[c].astype(float)
    f = b["crop_context_fraction"]
    w, h = b.xmax - b.xmin, b.ymax - b.ymin
    rebuilt = pd.DataFrame({
        "crop_xmin": np.floor((b.xmin - f * w).clip(lower=0)),
        "crop_ymin": np.floor((b.ymin - f * h).clip(lower=0)),
        "crop_xmax": np.ceil(np.minimum(b.xmax + f * w, b.source_width)),
        "crop_ymax": np.ceil(np.minimum(b.ymax + f * h, b.source_height)),
    })
    diff = (rebuilt - b[rebuilt.columns]).abs().max(axis=1)
    bad = diff > 1
    print(f"[check] legacy crops reproduced from boxes: {(~bad).sum()}/{len(b)} within 1px")
    if bad.any():
        print("[check] WARNING: these legacy boxes may already include context - inspect them:")
        print(b.loc[bad, ["record_id"] + BOX].head(20).to_string())


def convert_legacy(old, image_dir):
    o = old.copy()
    boxed = o["review_state"].eq("boxed") & o["xmin"].notna()
    box_id = o["box_id"].fillna(0).astype(int)
    n_boxes = o[boxed].groupby("image_id").size()

    out = pd.DataFrame({
        "annotation_id": o["image_id"] + ":" + box_id.astype(str),
        "sample_id": o["image_id"],
        "source_dataset": LEGACY_SOURCE,
        "source_image_id": o["image_id"],
        "image_path": o["scene_path"].map(lambda p: str(Path(image_dir) / p)),
        "relative_path": o["scene_path"],
        "width_px": o["source_width"],
        "height_px": o["source_height"],
        "review_state": o["review_state"],
        # don't claim human_verified unless you re-review them in the new editor
        "annotation_trust": "legacy_reviewed",
        "annotation_origin": "esp32_capture_manifest",
        "eligible_for_training": o["review_state"].isin(["boxed", "no_bird"]),
        "detector_class_id": o["class_id"].where(boxed),
        "detector_class_name": pd.Series("bird", index=o.index).where(boxed),
        "box_id": o["box_id"].where(boxed),
        "is_Bird": boxed,
        "mixed_species": False,
        "bird_count": o["image_id"].map(n_boxes).fillna(0).astype(int),
    })
    for c in BOX:
        out[c] = o[c].where(boxed)
    return out


def harmonize(df):
    for c in BOX + ["width_px", "height_px"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    boxed = df["review_state"].eq("boxed") & df["xmin"].notna()
    w, h = df.xmax - df.xmin, df.ymax - df.ymin

    df["projected_area_px2"] = (w * h).where(boxed)
    # resolution-independent size: use THIS for bird_size buckets / stratification
    df["area_frac"] = (w * h / (df.width_px * df.height_px)).where(boxed)
    # normalized YOLO-style coords so 160x120 and 640x480 live in one space
    df["cx_norm"] = ((df.xmin + df.xmax) / 2 / df.width_px).where(boxed)
    df["cy_norm"] = ((df.ymin + df.ymax) / 2 / df.height_px).where(boxed)
    df["w_norm"] = (w / df.width_px).where(boxed)
    df["h_norm"] = (h / df.height_px).where(boxed)

    # tight boxes everywhere; context is applied later when cropping
    df["context_percent"] = np.where(boxed, 0, np.nan)
    for c in BOX:
        df[f"context_{c}"] = df[c].where(boxed)

    df["capture_config"] = (df.width_px.astype("Int64").astype(str) + "x"
                            + df.height_px.astype("Int64").astype(str))
    elig = df["eligible_for_training"]
    default = df["review_state"].isin(["boxed", "no_bird"])
    elig = elig.map(lambda v: v if isinstance(v, bool) else
                    {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}.get(str(v).strip().lower()))
    df["eligible_for_training"] = elig.where(elig.notna(), default).astype(bool)
    return df


def assign_sessions(df, gap_min=SESSION_GAP_MIN):
    """Burst captures 1s apart are near-duplicates. Group them into sessions so
    a whole burst lands in the same train/val/test split (no leakage)."""
    s = df[["sample_id"]].drop_duplicates().copy()
    ts = s["sample_id"].str.extract(TS_RE)
    s["t"] = pd.to_datetime(ts[0] + ts[1], format="%Y%m%dT%H%M%S%f", errors="coerce")
    s = s.sort_values("t")
    new_session = s["t"].diff() > pd.Timedelta(minutes=gap_min)
    s["session"] = "session_" + new_session.cumsum().astype(str).str.zfill(4)
    session = df["sample_id"].map(s.set_index("sample_id")["session"])
    existing = df.get("split_group")
    if existing is not None:
        keep = existing.notna() & existing.astype(str).str.strip().ne("")
        df["split_group"] = existing.where(keep, session)
    else:
        df["split_group"] = session
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-image-dir", required=True, help="folder holding the legacy 160x120 captures")
    ap.add_argument("--out", default="combined_annotations.csv")
    ap.add_argument("--session-gap-min", type=float, default=SESSION_GAP_MIN)
    a = ap.parse_args()

    old, new = read_table(a.old), read_table(a.new)
    check_legacy_boxes_are_tight(old)

    merged = pd.concat([new, convert_legacy(old, a.old_image_dir)], ignore_index=True, sort=False)
    merged = harmonize(merged)
    merged = assign_sessions(merged, a.session_gap_min)

    dup = merged["annotation_id"].duplicated().sum()
    assert dup == 0, f"{dup} duplicate annotation_ids"
    has = merged.xmin.notna()
    eps = 0.5
    ok = ((merged.xmin >= -eps) & (merged.ymin >= -eps)
          & (merged.xmax <= merged.width_px + eps) & (merged.ymax <= merged.height_px + eps)
          & (merged.xmax > merged.xmin) & (merged.ymax > merged.ymin))
    print(f"[check] invalid boxes: {(has & ~ok).sum()}")

    cols = list(new.columns) + [c for c in merged.columns if c not in new.columns]
    merged[cols].to_csv(a.out, index=False)

    print(f"[done] wrote {len(merged)} rows -> {a.out}")
    print(merged.groupby(["source_dataset", "review_state"]).size().to_string())
    print(f"[info] sessions: {merged['split_group'].nunique()}")


if __name__ == "__main__":
    main()
