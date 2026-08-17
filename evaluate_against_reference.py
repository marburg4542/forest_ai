"""Compare outputs/trees.csv against a reference tree list.

    python evaluate_against_reference.py field_plot.csv
    python evaluate_against_reference.py fsct_output.csv --quality good --max-dist 1.5

The reference CSV needs x and y columns in the same coordinate system as the
point cloud, plus whichever attributes you want compared (dbh_cm, height_m).
Use --map to rename columns from another tool, e.g.
    --map DBH=dbh_cm --map Height=height_m --map X=x --map Y=y
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from forest_ai import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--pred", default="outputs/trees.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-dist", type=float, default=2.0)
    ap.add_argument("--quality", nargs="*", default=None,
                    help="restrict predictions, e.g. --quality good fair")
    ap.add_argument("--map", action="append", default=[],
                    metavar="OLD=NEW", help="rename a reference column")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    ref = pd.read_csv(args.reference)
    if args.map:
        ref = ref.rename(columns=dict(m.split("=", 1) for m in args.map))
    missing = {"x", "y"} - set(ref.columns)
    if missing:
        raise SystemExit(f"reference is missing column(s): {sorted(missing)}; "
                         f"it has {list(ref.columns)} - use --map to rename")

    text, matched, _ = evaluate.report(pred, ref, max_dist=args.max_dist,
                                       quality=args.quality)
    print(f"\nprediction: {args.pred}"
          + (f"  (quality in {args.quality})" if args.quality else ""))
    print(f"reference:  {args.reference}\n")
    print(text)

    os.makedirs(args.out, exist_ok=True)
    matched.to_csv(os.path.join(args.out, "matched_trees.csv"), index=False)
    evaluate.plot_agreement(os.path.join(args.out, "qc_06_agreement.png"), matched)
    print(f"\nwrote {args.out}/matched_trees.csv and qc_06_agreement.png")


if __name__ == "__main__":
    main()
