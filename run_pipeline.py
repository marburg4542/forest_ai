"""End-to-end forest inventory from a .las point cloud.

    python run_pipeline.py [--las Forest.las] [--out outputs] [--cache DIR]

Stages are cached in --cache keyed on the input file and the settings that
affect the terrain model, so a re-run only repeats what changed.  Use --force
to recompute everything.

The web interface (`python serve.py`) calls the same forest_ai.pipeline code,
so both produce identical numbers.
"""

from __future__ import annotations

import argparse
import json

from forest_ai import pipeline
from forest_ai.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", default="Forest.las")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-las", action="store_true",
                    help="skip writing the segmented point cloud (it is large)")
    ap.add_argument("--groups", type=int, default=4,
                    help="number of structural groups for the label-free clustering")
    args = ap.parse_args()

    print("=" * 72)
    print(f"forest_ai pipeline | {args.las}")
    print("=" * 72)

    cfg = Config(las_path=args.las, out_dir=args.out)
    result = pipeline.run(args.las, cfg, cache_dir=args.cache, force=args.force,
                          groups=args.groups, verbose=True)
    print(json.dumps(result.header, indent=2, default=str))

    written = pipeline.write_outputs(result, args.out,
                                     figures=not args.no_figures,
                                     segmented_las=not args.no_las)
    print(f"\nwrote to {args.out}/: " + ", ".join(written))
    print("\n" + pipeline.summary_text(result))


if __name__ == "__main__":
    main()
