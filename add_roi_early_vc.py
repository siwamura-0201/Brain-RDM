import argparse
from pathlib import Path

import bdpy

SOURCE_ROIS = ["ROI_V1", "ROI_V2", "ROI_V3"]
NEW_ROI = "Early_VC"
DATA_DIR = Path("data/god_raw")


def add_early_vc(h5_path):
    bdata = bdpy.BData(h5_path)

    missing = [roi for roi in SOURCE_ROIS if roi not in bdata.metadata.key]
    if missing:
        raise ValueError(f"{h5_path}: missing source ROI(s) {missing}")

    bdata.merge_metadata(
        key=NEW_ROI,
        sources=SOURCE_ROIS,
        description=f"1 = ROI {NEW_ROI} ({' | '.join(SOURCE_ROIS)})",
        where="VoxelData",
    )

    n_voxels = bdata.get(NEW_ROI).shape[1]
    bdata.save(h5_path)
    print(f"{h5_path}: added {NEW_ROI} ({n_voxels} voxels)")


def main(data_dir):
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    for h5_path in h5_files:
        add_early_vc(h5_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f"Merge {', '.join(SOURCE_ROIS)} into a new ROI '{NEW_ROI}' "
            "(logical OR) and add it to every BData (.h5) file in the "
            "target directory. Files are overwritten in place."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing BData .h5 files (default: {DATA_DIR})",
    )
    args = parser.parse_args()
    main(args.data_dir)
