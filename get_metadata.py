import argparse
import os
import re

import numpy as np
import pandas as pd
import bdpy

DATA_DIR = "data/THINGS"  # input dir
OUTPUT_DIR = "output"   # output dir

# THINGS stimulus names look like "mango_13s" or "candy_bar_04s":
# <category>_<instance number><exemplar letter>. The category itself may
# contain underscores, so we strip the trailing "_<digits><letters>" part
# instead of splitting on the first underscore.
STIMULUS_NAME_PATTERN = re.compile(r'^(.*)_\d+[a-zA-Z]*$')


def stimulus_to_category(stimulus_name):
    m = STIMULUS_NAME_PATTERN.match(stimulus_name)
    if m is None:
        raise ValueError(f"Unexpected stimulus name format: {stimulus_name!r}")
    return m.group(1)


def build_dataframe(bdata, subject_label):
    stimuli = bdata.get_label("stimulus_name")   # list[str], length N
    trial_type = bdata.get_label("trial_type")   # list[str] ("train"/"test"), length N
    trial_id = bdata.get("trial_id")             # ndarray (N, 1)
    session = bdata.get("session")               # ndarray (N, 1)
    run = bdata.get("run")                       # ndarray (N, 1)

    df = pd.DataFrame({
        "subject": subject_label,
        "trial_id": np.asarray(trial_id).ravel().astype(int),
        "stimulus_name": stimuli,
        "category": [stimulus_to_category(s) for s in stimuli],
        "session": np.asarray(session).ravel().astype(int),
        "run": np.asarray(run).ravel().astype(int),
        "trial_type": trial_type,
    })
    return df


def main(subject_num):
    subject_label = f"S{int(subject_num)}"

    bdata = bdpy.BData(os.path.join(DATA_DIR, f"sub-{subject_num}.h5"))
    df = build_dataframe(bdata, subject_label)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"sub-{subject_num}_stimuli.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_num", help="e.g. 01")
    args = parser.parse_args()
    main(args.subject_num)
