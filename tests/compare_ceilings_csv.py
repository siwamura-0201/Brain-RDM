#!/usr/bin/env python3
"""Two ceilings.csv files, compared column by column.

Used to hold the batched `permutation_null` to the implementation it
replaced: the same pipeline, the same seeds, the same n_perm, so every
column must agree. Float columns are compared on their own scale; the
permutation columns are called out separately because they are the ones the
change could move.

Usage: python tests/compare_ceilings_csv.py OLD.csv NEW.csv
"""
import sys

import numpy as np
import pandas as pd

KEYS = ["roi", "brain_metric", "model_rdm", "comparator", "subject", "row_type"]
PERM_COLUMNS = {"perm_p", "perm_z", "n_perm", "perm_seed"}


def main(old_path, new_path):
    old = pd.read_csv(old_path).sort_values(KEYS).reset_index(drop=True)
    new = pd.read_csv(new_path).sort_values(KEYS).reset_index(drop=True)

    if list(old.columns) != list(new.columns):
        print(f"FAIL: column sets differ\n  only in old: {set(old) - set(new)}"
              f"\n  only in new: {set(new) - set(old)}")
        return 1
    if len(old) != len(new):
        print(f"FAIL: row counts differ: {len(old)} vs {len(new)}")
        return 1
    for k in KEYS:
        if not old[k].equals(new[k]):
            print(f"FAIL: key column {k} differs")
            return 1

    worst = 0.0
    failures = []
    print(f"{'column':<26} {'kind':<5} {'max abs diff':>14}  {'max rel diff':>13}")
    print("-" * 64)
    for col in old.columns:
        if col in KEYS:
            continue
        a, b = old[col], new[col]
        if a.dtype == object or not np.issubdtype(a.dtype, np.number):
            same = a.fillna("<NA>").astype(str).eq(b.fillna("<NA>").astype(str)).all()
            print(f"{col:<26} {'text':<5} {'identical' if same else 'DIFFERENT':>14}")
            if not same:
                failures.append(col)
            continue

        av, bv = a.to_numpy(float), b.to_numpy(float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            print(f"{col:<26} {'num':<5} {'NaN PATTERN DIFFERS':>14}")
            failures.append(col)
            continue
        ok = ~np.isnan(av)
        if not ok.any():
            print(f"{col:<26} {'num':<5} {'all NaN':>14}")
            continue
        diff = np.abs(av[ok] - bv[ok])
        scale = np.maximum(np.abs(av[ok]), 1e-300)
        tag = " *perm" if col in PERM_COLUMNS else ""
        print(f"{col:<26} {'num':<5} {diff.max():>14.3e}  "
              f"{(diff / scale).max():>13.3e}{tag}")
        worst = max(worst, diff.max())
        # p 値と seed は完全一致でなければならない（順序統計量と乱数の引き）
        if col in ("perm_p", "n_perm", "perm_seed", "boot_seed") and diff.max() != 0:
            failures.append(col)

    print("-" * 64)
    print(f"worst absolute difference across all numeric columns: {worst:.3e}")
    if failures:
        print(f"FAIL: columns that must match exactly did not: {failures}")
        return 1
    print("PASS: identical keys, identical p-values and seeds, "
          "numeric columns agree to floating-point noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
