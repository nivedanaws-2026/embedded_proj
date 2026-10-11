"""CLI wrapper: regenerate the SIMULATED virtual-farm datasets.

Writes farm_clean.csv, farm_faulty.csv, farm_shift.csv into data/raw/ and the
parameter tables into configs/. Running with the default seed (1000) reproduces
the committed datasets.

Usage:
    python scripts/generate_datasets.py [--seed 1000] [--out data/raw]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_generation.virtual_farm import generate_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simulated farm datasets.")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default=str(config.DATA_RAW))
    args = ap.parse_args()
    res = generate_all(out_dir=Path(args.out), seed=args.seed)
    print("SIMULATED data written to", args.out)
    for name, df in res.items():
        print(f"  {name}: {df.shape[0]:>6} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
