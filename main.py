"""Stress-adaptive dimensionality selection experiment."""

import argparse
from dataclasses import replace
from pathlib import Path

from sads import constants as c
from sads.constants import OUTPUT_DIR as OUTPUT
from sads.experiment import Config, run

PARAMETERS = Config()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=c.DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start", default=c.SAMPLE_START)
    parser.add_argument("--end", default=c.SAMPLE_END)
    args = parser.parse_args()
    cfg = replace(
        PARAMETERS,
        returns_path=args.data_dir / c.RETURNS_PATH.name,
        stress_path=args.data_dir / c.STRESS_PATH.name,
        start=args.start,
        end=args.end,
    )
    results = run(cfg, args.output)
    for signal, result in results.items():
        print(f"\n{signal}\n{result['summary'].to_string(index=False)}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
