"""Stress-adaptive dimensionality selection experiment."""

from sads.constants import OUTPUT_DIR as OUTPUT
from sads.experiment import Config, run

PARAMETERS = Config()

if __name__ == "__main__":
    results = run(PARAMETERS, OUTPUT)
    for signal, result in results.items():
        print(f"\n{signal}\n{result['summary'].to_string(index=False)}")
    print(f"\nSaved to {OUTPUT}")
