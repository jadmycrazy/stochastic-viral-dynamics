"""Run all experiments and export figure data.

Usage: python run_experiments.py results/viral.json

Kept for backward compatibility. `viral-dynamics experiments` does the same.
"""

import sys

from viral.experiments import run_all, write_results

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/viral.json"
    write_results(run_all(verbose=True), output_path)
