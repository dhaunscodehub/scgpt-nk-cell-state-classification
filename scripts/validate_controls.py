#!/usr/bin/env python3
"""Run the control experiments and print a PASS/FAIL table.

These controls are what make the reported metrics interpretable. Run this
first on any new machine: if the negative control scores above chance, nothing
else in the repository can be trusted.

    python scripts/validate_controls.py
    python scripts/validate_controls.py --json results/controls.json

Exit status is 0 only if every control passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nkstate.io_utils import provenance, write_json  # noqa: E402
from nkstate.validation import run_controls  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", default=None, help="also write the full record to this path"
    )
    args = parser.parse_args(argv)

    report = run_controls(seed=args.seed)

    print()
    print("CONTROL EXPERIMENTS")
    print("=" * 78)
    print(f"{'':<4}{'result':<7}{'check':<48}{'measured':>10}{'chance':>9}")
    print("-" * 78)
    for number, check in enumerate(report["checks"], start=1):
        measured = (
            f"{check['balanced_accuracy']:.3f}"
            if "balanced_accuracy" in check
            else f"{check['optimism_balanced_accuracy']:.3f}"
            if "optimism_balanced_accuracy" in check
            else f"{check['adjusted_rand_index']:.3f}"
            if "adjusted_rand_index" in check
            else "-"
        )
        chance = f"{check['chance']:.3f}" if "chance" in check else "-"
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{number:<4}{status:<7}{check['name'][:47]:<48}{measured:>10}{chance:>9}")
    print("-" * 78)
    print(f"{report['n_passed']}/{report['n_checks']} controls passed")
    print()

    print("REPLICATE DETAIL")
    print("-" * 78)
    for check in report["checks"]:
        if "per_seed_balanced_accuracy" in check:
            print(
                f"  {check['split_kind']:<12} "
                f"mean {check['balanced_accuracy']:.3f} "
                f"sd {check['balanced_accuracy_sd']:.3f} "
                f"range [{check['balanced_accuracy_min']:.3f}, "
                f"{check['balanced_accuracy_max']:.3f}]  "
                f"seeds {check['seeds']}"
            )
    print()
    print("INTERPRETATION")
    print("-" * 78)
    for line in _wrap(report["interpretation"], 78):
        print(line)
    print()

    if args.json:
        path = write_json(args.json, {**report, "provenance": provenance()})
        print(f"wrote {path}")

    return 0 if report["all_passed"] else 1


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    raise SystemExit(main())
