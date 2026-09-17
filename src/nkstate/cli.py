"""Command-line interface.

Convention, shared with the other repositories in this portfolio: progress and
status messages go to **stderr**, machine-readable JSON goes to **stdout**. So
``nkstate run -c config.yaml | jq .comparison`` works, and redirecting stdout
to a file gives a valid JSON document with no log lines mixed in.

Exit codes:

==== ==========================================================
0    success
2    bad input or invalid configuration
3    a required optional dependency is missing
4    an external tool (scGPT) is unavailable or failed
5    data leakage detected
==== ==========================================================

Exit code 5 is distinct on purpose: a leaking split is not a crash and not a
bad config, it is a scientifically invalid run, and a caller in a batch script
should be able to tell that case apart without parsing a message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_MISSING_DEPENDENCY = 3
EXIT_EXTERNAL_TOOL = 4
EXIT_LEAKAGE = 5


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    from .io_utils import _Encoder, sanitise

    json.dump(
        sanitise(payload), sys.stdout, indent=2, cls=_Encoder, allow_nan=False
    )
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nkstate",
        description=(
            "NK cell functional state classification: marker panels, "
            "donor-grouped evaluation, baselines, and the scGPT integration layer."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the full pipeline from a config")
    run.add_argument("-c", "--config", required=True, help="path to a YAML config")
    run.add_argument("-o", "--output-dir", default=None, help="override output_dir")

    markers = subparsers.add_parser(
        "markers", help="report the marker panels and their coverage in a dataset"
    )
    markers.add_argument("-c", "--config", default=None, help="check coverage in this dataset")

    scgpt = subparsers.add_parser(
        "scgpt", help="report scGPT checkpoint status and the fine-tuning command"
    )
    scgpt.add_argument("-c", "--config", required=True, help="path to a YAML config")
    scgpt.add_argument(
        "--dry-run", action="store_true",
        help="build the command even with no checkpoint configured",
    )

    tokenise = subparsers.add_parser(
        "tokenise", help="validate scGPT input tokenisation against a vocabulary"
    )
    tokenise.add_argument("-c", "--config", required=True)

    validate = subparsers.add_parser(
        "validate", help="run the split and labelling control experiments"
    )
    validate.add_argument(
        "--seed", type=int, default=0, help="seed for the synthetic cohorts"
    )
    return parser


def _load(path: str):
    from .config import load_config

    return load_config(path)


def command_run(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .data.splits import LeakageError
    from .models.scgpt import ScgptUnavailable
    from .pipeline import PipelineError, run_pipeline

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    _log(f"[nkstate] running '{config.name}'")
    _log(f"[nkstate] data source: {config.data.source}, split: {config.split.kind}")
    try:
        result = run_pipeline(config, output_dir=args.output_dir)
    except LeakageError as error:
        _log(f"DATA LEAKAGE DETECTED: {error}")
        return EXIT_LEAKAGE
    except ScgptUnavailable as error:
        _log(f"scGPT unavailable: {error}")
        return EXIT_EXTERNAL_TOOL
    except ImportError as error:
        _log(f"missing dependency: {error}")
        return EXIT_MISSING_DEPENDENCY
    except (PipelineError, ValueError) as error:
        _log(f"pipeline error: {error}")
        return EXIT_BAD_INPUT

    _log(
        f"[nkstate] {result.n_cells} cells, {result.n_genes} genes, "
        f"{result.n_donors} donors"
    )
    for evaluation in result.evaluations:
        donor_part = (
            f"  donor macro-F1 {evaluation.donor.mean_macro_f1:.3f} "
            f"+/- {evaluation.donor.std_macro_f1:.3f}"
            if evaluation.donor else "  donor-level: not computed"
        )
        _log(
            f"[nkstate] {evaluation.model:<14} cell macro-F1 "
            f"{evaluation.cell.macro_f1:.3f}{donor_part}"
        )
    for warning in result.warnings:
        _log(f"[nkstate] warning: {warning}")

    _emit(result.to_dict())
    return EXIT_OK


def command_markers(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .markers import ALL_PANELS, NK_STATES, panel_overlap

    payload: dict = {
        "states": list(NK_STATES),
        "panels": {
            name: panel.to_dict()
            for name, panel in ALL_PANELS.items()
        },
        "panel_overlap": panel_overlap(),
    }

    if args.config:
        from .markers import panel_coverage
        from .pipeline import prepare_dataset

        try:
            config = _load(args.config)
        except ConfigError as error:
            _log(f"config error: {error}")
            return EXIT_BAD_INPUT
        _log(f"[nkstate] checking panel coverage in {config.data.source}")
        adata, _ = prepare_dataset(config)
        payload["coverage"] = panel_coverage(list(adata.var_names))
        payload["dataset"] = {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)}

    _emit(payload)
    return EXIT_OK


def command_scgpt(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .models.scgpt import (
        ScgptUnavailable, build_finetune_command, checkpoint_report,
    )

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    report = checkpoint_report(config.scgpt.checkpoint_dir)
    _log(f"[nkstate] scGPT checkpoint available: {report['available']}")
    try:
        command = build_finetune_command(
            checkpoint_dir=config.scgpt.checkpoint_dir,
            data_path=Path(config.output_dir) / "scgpt_input.h5ad",
            output_dir=Path(config.output_dir) / "scgpt",
            n_classes=len(config.labels.states),
            epochs=config.scgpt.epochs,
            batch_size=config.scgpt.batch_size,
            learning_rate=config.scgpt.learning_rate,
            max_seq_len=config.scgpt.max_seq_len,
            n_bins=config.scgpt.n_bins,
            freeze_encoder=config.scgpt.freeze_encoder,
            dry_run=args.dry_run or not report["available"],
        )
    except ScgptUnavailable as error:
        _log(f"scGPT unavailable: {error}")
        return EXIT_EXTERNAL_TOOL

    _emit({
        "checkpoint": report, "finetune_command": command.to_dict(),
        "executed": False,
        "note": (
            "this reports the command that would fine-tune scGPT; it is not run "
            "here and no transformer metric is produced"
        ),
    })
    return EXIT_OK


def command_tokenise(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .markers import all_state_genes
    from .models.scgpt import (
        ScgptUnavailable, TokenisationError, load_vocabulary, tokenise,
    )
    from .pipeline import _dense, prepare_dataset

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    if not config.scgpt.vocabulary_path:
        _log(
            "scgpt.vocabulary_path is not set. Tokenisation needs the "
            "checkpoint's vocab.json, which fixes which genes the model can "
            "represent; obtain a checkpoint from "
            "https://github.com/bowang-lab/scGPT and set the path."
        )
        return EXIT_EXTERNAL_TOOL

    try:
        vocabulary = load_vocabulary(config.scgpt.vocabulary_path)
    except ScgptUnavailable as error:
        _log(str(error))
        return EXIT_EXTERNAL_TOOL

    adata, _ = prepare_dataset(config)
    _log(f"[nkstate] tokenising {adata.n_obs} cells against {len(vocabulary)} vocabulary genes")
    try:
        tokenised = tokenise(
            _dense(adata.X), list(adata.var_names), vocabulary,
            max_seq_len=config.scgpt.max_seq_len, n_bins=config.scgpt.n_bins,
            pin_genes=all_state_genes(),
        )
    except TokenisationError as error:
        _log(f"tokenisation failed: {error}")
        return EXIT_BAD_INPUT

    _emit(tokenised.to_dict())
    return EXIT_OK


def command_validate(args: argparse.Namespace) -> int:
    from .validation import run_controls

    _log("[nkstate] running split and labelling control experiments")
    report = run_controls(seed=args.seed)
    for check in report["checks"]:
        _log(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['name']}")
    _emit(report)
    return EXIT_OK if report["all_passed"] else EXIT_BAD_INPUT


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": command_run, "markers": command_markers, "scgpt": command_scgpt,
        "tokenise": command_tokenise, "validate": command_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
