#!/usr/bin/env python3
"""Fast end-to-end smoke test: every module exercised on synthetic data.

Runs in well under a minute and needs no network access or external tool. It
answers one question — is this installation capable of producing a result at
all — and it is the first thing to run after cloning.

    python scripts/smoke_test.py

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))


def main() -> int:  # noqa: PLR0915 - a flat list of checks is the point
    from nkstate import markers
    from nkstate.config import ConfigError, config_from_dict
    from nkstate.data.splits import LeakageError, cell_split, donor_split
    from nkstate.evaluate import cell_metrics, compare_models, evaluate_predictions
    from nkstate.interpret import (
        benjamini_hochberg, differential_expression, interpret_states,
        marker_enrichment,
    )
    from nkstate.io_utils import write_json
    from nkstate.models.baselines import BASELINE_MODELS, ModelError, fit_baseline
    from nkstate.models.scgpt import (
        ScgptUnavailable, bin_expression, build_finetune_command,
        checkpoint_report, map_to_vocabulary, tokenise,
    )
    from nkstate.scoring import assign_states, score_states
    from nkstate.testing import SyntheticSpec, build_preprocessed

    # ---- markers ----------------------------------------------------------
    check(
        "marker panels defined for all four NK states",
        all(s in markers.STATE_PANELS for s in markers.NK_STATES),
        f"{len(markers.STATE_PANELS)} panels",
    )
    check(
        "every panel carries a literature reference",
        all(p.reference for p in markers.ALL_PANELS.values()),
        f"{len(markers.ALL_PANELS)} panels",
    )
    check(
        "lineage aliases resolve real annotation labels",
        markers.resolve_panel("CD4 T cells") == "T_cell"
        and markers.resolve_panel("NK cells") == "NK_cell",
    )
    check(
        "an unknown label resolves to None rather than raising",
        markers.resolve_panel("erythrocyte") is None,
    )

    # ---- synthetic data ---------------------------------------------------
    spec = SyntheticSpec(n_donors=6, cells_per_donor=120, state_effect=0.8, seed=0)
    data = build_preprocessed(spec)
    check(
        "synthetic dataset has the requested shape",
        data.adata.n_obs == 6 * 120,
        f"{data.adata.n_obs} cells x {data.adata.n_vars} genes",
    )
    check(
        "log-normalised expression is non-negative",
        float(data.features.min()) >= 0.0,
        f"min {float(data.features.min()):.3f}",
    )
    check(
        "raw counts retained in a layer",
        "counts" in data.adata.layers,
    )

    # ---- scoring ----------------------------------------------------------
    scores = score_states(data.adata, states=markers.NK_STATES)
    check(
        "all four panels reach full coverage on synthetic data",
        all(v > 0.9 for v in scores.coverage.values()),
        f"coverage {min(scores.coverage.values()):.2f}-{max(scores.coverage.values()):.2f}",
    )
    assignment = assign_states(scores)
    check(
        "pseudo-labels are tagged as pseudo-labels",
        assignment.label_type == "marker_pseudo_label",
    )
    from sklearn.metrics import adjusted_rand_score

    confident = assignment.confident_mask()
    ari = adjusted_rand_score(
        data.true_labels[confident], assignment.labels[confident]
    )
    check(
        "marker pseudo-labels recover the generating state",
        ari > 0.9,
        f"ARI {ari:.3f} on {int(confident.sum())}/{confident.size} confident cells",
    )

    # ---- splits -----------------------------------------------------------
    split = donor_split(data.donors, test_fraction=0.25, val_fraction=0.0, seed=0)
    train_donors = set(data.donors[np.asarray(split.train)].tolist())
    test_donors = set(data.donors[np.asarray(split.test)].tolist())
    check(
        "donor-grouped split shares no donor between train and test",
        not (train_donors & test_donors),
        f"{len(train_donors)} train / {len(test_donors)} test donors",
    )
    leaky = cell_split(data.donors, test_fraction=0.25, val_fraction=0.0, seed=0)
    shared = set(data.donors[np.asarray(leaky.train)].tolist()) & set(
        data.donors[np.asarray(leaky.test)].tolist()
    )
    check(
        "cell-random split does share donors (the documented failure mode)",
        len(shared) > 0,
        f"{len(shared)} shared donors",
    )
    try:
        donor_split(np.array(["one"] * 100), test_fraction=0.25, seed=0)
        refused = False
    except Exception:
        refused = True
    check("donor split on a single donor is refused", refused)

    # ---- baselines --------------------------------------------------------
    train_index = np.asarray(split.train, dtype=int)
    test_index = np.asarray(split.test, dtype=int)
    fitted_models = {}
    for name in BASELINE_MODELS:
        try:
            fitted_models[name] = fit_baseline(
                name, data.features[train_index], data.true_labels[train_index],
                data.gene_names, seed=0,
            )
        except ModelError as error:
            check(f"baseline {name} fits", False, str(error))
    check(
        "every baseline fits on synthetic data",
        len(fitted_models) == len(BASELINE_MODELS),
        f"{len(fitted_models)}/{len(BASELINE_MODELS)}",
    )
    check(
        "logistic regression exposes per-class gene coefficients",
        set(fitted_models["logistic"].gene_importance(5)) == set(markers.NK_STATES),
    )
    check(
        "marker_score baseline does no fitting yet predicts",
        fitted_models["marker_score"].predict(data.features[test_index]).size
        == test_index.size,
    )

    # ---- evaluation -------------------------------------------------------
    results = []
    for name, model in fitted_models.items():
        predicted = model.predict(data.features[test_index])
        results.append(
            evaluate_predictions(
                model=name, true_labels=data.true_labels[test_index],
                predicted_labels=predicted,
                donor_ids=data.donors[test_index],
                split_kind="donor", label_type="synthetic_ground_truth",
            )
        )
    check(
        "positive control is learnable by logistic regression",
        results[0].cell.macro_f1 > 0.85,
        f"macro-F1 {results[0].cell.macro_f1:.3f}",
    )
    comparison = compare_models(results)
    check(
        "model comparison names a best model and the marker floor",
        comparison["best_model"] in BASELINE_MODELS
        and comparison["marker_score_floor_macro_f1"] is not None,
        f"best {comparison['best_model']}",
    )
    single = evaluate_predictions(
        model="logistic", true_labels=["a", "b", "a", "b"],
        predicted_labels=["a", "b", "a", "b"], donor_ids=["d1"] * 4,
    )
    check(
        "one held-out donor yields no donor-level metric, with a caveat",
        single.donor is None and any("at least 2" in c for c in single.caveats),
    )
    metrics = cell_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    check(
        "chance accuracy is the majority-class rate",
        abs(metrics.chance_accuracy - 0.5) < 1e-12,
        f"{metrics.chance_accuracy:.3f}",
    )

    # ---- interpretation ---------------------------------------------------
    adjusted = benjamini_hochberg(np.array([0.001, 0.01, 0.5, 0.9]))
    check(
        "BH adjustment is monotone and bounded by 1",
        bool(np.all(np.diff(adjusted) >= -1e-12)) and float(adjusted.max()) <= 1.0,
        f"max {float(adjusted.max()):.3f}",
    )
    de = differential_expression(
        data.features[train_index], data.gene_names,
        data.true_labels[train_index], "cytotoxic", top_n=25,
    )
    check(
        "differential expression returns ranked genes with q-values",
        len(de) == 25 and de[0].q_value <= de[-1].q_value,
        f"top gene {de[0].gene} q={de[0].q_value:.2e}",
    )
    enrichment = marker_enrichment(
        [g.gene for g in de], "cytotoxic", data.gene_names, top_n=25
    )
    check(
        "cytotoxic DE genes are enriched for the cytotoxic panel",
        enrichment.is_enriched,
        f"{enrichment.n_overlap}/{enrichment.panel_size_in_universe} "
        f"fold {enrichment.fold_enrichment:.1f} p={enrichment.p_value:.1e}",
    )
    interpretation = interpret_states(
        data.features[train_index], data.gene_names,
        data.true_labels[train_index], top_n=25,
    )
    check(
        "every state's DE genes are enriched for its own panel",
        interpretation["all_states_enriched"],
        f"tested {len(interpretation['states_tested'])} states",
    )

    # ---- scGPT layer ------------------------------------------------------
    vocabulary = {g: i + 3 for i, g in enumerate(data.gene_names[:200])}
    mapping = map_to_vocabulary(
        data.gene_names, vocabulary, markers.all_state_genes()
    )
    check(
        "vocabulary mapping reports coverage and dropped marker genes",
        0 < mapping.coverage < 1 and isinstance(mapping.dropped_marker_genes, list),
        f"coverage {mapping.coverage:.3f}, "
        f"{len(mapping.dropped_marker_genes)} markers dropped",
    )
    binned = bin_expression(data.features[:20], n_bins=51)
    check(
        "expression bins are per-cell and zeros map to bin 0",
        int(binned.max()) <= 50
        and bool(np.all(binned[data.features[:20] == 0] == 0)),
        f"bin range {int(binned.min())}-{int(binned.max())}",
    )
    check(
        "a cell with no expression bins to all zeros",
        int(bin_expression(np.zeros((1, 10)), n_bins=51).max()) == 0,
    )
    try:
        bin_expression(np.array([[-1.0, 2.0]]), n_bins=51)
        rejected = False
    except Exception:
        rejected = True
    check("binning rejects negative (scaled) expression", rejected)
    tokenised = tokenise(
        data.features, data.gene_names, vocabulary,
        max_seq_len=64, pin_genes=markers.all_state_genes(),
    )
    check(
        "tokenisation truncates to max_seq_len and pins marker genes",
        tokenised.gene_ids.shape[1] == 64 and len(tokenised.pinned_genes) > 0,
        f"{len(tokenised.pinned_genes)} pinned of 64 positions",
    )
    report = checkpoint_report(None)
    check(
        "an absent scGPT checkpoint is reported, not assumed",
        report["available"] is False and "required_files" in report,
    )
    try:
        build_finetune_command(None, "d.h5ad", "out", 4, dry_run=False)
        refused_ft = False
    except ScgptUnavailable:
        refused_ft = True
    check("fine-tuning without a checkpoint is refused", refused_ft)
    command = build_finetune_command(None, "d.h5ad", "out", 4, dry_run=True)
    check(
        "dry-run still builds a complete fine-tuning command",
        "cell_annotation" in command.command_string and "--n-cls=4" in command.command_string,
    )

    # ---- config -----------------------------------------------------------
    config = config_from_dict({"data": {"n_top_genes": 500}})
    check("a minimal config loads with defaults", config.data.n_top_genes == 500)
    for payload, why in [
        ({"data": {"n_top_gene": 500}}, "misspelled key"),
        ({"nonsense": {}}, "unknown section"),
        ({"split": {"kind": "random"}}, "invalid split kind"),
        ({"data": {"max_mito_fraction": 15.0}}, "percentage instead of fraction"),
    ]:
        try:
            config_from_dict(payload)
            check(f"config rejects {why}", False, "accepted it")
        except ConfigError:
            check(f"config rejects {why}", True)

    # ---- serialisation ----------------------------------------------------
    with tempfile.TemporaryDirectory() as directory:
        path = write_json(
            Path(directory) / "out.json",
            {
                "bool": np.bool_(True), "int": np.int64(3),
                "float": np.float64(1.5), "nan": np.float64("nan"),
                "array": np.arange(3),
            },
        )
        import json

        loaded = json.loads(path.read_text())
    check(
        "numpy scalars, arrays and NaN serialise to valid JSON",
        loaded["bool"] is True and loaded["nan"] is None and loaded["array"] == [0, 1, 2],
    )

    # ---- report -----------------------------------------------------------
    width = 66
    print()
    print("SMOKE TEST")
    print("=" * (width + 12))
    for number, (name, passed, detail) in enumerate(CHECKS, start=1):
        status = "PASS" if passed else "FAIL"
        print(f"{number:>3}  {status}  {name[:width]:<{width}}  {detail}")
    print("=" * (width + 12))
    n_passed = sum(1 for _, passed, _ in CHECKS if passed)
    print(f"{n_passed}/{len(CHECKS)} checks passed")
    print()
    return 0 if n_passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
