"""scGPT integration: tokenisation, expression binning and fine-tuning commands.

scGPT (Cui et al., *Nat Methods* 21:1470, 2024) is a transformer pretrained on
33 million cells. Fine-tuning it requires:

* the ``scgpt`` package;
* a pretrained checkpoint (~1-2 GB), distributed by the authors through Google
  Drive under their own terms and not redistributable here;
* the checkpoint's gene vocabulary, which fixes what genes the model can see;
* a GPU with enough memory for a 1200-gene sequence per cell.

None of that is available in this repository, so this module implements the
parts that can be validated without it — the **input transformation**, which is
where the errors that silently ruin a fine-tune actually live — and builds the
documented command lines for the parts that cannot.

The three transformations that matter:

1. **Gene vocabulary mapping.** Genes absent from the checkpoint's vocabulary
   are invisible to the model. Silently dropping them changes what the
   classifier sees; :func:`map_to_vocabulary` reports exactly which marker
   genes were lost, because losing GNLY from a cytotoxic-state task is fatal
   and losing an anonymous background gene is not.

2. **Expression binning.** scGPT discretises each cell's expression into value
   bins computed **per cell**, so a cell's tokens encode its own expression
   ranking rather than an absolute level. Getting this wrong — binning globally,
   or including zeros — changes the representation of every cell.

3. **Highly-variable-gene truncation.** The input is a fixed-length sequence,
   so genes must be ranked and truncated. Marker genes are pinned so the task's
   own signal is not truncated away.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# scGPT's default number of expression bins.
DEFAULT_N_BINS = 51
# Default input sequence length used by the authors' cell-annotation task.
DEFAULT_MAX_SEQ_LEN = 1200
# Special tokens in the scGPT vocabulary.
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
UNKNOWN_TOKEN = "<unk>"


class ScgptUnavailable(RuntimeError):
    """Raised when the scGPT package or checkpoint is not available."""


class TokenisationError(ValueError):
    """Raised for an input that cannot be tokenised as requested."""


@dataclass
class VocabularyMapping:
    """Result of mapping a dataset's genes onto a checkpoint vocabulary."""

    kept_genes: list[str]
    dropped_genes: list[str]
    token_ids: list[int]
    n_vocabulary: int
    dropped_marker_genes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.kept_genes) + len(self.dropped_genes)
        return len(self.kept_genes) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "n_kept": len(self.kept_genes), "n_dropped": len(self.dropped_genes),
            "coverage": self.coverage, "n_vocabulary": self.n_vocabulary,
            "n_dropped_marker_genes": len(self.dropped_marker_genes),
            "dropped_marker_genes": self.dropped_marker_genes,
            "dropped_examples": self.dropped_genes[:10],
        }


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Read a checkpoint's gene vocabulary from its JSON file.

    scGPT checkpoints ship a ``vocab.json`` mapping gene symbol to token id.
    The vocabulary is part of the model: a gene outside it has no embedding and
    cannot be represented at all.
    """
    import json

    path = Path(path)
    if not path.is_file():
        raise ScgptUnavailable(
            f"gene vocabulary not found at {path}. scGPT checkpoints include a "
            "vocab.json; download a checkpoint from the scGPT repository "
            "(https://github.com/bowang-lab/scGPT) and point "
            "scgpt.vocabulary_path at it."
        )
    vocabulary = json.loads(path.read_text())
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise TokenisationError(f"{path}: expected a non-empty gene->id mapping")
    return {str(k): int(v) for k, v in vocabulary.items()}


def map_to_vocabulary(
    gene_names: Sequence[str],
    vocabulary: dict[str, int],
    marker_genes: Sequence[str] = (),
) -> VocabularyMapping:
    """Map dataset genes onto vocabulary token ids, reporting what was lost.

    Marker-gene losses are separated out because they are not equivalent to
    background-gene losses: a cytotoxic-state classifier that cannot see GNLY
    or PRF1 is crippled, and a summary coverage percentage would hide that.
    """
    markers = set(marker_genes)
    kept, dropped, ids = [], [], []
    for gene in gene_names:
        symbol = str(gene)
        if symbol in vocabulary:
            kept.append(symbol)
            ids.append(vocabulary[symbol])
        else:
            dropped.append(symbol)
    return VocabularyMapping(
        kept_genes=kept, dropped_genes=dropped, token_ids=ids,
        n_vocabulary=len(vocabulary),
        dropped_marker_genes=[g for g in dropped if g in markers],
    )


def bin_expression(
    expression: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> np.ndarray:
    """Discretise expression into per-cell value bins, as scGPT does.

    Binning is **per cell** and over **non-zero values only**. Both details
    matter and both are easy to get wrong:

    * Per cell, because the token then encodes where a gene sits in *that
      cell's* expression distribution. Binning against a global distribution
      would make the representation depend on library size, which is the
      technical axis normalisation exists to remove.
    * Non-zero only, because in a typical cell 90% of genes are zero. Including
      them would put almost every gene in the lowest bin and compress all real
      signal into the top few.

    Zeros map to bin 0; non-zero values map to bins 1..n_bins-1 by quantile.
    """
    if expression.ndim != 2:
        raise TokenisationError(
            f"expected a (cells, genes) matrix, got shape {expression.shape}"
        )
    if n_bins < 2:
        raise TokenisationError("n_bins must be at least 2")
    if np.any(expression < 0):
        raise TokenisationError(
            "negative expression values cannot be binned; scGPT expects "
            "normalised non-negative values, so pass log-normalised data or "
            "raw counts"
        )

    binned = np.zeros(expression.shape, dtype=np.int64)
    edges_count = n_bins - 1
    for row in range(expression.shape[0]):
        values = expression[row]
        non_zero = values > 0
        if not non_zero.any():
            continue
        observed = values[non_zero]
        # Quantile edges over this cell's non-zero values.
        quantiles = np.linspace(0, 1, edges_count + 1)[1:-1]
        edges = np.quantile(observed, quantiles) if quantiles.size else np.array([])
        binned[row, non_zero] = np.searchsorted(edges, observed, side="right") + 1
    return binned


@dataclass
class TokenisedCells:
    """Tokenised input ready for scGPT."""

    gene_ids: np.ndarray        # (n_cells, seq_len) vocabulary token ids
    expression_bins: np.ndarray  # (n_cells, seq_len) bin indices
    gene_names: list[str]
    n_bins: int
    max_seq_len: int
    pinned_genes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_cells": int(self.gene_ids.shape[0]),
            "seq_len": int(self.gene_ids.shape[1]),
            "n_bins": self.n_bins, "max_seq_len": self.max_seq_len,
            "n_pinned_marker_genes": len(self.pinned_genes),
            "bin_range": [int(self.expression_bins.min()), int(self.expression_bins.max())],
        }


def tokenise(
    expression: np.ndarray,
    gene_names: Sequence[str],
    vocabulary: dict[str, int],
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    n_bins: int = DEFAULT_N_BINS,
    pin_genes: Sequence[str] = (),
) -> TokenisedCells:
    """Turn an expression matrix into scGPT's (gene id, expression bin) input.

    Genes are ranked by mean expression and truncated to ``max_seq_len``, with
    ``pin_genes`` retained regardless of rank. Pinning matters: the marker genes
    that define the task are often not among the most highly expressed, and
    truncating them away leaves the model trying to classify functional states
    without seeing the genes that define them.
    """
    if expression.shape[1] != len(gene_names):
        raise TokenisationError(
            f"expression has {expression.shape[1]} genes but {len(gene_names)} names"
        )
    mapping = map_to_vocabulary(gene_names, vocabulary, pin_genes)
    if not mapping.kept_genes:
        raise TokenisationError(
            "no dataset gene is in the checkpoint vocabulary; check that gene "
            "symbols match the vocabulary's convention (HGNC symbols, not Ensembl ids)"
        )

    index = {g: i for i, g in enumerate(map(str, gene_names))}
    kept_positions = [index[g] for g in mapping.kept_genes]
    kept = expression[:, kept_positions]

    order = np.argsort(-kept.mean(axis=0))
    pinned = [g for g in pin_genes if g in set(mapping.kept_genes)]
    pinned_positions = [mapping.kept_genes.index(g) for g in pinned]

    selected: list[int] = list(pinned_positions)
    for position in order:
        if len(selected) >= max_seq_len:
            break
        if position not in set(selected):
            selected.append(int(position))
    selected = selected[:max_seq_len]

    return TokenisedCells(
        gene_ids=np.asarray([[mapping.token_ids[p] for p in selected]] * kept.shape[0]),
        expression_bins=bin_expression(kept[:, selected], n_bins),
        gene_names=[mapping.kept_genes[p] for p in selected],
        n_bins=n_bins, max_seq_len=max_seq_len, pinned_genes=pinned,
    )


@dataclass
class FineTuneCommand:
    """A constructed scGPT fine-tuning invocation."""

    command: list[str]
    checkpoint: str
    vocabulary: str
    n_classes: int
    settings: dict = field(default_factory=dict)

    @property
    def command_string(self) -> str:
        return shlex.join(self.command)

    def to_dict(self) -> dict:
        return {
            "command": self.command_string, "checkpoint": self.checkpoint,
            "vocabulary": self.vocabulary, "n_classes": self.n_classes,
            "settings": self.settings,
        }


def build_finetune_command(
    checkpoint_dir: str | Path | None,
    data_path: str | Path,
    output_dir: str | Path,
    n_classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    n_bins: int = DEFAULT_N_BINS,
    freeze_encoder: bool = False,
    dry_run: bool = False,
) -> FineTuneCommand:
    """Build the scGPT cell-annotation fine-tuning command.

    Follows the ``Tutorial_Annotation`` workflow in the scGPT repository. The
    settings exposed here are the ones that change the result:
    ``max_seq_len`` (how many genes the model sees), ``n_bins`` (expression
    resolution), and ``freeze_encoder`` (linear probe versus full fine-tune —
    with a few thousand cells, freezing is usually the honest choice, since a
    full fine-tune of a 50M-parameter encoder on 2000 cells overfits).
    """
    if checkpoint_dir is None and not dry_run:
        raise ScgptUnavailable(
            "scGPT is not configured. Obtain a pretrained checkpoint from "
            "https://github.com/bowang-lab/scGPT (the authors distribute them "
            "under their own terms and they are not redistributed here), then set "
            "scgpt.checkpoint_dir in your config. Pass --dry-run to build the "
            "command without it."
        )
    if n_classes < 2:
        raise TokenisationError("n_classes must be at least 2")

    root = Path(checkpoint_dir) if checkpoint_dir else Path("<scgpt_checkpoint_dir>")
    command = [
        "python", "-m", "scgpt.tasks.cell_annotation",
        f"--load-model={root}",
        f"--data-path={Path(data_path)}",
        f"--save-dir={Path(output_dir)}",
        f"--n-cls={n_classes}",
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--lr={learning_rate}",
        f"--max-seq-len={max_seq_len}",
        f"--n-bins={n_bins}",
        # Highly variable genes only: the input is a fixed-length sequence, so
        # gene selection is not optional.
        "--include-zero-gene=False",
    ]
    if freeze_encoder:
        command.append("--freeze-encoder=True")

    return FineTuneCommand(
        command=command, checkpoint=str(root),
        vocabulary=str(root / "vocab.json"), n_classes=n_classes,
        settings={
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "max_seq_len": max_seq_len,
            "n_bins": n_bins, "freeze_encoder": freeze_encoder,
            "dry_run": dry_run,
        },
    )


def checkpoint_report(checkpoint_dir: str | Path | None) -> dict:
    """Describe what is present and what is missing in a checkpoint directory."""
    if checkpoint_dir is None:
        return {
            "available": False,
            "reason": "scgpt.checkpoint_dir is not set",
            "required_files": ["args.json", "vocab.json", "best_model.pt"],
            "source": "https://github.com/bowang-lab/scGPT",
        }
    root = Path(checkpoint_dir)
    required = ["args.json", "vocab.json", "best_model.pt"]
    present = [f for f in required if (root / f).is_file()]
    return {
        "available": len(present) == len(required),
        "checkpoint_dir": str(root),
        "present_files": present,
        "missing_files": [f for f in required if f not in present],
        "package_installed": _scgpt_installed(),
    }


def _scgpt_installed() -> bool:
    try:
        import scgpt  # noqa: F401
    except ImportError:
        return False
    return True
