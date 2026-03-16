"""Utilities for merging feature CSVs and running StructScore evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ORIGIN_ALIASES = {
    "mnw": "multinews",
    "multi_news": "multinews",
    "multinews": "multinews",
    "govreport": "GovReport",
    "gov_report": "GovReport",
    "chem_sum": "ChemSum",
    "chemsum": "ChemSum",
    "arxiv": "arXiv",
}

DEFAULT_MERGE_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("origin", "summary"),
    ("origin", "source", "summary"),
    ("dataset", "summary"),
    ("id",),
    ("example_id",),
    ("sample_id",),
    ("summary",),
)


def _load_reweighting_helpers():
    """Import discourse reweighting utilities only when they are needed."""
    try:
        from .utils import process_sent_score, reweight_score
    except ImportError:
        from utils import process_sent_score, reweight_score
    return process_sent_score, reweight_score


def normalize_origin_name(value: object) -> object:
    """Map dataset aliases such as `mnw` to a single canonical name."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return ORIGIN_ALIASES.get(stripped.lower(), stripped)


def normalize_origin_column(df: pd.DataFrame, origin_column: str = "origin") -> pd.DataFrame:
    """Normalize split names in-place on a copy of the dataframe."""
    normalized = df.copy()
    if origin_column in normalized.columns:
        normalized[origin_column] = normalized[origin_column].map(normalize_origin_name)
    return normalized


def load_csv(path: str | Path, origin_column: str = "origin") -> pd.DataFrame:
    """Load a CSV and normalize dataset aliases when possible."""
    return normalize_origin_column(pd.read_csv(path), origin_column=origin_column)


def infer_merge_keys(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    merge_keys: Sequence[str] | None = None,
) -> list[str]:
    """Infer merge keys shared by both frames."""
    if merge_keys:
        missing = [key for key in merge_keys if key not in base_df.columns or key not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing merge keys in one of the files: {missing}")
        return list(merge_keys)

    for candidate in DEFAULT_MERGE_KEY_CANDIDATES:
        if all(column in base_df.columns and column in feature_df.columns for column in candidate):
            return list(candidate)

    shared_columns = [column for column in base_df.columns if column in feature_df.columns]
    raise ValueError(
        "Could not infer merge keys automatically. "
        f"Shared columns were {shared_columns}. Please pass --merge-keys explicitly."
    )


def _prepare_feature_frame(
    feature_df: pd.DataFrame,
    merge_keys: Sequence[str],
    existing_columns: set[str],
    suffix: str,
) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in feature_df.columns:
        if column in merge_keys:
            continue
        if column in existing_columns:
            rename_map[column] = f"{column}_{suffix}"
    prepared = feature_df.rename(columns=rename_map)
    keep_columns = list(merge_keys) + [column for column in prepared.columns if column not in merge_keys]
    return prepared[keep_columns]


def merge_feature_frames(
    base_df: pd.DataFrame,
    feature_frames: Sequence[tuple[str, pd.DataFrame]],
    merge_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge a base dataframe with one or more feature dataframes."""
    merged = base_df.copy()
    for feature_name, feature_df in feature_frames:
        keys = infer_merge_keys(merged, feature_df, merge_keys=merge_keys)
        prepared = _prepare_feature_frame(
            feature_df=feature_df,
            merge_keys=keys,
            existing_columns=set(merged.columns),
            suffix=Path(feature_name).stem,
        )
        merged = merged.merge(prepared, on=keys, how="left", validate="one_to_one")
    return merged


def merge_feature_files(
    base_path: str | Path,
    feature_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    merge_keys: Sequence[str] | None = None,
    origin_column: str = "origin",
) -> pd.DataFrame:
    """Merge multiple CSV feature files into one dataframe and optionally save it."""
    base_df = load_csv(base_path, origin_column=origin_column)
    feature_frames = [
        (str(path), load_csv(path, origin_column=origin_column))
        for path in feature_paths
    ]
    merged = merge_feature_frames(base_df, feature_frames, merge_keys=merge_keys)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
    return merged


def load_merged_features(
    path: str | Path,
    origins: Iterable[str] | None = None,
    origin_column: str = "origin",
) -> pd.DataFrame:
    """Load a merged CSV and optionally filter to selected origins."""
    df = load_csv(path, origin_column=origin_column)
    if origins is None:
        return df
    normalized_origins = {normalize_origin_name(origin) for origin in origins}
    return df[df[origin_column].isin(normalized_origins)].reset_index(drop=True)


def split_frames_by_origin(
    df: pd.DataFrame,
    origin_column: str = "origin",
) -> dict[str, pd.DataFrame]:
    """Return one dataframe per dataset split."""
    normalized = normalize_origin_column(df, origin_column=origin_column)
    return {
        str(origin): split.reset_index(drop=True)
        for origin, split in normalized.groupby(origin_column, dropna=False)
    }


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def evaluate_score_column(
    df: pd.DataFrame,
    score_column: str,
    label_column: str = "label",
    threshold: float = 0.5,
    use_discourse_reweighting: bool = False,
    alpha: float = 1.0,
    depth_factor: float = 1.0,
) -> dict[str, float]:
    """Evaluate a score column with optional discourse-aware reweighting."""
    working = df.dropna(subset=[score_column, label_column]).reset_index(drop=True)
    labels = working[label_column].tolist()

    if use_discourse_reweighting:
        process_sent_score, reweight_score = _load_reweighting_helpers()
        weighted_scores = []
        original_scores = []
        for _, row in working.iterrows():
            score_list = process_sent_score(row[score_column])
            weighted_mean, original_mean, _, _, _ = reweight_score(
                score_list=score_list,
                row=row,
                alpha=alpha,
                depth_factor=depth_factor,
            )
            weighted_scores.append(weighted_mean)
            original_scores.append(original_mean)
        scores = weighted_scores
        baseline = original_scores
    else:
        scores = working[score_column].astype(float).tolist()
        baseline = scores

    predictions = [1 if score >= threshold else 0 for score in scores]
    return {
        "rows": float(len(working)),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "auc": _safe_auc(labels, scores),
        "baseline_auc": _safe_auc(labels, baseline),
    }


def evaluate_by_origin(
    df: pd.DataFrame,
    score_column: str,
    origin_column: str = "origin",
    label_column: str = "label",
    threshold: float = 0.5,
    use_discourse_reweighting: bool = False,
    alpha: float = 1.0,
    depth_factor: float = 1.0,
) -> pd.DataFrame:
    """Evaluate one score column separately for each dataset split."""
    rows = []
    for origin, split_df in split_frames_by_origin(df, origin_column=origin_column).items():
        metrics = evaluate_score_column(
            df=split_df,
            score_column=score_column,
            label_column=label_column,
            threshold=threshold,
            use_discourse_reweighting=use_discourse_reweighting,
            alpha=alpha,
            depth_factor=depth_factor,
        )
        rows.append({"origin": origin, **metrics})
    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge multiple feature CSVs into one file.")
    merge_parser.add_argument("--base", required=True, help="Base CSV with shared metadata columns.")
    merge_parser.add_argument("--features", nargs="+", required=True, help="Feature CSVs to merge.")
    merge_parser.add_argument("--output", required=True, help="Where to write the merged CSV.")
    merge_parser.add_argument("--merge-keys", nargs="+", help="Explicit merge keys.")
    merge_parser.add_argument("--origin-column", default="origin", help="Dataset split column.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate one score column in a merged CSV.")
    eval_parser.add_argument("--input", required=True, help="Merged CSV to evaluate.")
    eval_parser.add_argument("--score-column", required=True, help="Score column to evaluate.")
    eval_parser.add_argument("--label-column", default="label", help="Binary label column.")
    eval_parser.add_argument("--origin-column", default="origin", help="Dataset split column.")
    eval_parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    eval_parser.add_argument(
        "--use-discourse-reweighting",
        action="store_true",
        help="Treat the score column as a serialized sentence-score list and apply StructScore reweighting.",
    )
    eval_parser.add_argument("--alpha", type=float, default=1.0, help="Reweighting alpha.")
    eval_parser.add_argument("--depth-factor", type=float, default=1.0, help="Depth factor.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "merge":
        merged = merge_feature_files(
            base_path=args.base,
            feature_paths=args.features,
            output_path=args.output,
            merge_keys=args.merge_keys,
            origin_column=args.origin_column,
        )
        print(f"Wrote {len(merged)} rows and {len(merged.columns)} columns to {args.output}")
        return

    if args.command == "evaluate":
        df = load_csv(args.input, origin_column=args.origin_column)
        results = evaluate_by_origin(
            df=df,
            score_column=args.score_column,
            origin_column=args.origin_column,
            label_column=args.label_column,
            threshold=args.threshold,
            use_discourse_reweighting=args.use_discourse_reweighting,
            alpha=args.alpha,
            depth_factor=args.depth_factor,
        )
        print(results.to_string(index=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
