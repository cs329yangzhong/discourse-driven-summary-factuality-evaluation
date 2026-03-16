"""Run a lightweight StructScore check across one or more dataset splits."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


DEFAULT_DATA_PATH = "data/diversumm_test.csv"
DEFAULT_ORIGINS = ["multinews", "qmsum", "GovReport", "arXiv", "ChemSum"]
DEFAULT_SYSTEMS = [
    "segment_0_align_score",
    # "segment_1_align_score",
    # "segment_minicheck_lvl_1",
    # "segment_minicheck_lvl_2",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="CSV file to evaluate.")
    parser.add_argument("--origin", help="Optional dataset split to filter.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        help="Score-list columns to evaluate.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="StructScore alpha.")
    parser.add_argument("--depth-factor", type=float, default=1.0, help="Depth factor.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    return parser


def format_macro_scores(scores: list[float]) -> str:
    return f"{scores} {float(np.mean(scores))}"


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        from . import discourse_utils as du
    except ImportError:  # pragma: no cover - supports running as a script
        try:
            import discourse_utils as du
        except ImportError as exc:
            if getattr(exc, "name", None) == "fuzzywuzzy":
                raise RuntimeError(
                    "This script requires the `fuzzywuzzy` package to be installed."
                ) from exc
            raise

    du.np = np

    df = pd.read_csv(args.data_path)
    available_origins = set(df["origin"].dropna().unique().tolist())
    if args.origin:
        origins = [args.origin]
    else:
        origins = [origin for origin in DEFAULT_ORIGINS if origin in available_origins]

    for system in args.systems:
        original_auc_scores = []
        weighted_auc_scores = []

        print(system)
        print(f"origins = {origins}")

        for origin in origins:
            origin_df = df[df.origin == origin].reset_index(drop=True)
            if origin_df.empty:
                raise ValueError(f"No rows found for origin={origin!r} in {args.data_path}")

            weighted_scores = []
            original_scores = []

            for _, row in origin_df.iterrows():
                score_list = du.process_sent_score(row[system])
                weighted_mean, original_mean, _, _, _ = du.reweight_score(
                    score_list,
                    row,
                    alpha=args.alpha,
                    depth_factor=args.depth_factor,
                )
                weighted_scores.append(weighted_mean)
                original_scores.append(original_mean)

            labels = origin_df["label"].tolist()
            original_auc_scores.append(round(roc_auc_score(labels, original_scores) * 100, 2))
            weighted_auc_scores.append(round(roc_auc_score(labels, weighted_scores) * 100, 2))

        print(f"old: {format_macro_scores(original_auc_scores)}")
        print(f"reweighted: {format_macro_scores(weighted_auc_scores)}")
        print()


if __name__ == "__main__":
    main()
