import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot average simulation completion time and normalized carbon "
            "usage by scheduler."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to carbon_power_models.csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the output plot",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()

    required_columns = {
        "experiment",
        "scheme",
        "power_model",
        "total_carbon_usage",
        "total_time_ms",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["experiment"] = pd.to_numeric(df["experiment"], errors="coerce")
    if "status" in df.columns:
        invalid_experiments = df.loc[
            df["status"].astype(str).str.strip().str.lower() == "invalid",
            "experiment",
        ].dropna().unique()
        if len(invalid_experiments):
            df = df[~df["experiment"].isin(invalid_experiments)].copy()
            print(
                "Ignoring experiments with invalid status: "
                f"{', '.join(map(str, invalid_experiments))}"
            )

    df["scheme"] = df["scheme"].astype(str).str.strip()
    df["power_model"] = df["power_model"].astype(str).str.strip()
    df["total_carbon_usage"] = pd.to_numeric(
        df["total_carbon_usage"], errors="coerce"
    )
    df["total_time_ms"] = pd.to_numeric(
        df["total_time_ms"], errors="coerce"
    )
    df = df.dropna(
        subset=[
            "experiment",
            "scheme",
            "power_model",
            "total_carbon_usage",
            "total_time_ms",
        ]
    ).copy()

    if df.empty:
        raise ValueError("No valid rows remain after filtering the CSV.")

    power_models = df["power_model"].drop_duplicates().tolist()
    ncols = min(3, len(power_models))
    nrows = math.ceil(len(power_models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, power_model in zip(axes, power_models):
        model_data = df[df["power_model"] == power_model]
        averages = (
            model_data.groupby("scheme")[
                ["total_time_ms", "total_carbon_usage"]
            ]
            .mean()
        )
        if "spark_fifo" not in averages.index:
            raise ValueError(
                f"No spark_fifo baseline rows found for {power_model}."
            )

        schemes = averages.index.tolist()
        times_hours = averages["total_time_ms"].to_numpy() / 3_600_000
        carbon_baseline = averages.loc["spark_fifo", "total_carbon_usage"]
        if carbon_baseline == 0:
            raise ValueError(
                f"spark_fifo carbon usage is zero for {power_model}; "
                "cannot normalize carbon usage."
            )
        normalized_carbon = (
            averages["total_carbon_usage"].to_numpy() / carbon_baseline
        )

        x = np.arange(len(schemes))
        width = 0.38
        time_bars = ax.bar(
            x - width / 2,
            times_hours,
            width,
            label="Completion time (hours)",
            color="cornflowerblue",
        )
        ax.set_ylabel("Completion time (hours)")
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, rotation=35, ha="right")
        ax.set_xlabel("Scheduler")

        carbon_ax = ax.twinx()
        carbon_bars = carbon_ax.bar(
            x + width / 2,
            normalized_carbon,
            width,
            label="Carbon usage (normalized to Spark FIFO)",
            color="mediumseagreen",
        )
        carbon_ax.set_ylabel("Carbon usage (normalized to Spark FIFO)")
        ax.set_title(power_model)

        ax.legend(
            [time_bars, carbon_bars],
            [
                "Completion time (hours)",
                "Carbon usage (normalized to Spark FIFO)",
            ],
            loc="upper left",
            fontsize=8,
        )

    for ax in axes[len(power_models):]:
        ax.set_visible(False)

    fig.suptitle(
        "End-to-end completion time and carbon usage by scheduler",
        fontsize=14,
    )
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main()
