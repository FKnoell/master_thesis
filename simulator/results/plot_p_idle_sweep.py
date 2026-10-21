import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to carbon_power_models.csv")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output plot (e.g., results/p_idle_savings_all.png)")
    parser.add_argument("--baseline", type=str, default="spark_fifo",
                        help="Baseline scheduler to compute savings against (default: spark_fifo)")
    args = parser.parse_args()

    # Load results
    df = pd.read_csv(args.csv_path)

    # Aggregate over experiments (mean per scheme + power_model)
    agg = df.groupby(["scheme", "power_model", "pidle", "pdyn"]).agg({
        "total_carbon_usage": "mean"
    }).reset_index()

    # Get baseline carbon (e.g., spark_fifo) per power_model
    baseline = agg[agg["scheme"] == args.baseline][["power_model", "total_carbon_usage"]].copy()
    baseline.rename(columns={"total_carbon_usage": "carbon_baseline"}, inplace=True)

    # Merge baseline into all rows
    merged = agg.merge(baseline, on="power_model")

    # Compute carbon savings vs baseline:
    # savings = 1 - (C_scheduler / C_baseline)
    merged["ratio"] = merged["total_carbon_usage"] / merged["carbon_baseline"]
    merged["savings"] = 1.0 - merged["ratio"]
    merged["savings_percent"] = merged["savings"] * 100.0

    # Sort for plotting
    merged = merged.sort_values("pidle")

    # Plot settings
    plt.figure(figsize=(10, 6))

    schemes = merged["scheme"].unique()
    markers = ["o", "s", "^", "D", "x", "+", "v", "<", ">", "p", "*"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))

    for idx, scheme in enumerate(schemes):
        sub = merged[merged["scheme"] == scheme]
        x = sub["pidle"].to_numpy()
        y = sub["savings_percent"].to_numpy()

        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        plt.plot(x, y, marker=marker, markersize=4, linewidth=1.5,
                 label=scheme, color=color)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    plt.xlabel(r"$P_{\mathrm{idle}} / (P_{\mathrm{idle}} + P_{\mathrm{dyn}})$")
    plt.ylabel(r"Carbon savings vs. {} (\%)".format(args.baseline))
    plt.title(r"Effect of $P_{\mathrm{idle}}$ share on carbon savings (all schedulers)")
    plt.legend(title="Scheduler", fontsize=9, title_fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    plt.savefig(args.output_path, dpi=300)
    plt.close()

    print(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main()