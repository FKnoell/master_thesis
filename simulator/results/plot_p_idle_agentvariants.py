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
                        help="Path to the agent variants CSV")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output filename prefix (e.g., results/p_idle_savings.png)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["scheme"] = df["agent"] + "," + df["variant"]

    agg = (
        df.groupby(["agent", "variant", "scheme", "power_model", "pidle", "pdyn"],
                   as_index=False)
        ["total_carbon_usage"]
        .mean()
    )

    output_dir = os.path.dirname(args.output_path) or "."
    output_stem, output_extension = os.path.splitext(os.path.basename(args.output_path))
    output_extension = output_extension or ".png"
    os.makedirs(output_dir, exist_ok=True)

    markers = ["o", "s", "^", "D", "x", "+", "v", "<", ">", "p", "*"]

    for agent in agg["agent"].unique():
        agent_data = agg[agg["agent"] == agent]
        baseline = agent_data[agent_data["variant"] == "default"][
            ["power_model", "pidle", "pdyn", "total_carbon_usage"]
        ].copy()
        if baseline.empty:
            raise ValueError(f"No default baseline found for agent '{agent}'")

        baseline.rename(columns={"total_carbon_usage": "carbon_baseline"}, inplace=True)
        merged = agent_data.merge(
            baseline, on=["power_model", "pidle", "pdyn"], validate="many_to_one"
        )

        merged["ratio"] = merged["total_carbon_usage"] / merged["carbon_baseline"]
        merged["savings_percent"] = (1.0 - merged["ratio"]) * 100.0
        merged = merged.sort_values("pidle")

        plt.figure(figsize=(10, 6))

        variants = merged["variant"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))

        for idx, variant in enumerate(variants):
            sub = merged[merged["variant"] == variant]
            plt.plot(
                sub["pidle"].to_numpy(),
                sub["savings_percent"].to_numpy(),
                marker=markers[idx % len(markers)],
                markersize=4,
                linewidth=1.5,
                label=variant,
                color=colors[idx % len(colors)],
            )

        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel(r"$P_{\mathrm{idle}} / (P_{\mathrm{idle}} + P_{\mathrm{dyn}})$")
        plt.ylabel(r"Carbon savings vs. default (\%)")
        plt.title(
            rf"Effect of $P_{{\mathrm{{idle}}}}$ share on carbon savings ({agent})"
        )
        plt.legend(title="Variant", fontsize=9, title_fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f"{output_stem}_{agent}{output_extension}"
        )
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
