from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "carbon_power_models.csv"
OUTPUT_PATH = RESULTS_DIR / "carbon_improvement_by_pidle.png"

SELECTED_PIDLE = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]


if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find:\n{CSV_PATH}")


df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

required_columns = {
    "experiment",
    "scheme",
    "power_model",
    "pidle",
    "total_carbon_usage",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )

# Clean data.
df["scheme"] = df["scheme"].astype(str).str.strip().str.lower()
df["power_model"] = df["power_model"].astype(str).str.strip()
df["experiment"] = pd.to_numeric(df["experiment"], errors="coerce")
df["pidle"] = pd.to_numeric(df["pidle"], errors="coerce")
df["total_carbon_usage"] = pd.to_numeric(
    df["total_carbon_usage"], errors="coerce"
)

df = df.dropna(
    subset=[
        "experiment",
        "scheme",
        "power_model",
        "pidle",
        "total_carbon_usage",
    ]
).copy()

# Avoid floating-point problems when selecting pidle values.
df["pidle_key"] = df["pidle"].round(8)
selected_keys = [round(value, 8) for value in SELECTED_PIDLE]
df = df[df["pidle_key"].isin(selected_keys)].copy()

if df.empty:
    raise ValueError("None of the selected pidle values were found.")

# Spark FIFO baseline for the same experiment, power model, and pidle.
baseline = (
    df[df["scheme"] == "spark_fifo"]
    [["experiment", "power_model", "pidle_key", "total_carbon_usage"]]
    .rename(columns={"total_carbon_usage": "spark_fifo_carbon"})
)

comparison = df[df["scheme"] != "spark_fifo"].merge(
    baseline,
    on=["experiment", "power_model", "pidle_key"],
    how="inner",
)

if comparison.empty:
    raise ValueError("No matching Spark FIFO baseline rows were found.")

# Calculate the improvement for each individual experiment and pidle value.
comparison["carbon_improvement"] = (
    (comparison["spark_fifo_carbon"] - comparison["total_carbon_usage"])
    / comparison["spark_fifo_carbon"]
    * 100
)

comparison = comparison.replace(
    [float("inf"), float("-inf")], pd.NA
).dropna(subset=["carbon_improvement"])

labels = {
    "cap_decima": "CAP DECIMA",
    "cap_fifo": "CAP FIFO",
    "cap_fifo_better": "CAP FIFO improved",
    "cap_fifo_backfill": "CAP FIFO backfill",
    "cap_partition": "CAP partition",
    "pcaps": "PCAPS",
    "spark_fifo_better": "Spark FIFO improved",
}

comparison["scheduler"] = (
    comparison["scheme"].map(labels).fillna(comparison["scheme"])
)

scheduler_order = [
    "CAP DECIMA",
    "CAP FIFO",
    "CAP FIFO improved",
    "CAP FIFO backfill",
    "CAP partition",
    "PCAPS",
    "Spark FIFO improved",
]
scheduler_order = [
    item for item in scheduler_order
    if item in comparison["scheduler"].unique()
]

# Create labels in the requested order.
pidle_labels = [f"{value:g}" for value in SELECTED_PIDLE]
pidle_label_map = dict(zip(selected_keys, pidle_labels))
comparison["pidle_label"] = comparison["pidle_key"].map(pidle_label_map)

# Print the 75th percentile for every pidle and scheduler.
percentile_75 = (
    comparison
    .groupby(["pidle_label", "scheduler"])["carbon_improvement"]
    .quantile(0.75)
    .reset_index(name="percentile_75")
)

print("\n75th-percentile carbon improvement by pidle and scheduler:\n")
print(percentile_75.round(2).to_string(index=False))

# Create 9 subplots: one graph per pidle value.
sns.set_theme(style="whitegrid", context="paper")
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(21, 15),
    sharey=True,
)
axes = axes.flatten()

for index, pidle_label in enumerate(pidle_labels):
    ax = axes[index]
    plot_data = comparison[
        comparison["pidle_label"] == pidle_label
    ]

    # Each box contains the individual experiment values.
    sns.boxplot(
        data=plot_data,
        x="scheduler",
        y="carbon_improvement",
        order=scheduler_order,
        showfliers=False,
        width=0.65,
        ax=ax,
    )

    # Optional: show the individual experiment values lightly.
    sns.stripplot(
        data=plot_data,
        x="scheduler",
        y="carbon_improvement",
        order=scheduler_order,
        color="black",
        alpha=0.25,
        size=3,
        jitter=0.18,
        ax=ax,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(rf"$P_{{idle}} = {pidle_label}$")
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Carbon improvement compared with Spark FIFO (%)")
    ax.tick_params(axis="x", rotation=35)

fig.suptitle(
    "Carbon improvement compared with Spark FIFO\n"
    "Experiments grouped for each selected $P_{idle}$ value",
    fontsize=17,
    y=1.02,
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(f"\nSaved plot to:\n{OUTPUT_PATH}")