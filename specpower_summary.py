# Copyright (c) 2026 Fynn [YOUR SURNAME]
#
# This file was developed as part of a master's thesis.
# It contains thesis-specific work and analysis.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path(
    "specpower_results.csv"
)

OUTPUT_PDF = Path(
    "specpower_results_summary.pdf"
)


POWER_COLUMNS = [
    "P_idle_W",
    "P_dyn_W",
    "P_peak_W",
    "ratio_Pidle_Ppeak_percent",
]


def convert_numeric_columns(data):
    """
    Convert the required power-model columns to numeric values.
    """
    for column in POWER_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def format_value(value):
    """
    Format values for display in the PDF table.
    """
    if pd.isna(value):
        return "n/a"

    return f"{value:.2f}"


def shorten_text(value, max_length=42):
    """
    Shorten long case descriptions for better readability.
    """
    value = str(value)

    if len(value) <= max_length:
        return value

    return value[:max_length - 3] + "..."


def create_pdf_visualization(summary):
    """
    Create and save the table as a PDF.

    The system column is intentionally excluded.
    """
    display_data = pd.DataFrame()

    display_data["Case"] = summary["case"].apply(
        shorten_text,
        max_length=42,
    )

    display_data["P_idle (W)"] = summary[
        "P_idle_W"
    ].apply(format_value)

    display_data["P_dyn (W)"] = summary[
        "P_dyn_W"
    ].apply(format_value)

    display_data["P_peak (W)"] = summary[
        "P_peak_W"
    ].apply(format_value)

    display_data["P_idle/P_peak (%)"] = summary[
        "ratio_Pidle_Ppeak_percent"
    ].apply(format_value)

    column_labels = list(display_data.columns)

    figure, axis = plt.subplots(
        figsize=(14, 7),
    )

    axis.axis("off")

    table = axis.table(
        cellText=display_data.values,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[
            0.44,
            0.14,
            0.14,
            0.14,
            0.18,
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.0)

    # Header styling.
    for column_index in range(
        len(column_labels)
    ):
        header_cell = table[
            0,
            column_index,
        ]

        header_cell.set_facecolor(
            "#1f4e78"
        )

        header_cell.set_text_props(
            color="white",
            weight="bold",
        )

    # Alternating row colors.
    for row_index in range(
        1,
        len(display_data) + 1,
    ):
        row_color = (
            "#eaf2f8"
            if row_index % 2 == 0
            else "#ffffff"
        )

        for column_index in range(
            len(column_labels)
        ):
            table[
                row_index,
                column_index,
            ].set_facecolor(row_color)

    # Highlight the average row.
    for column_index in range(
        len(column_labels)
    ):
        table[
            1,
            column_index,
        ].set_facecolor("#d9ead3")

        table[
            1,
            column_index,
        ].set_text_props(weight="bold")

    axis.set_title(
        "SPECpower Seven-Case Power Summary",
        fontsize=16,
        weight="bold",
        pad=20,
    )

    figure.tight_layout()

    # Save only the PDF.
    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE.resolve()}"
        )

    print(f"Reading: {INPUT_FILE.resolve()}")

    data = pd.read_csv(INPUT_FILE)

    data = convert_numeric_columns(data)

    required_columns = [
        "P_idle_W",
        "P_dyn_W",
        "P_peak_W",
        "ratio_Pidle_Ppeak_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    data = data.dropna(
        subset=required_columns
    ).copy()

    if data.empty:
        raise ValueError(
            "No complete valid rows were found."
        )

    # Case 1: average of all valid results.
    average_row = {
        "case": (
            "1 - Average of all valid results"
        ),
    }

    for column in POWER_COLUMNS:
        average_row[column] = data[column].mean()

    # Cases 2–7.
    #
    # Lower values appear before higher values within
    # each category.
    cases = [
        (
            "2 - Lowest P_idle/P_peak ratio",
            data["ratio_Pidle_Ppeak_percent"].idxmin(),
        ),
        (
            "3 - Highest P_idle/P_peak ratio",
            data["ratio_Pidle_Ppeak_percent"].idxmax(),
        ),
        (
            "4 - Lowest P_peak",
            data["P_peak_W"].idxmin(),
        ),
        (
            "5 - Highest P_peak",
            data["P_peak_W"].idxmax(),
        ),
        (
            "6 - Lowest P_idle",
            data["P_idle_W"].idxmin(),
        ),
        (
            "7 - Highest P_idle",
            data["P_idle_W"].idxmax(),
        ),
    ]

    summary_rows = [average_row]

    for case_name, row_index in cases:
        selected_row = data.loc[row_index]

        result_row = {
            "case": case_name,
        }

        for column in POWER_COLUMNS:
            result_row[column] = selected_row[column]

        summary_rows.append(result_row)

    summary = pd.DataFrame(summary_rows)

    create_pdf_visualization(summary)

    print("\nPDF visualization created successfully.")
    print(f"PDF file: {OUTPUT_PDF.resolve()}")

    print("\nReadable summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()