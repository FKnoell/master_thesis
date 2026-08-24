# Copyright (c) 2026 Fynn-Leonard Knöll
#
# This file was developed as part of a master's thesis.
# It contains thesis-specific work and analysis.
 
import re
from pathlib import Path

import pandas as pd


URL = (
    "https://www.spec.org/power_ssj2008/"
    "results/power_ssj2008/"
)

OUTPUT_MODEL = Path(
    "specpower/specpower_results.csv"
)


def flatten_columns(columns):
    """
    Convert simple or multi-level column names into plain strings.
    """
    flattened = []

    for column in columns:
        if isinstance(column, tuple):
            parts = []

            for part in column:
                text = str(part).strip()

                if text and text.lower() != "nan":
                    parts.append(text)

            column_name = " | ".join(parts)

        else:
            column_name = str(column).strip()

        flattened.append(column_name)

    return flattened


def normalize_column_name(name):
    """
    Normalize whitespace and line breaks in a column name.
    """
    name = str(name)
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", " ", name)

    return name.strip().lower()


def find_column(columns, required_parts):
    """
    Find a column whose name contains all required text fragments.
    """
    normalized_columns = {
        column: normalize_column_name(column)
        for column in columns
    }

    for column, normalized_name in normalized_columns.items():
        if all(
            part.lower() in normalized_name
            for part in required_parts
        ):
            return column

    raise KeyError(
        "No matching column found for: "
        + ", ".join(required_parts)
    )


def clean_numeric_column(series):
    """
    Convert values such as '1,234', '276 W', or 'NC' to numbers.
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(
            r"[^0-9.\-]",
            "",
            regex=True,
        )
        .replace("", pd.NA)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def remove_repeated_header_rows(data):
    """
    Remove repeated table-header rows that may occur within
    the extracted HTML data.
    """
    header_markers = [
        "hardware vendor",
        "cpu description",
        "avg. watts",
    ]

    text_data = (
        data
        .fillna("")
        .astype(str)
    )

    row_text = (
        text_data
        .apply(
            lambda row: " | ".join(row.tolist()),
            axis=1,
        )
        .str.lower()
    )

    repeated_header = row_text.apply(
        lambda text: any(
            marker in text
            for marker in header_markers
        )
    )

    return data.loc[~repeated_header].copy()


def main():
    print("Downloading SPECpower results...")

    tables = pd.read_html(URL)

    if not tables:
        raise RuntimeError(
            "No HTML tables were found on the webpage."
        )

    print(f"HTML tables found: {len(tables)}")

    cleaned_tables = []

    for table in tables:
        table = table.copy()

        table.columns = flatten_columns(
            table.columns
        )

        table = remove_repeated_header_rows(
            table
        )

        if not table.empty:
            cleaned_tables.append(table)

    if not cleaned_tables:
        raise RuntimeError(
            "No data rows could be extracted."
        )

    data = pd.concat(
        cleaned_tables,
        ignore_index=True,
    )

    watts_100_column = find_column(
        data.columns,
        ["avg.", "watts", "@", "100%"],
    )

    active_idle_column = find_column(
        data.columns,
        ["avg.", "watts", "active", "idle"],
    )

    print("\nDetected power columns:")
    print(f"100% load:   {watts_100_column}")
    print(f"Active Idle: {active_idle_column}")

    data["P_idle_W"] = clean_numeric_column(
        data[active_idle_column]
    )

    data["P_peak_W"] = clean_numeric_column(
        data[watts_100_column]
    )

    valid = (
        data["P_idle_W"].notna()
        & data["P_peak_W"].notna()
        & (data["P_peak_W"] > 0)
        & (data["P_idle_W"] >= 0)
        & (data["P_idle_W"] <= data["P_peak_W"])
    )

    model_result = data.loc[valid].copy()

    model_result["P_dyn_W"] = (
        model_result["P_peak_W"]
        - model_result["P_idle_W"]
    )

    model_result["ratio_Pidle_Ppeak"] = (
        model_result["P_idle_W"]
        / model_result["P_peak_W"]
    )

    model_result["ratio_Pidle_Ppeak_percent"] = (
        model_result["ratio_Pidle_Ppeak"]
        * 100
    )

    model_result["ratio_Pdyn_Ppeak"] = (
        model_result["P_dyn_W"]
        / model_result["P_peak_W"]
    )

    model_result["ratio_Pdyn_Ppeak_percent"] = (
        model_result["ratio_Pdyn_Ppeak"]
        * 100
    )

    preferred_columns = [
        "Hardware Vendor",
        "Test Sponsor",
        "System Enclosure (if applicable)",
        "Nodes",
        "JVM Vendor",
        "Processor",
        "Total Memory (GB)",
        "CPU Description",
        "MHz",
        "Chips",
        "Cores",
        "Total Threads",
        "ssj_ops @ 100%",
        watts_100_column,
        active_idle_column,
        "P_idle_W",
        "P_dyn_W",
        "P_peak_W",
        "ratio_Pidle_Ppeak",
        "ratio_Pidle_Ppeak_percent",
        "ratio_Pdyn_Ppeak",
        "ratio_Pdyn_Ppeak_percent",
        "Result (Overall ssj_ops/watt)",
    ]

    selected_columns = [
        column
        for column in preferred_columns
        if column in model_result.columns
    ]

    calculated_columns = [
        "P_idle_W",
        "P_dyn_W",
        "P_peak_W",
        "ratio_Pidle_Ppeak",
        "ratio_Pidle_Ppeak_percent",
        "ratio_Pdyn_Ppeak",
        "ratio_Pdyn_Ppeak_percent",
    ]

    for column in calculated_columns:
        if column not in selected_columns:
            selected_columns.append(column)

    model_result = model_result[
        selected_columns
    ]

    model_result = model_result.sort_values(
        by="ratio_Pidle_Ppeak",
        ascending=False,
    )

    model_result.to_csv(
        OUTPUT_MODEL,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print("\nAnalysis completed.")
    print(
        f"Power model file: "
        f"{OUTPUT_MODEL.resolve()}"
    )
    print(
        f"Valid records: "
        f"{len(model_result)}"
    )

    print("\nFirst five results:")
    print(
        model_result[
            [
                "P_idle_W",
                "P_dyn_W",
                "P_peak_W",
                "ratio_Pidle_Ppeak",
                "ratio_Pidle_Ppeak_percent",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()