from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/traffic_volume")
SITE_INFO_PATH = Path("data/traffic_sites/traffic_site_info.csv")
OUTPUT_DIR = Path("cleaned_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SUBURBS = [
    "East Melbourne", "Richmond", "Cremorne", "Jolimont", "Melbourne", 
    "South Yarra", "Southbank", "South Melbourne", "Fitzroy", "Collingwood",
]

CSV_DTYPES = {
    "NB_SCATS_SITE": "int16",
    "NB_DETECTOR": "int8",
    "CT_RECORDS": "int8",
}

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------

def load_selected_sites(site_info_path: Path) -> pd.Series:
    """Load site metadata and return selected site IDs."""
    site_df = pd.read_csv(site_info_path)

    selected_sites = (
        site_df.loc[site_df["suburb"].isin(SELECTED_SUBURBS), "site_id"]
        .astype("int16")
    )

    logger.info("Selected %d traffic sites", len(selected_sites))
    return selected_sites


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------

def process_csv(csv_file, selected_sites: pd.Series) -> pd.DataFrame:
    """Clean and transform a single SCATS CSV file."""

    df = pd.read_csv(csv_file, dtype=CSV_DTYPES)

    # Filter sites of interest
    df = df[df["NB_SCATS_SITE"].isin(selected_sites)]

    # Drop irrelevant columns
    df = df.drop(
        columns=["NM_REGION", "QT_VOLUME_24HOUR", "CT_ALARM_24HOUR"],
        errors="ignore",
    )

    # Rename columns
    df = df.rename(
        columns={
            "NB_SCATS_SITE": "site_id",
            "QT_INTERVAL_COUNT": "datetime",
            "NB_DETECTOR": "detector_id",
            "CT_RECORDS": "working_period_count",
        }
    )

    volume_cols = [c for c in df.columns if c.startswith("V")]

    # Remove rows with no positive volumes
    df = df[(df[volume_cols] > 0).any(axis=1)]

    # Handle NaNs and negative values
    df = df.fillna(0)

    neg_mask = df[volume_cols] < 0
    df["working_period_count"] -= neg_mask.sum(axis=1)
    df[volume_cols] = df[volume_cols].mask(neg_mask, 0)

    # Type conversions
    df["datetime"] = pd.to_datetime(df["datetime"]).astype("datetime64[us]")
    df[volume_cols] = df[volume_cols].astype("int16")

    # Aggregate 15-min volumes → hourly
    hourly_volume = {
        f"hour_{h+1}": df[volume_cols[h * 4 : (h + 1) * 4]].sum(axis=1)
        for h in range(24)
    }

    hourly_df = pd.DataFrame(hourly_volume, index=df.index)

    df = pd.concat([df.drop(columns=volume_cols), hourly_df], axis=1)

    # Long format
    df = df.melt(
        id_vars=["site_id", "datetime", "detector_id", "working_period_count"],
        var_name="hour",
        value_name="volume",
    )

    return df[["datetime", "site_id", "detector_id", "volume", "working_period_count"]]


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def process_zip_file(zip_path: Path, selected_sites: pd.Series) -> pd.DataFrame:
    """Process a top-level yearly ZIP file."""
    dfs: List[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zip_1:
        for i, inner_zip_name in enumerate(sorted(zip_1.namelist())):

            with zip_1.open(inner_zip_name) as zip_bytes:
                with zipfile.ZipFile(io.BytesIO(zip_bytes.read())) as zip_2:

                    for csv_name in sorted(zip_2.namelist()):
                        if not csv_name.lower().endswith(".csv"):
                            continue

                        with zip_2.open(csv_name) as csv_file:
                            df = process_csv(csv_file, selected_sites)

                            # Avoid duplicated headers across files
                            dfs.append(df.iloc[1:] if i > 0 else df)

                            logger.info("Processed %s", csv_name)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

SCHEMA = {
    "datetime": {"dtype": "datetime", "allow_na": False},
    "site_id": {"dtype": "int", "allow_na": False},
    "detector_id": {"dtype": "int", "allow_na": False},
    "volume": {"dtype": "int", "min": 0, "allow_na": False},
    "working_period_count": {"dtype": "int", "min": 0, "allow_na": False},
}


def validate_schema(df: pd.DataFrame, schema: dict) -> None:
    """Validate dataframe against schema rules."""
    errors = []

    for col, rules in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue

        s = df[col]

        if not rules.get("allow_na", True) and s.isna().any():
            errors.append(f"{col}: contains NaN values")

        if rules.get("dtype") == "int" and not pd.api.types.is_integer_dtype(s):
            errors.append(f"{col}: not integer dtype")

        if rules.get("dtype") == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            errors.append(f"{col}: not datetime dtype")

        if "min" in rules and (s < rules["min"]).any():
            errors.append(f"{col}: contains values < {rules['min']}")

    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors))


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    selected_sites = load_selected_sites(SITE_INFO_PATH)

    for zip_path in sorted(DATA_DIR.glob("*.zip")):
        year = zip_path.stem[-4:]
        logger.info("Processing year %s", year)

        traffic_df = process_zip_file(zip_path, selected_sites)

        validate_schema(traffic_df, SCHEMA)

        output_path = OUTPUT_DIR / f"traffic_volume_{year}.parquet"
        traffic_df.to_parquet(output_path, index=False)

        logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()