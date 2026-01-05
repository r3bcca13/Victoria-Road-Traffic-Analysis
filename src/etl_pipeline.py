# Import libraries

import io
from pathlib import Path
import zipfile
import pandas as pd

# Select sites of interest

site_df = pd.read_csv("../data/traffic_sites/traffic_site_info.csv")

# Select suburbs in which sites are located
selected_suburbs = ["East Melbourne", "Richmond", "Cremorne", "Jolimont", "Melbourne", "South Yarra",
                    "Southbank", "South Melbourne", "Fitzroy",  "Collingwood"]

selected_sites = site_df[site_df["suburb"].isin(selected_suburbs)]["site_id"].values

# Define functions

def process_csv(csv_file, selected_suburbs):
    
    # Read CSV file and change datatypes
    df = pd.read_csv(csv_file, dtype={"NB_SCATS_SITE": "int16",
                                      "NB_DETECTOR": "int8",
                                      "CT_RECORDS": "int8"})
    
    # Filter sites in selected suburbs
    df = df[df["NB_SCATS_SITE"].isin(selected_sites)]

    # Drop irrelavant columns
    df = df.drop(columns=["NM_REGION", "QT_VOLUME_24HOUR", "CT_ALARM_24HOUR"])

    # Rename column names
    df = df.rename(columns={"NB_SCATS_SITE": "site_id", 
                            "QT_INTERVAL_COUNT": "datetime",
                            "NB_DETECTOR": "detector_id", 
                            "CT_RECORDS": "working_period_count"})
    
    # Remove rows with all non-positive volume values
    volume_cols = [c for c in df.columns if c.startswith("V")]
    df = df[(df[volume_cols] > 0).any(axis=1)]

    # Replace NaN and negative volumes with zeros and adjust working period count
    df = df.fillna(0)

    neg_mask = df[volume_cols] < 0
    df["working_period_count"] -= neg_mask.sum(axis=1)
    df[volume_cols] = df[volume_cols].mask(neg_mask, 0)

    # Convert other datatypes
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[volume_cols] = df[volume_cols].astype("int16")

    # Sum volume columns by hour
    hourly_volume_df = pd.DataFrame({f"hour_{h+1}": df[volume_cols[h*4:(h+1)*4]].sum(axis=1) for h in range(24)})
    df = pd.concat([df.drop(columns=volume_cols), hourly_volume_df], axis=1)

    # Transform to long format
    df = df.melt(
        id_vars=["site_id", "datetime", "detector_id", "working_period_count"],
        value_vars=hourly_volume_df.columns,
        var_name="hour",
        value_name="volume")

    return df[["datetime", "site_id", "detector_id", "volume", "working_period_count"]]

# Clean traffic volume data

DATA_DIR = Path("../data/traffic_volume")

for zip_path in sorted(DATA_DIR.glob("*.zip")):

    year = str(zip_path)[-8:-4]
    dfs = []

    with zipfile.ZipFile(zip_path) as zip_1:
        for i, zip_name in enumerate(sorted(zip_1.namelist())):

            with zip_1.open(zip_name) as zip_file:
                with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zip_2:

                    for j, csv_name in enumerate(sorted(zip_2.namelist())):
                        if csv_name.lower().endswith(".csv"):
                            with zip_2.open(csv_name) as csv_file:
                                
                                df = process_csv(csv_file, selected_sites)

                                dfs.append(df.iloc[1:] if i > 0 else df)
                                print(f"Cleaned and added {csv_name} successfully.")

    # Save cleaned DataFrame
    traffic_df = pd.concat(dfs, ignore_index=True)
    traffic_df.to_csv(f"../cleaned_data/traffic_volume_{year}parquet", index=False)
    print(f"Saved traffic_volume_{year}.parquet successfully.")

# Check schema for improper values and datatypes

SCHEMA = {
    "datetime": {
        "dtype": "datetime",
        "allow_na": False
    },
    "site_id": {
        "dtype": "int",
        "allow_na": False
    },
    "detector_id": {
        "dtype": "int",
        "allow_na": False
    },
    "volume": {
        "columns": lambda df: df.filter(regex="^hour").columns,
        "dtype": "int",
        "min": 0,
        "allow_na": False,
    },
    "working_period_count": {
        "dtype": "int",
        "min": 0,
        "allow_na": False,
    }
}

def validate_schema(df, schema):
    errors = []

    for name, rules in schema.items():

        if name not in df.columns:
            errors.append(f"Missing column: {name}")
            continue

        s = df[name]

        if not rules.get("allow_na", True) and s.isna().any():
            errors.append(f"{name}: contains NaN values")

        if rules.get("dtype") == "int" and not pd.api.types.is_integer_dtype(s):
            errors.append(f"{name}: not integer dtype")

        if rules.get("dtype") == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            errors.append(f"{name}: not datetime dtype")

        if "min" in rules and (s < rules["min"]).any():
            errors.append(f"{name}: contains values < {rules['min']}")

        if "max" in rules and (s > rules["max"]).any():
            errors.append(f"{name}: contains values > {rules['max']}")

    return errors

errors = validate_schema(df, SCHEMA)

if errors:
    raise ValueError("Schema validation failed:\n" + "\n".join(errors))