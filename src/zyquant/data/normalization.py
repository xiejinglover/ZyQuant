from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from zyquant.core.exceptions import DataContractError

from .contracts import (
    DATE_COLUMNS,
    FIELD_SPECS,
    PRIMARY_KEYS,
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMNS,
)


def _coerce_column(table: str, column: str, series: pd.Series) -> pd.Series:
    spec = FIELD_SPECS[table][column]
    if pa.types.is_string(spec.arrow_type):
        converted = series.astype("string")
    elif pa.types.is_boolean(spec.arrow_type):
        invalid = series.notna() & ~series.isin([True, False, 0, 1])
        if invalid.any():
            raise DataContractError(f"{table}.{column} contains invalid booleans")
        converted = series.astype("boolean")
        if not spec.nullable:
            converted = converted.astype(bool)
    elif pa.types.is_integer(spec.arrow_type):
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = numeric.notna() & (numeric % 1 != 0)
        if invalid.any():
            raise DataContractError(f"{table}.{column} must contain whole numbers")
        converted = numeric.astype("Int64" if spec.nullable else "int64")
    elif pa.types.is_floating(spec.arrow_type):
        converted = pd.to_numeric(series, errors="coerce").astype(float)
    elif pa.types.is_date(spec.arrow_type):
        timestamps = pd.to_datetime(series, errors="coerce")
        converted = timestamps.map(
            lambda value: value.date() if pd.notna(value) else None
        ).astype(object)
    elif pa.types.is_timestamp(spec.arrow_type):
        converted = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        raise DataContractError(
            f"{table}.{column} uses unsupported type {spec.arrow_type}"
        )

    if not spec.nullable and converted.isna().any():
        raise DataContractError(f"{table}.{column} contains null or invalid values")
    if pa.types.is_floating(spec.arrow_type):
        finite = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise DataContractError(f"{table}.{column} must be finite")
    if spec.enum:
        values = set(converted.dropna().astype(str))
        unknown = values - set(spec.enum)
        if unknown:
            raise DataContractError(
                f"{table}.{column} contains unsupported values: {sorted(unknown)}"
            )
    if spec.minimum is not None and not (
        pa.types.is_date(spec.arrow_type)
        or pa.types.is_timestamp(spec.arrow_type)
        or pa.types.is_string(spec.arrow_type)
        or pa.types.is_boolean(spec.arrow_type)
    ):
        numeric = pd.to_numeric(converted, errors="coerce")
        if (numeric.dropna() < spec.minimum).any():
            raise DataContractError(
                f"{table}.{column} must be at least {spec.minimum}"
            )
        if not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise DataContractError(f"{table}.{column} must be finite")
    return converted


def normalize_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if name not in REQUIRED_COLUMNS:
        raise DataContractError(f"unsupported canonical table: {name}")
    result = frame.copy()
    missing = REQUIRED_COLUMNS[name] - set(result.columns)
    if missing:
        raise DataContractError(f"{name} missing required columns: {sorted(missing)}")
    unknown = set(result.columns) - set(FIELD_SPECS[name])
    if unknown:
        raise DataContractError(f"{name} contains unknown columns: {sorted(unknown)}")

    for column in tuple(result.columns):
        result[column] = _coerce_column(name, column, result[column])

    for column in DATE_COLUMNS[name]:
        if column not in result and FIELD_SPECS[name][column].required:
            raise DataContractError(f"{name} missing required date column: {column}")
    for column in TIMESTAMP_COLUMNS[name]:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)

    keys = list(PRIMARY_KEYS[name])
    if result.duplicated(keys).any():
        examples = result.loc[result.duplicated(keys, keep=False), keys].head(5)
        raise DataContractError(
            f"{name} contains duplicate keys: {examples.to_dict('records')}"
        )
    return result.sort_values(keys, ignore_index=True)


def normalize_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: normalize_table(name, frame) for name, frame in tables.items()}
