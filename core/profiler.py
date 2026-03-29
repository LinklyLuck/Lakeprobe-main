"""
LakeProbe — PartB: Profiler

Functions:
Scan CSV files and generate ProfileCard objects.
Minimize dependence on LLMs; everything here is based on computable facts.
Technology choice: DuckDB first, with Pandas as fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.models import ColumnProfile, ColumnRole, ProfileCard
from config import PROFILE_DIR


#Role inference rules

# Common keywords for time columns
_TIME_KEYWORDS = {"date", "time", "day", "month", "year", "quarter",
                  "timestamp", "created", "updated", "dt", "period"}

# Common keywords for ID columns
_ID_KEYWORDS = {"id", "key", "code", "uuid", "pk", "fk", "index", "no", "number"}

# PII keywords
_PII_KEYWORDS = {"email", "phone", "ssn", "address", "passport",
                 "credit_card", "ip_address", "name"}


def _infer_role(col_name: str, dtype: str, unique_rate: float,
                n_unique: int, row_count: int) -> ColumnRole:
    """Infer the column role based on column name, type, and cardinality."""
    name_lower = col_name.lower().replace("_", " ").replace("-", " ")
    tokens = set(name_lower.split())

    # Time columns
    if "datetime" in dtype or "date" in dtype:
        return ColumnRole.TIME
    if tokens & _TIME_KEYWORDS:
        return ColumnRole.TIME

    # ID columns
    if tokens & _ID_KEYWORDS:
        return ColumnRole.IDENTIFIER

    # Numeric columns → measure vs dimension
    if dtype in ("int64", "float64", "int32", "float32", "number"):
        # High-cardinality numeric → measure
        if unique_rate > 0.5 and n_unique > 20:
            return ColumnRole.MEASURE
        # Low-cardinality numeric → dimension (e.g., year=2021,2022,2023)
        if n_unique <= 20:
            return ColumnRole.DIMENSION
        return ColumnRole.MEASURE

    # String columns
    if dtype in ("object", "string", "str"):
        if unique_rate > 0.8 and n_unique > 100:
            return ColumnRole.TEXT
        return ColumnRole.DIMENSION

    # bool
    if dtype == "bool":
        return ColumnRole.DIMENSION

    return ColumnRole.UNKNOWN


def _detect_pii(col_name: str) -> bool:
    tokens = set(col_name.lower().replace("_", " ").replace("-", " ").split())
    return bool(tokens & _PII_KEYWORDS)


# DuckDB profiler

def _profile_with_duckdb(csv_path: str) -> ProfileCard:
    import duckdb

    dataset_id = Path(csv_path).stem
    conn = duckdb.connect()

    # Read CSV
    conn.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{csv_path}')")
    row_count = conn.execute("SELECT count(*) FROM data").fetchone()[0]
    col_info = conn.execute("PRAGMA table_info('data')").fetchall()

    columns: list[ColumnProfile] = []
    for col_row in col_info:
        cname = col_row[1]
        ctype = col_row[2].lower()

        # Basic statistics
        stats = conn.execute(f"""
            SELECT
                count(*) - count("{cname}") as n_miss,
                count(DISTINCT "{cname}") as n_unique
            FROM data
        """).fetchone()
        n_miss = stats[0]
        n_unique = stats[1]
        missing_rate = n_miss / row_count if row_count else 0
        unique_rate = n_unique / row_count if row_count else 0

        # min/max/mean
        min_val = max_val = mean_val = None
        dtype_str = _normalize_dtype(ctype)
        if dtype_str in ("int64", "float64"):
            try:
                mm = conn.execute(f'SELECT min("{cname}"), max("{cname}"), avg("{cname}") FROM data').fetchone()
                min_val, max_val, mean_val = mm[0], mm[1], float(mm[2]) if mm[2] is not None else None
            except Exception:
                pass

        # top-k values
        top_values = []
        try:
            rows = conn.execute(f"""
                SELECT "{cname}", count(*) as cnt
                FROM data WHERE "{cname}" IS NOT NULL
                GROUP BY "{cname}" ORDER BY cnt DESC LIMIT 5
            """).fetchall()
            top_values = [r[0] for r in rows]
        except Exception:
            pass

        # sample values
        sample_values = []
        try:
            rows = conn.execute(f'SELECT DISTINCT "{cname}" FROM data WHERE "{cname}" IS NOT NULL LIMIT 8').fetchall()
            sample_values = [r[0] for r in rows]
        except Exception:
            pass

        # Equal-frequency histogram (numeric columns) + value frequency table (categorical columns)
        histogram = []
        value_counts = {}
        try:
            if dtype_str in ("int64", "float64") and n_unique > 10:
                # Equal-frequency histogram: 10 buckets
                n_buckets = min(10, n_unique)
                hist_rows = conn.execute(f"""
                    WITH ranked AS (
                        SELECT "{cname}" as v,
                               NTILE({n_buckets}) OVER (ORDER BY "{cname}") as bucket
                        FROM data WHERE "{cname}" IS NOT NULL
                    )
                    SELECT bucket, MIN(v), MAX(v), COUNT(*) as cnt
                    FROM ranked GROUP BY bucket ORDER BY bucket
                """).fetchall()
                histogram = [(float(r[1]), float(r[2]), int(r[3])) for r in hist_rows]
            elif dtype_str == "object" and n_unique <= 200:
                # Categorical columns: full value frequency table
                vc_rows = conn.execute(f"""
                    SELECT "{cname}", COUNT(*) as cnt
                    FROM data WHERE "{cname}" IS NOT NULL
                    GROUP BY "{cname}" ORDER BY cnt DESC LIMIT 200
                """).fetchall()
                value_counts = {str(r[0]): int(r[1]) for r in vc_rows}
        except Exception:
            pass

        role = _infer_role(cname, dtype_str, unique_rate, n_unique, row_count)
        pii = _detect_pii(cname)

        columns.append(ColumnProfile(
            name=cname,
            dtype=dtype_str,
            missing_rate=round(missing_rate, 4),
            unique_rate=round(unique_rate, 4),
            n_unique=n_unique,
            min_val=min_val,
            max_val=max_val,
            mean_val=round(mean_val, 4) if mean_val is not None else None,
            top_values=top_values[:5],
            sample_values=sample_values[:8],
            inferred_role=role,
            pii_risk=pii,
            histogram=histogram,
            value_counts=value_counts,
        ))

    conn.close()

    return ProfileCard(
        dataset_id=dataset_id,
        file_path=csv_path,
        row_count=row_count,
        col_count=len(columns),
        columns=columns,
    )


# Pandas fallback

def _profile_with_pandas(csv_path: str) -> ProfileCard:
    import pandas as pd

    dataset_id = Path(csv_path).stem
    df = pd.read_csv(csv_path, low_memory=False)
    row_count = len(df)

    columns: list[ColumnProfile] = []
    for cname in df.columns:
        series = df[cname]
        dtype_str = _normalize_dtype(str(series.dtype))
        n_miss = int(series.isna().sum())
        n_unique = int(series.nunique())
        missing_rate = n_miss / row_count if row_count else 0
        unique_rate = n_unique / row_count if row_count else 0

        min_val = max_val = mean_val = None
        if dtype_str in ("int64", "float64"):
            try:
                min_val = float(series.min())
                max_val = float(series.max())
                mean_val = float(series.mean())
            except Exception:
                pass

        top_values = series.value_counts().head(5).index.tolist()
        sample_values = series.dropna().unique()[:8].tolist()

        role = _infer_role(cname, dtype_str, unique_rate, n_unique, row_count)
        pii = _detect_pii(cname)

        columns.append(ColumnProfile(
            name=cname,
            dtype=dtype_str,
            missing_rate=round(missing_rate, 4),
            unique_rate=round(unique_rate, 4),
            n_unique=n_unique,
            min_val=min_val,
            max_val=max_val,
            mean_val=round(mean_val, 4) if mean_val is not None else None,
            top_values=top_values[:5],
            sample_values=[str(v) for v in sample_values[:8]],
            inferred_role=role,
            pii_risk=pii,
        ))

    return ProfileCard(
        dataset_id=dataset_id,
        file_path=csv_path,
        row_count=row_count,
        col_count=len(columns),
        columns=columns,
    )


def _normalize_dtype(raw: str) -> str:
    """Normalize different dtype representations into a unified format."""
    raw = raw.lower().strip()
    if "int" in raw:
        return "int64"
    if "float" in raw or "double" in raw or "decimal" in raw or "numeric" in raw:
        return "float64"
    if "bool" in raw:
        return "bool"
    if "date" in raw or "time" in raw:
        return "datetime64"
    if raw in ("varchar", "text", "string", "object"):
        return "object"
    return raw


# Public entry point

def build_profile_card(csv_path: str, save: bool = True) -> ProfileCard:
    """
    Main profiler entry: scan CSV → ProfileCard

    DuckDB is used first, with Pandas as fallback.
    """
    try:
        card = _profile_with_duckdb(csv_path)
    except Exception:
        card = _profile_with_pandas(csv_path)

    if save:
        out_path = PROFILE_DIR / f"{card.dataset_id}.json"
        out_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

    return card


def load_profile_card(dataset_id: str) -> ProfileCard | None:
    """Load an existing ProfileCard from disk."""
    path = PROFILE_DIR / f"{dataset_id}.json"
    if not path.exists():
        return None
    return ProfileCard.model_validate_json(path.read_text(encoding="utf-8"))


def profile_all_csvs(csv_dir: str | Path | None = None) -> list[ProfileCard]:
    """Batch-profile all CSV files in a directory."""
    from config import CSV_DIR
    d = Path(csv_dir) if csv_dir else CSV_DIR
    cards = []
    for f in sorted(d.glob("*.csv")):
        cards.append(build_profile_card(str(f)))
    return cards
