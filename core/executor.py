"""
LakeProbe — Executor
Function:
Executes operator plans and supports DuckDB (preferred) and Polars backends.
An operator plan is not SQL, but rather a structured sequence of ops;
the Executor is responsible for translating it into concrete execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.models import ExecutablePlan, OpType, PlanOp
from config import CSV_DIR, EXECUTOR_BACKEND, MAX_RESULT_ROWS


# DuckDB Executor
class DuckDBExecutor:
    """
    1.Compile the operator plan into DuckDB SQL and execute it (parameterized queries, injection protection).
    2.Use thread-local storage to maintain a separate DuckDB connection for each thread,
    3.thereby avoiding race conditions caused by multiple concurrent FastAPI requests sharing a single connection.
    """

    _local = __import__("threading").local()

    @classmethod
    def get_instance(cls):
        """Thread-local pattern — each thread gets its own DuckDB connection."""
        conn = getattr(cls._local, "conn", None)
        if conn is None:
            import duckdb
            cls._local.conn = duckdb.connect()
            cls._local.executor = cls.__new__(cls)
            cls._local.executor.conn = cls._local.conn
        return cls._local.executor

    def __init__(self):
        import duckdb
        self.conn = duckdb.connect()

    def execute(self, plan: ExecutablePlan) -> dict:
        #Execute the plan and return {columns, rows, row_count, sql}.
        sql_parts: list[str] = []
        scan_table = None
        scan_alias = "t1"
        join_clauses: list[str] = []
        join_counter = 2
        group_keys: list[str] = []
        agg_exprs: list[str] = []
        select_cols: list[str] = []
        filters: list[str] = []
        params: list = []
        sort_clause = ""
        limit_clause = ""
        derive_exprs: dict[str, str] = {}

        for step in plan.steps:
            op = step.op
            p = step.params

            if op == OpType.SCAN:
                dataset = p["dataset"]
                csv_path = self._resolve_csv(dataset)
                scan_table = f"read_csv_auto('{csv_path}') AS {scan_alias}"

            elif op == OpType.JOIN:
                right_dataset = p.get("right_dataset", "")
                left_key = p.get("left_key", "")
                right_key = p.get("right_key", "")
                join_type = p.get("join_type", "LEFT")
                if join_type.upper() not in ("LEFT", "RIGHT", "INNER", "OUTER", "CROSS"):
                    join_type = "LEFT"
                right_csv = self._resolve_csv(right_dataset)
                right_alias = f"t{join_counter}"
                join_counter += 1
                join_clauses.append(
                    f"{join_type} JOIN read_csv_auto('{right_csv}') AS {right_alias} "
                    f"ON {scan_alias}.\"{left_key}\" = {right_alias}.\"{right_key}\""
                )

            elif op == OpType.DERIVE_TIME:
                source = p["source"]
                target = p["target"]
                unit = p.get("unit", "year")
                if unit == "year":
                    derive_exprs[target] = f'EXTRACT(YEAR FROM CAST("{source}" AS DATE))'
                elif unit == "month":
                    derive_exprs[target] = f'EXTRACT(MONTH FROM CAST("{source}" AS DATE))'
                elif unit == "quarter":
                    derive_exprs[target] = f'EXTRACT(QUARTER FROM CAST("{source}" AS DATE))'
                else:
                    derive_exprs[target] = f'CAST("{source}" AS VARCHAR)'

            elif op == OpType.FILTER:
                col = p["column"]
                op_str = p.get("op", "=")
                val = p.get("value")
                if col in derive_exprs:
                    col_expr = derive_exprs[col]
                elif join_clauses:
                    col_expr = f'{scan_alias}."{col}"'
                else:
                    col_expr = f'"{col}"'
                filter_sql, filter_params = self._build_filter(col_expr, op_str, val)
                filters.append(filter_sql)
                params.extend(filter_params)

            elif op == OpType.GROUPBY:
                group_keys = p.get("keys", [])

            elif op == OpType.AGGREGATE:
                metric = p["metric"]
                func = p.get("func", "sum")
                if join_clauses:
                    agg_sql = self._build_agg(f'{scan_alias}."{metric}"', func, qualified=True)
                else:
                    agg_sql = self._build_agg(metric, func)
                agg_exprs.append(agg_sql)

            elif op == OpType.SORT:
                col = p.get("column", "")
                order = p.get("order", "desc")
                if order.upper() not in ("ASC", "DESC"):
                    order = "DESC"
                func_hint = None
                for agg_e in agg_exprs:
                    if col in agg_e:
                        func_hint = agg_e
                        break
                if func_hint:
                    sort_col = func_hint
                elif join_clauses:
                    sort_col = f'{scan_alias}."{col}"'
                else:
                    sort_col = f'"{col}"'
                sort_clause = f" ORDER BY {sort_col} {order.upper()}"

            elif op == OpType.LIMIT:
                n = p.get("n", MAX_RESULT_ROWS)
                limit_clause = f" LIMIT {min(int(n), MAX_RESULT_ROWS)}"

            elif op == OpType.SELECT:
                select_cols = p.get("columns", [])

        if not scan_table:
            return {"columns": [], "rows": [], "row_count": 0, "sql": "-- no scan op"}

        has_join = len(join_clauses) > 0

        def _col(name: str) -> str:
            if has_join:
                return f'{scan_alias}."{name}"'
            return f'"{name}"'

        if agg_exprs and group_keys:
            sel_items = [_col(k) for k in group_keys] + agg_exprs
        elif agg_exprs:
            sel_items = agg_exprs
        elif select_cols:
            sel_items = [_col(c) for c in select_cols]
        else:
            if has_join:
                sel_items = [f"{scan_alias}.*"]
            else:
                sel_items = ["*"]

        sql = f"SELECT {', '.join(sel_items)} FROM {scan_table}"

        if join_clauses:
            sql += " " + " ".join(join_clauses)

        if filters:
            sql += f" WHERE {' AND '.join(filters)}"

        if group_keys:
            if has_join:
                sql += f" GROUP BY {', '.join(_col(k) for k in group_keys)}"
            else:
                sql += f" GROUP BY {', '.join(f'\"' + k + '\"' for k in group_keys)}"

        sql += sort_clause

        if not limit_clause:
            limit_clause = f" LIMIT {MAX_RESULT_ROWS}"
        sql += limit_clause

        try:
            result = self.conn.execute(sql, params if params else None)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return {
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
                "sql": sql,
            }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "sql": sql,
                "error": str(e),
            }

    def _resolve_csv(self, dataset: str) -> str:
        #Parses the dataset_id into a CSV file path
        import os
        p = Path(dataset)
        if p.is_absolute() and p.exists():
            csv_dir_resolved = os.path.realpath(str(CSV_DIR))
            target_resolved = os.path.realpath(str(p))
            if not target_resolved.startswith(csv_dir_resolved):
                return str(CSV_DIR / f"{Path(dataset).name}.csv")
            return str(p)
        safe_name = Path(dataset).name
        for suffix in ["", ".csv"]:
            candidate = CSV_DIR / f"{safe_name}{suffix}"
            if candidate.exists():
                return str(candidate)
        return str(CSV_DIR / f"{safe_name}.csv")

    def _build_filter(self, col_expr: str, op: str, value) -> tuple:
        #Build parameterized filter clause. Returns (sql_fragment, params_list).
        allowed_ops = {"=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "NOT IN", "IS", "IS NOT"}
        op_upper = op.upper().strip()
        if op_upper not in allowed_ops:
            op_upper = "="

        if value is None:
            if op_upper in ("=", "IS"):
                return f"{col_expr} IS NULL", []
            else:
                return f"{col_expr} IS NOT NULL", []
        elif isinstance(value, (list, tuple)):
            placeholders = ", ".join(["?"] * len(value))
            return f"{col_expr} IN ({placeholders})", list(value)
        else:
            return f"{col_expr} {op_upper} ?", [value]

    def _build_agg(self, metric: str, func: str, qualified: bool = False) -> str:
        func_map = {
            "sum": "SUM",
            "avg": "AVG",
            "count": "COUNT",
            "min": "MIN",
            "max": "MAX",
            "median": "MEDIAN",
            "count_distinct": "COUNT(DISTINCT",
        }
        sql_func = func_map.get(func, "SUM")
        col_ref = metric if qualified else f'"{metric}"'
        if func == "count_distinct":
            return f'{sql_func} {col_ref})'
        return f'{sql_func}({col_ref})'

    def close(self):
        self.conn.close()


# Polars Executor
class PolarsExecutor:
    #Compile the operator plan into a Polars lazy operation
    def execute(self, plan: ExecutablePlan) -> dict:
        import polars as pl

        lf = None
        group_keys: list[str] = []
        agg_exprs: list = []
        sort_col = None
        sort_desc = True
        limit_n = MAX_RESULT_ROWS
        select_cols: list[str] = []

        for step in plan.steps:
            op = step.op
            p = step.params

            if op == OpType.SCAN:
                csv_path = self._resolve_csv(p["dataset"])
                lf = pl.scan_csv(csv_path, try_parse_dates=True)

            elif op == OpType.JOIN:
                if lf is None:
                    continue
                right_dataset = p.get("right_dataset", "")
                left_key = p.get("left_key", "")
                right_key = p.get("right_key", "")
                join_type = p.get("join_type", "LEFT").lower()
                # Map to Polars join types
                polars_how = {"left": "left", "right": "right",
                              "inner": "inner", "outer": "full",
                              "cross": "cross"}.get(join_type, "left")
                right_csv = self._resolve_csv(right_dataset)
                right_lf = pl.scan_csv(right_csv, try_parse_dates=True)
                if polars_how == "cross":
                    lf = lf.join(right_lf, how="cross")
                else:
                    lf = lf.join(right_lf, left_on=left_key,
                                 right_on=right_key, how=polars_how)

            elif op == OpType.DERIVE_TIME:
                source = p["source"]
                target = p["target"]
                unit = p.get("unit", "year")
                if lf is not None:
                    try:
                        date_col = pl.col(source).cast(pl.Date)
                    except Exception:
                        date_col = pl.col(source).str.to_date(strict=False)
                    if unit == "year":
                        lf = lf.with_columns(date_col.dt.year().alias(target))
                    elif unit == "month":
                        lf = lf.with_columns(date_col.dt.month().alias(target))
                    elif unit == "quarter":
                        lf = lf.with_columns(
                            ((date_col.dt.month() - 1) // 3 + 1).alias(target)
                        )
                    else:
                        lf = lf.with_columns(pl.col(source).cast(pl.Utf8).alias(target))

            elif op == OpType.FILTER:
                col = p["column"]
                op_str = p.get("op", "=")
                val = p.get("value")
                if lf is not None:
                    lf = self._apply_filter(lf, col, op_str, val)

            elif op == OpType.GROUPBY:
                group_keys = p.get("keys", [])

            elif op == OpType.AGGREGATE:
                metric = p["metric"]
                func = p.get("func", "sum")
                agg_exprs.append(self._build_agg_expr(metric, func))

            elif op == OpType.SORT:
                sort_col = p.get("column")
                sort_desc = p.get("order", "desc") == "desc"

            elif op == OpType.LIMIT:
                limit_n = min(p.get("n", MAX_RESULT_ROWS), MAX_RESULT_ROWS)

            elif op == OpType.SELECT:
                select_cols = p.get("columns", [])

        if lf is None:
            return {"columns": [], "rows": [], "row_count": 0}

        try:
            if group_keys and agg_exprs:
                lf = lf.group_by(group_keys).agg(agg_exprs)

            if sort_col:
                # Handle sort by aggregated column
                try:
                    lf = lf.sort(sort_col, descending=sort_desc)
                except Exception:
                    # If sort col name changed due to agg, try to sort by last column
                    pass

            if select_cols:
                # Only select columns that exist
                try:
                    lf = lf.select(select_cols)
                except Exception:
                    pass

            lf = lf.limit(limit_n)
            df = lf.collect()

            return {
                "columns": df.columns,
                "rows": df.to_pandas().values.tolist(),
                "row_count": len(df),
            }
        except Exception as e:
            return {"columns": [], "rows": [], "row_count": 0, "error": str(e)}

    def _resolve_csv(self, dataset: str) -> str:
        #Safe CSV path resolution (consistent with DuckDB executor).
        import os
        p = Path(dataset)
        if p.is_absolute() and p.exists():
            csv_dir_resolved = os.path.realpath(str(CSV_DIR))
            target_resolved = os.path.realpath(str(p))
            if not target_resolved.startswith(csv_dir_resolved):
                return str(CSV_DIR / f"{Path(dataset).name}.csv")
            return str(p)
        safe_name = Path(dataset).name
        for suffix in ["", ".csv"]:
            candidate = CSV_DIR / f"{safe_name}{suffix}"
            if candidate.exists():
                return str(candidate)
        return str(CSV_DIR / f"{safe_name}.csv")

    def _apply_filter(self, lf, col, op_str, val):
        import polars as pl
        c = pl.col(col)
        ops = {
            "=": c == val, "!=": c != val, ">": c > val, "<": c < val,
            ">=": c >= val, "<=": c <= val,
            "LIKE": c.str.contains(str(val).replace("%", ".*") if val else ""),
        }
        expr = ops.get(op_str.upper() if isinstance(op_str, str) else op_str, c == val)
        return lf.filter(expr)

    def _build_agg_expr(self, metric, func):
        import polars as pl
        c = pl.col(metric)
        funcs = {
            "sum": c.sum(), "avg": c.mean(), "count": c.count(),
            "min": c.min(), "max": c.max(), "median": c.median(),
            "count_distinct": c.n_unique(),
        }
        return funcs.get(func, c.sum())

# Main entry
def execute_plan(plan: ExecutablePlan) -> dict:
    """
    Executor main entry point: Executes the operator plan
        1.Selects the DuckDB or Polars backend based on the configuration.
        2.DuckDB uses a singleton connection for performance.
    """
    if EXECUTOR_BACKEND == "polars":
        executor = PolarsExecutor()
    else:
        executor = DuckDBExecutor.get_instance()

    return executor.execute(plan)
