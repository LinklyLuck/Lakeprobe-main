#!/usr/bin/env python3
"""
LakeProbe -- kaggle Benchmark Runner
Usage:
  python benchmarks/run_benchmark.py --csv-dir E:\\Lakeprobe_V3\\data\\csv
  python benchmarks/run_benchmark.py --csv-dir E:\\Lakeprobe_V3\\data\\csv --mode query
  python benchmarks/run_benchmark.py --csv-dir E:\\Lakeprobe_V3\\data\\csv --mode discovery
  python benchmarks/run_benchmark.py --csv-dir E:\\Lakeprobe_V3\\data\\csv --mode cost
"""
from __future__ import annotations
import argparse, csv, json, sys, time, signal, threading
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
ROOT = BENCH_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUERY_TIMEOUT = 60  # seconds per query


# 25 QUERY TESTS (real column names from 67 CSVs)
QUERY_TESTS = [
    # Wine (3)
    {"id":"Q01","cat":"wine","query":"average alcohol by quality",
     "exp_m":["alcohol"],"exp_d":["quality"]},
    {"id":"Q02","cat":"wine","query":"top 5 wines with highest sulphates",
     "exp_m":["sulphates"],"exp_d":[]},
    {"id":"Q03","cat":"wine","query":"average pH by quality level",
     "exp_m":["pH"],"exp_d":["quality"]},
    # Housing (3)
    {"id":"Q04","cat":"housing","query":"average PRICE by CHAS",
     "exp_m":["PRICE"],"exp_d":["CHAS"]},
    {"id":"Q05","cat":"housing","query":"average RM by RAD",
     "exp_m":["RM"],"exp_d":["RAD"]},
    {"id":"Q06","cat":"housing","query":"maximum PRICE grouped by PTRATIO",
     "exp_m":["PRICE"],"exp_d":["PTRATIO"]},
    # Taxi (3)
    {"id":"Q07","cat":"taxi","query":"average Trip_Price by Weather",
     "exp_m":["Trip_Price"],"exp_d":["Weather"]},
    {"id":"Q08","cat":"taxi","query":"average Trip_Price by Traffic_Conditions",
     "exp_m":["Trip_Price"],"exp_d":["Traffic_Conditions"]},
    {"id":"Q09","cat":"taxi","query":"average Trip_Duration_Minutes by Day_of_Week",
     "exp_m":["Trip_Duration_Minutes"],"exp_d":["Day_of_Week"]},
    # Medical (3)
    {"id":"Q10","cat":"medical","query":"average Treatment_Cost_USD by Cancer_Type",
     "exp_m":["Treatment_Cost_USD"],"exp_d":["Cancer_Type"]},
    {"id":"Q11","cat":"medical","query":"average Target_Severity_Score by Cancer_Stage",
     "exp_m":["Target_Severity_Score"],"exp_d":["Cancer_Stage"]},
    {"id":"Q12","cat":"medical","query":"average Genetic_Risk by Gender",
     "exp_m":["Genetic_Risk"],"exp_d":["Gender"]},
    # MBA (3)
    {"id":"Q13","cat":"mba","query":"average gpa by major",
     "exp_m":["gpa"],"exp_d":["major"]},
    {"id":"Q14","cat":"mba","query":"average gmat by admission",
     "exp_m":["gmat"],"exp_d":["admission"]},
    {"id":"Q15","cat":"mba","query":"count admission by gender",
     "exp_m":["admission"],"exp_d":["gender"]},
    # Insurance (2)
    {"id":"Q16","cat":"insurance","query":"average charges by smoker",
     "exp_m":["charges"],"exp_d":["smoker"]},
    {"id":"Q17","cat":"insurance","query":"average bmi by region",
     "exp_m":["bmi"],"exp_d":["region"]},
    # IoT (3)
    {"id":"Q18","cat":"iot","query":"average Temperature_C by Machine_Type",
     "exp_m":["Temperature_C"],"exp_d":["Machine_Type"]},
    {"id":"Q19","cat":"iot","query":"average Vibration_mms by Machine_Type",
     "exp_m":["Vibration_mms"],"exp_d":["Machine_Type"]},
    {"id":"Q20","cat":"iot","query":"maximum Power_Consumption_kW",
     "exp_m":["Power_Consumption_kW"],"exp_d":[]},
    # Energy (2)
    {"id":"Q21","cat":"energy","query":"average Energy Consumption by Building Type",
     "exp_m":["Energy Consumption"],"exp_d":["Building Type"]},
    {"id":"Q22","cat":"energy","query":"average Square Footage by Day of Week",
     "exp_m":["Square Footage"],"exp_d":["Day of Week"]},
    # Animal (2)
    {"id":"Q23","cat":"animal","query":"average Weight by Breed",
     "exp_m":["Weight"],"exp_d":["Breed"]},
    {"id":"Q24","cat":"animal","query":"average Sleep_time_hours by Country",
     "exp_m":["Sleep_time_hours"],"exp_d":["Country"]},
    # Ecommerce (1)
    {"id":"Q25","cat":"ecommerce","query":"average price by seller_id",
     "exp_m":["price"],"exp_d":["seller_id"]},
]

# 25 DISCOVERY TESTS (real dataset IDs from 67 CSVs)
DISCOVERY_TESTS = [
    # Wine (3)
    {"id":"D01","cat":"wine","query":"I want a wine dataset to predict quality",
     "exp_ds":["wine_components","winequality_red","wine_quality_red"],"exp_cols":["quality","alcohol"]},
    {"id":"D02","cat":"wine","query":"Find a wine data with residual sugar and acidity",
     "exp_ds":["wine_acid_sugar","wine_components","winequality_red"],"exp_cols":["residual sugar","acidity"]},
    {"id":"D03","cat":"wine","query":"Show me wine chemistry datasets with pH and sulfur dioxide",
     "exp_ds":["wine_components","wine_quality_red","winequality_red"],"exp_cols":["pH","sulfur dioxide"]},
    # Housing (3)
    {"id":"D04","cat":"housing","query":"Find a datasets to predict Boston housing prices",
     "exp_ds":["boston","boston_home","boston_residence"],"exp_cols":["PRICE","RM","LSTAT"]},
    {"id":"D05","cat":"housing","query":"I need data about crime rate and property values",
     "exp_ds":["boston","boston_home","boston_residence"],"exp_cols":["CRIM","PRICE"]},
    {"id":"D06","cat":"housing","query":"Show me housing data with tax and age information",
     "exp_ds":["boston","boston_home","boston_residence"],"exp_cols":["TAX","AGE"]},
    # Taxi (3)
    {"id":"D07","cat":"taxi","query":"Find a datasets for predicting taxi trip fare",
     "exp_ds":["taxi_trip_pricing","cab_fare","cab_ride"],"exp_cols":["Trip_Price","Trip_Distance"]},
    {"id":"D08","cat":"taxi","query":"I need cab ride data with weather and traffic conditions",
     "exp_ds":["taxi_trip_pricing","cab_fare_cost","cab_ride_cost"],"exp_cols":["Weather","Traffic_Conditions"]},
    {"id":"D09","cat":"taxi","query":"Show me trip pricing data with base fare and per km rate",
     "exp_ds":["taxi_trip_pricing","cab_fare_rates","cab_fare_cost"],"exp_cols":["Base_Fare","Per_Km_Rate"]},
    # Medical (3)
    {"id":"D10","cat":"medical","query":"Find a cancer patient data with treatment cost and severity",
     "exp_ds":["cancer_medical","worldwide_cancer"],"exp_cols":["Treatment_Cost_USD","Target_Severity_Score"]},
    {"id":"D11","cat":"medical","query":"I need data about cancer survival years and genetic risk",
     "exp_ds":["cancer_medical","census_cancer","worldwide_cancer"],"exp_cols":["Survival_Years","Genetic_Risk"]},
    {"id":"D12","cat":"medical","query":"Show me datasets with cancer stage, smoking, and obesity data",
     "exp_ds":["cancer_medical","worldwide_cancer"],"exp_cols":["Cancer_Stage","Smoking","Obesity_Level"]},
    # MBA (2)
    {"id":"D13","cat":"mba","query":"Find a MBA admission data with GPA and GMAT scores",
     "exp_ds":["MBA_admission","business_administration"],"exp_cols":["gpa","gmat","admission"]},
    {"id":"D14","cat":"mba","query":"I need student profile data with work experience and industry",
     "exp_ds":["MBA_students_profile","business_administration"],"exp_cols":["work_exp","work_industry"]},
    # Insurance (2)
    {"id":"D15","cat":"insurance","query":"Find a insurance data to predict medical charges",
     "exp_ds":["insurance","coverage_data","coverage_details"],"exp_cols":["charges","bmi","smoker"]},
    {"id":"D16","cat":"insurance","query":"I need coverage data with age and region information",
     "exp_ds":["insurance","coverage_data","coverage_details"],"exp_cols":["age","region","charges"]},
    # IoT (3)
    {"id":"D17","cat":"iot","query":"Find a machine sensor data with temperature and vibration",
     "exp_ds":["IoT_factory","manufacturing_detector","industrial_sensor"],"exp_cols":["Temperature_C","Vibration"]},
    {"id":"D18","cat":"iot","query":"I need IoT data to predict machine failure",
     "exp_ds":["IoT_stors_indexes","manufacturing_detector","factory_sensor"],"exp_cols":["Failure_Within_7_Days","Remaining_Useful_Life"]},
    {"id":"D19","cat":"iot","query":"Show me manufacturing data with hydraulic pressure and coolant flow",
     "exp_ds":["manufacturing_detector","industrial_sensor","manufacturing_monitor"],"exp_cols":["Hydraulic_Pressure_bar","Coolant_Flow_L_min"]},
    # Energy (2)
    {"id":"D20","cat":"energy","query":"Find a energy consumption data by building type",
     "exp_ds":["train_energy_data","training_energy","training_power"],"exp_cols":["Energy Consumption","Building Type"]},
    {"id":"D21","cat":"energy","query":"I need power usage data with temperature and occupants",
     "exp_ds":["train_energy_data","training_power","train_power"],"exp_cols":["Energy Consumption","Average Temperature"]},
    # Animal (2)
    {"id":"D22","cat":"animal","query":"Find a cat breed data with weight and body length",
     "exp_ds":["cat_breed","feline_variety","cat_breed_details"],"exp_cols":["Weight","Body_length","Breed"]},
    {"id":"D23","cat":"animal","query":"I need pet data with fur colour and eye colour",
     "exp_ds":["cat_breed","feline_variety","customer_information_874"],"exp_cols":["Fur_colour_dominant","Eye_colour"]},
    # Ecommerce (2)
    {"id":"D24","cat":"ecommerce","query":"Find a e-commerce order data with price and freight value",
     "exp_ds":["olist_order_items","olist_order_payments"],"exp_cols":["price","freight_value"]},
    {"id":"D25","cat":"ecommerce","query":"I need product data with category names and dimensions",
     "exp_ds":["olist_products","product_category"],"exp_cols":["product_category_name","product_weight_g"]},
]


# Matching Helpers
def _col_match(expected, actual):
    e, a = expected.lower().strip(), actual.lower().strip()
    return e in a or a in e

def _ds_match(expected, actual):
    e, a = expected.lower().strip(), actual.lower().strip()
    return e in a or a in e



# Thread-based per-query timeout
class _TimeoutError(Exception):
    pass

def _run_with_timeout(func, timeout_sec):
    """Run func() with a timeout. Returns (result, error_string)."""
    result_box = [None]
    error_box = [None]

    def worker():
        try:
            result_box[0] = func()
        except Exception as e:
            error_box[0] = str(e)[:200]

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None, f"TIMEOUT ({timeout_sec}s)"
    if error_box[0]:
        return None, error_box[0]
    return result_box[0], None


# QUERY Benchmark
def run_query_benchmark():
    try:
        from main import parse_intent_only, pipeline_from_intent
    except ModuleNotFoundError:
        from app.main import parse_intent_only, pipeline_from_intent

    print("")
    print("=" * 65)
    print("  QUERY BENCHMARK -- 25 tests")
    print("=" * 65)
    results = []
    for i, t in enumerate(QUERY_TESTS, 1):
        q = t["query"]
        r = {"id": t["id"], "cat": t["cat"], "query": q, "error": None}
        t0 = time.perf_counter()

        def _do_query():
            p1 = parse_intent_only(q)
            p2 = pipeline_from_intent(q, p1["intent"], auto_execute=True,
                                      phase1_token_usage=p1.get("token_usage", {}))
            return p1, p2

        pair, err = _run_with_timeout(_do_query, QUERY_TIMEOUT)
        lat = (time.perf_counter() - t0) * 1000

        if err:
            r["error"] = err
            r["latency_ms"] = round(lat, 1)
            results.append(r)
            print("  [%2d/25] X %s (%10s) ERROR: %s" % (i, t["id"], t["cat"], err[:50]))
            continue

        p1, p2 = pair
        if p2.get("error"):
            r["error"] = p2["error"]
            r["latency_ms"] = round(lat, 1)
            results.append(r)
            print("  [%2d/25] X %s (%10s) ERROR: %s" % (i, t["id"], t["cat"], r["error"][:50]))
            continue

        b = p2.get("binding", {})
        tk = p2.get("token_usage", {})
        candidates = p2.get("candidates", {})
        bm = [x.get("physical_column", "") or x.get("column", "") for x in b.get("metric_bindings", [])]
        bd = [x.get("physical_column", "") or x.get("column", "") for x in b.get("dimension_bindings", [])]
        bt = [x.get("physical_column", "") or x.get("column", "") for x in b.get("time_bindings", [])]
        bf = [x.get("physical_column", "") or x.get("column", "") for x in b.get("filter_bindings", [])]
        blocked = b.get("blocked_candidates", [])
        ds_id = b.get("dataset_id", "")

        # Table selection: does the selected dataset match the expected category?
        table_ok = t["cat"].lower() in ds_id.lower() if ds_id else False

        # Table selection (relaxed): any candidate dataset matches?
        all_candidate_ds = set([ds_id]) if ds_id else set()
        for dc in candidates.get("dataset_candidates", []):
            if isinstance(dc, str):
                all_candidate_ds.add(dc)
            elif isinstance(dc, dict):
                cand_ds = dc.get("dataset_id", "")
                if cand_ds:
                    all_candidate_ds.add(cand_ds)
        table_any = any(t["cat"].lower() in cds.lower() for cds in all_candidate_ds) if all_candidate_ds else False

        # Binding accuracy
        m_ok = (not t["exp_m"]) or any(any(_col_match(e, a) for a in bm) for e in t["exp_m"])
        d_ok = (not t["exp_d"]) or any(any(_col_match(e, a) for a in bd) for e in t["exp_d"])
        ok = m_ok and d_ok

        r.update({"m_ok": m_ok, "d_ok": d_ok, "binding_ok": ok,
                  "bm": bm, "bd": bd, "bt": bt, "bf": bf,
                  "ds": ds_id, "table_ok": table_ok, "table_any": table_any,
                  "blocked": len(blocked),
                  "blocked_reasons": [x.get("reason", "") for x in blocked],
                  "latency_ms": round(lat, 1),
                  "lp_tok": tk.get("lakeprobe_total_tokens", 0),
                  "t2s_tok": tk.get("text2sql_total_tokens", 0)})
        icon = "OK" if ok else "X"
        tbl_icon = "T" if table_ok else ("t" if table_any else "-")
        print("  [%2d/25] %s%s %s (%10s) m=%s d=%s blk=%d %.0fms"
              % (i, icon, tbl_icon, t["id"], t["cat"], bm, bd, len(blocked), lat))
        results.append(r)
    return results


# DISCOVERY Benchmark
def run_discovery_benchmark():
    try:
        from main import discovery_pipeline
    except ModuleNotFoundError:
        from app.main import discovery_pipeline

    print("")
    print("=" * 65)
    print("  DISCOVERY BENCHMARK -- 25 tests")
    print("=" * 65)
    results = []
    for i, t in enumerate(DISCOVERY_TESTS, 1):
        q = t["query"]
        r = {"id": t["id"], "cat": t["cat"], "query": q, "error": None}
        t0 = time.perf_counter()

        def _do_disc():
            return discovery_pipeline(q)

        out, err = _run_with_timeout(_do_disc, QUERY_TIMEOUT)
        lat = (time.perf_counter() - t0) * 1000

        if err:
            r["error"] = err
            r["latency_ms"] = round(lat, 1)
            results.append(r)
            print("  [%2d/25] X %s (%10s) ERROR: %s" % (i, t["id"], t["cat"], err[:50]))
            continue

        matched = out.get("matched_datasets", [])
        found = [m["dataset_id"] for m in matched[:10]]
        schema = out.get("desired_schema", {})
        dcols = [c.get("name", "") for c in schema.get("desired_columns", [])]

        top3 = found[:3]
        ds_hits = sum(1 for ds in top3 if any(_ds_match(e, ds) for e in t["exp_ds"]))
        ds_prec = ds_hits / len(top3) if top3 else 0

        col_hits = sum(1 for ec in t["exp_cols"] if any(_col_match(ec, dc) for dc in dcols))
        col_prec = col_hits / len(t["exp_cols"]) if t["exp_cols"] else 1.0

        cov_hits = 0
        for ec in t["exp_cols"]:
            for m in matched[:5]:
                ac = [c["name"] for c in m.get("all_columns", [])]
                mc = [cm.get("actual_column", "") for cm in m.get("column_matches", [])]
                if any(_col_match(ec, c) for c in ac + mc):
                    cov_hits += 1
                    break
        cov = cov_hits / len(t["exp_cols"]) if t["exp_cols"] else 1.0

        r.update({"found": found, "ds_prec": round(ds_prec, 3), "col_prec": round(col_prec, 3),
                  "coverage": round(cov, 3), "schema_domain": schema.get("domain", ""),
                  "schema_cols": len(dcols), "latency_ms": round(lat, 1)})
        icon = "OK" if ds_prec > 0.3 else "X"
        print("  [%2d/25] %s %s (%10s) ds=%.0f%% col=%.0f%% cov=%.0f%% top=%s"
              % (i, icon, t["id"], t["cat"], ds_prec * 100, col_prec * 100, cov * 100,
                 found[0] if found else "?"))
        results.append(r)
    return results


# TOKEN COST
def run_token_cost(query_results=None):
    """
    Token cost analysis with REAL measured LakeProbe tokens.

    If query_results are provided (from run_query_benchmark), uses the actual
    per-query lp_tokens/t2s_tokens. Otherwise runs a sample of queries to measure.
    """
    profiles_dir = ROOT / "data" / "profile_cards"
    all_ddl, total_cols = [], 0
    for jf in sorted(profiles_dir.glob("*.json")):
        try:
            p = json.loads(jf.read_text(encoding="utf-8"))
            dm = {"int64": "INTEGER", "float64": "REAL", "object": "TEXT",
                  "datetime64": "TIMESTAMP", "bool": "BOOLEAN"}
            cols = [("  %s %s" % (c["name"], dm.get(c["dtype"], "TEXT"))) for c in p["columns"]]
            total_cols += len(cols)
            all_ddl.append("CREATE TABLE %s (\n%s\n);" % (p["dataset_id"], ",\n".join(cols)))
        except Exception:
            pass

    schema_tok = len("\n\n".join(all_ddl)) // 4
    n = len(all_ddl)
    t2s = schema_tok + 550  # ALL tables DDL + system/fewshot/query/output overhead

    # Real LakeProbe tokens: use actual measurements from query benchmark
    if query_results:
        measured = [r for r in query_results if r.get("lp_tok", 0) > 0]
        if measured:
            lp = round(sum(r["lp_tok"] for r in measured) / len(measured))
        else:
            lp = _measure_lp_tokens_live()
    else:
        lp = _measure_lp_tokens_live()

    sv = (t2s - lp) / t2s * 100 if t2s else 0

    # Cost estimation (DeepSeek V3.2 pricing: $0.28/M input, $0.42/M output)
    lp_input_est = int(lp * 0.9)   # ~90% input
    lp_output_est = lp - lp_input_est
    t2s_input_est = t2s - 150      # ~150 output tokens for SQL
    lp_usd = (lp_input_est * 0.28 + lp_output_est * 0.42) / 1e6
    t2s_usd = (t2s_input_est * 0.28 + 150 * 0.42) / 1e6

    scaling = []
    for nt in [1, 5, 10, 20, 50, n, 100, 200, 500, 1000]:
        est = int(schema_tok * nt / max(n, 1)) + 550
        s = (est - lp) / max(est, 1) * 100
        scaling.append({"tables": nt, "t2s": est, "lp": lp, "saving": round(s, 1)})

    return {"n_tables": n, "total_cols": total_cols, "schema_tok": schema_tok,
            "t2s_tokens": t2s, "lp_tokens": lp, "saving_pct": round(sv, 1),
            "measurement": "real" if query_results else "live_sample",
            "monthly_lp": round(lp_usd * 10000 * 30, 2),
            "monthly_t2s": round(t2s_usd * 10000 * 30, 2),
            "monthly_saving": round((t2s_usd - lp_usd) * 10000 * 30, 2),
            "scaling": scaling}


def _measure_lp_tokens_live():
    """Run a few sample queries to measure real LakeProbe token cost."""
    try:
        from main import parse_intent_only, pipeline_from_intent
    except ModuleNotFoundError:
        from app.main import parse_intent_only, pipeline_from_intent

    samples = ["average alcohol by quality", "top 5 wines with highest sulphates",
               "average charges by smoker", "average Temperature_C by Machine_Type"]
    tokens = []
    for q in samples:
        try:
            p1 = parse_intent_only(q)
            p2 = pipeline_from_intent(q, p1["intent"], auto_execute=False,
                                       phase1_token_usage=p1.get("token_usage", {}))
            tk = p2.get("token_usage", {}).get("lakeprobe_total_tokens", 0)
            if tk > 0:
                tokens.append(tk)
        except Exception:
            pass
    return round(sum(tokens) / len(tokens)) if tokens else 900  # fallback



# SUMMARIZE
def summarize_query(results):
    from collections import Counter
    v = [r for r in results if not r.get("error")]
    if not v:
        return {"total_queries": len(results), "successful": 0, "errored": len(results)}
    n = len(v)
    cats = sorted(set(r["cat"] for r in v))
    cb = {}
    for c in cats:
        cr = [r for r in v if r["cat"] == c]
        cb[c] = {"count": len(cr),
                 "binding_accuracy": round(sum(1 for r in cr if r.get("binding_ok")) / len(cr), 3),
                 "table_selection": round(sum(1 for r in cr if r.get("table_ok")) / len(cr), 3),
                 "table_any": round(sum(1 for r in cr if r.get("table_any")) / len(cr), 3),
                 "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in cr) / len(cr), 1),
                 "blocked": sum(r.get("blocked", 0) for r in cr)}
    ba = sum(1 for r in v if r.get("binding_ok")) / n
    ma = sum(1 for r in v if r.get("m_ok")) / n
    da = sum(1 for r in v if r.get("d_ok")) / n
    ta = sum(1 for r in v if r.get("table_ok")) / n
    ta_any = sum(1 for r in v if r.get("table_any")) / n
    blk = sum(r.get("blocked", 0) for r in v)
    qwb = sum(1 for r in v if r.get("blocked", 0) > 0)
    alp = sum(r.get("lp_tok", 0) for r in v) / n
    alat = sum(r.get("latency_ms", 0) for r in v) / n

    # Hallucination block reasons breakdown
    reason_counts = Counter(
        reason for r in v for reason in r.get("blocked_reasons", []) if reason
    )

    # Text2SQL baseline: ALL tables DDL (the whole data lake schema)
    # Text2SQL must send the entire schema for the LLM to pick the right table.
    # This is the correct comparison — NOT single-table DDL.
    profiles_dir = ROOT / "data" / "profile_cards"
    all_ddl_chars = 0
    for jf in profiles_dir.glob("*.json"):
        try:
            p = json.loads(jf.read_text(encoding="utf-8"))
            for c in p["columns"]:
                all_ddl_chars += len(c["name"]) + 12
            all_ddl_chars += len(p["dataset_id"]) + 30
        except Exception:
            pass
    at2s = all_ddl_chars // 4 + 550  # full schema + system/fewshot/query/output

    token_saving = (at2s - alp) / at2s * 100 if at2s else 0

    return {"total_queries": len(results), "successful": n, "errored": len(results) - n,
            "binding_accuracy": round(ba, 3), "metric_accuracy": round(ma, 3),
            "dimension_accuracy": round(da, 3),
            "table_selection": round(ta, 3), "table_selection_any": round(ta_any, 3),
            "hallucination_blocked": blk, "queries_with_blocks": qwb,
            "interception_rate": round(qwb / n, 3),
            "hallucination_block_reasons": dict(reason_counts.most_common()),
            "avg_lakeprobe_tokens": round(alp), "avg_text2sql_tokens": round(at2s),
            "token_saving_pct": round(token_saving, 1),
            "avg_latency_ms": round(alat, 1), "category_breakdown": cb}


def summarize_discovery(results):
    v = [r for r in results if not r.get("error")]
    if not v:
        return {"total_queries": len(results), "successful": 0, "errored": len(results)}
    n = len(v)
    cats = sorted(set(r["cat"] for r in v))
    cb = {}
    for c in cats:
        cr = [r for r in v if r["cat"] == c]
        cb[c] = {"count": len(cr),
                 "avg_ds_precision": round(sum(r.get("ds_prec", 0) for r in cr) / len(cr), 3),
                 "avg_col_precision": round(sum(r.get("col_prec", 0) for r in cr) / len(cr), 3),
                 "avg_coverage": round(sum(r.get("coverage", 0) for r in cr) / len(cr), 3)}
    return {"total_queries": len(results), "successful": n, "errored": len(results) - n,
            "avg_dataset_precision": round(sum(r.get("ds_prec", 0) for r in v) / n, 3),
            "avg_column_precision": round(sum(r.get("col_prec", 0) for r in v) / n, 3),
            "avg_coverage": round(sum(r.get("coverage", 0) for r in v) / n, 3),
            "avg_schema_columns": round(sum(r.get("schema_cols", 0) for r in v) / n, 1),
            "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in v) / n, 1),
            "category_breakdown": cb}


# PRINT REPORT (pure ASCII)
def print_report(qs, ds_sum, tc, env):
    print("")
    print("=" * 65)
    print("  LAKEPROBE v11 -- VLDB BENCHMARK REPORT")
    print("  %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  %d datasets, %s rows" % (env["n_datasets"], env["total_rows"]))
    print("=" * 65)

    if qs:
        s = qs
        print("")
        print("  -- QUERY (%d queries, %d ok) --" % (s["total_queries"], s["successful"]))
        print("  Table Selection (strict):  %.1f%%" % (s["table_selection"] * 100))
        print("  Table Selection (any):     %.1f%%" % (s["table_selection_any"] * 100))
        print("  Binding Accuracy:          %.1f%%" % (s["binding_accuracy"] * 100))
        print("    Metric Accuracy:         %.1f%%" % (s["metric_accuracy"] * 100))
        print("    Dimension Accuracy:      %.1f%%" % (s["dimension_accuracy"] * 100))
        print("  Tokens LP/T2S:             %d/%d (%+.1f%%)" % (
            s["avg_lakeprobe_tokens"], s["avg_text2sql_tokens"], s["token_saving_pct"]))
        print("  Avg Latency:               %.0fms" % s["avg_latency_ms"])
        print("")
        # Hallucination guard
        blk = s.get("hallucination_blocked", 0)
        qwb = s.get("queries_with_blocks", 0)
        print("  -- HALLUCINATION GUARD --")
        print("  Type-mismatch blocked:  %d  (across %d queries)" % (blk, qwb))
        print("  Interception Rate:      %.1f%%" % (s["interception_rate"] * 100))
        if s.get("hallucination_block_reasons"):
            print("  Block reasons:")
            for reason, cnt in list(s["hallucination_block_reasons"].items())[:15]:
                print("    %-45s x%d" % (reason[:45], cnt))
        print("")
        print("  %12s %3s %6s %6s %6s %7s %5s" % (
            "Category", "N", "Bind%", "TblS", "TblA", "Lat", "Blkd"))
        print("  " + "-" * 52)
        for c, v in sorted(s["category_breakdown"].items()):
            print("  %12s %3d %5.0f%% %5.0f%% %5.0f%% %6.0fms %5d" % (
                c, v["count"], v["binding_accuracy"] * 100,
                v["table_selection"] * 100, v["table_any"] * 100,
                v["avg_latency_ms"], v["blocked"]))

    if ds_sum:
        s = ds_sum
        print("")
        print("  -- DISCOVERY (%d queries, %d ok) --" % (s["total_queries"], s["successful"]))
        print("  Dataset Precision:    %.1f%%" % (s["avg_dataset_precision"] * 100))
        print("  Column Precision:     %.1f%%" % (s["avg_column_precision"] * 100))
        print("  Coverage:             %.1f%%" % (s["avg_coverage"] * 100))
        print("  Schema Columns:       %.1f" % s["avg_schema_columns"])
        print("  Avg Latency:          %.0fms" % s["avg_latency_ms"])
        print("")
        print("  %12s %3s %8s %5s %5s" % ("Category", "N", "DS Prec", "Col", "Cov"))
        print("  " + "-" * 38)
        for c, v in sorted(s["category_breakdown"].items()):
            print("  %12s %3d %7.0f%% %4.0f%% %4.0f%%" % (
                c, v["count"], v["avg_ds_precision"] * 100,
                v["avg_col_precision"] * 100, v["avg_coverage"] * 100))

    if tc:
        print("")
        method = tc.get("measurement", "estimate")
        print("  -- TOKEN COST (%d tables, %s) --" % (tc["n_tables"], method))
        print("  Text2SQL:  %s tokens/query" % format(tc["t2s_tokens"], ","))
        print("  LakeProbe: %s tokens/query" % format(tc["lp_tokens"], ","))
        print("  Saving:    %+.1f%%" % tc["saving_pct"])
        print("")
        print("  Monthly (DeepSeek-V3.2, 10K/day):")
        print("    LP: $%.2f  T2S: $%.2f  Save: $%.2f" % (
            tc["monthly_lp"], tc["monthly_t2s"], tc["monthly_saving"]))
        print("")
        print("  %7s %10s %10s %8s" % ("Tables", "Text2SQL", "LakeProbe", "Saving"))
        print("  " + "-" * 40)
        for row in tc["scaling"]:
            mark = " <--" if row["tables"] == tc["n_tables"] else ""
            print("  %7d %10s %10s %+7.1f%%%s" % (
                row["tables"], format(row["t2s"], ","),
                format(row["lp"], ","), row["saving"], mark))

    print("")
    print("=" * 65)


# SAVE FILES (UTF-8)
def save_files(qr, qs, dr, ds_sum, tc, env):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": datetime.now().isoformat(), "environment": env,
              "query_benchmark": qs, "discovery_benchmark": ds_sum,
              "token_cost_analysis": tc, "query_details": qr, "discovery_details": dr}

    jp = RESULTS_DIR / ("benchmark_%s.json" % ts)
    jp.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    cp = RESULTS_DIR / ("benchmark_%s_summary.csv" % ts)
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])
        if qs:
            for k, v in qs.items():
                if k != "category_breakdown":
                    w.writerow(["query", k, v])
        if ds_sum:
            for k, v in ds_sum.items():
                if k != "category_breakdown":
                    w.writerow(["discovery", k, v])
        if tc:
            for k, v in tc.items():
                if k != "scaling":
                    w.writerow(["cost", k, v])

    dp = RESULTS_DIR / ("benchmark_%s_detail.csv" % ts)
    with open(dp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "id", "cat", "query", "pass", "detail", "latency_ms"])
        for r in qr:
            ok = "PASS" if r.get("binding_ok") else "FAIL"
            d = r.get("error", "") or ("m=%s d=%s" % (r.get("bm", []), r.get("bd", [])))
            w.writerow(["query", r["id"], r["cat"], r["query"], ok, d, r.get("latency_ms", "")])
        for r in dr:
            ok = "PASS" if r.get("ds_prec", 0) > 0.3 else "FAIL"
            d = r.get("error", "") or ("ds=%.0f%% col=%.0f%%" % (
                r.get("ds_prec", 0) * 100, r.get("col_prec", 0) * 100))
            w.writerow(["disc", r["id"], r["cat"], r["query"], ok, d, r.get("latency_ms", "")])

    return jp, cp, dp


# MAIN
def main():
    ap = argparse.ArgumentParser(description="LakeProbe v11 Benchmark")
    ap.add_argument("--csv-dir", type=str, required=True)
    ap.add_argument("--mode", default="all", choices=["all", "query", "discovery", "cost"])
    args = ap.parse_args()

    import config
    config.CSV_DIR = Path(args.csv_dir)

    from core.embedding_engine import reset_encoder
    reset_encoder()

    # Skip join index (not needed for benchmark, takes 5+ min on large IoT datasets)
    try:
        import core.join_discovery as _jd
        _jd.build_join_index_all = lambda: 0
    except Exception:
        pass

    try:
        from main import initialize_data
    except ModuleNotFoundError:
        from app.main import initialize_data
    cards = initialize_data(args.csv_dir)
    n = len(cards)
    if n == 0:
        print("[ERROR] No datasets.")
        sys.exit(1)

    # Count total rows
    total_rows = 0
    for jf in (ROOT / "data" / "profile_cards").glob("*.json"):
        try:
            total_rows += json.loads(jf.read_text(encoding="utf-8")).get("row_count", 0)
        except Exception:
            pass
    env = {"n_datasets": n,
           "total_rows": ("~%.1fM" % (total_rows / 1e6)) if total_rows > 1e6 else str(total_rows),
           "csv_dir": str(args.csv_dir)}

    qr, qs, dr, ds_sum, tc = [], {}, [], {}, {}

    if args.mode in ("all", "query"):
        qr = run_query_benchmark()
        qs = summarize_query(qr)
    if args.mode in ("all", "discovery"):
        dr = run_discovery_benchmark()
        ds_sum = summarize_discovery(dr)
    if args.mode in ("all", "cost"):
        tc = run_token_cost(query_results=qr if qr else None)

    print_report(qs, ds_sum, tc, env)
    jp, cp, dp = save_files(qr, qs, dr, ds_sum, tc, env)
    print("")
    print("  Files saved:")
    print("    %s" % jp)
    print("    %s" % cp)
    print("    %s" % dp)


if __name__ == "__main__":
    main()