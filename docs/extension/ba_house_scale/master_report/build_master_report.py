"""Read-only consolidation of frozen BA-House evidence. No ML imports."""
# Report prose is kept as complete paragraphs.
# ruff: noqa: E501

import csv
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = "docs/extension/ba_house_scale/"
PG = BASE + "phase4b2_pg_stability/07_cell_stability_summary.json"
AGG = BASE + "phase4a_gnn_random/05_ba4a_aggregate_results.json"
GEN = BASE + "phase1_generation/01_generation_report.json"
TARGET = BASE + "phase3_targets/01_target_freeze_report.json"
BACK = BASE + "phase2_backbones/02_predictive_comparison.json"
MAIN = "results/metrics/ba_house_2x2/main.csv"
STAB = "results/metrics/ba_house_2x2/stability.csv"
RAUDIT = "docs/audit/ba_house_graphsage_random_baseline_audit.json"
METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_probability_delta",
    "retain_probability",
    "retain_delta",
    "retain_probability_ratio",
    "sparsity",
]
PG_NAMES = {"remove_probability_delta": "remove_delta"}
HIST_NAMES = {"retain_delta": "retain_probability_delta"}
TABLE_NAMES = [
    "01_dataset_protocol_overview",
    "02_recovery_master",
    "03_faithfulness_master",
    "04_stability_master",
    "05_pg_saturation_diagnostics",
    "06_backbone_comparison",
    "07_explainer_comparison",
    "08_scale_robustness",
]
CACHE = {}
SOURCES = {}
MAP = []
TABLES = {}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    if path not in CACHE:
        p = ROOT / path
        if p.suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as stream:
                CACHE[path] = list(csv.DictReader(stream))
        elif p.suffix == ".json":
            CACHE[path] = json.loads(p.read_text(encoding="utf-8-sig"))
        else:
            CACHE[path] = p.read_text(encoding="utf-8-sig")
        SOURCES[path] = {"file": path, "sha256": digest(p)}
    return CACHE[path]


def ref(file, *field):
    read(file)
    return {"file": file, "field": list(field)}


def calc(op, *args):
    return {"operation": op, "arguments": list(args)}


def resolve(expr):
    if "operation" in expr:
        values = [resolve(a) for a in expr["arguments"]]
        return {
            "mean": statistics.mean,
            "sample_std": statistics.stdev,
            "min": min,
            "max": max,
            "difference": lambda a: a[0] - a[1],
            "length": lambda a: len(a[0]),
        }[expr["operation"]](values)
    value = read(expr["file"])
    for part in expr["field"]:
        value = value[part]
    if isinstance(value, str):
        try:
            return float(value) if value else None
        except ValueError:
            pass
    return value


def put(table, row, field, expression, missing=None):
    index = len(TABLES.get(table, []))
    row[field] = None if expression is None else resolve(expression)
    MAP.append(
        {
            "output_file": table + ".json",
            "output_field": ["rows", index, field],
            "scale": row.get("scale", row.get("configuration")),
            "backbone": row.get("backbone"),
            "explainer": row.get("explainer"),
            "metric": row.get("metric", field),
            "historical_or_extension": "historical"
            if row.get("scale", row.get("configuration")) == "BA500"
            else "extension",
            "primary_or_contextual": row.get("primary_or_contextual", "primary"),
            "expression": expression,
            "missing_reason": missing,
        }
    )


def csv_metric(file, row, metric, stat="mean"):
    return ref(file, row, metric + "_" + stat)


def row_mean(file, metric, target_ids=None):
    data = read(file)
    rows = [i for i, r in enumerate(data) if target_ids is None or int(r["node_id"]) in target_ids]
    if not rows or metric not in data[0] or any(not data[i][metric] for i in rows):
        return None
    return calc("mean", *(ref(file, i, metric) for i in rows))


def formatted(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "/").replace("\n", " ")


def markdown(rows, fields=None):
    fields = fields or list(rows[0])
    return (
        "| "
        + " | ".join(fields)
        + " |\n| "
        + " | ".join("---" for _ in fields)
        + " |\n"
        + "\n".join("| " + " | ".join(formatted(row.get(f)) for f in fields) + " |" for row in rows)
        + "\n"
    )


def write_json(name, data):
    (OUT / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_md(name, content):
    (OUT / name).write_text(content + "\n", encoding="utf-8")


def table(name, rows, note):
    TABLES[name] = rows
    write_json(name + ".json", {"rows": rows, "notes": note})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_md(
        name + ".md",
        "# " + name[3:].replace("_", " ").title() + "\n\n" + markdown(rows, fields) + "\n" + note,
    )


def protection():
    previous = read(BASE + "phase4b2_pg_stability/hash_manifest_before.json")
    paths = {r["path"] for r in previous["files"]}
    for namespace in (ROOT / BASE).iterdir():
        if namespace.is_dir() and namespace != OUT:
            paths.update(
                p.relative_to(ROOT).as_posix() for p in namespace.rglob("*") if p.is_file()
            )
    paths.update(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "artifacts/extensions/ba_house_scale").rglob("*")
        if p.is_file()
    )
    paths.update(SOURCES)
    return {p: digest(ROOT / p) for p in sorted(paths)}


def collect():
    for p in (
        "docs/ba_house_2x2_freeze.json",
        "docs/audit/target_selection_protocol_audit.md",
        BASE + "phase1_generation/03_structural_clean_pool.json",
        BASE + "phase2_backbones/01_backbone_training_report.json",
        BASE + "phase3_targets/02_pg_pool_report.json",
        BASE + "phase4a_gnn_random/01_gnnexplainer_protocol_receipt.json",
        BASE + "phase4b1_pg_primary/06_pg_primary_aggregate.json",
        BASE + "phase4b1_pg_audit/12_root_cause_report.json",
        BASE + "phase4b1_pg_policy/05_primary_phase_decision.json",
        BASE + "phase4b1_pg_policy/01_policy_definition.json",
        BASE + "phase4b2_pg_stability/01_protocol_freeze.json",
        BASE + "phase4b2_pg_stability/12_stability_verifier_receipt.json",
        BASE + "phase4b2_pg_stability/09_seed42_context.json",
    ):
        read(p)
    cells = []
    ids = read("docs/ba_house_2x2_freeze.json")["target_ids"]
    for scale in ("BA500", "BA1000", "BA1500"):
        for backbone in ("GCN", "GraphSAGE"):
            for method in ("GNNExplainer", "PGExplainer", "Random"):
                cells.append({"scale": scale, "backbone": backbone, "explainer": method})
    datasets = []
    name = TABLE_NAMES[0]
    TABLES[name] = datasets
    for scale in ("BA500", "BA1000", "BA1500"):
        r = {
            "configuration": scale,
            "role": "historical frozen benchmark"
            if scale == "BA500"
            else "independent seeded scale robustness extension",
        }
        if scale == "BA500":
            meta = "artifacts/data/ba_house_metadata.json"
            for f, pointer in {
                "nodes": ("shapes", "num_nodes"),
                "directed_edges": ("shapes", "num_edges"),
                "feature_dimension": ("shapes", "num_node_features"),
                "train_split": ("split_counts", "train"),
                "validation_split": ("split_counts", "validation"),
                "test_split": ("split_counts", "test"),
                "gt_positive_directed_edges": ("ground_truth_counts", "edges"),
            }.items():
                put(name, r, f, ref(meta, *pointer))
            put(name, r, "final_target_count", ref("docs/ba_house_2x2_freeze.json", "target_count"))
            put(
                name,
                r,
                "historical_explainer_holdout_count",
                calc("length", ref("results/metrics/explainer_validation_targets.csv")),
            )
            r["target_selection_note"] = (
                "Historical GCN-correct motif test selection; confidence sort, seeded shuffle, 4 holdout/12 final; later GraphSAGE parity. NOT extension selector."
            )
            for b, source in (
                ("gcn", "results/metrics/gcn_model_summary.json"),
                ("graphsage", "results/metrics/graphsage_ba_house_training/summary.json"),
            ):
                r[b + "_checkpoint_status"] = "historical frozen best validation accuracy"
                put(name, r, b + "_test_accuracy", ref(source, "test_accuracy"))
                put(name, r, b + "_validation_accuracy", ref(source, "best_validation_accuracy"))
            for b, prefix in (("gcn", "pgexplainer"), ("graphsage", "pgexplainer_graphsage")):
                for split in ("train", "val"):
                    put(
                        name,
                        r,
                        b + "_pg_" + split + "_pool",
                        calc(
                            "length",
                            ref("configs/targets/" + prefix + "_" + split + "_targets.csv"),
                        ),
                    )
            r["pg_pool_note"] = (
                "Historical backbone-specific natural pools; common-pool sensitivity is post-hoc, not primary."
            )
        else:
            variant = next(
                i
                for i, v in enumerate(read(GEN)["variants"])
                if v["dataset_variant"].lower() == scale.lower() + "_ext"
            )
            for field, source in {
                "nodes": "num_nodes",
                "directed_edges": "directed_entries",
                "feature_dimension": "feature_dim",
                "gt_positive_directed_edges": "global_gt_positive_directed_entries",
            }.items():
                put(name, r, field, ref(GEN, "variants", variant, "structure", source))
            for field, split in (
                ("train_split", "train"),
                ("validation_split", "validation"),
                ("test_split", "test"),
            ):
                put(
                    name,
                    r,
                    field,
                    ref(GEN, "variants", variant, "structure", "split_counts", split),
                )
            ti = next(
                i
                for i, t in enumerate(read(TARGET)["scales"])
                if t["dataset_variant"].lower() == scale.lower() + "_ext"
            )
            put(name, r, "final_target_count", ref(TARGET, "scales", ti, "selected_count"))
            put(name, r, "pg_train_pool", ref(TARGET, "scales", ti, "pg_train_count"))
            put(name, r, "pg_validation_pool", ref(TARGET, "scales", ti, "pg_validation_count"))
            for b in ("gcn", "graphsage"):
                put(name, r, b + "_pg_train_pool", ref(TARGET, "scales", ti, "pg_train_count"))
                put(name, r, b + "_pg_val_pool", ref(TARGET, "scales", ti, "pg_validation_count"))
            r["target_selection_note"] = (
                read(TARGET)["scales"][ti]["eligibility_rule"]
                + "; "
                + read(TARGET)["scales"][ti]["selection_rule"]
            )
            for b in ("gcn", "graphsage"):
                bi = next(
                    i
                    for i, c in enumerate(read(BACK)["cells"])
                    if c["scale"] == scale.lower() + "_ext" and c["backbone"] == b
                )
                r[b + "_checkpoint_status"] = read(BACK)["cells"][bi]["status"]
                put(
                    name,
                    r,
                    b + "_test_accuracy",
                    ref(BACK, "cells", bi, "metrics", "test", "accuracy"),
                )
                put(
                    name,
                    r,
                    b + "_validation_accuracy",
                    ref(BACK, "cells", bi, "metrics", "val", "accuracy"),
                )
        r["ground_truth_edge_availability"] = (
            "Global internal-house edge GT; continuous ranking on full induced 3-hop candidate edges"
        )
        if scale == "BA500":
            raw = "results/metrics/frozen_phase4c_v2/per_node_metrics.csv"
            ids = read("docs/ba_house_2x2_freeze.json")["target_ids"]
            put(name, r, "mean_candidate_edges_final", row_mean(raw, "candidate_edge_count", ids))
            put(
                name,
                r,
                "mean_gt_prevalence_final",
                row_mean(raw, "ground_truth_edge_prevalence", ids),
            )
        else:
            ai = next(
                i
                for i, c in enumerate(read(AGG)["tables"][4]["rows"])
                if c[0] == scale.lower() + "_ext"
            )
            put(name, r, "mean_candidate_edges_final", ref(AGG, "tables", 4, "rows", ai, 2))
            put(name, r, "mean_gt_prevalence_final", ref(AGG, "tables", 4, "rows", ai, 6))
        r["primary_interpretation"] = (
            "Descriptive frozen cohort; not a nested sequence or causal graph-size comparison"
        )
        datasets.append(r)
    table(
        name,
        datasets,
        "BA500 is historical, not BA500_EXT experiments. The generation control reproduces tensors only, not historical selector chronology. Extension targets are class1 clean common-correct; selection-conditioned results do not cover all nodes/classes.",
    )

    recovery, faith, stability = [], [], []
    TABLES[TABLE_NAMES[1]] = recovery
    TABLES[TABLE_NAMES[2]] = faith
    TABLES[TABLE_NAMES[3]] = stability
    for cell in cells:
        s, b, m = cell.values()
        ext = s != "BA500"
        si = s.lower() + "_ext"
        bi = b.lower()
        n = next(d["final_target_count"] for d in datasets if d["configuration"] == s)
        notes = (
            "Random is a sanity/reference baseline, not a learned explainer."
            if m == "Random"
            else "Frozen learned explainer result."
        )
        expressions = {}
        stds = {}
        if ext and m == "PGExplainer":
            pi = next(
                i
                for i, c in enumerate(read(PG)["rows"])
                if c["scale"] == si and c["backbone"] == bi
            )
            for metric in METRICS:
                key = PG_NAMES.get(metric, metric)
                expressions[metric] = ref(PG, "rows", pi, "cross_seed_metrics", key, "mean")
                stds[metric] = ref(PG, "rows", pi, "cross_seed_metrics", key, "std")
            semantics = "mean and sample std across five seed-level target macro aggregates (seeds0-4); not pooled targets"
            source = PG
            notes += " Seed42 contextual only. Saturation remains a scientific limitation."
        elif ext:
            ti = 0 if m == "GNNExplainer" else 2
            cols = read(AGG)["tables"][ti]["columns"]
            ri = next(
                i
                for i, a in enumerate(read(AGG)["tables"][ti]["rows"])
                if a[0] == si and a[1] == bi
            )
            keys = [
                "EdgeAUROC",
                "EdgeAP",
                "FidelityPlus",
                "FidelityMinus",
                "OneMinusFidelityMinus",
                "RemoveDelta",
                "RetainProbability",
                "RetainDelta",
                "RetainRatio",
                "Sparsity",
            ]
            for metric, key in zip(METRICS, keys, strict=True):
                expressions[metric] = ref(AGG, "tables", ti, "rows", ri, cols.index(key))
                stds[metric] = None
            semantics = (
                "frozen seed42 target macro point estimate; std unavailable in source aggregate"
                if m == "GNNExplainer"
                else "frozen mean over10 random seeds per target then target macro mean; AUROC/AP SD over pooled target-seed rows"
            )
            if m == "Random":
                for metric, key in (("edge_auroc", "EdgeAUROC_SD"), ("edge_ap", "EdgeAP_SD")):
                    stds[metric] = ref(AGG, "tables", ti, "rows", ri, cols.index(key))
            source = AGG
        elif m != "Random":
            ri = next(
                i
                for i, a in enumerate(read(MAIN))
                if a["backbone"] == b and a["explainer"].startswith(m)
            )
            for metric in METRICS:
                key = HIST_NAMES.get(metric, metric)
                expressions[metric] = (
                    csv_metric(MAIN, ri, key) if key + "_mean" in read(MAIN)[ri] else None
                )
                stds[metric] = (
                    csv_metric(MAIN, ri, key, "std") if key + "_std" in read(MAIN)[ri] else None
                )
            raw = {
                ("GCN", "GNNExplainer"): "results/metrics/frozen_phase4c_v2/per_node_metrics.csv",
                (
                    "GCN",
                    "PGExplainer",
                ): "results/metrics/pgexplainer_gcn_seed42/per_node_metrics.csv",
                (
                    "GraphSAGE",
                    "GNNExplainer",
                ): "results/metrics/graphsage_gnnexplainer/seed42/per_node_metrics.csv",
                (
                    "GraphSAGE",
                    "PGExplainer",
                ): "results/metrics/graphsage_pgexplainer/seed42/per_node_metrics.csv",
            }[b, m]
            expressions["retain_probability"] = row_mean(raw, "retain_probability", ids)
            stds["retain_probability"] = None
            semantics = "historical seed42 target mean and sample SD across12 targets; NOT across-seed uncertainty"
            source = MAIN
            if m == "PGExplainer" and b == "GraphSAGE":
                notes += (
                    " Natural106/35 training/validation pools; common-pool sensitivity excluded."
                )
        else:
            if b == "GraphSAGE":
                for metric in METRICS:
                    key = HIST_NAMES.get(metric, metric)
                    expressions[metric] = (
                        ref(RAUDIT, "summary_node_level", key, "mean")
                        if key in read(RAUDIT)["summary_node_level"]
                        else None
                    )
                    stds[metric] = (
                        ref(RAUDIT, "summary_node_level", key, "std")
                        if key in read(RAUDIT)["summary_node_level"]
                        else None
                    )
                source = RAUDIT
            else:
                source = "results/metrics/ba_house_2x2/random_node_level_summary.csv"
                for metric in METRICS:
                    key = HIST_NAMES.get(metric, metric)
                    expressions[metric] = row_mean(source, key, ids)
                    stds[metric] = (
                        calc("sample_std", *(ref(source, i, key) for i in range(len(read(source)))))
                        if expressions[metric]
                        else None
                    )
            raw = (
                "results/metrics/"
                + ("graphsage_random_baseline" if b == "GraphSAGE" else "phase4d_random_baseline")
                + "/random_baseline_per_node.csv"
            )
            expressions["retain_probability"] = row_mean(raw, "retain_probability", ids)
            stds["retain_probability"] = None
            semantics = "historical mean over10 seeds within target then macro mean/sample SD across12 target means; retain probability from raw rows"
        r = dict(
            cell,
            metric_source=source,
            n_targets=n,
            mean_std_semantics=semantics,
            primary_or_contextual="primary",
            notes=notes,
        )
        f = dict(cell, n_targets=n, aggregation_semantics=semantics, notes=notes)
        n_expr = next(
            x["expression"]
            for x in MAP
            if x["output_file"] == TABLE_NAMES[0] + ".json"
            and x["scale"] == s
            and x["metric"] == "final_target_count"
        )
        put(TABLE_NAMES[1], r, "n_targets", n_expr)
        put(TABLE_NAMES[2], f, "n_targets", n_expr)
        for metric in METRICS:
            dest, name = (r, TABLE_NAMES[1]) if metric in METRICS[:2] else (f, TABLE_NAMES[2])
            put(
                name,
                dest,
                metric,
                expressions[metric],
                "Historical source does not contain this metric"
                if expressions[metric] is None
                else None,
            )
            put(
                name,
                dest,
                metric + "_std",
                stds[metric],
                "Source does not provide this dispersion statistic"
                if stds[metric] is None
                else None,
            )
        recovery.append(r)
        faith.append(f)
        t = dict(cell, n_targets=n, notes=notes)
        put(TABLE_NAMES[3], t, "n_targets", n_expr)
        if ext and m == "PGExplainer":
            for stat in ("mean", "std", "min"):
                put(TABLE_NAMES[3], t, "jaccard_" + stat, ref(PG, "rows", pi, "stability", stat))
            t.update(
                number_of_seeds=5,
                seed_ids=[0, 1, 2, 3, 4],
                pair_semantics="10 unordered pairs/target; mean10 per target; distribution/sample SD across target means; seed42 excluded",
            )
        elif ext and m == "GNNExplainer":
            gs = BASE + "phase4a_gnn_random/03_gnnexplainer_stability_results.json"
            gi = next(
                i
                for i, c in enumerate(read(gs)["target_jaccard"])
                if c["scale"] == si and c["backbone"] == bi
            )
            for stat in ("mean", "std"):
                put(TABLE_NAMES[3], t, "jaccard_" + stat, ref(gs, "target_jaccard", gi, stat))
            vals = read(gs)["target_jaccard"][gi]["values"]
            put(
                TABLE_NAMES[3],
                t,
                "jaccard_min",
                calc(
                    "min", *(ref(gs, "target_jaccard", gi, "values", k) for k in range(len(vals)))
                ),
            )
            t.update(
                number_of_seeds=5,
                seed_ids=[0, 1, 2, 3, 4],
                pair_semantics="10 unordered independent optimization pairs/target; distribution/sample SD across target means",
            )
        elif ext:
            cols = read(AGG)["tables"][2]["columns"]
            ri = next(i for i, a in enumerate(read(AGG)["tables"][2]["rows"]) if a[:2] == [si, bi])
            put(
                TABLE_NAMES[3],
                t,
                "jaccard_mean",
                ref(AGG, "tables", 2, "rows", ri, cols.index("Jaccard")),
            )
            for stat in ("std", "min"):
                put(
                    TABLE_NAMES[3],
                    t,
                    "jaccard_" + stat,
                    None,
                    "Frozen Random aggregate reports mean Jaccard only; no invented SD/min",
                )
            t.update(
                number_of_seeds=10,
                seed_ids=list(range(10)),
                pair_semantics="45 unordered random-score pairs/target; macro target mean; SD/min unavailable in frozen aggregate",
            )
        elif m == "Random":
            rs = (
                "results/metrics/"
                + ("graphsage_random_baseline" if b == "GraphSAGE" else "phase4d_random_baseline")
                + "/random_stability_summary.csv"
            )
            for stat in ("mean", "std", "min"):
                put(TABLE_NAMES[3], t, "jaccard_" + stat, ref(rs, 0, stat))
            t.update(
                number_of_seeds=10,
                seed_ids=list(range(10)),
                pair_semantics="45 unordered random-score pairs/target; distribution/sample SD across target means",
            )
        else:
            hi = next(
                i
                for i, a in enumerate(read(STAB))
                if a["backbone"] == b and a["explainer"].startswith(m)
            )
            for stat in ("mean", "std", "min"):
                put(TABLE_NAMES[3], t, "jaccard_" + stat, ref(STAB, hi, "jaccard_" + stat))
            t.update(
                number_of_seeds=5,
                seed_ids=list(range(5)),
                pair_semantics=read(STAB)[hi]["stochastic_source"]
                + "; "
                + read(STAB)[hi]["seeds_pairs"]
                + "; distribution/sample SD across target means",
            )
        if ext and m == "PGExplainer":
            seed_expr = ref(PG, "rows", pi, "seeds")
        elif s == "BA500" and m == "Random":
            seed_expr = ref(
                "results/metrics/phase4d_random_baseline/random_stability_manifest.json",
                "random_seeds",
            )
        else:
            seed_expr = ref(
                BASE + "phase4a_gnn_random/01_gnnexplainer_protocol_receipt.json",
                "phase4d",
                "random_baseline_seeds" if m == "Random" else "stability_seeds",
            )
        put(TABLE_NAMES[3], t, "number_of_seeds", calc("length", seed_expr))
        put(TABLE_NAMES[3], t, "seed_ids", seed_expr)
        stability.append(t)
    table(
        TABLE_NAMES[1],
        recovery,
        "SD semantics differ by row and are not interchangeable. Extension PG: official seeds0-4. BA500 PG: historical seed42 recovery. Random is baseline/reference. No statistical significance test.",
    )
    table(
        TABLE_NAMES[2],
        faith,
        "Fidelity+ = original-class flip after REMOVE (larger indicates necessity); Fidelity- = flip after RETAIN (smaller indicates sufficiency). Retain delta = p_original - p_retain; can be negative. Null means unavailable statistic, not zero. Physical edge deletion and unchanged x. Fixed20% candidate budget; sparsity is budget-driven, not evidence of learned adaptive compactness.",
    )
    table(
        TABLE_NAMES[3],
        stability,
        "High Jaccard does not establish correctness. No arbitrary good/bad threshold. Seed42 excluded from official five-seed stability. Random is reference only; absent extension SD/min remain null.",
    )

    saturation = []
    TABLES[TABLE_NAMES[4]] = saturation
    for i, c in enumerate(read(PG)["rows"]):
        r = {
            "scale": c["scale"].split("_")[0].upper(),
            "backbone": "GCN" if c["backbone"] == "gcn" else "GraphSAGE",
            "explainer": "PGExplainer",
            "seeds": c["seeds"],
        }
        put(TABLE_NAMES[4], r, "seeds", ref(PG, "rows", i, "seeds"))
        for output, source in {
            "n_targets": "N_targets",
            "mean_saturated_fraction": "saturated_fraction_mean",
            "targets_saturated_5_of_5": "targets_saturated_5_of_5",
            "targets_saturated_at_least_4_of_5": "targets_saturated_at_least_4_of_5",
            "targets_never_saturated": "targets_never_saturated",
            "ranking_collapse_total": "ranking_collapse_total",
        }.items():
            put(TABLE_NAMES[4], r, output, ref(PG, "rows", i, source))
        r["interpretation"] = (
            "Persistent across seeds for all targets"
            if c["backbone"] == "gcn"
            else "Present but much less severe; persistent subset, not all targets"
        )
        r["scientific_limitation"] = (
            "Saturation unresolved; absolute score magnitude requires caution. Saturation is NOT ranking collapse and does not automatically invalidate ranking/topK."
        )
        saturation.append(r)
    table(
        TABLE_NAMES[4],
        saturation,
        "Cross-seed saturation table covers BA1000/1500 only. No equivalent frozen BA500 five-seed saturation summary is asserted. Zero saturation: at least95% candidate scores below0.01; collapse: unique score count<=1. Criteria are inherited from BA4B2, not new quality thresholds.",
    )

    def metric_expression(index, metric):
        source_table = (
            TABLE_NAMES[1]
            if metric in METRICS[:2]
            else TABLE_NAMES[3]
            if metric == "stability"
            else TABLE_NAMES[2]
        )
        field = "jaccard_mean" if metric == "stability" else metric
        return next(
            x["expression"]
            for x in MAP
            if x["output_file"] == source_table + ".json"
            and x["output_field"] == ["rows", index, field]
        )

    comparisons = []
    TABLES[TABLE_NAMES[5]] = comparisons
    comparison_metrics = ["edge_auroc", "edge_ap", "fid_plus", "fid_minus", "stability"]
    for s in ("BA500", "BA1000", "BA1500"):
        for m in ("GNNExplainer", "PGExplainer", "Random"):
            indices = [
                next(
                    i
                    for i, c in enumerate(cells)
                    if c == {"scale": s, "backbone": b, "explainer": m}
                )
                for b in ("GCN", "GraphSAGE")
            ]
            for metric in comparison_metrics:
                r = {"scale": s, "explainer": m, "metric": metric}
                a, z = [metric_expression(i, metric) for i in indices]
                put(TABLE_NAMES[5], r, "GCN", a)
                put(TABLE_NAMES[5], r, "GraphSAGE", z)
                put(TABLE_NAMES[5], r, "difference_GraphSAGE_minus_GCN", calc("difference", z, a))
                r["interpretation_scope"] = (
                    "Descriptive association within configuration; not causal. BA500 PG natural pools differ; mean/SD semantics may differ across methods."
                )
                comparisons.append(r)
    table(
        TABLE_NAMES[5],
        comparisons,
        "Difference is GraphSAGE minus GCN; it is arithmetic, not a causal effect or significance estimate.",
    )
    methods = []
    TABLES[TABLE_NAMES[6]] = methods
    for s in ("BA500", "BA1000", "BA1500"):
        for b in ("GCN", "GraphSAGE"):
            indices = [
                next(
                    i
                    for i, c in enumerate(cells)
                    if c == {"scale": s, "backbone": b, "explainer": m}
                )
                for m in ("GNNExplainer", "PGExplainer", "Random")
            ]
            for metric in comparison_metrics:
                r = {"scale": s, "backbone": b, "metric": metric}
                exprs = [metric_expression(i, metric) for i in indices]
                for m, expression in zip(
                    ("GNNExplainer", "PGExplainer", "Random"), exprs, strict=True
                ):
                    put(TABLE_NAMES[6], r, m, expression)
                put(TABLE_NAMES[6], r, "highest_value", calc("max", *exprs))
                r["highest_value_methods"] = ", ".join(
                    m
                    for m in ("GNNExplainer", "PGExplainer", "Random")
                    if r[m] == r["highest_value"]
                )
                r["interpretation_note"] = (
                    "Numerically highest only; Fidelity- is preferably smaller. Random is reference, not learned. No global winner; aggregation differs."
                )
                methods.append(r)
    table(
        TABLE_NAMES[6],
        methods,
        "Method rankings depend on metric and backbone; highest Fidelity- is not a quality award. Recovery/fidelity can differ from stability. No universal winner.",
    )
    scales = []
    TABLES[TABLE_NAMES[7]] = scales
    for i, cell in enumerate(cells):
        r = dict(cell)
        for metric in comparison_metrics + ["sparsity"]:
            put(TABLE_NAMES[7], r, metric, metric_expression(i, metric))
        r["interpretation"] = (
            "Across independently generated BA-House configurations; descriptive robustness only. BA500 historical selector/pools differ."
        )
        scales.append(r)
    table(
        TABLE_NAMES[7],
        scales,
        "Các configuration BA500, BA1000, BA1500 không phải nested sequence. Không diễn giải sự thay đổi metric như causal effect của graph size. Chỉ kiểm tra robustness của pattern explainer/backbone; GT prevalence, candidate complexity, target chronology and training pools also differ.",
    )
    return datasets, recovery, faith, stability, saturation


def narratives(datasets, recovery, faith, stability, saturation):
    note = "PG extension: mean ± sample SD across five seed-level aggregates (0-4). BA500: frozen seed42 mean ± target SD. Extension GNNExplainer: frozen aggregate, no invented SD. Random: baseline/reference; historical target SD versus extension pooled target-seed SD for recovery. These SDs are not directly comparable. NA = unavailable."

    def compact(rows, metrics):
        output = []
        for r in rows:
            x = {k: r[k] for k in ("scale", "backbone", "explainer")}
            for metric in metrics:
                x[metric] = "NA" if r[metric] is None else f"{r[metric]:.4f}"
                if r.get(metric + "_std") is not None:
                    x[metric] += " ± " + formatted(r[metric + "_std"])
            output.append(x)
        return markdown(output)

    recovery_md = compact(recovery, METRICS[:2])
    faith_md = compact(faith, ["fid_plus", "fid_minus", "one_minus_fid_minus", "sparsity"])
    probability_md = compact(
        faith,
        [
            "remove_probability_delta",
            "retain_probability",
            "retain_delta",
            "retain_probability_ratio",
        ],
    )
    stability_md = compact(stability, ["jaccard_mean", "jaccard_min"])
    # Jaccard dispersion is across target means, never across seed-level metric aggregates.
    stability_rows = [
        dict(
            scale=r["scale"],
            backbone=r["backbone"],
            explainer=r["explainer"],
            Jaccard=formatted(r["jaccard_mean"])
            + (" ± " + formatted(r["jaccard_std"]) if r["jaccard_std"] is not None else ""),
            minimum=formatted(r["jaccard_min"]),
            seeds=r["number_of_seeds"],
            targets=r["n_targets"],
        )
        for r in stability
    ]
    stability_md = markdown(stability_rows)
    saturation_md = markdown(
        [dict(r, mean_saturated_fraction=float(r["mean_saturated_fraction"])) for r in saturation],
        [
            "scale",
            "backbone",
            "n_targets",
            "mean_saturated_fraction",
            "targets_saturated_5_of_5",
            "ranking_collapse_total",
        ],
    )
    for filename, content, foot in (
        ("11_thesis_table_recovery.md", recovery_md, note),
        (
            "12_thesis_table_faithfulness.md",
            faith_md,
            note + " Fidelity- smaller indicates retention sufficiency; sparsity budget-driven.",
        ),
        (
            "13_thesis_table_stability.md",
            stability_md,
            "Mean ± sample SD across target means of pairwise Jaccard;5 seeds=10 pairs, Random10 seeds=45 pairs. Extension Random SD/min unavailable; seed42 excluded from official stability.",
        ),
        (
            "14_thesis_table_saturation.md",
            saturation_md,
            "PG seeds0-4 only; fraction averaged across seeds. BA500 comparable cross-seed saturation summary unavailable. Saturation is not collapse; limitation remains.",
        ),
    ):
        write_md(
            filename,
            "# " + filename[3:-3].replace("_", " ").title() + "\n\n" + content + "\n" + foot,
        )
    rq = [
        {
            "rq": "RQ1",
            "title": "Ground-truth recovery vs Random",
            "result": "GNNExplainer và PGExplainer có AUROC/AP cao hơn Random trong các cell được tổng hợp. Với GCN, recovery của GNNExplainer cao hơn PG; với GraphSAGE, recovery của PG cao hơn GNN.",
            "interpretation": "Synthetic edge GT cho phép đối chiếu ranking AUROC/AP với baseline/reference. Không có universal winner cho mọi metric.",
            "limitation": "Cohort chọn lọc, GT prevalence và candidate complexity khác nhau; mean/SD semantics khác nhau; không phải kiểm định ý nghĩa thống kê.",
        },
        {
            "rq": "RQ2",
            "title": "Backbone association",
            "result": "Thứ hạng recovery tương đối của GNNExplainer và PGExplainer thay đổi giữa cấu hình GCN và GraphSAGE.",
            "interpretation": "Đây là descriptive association với backbone, không chứng minh tác động nhân quả.",
            "limitation": "BA500 PG natural training pools không bằng nhau; extension common pools giảm confound membership nhưng không tạo causal identification.",
        },
        {
            "rq": "RQ3",
            "title": "Fidelity, stability and sparsity",
            "result": "PG có Jaccard cao hơn GNN trong các cell, kể cả extension. Fidelity và recovery không luôn có cùng thứ hạng; sparsity tương đương theo budget cố định.",
            "interpretation": "Stability phải đọc cùng recovery/fidelity; Fidelity+ phản ánh REMOVE necessity và Fidelity- phản ánh RETAIN flip, nên nhỏ hơn thể hiện sufficiency tốt hơn.",
            "limitation": "GCN saturation vẫn persistent across seeds; high Jaccard không đủ chứng minh score magnitude đáng tin hoặc explanation đúng. Sparsity chủ yếu bị quyết định bởi budget.",
        },
        {
            "rq": "RQ4",
            "title": "Practical implication for transaction classification",
            "result": "BA-House cung cấp kiểm tra phương pháp trên synthetic GT, không phải bằng chứng trực tiếp về giao dịch thực.",
            "interpretation": "Không chọn explainer bằng một metric. Kiểm tra recovery khi có GT; fidelity/stability khi thiếu GT; kiểm tra backbone sensitivity và audit score saturation. Explanation ổn định không đồng nghĩa explanation đúng.",
            "limitation": "Recommendation là implication for later real-world evaluation, cần đánh giá thực riêng; không nhập metric ngoài BA-House.",
        },
    ]
    write_json("09_rq_summary.json", {"research_questions": rq, "evidence_tables": TABLE_NAMES[1:]})
    rqmd = "\n\n".join(
        "## "
        + r["rq"]
        + " — "
        + r["title"]
        + "\n\nResult: "
        + r["result"]
        + "\n\nInterpretation: "
        + r["interpretation"]
        + "\n\nLimitation: "
        + r["limitation"]
        for r in rq
    )
    write_md("09_rq_summary.md", "# RQ Summary\n\n" + rqmd)
    sections = [
        (
            "1. Scope and role of BA-House",
            "Chỉ tổng hợp frozen BA500, BA1000 và BA1500. Không train/retrain, chạy explainer hay sửa metric. Random là sanity/reference baseline, không phải learned explainer.",
        ),
        (
            "2. Dataset configurations",
            markdown(
                datasets,
                [
                    "configuration",
                    "role",
                    "nodes",
                    "directed_edges",
                    "feature_dimension",
                    "train_split",
                    "validation_split",
                    "test_split",
                    "final_target_count",
                ],
            )
            + "\nBA500 là historical frozen benchmark; BA500_EXT reproduction chỉ là generation control, không thay experiment lịch sử. Các configuration độc lập, không nested.",
        ),
        (
            "3. Backbone performance and frozen target protocol",
            markdown(
                datasets,
                [
                    "configuration",
                    "gcn_validation_accuracy",
                    "gcn_test_accuracy",
                    "graphsage_validation_accuracy",
                    "graphsage_test_accuracy",
                ],
            )
            + "\n"
            + markdown(
                datasets,
                [
                    "configuration",
                    "gcn_pg_train_pool",
                    "gcn_pg_val_pool",
                    "graphsage_pg_train_pool",
                    "graphsage_pg_val_pool",
                ],
            )
            + "\nBA500: lọc GCN-correct motif test, sort confidence, shuffle seeded, chia holdout/final; GraphSAGE parity hậu kiểm. Extension: structurally-clean, common-correct, class1, seeded CPU selection trước explanations. BA500 PG natural pools khác backbone; extension có common pools. Cohort không đại diện mọi class/node.",
        ),
        (
            "4. Ground-truth recovery",
            recovery_md
            + "\n"
            + note
            + "\n"
            + rq[0]["result"]
            + " Không suy ra chất lượng absolute score từ AUROC/AP.",
        ),
        (
            "5. Faithfulness and compactness",
            faith_md
            + "\n"
            + probability_md
            + "\n"
            + note
            + "\nFidelity+ lớn hơn thể hiện REMOVE necessity; Fidelity- nhỏ hơn thể hiện RETAIN sufficiency. Các delta xác suất có thể âm; không diễn giải prediction flip như probability drop. GCN có REMOVE flip dù mean probability delta nhỏ hoặc âm; vì vậy cần đọc binary và continuous metrics cùng nhau. X giữ nguyên; physical REMOVE/RETAIN edge columns. Cùng top20% candidate budget, sparsity không thể hiện ưu thế adaptive compactness.",
        ),
        (
            "6. Stability",
            stability_md
            + "\nOfficial PG extension seeds0-4: mean10 unordered pair Jaccards per target, cell distribution/sample SD across target means. Seed42 không nhập official stability. GNN stochastic mask optimization, PG independent training, Random random scores: khác nguồn stochasticity. Không đặt good/bad threshold.",
        ),
        (
            "7. PGExplainer saturation diagnostics",
            saturation_md
            + "\nGCN saturation persistent trên mọi target cả năm seed. GraphSAGE saturation present but much less severe, vẫn có subset persistent. Ranking collapse total bằng0 không nghĩa saturation đã giải quyết. Absolute score magnitude cần thận trọng; saturation không tự động invalidate ranking/topK. BA500 không có summary cross-seed tương đương trong bảng này.",
        ),
        (
            "8. Backbone sensitivity",
            rq[1]["result"]
            + " "
            + rq[1]["interpretation"]
            + "\n\nXem 06_backbone_comparison và 07_explainer_comparison cho từng metric/cell; highest value chỉ là số lớn nhất, không phải giải thưởng chất lượng. Đặc biệt Fidelity- nhỏ hơn mới thuận lợi.",
        ),
        (
            "9. Scale robustness",
            "Across the independently generated BA-House configurations, recovery ranking theo backbone và PG stability cao hơn GNN vẫn xuất hiện. Các configuration BA500, BA1000, BA1500 không phải nested sequence. Không diễn giải metric thay đổi như causal effect của graph size. Khác target chronology, training pools, GT prevalence và candidate complexity giới hạn so sánh; xem 08_scale_robustness.\n\n"
            + markdown(
                datasets,
                [
                    "configuration",
                    "final_target_count",
                    "mean_candidate_edges_final",
                    "mean_gt_prevalence_final",
                ],
            )
            + "\nGT prevalence được macro average theo target, không phải pooled edge prevalence. AP baseline chịu ảnh hưởng của prevalence; không diễn giải AP cross-scale như bài toán giống nhau hoàn toàn.",
        ),
        (
            "10. Answers to RQ1–RQ3",
            "\n\n".join(
                r["rq"] + ": " + r["result"] + " " + r["interpretation"] + " " + r["limitation"]
                for r in rq[:3]
            ),
        ),
        (
            "11. Implications for RQ4",
            rq[3]["result"] + " " + rq[3]["interpretation"] + " " + rq[3]["limitation"],
        ),
        (
            "12. Limitations",
            "Correctness-conditioned class1 cohorts; fixed seed backbone/dataset realizations; independent non-nested graphs; historical selector/pool chronology; heterogeneous SD semantics; Random missing extension Jaccard SD/min; fixed budget; no statistical tests; synthetic-to-transaction transfer not established. PG GCN saturation unresolved. Historical primary exact-equality verifier INVALID remains archived; audit attributed differences to device numerics. Subsequent numerical-equivalence policy accepted primary with saturation warning, did not retrain or rewrite metrics. BA4B2 independent verifier PASS is a separate verdict; no claim that historical INVALID receipt was changed. Seed42 primary historical reference is contextual for PG extension, not a sixth stability seed.",
        ),
        (
            "13. Final BA-House conclusion",
            "Không có explainer thắng tuyệt đối. Recovery, fidelity và stability phải đọc cùng nhau, theo backbone và frozen cohort. PG cho stability cao nhưng GCN saturation vẫn là limitation; kết quả descriptive robustness không xác lập causal graph-size effect hay hiệu quả trực tiếp trên giao dịch thực.",
        ),
    ]
    write_json(
        "10_BA_HOUSE_MASTER_REPORT.json",
        {
            "title": "BA-House Master Report",
            "sections": [{"heading": h, "body": b} for h, b in sections],
            "tables": TABLES,
            "research_questions": rq,
            "experiment_executed": False,
            "historical_artifacts_modified": False,
        },
    )
    write_md(
        "10_BA_HOUSE_MASTER_REPORT.md",
        "# BA-House Master Report\n\n" + "\n\n".join("## " + h + "\n\n" + b for h, b in sections),
    )
    paras = [
        (
            "A. Recovery BA-House",
            rq[0]["result"] + " " + rq[0]["interpretation"] + " " + rq[0]["limitation"],
        ),
        (
            "B. Backbone sensitivity",
            rq[1]["result"] + " " + rq[1]["interpretation"] + " " + rq[1]["limitation"],
        ),
        (
            "C. Fidelity",
            "Fidelity không luôn đồng thuận với recovery; tại BA1500 GraphSAGE, PG recovery cao hơn GNN nhưng Fidelity+ thấp hơn. Điều này phân biệt khớp motif GT với necessity/sufficiency đối với backbone. Prediction flip không thay thế delta xác suất, và sparsity theo cùng budget không chứng minh ưu thế compactness.",
        ),
        (
            "D. Stability",
            rq[2]["result"] + " " + rq[2]["interpretation"] + " " + rq[2]["limitation"],
        ),
        (
            "E. PG saturation",
            "PG GCN có mọi target saturated ở mọi seed official trong cả hai extension; ranking collapse không được ghi nhận. Stability cao vì thế có thể đồng thời tồn tại với absolute score magnitude rất nhỏ. Saturation vẫn là limitation chưa giải quyết; GraphSAGE cũng có subset saturated, không được mô tả là không saturation.",
        ),
        ("F. Scale robustness", sections[8][1].split("\n\n")[0]),
        ("G. Overall BA-House conclusion", sections[12][1]),
    ]
    write_md(
        "15_thesis_ready_results.md",
        "# Draft Results for Chapter 4\n\n" + "\n\n".join("## " + h + "\n\n" + b for h, b in paras),
    )
    write_md(
        "00_README.md",
        "# BA-House Master Consolidation\n\nRead-only source consolidation, no scientific execution. Start with 10_BA_HOUSE_MASTER_REPORT.md; 11-14 are compact four-decimal thesis tables and 15 is a Vietnamese draft. 01-08 JSON/CSV preserve full precision and explicit nulls; 16 maps every numeric metric to artifact fields and documented aggregation. 17 is an independent reporting verifier, not a new experiment verifier.\n\nBA500 chronology stays historical; BA1000/1500 are independent seeded configurations, not nested. PG extension primary recovery/fidelity uses five official seed-level aggregates; seed42 contextual only. Random baseline and unresolved GCN saturation are explicitly separated.\n\nRebuild: .venv/Scripts/python.exe -B docs/extension/ba_house_scale/master_report/build_master_report.py\nVerify: .venv/Scripts/python.exe -B docs/extension/ba_house_scale/master_report/verify_ba_house_master_report.py",
    )


def main():
    data = collect()
    print("Collected frozen evidence; hashing protected inputs", flush=True)
    before = protection()
    write_json("protected_inputs_before.json", {"files": before})
    narratives(*data)
    write_json(
        "16_source_map.json",
        {
            "sources": list(SOURCES.values()),
            "numeric_mappings": MAP,
            "mapping_contract": "JSON field paths; CSV index is zero-based data-row; mean/sample_std are reporting aggregation only. No metric implementation or experiment is executed.",
            "context_sources": [BASE + "phase4b2_pg_stability/09_seed42_context.json"],
            "phase_coverage": [
                "BA500 Task20 and historical GNN/PG/Random/stability",
                "BA-1",
                "BA-2",
                "BA-3",
                "BA-4A",
                "BA-4B1",
                "BA-4B1 audit",
                "BA-4B1 policy",
                "BA-4B2",
            ],
        },
    )
    lines = [
        "# Source Map",
        "",
        "Every numeric table value is traced below; repeated narrative/thesis values reuse these table cells. Protocol/chronology evidence is listed in source inventory. No memory/manual-note numeric source.",
        "",
    ]
    lines += ["- `" + p + "` SHA256 `" + record["sha256"] + "`" for p, record in SOURCES.items()]
    lines += [
        "",
        "## Numeric Field Mappings",
        "",
        markdown(
            [
                {
                    "output": m["output_file"],
                    "field": json.dumps(m["output_field"]),
                    "scale": m["scale"],
                    "backbone": m["backbone"],
                    "explainer": m["explainer"],
                    "source": json.dumps(m["expression"], ensure_ascii=False)
                    if m["expression"]
                    else "NULL: " + m["missing_reason"],
                }
                for m in MAP
            ]
        ),
    ]
    write_md("16_source_map.md", "\n".join(lines))
    after = {p: digest(ROOT / p) for p in before}
    write_json("protected_inputs_after.json", {"files": after, "unchanged": before == after})
    assert before == after
    write_json(
        "build_receipt.json",
        {
            "status": "BA_HOUSE_MASTER_REPORT_BUILD_COMPLETE",
            "sources": len(SOURCES),
            "protected_files": len(before),
            "numeric_mappings": len(MAP),
            "row_counts": {k: len(v) for k, v in TABLES.items()},
            "scientific_execution": False,
        },
    )
    print(
        json.dumps(
            {
                "sources": len(SOURCES),
                "protected_files": len(before),
                "row_counts": {k: len(v) for k, v in TABLES.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
