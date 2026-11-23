"""Explicit comparison plans, repeated observations, and reused spatial measurements."""
from copy import deepcopy
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def enabled(c):
    return bool(c.get("split_by") or c.get("extra_comparisons") or c.get("change", {}).get("enabled"))


def validate_design(c, names):
    columns = c.get("split_by", [])
    if not isinstance(columns, list) or len(columns) > 3 or len(set(columns)) != len(columns):
        raise ValueError("Choose up to three different columns to split comparisons by.")
    if c.get("split_depth", 1) not in (1, 2, 3):
        raise ValueError("Combine one, two, or three splitting columns.")
    if not c.get("include_overall", True) and not columns:
        raise ValueError("Choose splitting columns, or include the pooled comparison.")
    required = list(columns)
    extras = c.get("extra_comparisons", [])
    if not isinstance(extras, list) or len(extras) > 5:
        raise ValueError("Add up to five comparison columns.")
    for item in extras:
        required.append(item.get("group_col", ""))
        if not item.get("all_pairs") and (not item.get("group_a") or not item.get("group_b") or item["group_a"] == item["group_b"]):
            raise ValueError("Choose two different groups for every added comparison, or select All pairs.")
    groups = [e.get("group_col") for e in [c["endpoint"], *extras]
              if e.get("type", "binary") == "binary"]
    if c["sample_col"] in groups:
        raise ValueError("The comparison column must differ from the independent sample ID. Compare categories represented by multiple independent samples.")
    change = c.get("change", {})
    if change.get("enabled"):
        required.append(change.get("column", ""))
        if c["endpoint"]["type"] != "binary" or c["endpoint"]["paired"]:
            raise ValueError("Compare changes between two independent groups; the before/after observations are matched within each sample.")
        if not change.get("before") or not change.get("after") or change["before"] == change["after"]:
            raise ValueError("Choose different before and after observations.")
        if change.get("column") in (c["sample_col"], c["endpoint"].get("group_col")):
            raise ValueError("Before/after observations need a separate column from the independent sample ID and comparison groups.")
    if any(not name or name not in names for name in required):
        raise ValueError("Choose valid comparison, splitting, and repeated-observation columns.")
    return required


def plan(frame, c):
    """Enumerate only selected questions and observed strata, never all table columns."""
    validate_design(c, frame.columns)
    endpoints = [deepcopy(c["endpoint"])]
    for extra in c.get("extra_comparisons", []):
        values = sorted(frame[extra["group_col"]].dropna().astype(str).unique())
        if extra.get("all_pairs") and len(values) > 12:
            raise ValueError(f"{extra['group_col']} has more than 12 groups. Choose a specific pair rather than scanning identifiers.")
        pairs = combinations(values, 2) if extra.get("all_pairs") else [(extra["group_a"], extra["group_b"])]
        for a, b in pairs:
            e = dict(c["endpoint"], type="binary", group_col=extra["group_col"], group_a=a, group_b=b,
                     paired=bool(extra.get("paired")), change_col="", change_before="", change_after="")
            endpoints.append(e)
    change = c.get("change", {})
    if change.get("enabled"):
        endpoints.append(dict(c["endpoint"], change_col=change["column"], change_before=change["before"], change_after=change["after"]))
    columns = c.get("split_by", [])
    for col in columns:
        if frame[col].nunique() > 16:
            raise ValueError(f"{col} has more than 16 values. Choose a biological grouping rather than an image/cell identifier.")
    strata = [{}] if c.get("include_overall", True) else []
    for depth in range(1, min(c.get("split_depth", 1), len(columns)) + 1):
        for group in combinations(columns, depth):
            strata.extend(frame[list(group)].dropna().astype(str).drop_duplicates().to_dict("records"))
    plans, seen = [], set()
    for e in endpoints:
        for stratum in strata:
            # Fixing the comparison variable to one value cannot compare it.
            # Fixing the before/after column cannot estimate a within-person change.
            excluded = {e.get("group_col"), e.get("change_col")} if e["type"] == "binary" else set()
            if excluded & set(stratum):
                continue
            identity = json.dumps([e, stratum], sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            selected = frame
            for col, value in stratum.items():
                selected = selected[selected[col].astype("string") == value]
            reason = ""
            counts = {}
            if e["type"] == "binary":
                selected = selected[selected[e["group_col"]].astype("string").isin([e["group_a"], e["group_b"]])]
                people = selected[[c["sample_col"], e["group_col"]]].dropna().astype(str).drop_duplicates()
                counts = {str(k): int(v) for k, v in people.groupby(e["group_col"])[c["sample_col"]].nunique().items()}
                if not e["paired"] and (people.groupby(c["sample_col"])[e["group_col"]].nunique() > 1).any():
                    reason = "The same independent IDs occur in both groups; choose a matched comparison."
                if e["paired"]:
                    counts["matched"] = int((people.groupby(c["sample_col"])[e["group_col"]].nunique() == 2).sum())
                if min(counts.get(e["group_a"], 0), counts.get(e["group_b"], 0), counts.get("matched", float("inf"))) < c["min_samples"]:
                    reason = "Too few independent samples before zone coverage checks."
                if e.get("change_col"):
                    visits = selected[selected[e["change_col"]].astype("string").isin([e["change_before"], e["change_after"]])]
                    complete = visits.groupby(c["sample_col"])[e["change_col"]].nunique()
                    counts["complete_observations"] = int((complete == 2).sum())
                    complete_people = people[people[c["sample_col"]].isin(complete[complete == 2].index.astype(str))]
                    complete_counts = complete_people.groupby(e["group_col"])[c["sample_col"]].nunique()
                    counts["complete_by_group"] = {str(k): int(v) for k, v in complete_counts.items()}
                    if min(complete_counts.get(e["group_a"], 0), complete_counts.get(e["group_b"], 0)) < c["min_samples"]:
                        reason = "Too few independent samples with both observations before zone coverage checks."
            label = (f"{e['group_col']}: {e['group_a']} vs {e['group_b']}" if e["type"] == "binary" else e["type"])
            if e.get("change_col"):
                label += f" · change {e['change_after']} − {e['change_before']} ({e['change_col']})"
            if e.get("paired"):
                label += " · matched"
            label += " · " + (", ".join(f"{k}={v}" for k, v in stratum.items()) or "pooled")
            plans.append({"id": "comparison_" + hashlib.sha256(identity.encode()).hexdigest()[:12],
                          "label": label, "endpoint": e, "stratum": stratum, "samples": counts,
                          "status": "skipped" if reason else "ready", "reason": reason})
            if len(plans) > 64:
                raise ValueError("This expands to more than 64 comparisons. Select fewer grouping columns, pairs, or combinations.")
    if not plans:
        raise ValueError("No comparisons remain. Include pooled comparisons or choose different splitting columns.")
    return plans


class MeasurementCache:
    """Per-run disk cache: identical cell populations share counts and null draws."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def __call__(self, part, c, mpp, report):
        from . import screening
        identity = part.index.to_numpy(np.int64).tobytes() + str(mpp).encode()
        name = hashlib.sha256(identity).hexdigest()
        path = self.folder / (name + ".parquet")
        metadata = self.folder / (name + ".json")
        if path.exists():
            self.hits += 1
            meta = json.loads(metadata.read_text())
            return pd.read_parquet(path).to_dict("records"), [{"measurement_cache": "reused", "image": str(part._image.iloc[0])}], meta["annotation"]
        self.misses += 1
        neutral = part.copy()
        neutral["_group"] = "Measurement"  # outcomes must not change the spatial null
        rows, timing, annotation = screening._measure(neutral, c, mpp, report)
        pd.DataFrame(rows).to_parquet(path, index=False)
        metadata.write_text(json.dumps({"annotation": annotation}))
        return rows, timing, annotation


def run(settings, out, frame, report):
    """Run a declared comparison set, reusing geometry and correcting it jointly."""
    from . import screening as s, screening_figures as figures, screening_statistics as statistics
    import time
    started = time.perf_counter()
    c = settings["values"]
    plans = plan(frame, c)
    ready = [p for p in plans if p["status"] == "ready"]
    if not ready:
        raise ValueError("No comparison has enough independent samples. Review the comparison preview and matching choices.")
    cache = MeasurementCache(out / "measurements")
    completed = {}
    records = []
    # All frame indices must describe the same source rows in every subset.
    frame = frame.reset_index(drop=True)

    def expose():
        if not completed:
            return
        first = next(iter(completed.values()))["payload"]
        payload = dict(first, run=out.name, scan_folder=str(out), scan_settings=c,
                       comparisons=plans, revision=str(time.time_ns()))
        s._atomic_json(out / "result.json", payload)

    for number, item in enumerate(ready):
        cfg = deepcopy(c)
        cfg.update(split_by=[], extra_comparisons=[], include_overall=True)
        cfg["change"]["enabled"] = False
        cfg["endpoint"] = item["endpoint"]
        cfg["filters"] += [{"column": col, "values": [value]} for col, value in item["stratum"].items()]
        selected = frame
        for col, value in item["stratum"].items():
            selected = selected[selected[col].astype("string") == value]
        target = out / item["id"]
        (target / "tables").mkdir(parents=True)
        child_settings = dict(settings, values=cfg)
        entry = completed[item["id"]] = {"cfg": cfg, "folder": target, "settings": child_settings}
        progress = lambda message, *_, n=number: report(f"Comparison {n+1}/{len(ready)} · {message}", None)
        def preview(results, merged, regions, audit, message):
            results = results.assign(q=np.nan)
            payload = s._display_result(child_settings, target, results, merged, regions, audit, [], True, message)
            payload.update(run=out.name, comparison_id=item["id"], comparison_label=item["label"], scan_settings=c)
            entry["payload"] = payload
            s._atomic_json(target / "result.json", payload)
            expose()
        work = dict(cfg, resampling="off")
        try:
            result, _, regions, audit, merged = s.compute(selected, work, settings["mpp"], progress, preview, cache)
        except s.NoEligibleCells as error:
            del completed[item["id"]]
            item.update(status="skipped", reason=str(error))
            expose()
            continue
        result["comparison"], result["comparison_label"] = item["id"], item["label"]
        records.append(result)
        if sum(len(r) for r in records) * len(plans) / len(records) > 200000:
            raise ValueError("This plan exceeds 200,000 feature-zone-comparison tests. Narrow the comparison plan.")
        merged.to_csv(target / "tables/sample_features.csv", index=False)
        regions.to_csv(target / "tables/region_features.csv", index=False)
        entry.update(audit=audit, result=result)
        preview(result, merged, regions, audit, "Comparison measured; correction awaits the complete comparison plan")
        item["status"] = "measured"
        expose()
        # Re-read on demand below, rather than keeping all regions in memory.
        del regions, merged

    if not records:
        raise ValueError("No comparisons have cells after the selected filters. Review the filters and image selection.")
    template = records[0]
    for item in plans:
        if item["status"] == "skipped":
            skipped = template.copy()
            for col in ["effect", "p", "q", "q_global", "q_family", "q_classic", "median_a", "median_b"]:
                skipped[col] = np.nan
            skipped["comparison"], skipped["comparison_label"] = item["id"], item["label"]
            skipped["status"], skipped["test"] = item["reason"], "not tested"
            skipped[["n_samples", "n_a", "n_b"]] = 0
            records.append(skipped)
    result = statistics.adjust(pd.concat(records, ignore_index=True), c)
    if c["resampling"] != "off":
        contexts = {key: (pd.read_csv(entry["folder"] / "tables/sample_features.csv", dtype={"_sample": str, "_group": str}), entry["cfg"]) for key, entry in completed.items()}
        result = statistics.resample_joint(result, contexts, c, s.infer, report)
        del contexts
    stamps = [stamp for entry in completed.values() for stamp in [*entry["audit"]["annotations"], *([entry["audit"]["area_source"]] if entry["audit"].get("area_source") else [])]]
    if s._table_stamp(settings["table"]) != settings["source"] or any(s._stamp(p["path"]) != p for p in stamps):
        raise ValueError("Inputs changed during screening. Run again on the saved inputs.")
    result.to_csv(out / "tables/all_comparisons.csv", index=False)
    result[result.q < c["alpha"]].to_csv(out / "tables/all_significant_features.csv", index=False)
    for number, (key, entry) in enumerate(completed.items()):
        selected = result[result.comparison == key]
        target, cfg, audit = entry["folder"], entry["cfg"], entry["audit"]
        merged = pd.read_csv(target / "tables/sample_features.csv", dtype={"_sample": str, "_group": str})
        regions = pd.read_csv(target / "tables/region_features.csv", dtype={"_sample": str, "_group": str, "_image": str, "_occasion": str})
        selected.to_csv(target / "tables/screening_results_all.csv", index=False)
        selected[selected.q < c["alpha"]].to_csv(target / "tables/significant_features.csv", index=False)
        report(f"Drawing comparison {number+1}/{len(completed)}", .9)
        cfg["_scan_hypotheses"] = len(result)
        cfg["_comparison_label"] = next(p["label"] for p in plans if p["id"] == key)
        saved_figures = figures.save(target, selected, merged, cfg, settings.get("palette"))
        audit["seconds"] = time.perf_counter() - started
        payload = s._display_result(entry["settings"], target, selected, merged, regions, audit, saved_figures)
        item = next(p for p in plans if p["id"] == key)
        item.update(status="complete", significant=payload["significant"], tested=payload["tested"])
        payload.update(run=out.name, comparison_id=key, comparison_label=item["label"], scan_settings=c, scan_folder=str(out), scan_hypotheses=len(result))
        entry["payload"] = payload
        s._atomic_json(target / "result.json", payload)
        s._atomic_json(target / "design.json", {"settings": cfg, "comparison": item, "audit": audit, "scan_hypotheses": len(result)})
        expose()
    expose()
    s._atomic_json(out / "design.json", {"settings": c, "comparisons": plans, "measurement_cache": {"computed": cache.misses, "reused": cache.hits}, "tests": len(result)})
    report("Finished. All comparisons and figures are ready.", 1.)
    return json.loads((out / "result.json").read_text())
