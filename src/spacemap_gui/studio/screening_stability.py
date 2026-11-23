"""Read-only, ranked distance overview of saved screening tests and recovery.

Ranking is descriptive prioritization, never a new test or validation claim.
Both the browser and exported figures consume this same representation.
"""
from io import BytesIO
from collections import OrderedDict
from pathlib import Path
import textwrap
import threading

import numpy as np
import pandas as pd

from . import screening
from .screening_figures import zone_label
from .screening_statistics import SCOPES

RANKING = ("Ranked by the number of FDR-significant zones with the same effect direction, "
           "then median exact recovery in those zones, then smallest q and p. "
           "This prioritizes statistical support, not biological importance. "
           "Overlapping zones are not independent confirmations; no new p-value is calculated.")
STATES = {
    "fdr": ("FDR significant", "#23649b", "#ffffff"),
    "nominal": ("Raw p < 0.05 only", "#ffe590", "#191919"),
    "tested": ("Not significant", "#f3f5f8", "#384454"),
    "missing": ("Not tested", "#cbd1d8", "#384454"),
}
_plot_lock = threading.RLock()
_overview_lock = threading.RLock()
_overviews = OrderedDict()


def number(value):
    try:
        n = float(value)
        return n if np.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def direction(endpoint, effect):
    effect = number(effect)
    kind = endpoint["type"]
    if effect is None or kind == "describe":
        return 0, "—", "Direction unavailable"
    sign = int(np.sign(effect - 1 if kind == "survival" else effect))
    if not sign:
        return 0, "=", "No estimated direction (the test can still detect distribution differences)"
    if kind == "binary":
        group = endpoint["group_b"] if sign > 0 else endpoint["group_a"]
        meaning = "Larger after-minus-before change in " if endpoint.get("change_col") else "Higher in "
        return sign, str(group), meaning + str(group)
    if kind == "survival":
        return sign, "↑ hazard" if sign > 0 else "↓ hazard", "Higher feature values associated with " + ("higher" if sign > 0 else "lower") + " event hazard"
    return sign, "Positive" if sign > 0 else "Negative", ("Positive" if sign > 0 else "Negative") + " association with " + endpoint["value_col"]


def comparison_name(e, stratum=None):
    if e["type"] == "binary":
        label = f"{e['group_col']}: {e['group_a']} vs {e['group_b']}"
        if e.get("change_col"):
            label += f" · change {e['change_after']} − {e['change_before']} ({e['change_col']})"
        elif e.get("paired"):
            label += " · matched"
    elif e["type"] == "survival":
        label = f"Survival: {e['time_col']} · event: {e['event_col']}"
    elif e["type"] == "continuous":
        label = "Association: " + e["value_col"]
    else:
        label = "Descriptive"
    return label + (" · " + ", ".join(f"{k}={v}" for k, v in stratum.items()) if stratum else " · pooled")


def build(tests, c, plans=None, palette=None):
    """Combine feature-zone rows, preserving saved p/q and actual endpoint roles."""
    tests = tests.copy()
    if "comparison" not in tests:
        tests["comparison"] = ""
    lookup = {p["id"]: p for p in plans or []}
    zones = list(range(1 if c["zone_source"] == "whole" else len(c["widths"])))
    rows = []
    for (comparison, family, feature), part in tests.groupby(["comparison", "family", "feature"], sort=False):
        item = lookup.get(comparison, {})
        e = item.get("endpoint", c["endpoint"])
        name = comparison_name(e, item.get("stratum"))
        cells = []
        for zone, r in part.sort_values("zone").set_index("zone").iterrows():
            p, q, effect = number(r.get("p")), number(r.get("q")), number(r.get("effect"))
            state = "missing" if p is None else "fdr" if q is not None and q < c["alpha"] else "nominal" if p < .05 else "tested"
            sign, short, meaning = direction(e, effect)
            group = short if e['type'] == 'binary' and sign else None
            colour = (palette or {}).get(group, '#e79562' if sign > 0 else '#71b8df') if group is not None else None
            recovery = number(r.get("recovery")) if r.get("resampling") in ("subsample", "bootstrap") else None
            near = number(r.get("nearby_recovery")) if recovery is not None else None
            cells.append({"zone": int(zone), "state": state, "sign": sign, "direction": meaning, "short": short,
                "effect": effect, "p": p, "q": q, "recovery": recovery, "nearby_recovery": near,
                "group": group, "group_colour": colour, "ci_low": number(r.get('ci_low')), "ci_high": number(r.get('ci_high')),
                "resampling": r.get("resampling") if recovery is not None else None,
                "repeats": number(r.get("resamples")), "testable_repeats": number(r.get("valid_resamples")),
                "samples": number(r.get("n_samples")), "n_a": number(r.get("n_a")), "n_b": number(r.get("n_b")),
                "status": str(r.get("status", ""))})
        supports = [[v for v in cells if v["state"] == "fdr" and v["sign"] == sign] for sign in (-1, 1)]
        def strength(group):
            recoveries = [v["recovery"] for v in group if v["recovery"] is not None]
            return len(group), float(np.median(recoveries)) if recoveries else -1
        dominant = max(supports, key=strength)
        count, recovery = strength(dominant)
        qs, ps = [v["q"] for v in cells if v["q"] is not None], [v["p"] for v in cells if v["p"] is not None]
        label = str(part.label.iloc[0])
        rows.append({"comparison": str(comparison), "family": family, "feature": feature, "label": label,
            "comparison_label": name, "name": screening.FAMILIES[family] + " | " + label + " | " + name,
            "supported_zones": count, "fdr_zones": sum(v["state"] == "fdr" for v in cells),
            "tested_zones": sum(v["p"] is not None for v in cells), "total_zones": len(zones),
            "median_recovery": recovery if recovery >= 0 else None,
            "min_q": min(qs) if qs else None, "min_p": min(ps) if ps else None,
            "mixed_fdr_directions": all(bool(s) for s in supports), "cells": cells})
    rows.sort(key=lambda r: (-r["supported_zones"], -(r["median_recovery"] if r["median_recovery"] is not None else -1),
                             r["min_q"] if r["min_q"] is not None else 2,
                             r["min_p"] if r["min_p"] is not None else 2, r["name"], r["comparison"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return {"rows": rows, "zones": [{"zone": j, "label": zone_label(c, j).split(" (")[0]} for j in zones],
            "zone_mode": c["zone_mode"], "zone_source": c["zone_source"], "side": c["side"], "alpha": c["alpha"],
            "scope": SCOPES[c.get("correction_scope", "global")], "correction": c["correction"],
            "ranking": RANKING, "states": STATES,
            "resampling": c.get("resampling", "off"), "fraction": c.get("sample_fraction"), "repeats": c.get("resamples"),
            "comparisons": [{"id": k, "label": comparison_name(p["endpoint"], p.get("stratum"))} for k, p in lookup.items()]}


def load(root, run):
    saved = screening.result(root, run)
    c = saved.get("scan_settings", saved["settings"])
    folder = Path(saved.get("scan_folder", saved["folder"]))
    if saved.get("comparisons"):
        path = folder / "tables/all_comparisons.csv"
        if not path.exists() or saved.get("partial"):
            raise ValueError("The stability overview will be ready after all planned comparisons and resampling finish.")
    else:
        if saved.get("partial"):
            raise ValueError("The stability overview will be ready after the scan and resampling finish.")
        path = folder / "tables/screening_results_all.csv"
    # Pagination/search/export all read the same completed overview. Bound this
    # cache to two runs and invalidate it when saved results change; never scan
    # cell measurements or rebuild thousands of rows on every filter change.
    def stamp(p):
        stat = p.stat() if p.exists() else None
        return (str(p), stat.st_size, stat.st_mtime_ns) if stat else (str(p), None, None)
    key = (str(Path(root).resolve()), run, str(saved['revision']), stamp(folder/'result.json'), stamp(path))
    with _overview_lock:
        if key in _overviews:
            _overviews.move_to_end(key)
            return _overviews[key]
        tests = pd.read_csv(path) if saved.get('comparisons') else pd.DataFrame(saved.get('tests', saved['hits']))
        found = build(tests, c, saved.get("comparisons"), saved.get('palette'))
        found.update(run=run, revision=saved["revision"])
        _overviews[key] = found
        while len(_overviews) > 2:
            _overviews.popitem(last=False)
        return found


def select(found, comparison="", family="", search="", only_fdr=False, offset=0, limit=20):
    if family and family not in screening.FAMILIES:
        raise ValueError("Choose a feature type from this scan.")
    if offset < 0 or limit not in (10, 20, 40):
        raise ValueError("Choose 10, 20, or 40 features per page.")
    matches = [r for r in found["rows"] if (not comparison or r["comparison"] == comparison)
        and (not family or r["family"] == family) and (not search or search.casefold() in r["name"].casefold())
        and (not only_fdr or r["fdr_zones"])]
    offset = min(offset, max(0, (len(matches)-1)//limit*limit))
    return dict(found, rows=matches[offset:offset+limit], total=len(matches), offset=offset, limit=limit)


def csv_bytes(found):
    rows = []
    for row in found["rows"]:
        shared = {k: v for k, v in row.items() if k != "cells"}
        for cell in row["cells"]:
            rows.append(dict(shared, **cell, distance=found["zones"][cell["zone"]]["label"]))
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def figure_bytes(found, extension):
    if extension not in ("png", "svg", "pdf"):
        raise ValueError("Choose PNG, SVG, or PDF.")
    with _plot_lock:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle, Circle
        rows, zones = found["rows"], found["zones"]
        body_height = max(1.5, .55*len(rows))
        fig = plt.figure(figsize=(max(11, 6 + .85*len(zones)), body_height+3.5), layout="constrained")
        layout = fig.add_gridspec(3, 1, height_ratios=[.85, body_height, 1.0])
        header, ax, footer = [fig.add_subplot(layout[i, 0]) for i in range(3)]
        header.axis("off")
        footer.axis("off")
        labels = []
        for iy, row in enumerate(rows):
            labels.append(textwrap.fill(f"{row['rank']}. {row['name']}", 65) +
                f"\nFDR zones (same direction): {row['supported_zones']}/{row['total_zones']}" +
                (" · mixed directions" if row["mixed_fdr_directions"] else ""))
            values = {c["zone"]: c for c in row["cells"]}
            for ix, zone in enumerate(zones):
                cell = values.get(zone["zone"], {"state":"missing", "short":"—", "recovery":None})
                _, colour, ink = STATES[cell["state"]]
                ax.add_patch(Rectangle((ix-.5, iy-.5), 1, 1, facecolor=colour, edgecolor="white", linewidth=.7))
                if cell.get('group_colour'):
                    ax.add_patch(Circle((ix-.38, iy-.34), .055, facecolor=cell['group_colour'], edgecolor=ink, linewidth=.5))
                short = str(cell["short"])
                if len(short) > 18:
                    short = short[:17] + "…"
                short = textwrap.fill(short, 12)
                recovery = f"{cell['recovery']*100:.0f}%" if cell["recovery"] is not None else "—"
                ax.text(ix, iy, short + "\n" + recovery, color=ink, ha="center", va="center", fontsize=8)
        ax.set(xlim=(-.5, len(zones)-.5), ylim=(max(.5, len(rows)-.5), -.5))
        ax.set_xticks(range(len(zones)), [z["label"] for z in zones], rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(rows)), labels, fontsize=8)
        ax.set_xlabel("Whole region" if found["zone_source"] == "whole" else ("Rings" if found["zone_mode"] == "rings" else "Cumulative zones") + " · " + found["side"])
        header.text(.5, 1, "Feature stability across distances", ha="center", va="top", fontweight="bold", fontsize=13)
        mode = {'bootstrap': 'Bootstrap recovery', 'subsample': 'Sample-subset recovery', 'off': 'Not resampled'}.get(found['resampling'], 'Recovery')
        header.text(.5, .57, mode, ha='center', va='center', fontsize=9)
        header.legend(handles=[Patch(facecolor=colour, label=label) for label, colour, _ in STATES.values()],
                  loc="lower center", ncol=2 if len(zones)<8 else 4, fontsize=8, frameon=False)
        if not rows:
            ax.text(.5,.5,"No features match this filter",transform=ax.transAxes,ha="center")
        footer.text(0, .9, "Cell text: group with the higher estimated value (or higher change); % = exact recovery. — = unavailable.\n"
                 "Survival arrows show association with event hazard; Positive/Negative describe numeric associations.\n"
                 f"q < {found['alpha']:g} · {found['scope']} · {found['correction']}\n"
                 "Rank: same-direction FDR zone count, median exact recovery, smallest q/p. Overlapping zones are not independent validation.", fontsize=8, va="top")
        buffer = BytesIO()
        try:
            fig.savefig(buffer, format=extension, dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)
        return buffer.getvalue()
