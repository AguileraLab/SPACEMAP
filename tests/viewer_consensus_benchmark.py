"""Prepare a private consensus model and time read-only predictions on an existing table."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table")
    parser.add_argument("image")
    parser.add_argument("roi")
    parser.add_argument("--app-pid", type=int)
    args = parser.parse_args()
    if args.app_pid:
        allowed = {"SPACEMAP_HOME", "SPACEMAP_MODELS_DIR", "XDG_DATA_HOME"}
        for item in Path(f"/proc/{args.app_pid}/environ").read_bytes().split(b"\0"):
            key, _, value = item.partition(b"=")
            if key.decode() in allowed:
                os.environ[key.decode()] = value.decode()
    from spacemap import dataio, labels
    from spacemap.viewer import cells, consensus_model as m, reference
    from spacemap_gui.viewer.api import _markers
    import numpy as np

    table = str(Path(args.table).resolve())
    watched = [Path(table), dataio.fast_copy_path(table), labels.sidecar_path(table), reference.path_for(table)]

    def stamps():
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in watched]

    before = stamps()
    layer = {"roi": args.roi, "image": args.image}
    settings = m.settings_for(layer, table, _markers(layer, table, include_nuclear=True), "phenotype_ref1")
    target = m.path_for(settings)
    existed = target.exists()
    start = time.perf_counter()
    last = None
    while True:
        result = m.request(settings, prepare=True)
        if result.get("message") != last:
            last = result.get("message")
            print(json.dumps({"progress": last, "seconds": round(time.perf_counter() - start, 2)}), flush=True)
        if result["status"] != "preparing":
            break
        time.sleep(0.5)
    assert result["status"] == "ready", result
    prepare_s = time.perf_counter() - start
    rows, ids = cells.roi_index(table, args.roi)
    timings = []
    selected = ids[np.linspace(0, len(ids) - 1, 16, dtype=int)]
    for cell in selected:
        start = time.perf_counter()
        prediction = m.request(settings, label=int(cell))
        assert prediction["status"] == "ready", prediction
        timings.append(1000 * (time.perf_counter() - start))
    start = time.perf_counter()
    m.request(settings, label=int(selected[-1]))
    repeat_ms = 1000 * (time.perf_counter() - start)
    p = result["provenance"]
    assert before == stamps(), "Research table or saved labels changed"
    print(
        json.dumps(
            {
                "model_existed": existed,
                "prepare_s": prepare_s,
                "markers": len(p["markers"]),
                "examples": p["examples"],
                "available_examples": p["available_examples"],
                "phenotypes": len(p["training_counts"]),
                "oob_consensus_agreement": p["oob_balanced_accuracy"],
                "model_MB": target.stat().st_size / 1e6,
                "prediction_ms": timings,
                "median_prediction_ms": float(np.median(timings)),
                "repeat_ms": repeat_ms,
                "project_files_unchanged": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
