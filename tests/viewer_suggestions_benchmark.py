"""Time suggestions and least-sure scans on disposable 35-marker tables."""

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd


def main():
    with tempfile.TemporaryDirectory(prefix="spacemap-suggestion-speed-") as folder:
        root = Path(folder)
        os.environ["SPACEMAP_HOME"] = str(root / "state")
        os.environ["SPACEMAP_FAST_COPY"] = "on"
        from spacemap import dataio
        from spacemap.viewer import cells, suggestions as s
        from test_viewer_cell_profiles import marker_image

        image = root / "image.ome.tiff"
        marker_image(image, np.ones((35, 16, 16), np.uint16))
        for count in (100_000, 1_100_000):
            path = root / f"cells-{count}.csv"
            rng = np.random.default_rng(2)
            markers = [f"M{i}" for i in range(35)]
            frame = pd.DataFrame(rng.integers(0, 65535, (count, 35), dtype=np.uint16), columns=markers)
            frame["batch_id"] = "roi"
            frame["label"] = np.arange(1, count + 1)
            frame["phenotype_ref1"] = None
            frame.loc[:39, "phenotype_ref1"] = ["T", "B"] * 20
            frame.to_csv(path, index=False)
            del frame
            dataio.ensure_fast_copy(path, force=True)
            table = str(path)
            settings = s.settings_for(
                {"roi": "roi", "image": str(image), "pyramid": str(image), "segmentation": str(image)},
                table,
                {"column": "phenotype_ref1", "destination": "phenotype_ref1"},
                markers,
            )
            cells.roi_index(table, "roi")
            times = []
            for label in range(100, 111):
                start = time.perf_counter()
                assert s.run(settings, label=label)["status"] == "ready"
                times.append(1000 * (time.perf_counter() - start))
            start = time.perf_counter()
            result = s.request(settings, scan=True)
            while result["status"] == "preparing":
                time.sleep(0.02)
                result = s.request(settings, scan=True)
            assert result["status"] == "ready", result
            scan_ms = 1000 * (time.perf_counter() - start)
            start = time.perf_counter()
            assert s.request(settings, scan=True, exclude=[result["label"]])["label"] != result["label"]
            print(
                json.dumps(
                    {
                        "cells": count,
                        "markers": 35,
                        "examples": 40,
                        "first_suggestion_ms": times[0],
                        "later_suggestion_ms": float(np.median(times[1:])),
                        "first_scan_ms": scan_ms,
                        "cached_next_ms": 1000 * (time.perf_counter() - start),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
