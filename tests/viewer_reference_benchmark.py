"""Disposable large-table timing check; never edits a real project."""

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd


def main():
    with tempfile.TemporaryDirectory(prefix="spacemap-reference-speed-") as folder:
        root = Path(folder)
        os.environ["SPACEMAP_HOME"] = str(root / "state")
        os.environ["SPACEMAP_FAST_COPY"] = "on"
        from spacemap import dataio
        from spacemap.viewer import cells, reference

        for count in (100_000, 1_100_000):
            path = root / f"cells-{count}.csv"
            pd.DataFrame(
                {
                    "batch_id": "roi",
                    "label": np.arange(1, count + 1),
                    "phenotype_ref1": np.where(np.arange(count) % 2, "T", "B"),
                }
            ).to_csv(path, index=False)
            dataio.ensure_fast_copy(path, force=True)
            table = str(path)
            cells.roi_index(table, "roi")
            timings, review_times = [], []
            for batch in range(11):
                edits = {str(batch * 10 + i + 1): "NK" for i in range(10)}
                started = time.perf_counter()
                result = reference.save(table, "roi", "phenotype_ref1", edits, f"timing-{count}-{batch}")
                timings.append(1000 * (time.perf_counter() - started))
                assert result["assigned"] == 10
                started = time.perf_counter()
                listed = reference.review(table, "roi", "phenotype_ref1")
                review_times.append(1000 * (time.perf_counter() - started))
                assert listed["items"][0]["value"] == "NK"
            print(
                json.dumps(
                    {
                        "cells": count,
                        "cells_per_save": 10,
                        "first_save_ms": timings[0],
                        "later_save_ms": [min(timings[1:]), float(np.median(timings[1:])), max(timings[1:])],
                        "review_ms": [min(review_times), float(np.median(review_times)), max(review_times)],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
