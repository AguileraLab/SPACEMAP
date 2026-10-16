"""Optional 20-million-cell exact-rule timing on disposable data."""

import json
import os
from pathlib import Path
import tempfile
import time


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as csv
    from spacemap import dataio
    from spacemap_gui.studio import selection

    count = 20_000_000
    roi_size = 5_000_000
    with tempfile.TemporaryDirectory(prefix="spacemap-rule-speed-") as folder:
        root = Path(folder)
        os.environ["SPACEMAP_HOME"] = str(root / "state")
        os.environ["SPACEMAP_FAST_COPY"] = "on"
        table = root / "cells.csv"
        schema = pa.schema(
            [("batch_id", pa.string()), ("label", pa.uint32()), ("M0", pa.float64()), ("M1", pa.float64())]
        )
        with csv.CSVWriter(table, schema) as writer:
            for start in range(0, count, 250_000):
                rows = np.arange(start, start + 250_000, dtype=np.int64)
                writer.write(
                    pa.Table.from_arrays(
                        [
                            pa.array(["roi" + str(start // roi_size)] * len(rows)),
                            pa.array((rows % roi_size + 1).astype(np.uint32)),
                            pa.array((rows % 1000).astype(float)),
                            pa.array(((rows * 7) % 1000).astype(float)),
                        ],
                        schema=schema,
                    )
                )
        print(json.dumps({"prepared_cells": count, "csv_bytes": table.stat().st_size}), flush=True)
        dataio.ensure_fast_copy(table, force=True)
        settings = {
            "table": str(table),
            "roi": "roi0",
            "roi_column": "batch_id",
            "label_column": "label",
            "markers": ["M0", "M1"],
        }
        rule = {
            "scope": "project",
            "join": "all",
            "conditions": [
                {"marker": "M0", "transform": "Raw", "op": ">", "value": 500},
                {"marker": "M1", "transform": "Raw", "op": "<", "value": 250},
            ],
        }
        for phase in ("first", "repeat"):
            start = time.perf_counter()
            result = selection.evaluate("speed-check", settings, rule)
            elapsed = time.perf_counter() - start
            selected = selection.get(result["id"], channel="speed-check")
            rows = np.arange(roi_size)
            expected = rows[(rows % 1000 > 500) & ((rows * 7) % 1000 < 250)] + 1
            for group in selected["groups"]:
                np.testing.assert_array_equal(group["ids"], expected)
            assert result["count"] == 4 * len(expected)
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "seconds": round(elapsed, 3),
                        "selected": result["count"],
                        "cells": result["total"],
                        "exact_ids": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
