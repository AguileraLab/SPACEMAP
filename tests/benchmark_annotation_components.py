"""Read-only component benchmark: python tests/benchmark_annotation_components.py file.geojson."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from spacemap_gui.studio import annotation_components as components
from spacemap.viewer.render import TILE


def benchmark(path):
    path = Path(path)
    before = path.stat()
    start = perf_counter()
    overview = components.overview(path)
    overview_time = perf_counter() - start
    start = perf_counter()
    value = components.index(path)
    result = {'components': len(value['rows']), 'cold_index_seconds': perf_counter() - start, 'class_overview_seconds': overview_time}
    if not value['rows']:
        return result
    start = perf_counter()
    components.page(path)
    result['warm_page_seconds'] = perf_counter() - start
    point = value['geometries'][0].representative_point()
    start = perf_counter()
    for _ in range(100):
        components.hit(path, point.x, point.y)
    result['warm_hit_seconds'] = (perf_counter() - start) / 100
    start = perf_counter()
    chosen = components.selection(path, [r['id'] for r in value['rows']])
    result['selection_index_seconds'] = perf_counter() - start
    bounds = [r['bounds'] for r in value['rows']]
    image = SimpleNamespace(width=math.ceil(max(b[2] for b in bounds)) + 1,
                            height=math.ceil(max(b[3] for b in bounds)) + 1)
    z = max(0, math.ceil(math.log2(max(image.width, image.height) / TILE)))
    for key in ('overview_tile_seconds', 'warm_overview_tile_seconds'):
        start = perf_counter()
        components.selection_tile(path, chosen['selection'], image, z, 0, 0)
        result[key] = perf_counter() - start
    x, y = value['geometries'][0].exterior.coords[0]
    start = perf_counter()
    components.selection_tile(path, chosen['selection'], image, 0, int(x // TILE), int(y // TILE))
    result['detail_tile_seconds'] = perf_counter() - start
    result['source_unchanged'] = (before.st_mtime_ns, before.st_size) == (path.stat().st_mtime_ns, path.stat().st_size)
    assert result['source_unchanged']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geojson')
    args = parser.parse_args()
    print(json.dumps(benchmark(args.geojson), indent=2))
