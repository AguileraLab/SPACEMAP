"""Classification: Smallest island and Smallest hole clean the class map before it becomes polygons; runs are kept."""

import numpy as np

from spacemap_gui.studio import classification


def test_small_islands_join_their_surroundings_and_small_holes_fill():
    merged = np.zeros((40, 40), np.int32)  # class 0 everywhere
    merged[5:7, 5:7] = 1          # a 4-pixel island of class 1 inside class 0
    merged[20:30, 20:30] = 1      # a large piece of class 1
    merged[24:26, 24:26] = 0      # a 4-pixel hole in it (class 0)
    merged[0:3, 30:40] = -1       # outside the area (a tissue hole at the edge): never filled
    merged[12:14, 30:32] = -1     # a small tissue hole inside class 0: kept as well
    cleaned = classification.clean_map(merged, [0, 1], island_px=10, hole_px=0)
    assert (cleaned[5:7, 5:7] == 0).all()                 # the island joined class 0
    assert (cleaned[24:26, 24:26] == 1).all()             # the 0 inside class 1 was itself an island: joined class 1
    assert (cleaned[0:3, 30:40] == -1).all() and (cleaned[12:14, 30:32] == -1).all()
    holes = merged.copy()
    holes[24:26, 24:26] = 2                                # an excluded class (2) inside class 1
    filled = classification.clean_map(holes, [0, 1], island_px=0, hole_px=10)
    assert (filled[24:26, 24:26] == 1).all() and (filled[12:14, 30:32] == -1).all()
    kept = classification.clean_map(holes, [0, 1], island_px=0, hole_px=3)
    assert (kept[24:26, 24:26] == 2).all()                 # larger than Smallest hole: kept


def test_cleanup_is_checked_and_not_part_of_a_run():
    assert classification.cleanup({"hole_um2": 50}) == {"hole_um2": 50.0, "island_um2": 0.0}
    for bad in ({"hole_um2": -1}, {"island_um2": "big"}):
        try:
            classification.cleanup(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_a_run_is_saved_and_loaded_without_clustering(tmp_path):
    image = tmp_path / "roi.ome.tiff"
    found = {"run_id": "a" * 24, "cluster_map": np.array([[0, 1], [-1, 1]], np.int32), "counts": np.array([1, 2]),
             "z_sums": np.ones((2, 3)), "raw_sums": np.ones((2, 3)), "cluster_colors": ["#ff0000", "#00ff00"],
             "execution": {"device": "cpu"}, "provenance": {"scope": "tissue.geojson"}}
    prepared = {"factor": 4, "stack": {"marker_names": ["A", "B", "C"]}}
    classification.persist(tmp_path, image, prepared, found)
    loaded_prepared, loaded = classification.load(tmp_path, image, "a" * 24)
    assert loaded_prepared["factor"] == 4 and (loaded["cluster_map"] == found["cluster_map"]).all()
    assert classification.load(tmp_path, image, "b" * 24) is None and classification.load(tmp_path, image, "../x") is None
    shown = classification.render(loaded_prepared, loaded, None)
    assert shown["pixels"] == 3 and len(shown["classes"]) == 2
    assert classification.index_png(loaded["cluster_map"]).startswith("data:image/png;base64,")
