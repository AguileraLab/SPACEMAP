"""Blank marker names and black colours saved by an interrupted save are refilled from the image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from spacemap_gui.tabs.tab3_roi_extraction import _unset_colour


def test_black_is_only_unset_when_the_image_has_a_colour():
    assert _unset_colour(None, "#ff0000") and _unset_colour("", "#ff0000")
    assert _unset_colour("#000000", "#ff0000")
    assert not _unset_colour("#000000", "#000000")  # the image itself says black
    assert not _unset_colour("#00ff00", "#ff0000")  # a real choice is kept


def _image(path: Path) -> None:
    names = ["DAPI", "CD8"]
    ome = ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
           '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="64" SizeY="64" SizeC="2" SizeZ="1" SizeT="1">'
           + "".join(f'<Channel ID="Channel:0:{i}" Name="{n}" SamplesPerPixel="1"/>' for i, n in enumerate(names))
           + "".join(f'<TiffData FirstC="{i}" IFD="{i}" PlaneCount="1"/>' for i in range(2)) + "</Pixels></Image></OME>")
    with tifffile.TiffWriter(path) as writer:
        for c in range(2):
            writer.write(np.full((64, 64), 100 * (c + 1), np.uint16), description=ome if c == 0 else None, metadata=None)


def test_a_blank_saved_marker_is_refilled_on_the_page(tmp_path):
    from streamlit.testing.v1 import AppTest

    (tmp_path / "images").mkdir()
    (tmp_path / "data" / "boundaries").mkdir(parents=True)
    _image(tmp_path / "images" / "P1_merged.ome.tiff")

    def page(root):
        from pathlib import Path

        import streamlit as st

        from spacemap_gui.tabs import tab3_roi_extraction

        st.session_state.setdefault("tab3_input_dir", str(Path(root) / "images"))
        st.session_state.setdefault("tab3_geojson_dir", str(Path(root) / "data" / "boundaries"))
        st.session_state.setdefault("roi_name_1", "")  # left by a bad save
        st.session_state.setdefault("roi_color_1", "#000000")
        tab3_roi_extraction.render_single_image(Path(root))

    app = AppTest.from_function(page, args=(str(tmp_path),), default_timeout=180)
    app.run()
    next(box for box in app.selectbox if box.label == "Select Image").select("P1_merged.ome.tiff").run()
    assert not app.exception, app.exception
    assert app.session_state["roi_name_1"] == "CD8"
    assert app.session_state["roi_color_1"].lower() != "#000000"
