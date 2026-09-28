"""A registered (merged) image keeps its CODEX source's acquisition metadata.

A small QPTIFF-like source (one PerkinElmer description per channel page) and an H&E OME-TIFF are
merged as Tab 1 merges them. The merged OME-TIFF stays one pyramidal, tiled image, its colours are
opaque, exposure and wavelengths sit in OME's own places, and every source field is kept as a
key/value annotation (per channel, and for the slide).
"""

import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from spacemap.image_metadata import source_metadata
from spacemap.registration import merge_multi_round_ome_tiff
from spacemap.utils import rgb_to_color_int

OME = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
MARKERS = [("DAPI", "0,0,255", 2500, "DAPI"), ("CD3e", "255,0,0", 150000, "ATTO550"), ("CD8", "0,255,0", 200000, "CY5")]


def _qpi_description(name, color, exposure, filter_name):
    return f"""<?xml version="1.0" encoding="utf-16"?>
<PerkinElmer-QPI-ImageDescription>
  <DescriptionVersion>4</DescriptionVersion>
  <ImageType>FullResolution</ImageType>
  <SlideID>slide_42</SlideID>
  <StudyName>STUDY</StudyName>
  <ExposureTime>{exposure}</ExposureTime>
  <Name>{filter_name}</Name>
  <Color>{color}</Color>
  <Responsivity><Filter><Name>{filter_name}</Name><Response>421.9</Response></Filter></Responsivity>
  <Objective>20xLWD</Objective>
  <Biomarker>{name}</Biomarker>
  <ExcitationFilter><Name>{filter_name} set</Name><Bands><Band><Cuton>377.0</Cuton></Band><Band><Cuton>533.0</Cuton></Band></Bands></ExcitationFilter>
  <CameraSettings><Gain>1</Gain><Binning>2</Binning></CameraSettings>
</PerkinElmer-QPI-ImageDescription>"""


def _sources(tmp_path):
    codex = tmp_path / "S1_codex.qptiff"
    rng = np.random.default_rng(7)
    with tifffile.TiffWriter(codex) as writer:
        for name, color, exposure, filter_name in MARKERS:
            writer.write(rng.integers(0, 4000, (300, 400), dtype=np.uint16), photometric="minisblack",
                         description=_qpi_description(name, color, exposure, filter_name), metadata=None,
                         software="PerkinElmer-QPI")
    he = tmp_path / "S1_he_registered.ome.tiff"
    tifffile.imwrite(he, rng.integers(0, 255, (3, 300, 400), dtype=np.uint16), photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ["HE_Red", "HE_Green", "HE_Blue"]}})
    return codex, he


def test_the_source_is_read_whole(tmp_path):
    codex, _ = _sources(tmp_path)
    found = source_metadata(codex)
    assert found["image"]["SlideID"] == "slide_42" and found["image"]["Objective"] == "20xLWD"
    assert [record["Biomarker"] for record in found["channels"]] == ["DAPI", "CD3e", "CD8"]
    assert found["channels"][0]["ExcitationFilter/Name"] == "DAPI set"
    assert found["image"]["ExcitationFilter/Bands/Band[2]/Cuton"] == "533.0"  # the same in every channel: kept once, for the slide
    assert found["exposures"][1] == ("150000", "µs")


def test_the_merged_image_keeps_the_codex_metadata(tmp_path):
    codex, he = _sources(tmp_path)
    output = tmp_path / "S1_merged.ome.tiff"
    merge_multi_round_ome_tiff([str(codex), str(he)], str(output), 0.5, None, provenance={"CODEX": "S1_codex.qptiff", "H&E": "S1_he.tif"})
    with tifffile.TiffFile(output) as tif:
        series = tif.series[0]
        assert len(tif.series) == 1 and series.axes == "CYX" and series.shape == (6, 300, 400)
        assert len(series.levels) == 4 and tif.pages[0].is_tiled and not tif.is_shaped
        root = ET.fromstring(tif.ome_metadata)
    channels = root.findall(f".//{OME}Channel")
    assert [channel.get("Name") for channel in channels] == ["DAPI", "CD3e", "CD8", "HE_Red", "HE_Green", "HE_Blue"]
    assert channels[0].get("Color") == str(rgb_to_color_int((0, 0, 255))) and int(channels[0].get("Color")) & 0xFF == 0xFF
    assert channels[0].get("ExcitationWavelengthUnit") == "nm"
    planes = {int(plane.get("TheC")): plane for plane in root.iter(f"{OME}Plane")}
    assert planes[0].get("ExposureTime") == "2500" and planes[0].get("ExposureTimeUnit") == "µs"
    notes = {
        annotation.get("ID"): {m.get("K"): m.text for m in annotation.iter(f"{OME}M")}
        for annotation in root.iter(f"{OME}MapAnnotation")
    }
    dapi = notes[channels[0].find(f"{OME}AnnotationRef").get("ID")]
    assert dapi["Biomarker"] == "DAPI" and dapi["Source file"] == "S1_codex.qptiff" and dapi["ExcitationFilter/Name"] == "DAPI set"
    slide = next(record for record in notes.values() if record.get("SlideID"))
    assert slide["StudyName"] == "STUDY" and slide["CameraSettings/Binning"] == "2"
    spacemap = next(record for record in notes.values() if "Inputs" in record)
    assert spacemap["CODEX"] == "S1_codex.qptiff" and spacemap["H&E"] == "S1_he.tif"
    # OME's order inside a Channel: AnnotationRef before LightPath; the image's AnnotationRefs after Pixels.
    assert [child.tag.rsplit("}")[-1] for child in channels[0]] == ["AnnotationRef", "LightPath"]
    image = root.find(f"{OME}Image")
    tags = [child.tag.rsplit("}")[-1] for child in image]
    assert tags.index("Pixels") < tags.index("AnnotationRef")
    # And the merged image is itself a full source for the next steps.
    again = source_metadata(output)
    assert again["channels"][0]["Biomarker"] == "DAPI" and again["exposures"][0] == ("2500", "µs")


def test_the_matched_points_figure_is_saved_and_closed(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacemap.registration import _save_feature_matches, feature_matches_path

    output = tmp_path / "S1_he_registered.ome.tiff"
    plt.subplots()[1].plot([0, 1], [0, 1])  # stands in for palom's drawing
    _save_feature_matches(output)
    assert feature_matches_path(output) == tmp_path / "S1_he_registered_feature_matches.png"
    assert feature_matches_path(output).stat().st_size > 1000 and not plt.get_fignums()


def _notes(root):
    return {
        annotation.get("ID"): {m.get("K"): m.text for m in annotation.iter(f"{OME}M")}
        for annotation in root.iter(f"{OME}MapAnnotation")
    }


def _channel_notes(root, channel):
    found = {}
    everything = _notes(root)
    for reference in channel.findall(f"{OME}AnnotationRef"):
        found.update(everything.get(reference.get("ID"), {}))
    return found


def test_crops_and_background_removed_images_keep_the_parent_metadata(tmp_path):
    import json

    from spacemap.background_detection import subtract_background_image
    from spacemap.registration import process_single_image_rois

    codex, he = _sources(tmp_path)
    merged = tmp_path / "S1_merged.ome.tiff"
    merge_multi_round_ome_tiff([str(codex), str(he)], str(merged), 0.5, None, provenance={"CODEX": codex.name})
    boundary = tmp_path / "S1.geojson"
    boundary.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "roi_1", "classification": {"name": "roi_1"}},
        "geometry": {"type": "Polygon", "coordinates": [[[50, 40], [250, 40], [250, 200], [50, 200], [50, 40]]]}}]}))
    rois = tmp_path / "rois"
    rois.mkdir()
    process_single_image_rois(str(merged), str(boundary), str(rois), {"codex": 0.5},
                              channel_config={"keep": ["DAPI", "CD8", "HE_Red"], "rename": {"CD8": "CD8a"}})
    crop = next(rois.glob("*.ome.tiff"))
    with tifffile.TiffFile(crop) as tif:
        assert len(tif.series) == 1 and not tif.is_shaped and tif.series[0].shape[0] == 3 and len(tif.series[0].levels) == 4
        root = ET.fromstring(tif.ome_metadata)
    channels = root.findall(f".//{OME}Channel")
    assert [channel.get("Name") for channel in channels] == ["DAPI", "CD8a", "HE_Red"]
    assert all("Exposure" not in channel.attrib for channel in channels)  # OME's own field instead
    planes = {int(plane.get("TheC")): plane for plane in root.iter(f"{OME}Plane")}
    assert planes[0].get("ExposureTime") == "2500" and planes[1].get("ExposureTime") == "200000"  # CD8 is parent channel 2
    cd8 = _channel_notes(root, channels[1])
    assert cd8["Biomarker"] == "CD8" and cd8["Name in the parent image"] == "CD8" and cd8["Source file"] == "S1_codex.qptiff"
    everything = list(_notes(root).values())
    assert any(record.get("SlideID") == "slide_42" for record in everything)  # the slide's acquisition record
    assert any(record.get("CODEX") == "S1_codex.qptiff" for record in everything)  # how the parent was made
    assert any(record.get("Cropped from") == "S1_merged.ome.tiff" and record.get("ROI") for record in everything)
    # Removing background from the crop keeps all of it too.
    cleaned = tmp_path / "S1_roi_1_bg.ome.tiff"
    subtract_background_image(crop, {"DAPI": 10.0, "CD8a": 5.0}, cleaned)
    with tifffile.TiffFile(cleaned) as tif:
        assert len(tif.series) == 1 and not tif.is_shaped
        root = ET.fromstring(tif.ome_metadata)
    channels = root.findall(f".//{OME}Channel")
    assert _channel_notes(root, channels[1])["Biomarker"] == "CD8"
    assert {int(p.get("TheC")): p.get("ExposureTime") for p in root.iter(f"{OME}Plane")}[1] == "200000"
    records = list(_notes(root).values())
    assert any(record.get("Background removed from") == crop.name for record in records) and any(record.get("SlideID") for record in records)
    # SPACEMAP reads the exposure back from OME's field.
    from spacemap.background_subtraction_utils import get_channel_info

    assert get_channel_info(str(cleaned))[2][0]["exposure"] == "2"
