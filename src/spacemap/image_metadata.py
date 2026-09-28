"""An image's acquisition metadata, carried into the OME-TIFFs SPACEMAP writes.

``source_metadata(path)`` reads everything a source image records about itself: a QPTIFF's
per-channel PerkinElmer descriptions (marker, exposure, filters, camera, slide, study, scan
profile, ...) or an OME-TIFF's OME-XML (channel attributes, exposure planes, instrument and
key/value annotations). Fields every channel shares become the image's record; the others stay
with their channel.

``enrich_ome(path, channels, image_notes)`` adds these records to an OME-TIFF after it was
written, without touching its pixels: exposure times in OME's ``Plane/@ExposureTime``, excitation
and emission wavelengths on each ``Channel``, and every field as OME key/value annotations
(``MapAnnotation``), which QuPath, OMERO and Bio-Formats show.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import tifffile

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
MAP_NAMESPACE = "openmicroscopy.org/omero/client/mapAnnotation"
_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _q(tag):
    return f"{{{OME_NS}}}{tag}"


def _parse(text):
    return ET.fromstring(_DECLARATION.sub("", text or "", count=1))


def _flatten(element, prefix="", out=None):
    """Every text and attribute under *element* as "Path/To/Field" -> value (repeated tags numbered)."""
    out = {} if out is None else out
    for name, value in element.attrib.items():
        if value not in (None, ""):
            out[f"{prefix}@{_local(name)}" if prefix else _local(name)] = str(value)
    children = list(element)
    if not children:
        text = (element.text or "").strip()
        if text and prefix:
            out[prefix] = text
        return out
    counts, seen = Counter(_local(child.tag) for child in children), Counter()
    for child in children:
        name = _local(child.tag)
        seen[name] += 1
        key = name if counts[name] == 1 else f"{name}[{seen[name]}]"
        _flatten(child, f"{prefix}/{key}" if prefix else key, out)
    return out


def _split_shared(records):
    """(fields identical in every record, each record without them)."""
    if not records:
        return {}, []
    shared = {key: value for key, value in records[0].items() if all(record.get(key) == value for record in records[1:])}
    if len(records) == 1:
        return {}, records
    return shared, [{key: value for key, value in record.items() if key not in shared} for record in records]


def _qptiff(tif):
    records, exposures = [], []
    for page in tif.series[0].pages:
        try:
            record = _flatten(_parse(page.description))
        except ET.ParseError:
            record = {}
        records.append(record)
        exposure = record.get("ExposureTime")
        exposures.append((exposure, "µs") if exposure else None)  # PerkinElmer records microseconds
    shared, channels = _split_shared(records)
    return shared, channels, exposures


def _ome(tif):
    root = _parse(tif.ome_metadata)
    image = root.find(_q("Image"))
    pixels = image.find(_q("Pixels")) if image is not None else None
    if pixels is None:
        return {}, [], []
    annotations = {
        item.get("ID"): {m.get("K"): (m.text or "") for m in item.iter(_q("M"))}
        for item in root.iter(_q("MapAnnotation"))
    }

    def notes(element):
        found = {}
        for ref in element.findall(_q("AnnotationRef")):
            found.update(annotations.get(ref.get("ID"), {}))
        return found

    whole = {f"Image@{key}": value for key, value in image.attrib.items() if key != "ID"}
    for tag in ("AcquisitionDate", "Description"):
        item = image.find(_q(tag))
        if item is not None and (item.text or "").strip():
            whole[tag] = item.text.strip()
    for instrument in root.findall(_q("Instrument")):
        _flatten(instrument, "Instrument", whole)
    whole.update(notes(image))
    planes = {}
    for plane in pixels.findall(_q("Plane")):
        planes.setdefault(int(plane.get("TheC", 0)), plane)
    channels, exposures = [], []
    for index, channel in enumerate(pixels.findall(_q("Channel"))):
        record = {key: value for key, value in channel.attrib.items() if key not in {"ID", "SamplesPerPixel"}}
        for child in channel:
            if _local(child.tag) != "AnnotationRef":
                _flatten(child, _local(child.tag), record)
        record.update(notes(channel))
        plane = planes.get(index)
        if plane is not None:
            record.update({f"Plane@{key}": value for key, value in plane.attrib.items() if key not in {"TheC", "TheZ", "TheT"}})
        exposure = plane.get("ExposureTime") if plane is not None else record.get("Exposure")
        exposures.append((exposure, plane.get("ExposureTimeUnit", "s") if plane is not None else "ms") if exposure else None)
        channels.append(record)
    return whole, channels, exposures


def source_metadata(path):
    """What *path* records about its acquisition: {"image": {...}, "channels": [{...}, ...], "exposures": [(value, unit) | None]}.

    Unknown formats give their ImageDescription (when it is not tifffile's own shape note) as the image record.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            first = (tif.pages[0].description or "") if tif.pages else ""
            if tif.is_qpi or "<PerkinElmer-QPI-ImageDescription" in first[:400]:
                whole, channels, exposures = _qptiff(tif)
            elif tif.is_ome:
                whole, channels, exposures = _ome(tif)
            else:
                description = (tif.pages[0].description or "").strip() if tif.pages else ""
                whole = {"ImageDescription": description} if description and not description.startswith("{") else {}
                channels, exposures = [], []
    except (OSError, ValueError, ET.ParseError):
        whole, channels, exposures = {}, [], []
    return {"image": whole, "channels": channels, "exposures": exposures}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:g}" if number > 0 else None


def _map_annotation(parent, annotation_id, record):
    item = ET.SubElement(parent, _q("MapAnnotation"), ID=annotation_id, Namespace=MAP_NAMESPACE)
    values = ET.SubElement(item, _q("Value"))
    for key, value in record.items():
        entry = ET.SubElement(values, _q("M"), K=str(key))
        entry.text = str(value)
    return item


def enrich_ome(path, channels, image_notes=()):
    """Add acquisition metadata to the OME-TIFF at *path* (written by SPACEMAP), in place.

    *channels*: one dict per written channel, with optional ``exposure`` ((value, unit)),
    ``excitation_nm``, ``emission_nm`` and ``notes`` (key -> value, shown as the channel's
    annotation). *image_notes*: (title, {key: value}) pairs, each an annotation of the image.
    """
    ET.register_namespace("", OME_NS)
    with tifffile.TiffFile(path) as tif:
        text = tif.pages[0].description
    root = _parse(text)
    image = root.find(_q("Image"))
    pixels = image.find(_q("Pixels"))
    written = pixels.findall(_q("Channel"))
    if len(written) != len(channels):
        raise ValueError(f"{Path(path).name} has {len(written)} channels; metadata was given for {len(channels)}.")
    structured = root.find(_q("StructuredAnnotations"))
    if structured is None:
        structured = ET.SubElement(root, _q("StructuredAnnotations"))
    for index, (channel, given) in enumerate(zip(written, channels)):
        for side in ("excitation", "emission"):
            value = _number(given.get(f"{side}_nm"))
            if value:
                name = "ExcitationWavelength" if side == "excitation" else "EmissionWavelength"
                channel.set(name, value)
                channel.set(name + "Unit", "nm")
        exposure = given.get("exposure")
        if exposure and _number(exposure[0]):
            ET.SubElement(pixels, _q("Plane"), TheZ="0", TheT="0", TheC=str(index),
                          ExposureTime=_number(exposure[0]), ExposureTimeUnit=exposure[1])
        if given.get("notes"):
            annotation_id = f"Annotation:SPACEMAP:Channel:{index}"
            _map_annotation(structured, annotation_id, given["notes"])
            reference = ET.Element(_q("AnnotationRef"), ID=annotation_id)
            light_path = channel.find(_q("LightPath"))  # OME orders AnnotationRef before LightPath
            channel.insert(list(channel).index(light_path) if light_path is not None else len(channel), reference)
    for number, (title, record) in enumerate(image_notes):
        if record:
            annotation_id = f"Annotation:SPACEMAP:Image:{number}"
            _map_annotation(structured, annotation_id, record).set("Namespace", MAP_NAMESPACE)
            structured[-1].insert(0, ET.Element(_q("Description")))
            structured[-1][0].text = title
            ET.SubElement(image, _q("AnnotationRef"), ID=annotation_id)  # after Pixels, as OME orders it
    if not len(structured):
        root.remove(structured)
    xml = '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(root, encoding="unicode")
    tifffile.tiffcomment(path, xml.encode("utf-8"))
