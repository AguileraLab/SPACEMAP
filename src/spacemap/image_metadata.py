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
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
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
    ET.register_namespace("xsi", XSI_NS)
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


# Tags an OME Image holds before its Pixels, in the schema's order (copied from a parent when the child lacks them).
_IMAGE_HEAD = ("AcquisitionDate", "ExperimenterRef", "Description", "ExperimentRef", "ExperimenterGroupRef",
               "InstrumentRef", "ObjectiveSettings", "ImagingEnvironment", "StageLabel")


def _exposure_plane(pixels, index, value, unit):
    return ET.SubElement(pixels, _q("Plane"), TheZ="0", TheT="0", TheC=str(index), ExposureTime=str(value), ExposureTimeUnit=unit)


def inherit_ome(parent_path, child_path, channel_indices, notes=None):
    """Give an image made from another (an ROI crop, a background-removed copy) all of its parent's metadata, in place.

    ``channel_indices[j]`` is the parent channel written as child channel ``j``. Each child channel keeps
    its own name and colour (a rename or recolour chosen when it was made; the parent's name is then
    noted) and gets the parent channel's other attributes, its exposure plane and its annotations. The
    parent's image annotations, instruments, acquisition date and objective are copied, and *notes*
    (how this image was made) are added. SPACEMAP's non-standard ``Channel/@Exposure`` becomes OME's
    ``Plane/@ExposureTime``, so the result is valid OME. A parent that is not OME (a QPTIFF) is read
    with :func:`source_metadata` instead.
    """
    import copy

    ET.register_namespace("", OME_NS)
    ET.register_namespace("xsi", XSI_NS)
    with tifffile.TiffFile(parent_path) as tif:
        parent_xml = tif.ome_metadata if tif.is_ome else None
    with tifffile.TiffFile(child_path) as tif:
        child = _parse(tif.pages[0].description)
    image = child.find(_q("Image"))
    pixels = image.find(_q("Pixels"))
    channels = pixels.findall(_q("Channel"))
    if len(channels) != len(channel_indices):
        raise ValueError(f"{Path(child_path).name} has {len(channels)} channels; {len(channel_indices)} parent channels were given.")
    structured = child.find(_q("StructuredAnnotations"))
    if structured is None:
        structured = ET.SubElement(child, _q("StructuredAnnotations"))
    count = [0]

    def adopt(annotation):
        clone = copy.deepcopy(annotation)
        count[0] += 1
        clone.set("ID", f"Annotation:SPACEMAP:Inherited:{count[0]}")
        structured.append(clone)
        return clone.get("ID")

    def refer(channel, annotation_id):
        reference = ET.Element(_q("AnnotationRef"), ID=annotation_id)
        light_path = channel.find(_q("LightPath"))
        channel.insert(list(channel).index(light_path) if light_path is not None else len(channel), reference)

    if parent_xml is None:
        found = source_metadata(parent_path)
        for j, (channel, index) in enumerate(zip(channels, channel_indices)):
            exposure = found["exposures"][index] if index < len(found["exposures"]) else None
            if exposure and _number(exposure[0]):
                channel.attrib.pop("Exposure", None)
                _exposure_plane(pixels, j, _number(exposure[0]), exposure[1])
            record = found["channels"][index] if index < len(found["channels"]) else {}
            if record:
                annotation = _map_annotation(structured, f"Annotation:SPACEMAP:Inherited:C{j}", record)
                refer(channel, annotation.get("ID"))
        head = [(f"Acquisition: {Path(parent_path).name}", found["image"])] if found["image"] else []
    else:
        parent = _parse(parent_xml)
        parent_image = parent.find(_q("Image"))
        parent_pixels = parent_image.find(_q("Pixels"))
        parent_channels = parent_pixels.findall(_q("Channel"))
        planes = {}
        for plane in parent_pixels.findall(_q("Plane")):
            planes.setdefault(int(plane.get("TheC", 0)), plane)
        parent_structured = parent.find(_q("StructuredAnnotations"))
        annotations = {item.get("ID"): item for item in (list(parent_structured) if parent_structured is not None else [])}
        for j, (channel, index) in enumerate(zip(channels, channel_indices)):
            if index >= len(parent_channels):
                continue
            source = parent_channels[index]
            for key, value in source.attrib.items():
                if key not in {"ID", "Name", "Color", "SamplesPerPixel", "Exposure"}:
                    channel.set(key, value)
            old_exposure = channel.attrib.pop("Exposure", None) or source.get("Exposure")
            if index in planes:
                plane = copy.deepcopy(planes[index])
                plane.set("TheC", str(j))
                pixels.append(plane)
            elif _number(old_exposure):
                _exposure_plane(pixels, j, _number(old_exposure), "ms")
            for reference in source.findall(_q("AnnotationRef")):
                if reference.get("ID") in annotations:
                    refer(channel, adopt(annotations[reference.get("ID")]))
            if source.get("Name") and source.get("Name") != channel.get("Name"):
                renamed = _map_annotation(structured, f"Annotation:SPACEMAP:Renamed:C{j}", {"Name in the parent image": source.get("Name")})
                refer(channel, renamed.get("ID"))
        # The parent's instruments (before the Images, as OME orders them) and the image's own details.
        position = list(child).index(image)
        for offset, instrument in enumerate(parent.findall(_q("Instrument"))):
            if child.find(f"{_q('Instrument')}[@ID='{instrument.get('ID')}']") is None:
                child.insert(position + offset, copy.deepcopy(instrument))
        for tag in _IMAGE_HEAD:
            item = parent_image.find(_q(tag))
            if item is not None and image.find(_q(tag)) is None:
                order = [_local(child_item.tag) for child_item in image]
                after = [k for k, name in enumerate(order) if name in _IMAGE_HEAD[_IMAGE_HEAD.index(tag) + 1:] or name == "Pixels"]
                image.insert(after[0] if after else len(image), copy.deepcopy(item))
        for reference in parent_image.findall(_q("AnnotationRef")):
            if reference.get("ID") in annotations:
                ET.SubElement(image, _q("AnnotationRef"), ID=adopt(annotations[reference.get("ID")]))
        head = []
    for number, (title, record) in enumerate(head + ([("SPACEMAP", notes)] if notes else [])):
        if record:
            annotation = _map_annotation(structured, f"Annotation:SPACEMAP:Made:{number}", {str(k): str(v) for k, v in record.items()})
            description = ET.Element(_q("Description"))
            description.text = title
            annotation.insert(0, description)
            ET.SubElement(image, _q("AnnotationRef"), ID=annotation.get("ID"))
    # OME's order inside Pixels: Channels, then TiffData, then Planes.
    planes = [item for item in pixels if _local(item.tag) == "Plane"]
    for plane in planes:
        pixels.remove(plane)
    pixels.extend(sorted(planes, key=lambda plane: int(plane.get("TheC", 0))))
    if not len(structured):
        child.remove(structured)
    xml = '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(child, encoding="unicode")
    tifffile.tiffcomment(child_path, xml.encode("utf-8"))
