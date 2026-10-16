"""The project's lineage finds each ROI's slide; Samples links the study's own table by ROI or by slide."""

import json

import pytest

from spacemap_gui.studio import lineage, samples


def box(label, x):
    ring = [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"name": label}}


@pytest.fixture
def project(tmp_path):
    """Slide S1 registered and cut into two ROIs; S2 has one cell table but no ROI file."""
    root = tmp_path / "study"
    for name in ("data/codex/S1_codex.qptiff", "data/he/S1_he.qptiff", "outputs/registered/S1_merged.ome.tiff"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"")
    shapes = root / "data/boundaries"
    shapes.mkdir(parents=True)
    (shapes / "S1_roi_bounding_boxes_coords.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [box("roi_1", 0), box("roi_2", 20)]})
    )
    tables = root / "outputs/segmentation"
    tables.mkdir(parents=True)
    images = root / "outputs/rois"
    images.mkdir(parents=True)
    for roi in ("S1_roi_1", "S1_roi_2", "S2_roi_1"):
        (images / f"{roi}.ome.tiff").touch()
        (tables / f"{roi}.ome_cell_features.csv").write_text("label,CD3\n1,5\n")
    return root, {}


def test_the_lineage_traces_each_roi_to_its_slide_and_images(project):
    root, mappings = project
    found = lineage.trace(root, mappings)
    s1 = next(item for item in found["slides"] if item["slide"] == "S1")
    assert s1["codex"] == "S1_codex.qptiff" and s1["he"] == "S1_he.qptiff" and s1["registered"] == "S1_merged.ome.tiff"
    assert s1["rois"] == ["S1_roi_1", "S1_roi_2"]
    by_roi = {item["roi"]: item for item in found["rois"]}
    assert by_roi["S1_roi_1"]["found"] == "roi step" and by_roi["S1_roi_1"]["table"] == "S1_roi_1.ome_cell_features.csv"
    assert by_roi["S2_roi_1"]["slide"] == "S2" and by_roi["S2_roi_1"]["found"] == "name"


def test_samples_link_a_table_by_slide_and_keep_hand_corrections(project):
    root, mappings = project
    shown = samples.table(root, mappings)
    assert [(row["roi"], row["slide"]) for row in shown["rows"]] == [
        ("S1_roi_1", "S1"),
        ("S1_roi_2", "S1"),
        ("S2_roi_1", "S2"),
    ]
    assert not shown["saved"]
    # One row per slide (from the clinic), with an Excel byte-order mark and names as the scanner wrote them.
    text = "﻿Slide name;Outcome;Arm\nS1_codex.qptiff;CR;A\nS2;PD;B\nS9;SD;A\n"
    guess = samples.suggest_link(root, mappings, text)
    assert guess["link_column"] == "Slide name" and guess["link_to"] == "slide" and guess["matches"] == 2
    preview = samples.link(root, mappings, text, "Slide name", "slide", {"Outcome": "response", "Arm": ""})
    assert preview["matched"] == 3 and preview["unused"] == ["S9"] and not samples.path(root).exists()
    samples.link(root, mappings, text, "Slide name", "slide", {"Outcome": "response", "Arm": ""}, apply=True)
    assert samples.values(root)["S1_roi_2"] == {"slide": "S1", "response": "CR", "Arm": "A"}
    samples.edit(root, mappings, "S2_roi_1", "slide", "S1")
    samples.rename(root, mappings, "Arm", "arm")
    kept = samples.table(root, mappings)
    row = next(item for item in kept["rows"] if item["roi"] == "S2_roi_1")
    assert row["slide"] == "S1" and row["_suggested"] == "S2" and "arm" in kept["columns"]
    assert samples.drop(root, mappings, "arm")["columns"] == ["roi", "slide", "response"]


def test_samples_refuse_what_would_be_ambiguous(project):
    root, mappings = project
    with pytest.raises(ValueError, match="more than one row"):
        samples.link(root, mappings, "roi,x\nS1_roi_1,1\nS1_roi_1,2\n", "roi", "roi", {"x": ""})
    with pytest.raises(ValueError, match="roi and slide"):
        samples.link(root, mappings, "roi,x\nS1_roi_1,1\n", "roi", "roi", {"x": "Slide"})
    with pytest.raises(ValueError, match="No roi"):
        samples.link(root, mappings, "name,x\nnope,1\n", "name", "roi", {"x": ""})
    with pytest.raises(ValueError):
        samples.read_table("names\nS1_roi_1\n")
    with pytest.raises(ValueError):
        samples.edit(root, mappings, "S1_roi_1", "slide", "")
    with pytest.raises(ValueError):
        samples.rename(root, mappings, "roi", "name")


def test_roi_step_manual_file_match_defines_parent_slide(project):
    from spacemap_gui.studio import rois
    from spacemap_gui.studio.settings import config_path
    root, mappings = project
    (root / 'data/boundaries/custom_regions.geojson').write_text(json.dumps({
        'type': 'FeatureCollection', 'features': [box('Tumor centre', 0)]}))
    config = {'tab_configurations': {'tab3_roi_extraction': {
        rois.FILES_KEY: {'P17_P19_merged.ome.tiff': 'custom_regions.geojson'}}}}
    config_path(root).parent.mkdir(parents=True, exist_ok=True)
    config_path(root).write_text(json.dumps(config))
    (root / 'outputs/rois/P17_P19_Tumor_centre.ome.tiff').touch()
    row = next(row for row in samples.table(root, mappings)['rows'] if row['roi'].startswith('P17_P19_'))
    assert row['slide'] == 'P17_P19' and row['_found'] == 'roi step'
    assert row['_source'] == 'custom_regions.geojson'


def test_manual_clinical_column_is_saved_and_editable(project):
    root, mappings = project
    found = samples.add(root, mappings, 'Stage')
    assert 'Stage' in found['columns']
    samples.edit(root, mappings, 'S1_roi_1', 'Stage', 'II')
    assert samples.values(root)['S1_roi_1']['Stage'] == 'II'
    with pytest.raises(ValueError, match='already'):
        samples.add(root, mappings, 'Stage')


def test_samples_includes_tables_from_a_relative_correction_folder(project):
    root, mappings = project
    folder = root / 'custom_measurements'
    folder.mkdir()
    (folder / 'P17_P19_roi_3.ome_cell_features.csv').write_text('label,CD3\n1,5\n')
    (root / 'outputs/rois/P17_P19_roi_3.ome.tiff').touch()
    found = samples.table(root, {**mappings, 'bc_input_dir': 'custom_measurements'})
    row = next(row for row in found['rows'] if row['roi'] == 'P17_P19_roi_3')
    assert row['slide'] == 'P17_P19' and row['_found'] == 'name'


def test_saved_tagged_alias_is_coalesced_without_losing_clinical_values(project):
    import pandas as pd
    root, mappings = project
    target = samples.path(root)
    target.parent.mkdir(parents=True)
    pd.DataFrame([
        {'roi':'S1_roi_1','slide':'S1','Stage':''},
        {'roi':'S1_roi_1.ome_cell_features_tagged','slide':'S1_roi_1.ome_cell_features_tagged','Stage':'II'},
    ]).to_csv(target,index=False)
    shown = samples.table(root,mappings)
    assert len(shown['rows']) == 3
    assert shown['rows'][0]['slide'] == 'S1' and shown['rows'][0]['Stage'] == 'II'
    samples.edit(root,mappings,'S1_roi_1','Stage','III')
    assert len(pd.read_csv(target)) == 3


def test_dataset_column_join_is_applied_per_cell_and_survives_reload(project):
    import pandas as pd
    from spacemap_gui.studio import correction, sample_links
    root, mappings = project
    for source in (root/'outputs/segmentation').glob('*.csv'):
        pd.DataFrame({'label':[1,2,3], 'PatientID':['001','002','NA'], 'CD3':[2,4,8]}).to_csv(source,index=False)
    from spacemap import dataio
    dataio.ensure_fast_copy(source, force=True)
    text = 'Patient,Response\n001,CR\n002,PD\nNA,SD\n'
    options = samples.suggest_link(root,mappings,text)['targets']
    assert any(item['value']=='column:PatientID' for item in options)
    preview = samples.link(root,mappings,text,'Patient','column:PatientID',{'Response':'response'})
    assert preview['matched']==3 and preview['matched_cells']==9
    assert not sample_links.path(root).exists()
    applied = samples.link(root,mappings,text,'Patient','column:PatientID',{'Response':'response'},apply=True)
    assert all(row['response']=='Varies by cell' for row in applied['table']['rows'])
    config = correction.config_of({})
    settings = correction.job_settings(root,mappings,config,['CD3'])
    cells,_ = correction._load(settings['tables'], settings['samples'], 'batch_id', config, lambda *a:None, settings['sample_links'])
    assert len(cells)==9 and cells.response.tolist()==['CR','PD','SD']*3
    samples.rename(root,mappings,'response','outcome')
    samples.edit(root,mappings,'S1_roi_1','outcome','Reviewed')
    frame=pd.DataFrame({'PatientID':['001','002']})
    assert sample_links.apply(frame,'S1_roi_1',sample_links.read(root)).outcome.tolist()==['Reviewed','Reviewed']
    samples.drop(root,mappings,'outcome')
    assert sample_links.read(root)==[]


def test_parquet_matching_column_and_table_filenames(project):
    import pandas as pd
    root,mappings=project
    folder=root/'parquet'
    folder.mkdir()
    mappings={'bc_input_dir':str(folder)}
    for roi in ('S1_roi_1','S1_roi_2','S2_roi_1'):
        pd.DataFrame({'Patient':['P1' if roi.startswith('S1') else 'P2'], 'CD3':[4]}).to_parquet(folder/f'{roi}.ome_cell_features_tagged.parquet')
    result=samples.link(root,mappings,'Case,Age\nP1,60\nP2,70\n','Case','column:Patient',{'Age':'age'},apply=True)
    assert result['matched']==3 and result['table']['rows'][0]['age']=='60'
    result=samples.link(root,mappings,'File,Region\nS1_roi_1.ome_cell_features_tagged.csv,Tumor\n', 'File','table',{'Region':'region'},apply=True)
    assert result['matched']==1 and result['table']['rows'][0]['region']=='Tumor'
    with pytest.raises(ValueError,match='more than one row'):
        samples.link(root,mappings,'Case,Age\nP1,60\nP1,70\n','Case','column:Patient',{'Age':'age'})
