"""Image-driven matching: 44 tagged ROI tables remain 44 ROIs on 12 parent slides."""
import json

import pandas as pd
import pytest

from spacemap.viewer.catalog import sample_of
from spacemap_gui.studio import cell_inputs, correction, samples


@pytest.mark.parametrize('filename', ['P17_P19_roi_3.ome_cell_features_tagged.csv',
    'P17_P19_roi_3.ome_cell_features_tagged.parquet', 'P17_P19_roi_3.ome_cell_features_tagged.csv.parquet',
    'P17_P19_roi_3.ome_cell_features_annotated.csv', 'P17_P19_roi_3.ome.tiff'])
def test_measurements_and_images_share_one_roi_name(filename):
    assert sample_of(filename) == 'P17_P19_roi_3'


def test_44_tagged_tables_match_44_images_and_12_recorded_slides(tmp_path):
    images, tables, boundaries = (tmp_path / p for p in ('images', 'tables', 'data/boundaries'))
    for path in (images, tables, boundaries):
        path.mkdir(parents=True)
    ring = [[0,0],[10,0],[10,10],[0,10],[0,0]]
    for slide in range(12):
        features = []
        for roi in range(4 if slide < 8 else 3):
            name = f'P{slide}_A_roi_{roi+1}'
            (images / f'{name}.ome.tiff').touch()
            (tables / f'{name}.ome_cell_features_tagged.csv').write_text('label,A,B\n1,10,20\n')
            features.append({'type':'Feature', 'properties':{'name':f'roi_{roi+1}'}, 'geometry':{'type':'Polygon','coordinates':[ring]}})
        (boundaries / f'P{slide}_A_roi_bounding_boxes_coords.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    (tables / 'Unrelated.ome_cell_features_tagged.csv').write_text('label,A,B\n1,10,20\n')
    mappings = {'bc_images_dir':str(images),'bc_input_dir':str(tables)}
    plan = correction.plan(tmp_path,mappings,{})
    sheet = samples.table(tmp_path,mappings)
    assert len(plan['rows']) == plan['tables'] == len(sheet['rows']) == 44
    assert len({row['slide'] for row in sheet['rows']}) == 12
    assert next(u for u in plan['units'] if u['column']=='slide')['units'] == 12
    assert all(row['match']=='matched' for row in plan['rows']) and not plan['missing']
    # A missing measurement stays as an image row, never silently disappearing.
    next(tables.glob('P0*.csv')).unlink()
    plan = correction.plan(tmp_path,mappings,{})
    assert len(plan['rows']) == 44 and plan['tables'] == 43 and len(plan['missing']) == 1


def test_parquet_pairs_and_verified_caches_are_not_duplicate_samples(tmp_path):
    from spacemap import dataio
    csv = tmp_path / 'P1_roi_1.ome_cell_features_tagged.csv'
    pd.DataFrame({'label':[1,2], 'patient':['001','002']}).to_csv(csv,index=False)
    twin = dataio.ensure_fast_copy(csv, force=True)
    assert cell_inputs.table_index(tmp_path) == {'P1_roi_1':[str(csv)]}
    assert cell_inputs.read_path(csv) == twin
    csv.write_text(csv.read_text()+'\n')
    assert cell_inputs.read_path(csv) == csv  # stale cache must never be used
    direct = csv.with_suffix('.parquet')
    pd.read_csv(csv).to_parquet(direct,index=False)
    assert cell_inputs.table_index(tmp_path) == {'P1_roi_1':[str(direct)]}


def test_multiple_distinct_tables_are_shown_as_ambiguous(tmp_path):
    image = tmp_path / 'P1_roi_1.ome.tiff'
    image.touch()
    for suffix in ('_cell_features', '_cell_features_tagged'):
        (tmp_path / f'P1_roi_1.ome{suffix}.csv').write_text('label,A\n1,4\n')
    rows = cell_inputs.matches(tmp_path, {'bc_images_dir':str(tmp_path)}, tmp_path)
    assert rows[0]['match'] == 'ambiguous' and not rows[0]['path']
