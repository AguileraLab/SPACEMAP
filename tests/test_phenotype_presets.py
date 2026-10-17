import pytest

from spacemap import phenotype_presets as presets


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path))
    return tmp_path


def test_default_names_and_alias_colours(home):
    assert [item["name"] for item in presets.read()["entries"]] == list(presets.BASE_COLOURS)
    assert len(presets.read()["entries"]) == 12
    assert presets.colours()["Endothelia"] == presets.colours()["Endothelial"]
    assert not presets.path().exists()


def test_user_preset_survives_reload_and_applies_across_projects(home):
    from spacemap.acc_utils import build_color_mapping
    from spacemap_gui.studio import colours
    first = presets.read()
    entries = [{"name":"Epithelial", "color":"#123ABC"}, {"name":"New type", "color":"#abcdef"}]
    saved = presets.save(entries, first["revision"])
    presets._read.cache_clear()
    assert presets.read() == saved
    assert build_color_mapping(["Epithelial", "Epithelia", "New type"]) == {
        "Epithelial":"#123abc", "Epithelia":"#123abc", "New type":"#abcdef"}
    assert colours.saved({})["New_type"] == "#abcdef"
    assert colours.saved({"tab_configurations":{"tab9_visualization":{"tab9_actual_cell_color_val_New_type":"#999999"}}})["New_type"] == "#abcdef"
    assert [item["name"] for item in saved["entries"]] == ["Epithelial", "New type"]
    with pytest.raises(ValueError, match="changed"):
        presets.save(entries, first["revision"])


@pytest.mark.parametrize("entries", [None, [{"name":"A","color":"red"}], [{"name":"A\nB","color":"#123456"}],
    [{"name":"T cell","color":"#123456"},{"name":"T-cell","color":"#234567"}]])
def test_invalid_presets_do_not_replace_saved_entries(home, entries):
    before = presets.read()
    with pytest.raises(ValueError):
        presets.save(entries, before["revision"])
    assert presets.read() == before


def test_legacy_default_palette_upgrades_but_custom_edits_survive(home):
    import json
    target = presets.path()
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'entries':[
        {'name':'Epithelial','color':presets._OLD_BASE_COLOURS['Epithelial']},
        {'name':'CD8','color':'#123456'},
    ]}))
    palette = presets.colours()
    assert palette['Epithelial'] == '#7a3e85'
    assert palette['Epithelia'] == '#7a3e85'
    assert palette['CD8'] == '#123456'


def test_new_colours_are_distinct_and_stable_across_subsets_and_restarts(home):
    first = presets.ensure(['New alpha', 'New beta', 'epithelia'])
    assert first['epithelia'] == presets.BASE_COLOURS['Epithelial']
    assert first['New alpha'] != first['New beta']
    assert first['New alpha'] not in presets.BASE_COLOURS.values()
    presets._read.cache_clear()
    assert presets.ensure(['New beta'])['New beta'] == first['New beta']
    assert presets.ensure(['New gamma', 'New alpha'])['New alpha'] == first['New alpha']
    assert len(presets.read()['entries']) == 12  # category colours do not pollute the vocabulary
    many = presets.ensure([f'Novel {i}' for i in range(65)])
    assert len(set(many.values())) == 65  # no wrapping the former 20/60-colour cycle


def test_preset_auto_colour_and_alias_edit_propagate(home):
    initial = presets.read()
    entries = initial['entries'] + [{'name':'Novel phenotype','color':None}]
    presets.save(entries, initial['revision'])
    chosen = presets.ensure(['Novel phenotype'])['Novel phenotype']
    assert chosen not in presets.BASE_COLOURS.values()
    presets.update({'Epithelia':'#123456'})
    assert presets.ensure(['Epithelial','Epithelia','EPITHELIA']) == dict.fromkeys(
        ['Epithelial','Epithelia','EPITHELIA'], '#123456')
    assert presets.ensure(['Novel phenotype'])['Novel phenotype'] == chosen


def test_layer_colour_lookup_ignores_old_method_colours_and_reloads_edits(home, tmp_path):
    import numpy as np
    import pandas as pd
    from spacemap_gui.viewer import api
    table = tmp_path / 'cells.csv'
    pd.DataFrame({'batch_id':['roi']*3, 'label':[1,2,3],
                  'phenotype_som':['Epithelia','CD8','New phenotype'],
                  'phenotype_leiden':['Epithelial','CD8','New phenotype']}).to_csv(table,index=False)
    layer = {'roi':'roi','tables':{'combined':str(table)},'colors':{'Epithelia':'#ff00ff','Epithelial':'#0000ff','CD8':'#00ff00'}}
    som, _ = api._cell_lut(layer,'combined','phenotype_som',frozenset(),'viridis')
    leiden, _ = api._cell_lut(layer,'combined','phenotype_leiden',frozenset(),'viridis')
    assert np.array_equal(som, leiden)
    assert som[1].tolist() == [122,62,133,255]
    assert som[2].tolist() == [214,39,40,255]
    # Another project can have a different cached project palette; the user-wide edit wins.
    presets.update({'Epithelial':'#123456'})
    updated, _ = api._cell_lut(layer,'combined','phenotype_som',frozenset(),'viridis')
    assert updated[1].tolist() == [18,52,86,255]
    assert updated[3].tolist() == som[3].tolist()


def test_concurrent_allocations_preserve_all_names(home):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(presets.ensure, [['Novel A'],['Novel B'],['Novel A','Novel C']]))
    combined = presets.ensure(['Novel A','Novel B','Novel C'])
    for value in values:
        assert all(combined[name] == colour for name,colour in value.items())
    assert len(set(combined.values())) == 3
