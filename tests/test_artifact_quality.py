"""Quality models must work beyond a named marker or an H&E appearance."""
import json
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from spacemap_gui.studio import artifact_learning as learning, artifact_quality as quality, artifacts
from test_artifact_learning import polygon
from test_studio_artifacts import marker_image


def quality_slide(path, kind, *, seed=3, affected=0, names=None, blank=False):
    rng = np.random.default_rng(seed)
    size = 192
    channels = [gaussian_filter(rng.gamma(3, 200, (size, size)), .7) for _ in range(3)]
    patch = (slice(40, 150), slice(40, 150))
    for index in [affected] if isinstance(affected, int) else affected:
        if kind == 'blur':
            channels[index][patch] = gaussian_filter(channels[index], 5)[patch]
        else:
            channels[index][patch] = 4800 + rng.normal(0, 5, (110, 110))
    if blank:
        channels.append(np.zeros((size, size)))
    # H&E is healthy on both sides of the artifact, including its texture.
    he = [np.clip(180 + rng.normal(0, 12, (size, size)), 0, 255) for _ in range(3)]
    names = names or ['CD3', 'PanCK', 'DAPI']
    if blank:
        names = [*names, 'Empty']
    marker_image(path, np.stack([*he, *channels]).astype(np.uint16), ['HE_Red', 'HE_Green', 'HE_Blue', *names])
    return path


@pytest.mark.parametrize('kind', ['blur', 'debris'])
@pytest.mark.parametrize('affected', [2, (0, 1, 2)])
def test_unseen_marker_panel_single_channel_defect(tmp_path, kind, affected):
    train = quality_slide(tmp_path / 'train.ome.tiff', kind)
    positive = [polygon(60, 60, 130, 130)]
    negative = [polygon(5, 5, 185, 25), polygon(5, 165, 185, 185)]
    started = time.perf_counter()
    learning.snapshot(tmp_path, kind, train, 1, positive, negative, [polygon(0, 0, 192, 192)], False)
    model = learning.train(learning.training_request(tmp_path, kind), lambda *a: None)['version']
    assert model['feature_schema'] == quality.SCHEMA_V2
    assert model['mode'] == 'supervised' and not model['he']
    # New texture, no shared marker names, a different affected channel, and an
    # entirely empty channel: H&E-only fallback cannot solve this example.
    test = quality_slide(tmp_path / 'test.ome.tiff', kind, seed=31, affected=affected,
                         names=['CD8', 'CD20', 'CD68'], blank=True)
    found = learning.predict(tmp_path, kind, model['id'], test, 1, [polygon(0, 0, 192, 192)], minimum_um2=0)
    assert found['matched_markers'] == [] and len(found['read_markers']) == 4
    assert found['used'] == 'Cross-marker quality model'
    area = learning._geometry(found['features'])
    inside, outside = box(60, 60, 130, 130), box(0, 0, 192, 192).difference(box(30, 30, 160, 160))
    recall = area.intersection(inside).area / inside.area
    false_positive = area.intersection(outside).area / outside.area
    print(f'{kind}: recall={recall:.3f}, clear-area false-positive={false_positive:.3f}, train+predict={time.perf_counter()-started:.2f}s')
    assert recall > .8 and false_positive < .15
    assert all(f['properties']['classification']['color'] == [230, 25, 75] for f in found['features'])


def test_quality_tile_seams_empty_channels_and_legacy_features(tmp_path):
    image = quality_slide(tmp_path / 'slide.ome.tiff', 'blur', blank=True)
    reader = quality.extractor(image, 1)
    full = reader.tile(0, 0, 192, 192)
    piece = reader.tile(70, 50, 60, 65)
    np.testing.assert_allclose(piece, full[50:115, 70:130], atol=2e-5)
    legacy = learning.Features(image, 1)
    np.testing.assert_allclose(piece[..., :len(legacy.names)], legacy.tile(70, 50, 60, 65), atol=2e-5)
    assert np.isfinite(full).all()
    fraction = full[..., reader.names.index('quality:markers:signal_fraction')]
    np.testing.assert_allclose(fraction, .75)
    cross = quality.extractor_v2(image, 1)  # the cross-marker set: also seamless, and no measurement names a marker
    whole = cross.tile(0, 0, 192, 192)
    np.testing.assert_allclose(cross.tile(70, 50, 60, 65), whole[50:115, 70:130], atol=2e-5)
    assert np.isfinite(whole).all() and not any(n.startswith(('marker:', 'texture:')) for n in cross.names)


def test_combined_detector_settings_and_scope(tmp_path, monkeypatch):
    before = learning.state(tmp_path, 'debris')
    values = {'detector_values': {'bright': 98., 'lint': 6.}, 'detectors': ['lint'], 'scale': 4}
    saved = learning.save_options(tmp_path, 'debris', values, before['revision'])
    assert saved['settings']['detectors'] == ['lint']
    assert learning.options(tmp_path, 'blur')['scale'] == 2
    with pytest.raises(ValueError):
        learning.save_options(tmp_path, 'debris', {'detectors': []}, saved['revision'])
    prepared = {'shape': (64, 64), 'inputs': {'microns_per_pixel': .5}}
    monkeypatch.setattr(artifacts, 'prepare', lambda *a, **kw: prepared)
    seen = []
    def lint(p, value, *, domain=None):
        seen.append((value, domain.copy()))
        return np.ones((64, 64), bool)
    monkeypatch.setattr(artifacts, 'lint_mask', lint)
    monkeypatch.setattr(artifacts, 'bright_mask', lambda *a: pytest.fail('Disabled bright detector ran'))
    found = learning.detect('fake', {}, 'debris', 2, None, [polygon(0, 0, 64, 64)], 0,
                            detector_values=values['detector_values'], detectors=['lint'])
    assert seen[0][0] == 6 and seen[0][1].sum() == 32 * 32
    area = learning._geometry(found['features'])
    assert area.difference(box(0, 0, 64, 64)).is_empty and area.area > 64 * 64 * .9


def test_shared_quality_blur_preview_and_legacy_settings(tmp_path, monkeypatch):
    home = learning.folder(tmp_path, 'debris')
    home.mkdir(parents=True)
    old = {'scale': 4, 'detector_values': {'bright': 98., 'lint': 6.}, 'detectors': ['lint']}
    path = home / 'settings.json'
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    state = learning.state(tmp_path, 'debris')
    assert path.read_bytes() == original  # Opening the new panel does not migrate files.
    assert set(state['detector_ranges']) == {'blur', 'bright', 'lint'}
    assert state['settings']['scale'] == 4 and state['settings']['detectors'] == ['lint']
    values = {**state['settings'], 'detectors': ['blur']}
    learning.save_options(tmp_path, 'debris', values, state['revision'])
    prepared = {'shape': (64, 64), 'inputs': {'microns_per_pixel': .5}}
    monkeypatch.setattr(artifacts, 'prepare', lambda *a, **kw: prepared)
    monkeypatch.setattr(artifacts, 'blur_mask', lambda *a: np.ones((64, 64), bool))
    for name in ('bright_mask', 'lint_mask'):
        monkeypatch.setattr(artifacts, name, lambda *a, **kw: pytest.fail('Disabled detector ran'))
    found = learning.detect('fake', {}, 'debris', 2, None, [polygon(0, 0, 64, 64)], 0,
                            detector_values=values['detector_values'], detectors=['blur'])
    assert found['features']
    assert learning._geometry(found['features']).difference(box(0, 0, 64, 64)).is_empty
    assert found['features'][0]['properties']['classification']['name'] == 'Image_quality'


def test_positive_only_quality_model_preserves_snapshot(tmp_path):
    image = quality_slide(tmp_path / 'one.ome.tiff', 'debris')
    learning.snapshot(tmp_path, 'debris', image, 2, [polygon(60, 60, 130, 130)], [], [polygon(0, 0, 192, 192)], False)
    model = learning.train(learning.training_request(tmp_path, 'debris'), lambda *a: None)['version']
    assert model['mode'] == 'artifact_only'
    saved = json.loads((learning.folder(tmp_path, 'debris') / (model['examples'][0] + '.json')).read_text())
    assert saved['annotations']['negative'] == [] and saved['negative'] == 0
    assert saved['feature_schema'] == 'quality_v2'
    assert 'artifact_quality.py' in model['software']['source_sha256']
    assert learning.predict(tmp_path, 'debris', model['id'], image, 2, [polygon(0, 0, 192, 192)], minimum_um2=0)['features']


def test_one_quality_model_learns_both_blur_and_bright_regions(tmp_path):
    domain = [polygon(0, 0, 192, 192)]
    red = [polygon(60, 60, 130, 130)]
    green = [polygon(5, 5, 185, 25), polygon(5, 165, 185, 185)]
    for kind in ('blur', 'debris'):
        image = quality_slide(tmp_path / (kind + '.ome.tiff'), kind)
        learning.snapshot(tmp_path, 'debris', image, 1, red, green, domain, False)
    version = learning.train(learning.training_request(tmp_path, 'debris'), lambda *a: None)['version']
    assert version['images'] == 2 and version['artifact_targets'] == ['blur', 'bright', 'lint']
    for kind in ('blur', 'debris'):
        image = quality_slide(tmp_path / ('unseen-' + kind + '.ome.tiff'), kind, seed=31,
                              affected=2, names=['CD8', 'CD20', 'CD68'])
        found = learning.predict(tmp_path, 'debris', version['id'], image, 1, domain, minimum_um2=0)
        area = learning._geometry(found['features'])
        inside = box(60, 60, 130, 130)
        outside = box(0, 0, 192, 192).difference(box(30, 30, 160, 160))
        assert area.intersection(inside).area / inside.area > .8
        assert area.intersection(outside).area / outside.area < .15
