from pathlib import Path
from urllib.parse import quote
import pytest
from spacemap_gui.studio.flow_export import export,validate_svg


SVG='<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect x="10" y="10" width="180" height="80" fill="white"/><text x="20" y="50">Marker A</text></svg>'


@pytest.mark.parametrize('kind',['svg','pdf','png','tiff'])
def test_publication_formats_and_manifest(tmp_path,kind):
    if kind!='svg':pytest.importorskip('cairosvg')
    ws={'output_dir':str(tmp_path),'revision':3,'sources':[],'gates':[]}
    found=export(ws,{'name':'figure','format':kind,'dpi':300,'svg':'data:image/svg+xml,'+quote(SVG),'manifest':{'test':'immutable data'}})
    assert Path(found['file']).is_file() and Path(found['manifest']).is_file()
    if kind in ('png','tiff'):
        from PIL import Image
        with Image.open(found['file']) as image:
            assert image.width==625
    assert 'immutable data' in Path(found['manifest']).read_text()


def test_external_svg_resources_rejected():
    with pytest.raises(ValueError,match='External'):
        validate_svg(b'<svg><image href="file:///tmp/private"/></svg>')
    with pytest.raises(ValueError,match='Only figure'):
        validate_svg(b'<svg><script>alert(1)</script></svg>')
    validate_svg(b'<svg><g clip-path="url( \'#clip\' )"/></svg>')
    with pytest.raises(ValueError,match='External style'):
        validate_svg(b'<svg><g fill="url( https://example.org/image )"/></svg>')


def test_large_raster_export_is_bounded(tmp_path):
    with pytest.raises(ValueError,match='too large'):
        export({'output_dir':str(tmp_path)}, {'format':'png','dpi':1200,
            'svg':'data:image/svg+xml,'+quote(SVG.replace('200','20000'))})
