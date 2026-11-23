"""Publication exports and their reproducible plot/layout manifests."""
import base64
from io import BytesIO
import json
from pathlib import Path
import re
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import unquote_to_bytes


def decode(value, expected):
    if not isinstance(value,str) or not value.startswith('data:'+expected) or len(value)>100_000_000:
        raise ValueError('Export needs a valid image from the figure.')
    header,body=value.split(',',1)
    return base64.b64decode(body,validate=True) if ';base64' in header else unquote_to_bytes(body)


def validate_svg(raw,depth=0):
    if depth>8 or re.search(rb'<!ENTITY|<!DOCTYPE',raw,re.I):
        raise ValueError('Unsupported SVG document.')
    root=ET.fromstring(raw)
    for element in root.iter():
        if element.tag.split('}')[-1].lower() in ('script','foreignobject'):
            raise ValueError('Only figure graphics are allowed in an export.')
        for name,value in element.attrib.items():
            local=name.split('}')[-1].lower()
            if local.startswith('on'):
                raise ValueError('SVG event handlers are not allowed.')
            if local=='href' and not value.startswith('#'):
                if value.startswith('data:image/svg+xml'):
                    validate_svg(decode(value,'image/svg+xml'),depth+1)
                elif not value.startswith(('data:image/png','data:image/jpeg')):
                    raise ValueError('External image references are not allowed.')
            for reference in re.findall(r'url\((.*?)\)',value,re.I):
                if not reference.strip().strip('\"\x27').startswith('#'):
                    raise ValueError('External style references are not allowed.')


def export(workspace,body):
    from .flow_gating import atomic_json
    folder=Path(workspace['output_dir']).expanduser()
    folder.mkdir(parents=True,exist_ok=True)
    name=re.sub(r'[^\w .-]+','_',str(body.get('name','flow-figure'))).strip(' .')[:100] or 'flow-figure'
    kind=body.get('format','png')
    if kind not in ('png','svg','pdf','tiff'):
        raise ValueError('Choose PNG, SVG, PDF or TIFF.')
    dpi=int(body.get('dpi',300))
    if not 72<=dpi<=1200:
        raise ValueError('Use 72–1200 DPI.')
    if body.get('svg'):
        svg=decode(body['svg'],'image/svg+xml');validate_svg(svg)
        root=ET.fromstring(svg)
        dimensions=[float(re.sub(r'px$', '', root.get(key,'0'))) for key in ('width','height')]
        if min(dimensions)<=0 or dimensions[0]*dimensions[1]*(dpi/96)**2>80_000_000:
            raise ValueError('Export is too large. Reduce the page size or DPI (maximum 80 megapixels).')
    if kind in ('svg','pdf'):
        raw=decode(body.get('svg'),'image/svg+xml')
        validate_svg(raw)
        if kind=='pdf':
            try:
                import cairosvg
            except ImportError as exc:
                raise ValueError('PDF export needs CairoSVG. SVG preserves vector text and is available now.') from exc
            raw=cairosvg.svg2pdf(bytestring=raw,unsafe=False)
    else:
        from PIL import Image
        if body.get('png'):
            source=decode(body['png'],'image/png')
        else:
            import cairosvg
            svg=decode(body.get('svg'),'image/svg+xml');validate_svg(svg)
            source=cairosvg.svg2png(bytestring=svg,scale=dpi/96,unsafe=False)
        with Image.open(BytesIO(source)) as image:
            buffer=BytesIO()
            image.save(buffer,format='TIFF' if kind=='tiff' else 'PNG',dpi=(dpi,dpi),**({'compression':'tiff_lzw'} if kind=='tiff' else {}))
            raw=buffer.getvalue()
    # New exports never silently replace a publication file.
    stem=name+'-'+uuid.uuid4().hex[:8]
    target=folder/(stem+'.'+kind)
    temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_bytes(raw)
    manifest={'version':1,'workspace_revision':workspace['revision'],'sources':workspace['sources'],
              'gates':workspace['gates'],'dpi':dpi,**body.get('manifest',{})}
    atomic_json(folder/(stem+'.json'),manifest)
    temporary.replace(target)
    mime={'png':'image/png','tiff':'image/tiff','svg':'image/svg+xml','pdf':'application/pdf'}[kind]
    return {'file':str(target),'manifest':str(folder/(stem+'.json')),
            'download':'data:'+mime+';base64,'+base64.b64encode(raw).decode()}
