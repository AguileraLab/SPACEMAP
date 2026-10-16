"""Browser-check that a layer in pieces is used as one region: Auto's "in" list and Segmentation's "Segment" list offer
the whole tissue (not each piece), Selected polygons measures only the picked pieces.

Usage: PYTHONPATH=src:tests python tests/viewer_measure_regions_browser.py
"""

from viewer_studio_regions_browser import setup as marker_image
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0; const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
setTimeout(async()=>{try {
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>tissue()?.features?.length===3 && s.imagesReady);
 const image=()=>s.layers.find(l=>l.type==='image');
 const scope=()=>document.querySelector('#panel [aria-label="Measure within"]');
 await until(()=>scope() && Array.from(scope().options).some(o=>o.textContent.includes('3 polygons')));
 const texts=Array.from(scope().options,o=>o.textContent);
 assert(texts.includes(tissue().name+' (3 polygons)') && texts.includes('Selected polygons') && !texts.some(t=>/tissue_2/.test(t))
   && Array.from(scope().options).find(o=>o.textContent===tissue().name+' (3 polygons)').parentElement.label==='markers',
   'The "in" list offers the whole tissue under the image\'s name, and Selected polygons, not each piece: '+texts.join(' | '));
 const auto=async value=>{
   scope().value=value; scope().dispatchEvent(new Event('change'));
   image().autoMessage='';
   button(document.querySelector('#panel'),'Auto all markers').click();
   await until(()=>!image().autoRun && image().autoMessage);
   return image().autoMessage;
 };
 const pixels=text=>Number((text.match(/([\d,]+) pixels/)||[])[1]?.replace(/,/g,''));
 const whole=pixels(await auto(JSON.stringify({shape:tissue().token,group:tissue().name})));
 s.selected=null; s.picks=null;
 const none=await auto(JSON.stringify({selected:true}));
 assert(/Select polygons on the image first/.test(none),'Selected polygons asks for a selection first: '+none);
 s.selected={layer:tissue(),id:tissue().features[0].properties.id};
 const oneText=await auto(JSON.stringify({selected:true})), one=pixels(oneText);
 assert(whole>0 && one>0 && one<whole,'The whole tissue counts all pieces; one picked piece counts less ('+one+' < '+whole+'): '+oneText);
 document.querySelector('[data-step=tab5]').click();
 const all=()=>tools().querySelector('[aria-label="Area of the ticked images"]');
 await until(()=>all() && Array.from(all().options).some(o=>o.value==='local:tissue'));
 assert(!tools().querySelector('[aria-label="Where to segment"]'),'One place to choose the area: each image\'s row (and Set all)');
 all().value='local:tissue'; all().dispatchEvent(new Event('change'));
 await until(()=>/without their area/.test(tools().textContent) && tools().querySelector('[aria-label="Where to segment markers.ome.tiff"]').value==='local:tissue');
 assert(/without their area/.test(tools().textContent),'Images without a saved tissue are counted, so nothing runs on the whole image by mistake');
 assert(button(document.querySelector('.studio-runbar'),'Selected polygons'),'Segmentation offers inside tissue, and Selected polygons for this image');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(0,600)));}},100);
</script>
"""


def setup(root):
    """The marker image and a saved tissue in three separate pieces."""
    import json

    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    from spacemap_gui.studio import annotation

    marker_image(root)
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    path = annotation.valid_tissue(output, "markers")
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = unary_union([box(20, 20, 120, 300), box(150, 20, 250, 300), box(280, 20, 380, 300)])
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(pieces)}]}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
