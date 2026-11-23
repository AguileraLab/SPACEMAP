"""Ambiguous tissue names require an explicit GeoJSON in the existing Area menu."""
import json

from viewer_threshold_browser import main


def setup(root):
    from spacemap_gui.studio import annotation_inputs
    image = root / 'roi.ome.tiff'
    folder = annotation_inputs.import_folder(root, image)
    folder.mkdir(parents=True)
    for filename, x in [('aaaa_review.geojson', 10), ('bbbb_merged.geojson', 100), ('cccc_merged.geojson', 180)]:
        (folder / filename).write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'properties': {'classification': {'name': 'tissue'}},
            'geometry': {'type': 'Polygon', 'coordinates': [[[x,10],[x+40,10],[x+40,60],[x,60],[x,10]]]}}]}))
    (root / (root.name + '.json')).write_text(json.dumps({'directory_mappings': {'tab4_input_dir': str(root)},
        'tab_configurations': {'tab5_segmentation': {'tab4_mask_layer_matches': {'roi.ome.tiff': 'imported:tissue'}}}}))


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(n,v)=>{n.value=v;n.dispatchEvent(new Event('change'));};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 const area=()=>document.querySelector('[aria-label="Where to segment roi.ome.tiff"]');
 const row=()=>area()?.closest('tr');
 const data=async()=>(await(await fetch('/spacemap-viewer/api/studio/browser-check/segmentation_images')).json());
 await until(()=>area()&&row().textContent.includes('Choose source'));
 assert(area().value==='imported:tissue'&&area().selectedOptions[0].disabled,'An ambiguous saved name stays unresolved rather than selecting an arbitrary file');
 const groups=[...area().querySelectorAll('optgroup')].filter(g=>g.label.startsWith('File · '));
 assert(groups.length===3&&new Set(groups.map(g=>g.label)).size===3,'Each GeoJSON has a distinct source group, including identically named imports');
 const choices=groups.flatMap(g=>[...g.querySelectorAll('option')]);
 assert(choices.length===3&&choices.every(o=>o.textContent==='tissue'),'Source groups retain the annotation name');
 const bulk=document.querySelector('[aria-label="Area of the ticked images"]');
 assert(![...bulk.options].some(o=>o.value.startsWith('source:')),'A file tied to one image cannot be applied to every image');
 const chosen=groups.find(g=>g.label==='File · review').querySelector('option').value;
 change(area(),chosen);
 await until(async()=>{const p=await data();return p.rows.find(r=>r.image==='roi.ome.tiff').area===chosen;});
 await until(()=>area()?.value===chosen&&!row().textContent.includes('Choose source'));
 assert(area().title.includes('/aaaa_review.geojson'),'The selected source full path is available in help');
 document.querySelector('[data-step=tab7]').click();await delay(150);document.querySelector('[data-step=tab5]').click();
 await until(()=>area()?.value===chosen);
 assert(!area().selectedOptions[0].disabled,'Explicit source survives navigation');
 const p=await data(),r=p.rows.find(r=>r.image==='roi.ome.tiff');
 assert(!r.ambiguous&&!r.missing&&r.area_source.endsWith('/aaaa_review.geojson'),'The selected source agrees between the image row and API');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=150, setup=setup)
