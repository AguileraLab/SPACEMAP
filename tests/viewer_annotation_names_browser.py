"""Browser-check Studio's Annotations in Layers by name: the image's block, rename moves polygons between names, hiding
and colouring a name, Export ticks (the project's rules), Duplicate, and Export by the rules.

Usage: PYTHONPATH=src:tests python tests/viewer_annotation_names_browser.py
"""

from viewer_measure_regions_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0; const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
const studioApi=path=>'/spacemap-viewer/api/studio/browser-check'+path;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent('page error: '+e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try {
 const s=window.__state, tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady && document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 fetch('/result?progress='+encodeURIComponent('step open'));
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 fetch('/result?progress='+encodeURIComponent('image clicked'));
 const tissue=()=>s.layers.find(l=>l.id==='tissue-regions');
 await until(()=>tissue()?.features?.length===3 && s.imagesReady);
 s.layerGroups.annotations=true; buildPanel();
 fetch('/result?progress='+encodeURIComponent('tissue loaded, panel built'));
 const row=name=>document.querySelector('#panel [data-annotation-name="'+name+'"]');
 await until(()=>row('tissue'));
 assert(document.querySelector('#panel [data-annotation-source="markers"]') && row('tissue').querySelector('.kind').textContent==='3' && /Tissue/.test(row('tissue').querySelector('.layer-head').title),
   'Annotations has a block for the open image, and tissue is one row of 3 polygons, from Tissue');
 // Rename one piece: it moves to its own name row.
 const id=tissue().features[0].properties.id;
 nameRegions(tissue(), new Set([id]), 'lymph_node');
 await until(()=>row('lymph_node'));
 assert(row('lymph_node').querySelector('.kind').textContent==='1' && row('tissue').querySelector('.kind').textContent==='2','Renamed polygons form their own name row');
 const drawn=()=>document.querySelectorAll('#shapes g[data-layer="tissue-regions"] path').length;
 row('lymph_node').querySelector('[aria-label="Show lymph_node"]').click();
 await until(()=>drawn()===2);
 assert(tissue().visible,'Hiding a name hides only its polygons');
 const colour=row('tissue').querySelector('[aria-label="Colour of tissue"]');colour.value='#00ff00';colour.dispatchEvent(new Event('change'));
 await until(()=>tissue().features.filter(f=>annotationName(tissue(),f)==='tissue').every(f=>shapeColor(f)==='#00ff00'));
 assert(true,'A name takes one colour');
 const ticks=async()=>(await (await fetch(studioApi('/annotation_names?image_token='+encodeURIComponent(s.layers.find(l=>l.type==='image').token)))).json()).choices;
 assert(!row('lymph_node').querySelector('[aria-label="Export lymph_node"]').checked,'Nothing is ticked for Export until chosen');
 row('lymph_node').querySelector('[aria-label="Export lymph_node"]').click();
 await until(async()=>(await ticks())['::lymph_node::']===true);
 await until(()=>row('lymph_node').querySelector('[aria-label="Export lymph_node"]').checked);
 row('lymph_node').querySelector('[aria-label="Export lymph_node"]').click();
 await until(async()=>(await ticks())['::lymph_node::']===false);
 assert(true,'Export ticks are saved with the project');
 window.prompt=()=>'primary';
 row('tissue').querySelector('[aria-label="Duplicate tissue"]').click();
 await until(()=>row('primary'));
 assert(row('primary').querySelector('.kind').textContent==='2' && /Copy of tissue/.test(row('primary').querySelector('.layer-head').title),'Duplicate makes a new name row with a copy of the polygons');
 // Delete: a copy leaves the project whole; a name in the tissue loses its polygons only.
 window.confirm=()=>true;
 const copyLayer=s.layers.find(l=>l.origin==='Copy of tissue');
 row('primary').querySelector('[aria-label="Delete primary"]').click();
 await until(()=>!row('primary')&&!s.layers.includes(copyLayer));
 await until(async()=>!(await (await fetch(studioApi('/annotation_export'))).json()).names.some(n=>n.name==='primary'));
 assert(true,'Deleting a copy takes its layer out of the project');
 nameRegions(tissue(), new Set([tissue().features[1].properties.id]), 'spare');
 await until(()=>row('spare'));
 row('spare').querySelector('[aria-label="Delete spare"]').click();
 await until(()=>!row('spare')&&tissue().features.length===2);
 assert(true,'Deleting a name in the tissue deletes its polygons and keeps the tissue');
 // Export: the rules table lists the open image's annotations; Layers' ticks are its Export ticks.
 button(document.querySelector('.studio-stage-tabs'),'Export').click();
 const ruleRow=rule=>tools().querySelector('.annotation-rule[data-rule="'+rule+'"]');
 await until(()=>ruleRow('local:tissue')&&ruleRow('local:lymph_node'));
 assert(!ruleRow('local:lymph_node').querySelector('input[type=checkbox]').checked,'The rules show lymph_node unticked, as in Layers');
 if(!ruleRow('local:tissue').querySelector('input[type=checkbox]').checked){ruleRow('local:tissue').querySelector('input[type=checkbox]').click();
   await until(async()=>(await (await fetch(studioApi('/annotation_export'))).json()).settings.rules['local:tissue']?.export===true);}
 assert(true,'Ticking Export saves the rule for every image');
 await until(()=>row('tissue').querySelector('[aria-label="Export tissue"]').checked);
 assert(true,'Layers shows the rule as its Export tick');
 let exported=null;const originalFetch=window.fetch;
 window.fetch=async(url,options)=>{const response=await originalFetch(url,options);if(String(url).endsWith('/annotation_export')&&options?.method==='POST'&&!String(options.body).includes('"action"'))exported=await response.clone().json();return response;};
 const click=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(!this.download)click.call(this);};
 button(document.querySelector('.studio-runbar'),'This image').click();
 await until(()=>exported);
 assert(exported.names&&exported.names.join()==='tissue','Export writes the ticked annotations only: '+JSON.stringify(exported.names||exported.error));
 assert(exported.geojson.features.find(f=>f.properties.name==='tissue').properties.classification.color.join()==='0,255,0','Each name keeps its colour in the file');
 assert(/as markers\.geojson/.test(document.querySelector('.studio-runbar').textContent),'The file is named after its image');
 // Export name and colour in the rules change the file only.
 const exportName=ruleRow('local:tissue').querySelector('input[type=text]');exportName.value='Tissue_new';exportName.dispatchEvent(new Event('change'));
 await until(async()=>(await (await fetch(studioApi('/annotation_export'))).json()).settings.rules['local:tissue']?.name==='Tissue_new');
 exported=null;button(document.querySelector('.studio-runbar'),'This image').click();await until(()=>exported);
 assert(exported.names.join()==='Tissue_new'&&row('tissue'),'The export name changes the file, not the polygons');
 // File names: <prefix><image name><suffix>.geojson.
 const affix=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
 affix('File name prefix','QC_');await until(()=>/e\.g\. QC_markers\.geojson/.test(tools().textContent));
 affix('File name suffix','_v2');await until(()=>/e\.g\. QC_markers_v2\.geojson/.test(tools().textContent));
 exported=null;button(document.querySelector('.studio-runbar'),'This image').click();await until(()=>exported);
 assert(exported.filename==='QC_markers_v2.geojson'&&/stage0_tissue_annotation\/QC_markers_v2\.geojson$/.test(exported.saved_as),'Prefix and suffix name the exported file, in the output folder itself: '+exported.saved_as);
 // A name changed in Layers shows in the rules without pressing anything.
 nameRegions(tissue(), new Set(tissue().features.filter(f=>f.properties.name==='lymph_node').map(f=>f.properties.id)), 'node');
 await until(()=>ruleRow('local:node')&&!ruleRow('local:lymph_node'));
 assert(true,'A rename in Layers shows in Export at once');
 window.fetch=originalFetch;HTMLAnchorElement.prototype.click=click;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#panel')?.textContent||'').slice(0,600)));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
