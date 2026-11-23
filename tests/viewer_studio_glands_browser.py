"""Real browser: configure, run, resume, plot, sort, and select gland cells."""
import json
import pandas as pd
from spacemap import dataio
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    from spacemap_gui.studio import gland_clonality as g
    from spacemap_gui.studio import phenotyping_inputs
    images = root / 'images'; images.mkdir()
    (images / 'roi.ome.tiff').symlink_to(root / 'roi.ome.tiff')
    phenotyping_inputs.save_folders(root, {'images':str(images)})
    f = pd.read_csv(root / "cells.csv")
    index = np.arange(len(f))
    inside = index % 8 < 6
    f["tumor_roi"] = np.where(inside, index//8+1, 0)
    f["tumor_bed_id"] = np.where(inside, 0, index//8+1)
    f["tumor_bed"] = (~inside).astype(int)
    f["phenotype_sub"] = np.where(inside, np.where(index%8 < np.array([5, 2, 4, 1, 3])[index//8], "A", "B"), None)
    f["phenotype_ml_mr_refined"] = np.where(inside, "Epithelial", np.where(index%8-6 < index//8%3, "Fibroblast", "Endothelial"))
    f["patient"] = "p1"; f["response"] = "Yes"; f["timepoint"] = "pre"
    f.to_csv(root / "cells.csv", index=False);dataio.ensure_fast_copy(root / "cells.csv", force=True)
    c = g.defaults(list(f)); c.update(min_gland=4, min_bed=1, interior_k=2, exterior_k=1,
        interior_classes=["A", "B"], exterior_classes=["Fibroblast", "Endothelial"], group_a="Yes", group_b="No")
    previous = g.settings(root, root / "cells.csv")
    g.settings(root, root / "cells.csv", c, previous["revision"])
    outlines = root / "outputs/segmentation/roi.ome_cell_boundaries.geojson"
    outlines.parent.mkdir(parents=True, exist_ok=True);outlines.write_text((root / "cells.geojson").read_text())


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async fn=>{for(let i=0;i<2400;i++){if(await fn())return;await delay(50);}throw Error('Timed out '+fn+'; '+document.querySelector('#message')?.textContent);};
let checks=0;const assert=(ok,message)=>{if(!ok)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(root,label)=>[...root.querySelectorAll('button')].find(b=>b.textContent===label);
setTimeout(async()=>{try{
 const state=window.__state,tools=()=>document.querySelector('#studio-tools'),base='/spacemap-viewer/api/studio/browser-check';
 const input=label=>tools().querySelector('[aria-label="'+label+'"]');
 const request=async(action,body={})=>{const r=await fetch(base+'/spatial/gland_'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const out=await r.json();if(!r.ok)throw Error(out.error);return out;};
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 const imageToken=state.layers.find(l=>l.type==='image').token;
 document.querySelector('[data-step=tab8]').click();await until(()=>button(tools(),'Gland types'));button(tools(),'Gland types').click();
 await until(()=>input('Interior classes')&&input('Interior classes').querySelector('summary').textContent==='2 selected');
 assert(input('Gland types (K)').value==='2','Shared saved configuration loaded with plain K labels');
 assert(input('Niche types (K)').closest('.field').textContent.includes('?'),'Niche K has individual help');
 assert(state.layers.find(l=>l.type==='image').token===imageToken,'Entering Gland types preserves the current viewer image');
 input('Interior classes').open=true;const search=input('Find Interior classes');search.value='A';search.dispatchEvent(new Event('input'));
 assert([...input('Interior classes').querySelectorAll('.threshold-checks label')].filter(r=>!r.hidden).length===1,'Class picker is searchable');
 input('Interior classes').open=false;
 await until(()=>input('Pick roi'));input('Pick roi').click();
 button(document.querySelector('.studio-runbar'),'Selected').click();
 await until(()=>document.querySelector('.gland-done'));
 let jobs=(await(await fetch(base+'/jobs')).json()).jobs,job=jobs.find(j=>j.kind==='gland_types');
 assert(job.status==='done','Gland background job finished');
 await until(()=>tools().querySelector('tr[data-roi="roi"]')?.textContent.includes('5 typed glands'));assert(true,'Image list shows the completed gland result');
 assert(job.result.glands===5&&job.result.beds===5,'Correct gland and bed counts');
 await until(()=>document.querySelectorAll('.gland-profiles input[type=checkbox]').length>=3);
 for(const kind of ['interior','exterior']){
   const panel=document.querySelector('[data-figure="gland-'+kind+'"]');
   panel.querySelector('.gland-profiles input[type=checkbox]').click();button(panel,'Flag for review').click();
   await until(async()=>(await(await fetch(base+'/review')).json()).items.some(i=>i.kind===(kind==='interior'?'gland_type':'niche_type')));
 }
 assert(true,'Both gland-type and niche-type heatmaps flag their selected cells');
 assert(job.result.columns.join()==='gland_clone,gland_bed_niche','Both linked categorical columns published');
 assert(state.layers.find(l=>l.type==='image').token===imageToken,'Background fitting keeps the image open');
 await until(()=>document.querySelector('.gland-profiles'));
 assert(document.querySelectorAll('.gland-profiles').length===2,'Both profile heatmaps appear beside the image');
 assert(document.querySelector('.gland-profiles').textContent.includes('%'),'Heatmap text gives measured proportions');
 button(tools(),'Fit quality').click();await until(()=>document.querySelector('.gland-silhouette-chart'));
 assert(document.querySelectorAll('.gland-silhouette-chart circle[data-score]').length===5,'Fixed-K fit quality shows one dot for each scored gland');
 assert(document.querySelector('.gland-silhouette-chart').getBoundingClientRect().width>=620,'Narrow docks scroll the score chart instead of shrinking its labels');
 assert(document.querySelector('[data-figure="gland-exterior"] .gland-silhouette-empty').textContent.includes('K=1'),'One-type niches explain unavailable silhouette rather than reporting zero');
 assert(state.layers.find(l=>l.type==='image').token===imageToken,'Fit quality does not switch the viewer image');
 button(tools(),'Profiles').click();await until(()=>document.querySelectorAll('.gland-profiles').length===2);
 const typeButtons=()=>[...document.querySelectorAll('[data-gland-kind="interior"]')];
 const viewWidth=state.viewer.viewport.getBounds(true).width;
 typeButtons()[0].click();
 await until(()=>document.querySelector('[data-figure="gland-interior"] .gland-view-status')?.textContent.includes('cells in'));
 const coloured=state.layers.find(l=>l.type==='cells'&&l.flowSelection);
 assert(coloured.color_by==='gland_clone'&&coloured.mode==='both','Clicking a gland type colours assigned cells through the shared layer');
 assert(state.layers.find(l=>l.type==='image').token===imageToken,'Type click keeps the current image');
 await until(()=>state.viewer.viewport.getBounds(true).width<viewWidth*.9);
 assert(true,'Type click zooms to an example within the open image');
 assert(window.__glandData.flowSelection===coloured.flowSelection&&window.__glandData.result.matched===coloured.flowPopulation.count,'Type selection gives Data the same exact cells');
 await until(()=>performance.getEntriesByType('resource').some(r=>r.name.includes(coloured.flowSelection)&&r.name.includes('/tile/')));
 const tileUrl=performance.getEntriesByType('resource').filter(r=>r.name.includes(coloured.flowSelection)&&r.name.includes('/tile/')).at(-1).name;
 const tile=new Image();tile.src=tileUrl;await tile.decode();const pixels=document.createElement('canvas');pixels.width=tile.width;pixels.height=tile.height;
 const ctx=pixels.getContext('2d');ctx.drawImage(tile,0,0);const rgba=ctx.getImageData(0,0,tile.width,tile.height).data;
 assert(rgba.some((v,i)=>i%4===3&&v>0),'Selected gland overlay contains visible coloured pixels');
 const rgb=job.result.palette[typeButtons()[0].dataset.glandType].slice(1).match(/../g).map(v=>parseInt(v,16));
 assert(rgba.some((v,i)=>i%4===3&&v>0&&rgb.every((n,j)=>Math.abs(rgba[i-3+j]-n)<3)),'Viewer cell fills use the same gland type colour as the map legend');
 document.querySelector('[data-gland-kind="exterior"]').click();
 await until(()=>coloured.color_by==='gland_bed_niche'&&document.querySelector('[data-figure="gland-exterior"] .gland-view-status')?.textContent.includes('cells in'));
 assert(coloured.flowPopulation.count===10,'Niche click selects all assigned exterior cells in this image');
 const nativeFetch=window.fetch,previousToken=coloured.flowSelection,previousZoom=state.viewer.viewport.getZoom();
 window.fetch=async(url,options)=>String(url).endsWith('/gland_type_cells')?new Response(JSON.stringify({missing:true,message:'This type is not present in roi.'}),{headers:{'Content-Type':'application/json'}}):nativeFetch(url,options);
 typeButtons()[0].click();await until(()=>document.querySelector('[data-figure="gland-interior"] .gland-view-status').textContent.includes('not present'));
 window.fetch=nativeFetch;
 assert(coloured.flowSelection===previousToken&&coloured.color_by==='gland_bed_niche'&&state.layers.find(l=>l.type==='image').token===imageToken,'Absent type reports a message without changing image, colours or selected cells');
 assert(Math.abs(state.viewer.viewport.getZoom()-previousZoom)<1e-5,'Absent type preserves zoom');
 let release,waiting=false;
 window.fetch=async(url,options)=>{const response=await nativeFetch(url,options);if(String(url).endsWith('/gland_type_cells')){waiting=true;await new Promise(r=>release=r);}return response;};
 typeButtons()[0].click();await until(()=>waiting);window.__glandReset(coloured);release();window.fetch=nativeFetch;await delay(500);
 assert(!coloured.flowSelection,'A later Show all cells defeats a delayed type selection');
 button(tools(),'Explore').click();await until(()=>document.querySelector('[aria-label="Interior gland types"]'));
 assert(document.querySelector('[aria-label="Exterior niches"]'),'Both cell-position maps are rendered');
 const table=tools().querySelector('.gland-structures');assert(table.querySelectorAll('[data-gland]').length===5,'Clickable gland table is populated');
 table.querySelector('[data-gland] input[type=checkbox]').click();
 button(tools(),'Flag glands for review').click();button(tools(),'Flag niches for review').click();
 await until(async()=>{const items=(await(await fetch(base+'/review')).json()).items;return items.some(i=>i.kind==='gland')&&items.some(i=>i.kind==='niche');});
 assert(true,'Individual glands and their beds can also be flagged separately');
 button(table,'Cells ↓').click();assert(button(tools().querySelector('.gland-structures'),'Cells ↑'),'Column headers sort in either direction');
 const row=tools().querySelector('.gland-structures [data-gland]');button(row,'Interior').click();
 await until(()=>state.layers.find(l=>l.type==='cells')?.flowSelection);
 const layer=state.layers.find(l=>l.type==='cells');assert(layer.flowSelection,'Gland selection links to viewer and Data');
 const selected=await request('cells',{job_id:job.id,structure:row.dataset.gland,role:'gland'});
 assert(selected.bounds.length===4&&selected.image==='roi','Selection returns exact image bounds');
 assert(document.querySelector('#studio-tools'),'Studio remains responsive after selection');
 button(tools(),'Compare').click();await until(()=>tools().textContent.includes('A median'));
 assert(tools().textContent.includes('Higher in'),'Comparisons state direction by group name');
 button(tools(),'Density').click();await delay(200);button(tools(),'Gland types').click();
 await until(()=>input('Gland saved run')?.value===job.id);
 assert(true,'Saved result resumes after switching methods without refitting');
 const stats=await request('structures',{job_id:job.id});assert(stats.rows.length===5,'Saved structure table remains available');
 const wrong=await fetch(base+'/spatial/gland_download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:job.id,file:'../../cells.csv'})});
 assert(wrong.status===400,'Download endpoint refuses paths outside its allowed exports');
 const res=await fetch(base+'/spatial/gland_download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:job.id,file:'gland_structures.csv'})});
 assert(res.ok&&(await res.text()).includes('gland_type'),'CSV export is available');
 const before=job.result.source_after;
 input('Niche types (K)').value=4;input('Niche types (K)').dispatchEvent(new Event('change'));
 button(document.querySelector('.studio-runbar'),'Review optimal K').click();
 await until(async()=>{const list=(await(await fetch(base+'/jobs')).json()).jobs;const review=list.find(j=>j.kind==='gland_types'&&j.id!==job.id);return review?.status==='done';});
 const latest=(await(await fetch(base+'/jobs')).json()).jobs.find(j=>j.kind==='gland_types'&&j.id!==job.id);
 assert(latest.result.diagnose&&latest.result.columns.length===0,'Auto-K review publishes diagnostics without changing columns');
 assert(latest.result.source_after===before,'K review preserves previously saved cell results');
 await until(()=>input('Gland saved run').value===latest.id&&document.querySelectorAll('.gland-k-chart').length===2);
 assert(tools().querySelector('[data-view="K review"]').classList.contains('active'),'Review opens the score charts by default');
 assert(document.querySelector('[data-figure="gland-interior"] .gland-k-chart rect[data-k]')&&document.querySelector('[data-figure="gland-exterior"] .gland-k-chart rect[data-k]'),'Both right-dock score charts render bars');
 assert(input('Niche types (K)').value==='4','Review leaves the manual niche K unchanged');
 const details=await request('result',{job_id:latest.id}),review=details.k_reviews.exterior;
 assert(document.querySelector('[data-figure="gland-exterior"]').textContent.includes('Suggested K = '+review.suggested_k),'Chart states the suggested K');
 await until(()=>document.querySelectorAll('.gland-review-charts .gland-silhouette-chart').length===2);
 assert(document.querySelectorAll('.gland-review-charts .gland-silhouette-chart circle[data-score]').length===10,'Individual gland and bed distributions render beside K charts');
 const scores=await request('silhouette',{job_id:latest.id,kind:'interior'});
 assert(Math.abs(scores.mean-latest.result.models.interior.silhouette)<1e-10,'Individual scores average to the exact K-review score');
 const scoreCSV=await fetch(base+'/spatial/gland_download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:latest.id,file:'interior_silhouette.csv'})});
 assert(scoreCSV.ok&&(await scoreCSV.text()).includes('reference_structures'),'Individual scores export with sample scope');
 const scorePNG=await fetch(base+'/spatial/gland_download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:latest.id,file:'interior_silhouette.png'})});
 assert(scorePNG.ok&&(await scorePNG.blob()).size>1000,'Individual score figure exports as PNG');
 const png=await fetch(base+'/spatial/gland_download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:latest.id,file:'exterior_k_review.png'})});
 assert(png.ok&&png.headers.get('content-type').includes('image/png')&&(await png.blob()).size>1000,'K score chart exports as PNG');
 button(tools(),'Profiles').click();await until(()=>document.querySelectorAll('.gland-profiles').length===2);
 assert(!document.querySelector('.gland-k-chart'),'Profiles remain separately accessible');
 button(tools(),'K review').click();await until(()=>document.querySelector('.gland-k-chart'));
 input('Auto niche K').checked=true;input('Auto niche K').dispatchEvent(new Event('change'));
 button(tools(),'Use niche K='+review.suggested_k).click();
 assert(input('Niche types (K)').value===String(review.suggested_k)&&!input('Auto niche K').checked,'Use suggested K explicitly sets manual K and turns off Auto');
 const settings=await fetch(base+'/spatial/gland_settings').then(r=>r.json());
 assert(settings.values.exterior_k===4,'Using a recommendation only changes the draft until Save or Fit');
 fetch('/result?ok='+checks);
}catch(e){fetch('/result?error='+encodeURIComponent(e.message+'\n'+e.stack));}},100);
</script>
"""

if __name__ == "__main__":
    import viewer_threshold_browser as harness
    harness.EXPOSE += '<script>window.__glandData=dataView;window.__glandReset=resetCellView;</script>'
    main(DRIVER, studio=True, setup=setup, timeout=240, screenshot=True, capture="/tmp/spacemap-gland-browser.png")
