"""Browser-check Spatial analysis (Density with its outline, By distance, a density map, Triads drawn on the image)
and Visualization (Proportions, Heatmap, Method agreement, UMAP), each linked to the open image's cells.

Usage: PYTHONPATH=src:tests python tests/viewer_spatial_browser.py
"""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),bar=()=>document.querySelector('.studio-runbar');
 const row=label=>Array.from(bar().querySelectorAll('.row')).find(r=>r.querySelector('.lbl')?.textContent===label);
 const tab=name=>Array.from(tools().querySelectorAll('.studio-stage-tabs button')).find(b=>b.textContent===name).click();
 const figure=key=>document.querySelector('#studio-figures [data-figure="'+key+'"]');
 const cells=()=>s.layers.find(l=>l.type==='cells');
 await until(()=>s.imagesReady&&s.cellsReady&&document.querySelector('[data-step=tab8]'));
 document.querySelector('[data-step=tab8]').click();
 await until(()=>tools().querySelector('.studio-stage-tabs'));
 assert(['Density','Triads','Feature screening'].every(n=>Array.from(tools().querySelectorAll('.studio-stage-tabs button')).some(b=>b.textContent===n)),'Spatial analysis has Density, Triads and Feature screening');
 tab('Density');
 await until(()=>tools().querySelector('[aria-label="Density group column"]')&&row('Density:'));
 const pick=(label,value)=>{const e=tools().querySelector('[aria-label="'+label+'"]');e.value=value;e.dispatchEvent(new Event('change'));};
 pick('Density group column','phenotype_ml_consensus');
 button(row('Density:'),'This image').click();
 await until(()=>figure('density')?.querySelector('.row[data-row="roi"]'));
 figure('density').querySelector('.row[data-row="roi"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>s.previews?.['density-outline']?.outlines?.length);
 assert(/tissue area/.test(bar().textContent),'Density: cells per mm² per image; a click draws the tissue outline used on the image');
 button(row('By distance:'),'This image').click();
 await until(()=>figure('distance')?.querySelector('.row[data-row="A"]'));
 figure('distance').querySelector('.row[data-row="A"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>(cells().filters||[]).some(f=>f.col==='phenotype_ml_consensus'&&f.values?.includes('A')));
 assert(true,'By distance: a group clicked in the heatmap is shown alone on the image');
 await until(()=>tools().querySelector('[aria-label="Map group"]')?.options.length);
 button(tools(),'Show on the open image').click();
 await until(()=>s.layers.some(l=>l.type==='density'));
 assert(true,'A density map of the group is added to the image');
 tab('Triads');
 await until(()=>tools().querySelector('[aria-label="Phenotype C"]')?.options.length);
 const corner=(c,v)=>{const e=tools().querySelector('[aria-label="Phenotype '+c+'"]');e.value=v;e.dispatchEvent(new Event('change'));};
 corner('A','A');corner('B','A');corner('C','B');
 const shuffles=tools().querySelector('[aria-label="Triad shuffles"]');shuffles.value='20';
 const near=tools().querySelector('[aria-label="Triad distance up to µm"]');near.value='40';
 button(row('Find triads:'),'This image').click();
 await until(()=>Array.from(tools().querySelectorAll('button')).some(b=>b.textContent==='roi'&&b.title==='Open with its triangles'));
 Array.from(tools().querySelectorAll('button')).find(b=>b.textContent==='roi'&&b.title==='Open with its triangles').click();
 await until(()=>s.previews?.triads?.outlines?.length&&/triangles drawn/.test(bar().textContent));
 await until(()=>s.layers.some(l=>l.type==='density'&&/::/.test(l.name||'')));
 assert(true,'Triads: counted, tested, and each image’s triangles drawn on it');
 // Visualization
 document.querySelector('[data-step=tab9]').click();
 await until(()=>tools().querySelector('.studio-stage-tabs'));
 tab('Proportions');await until(()=>row('Proportions:'));
 button(row('Proportions:'),'Draw').click();
 await until(()=>figure('viz-proportions')?.querySelector('.spatial-segment'));
 figure('viz-proportions').querySelector('.spatial-segment[title^="B "]').click();
 await until(()=>cells().color_by==='phenotype_ml_consensus'&&cells().hidden.has('A')&&!cells().hidden.has('B'));
 assert(true,'Proportions: a bar segment shows that group’s cells on the image');
 tab('Heatmap');await until(()=>row('Heatmap:'));
 button(row('Heatmap:'),'Draw').click();
 await until(()=>figure('viz-heatmap')?.querySelector('.row[data-row="A"]'));
 figure('viz-heatmap').querySelector('.row[data-row="A"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>cells().hidden.has('B')&&!cells().hidden.has('A'));
 assert(true,'Heatmap: a group clicked shows its cells');
 tab('Method agreement');await until(()=>row('Agreement:'));
 button(row('Agreement:'),'Compare').click();
 await until(()=>/accuracy 100.0%/.test(tools().textContent)&&figure('viz-agreement'));
 assert(true,'Method agreement: accuracy, F1 per phenotype and the confusion table');
 tab('UMAP');await until(()=>row('UMAP:'));
 button(row('UMAP:'),'Make').click();
 await until(()=>figure('viz-umap')?.querySelector('canvas.spatial-umap'),);
 const canvas=figure('viz-umap').querySelector('canvas'),r=canvas.getBoundingClientRect();
 const ev=(type,x,y)=>canvas.dispatchEvent(new PointerEvent(type,{bubbles:true,clientX:r.left+x*r.width,clientY:r.top+y*r.height,pointerId:1}));
 ev('pointerdown',0,0);ev('pointermove',1,1);ev('pointerup',1,1);
 await until(()=>/cells selected/.test(figure('viz-umap').textContent)&&(cells().filters||[]).some(f=>f.col==='label'));
 assert(true,'UMAP: a box of points shows those cells on the image');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('.studio-runbar')?.textContent));}},100);
</script>
"""


def setup(root):
    """The harness table with a distance from an annotation (Cell tagging's column)."""
    import pandas as pd

    from spacemap import dataio

    table = root / "cells.csv"
    frame = pd.read_csv(table)
    frame["tumor_distance"] = frame["x_centroid"] - 160.0
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=420, setup=setup)
