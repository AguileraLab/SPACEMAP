"""Review without required metadata; exact ranges linked to Data and image pixels."""
import viewer_threshold_browser as harness

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<3000;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(c,m)=>{if(!c)throw Error(m);checks++;};
const btn=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const change=(el,value)=>{el.value=String(value);el.dispatchEvent(new Event('change'));};
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 // Hold MR-ML loading, switch methods, then release it after the new footer exists.
 const originalFetch=window.fetch;let releaseMethod,methodWaiting=false;
 window.fetch=async(url,options)=>{const response=await originalFetch(url,options);
   if(String(url).endsWith('/mrml_settings')&&(!options||!options.method||options.method==='GET')){
     methodWaiting=true;await new Promise(r=>releaseMethod=r);
   }return response;};
 change(document.querySelector('[aria-label="Phenotyping method"]'),'mr_ml');
 await until(()=>methodWaiting);
 change(document.querySelector('[aria-label="Phenotyping method"]'),'subphenotyping');
 const tools=()=>document.querySelector('#studio-tools');
 await until(()=>tools().querySelector('[aria-label="Generate Marker review"]'));
 await until(()=>btn(document.querySelector('.studio-runbar'),'Run sub-phenotyping'));
 releaseMethod();window.fetch=originalFetch;await delay(300);
 assert(!btn(document.querySelector('.studio-runbar'),'Run MR-ML')&&btn(document.querySelector('.studio-runbar'),'Run sub-phenotyping'), 'Late MR-ML load cannot replace the current method footer');
 assert(!btn(document.querySelector('.studio-runbar'),'Find markers'),'Only one marker decision workflow');
 const box=tools().querySelector('[data-marker-figures]');box.open=true;
 assert(tools().querySelector('[aria-label="Figure patient column"]').value==='','No grouping required or inferred from metadata');
 assert([...tools().querySelector('[aria-label="Figure patient column"]').options].some(o=>o.value==='specimen_code'),'Custom metadata available');
 change(tools().querySelector('[aria-label="Minimum cells per sample"]'),2);
 btn(box,'None').click();tools().querySelector('[aria-label="Generate Marker review"]').click();
 btn(box,'Generate selected').click();
 await until(()=>document.querySelector('[data-marker-review] .studio-review-list table'));
 const gallery=()=>document.querySelector('.studio-marker-gallery'),review=()=>gallery().querySelector('[data-marker-review]');
 assert(review().textContent.includes('Sample comparison · None'),'Pooled review works without metadata');
 assert(review().querySelectorAll('[aria-label^="Review use "]').length===2,'Every selected marker can be reviewed');
 assert(!review().querySelector('[aria-label="Review use M0"]').checked,'Candidates do not change clustering inputs automatically');
 btn(review(),'M0').click();
 assert(review().querySelector('svg rect'),'Interactive histogram renders');
 review().querySelector('svg rect').dispatchEvent(new MouseEvent('click'));
 assert(review().querySelector('[aria-label="Review range from"]').value!=='','Clicking a bin fills range');
 // Stored marker values are 1,11,...191 in phenotype A. Pick labels 3 through 6.
 change(review().querySelector('[aria-label="Review range from"]'),21);
 change(review().querySelector('[aria-label="Review range to"]'),51);
 const imageId=s.spec.id, zoom=s.viewer.viewport.getZoom();
 btn(review(),'Show cells').click();
 await until(()=>s.layers.some(l=>l.flowSelection)||review().textContent.includes('Error'));
 const layer=s.layers.find(l=>l.flowSelection);
 assert(layer.flowPopulation.count===4&&layer.flowPopulation.by_image.roi===4,'Range gives exactly four cells');
 assert(s.spec.id===imageId&&Math.abs(s.viewer.viewport.getZoom()-zoom)<1e-5,'Current image and zoom remain unchanged');
 await until(()=>window.__reviewData.result?.matched===4);
 assert(window.__reviewData.flowSelection===layer.flowSelection,'Data and image share identical membership');
 assert(window.__reviewData.result.matched===4,'Data shows exactly four matching rows');
 const tile=s.viewer.world.getItemAt(s.layers.indexOf(layer));
 const resources=performance.getEntriesByType('resource').filter(r=>r.name.includes('flow_selection='));
 await until(()=>performance.getEntriesByType('resource').some(r=>r.name.includes(layer.flowSelection)&&r.name.includes('/tile/')));
 const url=performance.getEntriesByType('resource').filter(r=>r.name.includes(layer.flowSelection)&&r.name.includes('/tile/')).at(-1).name;
 const img=new Image();img.src=url;await img.decode();const canvas=document.createElement('canvas');canvas.width=img.width;canvas.height=img.height;const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);
 assert(ctx.getImageData(0,0,img.width,img.height).data.some((v,i)=>i%4===3&&v>0),'Linked cell tile contains visible pixels');
 // A later explicit reset defeats an earlier slow review selection.
 const realFetch=window.fetch;let release,waiting;
 window.fetch=async(url,options)=>{const response=await realFetch(url,options);if(String(url).endsWith('/marker_review')&&JSON.parse(options.body).action==='select'){
   waiting=true;await new Promise(r=>release=r);}return response;};
 btn(review(),'Show cells').click();await until(()=>waiting);window.__reviewReset(layer);release();
 await until(()=>!btn(review(),'Show cells').disabled);window.fetch=realFetch;
 assert(!layer.flowSelection,'A later Show all cells wins over delayed range selection');
 review().querySelector('[aria-label="Review use M0"]').click();
 btn(review(),'Use selected for clustering').click();
 await until(()=>review().textContent.includes('Saved 1 markers'));
 assert(tools().textContent.includes('A: 1 of 2 markers for clustering'),'Explicit review choices update existing marker settings');
 const png=gallery().querySelector('a[download]');const file=await fetch(png.href);assert(file.ok&&(await file.blob()).size>1000,'Composite PNG is downloadable');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(error+'\n'+error.stack+'\n'+document.querySelector('[data-marker-review]')?.textContent));}},100);
</script>
'''


def setup(root):
    import json
    import pandas as pd
    from spacemap import dataio
    f=pd.read_csv(root/'cells.csv')
    f['specimen_code']=['S1']*10+['S2']*30
    f.to_csv(root/'cells.csv', index=False)
    dataio.ensure_fast_copy(root/'cells.csv', force=True)
    (root/'outputs/phenotyping/cache/subphenotyping_params.json').write_text(json.dumps({
        'parent_col':'phenotype_ml_consensus','parent_values':['A'],'markers':['M0','M1'],
        'transform':'stored','normalization':'none'}))


if __name__=='__main__':
    harness.EXPOSE += '<script>window.__reviewData=dataView;window.__reviewReset=resetCellView;</script>'
    harness.main(DRIVER,studio=True,timeout=260,setup=setup)
