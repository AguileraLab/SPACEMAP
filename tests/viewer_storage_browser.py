"""Jobs storage preview and cleanup in a disposable Firefox project."""
import json
import os
from viewer_threshold_browser import main


def setup(root):
    os.environ['SPACEMAP_CACHE_DIR']=str(root/'user_cache')
    os.environ['NUMBA_CACHE_DIR']=str(root/'user_cache/numba_cache')
    os.environ['CELLPOSE_LOCAL_MODELS_PATH']=str(root/'user_cache/cellpose_cache')
    cache=root/'outputs/viewer_cache/cell_labels/image.cell_labels.tif';cache.parent.mkdir(parents=True);cache.write_bytes(b'cached outlines')
    summaries=root/'user_cache/studio-summaries-v1';summaries.mkdir(parents=True);(summaries/'example.json').write_text('{}')
    table=root/'saved_cells.csv';table.write_text('id,phenotype_leiden_clusters\n1,Cluster 0\n')
    folder=root/'outputs/studio/jobs'/('f'*32);folder.mkdir(parents=True)
    (folder/'results.parquet').write_bytes(b'working data already summarized')
    (folder/'request.json').write_text(json.dumps({'kind':'leiden','settings':{}}))
    profile={'version':2,'values':[[0]],'summaries':{k:{'rows':['Cluster 0'],'columns':['M0'],'values':[[0]],'raw':[[1]],'top_markers':{}} for k in ('mean','median')}}
    (folder/'status.json').write_text(json.dumps({'id':'f'*32,'kind':'leiden','status':'done','created':'2026-10-23T00:00:00Z','progress':1,
        'result':{'table':str(table),'column':'phenotype_leiden_clusters','profile':profile}}))


DRIVER=r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(ok,msg)=>{if(!ok)throw Error(msg);checks++;fetch('/result?progress='+encodeURIComponent(msg));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
setTimeout(async()=>{try{
 await until(()=>window.__state?.imagesReady&&document.querySelector('[data-tab=jobs]'));
 document.querySelector('[data-tab=jobs]').click();
 await until(()=>document.querySelector('[data-temporary-files]'));
 const panel=document.querySelector('[data-temporary-files]');
 assert(!panel.open&&!performance.getEntriesByType('resource').some(r=>r.name.endsWith('/storage')),'Storage scanning waits until requested');
 assert(!button(panel,'Clean selected').getClientRects().length,'Cleanup actions stay hidden while collapsed');
 const caches=document.querySelector('[data-performance-caches]');
 assert(!caches.open&&!performance.getEntriesByType('resource').some(r=>r.name.includes('storage?caches=1')),'Performance-cache scans also wait for expansion');
 caches.open=true;await until(()=>caches.querySelector('[data-cache-group=cell_labels]'));
 assert(caches.querySelector('[role=status]').textContent.startsWith('Kept'),'Performance caches are explicitly retained');
 assert(!caches.querySelector('input[type=checkbox]'),'Performance caches are excluded from cleanup selection');
 const outlines=caches.querySelector('[data-cache-group=cell_labels]');outlines.open=true;
 await until(()=>outlines.querySelector('.storage-file'));
 assert(outlines.querySelector('summary').title.includes('/outputs/viewer_cache/cell_labels')&&outlines.querySelector('summary').title.includes('Rebuilding'),'Cache groups show full path, purpose and impact');
 const cacheFile=outlines.querySelector('.storage-file');
 assert(cacheFile.title.includes('/image.cell_labels.tif')&&cacheFile.textContent.includes('15 B'),'Separate cache files have paths and measured sizes');
 cacheFile.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:500,clientY:200}));
 assert(document.querySelector('.lists-context [role=menuitem]').textContent==='Copy path','Cache files offer Copy path');document.querySelector('.lists-context').remove();
 assert(caches.querySelector('[data-cache-group=summaries] summary').title.includes('Shared'),'Shared caches identify their cross-project scope');
 assert(outlines.querySelector('summary').getBoundingClientRect().left>caches.querySelector('summary').getBoundingClientRect().left,'Cache components are indented beneath their parent');
 caches.open=false;
 panel.open=true;await until(()=>panel.querySelector('input[type=checkbox]'));
 assert(panel.querySelectorAll('input').length===3,'Cleanup categories are shown with sizes');
 assert(button(panel,'Clean selected').disabled,'Cleanup requires an explicit category choice');
 const picked=panel.querySelector('[aria-label="Clean Saved Leiden working tables"]');
 const group=panel.querySelector('[data-storage-group=leiden]');group.open=true;await until(()=>group.querySelector('.storage-file'));
 const file=group.querySelector('.storage-file');
 assert(file.title.includes('/results.parquet')&&file.title.includes('heatmaps are preserved'),'Every cleanup item explains its purpose, impact and full path on hover');
 file.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:500,clientY:200}));
 assert(document.querySelector('.lists-context [role=menuitem]').textContent==='Copy path','Cleanup items offer Copy path on right-click');document.querySelector('.lists-context').remove();
 const itemPick=file.querySelector('input');itemPick.click();
 assert(picked.checked,'Individual file selection updates its group checkbox');itemPick.click();
 assert(!picked.disabled,'A completed job with saved outputs can be cleaned');picked.click();
 assert(!button(panel,'Clean selected').disabled,'Selecting a category enables cleanup');
 button(panel,'Clean selected').click();await until(()=>panel.querySelector('[role=status]').textContent.includes('freed'));
 assert(panel.querySelector('[aria-label="Clean Saved Leiden working tables"]').disabled,'The cleaned working file is no longer listed');
 const result=await fetch(api('/studio/browser-check/jobs/'+ 'f'.repeat(32))).then(r=>r.json());
 assert(result.job.status==='done'&&result.job.result.profile.summaries.mean,'Job metadata and both heatmap summaries survive cleanup');
 assert(!button(panel,'Clean selected').getClientRects().length||button(panel,'Clean selected').disabled,'Cleanup cannot reuse the previous selection');
 const kept=await fetch(api('/studio/browser-check/storage?caches=1')).then(r=>r.json());
 assert(kept.groups.find(g=>g.id==='cell_labels').bytes===15,'Cleaning temporary files leaves performance caches intact');
 document.querySelector('#app').style.setProperty('--studio-right','320px');caches.open=true;await delay(100);
 assert(caches.scrollWidth<=caches.clientWidth+2&&panel.scrollWidth<=panel.clientWidth+2,'Storage lists and controls fit a narrow dock');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(e){fetch('/result?error='+encodeURIComponent(e.message+'\n'+e.stack));}},100);
</script>
"""

if __name__=='__main__':main(DRIVER,studio=True,setup=setup,timeout=160)
