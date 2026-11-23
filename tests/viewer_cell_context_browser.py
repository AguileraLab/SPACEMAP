"""Shared Cell data, density scope and reliable Flow/Data/cluster switching."""
from viewer_threshold_browser import main

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));},control=n=>document.querySelector('[aria-label="'+n+'"]');
const read=async url=>{const response=await fetch(url),result=await response.json();if(!response.ok)throw Error(result.error);return result;};
(async()=>{try{
 const s=window.__state;await until(()=>s.cellsReady&&s.layers.some(l=>l.combinedTable));await s.phenotypingLink;
 const cells=s.layers.find(l=>l.type==='cells'&&l.combinedTable);s.layerGroups.cells=true;buildPanel();
 check(document.querySelectorAll('select[aria-label="Cell data"]').length===1,'There is one shared Cell data selector');
 check(control('Cell data').selectedOptions[0].textContent==='Phenotyping results','The shared source has the same plain name used by Phenotyping');
 const sourceBox=document.querySelector('[data-shared-cell-data]');
 check(!sourceBox.querySelector('p,details,small')&&!!sourceBox.querySelector('.help[data-tip]'),'Source explanations use one info tooltip instead of visible paragraphs');
 const validSource=cells.table;cells.table='missing-source';buildPanel();
 check(control('Cell data').value===''&&document.querySelector('[data-shared-cell-data] button').disabled,'Missing source shows a placeholder and disables Open data instead of displaying a false choice');
 cells.table=validSource;buildPanel();
 check(!document.querySelector('[data-shared-cell-data] button').disabled,'Available Cell data can be opened directly');
 await until(()=>document.querySelector('.cell-filter-count')?.textContent==='40 of 40 cells shown');check(true,'Unfiltered cells have an accurate count');
 cells.tables.push({id:'t0',name:"This segmentation's cells",file:'cells.csv'});
 await addLayerToView({id:'restored-map',type:'density',name:'Density map 1',token:cells.token,image_token:cells.image_token,
   tables:[{id:'expired',name:'Combined phenotyping table'}],table:'expired',fcol:'',fvals:[],radius:30,cmap:'inferno',visible:true,opacity:.7,added:true});
 const map=s.layers.find(l=>l.id==='restored-map');cellRowsOpen()[map.id]=true;buildPanel();
 const density=()=>read(api('/cells/'+map.token+'/density/legend?'+densityQuery(map)));
 await until(()=>document.querySelector('[data-layer-id="restored-map"] .gradient'));
 check(map.table===cells.table&&(await density()).cells===40,'A restored density map repairs its old data reference and computes all image cells');
 check(!control('Density cell data'),'Density maps have no competing cell-data selector');
 showDataTab();await dataChoose('combined');await until(()=>dataView.queryReady);
 check(!document.querySelector('select[aria-label="Data source"]'),'The Data tab displays the shared source and links back to its single control');
 change(control('Cell data'),cells.id+'|t0');await until(()=>cells.table==='t0'&&map.table==='t0'&&dataView.source?.table==='t0'&&dataView.queryReady);
 check(document.querySelector('[aria-label="Data source"]').textContent==='Segmentation measurements','One source change updates Data, cell overlay and density map');
 check((await density()).cells===40,'Density remains valid after changing the shared source');
 change(control('Cell data'),cells.id+'|'+cells.combinedTable);await until(()=>dataView.source?.combined&&dataView.queryReady);
 change(control('Density cells included'),'viewer');await until(()=>map.cellScope==='viewer');
 await showViewerCellGroups('phenotype_training',['A'],['A','B']);
 await until(()=>dataView.groupPick?.values[0]==='A'&&dataView.queryReady);
 check((await density()).cells===20,'Density can follow the selected phenotype cluster');
 const alpha=async label=>{const response=await fetch(tileUrl(cells,0,0,0,Date.now()));if(!response.ok)throw Error(await response.text());const image=await createImageBitmap(await response.blob());const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);image.close();const i=label-1;return ctx.getImageData(40*(i%8)+20,40*Math.floor(i/8)+20,1,1).data[3];};
 const firstAlpha=[await alpha(1),await alpha(21)];check(firstAlpha[0]>0&&firstAlpha[1]===0,'The rendered overlay contains only the clicked cluster: '+JSON.stringify({firstAlpha,mode:cells.mode,filters:cells.filters,hidden:[...cells.hidden]}));
 window.launchFlowWorkspace(document.createElement('div'));
 const endpoint=api('/studio/browser-check/flow_gating'),meta=await read(endpoint+'?metadata=1');
 const selection=await(await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'select',revision:meta.workspace.revision,populations:['root'],rule:{column:'phenotype_training',values:['B']}})})).json();
 const bus=new BroadcastChannel('spacemap-flow-browser-check');bus.postMessage({sender:'test-flow',type:'selection',selection,mode:'filter',table:dataView.combined.path});
 await until(()=>cells.flowSelection===selection.token&&dataView.flowSelection===selection.token);
 check((await density()).cells===20,'Density follows a Flow population using the same shared cell data');
 await showViewerCellGroups('phenotype_training',['A'],['A','B']);await until(()=>!cells.flowSelection&&!dataView.flowSelection&&dataView.queryReady);
 check(await alpha(1)>0&&await alpha(21)===0,'A cluster click replaces an earlier Flow mask in the actual overlay');
 check(document.querySelector('[data-cell-view-status]').textContent.includes('Phenotyping clusters'),'The viewer states that Phenotyping clusters control the current cells');
 await Promise.all([showViewerCellGroups('phenotype_training',['A'],['A','B']),showViewerCellGroups('phenotype_training',['B'],['A','B'])]);
 await until(()=>dataView.groupPick?.values[0]==='B'&&dataView.queryReady);
 check(await alpha(1)===0&&await alpha(21)>0,'Rapid cluster clicks keep the latest selection');
 dataView.groupPick=null;dataView.onlySelected=false;dataView.filters=[{col:'M0',lo:1,hi:91}];dataFiltersChanged();await until(()=>dataView.queryReady);
 check((await density()).cells===10&&cells.viewSource==='Data filters','Data filters replace the cluster selection and update density');
 change(control('Density cells included'),'all');check((await density()).cells===40,'All cells in this image ignores viewer filters explicitly');
 control('Show all image cells').click();await until(()=>!cells.filters.length&&!cells.hidden.size&&dataView.queryReady);check(await alpha(1)>0&&await alpha(21)>0,'Show all cells clears selection conflicts');
 check(control('Cell data').value===cells.id+'|'+cells.combinedTable,'Clearing filters preserves the shared Cell data choice');
 // Delayed responses must not restore an old source or selection.
 await showViewerCellGroups('phenotype_training',['A'],['A','B']);await until(()=>dataView.queryReady);
 await inspectReviewedCell(cells,1);await until(()=>s.inspectedCell?.result.profile?.status==='ready');
 const originalFetch=window.fetch;let release,held=false;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);if(String(url).includes('/columns?')&&!held){held=true;await new Promise(r=>release=r);}return response;};
 const pending=showViewerCellGroups('phenotype_training',['A'],['A','B']);await until(()=>held);
 change(control('Cell data'),cells.id+'|t0');await until(()=>cells.table==='t0'&&dataView.queryReady);release();await pending;window.fetch=originalFetch;
 check(cells.table==='t0'&&map.table==='t0'&&!cells.filters.length,'An older cluster response cannot undo a newer Cell data choice');
 check(!dataView.groupPick&&!dataView.onlySelected&&!s.inspectedCell&&dataView.result.total===40,'Changing Cell data clears selections and inspection from the old source');
 const link=s.phenotypingLink;let releaseLink;s.phenotypingLink=new Promise(r=>releaseLink=r);
 const sequence=s.groupPickSequence;bus.postMessage({sender:'test-flow',type:'selection',selection,mode:'filter',table:dataView.combined.path});await until(()=>s.groupPickSequence>sequence);
 const newer=showViewerCellGroups('phenotype_training',['B'],['A','B']);releaseLink();await newer;s.phenotypingLink=link;
 await until(()=>dataView.groupPick?.values[0]==='B'&&dataView.queryReady);
 check(!cells.flowSelection&&await alpha(1)===0&&await alpha(21)>0,'A delayed Flow selection cannot override a newer cluster click');
 const saved=captureView();check(saved,'The shared setup remains saveable');
 bus.close();fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+dataView.status?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(DRIVER,studio=True,timeout=240,screenshot=False,browser_host='cell-context.test')
