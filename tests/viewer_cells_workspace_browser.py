"""Approved Cells layout and explicit imported-source switching, on disposable cells."""
from viewer_threshold_browser import main

DRIVER = r'''<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<800;i++){if(await f())return;await wait(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const control=n=>document.querySelector('[aria-label="'+n+'"]');
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
(async()=>{try{
 const s=window.__state;await until(()=>s.cellsReady&&s.layers.some(l=>l.combinedTable));await s.phenotypingLink;
 const cells=activeCellLayer(),table=cells.table,token=cells.token,image=s.layers.find(l=>l.type==='image');
 const panel=()=>document.querySelector('[data-layer-group=cells]');
 check(!panel().open,'Cells starts collapsed');
 const visibleBefore=cells.visible;
 check(control('Show cell outlines').getClientRects().length,'Visibility checkbox remains available while collapsed');
 control('Show cell outlines').click();
 check(cells.visible!==visibleBefore&&!panel().open,'Visibility can change without expanding Cells');
 control('Show cell outlines').click();
 check(cells.visible===visibleBefore&&!panel().open&&!control('Cell top markers').getClientRects().length,'Visibility restores independently of folded marker controls');
 panel().querySelector('summary').click();
 check(panel().querySelector('summary>.name').textContent==='Cells','Panel is named Cells');
 check(panel().querySelector('summary>.name').previousElementSibling===control('Show cell outlines')&&!/\bShow\b/.test(panel().querySelector('summary').textContent),'Unlabelled visibility checkbox is immediately before Cells');
 const nameBounds=panel().querySelector('summary>.name').getBoundingClientRect(), markerBounds=control('Cell top markers').getBoundingClientRect();
 check(markerBounds.left>nameBounds.right&&markerBounds.left-nameBounds.right<24,'Top markers sits directly beside Cells when expanded');
 check(!panel().querySelector('[data-cell-inspection]')&&!panel().querySelector('[data-cell-layer="'+cells.id+'"]>summary'),'No duplicate cell or inspection panel');
 check(!Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Density map'),'Density removed from Add to the view in Studio');
 const before=[cells.table,cells.color_by,cells.visible];panel().querySelector('summary').click();
 check(!panel().open&&!control('Cell top markers').getClientRects().length,'Header controls disappear when folded');
 check(JSON.stringify(before)===JSON.stringify([cells.table,cells.color_by,cells.visible]),'Folding preserves data, colours and visibility');panel().querySelector('summary').click();
 if(!control('Cell top markers').checked)control('Cell top markers').click();
 check(control('Cell top marker set').getClientRects().length,'Top markers reveals settings');
 control('Cell top markers').click();check(!control('Cell top marker set').getClientRects().length,'Top marker settings disappear when off');
 await until(()=>control('Colour the cells by')?.querySelector('option[value=phenotype_training]'));
 const filter=control('Add a filter'),colour=control('Colour the cells by');check(!!(filter.compareDocumentPosition(colour)&Node.DOCUMENT_POSITION_FOLLOWING),'Filters precede Colour by');
 change(colour,'phenotype_training');await until(()=>control('Show group A'));
 const search=control('Find cell groups');search.value='B';search.dispatchEvent(new Event('input'));
 panel().querySelector('.cell-group-search input[type=checkbox]').click();
 await until(()=>cells.hidden.has('B'));check(!cells.hidden.has('A'),'Select all matching preserves groups outside the search');
 await until(()=>control('Colour for A'));change(control('Colour for A'),'#12ab34');
 await until(()=>cellColourQuery(cells).includes('12ab34'));check(true,'Group colours reach the actual cell tile request');
 const densityButton=()=>control('Density from shown cells');
 check(!document.querySelector('.cell-density-maps')&&densityButton().getAttribute('aria-pressed')==='false','No density panel before the button is clicked');
 densityButton().click();await until(()=>s.layers.some(l=>l.type==='density')&&!densityButton().disabled);
 const removed=s.layers.find(l=>l.type==='density');await until(()=>removed.item);
 check(densityButton().getAttribute('aria-pressed')==='true'&&getComputedStyle(densityButton()).backgroundColor==='rgb(36, 75, 50)','Density turns green when enabled');
 densityButton().click();
 check(!s.layers.includes(removed)&&!removed.item&&!document.querySelector('.cell-density-maps')&&densityButton().getAttribute('aria-pressed')==='false','Second click removes the map, its panel and green state');
 densityButton().click();await until(()=>s.layers.some(l=>l.type==='density')&&!densityButton().disabled);
 const closed=s.layers.find(l=>l.type==='density');
 document.querySelector('[data-layer-id="'+closed.id+'"] button[aria-label^="Remove "]').click();
 check(!s.layers.includes(closed)&&!document.querySelector('.cell-density-maps')&&densityButton().getAttribute('aria-pressed')==='false','Close button removes the map and resets Density');
 densityButton().click();await until(()=>s.layers.some(l=>l.type==='density')&&!densityButton().disabled);
 let map=s.layers.find(l=>l.type==='density');
 const density=async m=>{const r=await fetch(api('/cells/'+m.token+'/density/legend?'+densityQuery(m)));if(!r.ok)throw Error(await r.text());return r.json();};
 check(map.cellsId===cells.id&&map.table===table&&map.cellScope==='viewer'&&!map.vcol,'New density uses the current viewer table and selected cells');
 check((await density(map)).cells===20,'Density counts only the selected category');
 const mapBody=()=>document.querySelector('[data-layer-id="'+map.id+'"] .layer-body');
 await until(()=>mapBody()?.querySelector('.gradient'));
 check(!!mapBody().querySelector('.density-control-grid')&&!mapBody().querySelector('.density-more').open,'Compact map controls, independent filters optional');
 change(control('Density cells included'),'snapshot');await until(()=>map.dataMode==='snapshot');
 await showViewerCellGroups('phenotype_training',['B'],['A','B']);
 check(map.densitySnapshot.hidden.includes('B')&&(await density(map)).cells===20,'Fixed map keeps its original group while the viewer changes');
 const frozenTable=map.table;cells.tables.push({id:'t0',name:'Segmentation measurements',file:'cells.csv'});
 await chooseCellSource(cells,'t0');check(map.table===frozenTable&&cells.table==='t0','Fixed map keeps its original table');
 change(control('Density cells included'),'viewer');check(map.table==='t0','Following current cells explicitly restores shared source');
 await chooseCellSource(cells,table);
 // Registered imports must not switch or replace the loaded source.
 const originalVisible=cells.visible;
 const extra=await registerImportedCells({id:'alternate-outlines',type:'cells',name:'Imported outlines',token,image_token:image.token,roi:'roi',tables:[{id:'t0',name:'Imported measurements'}],table:'t0',mode:'both'});
 check(activeCellLayer()===cells&&cells.table===table&&cells.visible===originalVisible&&!extra.visible,'Import leaves the current viewer table and outlines intact');
 check([...control('Import cells').options].some(o=>o.value==='source:alternate-outlines')&&[...control('Import cells').options].some(o=>o.value==='table:alternate-outlines|t0'),'Bottom list contains imported geometry and linked data');
 change(control('Import cells'),'source:alternate-outlines');await until(()=>activeCellLayer()===extra&&extra.visible);
 check(control('Cell data').value==='alternate-outlines|t0','Explicit bottom-list switch changes the active pair');
 await until(()=>control('Colour the cells by')?.querySelector('option[value=M0]'));change(control('Colour the cells by'),'M0');
 await until(()=>control('Show cells from')&&!document.querySelector('.cell-range').parentElement.hidden);
 check(densityButton().getAttribute('aria-pressed')==='false','Imported source has its own Density toggle');
 const previous=s.layers.filter(l=>l.type==='density').length;densityButton().click();await until(()=>s.layers.filter(l=>l.type==='density').length>previous&&!densityButton().disabled);
 const mean=s.layers.filter(l=>l.type==='density').find(m=>m.cellsId===extra.id);
 check(mean&&mean.vcol==='M0'&&densityCellLayer(mean)===extra,'Numeric mean uses the selected imported source, not the first segmentation');
 check((await density(mean)).cells===40,'Imported source map is usable by the backend');
 await chooseCellSource(cells,table,false);
 await registerImportedCells({id:cells.id,type:'cells',token,image_token:image.token,tables:[{id:'t0',name:'Another imported table'}]});
 check(cells.table===table&&activeCellLayer()===cells,'Adding a linked table keeps the current data choice');
 // Clear images/chrome only for a useful, genuinely rendered panel screenshot.
 s.layerGroups.cells=true;cellRowsOpen()[map.id]=true;
 setCellColour(cells,'phenotype_training');await until(()=>control('Show group A'));
 cells.legendSearch={};buildPanel();
 const grid=document.querySelector('.density-control-grid');
 const dock=document.querySelector('#app');dock.style.setProperty('--studio-right','760px');
 document.querySelectorAll('#panel .layer-group:not([data-layer-group=cells]),#panel>.layer').forEach(e=>e.style.display='none');
 panel().scrollIntoView({block:'start'});await wait(300);
 check(panel().scrollWidth<=panel().clientWidth+1&&panel().getBoundingClientRect().right<=innerWidth,'Wide panel stays inside the dock without horizontal overflow');
 dock.style.setProperty('--studio-right','340px');await wait(150);
 check(panel().scrollWidth<=panel().clientWidth+1,'Narrow panel wraps without horizontal overflow');
 dock.style.setProperty('--studio-right','760px');await wait(150);
 check(image===s.layers.find(l=>l.type==='image')&&image.item,'Source changes preserve the open image');
 // Reopening retains the previous button's settings, but never starts a map.
 delete map.cellDensity;map.radius=70;saveView();await flushViewerSettings();
 await init(s.spec);await until(()=>s.cellsReady);await s.phenotypingLink;
 check(s.layers.some(l=>l.id===map.id)&&s.layers.filter(isCellDensityMap).every(l=>!l.visible&&!l.item),'Saved Cells density maps stay off when the image reopens');
 s.layerGroups.cells=true;buildPanel();
 check(!document.querySelector('.cell-density-maps')&&densityButton().getAttribute('aria-pressed')==='false','Saved density panels also stay hidden until clicked');
 densityButton().click();await until(()=>s.layers.find(l=>l.id===map.id)?.item);
 const restored=s.layers.find(l=>l.id===map.id);
 check(restored.radius===70&&!restored.densityDormant&&densityButton().getAttribute('aria-pressed')==='true','Click restores existing density settings and green state');
 document.querySelectorAll('#panel .layer-group:not([data-layer-group=cells]),#panel>.layer').forEach(e=>e.style.display='none');
 panel().scrollIntoView({block:'start'});await wait(300);
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(e.stack));}})();
</script>'''

if __name__ == '__main__':
    main(DRIVER,studio=True,timeout=150,capture='/tmp/spacemap-cells-workspace.png')
