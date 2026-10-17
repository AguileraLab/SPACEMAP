"""Browser-check the Data tab below the image: the cells' table in pages, filters, sorting, rows and cells linked both
ways, Show on image, an edit and an export.

Usage: PYTHONPATH=src:tests python tests/viewer_data_view_browser.py
"""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<600;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.cellsReady&&s.studio);
 const cells=s.layers.find(l=>l.type==='cells');
 showDataTab();
 const grid=()=>document.querySelector('.data-grid');
 await until(()=>grid()&&grid().querySelectorAll('tbody tr').length===40);
 assert(document.querySelector('.split-tab.on').dataset.tab==='data'&&/1–40 of 40 rows/.test(document.querySelector('.data-status').textContent),
   'The Data tab shows the image cells\' table, 40 rows');
 const status=()=>document.querySelector('.data-status').textContent;
 const columns=Array.from(grid().querySelectorAll('thead tr:first-child th')).length-1;
 assert(columns===dataView.result.columns.length&&getComputedStyle(document.querySelector('.data-scroll')).overflowX==='auto','Every column is shown ('+columns+'); the grid scrolls sideways');
 const menu=()=>document.querySelector('.data-filter-menu');
 grid().querySelector('[aria-label="Filter M0"]').click();
 await until(()=>menu()?.querySelector('[aria-label=From]'));
 assert(menu().querySelectorAll('.data-bar').length>1,'A number column\'s filter shows its histogram');
 const from=menu().querySelector('[aria-label=From]');from.value='200';from.dispatchEvent(new Event('input'));
 await until(()=>/of 20 \(40 rows\)/.test(status()));
 assert(/≥ 200/.test(grid().querySelector('[aria-label="Filter M0"]').textContent),'From 200 keeps 20 rows; the column shows its filter');
 await until(()=>(cells.filters||[]).some(filter=>filter.col==='M0'&&filter.lo===200));
 assert(cells.filters.some(filter=>filter.col==='M0'&&filter.lo===200),'The image follows the filters: the cell layer draws only these rows\' cells');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));
 grid().querySelector('[aria-label="Filter phenotype_training"]').click();
 await until(()=>menu()?.querySelector('.data-value input'));
 const value=menu().querySelector('.data-value'),count=Number(value.lastChild.textContent.replace(/,/g,''));
 value.querySelector('input').click();
 await until(()=>status().includes('of '+(20-count)+' (40 rows)'));
 assert(count>0,'Unticking a group keeps the other rows ('+(20-count)+')');
 Array.from(menu().querySelectorAll('button')).find(b=>b.textContent==='All').click();
 await until(()=>/of 20 \(40 rows\)/.test(status())&&!cells.filters.some(filter=>filter.col==='phenotype_training'));
 assert(true,'All ticks every value again (no filter on that column)');
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));
 Array.from(grid().querySelectorAll('thead th')).find(th=>th.textContent.startsWith('M0')).click();
 await until(()=>/▲/.test(grid().querySelector('thead').textContent));
 const first=()=>grid().querySelector('tbody tr');
 assert(first().querySelector('td[data-col="M0"]').textContent==='201','Clicking a header sorts');
 first().click();
 await until(()=>s.cellOutline&&s.cellOutline.length);
 assert(first().classList.contains('selected'),'Clicking a row outlines its cell on the image');
 await inspectReviewedCell(cells,Number(first().querySelector('td[data-col="label"]').textContent));
 await until(()=>splitMarkers.pinned);
 first().dispatchEvent(new MouseEvent('click',{bubbles:true,ctrlKey:true}));
 assert(!dataView.selected.size&&!s.inspectedCell&&!splitMarkers.pinned,'Deselecting the final Data row clears the inspected cell and releases marker tiles');
 // A cell picked on the image selects its row.
 document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer:cells,labels:[39]}}));
 await until(()=>grid().querySelector('tr.selected td[data-col="label"]')?.textContent==='39');
 assert(true,'A cell clicked on the image selects its row');
 const cell=grid().querySelector('tbody tr td[data-col="phenotype_training"]');
 cell.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
 await until(()=>grid().querySelector('.data-edit'));
 const input=grid().querySelector('.data-edit');input.value='Edited';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}));
 await until(()=>/1 edits not saved/.test(document.querySelector('.data-status').textContent));
 assert(grid().querySelector('tbody tr td[data-col="phenotype_training"]').textContent==='Edited','A value is edited in place, kept until saved');
 const q=encodeURIComponent(JSON.stringify(dataBody({columns:['label','M0']})));
 const csv=await (await fetch(api(dataRoute('export','?q='+q)))).text();
 assert(csv.split('\n')[0].trim()==='label,M0'&&csv.trim().split('\n').length===21,'Export writes these rows and the columns asked for');
 // Several cells picked on the image must all be renamed, even though Data scrolls to the last one.
 document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer:cells,labels:[37,38]}}));
 await until(()=>dataView.queryReady&&!dataView.loading&&dataView.result.identity?.[[...dataView.selected][0]]?.label===38);
 const ph=dataView.result.shown.indexOf('phenotype_training')+1,label=dataView.result.shown.indexOf('label')+1;
 const previous=dataView.result.rows.find(r=>r[label]===37)[ph];
 grid().querySelector('[aria-label="Filter phenotype_training"]').click();
 await until(()=>menu()?.querySelector('.data-value-name'));
 menu().querySelector('[aria-label="Rename selected cells only"]').click();
 const rename=[...menu().querySelectorAll('.data-value-name')].find(i=>i.value===previous);
 rename.value='Viewer reviewed';rename.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>!dataView.busy&&!dataView.loading&&dataView.result.edited===2&&menu()?.querySelector('.data-value-name:not(:disabled)'));
 const changed=dataView.result.rows.filter(r=>r[ph]==='Viewer reviewed').map(r=>r[label]).sort();
 assert(JSON.stringify(changed)==='[37,38]','Selected-only renaming honours all picked viewer cells, not just the last row');
 closeContextMenu();await dataHistory('undo');
 assert(dataView.result.edited===1&&!dataView.result.rows.some(r=>r[ph]==='Viewer reviewed'),'Undo restores the viewer-selected labels');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=120)
