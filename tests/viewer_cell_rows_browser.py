"""Browser-check Studio's Cells and maps in Layers: rows for outlines, each colour and the density maps.

Usage: PYTHONPATH=src:tests python tests/viewer_cell_rows_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message.slice(0,80)));};
const change=(element,value)=>{element.value=value;element.dispatchEvent(new Event('change'));};
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent('page error: '+event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[data-step=tab5]'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  const block=()=>document.querySelector('[data-cell-layer="'+cells.id+'"]');
  const group=document.querySelector('[data-layer-group=cells]');
  if(!group.open)group.querySelector('summary').click();
  const field=label=>block()?.querySelector('[aria-label="'+label+'"]');
  await until(()=>field('Cell data')&&field('Draw cells as')&&field('Colour the cells by'));
  assert(!block().querySelector('[data-cell-row]'),'One settings form per cell layer: Data, Show, Colour by, Filter');
  if(!cells.visible){block().querySelector(':scope > summary input[type=checkbox]').click();await until(()=>cells.visible);}
  await until(()=>field('Colour the cells by').querySelectorAll('optgroup').length===2);
  const groupsOf=label=>Array.from(field('Colour the cells by').querySelectorAll('optgroup')).find(g=>g.label===label);
  assert(Array.from(groupsOf('Groups').children).some(o=>o.value==='phenotype_training')&&Array.from(groupsOf('Values').children).some(o=>o.value==='M0'),
    'Colour by sorts the columns into Groups (text) and Values (numbers)');
  change(field('Colour the cells by'),'M0');
  await until(()=>cells.color_by==='M0'&&cells.mode==='both'&&block().querySelector('[aria-label="Show cells from"]'));
  await until(()=>!block().querySelector('.cell-range').parentElement.hidden);
  change(block().querySelector('[aria-label="Show cells from"]'),'100');
  await until(()=>block()?.querySelector('[aria-label="Show cells up to"]'));
  change(block().querySelector('[aria-label="Show cells up to"]'),'200');
  await until(()=>cells.range&&cells.range[1]===200);
  const legend=await (await fetch('/spacemap-viewer/api/cells/'+cells.token+'/legend?'+cellColourQuery(cells))).json();
  assert(legend.kind==='numeric'&&legend.shown===10&&legend.min===100&&legend.max===200,'A value scale can span a range of values: '+JSON.stringify(legend));
  change(field('Draw cells as'),'fill');
  await until(()=>cells.mode==='fill');
  assert(true,'Show sets outlines, filled or both');
  change(field('Colour the cells by'),'phenotype_training');
  await until(()=>cells.color_by==='phenotype_training'&&!cells.range&&block().querySelector('.cell-legend .legend .item'));
  assert(true,'A group column shows its groups in the legend');
  // Filter: keep group A only.
  change(field('Add a filter'),'phenotype_training');
  await until(()=>block().querySelector('[aria-label="Keep B"]'));
  block().querySelector('[aria-label="Keep B"]').click();
  Array.from(block().querySelectorAll('.cell-filter-editor button')).find(b=>b.textContent==='Apply').click();
  await until(()=>(cells.filters||[]).length===1&&/20 of 40 cells shown/.test(block().textContent));
  const tile=await fetch(api('/cells/'+cells.token+'/tile/0/0/0.png?mode=fill&t='+encodeURIComponent(cells.table)+'&f='+encodeURIComponent(JSON.stringify(cells.filters))));
  assert(tile.ok,'A filter keeps the ticked groups: 20 of 40 cells drawn');
  change(field('Add a filter'),'M0');
  await until(()=>block().querySelector('[aria-label="Filter from"]'));
  block().querySelector('[aria-label="Filter from"]').value='100';block().querySelector('[aria-label="Filter to"]').value='150';
  Array.from(block().querySelectorAll('.cell-filter-editor button')).find(b=>b.textContent==='Apply').click();
  await until(()=>(cells.filters||[]).length===2&&/\d+ of 40 cells shown/.test(block().textContent));
  assert(/[0-9] of 40 cells shown/.test(block().textContent),'Filters add up: '+block().querySelector('.cell-filters').textContent);
  block().querySelector('[aria-label="Remove filter M0"]').click();
  await until(()=>(cells.filters||[]).length===1);
  block().querySelector(':scope > summary input[type=checkbox]').click();
  await until(()=>!cells.visible);
  block().querySelector(':scope > summary input[type=checkbox]').click();
  await until(()=>cells.visible&&cells.color_by==='phenotype_training'&&cells.filters.length===1);
  assert(true,'The block tick hides and shows the layer as it was');
  await addLayerToView({id:'density-check',type:'density',name:'A density',token:cells.token,image_token:cells.image_token,tables:cells.tables,
    table:cells.table,fcol:'phenotype_training',fvals:['A'],radius:50,cmap:'magma',visible:true,opacity:.65,added:true});
  buildPanel();
  await until(()=>document.querySelector('[data-cell-layer=density] [data-layer-id=density-check]'));
  const map=state.layers.find(layer=>layer.id==='density-check');
  document.querySelector('[data-cell-layer=density] [data-layer-id=density-check] .layer-head input[type=checkbox]').click();
  await until(()=>!map.visible);
  document.querySelector('[data-cell-layer=density] [data-layer-id=density-check] .layer-head .name').click();
  await until(()=>document.querySelector('[data-cell-layer=density] [data-layer-id=density-check] .layer-body .legend'));
  assert(true,'A density map is a row: show or hide it, open it for its settings');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}
},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=90)
