"""Saved SOM names, heatmap keys, cell legends and shared editable colours agree."""
from viewer_studio_som_browser import DRIVER as SOM_DRIVER, setup
from viewer_threshold_browser import main

# Exercise the real SOM / naming workflow, then verify colour changes without rerunning clustering.
DRIVER = SOM_DRIVER[:SOM_DRIVER.index("  tools().querySelectorAll('[aria-label^=\"Phenotype of Cluster\"]').forEach(input=>{input.value='Revised'")]
DRIVER = DRIVER.replace("index?'Tumor':'T cell'", "index?'CD8':'Epithelia'")
DRIVER += r"""
  const swatches=()=>Array.from(tools().querySelectorAll('[aria-label^="Colour of Cluster"]'));
  await until(()=>swatches()[0]?.value==='#7a3e85' && swatches()[1]?.value==='#d62728');
  assert(true,'SOM cluster names use the supplied Epithelial and CD8 colours');
  const heatColours=()=>Array.from(document.querySelectorAll('#studio-figures .studio-heatmap .row rect[width="10"]')).map(r=>r.getAttribute('fill'));
  assert(heatColours().includes('#7a3e85') && heatColours().includes('#d62728'),'Heatmap phenotype keys match the naming fields');
  const legend=async()=>(await (await fetch('/spacemap-viewer/api/cells/'+cells.token+'/legend?t='+encodeURIComponent(cells.table)+'&col=phenotype_som_check')).json()).items;
  const original=await legend();
  assert(original.find(i=>i.value==='Epithelia').color==='#7a3e85' && original.find(i=>i.value==='CD8').color==='#d62728','SOM cell legend and mask lookup match the phenotype preset');
  // Reproduce the old Show result overlay on the same segmentation.
  await addLayerToView({...state.spec.layers.find(l=>l.type==='cells'),id:'result-legacy',
    visible:true,color_by:run.result.column,hidden:[]});
  assert(state.layers.filter(l=>l.type==='cells').length===2,'Legacy duplicate overlay reproduced');
  button(document.querySelector('[data-figure=som]'),'Show all').click();
  await until(()=>cells.color_by===run.result.column && state.layers.filter(l=>l.type==='cells').length===1);
  const first=run.result.clusters[0].name;
  tools().querySelector('[data-group="'+first+'"]').click();
  await until(()=>cells.hidden.has(second)&&!cells.hidden.has(first));
  const rawLegend=async()=>(await (await fetch('/spacemap-viewer/api/cells/'+cells.token+'/legend?'+cellColourQuery(cells))).json()).items;
  const reviewed=await rawLegend();
  assert(reviewed.find(i=>i.value===first).color==='#7a3e85'&&reviewed.find(i=>i.value===second).color==='#d62728',
    'Picking a saved named cluster keeps the heatmap colours on the raw cluster legend');
  showDataTab();
  const dataReady=()=>dataView.queryReady&&!dataView.loading;
  const dataGroups=()=>dataView.result.rows.map(row=>row[dataView.result.shown.indexOf(run.result.column)+1]);
  await until(()=>dataReady()&&dataView.result.shown.includes(run.result.column)&&
    dataView.result.rows.length===reviewed.find(i=>i.value===first).count&&dataGroups().every(value=>value===first));
  assert(dataView.onlySelected&&document.querySelector('[aria-label="Only selected cells"]').checked,
    'Opening Data after a cluster click automatically shows only that cluster’s cells');
  button(document.querySelector('[data-figure=som]'),'Show all').click();
  await until(()=>dataReady()&&dataView.result.rows.length===40);
  assert(!dataView.onlySelected,'Show all clears the cluster filter in Data');
  const legendRow=group=>[...document.querySelectorAll('[data-cell-layer="'+cells.id+'"] .legend .item')]
    .find(row=>row.textContent.includes('('+group+')'));
  await until(()=>legendRow(first));
  legendRow(first).click();
  await until(()=>dataReady()&&dataView.onlySelected&&dataView.result.rows.length===reviewed.find(i=>i.value===second).count);
  assert(dataGroups().every(value=>value===second),'Cell-legend toggles filter Data to the visible group');
  await until(()=>legendRow(second));
  legendRow(second).click();
  await until(()=>dataReady()&&dataView.result.rows.length===0);
  assert(true,'Hiding every legend group shows zero Data rows');
  tools().querySelector('[data-group="'+first+'"]').click();
  await until(()=>dataReady()&&dataGroups().length>0&&dataGroups().every(value=>value===first));
  assert(cells.hidden.has(second)&&!cells.hidden.has(first),'A heatmap/list pick restores the same cells in Data and the image');
  const pixelColours=async()=>{
    const blob=await (await fetch(tileUrl(cells,0,0,0,state.versions[cells.id]))).blob();
    const bitmap=await createImageBitmap(blob), canvas=document.createElement('canvas');
    canvas.width=bitmap.width;canvas.height=bitmap.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);
    const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data, colours=new Set();
    for(let i=0;i<rgba.length;i+=4)if(rgba[i+3])colours.add([...rgba.slice(i,i+3)].map(v=>v.toString(16).padStart(2,'0')).join(''));
    bitmap.close();return colours;
  };
  assert((await pixelColours()).has('7a3e85')&&!(await pixelColours()).has('d62728'),
    'Rendered image tiles use the chosen cluster colour and hide the other cluster');
  const layerRows=()=>[...document.querySelectorAll('[data-cell-layer="'+cells.id+'"] .legend .item')];
  await until(()=>layerRows().some(row=>row.textContent.includes('Epithelia ('+first+')')));
  assert(layerRows().find(row=>row.textContent.includes('Epithelia ('+first+')')).querySelector('.swatch').style.background==='rgb(122, 62, 133)',
    'The visible Cells legend shows the same edited name and colour');
  button(tools(),'Show on the image').click();
  await delay(400);
  assert(state.layers.filter(l=>l.type==='cells').length===1&&cells.hidden.has(second),
    'Show on the image preserves the picked cluster without duplicating the layer');
  const palette=await (await fetch('/spacemap-viewer/api/studio/browser-check/colours')).json();
  assert(palette.colours.Epithelial==='#7a3e85' && palette.colours.Epithelia==='#7a3e85','Save names does not overwrite the preset with temporary cluster colours');
  swatches()[0].value='#123456';swatches()[0].dispatchEvent(new Event('input'));
  assert((await rawLegend()).find(i=>i.value===first).color==='#123456' && (await pixelColours()).has('123456'),
    'Unsaved colour edits preview on the image as well as the heatmap');
  button(tools(),'Save names').click();
  await until(async()=>{const value=await legend();return value?.find(i=>i.value==='Epithelia')?.color==='#123456';});
  assert(true,'A deliberate cluster colour edit updates the cell legend');
  await until(()=>!button(tools(),'Save names').disabled);
  const shared=await (await fetch('/spacemap-viewer/api/studio/browser-check/colours')).json();
  assert(shared.colours.Epithelial==='#123456' && shared.colours.Epithelia==='#123456','An alias colour edit updates the canonical phenotype too');
  picker.value='leiden';picker.dispatchEvent(new Event('change'));
  await delay(300);
  picker.value='som';picker.dispatchEvent(new Event('change'));
  await until(()=>swatches()[0]?.value==='#123456');
  assert(heatColours().includes('#123456'),'Changing methods reloads the shared phenotype colour');
  const names=tools().querySelectorAll('[aria-label^="Phenotype of Cluster"]');
  names[0].value='Brand new phenotype';names[0].dispatchEvent(new Event('change'));
  const reused=await (await fetch('/spacemap-viewer/api/studio/browser-check/colours',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names:['Brand new phenotype']})})).json();
  await until(()=>swatches()[0]?.value === reused.mapping['Brand new phenotype']);
  assert(reused.mapping['Brand new phenotype']!=='#123456','A new phenotype receives a persistent distinct colour');
  // Two raw clusters may share a draft name without losing separate membership.
  names[1].value='Brand new phenotype';names[1].dispatchEvent(new Event('change'));
  await until(()=>swatches()[1]?.value===swatches()[0]?.value);
  button(document.querySelector('[data-figure=som]'),'Show all').click();
  await until(()=>cells.hidden.size===0);
  const rows=()=>[...document.querySelectorAll('[data-figure=som] .studio-heatmap .row')];
  const rowFor=name=>rows().find(row=>row.dataset.row===name);
  rowFor(first).dispatchEvent(new MouseEvent('click',{bubbles:true}));
  await until(()=>cells.hidden.has(second)&&!cells.hidden.has(first));
  const merged=await rawLegend();
  assert(merged.every(i=>i.color===reused.mapping['Brand new phenotype']),
    'Shared draft names share colours while one raw cluster remains independently selectable');
  rowFor(second).dispatchEvent(new MouseEvent('click',{bubbles:true,ctrlKey:true}));
  await until(()=>cells.hidden.size===0);
  assert(true,'Ctrl-click selects both heatmap clusters');
  button(document.querySelector('[data-figure=som]'),'Show all').click();
  await delay(100);
  assert(cells.hidden.size===0&&!document.querySelector('.studio-namer').textContent.includes('null'),'Show all clears the selection cleanly');
  // A late legend response from a previous click cannot replace the newer pick.
  const fetchBeforePick=window.fetch;let delayed=false;
  window.fetch=async(url,options)=>{
    if(!delayed&&String(url).includes('/legend?')&&!String(url).includes('group_colours')){
      delayed=true;await delay(250);
    }
    return fetchBeforePick(url,options);
  };
  rowFor(first).dispatchEvent(new MouseEvent('click',{bubbles:true}));
  await until(()=>delayed);
  rowFor(second).dispatchEvent(new MouseEvent('click',{bubbles:true}));
  await until(()=>cells.hidden.has(first)&&!cells.hidden.has(second));
  await delay(400);window.fetch=fetchBeforePick;
  assert(cells.hidden.has(first)&&!cells.hidden.has(second),'Rapid heatmap clicks keep the latest selection');
  tools().querySelector('[aria-label="Overwrite existing results"]').click();
  button(document.querySelector('.studio-runbar'),'Run SOM').click();
  await until(async()=>{const j=(await jobs()).find(j=>j.kind==='som');return j?.id!==run.id&&j?.status==='done';});
  const rerun=(await jobs()).find(j=>j.kind==='som');
  await until(()=>state.clusterReview?.job===rerun.id);
  assert(cells.hidden.size===0&&Object.keys(state.clusterReview.names).length===0,'A new run clears previous picks and draft names for reused cluster IDs');
  const rerunLegend=await rawLegend();
  assert(rerunLegend.every(i=>i.color===tools().querySelector('[aria-label="Colour of '+i.value+'"]').value),
    'The new run uses the same colours in its heatmap, naming list and cells');
  assert(state.layers.filter(l=>l.type==='cells').length===1,'Rerunning keeps one cell overlay');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}
},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=360, setup=setup)
