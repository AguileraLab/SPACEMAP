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
  const palette=await (await fetch('/spacemap-viewer/api/studio/browser-check/colours')).json();
  assert(palette.colours.Epithelial==='#7a3e85' && palette.colours.Epithelia==='#7a3e85','Save names does not overwrite the preset with temporary cluster colours');
  swatches()[0].value='#123456';swatches()[0].dispatchEvent(new Event('input'));
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
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}
},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=360, setup=setup)
