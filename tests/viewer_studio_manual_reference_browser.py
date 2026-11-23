"""Create a reference, label Data/viewer cells, score it and write the dataset."""
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap import dataio
    from spacemap_gui.viewer import api
    from spacemap_gui.studio import phenotyping_inputs
    # Keep the registered image identity, without the harness's same-sample
    # background-corrected image making the assignment ambiguous.
    images = root / 'images'; images.mkdir()
    (images / 'roi.ome.tiff').symlink_to(root / 'roi.ome.tiff')
    phenotyping_inputs.save_folders(root, {'images':str(images)})
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    frame['cell_id'] = range(1001, 1041)
    frame['phenotype_ref1'] = float('nan')
    frame.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    async def check(request):
        values=dataio.load(table)
        return JSONResponse({'labels':values['review_labels'].fillna('').tolist() if 'review_labels' in values else [],
            'reference':values['phenotype_ref1'].fillna('').tolist(),
            'written':'review_labels' in pd.read_csv(table,nrows=0).columns})
    api.app.router.routes.append(Route('/manual-reference-test',check))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const card=()=>document.querySelector('.manual-reference-card');
const button=text=>Array.from(card()?.querySelectorAll('button') || []).find(b=>b.textContent===text);
const change=(el,v)=>{el.value=v;el.dispatchEvent(new Event('change'));};
const read=async()=>await(await fetch('/spacemap-viewer/api/manual-reference-test')).json();
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent(event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{try{
  const s=window.__state;
  await until(()=>s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
  showDataTab();await until(()=>dataView.combined);await dataChoose('combined');await until(()=>document.querySelector('.data-grid tbody tr')&&dataView.source?.combined);
  const rows=document.querySelectorAll('.data-grid tbody tr');rows[0].click();rows[1].dispatchEvent(new MouseEvent('click',{ctrlKey:true,bubbles:true}));
  document.querySelector('[aria-label="Label Data cells"]').click();
  await until(()=>card()?.querySelector('[aria-label="Create reference column"]'));
  assert(document.querySelector('[aria-label="Phenotyping method"]').value==='reference','Data has a direct route to manual labelling');
  const cells=s.layers.find(l=>l.type==='cells');cells.mode='both';s.layerGroups.cells=true;
  setCellColour(cells,'');
  const pixel=async(x,y)=>{const response=await fetch(tileUrl(cells,0,0,0,Date.now()));if(!response.ok)throw Error(await response.text());
    const image=await createImageBitmap(await response.blob()),canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);image.close();return ctx.getImageData(x,y,1,1).data;};
  change(card().querySelector('[aria-label="Save labels in column"]'),'phenotype_ref1');
  await until(()=>button('Assign 2 cells')&&!button('Assign 2 cells').disabled);
  button('Flag for review').click();
  const reviewQueue=async()=>(await(await fetch('/spacemap-viewer/api/studio/browser-check/review')).json()).items;
  await until(async()=>(await reviewQueue()).filter(i=>i.kind==='cells').length===2);
  assert((await reviewQueue()).filter(i=>i.kind==='cells').every(i=>i.target.count===1),'Selected manual-reference cells are individually flagged without assigning a phenotype');
  const existingName=card().querySelector('[aria-label="Phenotype name to assign"]');existingName.value='Treg';existingName.dispatchEvent(new Event('input'));
  button('Assign 2 cells').click();
  await until(async()=>(await read()).reference.slice(0,2).every(v=>v==='Treg'));
  await until(()=>dataView.result?.kinds?.phenotype_ref1==='group');
  await until(()=>cells.referenceOverlay==='phenotype_ref1');
  assert(!cells.color_by&&document.querySelector('[aria-label="Colour cells by"]').value===''&&document.querySelector('[aria-label="Colour the cells by"]').value==='',
    'Assigning manual labels preserves Single colour on both controls');
  assert((await pixel(20,20))[3]>0&&(await pixel(100,10))[3]>0&&(await pixel(100,20))[3]===0,
    'Assigned cells are filled while unassigned outlines stay visible');
  const overlay=document.querySelector('[aria-label="Colour assigned cells"]');overlay.click();
  assert((await pixel(20,20))[3]===0&&(await pixel(20,10))[3]>0,'Manual-label overlay can be hidden without hiding outlines');
  document.querySelector('[aria-label="Colour assigned cells"]').click();
  dataView.filters=[{col:'phenotype_ref1',values:['Treg']}];await dataLoad();
  await until(()=>dataView.queryReady&&dataView.result.matched===2);
  assert(true,'Text labels in an empty numeric reference appear immediately and filter correctly in Data');
  await until(()=>button('Undo')&&!button('Undo').disabled);button('Undo').click();
  await until(async()=>!(await read()).reference.some(Boolean));
  assert(!cells.color_by,'Undo preserves the colour choice');
  dataView.filters=[];await dataLoad();
  await until(()=>dataView.queryReady&&document.querySelectorAll('.data-grid tbody tr').length>=2);
  const restoredRows=document.querySelectorAll('.data-grid tbody tr');restoredRows[0].click();restoredRows[1].dispatchEvent(new MouseEvent('click',{ctrlKey:true,bubbles:true}));
  assert(true,'Undo clears the numeric-origin reference without a conversion error');
  change(card().querySelector('[aria-label="Save labels in column"]'),'review_labels');
  card().querySelector('[aria-label="Create reference column"]').click();
  await until(()=>card().querySelector('datalist option[value="review_labels"]'));
  await until(()=>button('Assign 2 cells')&&!button('Assign 2 cells').disabled);
  const name=card().querySelector('[aria-label="Phenotype name to assign"]');name.value='A';name.dispatchEvent(new Event('input'));
  button('Assign 2 cells').click();
  await until(async()=>{const result=await read();return result.labels[0]==='A'&&result.labels[1]==='A';});
  assert((await read()).labels.filter(Boolean).length===2,'Only the two selected Data rows receive the typed phenotype');
  await until(()=>dataView.result.columns.includes('review_labels'));
  assert(dataView.result.columns.includes('review_labels'),'The custom reference appears in the combined Data table immediately');
  await until(()=>button('Undo'));button('Undo').click();
  await until(async()=>!(await read()).labels.some(Boolean));
  assert(true,'Undo restores the previous labels');
  const layer=s.layers.find(l=>l.type==='cells');
  document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer,labels:[3,4]}}));
  await until(()=>card().querySelector('[aria-label="Cells to label"]').value==='viewer'&&button('Assign 2 cells')&&!button('Assign 2 cells').disabled);
  button('Assign 2 cells').click();
  await until(async()=>{const result=await read();return result.labels[2]==='A'&&result.labels[3]==='A';});
  assert((await read()).labels.filter(Boolean).length===2,'Viewer cell selection uses the same reference assignment');
  assert(!!button('Compare predictions (F1)…'),'Reference panel links to the unified Visualization assessment');
  dataView.filters=[{col:'label',lo:5,hi:7}];await dataLoad();
  await until(()=>dataView.queryReady&&dataView.result.matched===3);
  change(card().querySelector('[aria-label="Cells to label"]'),'filtered');
  const filteredName=card().querySelector('[aria-label="Phenotype name to assign"]');filteredName.value='B';filteredName.dispatchEvent(new Event('input'));
  await until(()=>button('Assign 3 cells')&&!button('Assign 3 cells').disabled);button('Assign 3 cells').click();
  await until(async()=>(await read()).labels.slice(4,7).every(value=>value==='B'));
  assert((await read()).labels.filter(Boolean).length===5,'Filtered Data rows assign exactly the displayed matches and preserve other labels');
  await until(()=>button('Undo'));button('Undo').click();
  await until(async()=>(await read()).labels.filter(Boolean).length===2);
  const models=card().querySelector('.reference-model-training');models.open=true;
  await until(()=>card().querySelector('[aria-label="Phenotyping model to load"]'));
  assert(!!button('Current image')&&!!button('Selected images'),'Training offers current image or selected images');
  card().querySelector('.reference-training-details').open=true;
  button('Predict other cells…').click();
  await until(()=>document.querySelector('[aria-label="Training reference column"]')?.value==='review_labels');
  assert(true,'Predict other cells opens MR-ML with the custom reference already selected');
  change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
  await until(()=>button('Write into table')&&!button('Write into table').disabled);button('Write into table').click();
  await until(async()=>(await read()).written);
  assert((await read()).labels[2]==='A','Write into table preserves the reference column in the dataset');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+card()?.textContent));}},100);
</script>
"""


if __name__=='__main__':
    main(DRIVER,studio=True,timeout=240,setup=setup,screenshot=False)
