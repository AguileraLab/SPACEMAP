"""Create a reference, label Data/viewer cells, score it and write the dataset."""
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap import dataio
    from spacemap_gui.viewer import api
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    frame['cell_id'] = range(1001, 1041)
    frame.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    async def check(request):
        values=dataio.load(table)
        return JSONResponse({'labels':values['review_labels'].fillna('').tolist() if 'review_labels' in values else [],
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
  const evaluation=Array.from(card().querySelectorAll('details')).find(d=>d.querySelector('summary')?.textContent==='Check agreement (F1)');evaluation.open=true;
  await until(()=>card().querySelector('[aria-label="Prediction column to check"] option[value="phenotype_training"]')&&!button('Calculate F1').disabled);
  change(card().querySelector('[aria-label="Prediction column to check"]'),'phenotype_training');button('Calculate F1').click();
  await until(()=>card().textContent.includes('Macro F1 1.000'));
  assert(card().textContent.includes('2 labelled cells'),'F1 compares the named predictions only on the two reference cells');
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
  await until(()=>card().querySelector('[aria-label="Reference model library folder"]').value);
  assert(!!button('Train from this reference')&&!!button('Predict other cells…'),'Training, prediction and model folder are available beside the reference');
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
