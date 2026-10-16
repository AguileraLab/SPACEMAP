"""Right-click phenotype assignment, real model scores, multi-selection and Undo."""
from viewer_threshold_browser import main
from viewer_studio_cell_review_browser import setup as review_setup


def setup(root):
    review_setup(root)
    from spacemap_gui.studio import reference_models, tasks
    from spacemap_gui.viewer import api
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    settings = {"table":str(root/"cells.csv"),"training_column":"phenotype_ref1",
                "roi_column":"batch_id","label_column":"label","markers":["M0","M1"]}
    settings["revision"] = tasks.source_revision(settings)
    trained = reference_models.train(settings, root, lambda *args: None)
    api.app.router.routes.append(Route('/menu-model', lambda request: JSONResponse(trained)))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change',{bubbles:true}));};
const read=async()=>await(await fetch('/spacemap-viewer/api/cell-review-test')).json();
const card=()=>document.querySelector('.manual-reference-card');
const button=(text)=>Array.from(card()?.querySelectorAll('button') || []).find(b=>b.textContent===text);
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent(event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{try{
  const s=window.__state;
  await until(()=>s.imagesReady&&s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
  change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
  const image=()=>s.layers.find(l=>l.type==='image'), cells=()=>s.layers.find(l=>l.type==='cells');
  await until(()=>cells().combinedTable&&cells().table===cells().combinedTable&&card()?.referenceInfo);
  s.layers.filter(l=>l.type==='annotations').forEach(l=>l.visible=false);setTool('off');
  const model=await(await fetch('/spacemap-viewer/api/menu-model')).json();
  cells().suggestionsOn=true;cells().suggestionMode='saved_reference';cells().suggestionModelId=model.models[0].model_id;
  const click=(x=55,y=15,ctrl=false)=>s.viewer.raiseEvent('canvas-click',{quick:true,
    position:s.viewer.viewport.pixelFromPoint(image().item.imageToViewportCoordinates(x,y)),originalEvent:{ctrlKey:ctrl}});
  const menu=()=>document.querySelector('#ctxmenu');
  const open=()=>{const bounds=s.viewer.element.getBoundingClientRect();s.viewer.element.dispatchEvent(new MouseEvent('contextmenu',
    {bubbles:true,cancelable:true,clientX:bounds.left+140,clientY:bounds.top+180}));};
  click();await until(()=>s.inspectedCell?.result.suggestions?.status==='ready');
  open();await until(()=>menu()?.querySelector('[data-phenotype="Treg"]'));
  const first=menu().querySelector('.ctx-phenotype-suggestion'), choice=s.inspectedCell.result.suggestions.choices[0];
  assert(first.dataset.phenotype===choice.phenotype&&first.textContent.includes((100*choice.probability).toFixed(1)+'%'),
    'Right-click places the current model suggestion and its score above the preset names');
  assert(menu().textContent.includes('Assign to 1 cell · phenotype_ref1')&&menu().textContent.includes('Image zoom'),
    'The menu names the destination and selection size while preserving image controls');
  menu().querySelector('[data-phenotype="Treg"]').click();
  await until(async()=>(await read()).live[1]==='Treg');
  await until(()=>s.manualReferenceEdit&&s.inspectedCell?.result.details.phenotype_ref1==='Treg');
  assert(s.inspectedCell.result.details.phenotype_ref1==='Treg','A quick assignment saves the selected cell and immediately updates its cell card');
  button('Undo').click();await until(async()=>(await read()).live[1]==='A');
  await until(()=>s.inspectedCell.result.details.phenotype_ref1==='A');
  assert(true,'Undo refreshes the displayed reference label on the selected cell');
  click(215,95,true);await until(()=>s.inspectedCell?.result.members?.length===2&&s.inspectedCell.result.suggestions?.status==='ready');
  open();await until(()=>menu()?.querySelector('.ctx-phenotype-suggestion'));
  const best=s.inspectedCell.result.suggestions.choices[0];
  assert(menu().textContent.includes('Assign to 2 cells')&&menu().textContent.includes('average scores')&&
    menu().querySelector('.ctx-phenotype-suggestion').textContent.includes((100*best.probability).toFixed(1)+'%'),
    'Multiple selected cells show their average model scores and the exact assignment count');
  menu().querySelector('.ctx-phenotype-suggestion').click();
  await until(async()=>{const v=(await read()).live;return v[1]===best.phenotype&&v[21]===best.phenotype;});
  const after=(await read()).live;
  assert(after.every((name,index)=>[1,21].includes(index)||name===(index<20?'A':'B')),'Accepting a suggestion changes exactly the selected cells');
  button('Undo').click();await until(async()=>{const v=(await read()).live;return v[1]==='A'&&v[21]==='B';});
  open();await until(()=>menu()?.querySelector('[data-phenotype="Treg"]'));
  const old=menu().querySelector('[data-phenotype="Treg"]');
  click(95);await until(()=>s.inspectedCell?.result.label===3&&s.inspectedCell.result.profile?.status==='ready');
  old.click();await delay(300);
  assert((await read()).live.every((name,index)=>name===(index<20?'A':'B')),'A menu opened for an earlier selection cannot label newly selected cells');
  open();await until(()=>menu()?.querySelector('[data-phenotype="Treg"]'));
  const search=menu().querySelector('[aria-label="Find phenotype to assign"]');search.value='Custom from menu';
  search.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(async()=>(await read()).live[2]==='Custom from menu');
  await until(()=>s.manualReferenceEdit&&s.inspectedCell.result.details.phenotype_ref1==='Custom from menu');
  assert(true,'A typed phenotype can also be assigned directly from the context menu');
  button('Undo').click();await until(async()=>(await read()).live[2]==='A');
  state.manualReferenceSource='data';dataView.selected=new Set([12,13]);dataView.pickedLabels=[13,14];dataView.pickedRoi='roi';
  open();
  assert(!menu().querySelector('[data-phenotype]')&&menu().textContent.includes('Data rows selected'),
    'A different Data selection cannot accidentally relabel an old inspected cell');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+card()?.textContent+'\n'+document.querySelector('#ctxmenu')?.textContent));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER,studio=True,timeout=180,setup=setup,screenshot=False)
