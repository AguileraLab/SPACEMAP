"""Browser-check Studio's Cellpose training (Segmentation) on a disposable project in headless Firefox.

The user's workflow: start a session from a base model, place a training tile and a validation tile, correct their
cells, validate, train a version, use it, and predict the next tile with it.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_training_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<4800;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab5]'));
  document.querySelector('[data-step=tab5]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Model name"]'));
  const image=state.layers.find(layer=>layer.type==='image'), viewer=state.viewer;
  const at=(x,y)=>viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y));
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
  set('Model name','browser_check');set('Start from','nuclei');
  button(tools(),'Start').click();
  await until(()=>/not trained yet/.test(tools().textContent));
  assert(/starts from nuclei/.test(tools().textContent),'A session starts from the chosen base model');
  // A tile: placed by a click, predicted, corrected (a cell drawn), kept.
  const tile=async(role,x,y)=>{
    set('Tile size','256');set('Tile role',role);
    button(tools(),'Place a tile').click();
    await delay(50);
    viewer.raiseEvent('canvas-click',{quick:true,position:at(x,y)});
    await until(()=>/found \d+ cells|rror/.test(tools().textContent));
    const cells=state.layers.find(layer=>layer.local==='Cells to correct');
    document.querySelector('[data-tool=polygon]').click();
    const before=cells.features.length;
    [[x-20,y-20],[x+20,y-20],[x+20,y+20],[x-20,y+20]].forEach(([px,py])=>viewer.raiseEvent('canvas-click',{quick:true,position:at(px,py)}));
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
    await until(()=>cells.features.length===before+1);
    if(role==='train'){
      // With Freehand on, a right-click deletes the cell under it; Ctrl+Z brings it back, Ctrl+Y deletes it again.
      document.querySelector('[data-tool=freehand]').click();
      const bounds=viewer.element.getBoundingClientRect(), spot=at(x,y);
      const canvas=viewer.canvas||viewer.element;
      canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:bounds.left+spot.x,clientY:bounds.top+spot.y}));
      await until(()=>cells.features.length===before);
      document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
      await until(()=>cells.features.length===before+1);
      document.dispatchEvent(new KeyboardEvent('keydown',{key:'y',ctrlKey:true,bubbles:true}));
      await until(()=>cells.features.length===before);
      document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
      await until(()=>cells.features.length===before+1);
      assert(true,'Right-click deletes a cell, Ctrl+Z brings it back and Ctrl+Y redoes');
      // Split cuts the drawn cell along a dragged line; Merge joins the halves again (only the drawn cell kept).
      cells.features=cells.features.slice(-1);const drawn=cells.features[0].properties.name;
      document.querySelector('[data-tool=split]').click();
      viewer.raiseEvent('canvas-press',{position:at(x,y-30)});
      viewer.raiseEvent('canvas-drag',{position:at(x,y+30)});
      viewer.raiseEvent('canvas-release',{position:at(x,y+30)});
      await until(()=>cells.features.length===2);
      assert(cells.features.some(f=>f.properties.name===drawn+' part 2'),'Split cuts a cell in two along the dragged line');
      document.querySelector('[data-tool=merge]').click();
      viewer.raiseEvent('canvas-click',{quick:true,position:at(x-10,y)});
      viewer.raiseEvent('canvas-click',{quick:true,position:at(x+10,y)});
      await until(()=>cells.features.length===1);
      assert(cells.features[0].properties.name===drawn,'Merge joins the two halves into one cell');
      document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
      await until(()=>cells.features.length===2);
      assert(true,'Ctrl+Z undoes a merge');
      document.querySelector('[data-tool=off]').click();
    }
    button(tools(),'Keep tile').click();
    await until(()=>/Kept \d+ cells|rror/.test(tools().textContent));
  };
  await tile('train',130,130);
  await until(()=>/1 training · 0 validation/.test(tools().textContent));
  assert(/1 training · 0 validation/.test(tools().textContent) && state.previews['training-tiles-train'],'A training tile is kept and outlined on the image');
  await tile('validate',270,270);
  await until(()=>/1 training · 1 validation/.test(tools().textContent));
  assert(/1 training · 1 validation/.test(tools().textContent) && state.previews['training-tiles-validate'],'A validation tile is kept and outlined');
  tools().querySelector('[aria-label="Show the corrected cells in the tiles"]').click();
  await until(()=>state.previews['training-tile-cells']?.outlines?.length>0);
  tools().querySelector('[aria-label="Show the tiles on the image"]').click();
  await until(()=>!state.previews['training-tiles-train']);
  assert(state.previews['training-tile-cells'],'Tiles can be hidden, and the corrected cells inside them shown');
  tools().querySelector('[aria-label="Show the tiles on the image"]').click();
  tools().querySelector('[aria-label="Show the corrected cells in the tiles"]').click();
  await until(()=>state.previews['training-tiles-train']&&!state.previews['training-tile-cells']);
  button(tools(),'Validate').click();
  await until(()=>/Precision \d+%/.test(tools().textContent));
  assert(/\d+ missed, over 1 validation tile/.test(tools().textContent),'Validate compares the prediction with the corrected cells');
  set('Epochs','1');
  button(tools(),'Train').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  await until(async()=>{const job=(await jobs()).find(item=>item.kind==='train_cellpose');return job && /done|failed/.test(job.status);});
  const job=(await jobs()).find(item=>item.kind==='train_cellpose');
  assert(job.status==='done' && job.result.name==='browser_check_v1' && job.result.validation_tiles===1,'Train makes version 1 as a job: '+job.message);
  await until(()=>/browser_check_v1/.test(tools().textContent) && button(tools(),'Use'));
  button(tools(),'Use').click();
  await until(()=>/is now Segmentation.s model/.test(tools().textContent));
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/segmentation_settings')).json();
  assert(saved.values.tab4_model_type==='custom' && saved.values.tab4_custom_model.endsWith('browser_check_v1'),'Use makes Segmentation use the version: '+JSON.stringify({type:saved.values.tab4_model_type,path:saved.values.tab4_custom_model}));
  // The scores table: the base model and version 1 on the same validation tile.
  button(tools(),'Score all versions').click();
  await until(()=>tools().querySelectorAll('.studio-scores tr').length===3 || /stopped|rror/.test(tools().textContent));
  const rows=Array.from(tools().querySelectorAll('.studio-scores tr')).map(tr=>tr.firstChild.textContent);
  assert(rows[0]==='Version' && /^nuclei/.test(rows[1]) && /^browser_check_v1/.test(rows[2]) && /1 validation tile/.test(tools().textContent),'Score all versions lists each version on the same tiles: '+rows.join(' | '));
  set('Tile role','train');
  button(tools(),'Place a tile').click();await delay(50);
  viewer.raiseEvent('canvas-click',{quick:true,position:at(200,130)});
  await until(()=>/found \d+ cells|rror/.test(tools().textContent));
  assert(/browser_check_v1 found/.test(tools().textContent),'The next tile is predicted by the version just trained');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,ap:[job.result.ap_before,job.result.ap_after]})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-700)));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=600)
