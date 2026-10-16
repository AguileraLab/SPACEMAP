"""End-to-end TLS candidates tab on a disposable image/cohort."""
import pandas as pd
from spacemap import dataio
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import tifffile
    table = root / "cells.csv"
    frame = pd.read_csv(table)
    frame["phenotype_ml_consensus"] = ["B", "CD4"] * 20
    frame["tissue_roi"] = [1 if i % 8 < 4 else 2 for i in range(40)]
    second = frame.copy()
    second["batch_id"], second["tissue_roi"] = "roi2", 1
    second["cell_id"] += 40
    pd.concat([frame, second], ignore_index=True).to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    tifffile.imwrite(root / "roi2.ome.tiff", np.zeros((2, 400, 400), dtype=np.uint16),
                     metadata={"axes": "CYX", "PhysicalSizeX": .6, "PhysicalSizeY": .6,
                               "Channel": {"Name": ["M0", "M1"]}})
    for name in ("roi", "roi2"):
        outlines = root / f"outputs/segmentation/{name}.ome_cell_boundaries.geojson"
        outlines.parent.mkdir(parents=True, exist_ok=True)
        outlines.write_text((root / "cells.geojson").read_text())


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(ok,message)=>{if(!ok)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const button=(parent,text)=>Array.from(parent.querySelectorAll('button')).find(b=>b.textContent===text);
setTimeout(async()=>{try{
 const state=window.__state, tools=()=>document.querySelector('#studio-tools');
 const base='/spacemap-viewer/api/studio/browser-check';
 const jobs=async()=>(await(await fetch(base+'/jobs')).json()).jobs;
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 document.querySelector('[data-step=tab8]').click();
 await until(()=>button(tools(),'TLS candidates'));
 button(tools(),'TLS candidates').click();
 await until(()=>tools().querySelector('[aria-label="TLS maximum gap"]'));
 const input=name=>tools().querySelector('[aria-label="'+name+'"]');
 const set=(name,value)=>{const el=input(name);el.value=String(value);el.dispatchEvent(new Event('change'));};
 assert(input('TLS tissue region').value==='tissue_roi','Defaults to tissue-region grouping');
 assert(input('TLS output prefix').value==='TLS','Expected output column names');
 assert(!input('Overwrite TLS results').checked,'Overwrite is explicit and off by default');
 assert(button(tools().querySelector('.studio-method'),'Save'),'Settings have the same header Save control as other panels');
 assert(tools().textContent.includes('TLS_id')&&tools().textContent.includes('TLS_distance'),'All output columns are disclosed compactly');
 await until(()=>input('TLS B cells selector')&&input('TLS MPP').value==='0.5');
 const b=input('TLS B cells selector'),t=input('TLS T cells selector');
 assert(!b.open&&!t.open,'Both phenotype pickers start collapsed');
 b.querySelector('summary').click();await until(()=>b.open);
 assert(input('B cells: B').checked&&b.querySelector('summary').textContent==='B','B selection is visible in the compact summary');
 t.querySelector('summary').click();await until(()=>t.open&&!b.open);
 const search=input('Find TLS T cells');search.value='CD4';search.dispatchEvent(new Event('input'));
 assert([...t.querySelectorAll('.threshold-checks label')].filter(l=>!l.hidden).map(l=>l.textContent).join()==='CD4','Dropdown search narrows phenotype choices');
 search.value='';search.dispatchEvent(new Event('input'));
 input('T cells: CD8').click();input('T cells: Treg').click();
 button(t,'Done').click();assert(!t.open&&t.querySelector('summary').textContent==='CD4','Done closes the picker and updates selected names');
 assert(input('TLS MPP').readOnly&&input('TLS MPP source').textContent.startsWith('Image'),'Auto MPP reads the open image metadata');
 assert(input('TLS maximum gap').closest('.field').nextElementSibling.contains(input('TLS MPP')),'MPP is immediately below Maximum gap');
 const scales=(await(await fetch(base+'/spatial/tls_mpp',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json()).images;
 assert(scales.roi.mpp===.5&&scales.roi2.mpp===.6,'Calibration is resolved separately for each project image');
 set('TLS MPP mode','manual');set('TLS MPP',0);
 button(tools().querySelector('.studio-method'),'Save').click();
 await until(()=>document.querySelector('.studio-runbar').textContent.includes('Enter a positive MPP'));
 assert(true,'Invalid manual MPP produces a readable error');
 set('TLS MPP mode','auto');
 set('TLS maximum gap',60);set('TLS minimum lymphocytes',20);set('TLS minimum B cells',5);set('TLS minimum T cells',5);
 button(tools().querySelector('.studio-method'),'Save').click();
 await until(()=>document.querySelector('.studio-runbar').textContent.includes('Settings saved.'));
 const saved=await(await fetch(base+'/spatial/tls_settings')).json();
 assert(saved.values.max_gap===60&&saved.values.min_b===5&&saved.values.pixel_um===0,'Settings persist with automatic per-image calibration and separate cell minima');
 for(const name of ['roi','roi2']){const tick=input('Pick '+name);if(!tick.checked)tick.click();}
 button(document.querySelector('.studio-runbar'),'Selected').click();
 let job;await until(async()=>{job=(await jobs()).find(j=>j.kind==='tls_candidates');return job&&['done','failed'].includes(job.status);});
 assert(job.status==='done','Job completes: '+job.message);
 assert(job.result.total===3&&job.result.counts.roi===2&&job.result.counts.roi2===1,'Each image has its own candidate count');
 assert(job.result.columns.join()==='TLS,TLS_id,TLS_size,TLS_distance','All four columns publish together');
 assert(job.result.structures[0].b_cells===10&&job.result.structures[0].t_cells===10,'B and T counts derive from selected phenotypes');
 await until(()=>tools().querySelector('tr[data-roi="roi"]')?.textContent.includes('2 TLS candidates'));
 assert(true,'Top image list shows TLS region count');
 assert(getComputedStyle(input('Show TLS in roi')).cursor==='pointer','TLS count badges use the pointer cursor');
 assert(input('TLS edge trim')?.value==='0','Automatic edge trim defaults to the original boundary');
 await until(()=>button(tools(),'Review'));
 const review=button(tools(),'Review');
 const header=review.closest('table').querySelector('tr').textContent;
 assert(header.includes('Area')&&header.includes('B/T per mm²'),'Review table labels area and density units');
 await until(()=>state.previews?.['tls-candidates']?.outlines?.length===2);
 assert(!state.previews['tls-selected'],'All candidates show automatically without selecting a row');
 review.closest('tr').click();
 await until(()=>state.previews?.['tls-selected']?.outlines?.length===1);
 assert(state.previews['tls-candidates'].outlines.length===2&&state.previews['tls-candidates'].colour==='#00dcc8','Review highlights one TLS while retaining all boundaries');
 const cells=()=>state.layers.find(l=>l.type==='cells');
 await until(()=>cells().color_by==='TLS');
 assert(!cells().filters?.some(f=>f.col==='TLS_id'||f.col==='tissue_roi'),'Review never hides other structures using ID or region filters');
 await until(()=>state.viewer.viewport.getZoom()>1.1);
 assert(true,'Clicking a TLS zooms to its boundary');
 assert(true,'Review colours cells using TLS membership');
 const shapes=await(await fetch(base+'/spatial/tls_shapes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:job.id,image:'roi'})})).json();
 assert(shapes.features.length===2&&shapes.features.every(f=>f.properties.id===1),'Saved geometry carries IDs local to each tissue region');
 assert(shapes.features[0].geometry.type==='Polygon','Review and tags share a saved polygon boundary');
 const csv=await(await fetch(base+'/spatial/tls_download?job_id='+job.id+'&format=csv')).text();
 assert(csv.includes('area_um2')&&csv.includes('b_cells')&&csv.includes('density_per_mm2'),'CSV exports candidate measurements');
 const geo=await(await fetch(base+'/spatial/tls_download?job_id='+job.id+'&format=geojson')).json();
 assert(geo.features[0].properties.classification.name==='TLS candidate','GeoJSON is clearly labelled as candidate');
 input('TLS boundary visibility').click();assert(!state.previews['tls-candidates']&&!state.previews['tls-selected'],'Hide TLS clears all candidate and highlight boundaries');
 input('TLS boundary visibility').click();await until(()=>state.previews?.['tls-candidates']?.outlines?.length===2);
 assert(true,'Show TLS restores every boundary');
 input('TLS review image').value='roi2';input('TLS review image').dispatchEvent(new Event('change'));
 await until(()=>state.imagesReady&&state.cellsReady&&cells()?.roi==='roi2'&&state.previews?.['tls-candidates']?.outlines?.length===1);
 assert(input('TLS review image').value==='roi2'&&tools().querySelectorAll('tr[data-tls]').length===1,'Review dropdown opens its image and updates the TLS list');
 if(input('TLS MPP').value!=='0.6')fetch('/result?progress='+encodeURIComponent('Calibration pending: '+input('TLS MPP source').title));
 await until(()=>input('TLS MPP').value==='0.6');
 assert(input('TLS MPP').value==='0.6','MPP follows the newly opened image');
 tools().querySelector('tr[data-roi="roi"] button.studio-item-name').click();
 await until(()=>state.imagesReady&&state.cellsReady&&cells()?.roi==='roi'&&input('TLS review image').value==='roi'&&state.previews?.['tls-candidates']?.outlines?.length===2);
 assert(tools().querySelectorAll('tr[data-tls]').length===2&&input('TLS MPP').value==='0.5','Top image navigation automatically follows the image and calibration');
 button(tools(),'Neighbourhoods').click();await until(()=>input('Phenotype column'));
 input('Show TLS in roi2').click();
 await until(()=>input('TLS review image')?.value==='roi2'&&cells()?.roi==='roi2'&&state.previews?.['tls-candidates']?.outlines?.length===1);
 assert(true,'TLS count badge opens its image, activates TLS and previews every candidate');
 input('TLS boundary visibility').click();input('Show TLS in roi').click();
 await until(()=>input('TLS review image')?.value==='roi'&&cells()?.roi==='roi'&&state.previews?.['tls-candidates']?.outlines?.length===2);
 assert(input('TLS boundary visibility').textContent==='Hide TLS','Clicking a badge restores a hidden preview and synchronizes the list');
 button(document.querySelector('.studio-runbar'),'This image').click();
 await until(()=>document.querySelector('.studio-runbar').textContent.includes('already exists'));
 assert((await jobs()).filter(j=>j.kind==='tls_candidates').length===1,'Collision is rejected without overwrite');
 input('Overwrite TLS results').checked=true;
 input('TLS output prefix').value='TLS_test';input('TLS output prefix').dispatchEvent(new Event('input'));
 assert(!input('Overwrite TLS results').checked,'Changing the prefix clears overwrite');
 input('TLS output prefix').value='TLS';input('TLS output prefix').dispatchEvent(new Event('input'));
 input('Overwrite TLS results').checked=true;
 button(document.querySelector('.studio-runbar'),'This image').click();
 let rerun;await until(async()=>{rerun=(await jobs()).find(j=>j.kind==='tls_candidates'&&j.id!==job.id);return rerun&&['done','failed'].includes(rerun.status);});
 assert(rerun.status==='done'&&rerun.result.total===2,'Explicit overwrite succeeds: '+rerun.message);
 assert(!input('Overwrite TLS results').checked,'Overwrite clears after launch');
 button(tools(),'Neighbourhoods').click();await until(()=>input('Phenotype column'));
 button(tools(),'TLS candidates').click();await until(()=>button(tools(),'Review'));
 assert(input('TLS maximum gap').value==='60','Reopening restores settings and saved candidates');
 // Manual review uses the real Annotation brush and polygon tools, with a separate guarded save.
 button(tools(),'Review').click();
 button(tools(),'Edit boundaries').click();
 const editor=()=>state.layers.find(l=>l.tlsDraft);
 await until(()=>editor()?.features.length===1&&state.selected?.layer===editor());
 assert(editor().editable&&editor().regions,'TLS opens as an editable annotation draft for the selected tissue region');
 await until(()=>state.previews?.['tls-candidates']?.outlines.length===1);
 assert(true,'The other tissue region stays visible while editing');
 const snapshot=()=>JSON.stringify(editor().features), initialDraft=snapshot();
 const at=(x,y)=>state.viewer.viewport.pixelFromPoint(state.layers.find(l=>l.type==='image').item.imageToViewportCoordinates(x,y));
 const stroke=async(mode,from,to,radius)=>{
   state.brushRadius=radius;const originalEvent=mode==='add'?{shiftKey:true}:{altKey:true};
   state.viewer.raiseEvent('canvas-press',{position:at(...from),originalEvent});
   assert(!!state.brushStroke,mode+' modifier starts the TLS brush');
   state.viewer.raiseEvent('canvas-drag',{position:at(...to),originalEvent});
   state.viewer.raiseEvent('canvas-release',{position:at(...to),originalEvent});
   await until(()=>!state.brushBusy);
 };
 await stroke('add',[140,100],[220,100],15);
 const expanded=snapshot();assert(expanded!==initialDraft,'Shift grows the chosen TLS boundary');
 await stroke('erase',[80,-90],[80,280],8);
 assert(editor().features.length===2,'Alt cuts the draft into two pieces');
 button(tools(),'Undo').click();assert(snapshot()===expanded,'Undo restores the boundary before the cut');
 button(tools(),'Redo').click();assert(editor().features.length===2,'Redo restores the cut');
 await delay(550);button(tools(),'Merge').click();state.mergeGap=30;
 state.viewer.raiseEvent('canvas-click',{quick:true,position:at(20,100),originalEvent:{}});
 state.viewer.raiseEvent('canvas-click',{quick:true,position:at(120,100),originalEvent:{}});
 await until(()=>!state.brushBusy&&editor().features.length===1);
 assert(true,'Merge joins selected TLS pieces with the existing Annotation tool');
 const beforeTrim=snapshot();set('TLS draft trim',2);
 button(tools(),'Trim edge').click();await until(()=>!input('TLS draft trim').disabled&&snapshot()!==beforeTrim);
 assert(true,'Trim reduces an existing TLS draft in microns');
 button(tools(),'Undo').click();assert(snapshot()===beforeTrim,'Draft trim is undoable');
 tools().querySelector('tr[data-roi="roi2"] button.studio-item-name').click();
 await until(()=>cells()?.roi==='roi2'&&state.imagesReady&&button(tools(),'Resume edits'));
 assert(!editor(),'Changing images suspends the local editor');
 button(tools(),'Resume edits').click();
 await until(()=>cells()?.roi==='roi'&&editor());
 assert(JSON.stringify(editor().features.map(f=>f.geometry))===JSON.stringify(JSON.parse(beforeTrim).map(f=>f.geometry)),'Resume returns to the edited image and preserves draft geometry');
 button(tools(),'Save edits').click();
 let reviewed;await until(async()=>{reviewed=(await jobs()).find(j=>j.kind==='tls_candidates'&&j.id!==job.id&&j.id!==rerun.id&&['done','failed'].includes(j.status));return !!reviewed;});
 assert(reviewed.status==='done'&&reviewed.result.reviewed&&reviewed.result.parent_job===rerun.id,'Save publishes a manual review job with original-run provenance: '+reviewed.message);
 await until(()=>!editor()&&button(tools(),'Edit boundaries'));
 assert(reviewed.result.columns.join()==='TLS,TLS_id,TLS_size,TLS_distance','Manual saving updates all four dataset columns');
 assert(reviewed.result.structures.some(s=>s.reviewed)&&reviewed.result.structures.some(s=>!s.reviewed),'Only the edited tissue region is marked as manually reviewed');
 button(tools(),'Edit boundaries').click();await until(()=>editor());
 button(tools(),'Discard edits').click();await until(()=>!editor());
 assert(button(tools(),'Edit boundaries')&&!button(tools(),'Resume edits'),'Discard leaves the saved result and removes the draft');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent+'\n'+document.querySelector('.studio-runbar')?.textContent));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
