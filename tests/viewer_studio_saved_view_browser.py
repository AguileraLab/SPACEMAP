"""Revisit through Studio's actual Annotation image rows, not direct init calls."""
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import tifffile
    from spacemap.viewer import display

    names = ['DAPI', 'CD3', 'E-cadherin']
    for name in ('first', 'second', 'unseen'):
        data = np.arange(3 * 400 * 400, dtype=np.uint16).reshape(3, 400, 400)
        tifffile.imwrite(root / f'{name}.ome.tiff', data, metadata={'axes': 'CYX',
            'PhysicalSizeX': .51, 'PhysicalSizeY': .51,
            'Channel': {'Name': names, 'Color': [65535, -16776961, -16711681]}})
        path = root / f'outputs/segmentation/{name}.ome_cell_boundaries.geojson'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((root / 'cells.geojson').read_text())
    sets = root / 'outputs/viewer/display/marker_sets.json'
    display.save_set(sets, 'Codex', [{'name': name, 'color': '#123456', 'low': 125.25, 'high': 45000} for name in names])


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=Number(sessionStorage.getItem('studio-view-checks')||0);
const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools'),panel=()=>document.querySelector('#panel');
 const img=()=>s.layers.find(l=>l.id==='image'),cells=()=>s.layers.find(l=>l.id==='cells');
 const cellBox=()=>panel().querySelector('[data-cell-layer=cells] > summary input[type=checkbox]');
 const chosen=()=>panel().querySelector('.sets select');
 const channelState=()=>img().channels.map(c=>({name:c.name,color:c.color,visible:c.visible,low:c.low,high:c.high}));
 const goto=async name=>{
  await until(()=>tools().querySelector('tr[data-image="'+name+'.ome.tiff"] button.studio-item-name'));
  tools().querySelector('tr[data-image="'+name+'.ome.tiff"] button.studio-item-name').click();
  await until(()=>!s.commandBusy&&s.imagesReady&&img()?.name===name+'.ome.tiff'&&cells()&&chosen());
  await delay(80);
 };
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));
 document.querySelector('[data-step=tab4]').click();
 if(sessionStorage.getItem('studio-view-phase')==='reload'){
  await until(()=>!s.commandBusy&&s.imagesReady&&img()?.name==='first.ome.tiff'&&cells());
  assert(!cells().visible,'Polling the saved Studio command after a real reload keeps Cells off');
  assert(JSON.stringify(channelState())===sessionStorage.getItem('studio-view-channels'),'Session polling after cache-cleared reload preserves chosen marker styles');
  await until(()=>chosen()?.value==='Codex');
  assert(chosen().value==='Codex','The marker-set name restores without reapplying its saved colours');
  await goto('second');await goto('first');
  assert(!cells().visible&&JSON.stringify(channelState())===sessionStorage.getItem('studio-view-channels'),'Repeated actual row navigation after reload keeps the complete image display');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));return;
 }
 await goto('first');
 assert(!!cells()&&!!cellBox(),'The reproduction uses a catalogued image with an actual Cells layer');
 await until(()=>Array.from(chosen().options).some(o=>o.value==='Codex'));
 chosen().value='Codex';chosen().dispatchEvent(new Event('change'));
 await until(()=>img().channels.every(c=>c.visible&&c.color==='#123456'));
 button(panel(),'Colours from the image').click();
 if(cellBox().checked)cellBox().click();
 img().inputs[1].low.value='377.375';img().inputs[1].low.dispatchEvent(new Event('change'));
 const expected=JSON.stringify(channelState());sessionStorage.setItem('studio-view-channels',expected);
 assert(!cells().visible&&img().channels.every(c=>c.color===img().info.colors[c.index]),'Original colours, all selected markers and Cells off match the reported starting view');
 s.viewer.viewport.zoomTo(2.5,null,true);s.viewer.viewport.panTo(new OpenSeadragon.Point(.4,.45),true);saveView();
 await goto('second');await goto('first');
 assert(!cells().visible&&!cellBox().checked,'Returning through the Annotation image list keeps Cells off');
 assert(JSON.stringify(channelState())===expected,'Returning through the image list keeps original colours, checked markers and precise thresholds');
 await until(()=>chosen().value==='Codex');assert(chosen().value==='Codex','The chosen marker-set name survives revisiting the image');
 assert(Math.abs(s.viewer.viewport.getZoom()-2.5)<.05,'Row navigation keeps the saved zoom');
 await goto('first');
 assert(!cells().visible&&JSON.stringify(channelState())===expected,'Clicking the already-open image does not reapply defaults');
 await flushViewerSettings();
 const saved=(await getJSON('/workspace/'+s.spec.workspace+'/view')).view;
 assert(saved.layers.cells.visible===false&&JSON.stringify(saved.layers.image.channels.map(c=>({name:c.name,color:c.color,visible:c.visible,low:c.low,high:c.high})))===expected,'The project JSON is not overwritten by navigation defaults');
 await goto('unseen');
 assert(img().channels.filter(c=>c.visible).map(c=>c.name).join(',')==='DAPI'&&s.layers.filter(l=>l.id!=='image'&&!l.added).every(l=>!l.visible)&&s.layers.filter(l=>l.added).every(l=>l.visible),'A never-opened image keeps DAPI only and Cells and other overlays off; the step\'s own layers (tissue, artifacts, regions) show');
 await goto('first');
 // Explicit channel commands must still execute and persist; opening alone must not.
 const changed=await runCommand('show_markers',{markers:['CD3'],replace:true});
 assert(changed.status==='done'&&img().channels.filter(c=>c.visible).map(c=>c.name).join(',')==='CD3','An explicit Show markers command still changes the requested channels');
 await goto('second');await goto('first');
 assert(!cells().visible&&img().channels.filter(c=>c.visible).map(c=>c.name).join(',')==='CD3','An explicit channel change persists without resetting unrelated layers');
 await runCommand('show_markers',{markers:['CD3'],replace:true});
 // A later manual change must win over that historical command on reload.
 img().channels.forEach(c=>c.visible=true);remember(img());refreshLayer(img());
 sessionStorage.setItem('studio-view-channels',JSON.stringify(channelState()));
 await flushViewerSettings();
 sessionStorage.setItem('studio-view-phase','reload');sessionStorage.setItem('studio-view-checks',String(checks));
 window.addEventListener('pagehide',()=>localStorage.clear());localStorage.clear();location.reload();
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=210, screenshot=False)
