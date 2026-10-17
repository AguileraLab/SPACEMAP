"""Browser-check aligning a registration pair by eye: the H&E over the CODEX, scaled, turned, moved; right-click zoom.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_align_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message.slice(0,90)));};
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent('page error: '+event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab1]'));
  document.querySelector('[data-step=tab1]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('tr[data-sample="P1"] button.studio-item-name'));
  // First the CODEX alone (its name), then the row: the same image stays open and the H&E is added to it.
  tools().querySelector('tr[data-sample="P1"] td:nth-child(3) button').click();
  await until(()=>state.layers.some(layer=>layer.type==='image' && layer.item) && /Raw CODEX/.test(document.querySelector('#title')?.textContent || ''));
  tools().querySelector('tr[data-sample="P1"] td:nth-child(5)').click();  // the row, not a name
  const he=()=>state.layers.find(layer=>layer.id==='he' && layer.align);
  await until(()=>he() && he().item && state.layers.some(layer=>layer.type==='image' && !layer.align && layer.item));
  const base=state.layers.find(layer=>layer.type==='image' && !layer.align);
  await until(()=>Math.abs(he().item.getBounds(true).width/base.item.getBounds(true).width-0.25)<0.01);
  assert(true,'The H&E lies over the CODEX, scaled by the pixel sizes (200 px at 0.25 over 400 px at 0.5)');
  await until(()=>tools().querySelector('[aria-label="H&E turn in degrees"]'));
  assert(Array.from(tools().querySelectorAll('button')).some(button=>button.textContent==='Quick align'),'Quick align sits with the alignment controls');
  const degrees=tools().querySelector('[aria-label="H&E turn in degrees"]');degrees.value='30';degrees.dispatchEvent(new Event('change'));
  await until(()=>Math.abs(he().item.getRotation()-30)<0.01);
  assert(true,'Turning the H&E turns it on the image');
  tools().querySelector('[aria-label="Move the H&E"]').click();
  const before=he().item.getBounds(true).x;
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
  await until(()=>he().item.getBounds(true).x>before);
  assert(he().align.x===10,'With Move H&E on, the arrow keys move the H&E');
  tools().querySelector('[aria-label="Scale the H&E by the pixel sizes"]').click();
  await until(()=>Math.abs(he().item.getBounds(true).width/base.item.getBounds(true).width-0.5)<0.01);
  assert(true,'Unticking Scale shows the H&E at its own pixel size');
  // Right click on the image: a menu, and zoom to a percentage (100%: one image pixel per screen pixel).
  const canvas=document.querySelector('#osd .openseadragon-canvas');const box=canvas.getBoundingClientRect();
  canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:box.left+box.width/2,clientY:box.top+box.height/2}));
  await until(()=>document.querySelector('#ctxmenu'));
  const zoom=Array.from(document.querySelectorAll('#ctxmenu button')).find(button=>button.textContent==='100% (native)');
  assert(zoom && !zoom.disabled,'Right click shows the menu with the zoom percentages');
  zoom.click();
  await until(()=>{const imageZoom=base.item.viewportToImageZoom(state.viewer.viewport.getZoom(true));return Math.abs(imageZoom-1)<0.05;});
  assert(!document.querySelector('#ctxmenu'),'100% shows one image pixel per screen pixel, and the menu closes');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(-600)));}
},100);
</script>
"""


def setup(root):
    """A 400 px CODEX at 0.5 µm/px and a 200 px RGB H&E at 0.25 µm/px, named as a pair."""
    import numpy as np
    import tifffile

    for folder in ("data/codex", "data/he"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    codex = np.zeros((2, 400, 400), np.uint16)
    codex[0, 100:300, 100:300] = 3000
    tifffile.imwrite(root / "data/codex/P1_codex.ome.tif", codex, metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                                                                          "Channel": {"Name": ["DAPI", "CD3"]}})
    he = np.full((200, 200, 3), 230, np.uint8)
    he[50:150, 50:150] = (120, 60, 150)
    tifffile.imwrite(root / "data/he/P1_he.ome.tif", he, photometric="rgb", metadata={"axes": "YXS", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25})


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
