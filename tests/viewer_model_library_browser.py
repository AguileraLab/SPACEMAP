"""Browser-check the model library in Studio's Project step: share a model, see it in the library, move the library.

Usage: PYTHONPATH=src:tests python tests/viewer_model_library_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab0]'));
  document.querySelector('[data-step=tab0]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Share demo_v1"]'));
  assert(/Segmentation \(Cellpose\)/.test(tools().textContent) && tools().querySelector('[aria-label="Model library folder"]').value.endsWith('/library'),
    'The Project step lists this project\'s Cellpose model and the library folder');
  tools().querySelector('[aria-label="Share demo_v1"]').click();
  await until(()=>/demo_v1 is in the library/.test(tools().textContent));
  assert(!tools().querySelector('[aria-label="Share demo_v1"]') && /In the library/.test(tools().textContent) && /In this project/.test(tools().textContent),
    'Sharing copies the model into the library and marks it on both sides');
  const folder=tools().querySelector('[aria-label="Model library folder"]');
  folder.value=folder.value.replace(/library$/,'lab_library');folder.dispatchEvent(new Event('change'));
  await until(()=>/Library folder changed/.test(tools().textContent));
  assert(tools().querySelector('[aria-label="Share demo_v1"]'),'Another library folder starts empty, so the model can be shared there too');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(0,800)));}
},100);
</script>
"""


def setup(root):
    """A Cellpose model trained in this project."""
    models = root / "outputs/segmentation/cellpose_models/models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "demo_v1").write_bytes(b"weights")


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup)
