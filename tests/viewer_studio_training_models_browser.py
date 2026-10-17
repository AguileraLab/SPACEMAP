"""Saved-weight selection in real Firefox; does not run training or require a GPU."""
import json
import os
from viewer_threshold_browser import main


def setup(root):
    models = root / 'data/seg_model'
    models.mkdir(parents=True)
    for name, age in (('studio_old_v9', 10), ('studio_recent_v26', 20)):
        path = models / name
        path.write_bytes(b'0' * 1_100_000)
        os.utime(path, (age, age))
    local = root / 'outputs/segmentation/cellpose_models/models/project_v3'
    local.parent.mkdir(parents=True)
    local.write_bytes(b'fixture weights')
    (root / f'{root.name}.json').write_text(json.dumps({'tab_configurations': {'tab5_segmentation': {
        'tab4_model_type': 'custom', 'tab4_custom_model': str(models / 'studio_recent_v26')}}}))


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const button=(text)=>[...document.querySelectorAll('#studio-tools button')].find(b=>b.textContent===text);
const input=label=>document.querySelector('[aria-label="'+label+'"]');
const change=(el,value)=>{el.value=value;el.dispatchEvent(new Event('change'));};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 await until(()=>input('Start from')&&input('Cellpose model'));
 let select=input('Start from');
 assert(select.value.endsWith('studio_recent_v26'),'Start from uses the saved selected model on first render');
 const folder=[...select.querySelector('optgroup[label="Models folder"]').children];
 assert(folder[0].value.endsWith('studio_recent_v26')&&folder[1].value.endsWith('studio_old_v9'),'Models folder versions are listed newest first');
 assert([...select.options].some(o=>o.value.endsWith('project_v3')),'Models trained in this project are also offered');
 assert([...select.options].some(o=>o.value==='nuclei'),'Built-in models remain available');
 change(input('Cellpose model'),'folder:studio_old_v9');
 await until(()=>input('Start from').value.endsWith('studio_old_v9'));
 assert(true,'Changing Segmentation model updates the starting model');
 change(input('Start from'),folder[0].value);input('Refresh starting models').click();await delay(500);
 assert(input('Start from').value.endsWith('studio_recent_v26'),'Refresh preserves an explicitly chosen starting model');
 change(input('Model name'),'improved_cells');button('Start').click();
 await until(()=>button('Start over'));
 let saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/training')).json();
 assert(saved.session.base.endsWith('studio_recent_v26')&&saved.session.current===saved.session.base,'Training session starts with the selected saved weights');
 button('Start over').click();await until(()=>input('Start from'));
 assert(input('Start from').value.endsWith('studio_old_v9'),'Start over uses the current segmentation model');
 button('Cancel').click();await until(()=>button('Start over'));
 saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/training')).json();
 assert(saved.session.name==='improved_cells','Cancelling a new session keeps the existing session');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
