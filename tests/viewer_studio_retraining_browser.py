"""Tile selection and refreshed retraining submission in Firefox; no real training job."""
import json
import shutil
from viewer_threshold_browser import main


def setup(root):
    from spacemap_gui.studio import jobs, segmentation, training
    settings = {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['M0'], 'tab4_gpu': False}
    training.start(root, 'reuse_browser', 'nuclei')
    other = root / 'second.ome.tiff'
    shutil.copyfile(root / 'roi.ome.tiff', other)
    for image, x, role in ((root / 'roi.ome.tiff', 0, 'train'), (root / 'roi.ome.tiff', 64, 'train'),
                           (other, 0, 'train'), (other, 64, 'validate')):
        training.save_example(root, image, ['M0', 'M1'], [x, 0, 64, 64],
            [[[x+5, 5], [x+25, 5], [x+25, 25], [x+5, 25]]], settings, role=role)
    (root / f'{root.name}.json').write_text(json.dumps({'tab_configurations': {'tab5_segmentation': settings}}))
    # Exercise the real endpoint and its snapshot without spawning Cellpose.
    jobs.start = lambda project, kind, settings: {'id': 'fixture', 'kind': kind, 'status': 'starting', 'request': settings}


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const input=label=>document.querySelector('[aria-label="'+label+'"]');
const button=text=>[...document.querySelectorAll('#studio-tools button')].find(b=>b.textContent===text);
const change=(el,v,type='change')=>{el.value=v;el.dispatchEvent(new Event(type));};
let submission, response;
const originalFetch=window.fetch;
window.fetch=async(url,opts)=>{
 const result=await originalFetch(url,opts);
 if(String(url).endsWith('/training/retrain')){submission=JSON.parse(opts.body);response=await result.clone().json();}
 return result;
};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 await until(()=>input('Find training tiles')&&input('Nucleus marker M1'));
 let list=document.querySelector('.studio-tile-list');
 assert(!list.open&&!input('Find training tiles').getClientRects().length,'Selection controls hidden until Tiles expands');
 assert(button('Train')&&!button('Retrain selected')&&input('Training tiles').value==='all','One Train action defaults to all training tiles');
 change(input('Training tiles'),'selected');
 assert(button('Train').disabled,'An empty selection cannot silently train all tiles');
 list.querySelector('summary').click();
 assert(input('Find training tiles').getClientRects().length&&button('Train').disabled,'Expanded header shows selection controls');
 const centre=el=>{const r=el.getBoundingClientRect();return r.top+r.height/2;};
 assert(input('Find training tiles').getBoundingClientRect().width<150&&Math.abs(centre(input('Select all matching training tiles'))-centre(input('Find training tiles')))<4,'Select-all and search share one compact controls row');
 assert(Math.abs(centre(button('Train'))-centre(button('Keep tile')))<4,'Train stays next to Keep tile on the action row');
 const checksForTiles=()=>[...document.querySelectorAll('[aria-label^="Retrain tile "]')];
 assert(checksForTiles().length===4&&checksForTiles().filter(t=>t.disabled).length===1,'Every tile has a checkbox; validation is held out');
 input('Select all matching training tiles').click();
 assert(list.open&&checksForTiles().filter(t=>t.checked).length===3&&!button('Train').disabled,'Select all selects training tiles across images without folding the list');
 change(input('Find training tiles'),'second','input');
 assert([...list.querySelectorAll('.studio-folder')].filter(r=>r.getClientRects().length).length===2,'Search displays matching image tiles');
 input('Select all matching training tiles').click();
 assert(checksForTiles().filter(t=>t.checked).length===2,'Unselect matching preserves selections for other images');
 change(input('Find training tiles'),'','input');
 assert(input('Select all matching training tiles').indeterminate,'Select-all shows partial selection');
 checksForTiles().find(t=>!t.disabled&&!t.checked).click();
 assert(input('Select all matching training tiles').checked,'Individual selection synchronizes select-all');
 // Reload the tile list through session controls; selection survives.
 button('New training').click();await until(()=>input('Start from'));
 assert(button('Train').disabled,'Cannot train while choosing a new training session');
 button('Cancel').click();await until(()=>button('New training'));
 assert(checksForTiles().filter(t=>t.checked).length===3&&!button('Train').disabled,'Selection survives reload and cancelled session change');
 input('Nucleus marker M1').click();
 change(input('Epochs'),'2');
 button('Train').click();
 await until(()=>response);
 assert(response.job&&submission.names.length===3&&submission.epochs===2&&!submission.missing_markers,'Train posts exactly the selected IDs, epochs and fixed-marker mode');
 const settings=response.job.request;
 assert(settings.refresh_inputs&&settings.input_settings.tab4_last_nuclear_names.includes('M1'),'Current marker edits saved before retraining snapshot');
 assert(settings.tiles.filter(t=>t.role==='train').length===3&&settings.tiles.filter(t=>t.role==='validate').length===1,'Validation stays separate in server snapshot');
 assert(settings.training_session.current==='nuclei','Continues the Training model');
 assert(settings.tiles.every(t=>t.image_path.startsWith('/')&&t.mask_stamp),'Source paths and corrected-mask versions recorded');
 assert(!/failed|error/i.test(document.querySelector('#studio-tools').textContent),'No UI error after submission');
 // Leave a compact expanded list visible for the screenshot.
 document.querySelector('.studio-tile-list').open=true;
 document.querySelector('.studio-examples').scrollIntoView({block:'center'});
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-900)));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
