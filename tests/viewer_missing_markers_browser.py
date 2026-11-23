"""Missing-marker option, selected/all submissions and saved validation in Firefox."""
import json
from pathlib import Path
import numpy as np
from viewer_threshold_browser import main
from viewer_studio_retraining_browser import setup as base_setup


def setup(root):
    from test_studio_artifacts import marker_image
    from spacemap_gui.studio import jobs, training
    marker_image(root/'roi.ome.tiff', np.full((2, 400, 400), 60, np.uint16), ['DAPI', 'HE_Red'])
    base_setup(root)
    config = json.loads((root/f'{root.name}.json').read_text())
    config['tab_configurations']['tab5_segmentation']['tab4_last_nuclear_names'] = ['DAPI', 'HE_Red']
    (root/f'{root.name}.json').write_text(json.dumps(config))
    source, models = training.folders(root)
    path = models/'models/reuse_browser_v1'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fixture weights')
    scores = {mode: {'before': 0.5, 'after': 0.7} for mode in ['full', 'without_he', 'fewer_markers', 'dapi_only']}
    note = {'name': path.name, 'path': str(path), 'missing_marker_training': True,
            'required_nuclear': 'DAPI', 'validation_tiles': 1, 'subset_validation': scores}
    Path(str(path)+'.json').write_text(json.dumps(note))
    session = training.session(root)
    older = {**note, 'name': 'reuse_browser_v0', 'path': str(path.with_name('reuse_browser_v0')),
             'subset_validation': {'full': {'before': None, 'after': 0.6}}}
    session.update(current=str(path), versions=[older, note])
    training._write(source/'session.json', session)
    history = []
    def start(project, kind, settings):
        job = {'id': f'fixture_{len(history)}', 'kind': kind, 'status': 'done',
               'created': '2026-10-23T00:00:00', 'message': 'Fixture submission', 'request': settings}
        history.insert(0, job)
        return job
    jobs.start = start
    jobs.recent = lambda *args, **kwargs: history


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const input=label=>document.querySelector('[aria-label="'+label+'"]');
const button=text=>[...document.querySelectorAll('#studio-tools button')].find(b=>b.textContent===text);
const change=(el,value)=>{el.value=value;el.dispatchEvent(new Event('change'));};
const responses=[];const originalFetch=window.fetch;
window.fetch=async(url,opts)=>{const result=await originalFetch(url,opts);
 if(String(url).endsWith('/training/retrain'))responses.push(await result.clone().json());return result;};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 await until(()=>input('Training input mode')&&input('Find training tiles')&&input('Nucleus marker DAPI'));
 const option=input('Training input mode');
 assert(option.value==='fixed','Fixed markers remains the default');
 assert(!document.querySelector('.studio-training-validation').open&&!document.querySelector('.studio-model-history').open,'Validation and model history start folded');
 assert([...document.querySelectorAll('[data-training-action]')].length===1&&!button('Retrain selected'),'Only one training action is offered');
 change(option,'optional');assert(option.value==='optional','Missing-marker mode has an explicit descriptive choice');
 document.querySelector('.studio-tile-list').open=true;
 input('Select all matching training tiles').click();
 [...document.querySelectorAll('[aria-label^="Retrain tile "]')].find(t=>t.checked).click();
 assert(input('Training tiles').value==='selected','Tile selection updates the visible training scope');
 button('Train').click();await until(()=>responses.length===1);
 let request=responses[0].job?.request;
 assert(request?.missing_markers&&request.tiles.filter(t=>t.role==='train').length===2,'Selected retraining sends the missing-marker option and selected tiles');
 assert(request.tiles.filter(t=>t.role==='validate').length===1&&request.input_settings.tab4_last_nuclear_names.includes('DAPI'),'DAPI settings snapshotted; validation remains separate');
 await until(()=>!button('Train').disabled);
 change(input('Training tiles'),'all');
 change(option,'fixed');
 button('Train').click();await until(()=>responses.length===2);
 request=responses[1].job?.request;
 assert(!request?.missing_markers&&request.refresh_inputs&&request.tiles.filter(t=>t.role==='train').length===3,'All-tile fixed-marker training also refreshes current inputs');
 assert(request.training_session.current.endsWith('reuse_browser_v1'),'Both actions continue the latest Training model');
 document.querySelector('.studio-training-validation').open=true;
 let scores=document.querySelector('.studio-subset-scores');
 assert(scores&&input('Validation model').value.endsWith('reuse_browser_v1'),'Validation defaults to the latest trained model');
 const rows=[...scores.querySelectorAll('tr')];
 assert(rows.length===5&&/Without H&E/.test(scores.textContent)&&/DAPI only/.test(scores.textContent),'All four validation scenarios displayed separately');
 assert(rows.slice(1).every(r=>r.cells[1].textContent==='0.50'&&r.cells[2].textContent==='0.70'),'Before/after scores displayed correctly');
 assert(rows[0].textContent.includes('Starting model')&&rows[0].textContent.includes('Trained model'),'Score columns identify the models being compared');
 change(input('Validation model'),input('Validation model').options[1].value);
 assert(document.querySelectorAll('.studio-subset-scores').length===1&&document.querySelector('.studio-subset-scores td:nth-child(2)').textContent==='—','Only the chosen version is shown, including missing historical scores');
 input('Nucleus marker HE_Red').click();await delay(250);
 assert(input('Validation model').value.endsWith('reuse_browser_v0')&&document.querySelector('.studio-subset-scores td').getClientRects().length,'Validation selection and expansion survive settings redraw');
 change(input('Validation model'),input('Validation model').options[0].value);
 // Visible controls remain contained in a narrow dock.
 document.querySelector('#app').style.setProperty('--studio-tools','330px');await delay(100);
 const panel=document.querySelector('.studio-training');
 assert([...panel.querySelectorAll('.studio-training-actions > *, .studio-training-choices select')].filter(el=>el.getClientRects().length).every(el=>el.getBoundingClientRect().right<=panel.getBoundingClientRect().right+1),'Tile and training controls wrap inside a narrow dock');
 document.querySelector('#app').style.setProperty('--studio-tools','620px');
 document.querySelector('.studio-tile-list').open=false;
 document.querySelector('.studio-training').scrollIntoView({block:'start'});await delay(100);
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-1000)));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
