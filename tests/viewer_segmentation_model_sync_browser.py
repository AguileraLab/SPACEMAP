"""Real model/settings APIs with disposable saved weights; no inference or research data."""
import json

from viewer_threshold_browser import main


def setup(root):
    from starlette.responses import JSONResponse
    from spacemap_gui.studio import training
    from spacemap_gui.viewer import api

    folder, models = training.folders(root)
    training.start(root, 'studio_20260927', 'nuclei')
    (models / 'models').mkdir(parents=True, exist_ok=True)

    def add_version(number):
        name = f'studio_20260927_v{number}'
        path = models / 'models' / name
        path.write_bytes(b'disposable catalogue fixture; not inference weights')
        meta = {'name': name, 'path': str(path), 'ap_after': .94, 'ap_before': .94,
                'created': f'2026-10-23T12:00:{number:02d}', 'training_tiles': 56, 'validation_tiles': 1}
        path.with_name(name + '.json').write_text(json.dumps(meta))
        session = training.session(root)
        session['current'] = str(path)
        session['versions'].append(meta)
        (folder / 'session.json').write_text(json.dumps(session))
        return path

    active = add_version(31)
    add_version(32)
    (root / f'{root.name}.json').write_text(json.dumps({'tab_configurations': {'tab5_segmentation': {
        'tab4_model_type': 'custom', 'tab4_custom_model': str(active)}}}))

    async def finish(request):
        add_version(33)
        return JSONResponse({'ok': True})

    api.app.add_route('/studio/browser-check/test-finish-training', finish, methods=['POST'])


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const input=label=>document.querySelector('[aria-label="'+label+'"]');
const change=(el,value)=>{el.value=value;el.dispatchEvent(new Event('change'));};
const base='/spacemap-viewer/api/studio/browser-check';
const id=n=>'project:studio_20260927_v'+n;
const use=n=>input('Use for segmentation studio_20260927_v'+n);
const active=n=>input('Active model studio_20260927_v'+n);
const settings=async()=>(await(await fetch(base+'/segmentation_settings')).json()).values;
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 await until(()=>input('Cellpose model')&&use(32));
 const history=document.querySelector('.studio-model-history');
 assert(input('Cellpose model').value===id(31)&&active(31).disabled,'Previously selected v31 remains the active segmentation model');
 assert(!history.open&&!input('Trained version'),'History starts folded and there is no second model selector');
 assert(document.querySelector('#studio-seg .field').textContent.startsWith('Active model'),'Active model is the first settings control');
 assert(document.querySelector('[data-training-model]').textContent.endsWith('v32'),'Training model is identified separately as a read-only value');
 assert(!document.querySelector('#studio-tools').textContent.includes('Place the next tile: this version predicts it'),'No repeated training explanation under the controls');
 history.open=true;
 use(32).click();
 await until(async()=>input('Cellpose model').value===id(32)&&active(32)?.disabled&&(await settings()).tab4_custom_model.endsWith('_v32'));
 assert(true,'Use for segmentation immediately updates the active selector, history and saved settings');
 change(input('Cellpose model'),id(31));
 await until(async()=>active(31)?.disabled&&(await settings()).tab4_custom_model.endsWith('_v31'));
 assert(use(32)&&document.querySelector('[data-training-model]').textContent.endsWith('v32'),'Selecting an older segmentation model updates history without restarting the training series');
 // Delay a model response, then choose a newer model and edit a numeric setting.
 const originalFetch=window.fetch;let release,held=false;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);if(String(url).endsWith('/segmentation_images')&&args?.method==='POST'&&!held){held=true;await new Promise(r=>release=r);}return response;};
 change(input('Cellpose model'),id(32));await until(()=>held);
 change(input('Cellpose model'),id(31));change(input('Flow threshold'),'0.6');release();
 await until(async()=>{const s=await settings();return s.tab4_custom_model.endsWith('_v31')&&s.tab4_flow===.6&&active(31)?.disabled;});
 window.fetch=originalFetch;
 assert(input('Cellpose model').value===id(31)&&input('Flow threshold').value==='0.6','Latest model and settings changes survive a delayed save');
 // A rejected activation must never display Active or claim success.
 window.fetch=(url,args)=>String(url).endsWith('/segmentation_images')&&args?.method==='POST'
   ?Promise.resolve(new Response(JSON.stringify({error:'Fixture activation rejected'}),{status:400,headers:{'Content-Type':'application/json'}}))
   :originalFetch(url,args);
 use(32).click();await until(()=>document.querySelector('#studio-tools').textContent.includes('Fixture activation rejected')&&active(31)?.disabled&&input('Cellpose model').value===id(31));
 window.fetch=originalFetch;
 assert(!active(32)&&(await settings()).tab4_custom_model.endsWith('_v31'),'Failed activation retains the actual saved model and reports the error');
 // A completed training job must publish a new choice without activation or navigation.
 await fetch(base+'/test-finish-training',{method:'POST'});
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);if(String(url).endsWith('/jobs')&&!args?.method){const data=await response.json();data.jobs.unshift({id:'fixture-completed-training',kind:'train_cellpose',status:'done',message:'Training complete',created:'2026-10-23T12:00:33',result:{name:'studio_20260927_v33'}});return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});}return response;};
 await until(()=>[...input('Cellpose model').options].some(o=>o.value===id(33))&&use(33));
 window.fetch=originalFetch;
 assert(input('Cellpose model').value===id(31)&&active(31)?.disabled,'A completed training job refreshes the model list without replacing the chosen model');
 assert(history.open&&document.querySelector('[data-training-model]').textContent.endsWith('v33'),'History stays open and the next training model updates');
 const form=document.querySelector('#studio-seg'), model=input('Cellpose model');
 assert(model.getBoundingClientRect().right<=form.getBoundingClientRect().right+1,'The active model control fits the Studio panel');
 document.querySelector('[data-step=tab4]').click();await delay(150);document.querySelector('[data-step=tab5]').click();
 await until(()=>active(31)&&use(33));
 assert(input('Cellpose model').value===id(31),'The active model survives navigation');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
