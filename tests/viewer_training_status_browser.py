"""Training status follows jobs across navigation, completion and failure in Firefox."""
from viewer_threshold_browser import main
from viewer_missing_markers_browser import setup as base_setup


def setup(root):
    from starlette.responses import JSONResponse
    from spacemap_gui.studio import jobs
    from spacemap_gui.viewer import api
    base_setup(root)
    history = []
    control = {'reject': False}

    def start(project, kind, settings):
        if control['reject']:
            raise ValueError('Fixture cannot start')
        session = settings['training_session']
        job = {'id':f'status_{len(history)}', 'kind':kind, 'status':'running',
               'model_name':f"{session['name']}_v{len(session['versions']) + 1}",
               'created':'2026-10-24T00:00:00', 'message':'Reading training tiles…'}
        history.insert(0, job)
        return job

    async def change(request):
        value = await request.json()
        if 'reject' in value:
            control['reject'] = value.pop('reject')
        history[0].update(value)
        if value.get('status') == 'done':
            history[0]['result'] = {'name':history[0]['model_name']}
        return JSONResponse({'ok':True})

    jobs.start = start
    jobs.recent = lambda *args, **kwargs: history
    api.app.add_route('/studio/browser-check/test-training-status', change, methods=['POST'])


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const button=text=>[...document.querySelectorAll('#studio-tools button')].find(b=>b.textContent===text);
const status=()=>document.querySelector('.studio-training-run');
const change=body=>fetch('/spacemap-viewer/api/studio/browser-check/test-training-status',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 await until(()=>button('Train')&&!button('Train').disabled);
 assert(status().hidden,'No empty status line before a training job');
 button('Train').click();
 await until(()=>status().textContent.includes('reuse_browser_v3 in the background'));
 assert(status().dataset.phase==='running'&&getComputedStyle(status()).color==='rgb(244, 174, 98)','Running training is orange and identifies the version');
 const controls=document.querySelector('.studio-training-actions').getBoundingClientRect();
 assert(status().getBoundingClientRect().top-controls.bottom<12&&button('Train').disabled,'Status is directly under tile/Train controls and blocks duplicate training');
 await change({message:'Scoring held-out tiles…'});
 await until(()=>status().title==='Scoring held-out tiles…');
 assert(status().dataset.phase==='running','Scoring phase remains in progress, not falsely complete');
 document.querySelector('[data-step=tab4]').click();await delay(100);document.querySelector('[data-step=tab5]').click();
 await until(()=>status()?.textContent.includes('reuse_browser_v3 in the background'));
 assert(status().dataset.phase==='running'&&button('Train').disabled,'Running version and state restore after navigation');
 await change({status:'done',message:'Saved model'});
 await until(()=>status().dataset.phase==='done');
 assert(getComputedStyle(status()).color==='rgb(123, 216, 143)'&&status().textContent.includes('Training complete · reuse_browser_v3'),'Completed training turns green with the saved model name');
 await until(()=>!button('Train').disabled);
 button('Train').click();await until(()=>status().textContent.includes('in the background'));
 await change({status:'cancelled',message:'Stopped by user'});
 await until(()=>status().dataset.phase==='cancelled');
 assert(status().textContent.includes('cancelled')&&getComputedStyle(status()).color!=='rgb(123, 216, 143)','Cancelled runs do not appear successful');
 await until(()=>!button('Train').disabled);button('Train').click();
 await until(()=>status().textContent.includes('in the background'));
 await change({status:'failed',message:'Fixture worker failure'});
 await until(()=>status().dataset.phase==='failed');
 assert(status().textContent.includes('Fixture worker failure'),'Worker failure stays visible with its reason');
 await change({status:'done',message:'Saved model',reject:true});
 await until(()=>status().dataset.phase==='done'&&!button('Train').disabled);
 button('Train').click();await until(()=>status().textContent.includes('Fixture cannot start'));
 await delay(3500);
 assert(status().dataset.phase==='failed'&&!button('Train').disabled,'A rejected launch stays red after polling an older completed job');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=150, setup=setup)
