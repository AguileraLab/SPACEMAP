"""Shared Train / Predict switches, search selection, persistence and label safety."""
from viewer_threshold_browser import main


def setup(root):
    import os
    import pandas as pd
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap import dataio
    from spacemap_gui.studio import reference_models, reference_setup, tasks
    from spacemap_gui.viewer import api
    table = root/'cells.csv'
    frame = pd.read_csv(table)
    frame['phenotype_ref1'] = ['A']*15 + ['Others']*10 + ['B']*15
    frame.to_csv(table, index=False); dataio.ensure_fast_copy(table, force=True)
    os.environ['SPACEMAP_MODELS_DIR'] = str(root/'models')
    settings = {'project_root':str(root), 'table':str(table), 'roi_column':'batch_id', 'label_column':'label',
                'markers':['M0','M1'], 'training_column':'phenotype_ref1', 'checked_only':False}
    settings['revision'] = tasks.source_revision(settings)
    reference_models.train(settings, root/'fit', lambda *a: None)
    original = table.read_bytes()
    async def inspect(request):
        return JSONResponse({**reference_setup.phenotype_policy(root), 'unchanged':original == table.read_bytes(),
                             'labels':dataio.load(table).phenotype_ref1.tolist()})
    api.app.router.routes.append(Route('/phenotype-policy-test', inspect))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const read=async()=>await(await fetch('/spacemap-viewer/api/phenotype-policy-test')).json();
const change=(control,value)=>{control.value=value;control.dispatchEvent(new Event('change'));};
const get=(holder,label)=>holder.querySelector('[aria-label="'+label+'"]');
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const image=s.layers.find(l=>l.type==='image').token;
 const originalFetch=window.fetch;let releaseOldSetup=null, delayed=false;
 window.fetch=(url,options)=>{
   const response=originalFetch(url,options);
   if(!delayed&&String(url).endsWith('/reference_setup')&&(!options?.method||options.method==='GET')){
     delayed=true;return response.then(value=>new Promise(resolve=>{releaseOldSetup=()=>resolve(value);}));
   }return response;
 };
 change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
 await until(()=>document.querySelector('.reference-model-training'));
 document.querySelector('.reference-model-training').open=true;
 const switches=document.querySelector('.manual-reference-card .reference-phenotype-switches');switches.open=true;
 await until(()=>get(switches,'Train Others')&&!get(switches,'Train Others').disabled);
 assert(get(switches,'Train Others').checked&&get(switches,'Predict Others').checked,'Existing models start with unchanged class switches');
 get(switches,'Train Others').click();
 await until(async()=>(await read()).excluded_training_phenotypes.includes('Others')&&!get(switches,'Predict Others').disabled);
 await until(()=>releaseOldSetup);releaseOldSetup();await delay(150);window.fetch=originalFetch;
 assert(!get(switches,'Train Others').checked,'A delayed older model reply cannot restore a switched-off training class');
 assert(get(switches,'Predict Others').checked,'Train and Predict are independent switches');
 get(switches,'Predict Others').click();
 await until(async()=>(await read()).excluded_prediction_phenotypes.includes('Others')&&!get(switches,'Predict Others').disabled);
 const before=await read();
 assert(before.unchanged&&before.labels.filter(v=>v==='Others').length===10,'Switching a class off preserves its manual labels and dataset');
 const search=get(switches,'Find model phenotype');search.value='Others';search.dispatchEvent(new Event('input'));
 await until(()=>switches.querySelectorAll('tbody tr').length===1);
 get(switches,'Train all matching phenotypes').click();
 await until(async()=>!(await read()).excluded_training_phenotypes.includes('Others')&&!get(switches,'Train all matching phenotypes').disabled);
 assert(!(await read()).excluded_training_phenotypes.includes('B')&&(await read()).excluded_prediction_phenotypes.includes('Others'),'Select-all only changes names in the search and its chosen action');
 get(switches,'Train all matching phenotypes').click();
 await until(async()=>(await read()).excluded_training_phenotypes.includes('Others')&&!get(switches,'Train all matching phenotypes').disabled);
 search.value='';search.dispatchEvent(new Event('input'));
 assert(get(switches,'Train A').checked&&!get(switches,'Train Others').checked,'Re-enabling preserves unrelated class choices');
 const header=switches.querySelector('thead').getBoundingClientRect();
 assert(header.height<65,'Train and Predict share one compact table row');
 change(document.querySelector('[aria-label="Phenotyping method"]'),'mr_ml');
 await until(()=>document.querySelector('[aria-label="MR-ML workflow"]'));
 change(document.querySelector('[aria-label="MR-ML workflow"]'),'saved_model');
 const model=document.querySelector('[aria-label="Saved reference model"]');
 await until(()=>model.options.length>1);change(model,model.options[1].value);
 await until(()=>document.querySelector('#studio-training .reference-phenotype-switches'));
 const second=document.querySelector('#studio-training .reference-phenotype-switches');
 second.parentElement.open=true;second.open=true;
 await until(()=>get(second,'Predict Others')&&!get(second,'Predict Others').disabled);
 assert(!get(second,'Predict Others').checked&&!get(second,'Train Others').checked,'MR-ML uses the same saved switches as Manual reference');
 get(second,'Predict Others').click();
 await until(async()=>!(await read()).excluded_prediction_phenotypes.includes('Others')&&!get(second,'Predict Others').disabled);
 assert((await read()).excluded_training_phenotypes.includes('Others'),'Prediction can be re-enabled without adding the class to training');
 change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
 await until(()=>document.querySelector('.manual-reference-card .reference-phenotype-switches'));
 const restored=document.querySelector('.manual-reference-card .reference-phenotype-switches');
 await until(()=>get(restored,'Predict Others')&&!get(restored,'Predict Others').disabled);
 assert(get(restored,'Predict Others').checked&&!get(restored,'Train Others').checked,'Switches persist after navigating back');
 assert(s.layers.find(l=>l.type==='image').token===image&&(await read()).unchanged,'Changing model policy preserves the open image and original file');
 document.querySelector('.reference-model-training').open=true;restored.open=true;
 restored.scrollIntoView({block:'center'});await delay(250);
 fetch('/result?ok='+checks);
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup, screenshot=True, capture='/tmp/spacemap-phenotype-policy.png')
