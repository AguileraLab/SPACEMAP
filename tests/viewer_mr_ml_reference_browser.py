"""Run reference-only MR-ML with morphology, named scores and CSV commit in Jobs."""
from viewer_threshold_browser import main


def setup(root):
    import json
    import os
    import pandas as pd
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap import dataio, pending_columns
    from spacemap_gui.viewer import api
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    frame['area'] = [10.] * 20 + [100.] * 20
    frame['eccentricity'] = [.1] * 20 + [.9] * 20
    frame.loc[[15,16,17,18,19,35,36,37,38,39], 'phenotype_ref1'] = None
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    from spacemap_gui.studio import mr_ml, reference_models, tasks
    os.environ['SPACEMAP_MODELS_DIR'] = str(root / 'models')
    chosen = {**mr_ml.DEFAULTS, 'markers':['M0','M1'], 'n_estimators':20, 'cpu_workers':2, 'enable_filter':False,
              'feature_weights':{'markers':1.,'morphology':1.,'compartments':0.}}
    mr_ml.cache_path(root).write_text(json.dumps(chosen))
    train = {'table':str(table),'roi_column':'batch_id','label_column':'label','markers':['M0','M1'],
             'training_column':'phenotype_ref1','checked_only':False}
    train['revision'] = tasks.source_revision(train)
    reference_models.train(train,root/'model-training',lambda *args:None)
    original = table.read_bytes()
    async def check(request):
        values = dataio.load(table)
        score = 'my_cell_score'
        return JSONResponse({'csv_unchanged':table.read_bytes() == original,
            'pending':[item['column'] for item in pending_columns.pending(table)],
            'scored':int(values[score].notna().sum()) if score in values else None,
            'kept':bool((values.loc[frame.phenotype_ref1.notna(), 'phenotype_checked'] == frame.loc[frame.phenotype_ref1.notna(), 'phenotype_ref1']).all()) if 'phenotype_checked' in values else None,
            'csv_columns':list(pd.read_csv(table, nrows=0).columns),
            'saved_scored':int(values['saved_model_score'].notna().sum()) if 'saved_model_score' in values else None,
            'models':len(reference_models.available(table))})
    api.app.router.routes.append(Route('/mr-reference-test', check))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const initialFetch=window.fetch;let earlyFailures=0;
 window.fetch=(url,args)=>String(url).endsWith('/selection_options')&&earlyFailures++<1 ?
   Promise.resolve(new Response(JSON.stringify({error:'Open an image first.'}),{status:400,headers:{'Content-Type':'application/json'}})) : initialFetch(url,args);
 const useMethod=document.querySelector('[aria-label="Phenotyping method"]');useMethod.value='mr_ml';useMethod.dispatchEvent(new Event('change'));
 await until(()=>document.querySelector('[aria-label="MR-ML mode"]'));
 assert(earlyFailures>=2&&!document.querySelector('#studio-training').textContent.includes('Open an image first.'),'MR-ML recovers from an early empty session without a page refresh');
 window.fetch=initialFetch;
 const form=document.querySelector('#studio-training');
 const input=label=>form.querySelector('[aria-label="'+label+'"]');
 const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
 assert(input('MR-ML mode').value==='reference','Reference-only is the initial MR-ML mode');
 assert(input('MR-ML workflow').options.length===2&&input('MR-ML trees').value==='20','The two workflows load Classic settings, including tree count');
 assert(button(document.querySelector('.studio-runbar'),'Run MR-ML')&&!button(form,'Run MR-ML'),'Run MR-ML uses the shared footer');
 const overwriteHelp=form.querySelector('[aria-label="Overwrite existing results"]').parentElement.querySelector('.help').getBoundingClientRect();
 assert(overwriteHelp.width<=18&&Math.abs(overwriteHelp.width-overwriteHelp.height)<2,'Overwrite help stays a compact round icon');
 input('Training reference column').value='phenotype_ref1';
 input('Use morphology area').checked=true;input('Use morphology eccentricity').checked=true;
 input('MR-ML result column').value='phenotype_predicted';input('MR-ML result column').dispatchEvent(new Event('input'));
 assert(input('MR-ML score column').value==='phenotype_predicted_confidence','Suggested output names follow the chosen base name');
 input('MR-ML score column').value='my_cell_score';input('MR-ML filtered column').value='phenotype_checked';
 buildPanel();assert(input('MR-ML score column').value==='my_cell_score','Layer refresh keeps entered MR-ML settings');
 input('MR-ML mode').value='ensemble';input('MR-ML mode').dispatchEvent(new Event('change'));
 assert(form.textContent.includes('per-phenotype F1'),'Classic mode explains reference-weighted method votes');
 input('MR-ML mode').value='reference';input('MR-ML mode').dispatchEvent(new Event('change'));
 input('MR-ML minimum score').value='.9';input('MR-ML low-score label').value='Others';
 let sent;const originalFetch=window.fetch;
 window.fetch=(url,args)=>{if(String(url).includes('/command')&&args?.body){const body=JSON.parse(args.body);if(body.name==='run_mr_ml')sent=body;}return originalFetch(url,args);};
 button(document.querySelector('.studio-runbar'),'Run MR-ML').click();
 await until(()=>document.querySelector('#studio-jobs').textContent.includes('Show result'));
 const config=await(await fetch(api('/studio/browser-check/mrml_settings'))).json();
 assert(sent?.args.settings_revision&&config.values.phenotyping_methods.length===0&&config.values.morphology_cols.join(',')==='area,eccentricity','Run uses the saved Classic settings with selected morphology and no votes');
 const read=async()=>await(await fetch(api('/mr-reference-test'))).json();
 let check=await read();
 assert(check.csv_unchanged&&check.pending.length===3,'Prediction and score columns are saved in Parquet without rewriting the CSV');
 assert(check.scored===10&&check.kept,'Only the remaining cells get prediction scores; references are retained');
 const jobs=document.querySelector('#studio-jobs');button(jobs,'Show result').click();
 await until(()=>s.layers.some(layer=>layer.color_by==='phenotype_checked'));
 assert(true,'Show result uses the filtered label column');
 await until(()=>button(jobs,'Write into the CSV'));
 window.confirm=()=>true;button(jobs,'Write into the CSV').click();
 await until(async()=>{check=await read();return check.csv_columns.includes('my_cell_score');});
 assert(check.csv_columns.includes('phenotype_predicted')&&check.csv_columns.includes('phenotype_checked'),'Jobs writes all three named columns into the CSV on request');
 assert(check.scored===10&&check.kept,'CSV commit preserves numeric scores and reference labels');
 input('MR-ML workflow').value='saved_model';input('MR-ML workflow').dispatchEvent(new Event('change'));
 const savedModel=input('Saved reference model');savedModel.value=savedModel.options[1].value;savedModel.dispatchEvent(new Event('change'));
 input('Keep existing reference labels').checked=false;
 input('MR-ML result column').value='phenotype_saved_model';input('MR-ML result column').dispatchEvent(new Event('input'));
 input('MR-ML score column').value='saved_model_score';input('MR-ML filtered column').value='phenotype_saved_filtered';
 button(document.querySelector('.studio-runbar'),'Run MR-ML').click();
 await until(async()=>{check=await read();return check.saved_scored===40;});
 assert(check.saved_scored===40&&check.models===2,'Saved-model workflow predicts every cohort cell without creating another model');
 const chosenModel=savedModel.value;
 window.dispatchEvent(new CustomEvent('studio-reference-method',{detail:{method:'mr_ml',column:'phenotype_ref1',model_id:chosenModel}}));
 await until(()=>document.querySelector('#studio-training')!==form&&document.querySelector('[aria-label="MR-ML workflow"]')?.value==='saved_model');
 await until(()=>document.querySelector('[aria-label="Saved reference model"]')?.value===chosenModel);
 assert(document.querySelector('[aria-label="Training reference column"]').value==='phenotype_ref1'&&!document.querySelector('[aria-label="Limit rows"]').checked,
   'Manual Reference handoff selects the saved model and whole-cohort prediction');
 window.fetch=originalFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-jobs')?.textContent+'\n'+document.querySelector('#studio-training')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup)
