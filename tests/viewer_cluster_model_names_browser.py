"""Offline model names, source races and AI evidence for all three cluster explorers."""
from viewer_threshold_browser import main


def setup(root):
    import os
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui.studio import reference_models, tasks, cluster_profiles, chat
    from test_studio_chat import _job

    os.environ['SPACEMAP_MODELS_DIR'] = str(root / 'models')
    # Only image capability is simulated. Every chat POST is intercepted by the
    # driver, so this check never contacts an AI provider.
    chat.sees_images = lambda: True
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    for method in ['leiden', 'som', 'vortex']:
        frame['phenotype_' + method + '_clusters'] = ['Cluster 0'] * 20 + ['Cluster 1'] * 20
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    settings = {'table': str(table), 'roi_column': 'batch_id', 'label_column': 'label',
                'markers': ['M0', 'M1'], 'training_column': 'phenotype_ref1', 'checked_only': False}
    settings['revision'] = tasks.source_revision(settings)
    reference_models.train(settings, root / 'trained', lambda *_: None)
    for method in ['leiden', 'som', 'vortex']:
        features = ['M0'] if method == 'som' else ['M0', 'M1']
        column = 'phenotype_' + method + '_clusters'
        clusters, profile = cluster_profiles.build(frame, features, column, markers=features, name=str)
        _job(root, method, {'table': str(table), 'column': column, 'phenotype_column': 'phenotype_' + method,
            'markers': features, 'clusters': clusters, 'cells': len(frame), 'profile': profile, 'where': {}})


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
const button=(box,text)=>[...box.querySelectorAll('button')].find(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&tools().querySelector('[aria-label="Phenotyping method"]'));
 const realFetch=window.fetch;let aiCalls=0,modelCalls=0;
 window.fetch=(url,args)=>{if(String(url).endsWith('/chat')&&args?.method==='POST')aiCalls++;
  if(String(url).endsWith('/cluster_suggestions')&&args?.method==='POST'&&JSON.parse(args.body).model_id)modelCalls++;
  return realFetch(url,args);};
 const picker=()=>tools().querySelector('[aria-label="Cluster naming model"]');
 const box=()=>tools().querySelector('.studio-cluster-suggestions');
 const row=group=>tools().querySelector('[data-group="'+group+'"]').closest('.studio-class-row');
 const input=group=>tools().querySelector('[aria-label="Phenotype of '+group+'"]');
 for(const method of ['leiden','som','vortex']){
  change(tools().querySelector('[aria-label="Phenotyping method"]'),method);
  await until(()=>picker()?.options.length&&input('Cluster 0'));
  const model=[...picker().options].find(o=>o.value.startsWith('model:')&&!o.disabled);
  assert(!!model,method+' offers the trained Manual reference model');
  assert(tools().querySelectorAll('.studio-cluster-suggestions').length===1&&box().querySelector('label').textContent==='Model',method+' uses one panel and the shared Model label');
  if(method==='som')assert(model.title.startsWith('1/2 matching features'),'A model trained on more features remains available');
  const before=modelCalls;await delay(150);assert(modelCalls===before,method+' does not load model predictions while opening the panel');
  change(picker(),model.value);input('Cluster 0').value='Keep my edit';input('Cluster 0').dispatchEvent(new Event('input'));
  button(box(),'Suggest names').click();await until(()=>box().textContent.includes('clusters scored')&&!button(box(),'Suggest names').disabled);
  assert(aiCalls===0,method+' suggests names without any AI request');
  assert(input('Cluster 0').value==='Keep my edit',method+' suggestions preserve the typed name');
  if(method==='som')assert(row('Cluster 0').querySelector('.studio-hints .help').getAttribute('data-tip').includes('1 feature.'),'Predictions use the feature subset in this clustering');
  assert(['Cluster 0','Cluster 1'].every(group=>row(group).querySelectorAll('.studio-hints .studio-suggestion').length===1),method+' keeps one compact suggestion per cluster');
  assert(!row('Cluster 0').querySelector('[aria-label^="Ask AI"]')&&row('Cluster 0').querySelector('.studio-hints .help').getAttribute('data-tip').includes('sampled cells'),method+' moves evidence into help and removes repeated AI buttons');
  row('Cluster 0').querySelector('.studio-suggestion').click();await until(()=>input('Cluster 0').value==='A');
  assert(input('Cluster 1').value==='',method+' accepting one suggestion leaves other cluster names alone');
  if(method==='leiden'){
   let release,held=false,requests=0;
   window.fetch=async(url,args)=>{
    if(String(url).endsWith('/chat')&&args?.method==='POST'){aiCalls++;return new Response(JSON.stringify({done:{say:'Unexpected AI request',source:'none'}})+'\n',{headers:{'Content-Type':'application/x-ndjson'}});}
    if(String(url).endsWith('/cluster_suggestions')&&args?.method==='POST'&&JSON.parse(args.body).model_id)requests++;
    const response=await realFetch(url,args);
    if(String(url).endsWith('/cluster_suggestions')&&args?.method==='POST'&&JSON.parse(args.body).model_id&&!held){held=true;await new Promise(r=>release=r);}return response;};
   button(box(),'Suggest names').click();await until(()=>held);
   box().querySelector('[aria-label="Ask AI about clusters"]').click();await delay(150);
   assert(requests===1,'Ask AI reuses an in-flight model prediction');
   change(picker(),'reference:phenotype_ref1');button(box(),'Suggest names').click();
   await until(()=>box().textContent.includes('labelled cells'));release();await delay(150);
   assert(aiCalls===0,'Changing the model source cancels a pending AI naming request');
   assert(picker().value==='reference:phenotype_ref1'&&row('Cluster 0').querySelector('.studio-hints .help').getAttribute('data-tip').startsWith('Reference-cell'),'A delayed model response cannot replace a newer reference-source choice');
   assert(input('Cluster 0').value==='A','Changing suggestion sources preserves chosen names');
   window.fetch=realFetch;
   row('Cluster 1').querySelector('.studio-suggestion').click();await until(()=>input('Cluster 1').value==='B');
   button(tools(),'Save names').click();
   await until(async()=>{const found=await(await realFetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();return found.jobs.some(j=>j.kind==='name_clusters'&&j.status==='done');});
   assert(true,'Accepted suggestions save through the existing name-clusters job');
  }
 }
 let sent;
 window.fetch=async(url,args)=>String(url).endsWith('/chat')&&args?.method==='POST'?(sent=JSON.parse(args.body),new Response(JSON.stringify({done:{conversation:'check',say:'Review the model and marker evidence.',source:'model',remember:[]}})+'\n',{status:200,headers:{'Content-Type':'application/x-ndjson'}})):realFetch(url,args);
 row('Cluster 0').querySelector('[data-group]').click();
 box().querySelector('[aria-label="Ask AI about clusters"]').click();await until(()=>sent);
 assert(sent.page.explorer.model_id&&sent.page.explorer.method==='vortex'&&sent.page.explorer.picked.includes('Cluster 0'),'AI receives the selected trained model, run and cluster');
 assert(sent.text.includes('Model votes are not independent accuracy')&&sent.text.includes('heatmap'),'AI is instructed to compare model evidence with the heatmap');
 assert(sent.images?.some(value=>value.startsWith('data:image/')),'The heatmap is attached for an image-capable AI model');
 assert(![...document.querySelectorAll('#studio-figures .studio-namer button')].some(b=>b.textContent==='Ask AI')&&tools().querySelectorAll('[aria-label="Ask AI about clusters"]').length===1,'AI naming has one control in the naming panel');
 window.fetch=realFetch;
 assert(box().scrollWidth<=box().clientWidth+1,'Naming controls fit the Studio dock');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=240, setup=setup)
