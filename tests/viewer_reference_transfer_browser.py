"""Switch real viewer ROI layers and refit a saved model with extra/missing inputs."""
from viewer_threshold_browser import main


def setup(root):
    import copy
    import shutil
    import numpy as np
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui.viewer import api, registry
    from spacemap_gui.studio import reference_models, tasks
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    table=root/'cells.csv'
    frame=pd.read_csv(table).drop(columns='phenotype_ml_consensus')
    frame['area']=np.arange(40)+100
    frame['eccentricity']=np.linspace(.1,.8,40)
    frame['HE_Red']=np.arange(40)+10.
    other=frame.copy();other['batch_id']='roi2';other['cell_id']+=40
    other['phenotype_ref1']=None;other['M1']=np.nan
    pd.concat([frame,other],ignore_index=True).to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    original=table.read_bytes()
    settings={'table':str(table),'training_column':'phenotype_ref1',
              'roi_column':'batch_id','label_column':'label','markers':['M0','M1']}
    settings['revision']=tasks.source_revision(settings)
    trained=reference_models.train(settings,root,lambda *args:None)
    second=root/'roi2.ome.tiff';shutil.copyfile(root/'roi.ome.tiff',second)
    image_token=registry.register(second)

    async def info(request):
        return JSONResponse({'trained':trained,'unchanged':table.read_bytes()==original})

    async def switch(request):
        spec=copy.deepcopy(registry.session_view('browser-check',mark_seen=False)['spec'])
        old=next(layer for layer in spec['layers'] if layer['type']=='cells')
        props={key:value for key,value in registry.resolve_object(old['token'],'cells').items() if key!='kind'}
        token=registry.register_object('cells',**{**props,'roi':'roi2','image':str(second)})
        spec['id']='second-roi';spec['title']='Second ROI';spec['layers']=[layer for layer in spec['layers'] if layer['type']!='annotations']
        for layer in spec['layers']:
            if layer['type']=='image':layer.update(token=image_token)
            if layer['type']=='cells':
                layer.update(token=token,roi='roi2',image_token=image_token,suggestionMode='consensus_rf',suggestionsOn=True)
                layer.pop('resolve',None)
        registry.publish('browser-check',spec,project=registry.session_project('browser-check'))
        return JSONResponse(spec)
    api.app.router.routes.extend([Route('/transfer-model',info),Route('/transfer-switch',switch)])


DRIVER=r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change',{bubbles:true}));};
const button=(parent,text)=>Array.from(parent.querySelectorAll('button')).find(b=>b.textContent===text);
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent(event.message+' '+event.filename+':'+event.lineno)));
setTimeout(async()=>{try{
 const s=window.__state, cells=()=>s.layers.find(l=>l.type==='cells'), image=()=>s.layers.find(l=>l.type==='image');
 const card=()=>document.querySelector('.manual-reference-card');
 await until(()=>s.imagesReady&&s.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 change(document.querySelector('[aria-label="Phenotyping method"]'),'reference');
 await until(()=>cells().combinedTable&&card()?.referenceInfo);
 s.layers.filter(l=>l.type==='annotations').forEach(l=>l.visible=false);setTool('off');
 const read=async()=>await(await fetch('/spacemap-viewer/api/transfer-model')).json();
 const first=(await read()).trained;
 document.dispatchEvent(new CustomEvent('spacemap-reference-trained',{detail:{id:'trained-first',result:first}}));
 await until(()=>cells().suggestionModelId===first.models[0].model_id);
 const click=()=>s.viewer.raiseEvent('canvas-click',{quick:true,position:s.viewer.viewport.pixelFromPoint(image().item.imageToViewportCoordinates(55,15)),originalEvent:{}});
 click();await until(()=>s.inspectedCell?.result.suggestions?.status==='ready');
 assert(!document.querySelector('#cellcard').textContent.includes('phenotype_ml_consensus'),'Manual-reference predictions need no old consensus column');
 const training=card().querySelector('.reference-model-training');training.open=true;
 await until(()=>card().querySelector('[aria-label="Train with area"]'));
 const features=card().querySelector('.reference-training-features');features.open=true;
 card().querySelector('[aria-label="Train with area"]').click();
 card().querySelector('[aria-label="Train with eccentricity"]').click();
 card().querySelector('[aria-label="Train with HE_Red"]').click();
 assert(features.querySelector('summary').textContent.includes('5 selected'),'Training can include morphology and H&E alongside hidden markers');
 button(card(),'Train from this reference').click();
 await until(()=>s.inspectedCell?.result.suggestions?.provenance?.markers?.length===5);
 const model=cells().suggestionModelId;
 assert((await read()).unchanged,'Retraining with extra features preserves existing manual labels and data');
 assert(s.inspectedCell.result.suggestions.provenance.markers.includes('area')&&s.inspectedCell.result.suggestions.provenance.markers.includes('HE_Red'),
   'The replacement model actually uses the additional features');
 const next=await(await fetch('/spacemap-viewer/api/transfer-switch')).json();
 await window.__init(next);
 await until(()=>s.imagesReady&&s.cellsReady&&cells().combinedTable&&cells().table===cells().combinedTable);
 assert(cells().suggestionMode==='saved_reference'&&cells().suggestionModelId===model&&cells().suggestionsOn,
   'Switching image keeps the trained reference model and overrides the old per-image consensus default');
 setTool('off');click();await delay(500);
 fetch('/result?progress='+encodeURIComponent('Second ROI selection: '+JSON.stringify({id:s.spec.id,roi:cells().roi,table:cells().table,cell:s.inspectedCell?.result?.label,mode:cells().suggestionMode,profile:s.inspectedCell?.result?.profile?.status,suggestions:s.inspectedCell?.result?.suggestions})));
 await until(()=>s.inspectedCell?.result.suggestions?.status==='ready');
 const result=s.inspectedCell.result.suggestions;
 assert(result.provenance.features_used===4&&result.provenance.missing_markers.join(',')==='M1',
   'A cell in the new ROI receives a prediction using its four available features');
 assert(document.querySelector('#cellcard').textContent.includes('missing: M1'),'The cell card identifies the missing feature');
 // A click made before combined-table linking may already contain the old
 // consensus error. Completing the link must refresh that card itself.
 delete cells().referenceChoiceTable;cells().suggestionMode='consensus_rf';
 s.inspectedCell.result.suggestions={status:'error',message:'old consensus column missing'};
 await dataRefreshCombined();
 await until(()=>s.inspectedCell?.result.suggestions?.status==='ready');
 assert(cells().suggestionMode==='saved_reference'&&!document.querySelector('#cellcard').textContent.includes('old consensus'),
   'Completing the table link refreshes an early click without another user click');
 // The same completed Auto-train run must also activate on a new layer object.
 document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:{enabled:true,table:first.table,completed:{id:'auto-same-run',result:first}}}));
 await until(()=>cells().suggestionModelId===first.models[0].model_id);
 await window.__init({...next,id:'second-roi-reopened'});
 await until(()=>s.imagesReady&&s.cellsReady&&cells().combinedTable);
 cells().suggestionMode='consensus_rf';cells().suggestionModelId=null;
 document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:{enabled:true,table:first.table,completed:{id:'auto-same-run',result:first}}}));
 await until(()=>cells().suggestionMode==='saved_reference'&&cells().suggestionModelId===first.models[0].model_id);
 assert(true,'An already completed Auto-train model activates on newly opened cell layers');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#cellcard')?.textContent+'\n'+document.querySelector('.manual-reference-card')?.textContent));}},100);
</script>
"""

if __name__=='__main__':
    main(DRIVER,studio=True,timeout=240,setup=setup,screenshot=False)
