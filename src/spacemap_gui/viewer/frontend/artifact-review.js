/* Reviewed artifact layers and independent saved model libraries. No image work
   starts just because a panel opens. Left and right controls share one state. */
window.createArtifactReview = h => {
  'use strict';
  const {node, field, choice, help, get, post, route} = h;
  const names={clot:'Blood clots',blur:'Blur',bright:'Bright spots',lint:'Lint',debris:'Image quality'};
  const tabsShown=['clot','debris'];
  const goodName=e=>e.part==='blur'?'Sharp tissue':e.part==='lint'?'Good glass':'Good tissue';
  const labels={clot:'Pinker by',blur:'Blur level',bright:'Bright level',lint:'Lint signal'};
  const entries=new Map(); let selected='clot'; let saveQueue=Promise.resolve(); let drawingRequest=0;
  const image=()=>h.image();
  const key=part=>image()?.token+':'+part;
  const entry=part=>entries.get(key(part));
  const valid=e=>image()?.token===e.image_token;
  const path=(e,action)=>route('/artifact_review/'+e.part+'/'+action);
  const postFor=(e,action,body={})=>post(path(e,action),{image_token:e.image_token,...body});
  const refresh=e=>{e.views=e.views.filter(v=>v.root.isConnected);e.views.forEach(v=>v.sync());};
  const modelState=(e,data)=>{
    e.versions=data.versions;e.deleted_versions=data.deleted_versions||[];
    if(!e.modelBusy)e.version=data.selected_version??e.versions.at(-1)?.id??'';
  };
  async function manageModel(e,action,version){
    if(e.modelBusy||e.read_only||!valid(e))return;
    const chosen=e.versions.find(v=>v.id===version);
    if(action==='delete_model'&&!window.confirm('Delete '+(chosen?modelName(e,chosen):'this model')+'? Saved polygons and examples stay intact. You can restore it under Deleted models.'))return;
    e.modelBusy=true;e.lastAction='training';e.trainingDetails='';e.sequence++;if(action==='select_model')e.choices=(e.choices||0)+1;refresh(e);
    try{
      const result=await postFor(e,action,{version});e.modelBusy=false;modelState(e,result);
      e.trainingMessage=action==='delete_model'?'Model deleted.':action==='restore_model'?'Model restored.':'';
    }catch(error){e.modelBusy=false;e.trainingMessage=error.message;try{modelState(e,await get(path(e,'state')));}catch(_){} }
    finally{e.modelBusy=false;refresh(e);}
  }
  async function load(part) {
    const id=key(part); if(entries.has(id)) {
      const e=await entries.get(id).ready;
      if(valid(e)){
        const sequence=e.sequence,busy=e.modelBusy,found=await get(path(e,'state'));
        if(valid(e)&&sequence===e.sequence&&!busy&&!e.modelBusy){
          // The layer as the server gives it now (its file follows the Output folder and Sub-directory).
          e.examples=found.examples;modelState(e,found);e.microns_per_pixel=found.microns_per_pixel;
          if(found.layer)e.layer=found.layer;if(found.example_layer)e.example_layer=found.example_layer;
          await h.ensureLayer(e.layer);
        }
      }
      return e;
    }
    const e={part,image_token:image()?.token,views:[],message:'',trainingMessage:'',busy:false,sequence:0}; entries.set(id,e);
    e.ready=(async()=>{try {
      Object.assign(e,await get(path(e,'state'))); e.values={...e.settings}; e.saved={...e.settings};
      modelState(e,e);
      if(valid(e)) await h.ensureLayer(e.layer); return e;
    } catch(error) {entries.delete(id);throw error;}})();return e.ready;
  }
  function save(e) {
    clearTimeout(e.saveTimer);
    const work=async()=>{
      if(JSON.stringify(e.values)===JSON.stringify(e.saved)) return;
      const snapshot={...e.values}; e.saving=true;refresh(e);
      try {const result=await postFor(e,'settings',{values:snapshot,revision:e.revision});e.revision=result.revision;e.saved=snapshot;e.saveError='';}
      catch(error){e.saveError=error.message;throw error;}
      finally{e.saving=false;refresh(e);}
    };
    const result=saveQueue.catch(()=>{}).then(work);saveQueue=result;return result;
  }
  function update(e,key,value) {
    e.values[key]=value;e.sequence++;refresh(e);clearTimeout(e.saveTimer);
    e.saveTimer=setTimeout(()=>save(e).catch(()=>{}),400);
  }
  async function openExamples(e) {
    if(e.examplesReady){await h.ensureLayer(e.example_layer);return;}
    if(!e.examplesLoading)e.examplesLoading=(async()=>{
      await h.flush();
      const result=await postFor(e,'open_examples',{artifact_preview:h.exampleState(e.part).preview});
      if(!valid(e))return;
      e.example_layer=result.layer;await h.ensureLayer(result.layer);e.examplesReady=true;if(result.combined_red)await h.reloadExamples(e.part);h.rebuild();
    })().finally(()=>{e.examplesLoading=null;});
    await e.examplesLoading;
  }
  const modelName=(e,v)=>(v.source_part==='blur'?'Blur':e.part==='debris'&&!v.artifact_targets?'Bright spots & lint':names[e.part])+' · '+new Date(v.created).toLocaleString(undefined,
    {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',fractionalSecondDigits:3,hour12:false})+(v.mode==='artifact_only'?' · Artifact-only':'');
  async function watchTraining(e) {
    clearTimeout(e.jobTimer);
    if(!valid(e)||!e.trainingJob)return;
    // Several watchers can be waiting at once (training, a refresh, reopening the panel): only the first to see the
    // job end acts on it, so a model chosen meanwhile is not replaced by the new one again.
    const id=e.trainingJob;
    try{
      const found=await get(route('/jobs'));
      if(e.trainingJob!==id)return;
      const job=found.jobs.find(j=>j.id===id);
      if(job?.status==='done'){
        const result=await get(path(e,'state'));
        if(e.trainingJob!==id)return;
        e.trainingJob=null;e.examples=result.examples;modelState(e,result);
        // The new model is chosen, unless another one was chosen by hand since Train was pressed.
        if((e.choices||0)===(e.choicesAtTraining||0))e.version=job.result?.version?.id || e.versions.at(-1)?.id || '';
        e.trainingDetails=job.result?.preparation?.device==='cuda'?'Examples prepared on GPU: '+job.result.preparation.name:'Examples prepared on CPU.';
        e.trainingJob=null;e.trainingMessage='Model ready. Predict to test it.';
      }else if(job&&['failed','cancelled'].includes(job.status)){
        e.trainingJob=null;e.trainingMessage=job.message||'Training stopped. Check Jobs for details.';
      }else e.trainingMessage=job?.message||'Training…';
    }catch(error){e.trainingMessage=error.message;}
    refresh(e);
    if(e.trainingJob&&valid(e))e.jobTimer=setTimeout(()=>watchTraining(e),1500);
  }
  // Train on the ticked images: each one's saved (hand-fixed) polygons are its red examples and its green drawings
  // its good tissue; the open image's edits are saved first. The base model's other images are kept.
  async function trainSelected(e) {
    if(e.trainingBusy||e.trainingJob||e.modelBusy||!valid(e))return;
    const images=h.picked();
    if(!images.length){e.lastAction='training';e.trainingMessage='Tick the reviewed images in the list first.';refresh(e);return;}
    const base=e.values.training_source==='library'?null:e.version||null;
    e.trainingBusy=true;e.lastAction='training';e.trainingDetails='';e.trainingMessage='Saving this image’s edits…';refresh(e);
    try{
      await h.flush();
      if(h.annotationState()[e.layer.id]?.pending)await h.savePolygons(e.part);
      if(e.examplesReady)await h.finishExamples(e.part);
      await save(e);
      const result=await postFor(e,'train',{images,base_version:base,scale:e.values.scale,remaining_good:e.values.remaining_good});
      e.examples=result.examples;e.trainingJob=result.job.id;e.choicesAtTraining=e.choices||0;
      e.trainingMessage='Training on '+result.images+' reviewed image'+(result.images===1?'':'s')+'…';
      e.trainingDetails=result.skipped.length?'Left out: '+result.skipped.join('; '):'';
      watchTraining(e);
    }catch(error){e.trainingMessage=error.message;}
    finally{e.trainingBusy=false;refresh(e);}
  }
  async function trainCurrent(e) {
    if(e.trainingBusy||e.trainingJob||e.modelBusy||!valid(e))return;
    const base=e.values.training_source==='library'?null:e.version||null;
    const capture=e.values.include_current!==false;
    e.trainingBusy=true;e.lastAction='training';e.trainingDetails='';e.trainingMessage=capture?'Saving examples…':'Reusing saved examples…';refresh(e);
    try{
      let request={base_version:base,capture_current:capture};
      if(capture){
        await openExamples(e);await h.finishExamples(e.part);
        const drawn=h.exampleState(e.part);
        request={...request,example_revision:drawn.revision,scale:e.values.scale,use_reviewed:true,
          remaining_good:e.values.remaining_good,...h.reviewed(e.part)};
      }
      await save(e);
      const result=await postFor(e,'train',request);
      e.examples=result.examples;e.trainingJob=result.job.id;e.choicesAtTraining=e.choices||0;e.trainingMessage='Training…';
      watchTraining(e);
    }catch(error){e.trainingMessage=error.message;}
    finally{e.trainingBusy=false;refresh(e);}
  }
  async function preview(e,model=false,replace=false) {
    if(!valid(e)||e.modelBusy)return;
    if(e.busy){e.again={model};return;}
    if(h.needsReview(e.part) && !replace && !window.confirm('Replace the edited preview with a new detection? Save polygons first if you want to keep it in History.'))return;
    e.busy=true;e.lastAction='preview';e.details='';e.message=model?'Predicting…':'Detecting…';refresh(e);
    const sequence=e.sequence;
    try {
      await openExamples(e);await save(e);
      const result=await postFor(e,model?'predict':'detect',{...e.values,version:e.version,manual_features:h.manualArtifacts(e.part)});
      if(!valid(e))return;
      if(sequence!==e.sequence){e.again=e.again||{model};return;}
      await h.showPreview(result);e.detected=true;
      e.message=result.features.length+' polygons · '+(result.mode==='artifact_only'?'Artifact-only similarity':result.used);
      e.details=[result.missing_markers?.length?'Markers not available: '+result.missing_markers.join(', '):'',result.execution?.device==='cuda'?'GPU measurements: '+result.execution.name:'CPU measurements',result.execution?.forest_device==='cuda'?'Forest evaluated on GPU':''].filter(Boolean).join(' · ');
    }catch(error){e.message=error.message;}
    finally{e.busy=false;refresh(e);if(e.again){const next=e.again;e.again=null;preview(e,next.model,true);}}
  }
  function controls(e,root,teach=true,goodHolder=root) {
    root.dataset.artifactImage=e.image_token;
    const modelOnly=e.part==='clot';
    const scale=choice(e.scales.map(n=>[String(n),h.scaleLabel(n,e.microns_per_pixel || h.pixelSize?.())]),String(e.values.scale),'Artifact scale');
    scale.onchange=()=>update(e,'scale',Number(scale.value));
    const sliders=[];
    const schedule=()=>{clearTimeout(e.previewTimer);e.previewTimer=setTimeout(()=>preview(e),450);};
    let detectorChoice;
    const detectors=['blur','bright','lint'];
    const detectorSelection=()=>detectors.filter(p=>e.values.detectors.includes(p)).join('+');
    if(e.part==='debris'){
      detectorChoice=choice([['blur+bright+lint','All three'],['blur','Blur'],['bright','Bright spots'],['lint','Lint'],['bright+lint','Bright spots & lint'],['blur+bright','Blur & bright spots'],['blur+lint','Blur & lint']],detectorSelection(),'Preview detector');
      detectorChoice.onchange=()=>{update(e,'detectors',detectorChoice.value.split('+'));schedule();};
    }
    for(const part of modelOnly?[]:e.part==='debris'?detectors:[e.part]){
      const range=e.part==='debris'?e.detector_ranges[part]:e.range;
      const current=()=>e.part==='debris'?e.values.detector_values[part]:e.values.value;
      const slider=node('input',{type:'range',...range,value:current(),'aria-label':labels[part]});
      const value=node('input',{type:'number',...range,value:current(),'aria-label':labels[part]+' value',style:'width:64px;flex:0 0 64px'});
      const slide=n=>{if(e.part==='debris')update(e,'detector_values',{...e.values.detector_values,[part]:Number(n)});else update(e,'value',Number(n));schedule();};
      slider.oninput=()=>slide(slider.value);value.onchange=()=>slide(value.value);
      const tip=part==='blur'?'A higher cut includes more regions with low overall sharpness. This classical slider is a starting point; the learned model also measures individual markers.':part==='bright'?'Lower the percentile to include more bright regions. Flat glare and signal across many markers are candidate artifacts. Review these before training.':'Lower the signal threshold to include fainter fibres. This slider looks for elongated bright objects inside Work within, including on tissue. Use the model for other shapes.';
      sliders.push({part,slider,value,current,row:field(labels[part],node('div',{class:'inline'},[slider,value]),tip+' Moving the slider previews after a short pause. Edited previews require confirmation before replacement.')});
    }
    const minimum=node('input',{type:'number',min:0,step:10,value:e.values.minimum_um2,'aria-label':'Smallest artifact'});
    minimum.onchange=()=>update(e,'minimum_um2',Number(minimum.value));
    const holes=node('input',{type:'number',min:0,step:5,value:e.values.fill_holes_um2??0,'aria-label':'Fill artifact holes smaller than'});
    holes.onchange=()=>update(e,'fill_holes_um2',Math.max(0,Number(holes.value)||0));
    const detect=modelOnly?null:node('button',{class:'btn primary small',text:'Preview',onclick:()=>preview(e)});
    const saved=node('span',{class:'chip'}),status=node('div',{class:'muted artifact-process-status',role:'status','aria-atomic':'true'});
    const retry=node('button',{class:'btn small',text:'Save settings',hidden:true,onclick:()=>save(e).catch(()=>{})});
    const detectorFields=[...(detectorChoice?[field('Preview detector',detectorChoice,'Choose which sliders propose candidates for the same red artifact list. The shared model learns blur, bright spots and lint from your reviewed examples, independently of these sliders.')]:[]),...sliders.map(v=>v.row)];
    root.append(field('Scale',scale,'Detection resolution. Full keeps the finest detail; coarser scales are faster. Blur needs fine detail: start at Full or 1/2 and keep the scale consistent while comparing models. This does not change Tissue settings.'),
      field('Smallest',node('div',{class:'inline'},[minimum,node('span',{class:'muted',text:'µm²'})]),'Exclude artifact polygons below this area.'),
      field('Fill holes',node('div',{class:'inline'},[holes,node('span',{class:'muted',text:'µm²'})]),'Fill holes smaller than this inside found polygons (0: keep every hole). Predictions often have thousands of one-pixel gaps inside a clot; filling those below 10 to 20 µm² makes the polygons several times lighter, so editing and marking them is faster. It slightly enlarges what is subtracted from the tissue.'),
      node('div',{class:'row'},[saved,retry]));
    if(detect)root.append(node('details',{'data-artifact-sliders':e.part},[node('summary',{text:'Detection sliders'}),...detectorFields,detect]));
    let versions,library,train,trainSome,predict,remaining,cut,exampleCounts,remove,deleted,restore,deletedList,include,source,forget,marks=[];
    if(teach){
      const details=node('details',{'data-artifact-training':e.part});details.open=e.teachOpen??true;
      details.ontoggle=()=>{if(details.isConnected){e.teachOpen=details.open;if(details.open&&!e.read_only)openExamples(e).then(()=>refresh(e)).catch(error=>{if(!e.trainingMessage){e.lastAction='training';e.trainingDetails='';e.trainingMessage=error.message;}refresh(e);});}};
      details.append(node('summary',{text:'Train from examples'}));
      marks=['positive','negative'].map(role=>node('button',{class:'btn small', 'data-example-role':role,
        'aria-label':role==='positive'?'Draw artifact examples':'Draw good tissue examples',onclick:async()=>{
          const request=++drawingRequest, previous=h.drawingState(), pending='artifact-examples-loading-'+request;
          e.drawingTool=['rect','polygon','freehand'].includes(previous.tool)?previous.tool:e.drawingTool;
          h.setTarget(pending);e.lastAction='training';e.trainingDetails='';e.trainingMessage='';refresh(e);
          const current=()=>request===drawingRequest&&valid(e)&&root.isConnected&&h.drawingState().id===pending;
          try{
            await openExamples(e);
            // The artifact layer may have been dropped by a refresh of the image's layers (no file saved yet): add it again.
            if(role==='positive')await h.ensureLayer(e.layer);
            if(!current())return;
            if(!h.drawExamples(e.part,role,e.drawingTool))throw Error('The example layer could not be opened.');
            e.trainingMessage='';refresh(e);
          }catch(error){if(current()){e.trainingMessage='Drawing paused. Try choosing '+(role==='positive'?'Artifact':'Good tissue')+' again.';refresh(e);}}
        }},[node('span',{class:'swatch',style:'display:inline-block;background:'+(role==='positive'?'#e6194b':'#2ecc71')}),
          node('span',{text:role==='positive'?'Artifact':goodName(e)})]));
      exampleCounts=node('span',{class:'muted'});
      remaining=node('input',{type:'checkbox','aria-label':'Remaining area is good'});remaining.checked=e.values.remaining_good;
      remaining.onchange=()=>update(e,'remaining_good',remaining.checked);
      train=node('button',{class:'btn small primary',text:'Train model',title:'Train with this image’s reviewed polygons and examples',onclick:()=>trainCurrent(e)});
      trainSome=node('button',{class:'btn small',text:'Train on selected',title:'Train one model with the reviewed polygons and examples of each image ticked in the list',onclick:()=>trainSelected(e)});
      versions=choice([],e.version,'Artifact model version');versions.onchange=()=>manageModel(e,'select_model',versions.value);
      remove=node('button',{class:'btn small',text:'Delete model',onclick:()=>manageModel(e,'delete_model',e.version)});
      deletedList=choice([],'','Deleted artifact model');
      restore=node('button',{class:'btn small',text:'Restore',onclick:()=>manageModel(e,'restore_model',deletedList.value)});
      deleted=node('details',{'data-deleted-models':e.part},[node('summary',{text:'Deleted models'}),node('div',{class:'inline'},[deletedList,restore]),help('Delete removes a version from the model picker. Its weights, examples and lineage are retained so you can restore it and earlier predictions remain traceable.')]);
      predict=node('button',{class:'btn small primary',text:'Predict',onclick:()=>preview(e,true)});
      cut=node('input',{type:'number',min:.05,max:.95,step:.05,value:e.values.cut,'aria-label':'Model cut'});cut.onchange=()=>update(e,'cut',Number(cut.value));
      library=node('span',{class:'muted'});
      include=node('input',{type:'checkbox','aria-label':'Include current image'});
      include.onchange=()=>update(e,'include_current',include.checked);
      source=choice([['selected','Selected model'],['library','Reviewed library']],e.values.training_source||'selected','Training source');
      source.onchange=()=>update(e,'training_source',source.value);
      forget=node('button',{class:'btn small',text:'Forget this image',title:'Remove this image from the reviewed library. Existing model snapshots remain intact.',onclick:async()=>{e.lastAction='training';e.trainingDetails='';try{const result=await postFor(e,'forget');e.examples=result.examples;e.trainingMessage='Removed from the reviewed library.';refresh(e);}catch(error){e.trainingMessage=error.message;refresh(e);}}});
      const advanced=node('details',{'data-training-options':e.part},[node('summary',{text:'Training options'}),
        field('Train from',source,'Selected model reuses exactly that version’s saved examples, excluding later submissions. Reviewed library uses the latest retained contribution from every image, independent of the selected model. With no model yet, training uses the reviewed library.'),
        node('label',{class:'row'},[include,node('span',{text:'Include current image'}),help('Add this image’s reviewed polygons, replacing its older contribution if present. Untick to retrain the selected version from its saved examples alone; current drawings and previews remain untouched.')]),
        node('label',{class:'row'},[remaining,node('span',{text:e.part==='lint'?'Remaining glass is good':'Remaining tissue is good'}),help('Use the selected annotation outside these artifacts as good examples. Turn this off if other artifacts are still unmarked; draw only known good areas instead.')]),
        forget,
        field('Model cut',cut,'A higher cut keeps fewer detections. This is not a calibrated probability. All available channels are read, including hidden markers. Quality models use intensity, local edges and texture across individual channels and pooled channel summaries; unfamiliar panels need testing with new reviewed examples. An Artifact-only model matches the appearance of your red examples; adding good examples enables a two-class random forest.')]);
      const exampleList=node('div',{class:'artifact-example-list'});
      e.exampleList=exampleList;
      const goodReview=node('details',{'data-artifact-good':e.part},[node('summary',{text:goodName(e)}),exampleList]);
      goodReview.open=!!e.goodOpen;
      goodReview.ontoggle=()=>{
        if(!goodReview.isConnected)return;e.goodOpen=goodReview.open;
        if(goodReview.open)openExamples(e).then(()=>{
          if(exampleList.isConnected&&valid(e))h.renderLayer(exampleList,e.example_layer.id);
        }).catch(error=>{e.lastAction='training';e.trainingMessage=error.message;refresh(e);});
      };
      details.append(field('Draw',node('div',{class:'segmented',role:'group','aria-label':'Example class'},marks),(e.part==='debris'?'Red includes blur, bright spots, lint and hairs. Green should be clear of all these defects; review earlier green examples before using them for a broader model. ':'')+'Choose red Artifact or green Good tissue, then use Rectangle, Polygon or Freehand above the image. Select any example to edit it: Shift paints, Alt erases, and the wheel changes brush size. Right-click offers Delete or a class change for the selected polygons. The list buttons apply to all checked polygons. Ctrl+Z undoes.'),
        node('div',{class:'row'},[exampleCounts]),
        field('Model',node('div',{class:'inline'},[versions,remove]),(e.part==='debris'?'Earlier Blur and Bright spots & lint models remain here with their original names. Select one to predict or start a new Image quality version from its examples; the two libraries are not automatically mixed. ':'')+'Choose an older version to Predict or train from its original examples. The selection is saved across images and reloads. Training creates a new dated version and never overwrites its parent. Dates use local time; hover for UTC, image count and parent version.'),
        node('div',{class:'row wrap'},[predict,train,trainSome,library,help('Train on selected does the same with each image ticked in the list at once: each image’s saved polygons (as you fixed them) are its red examples and its green drawings its good tissue, inside its Work within; images never reviewed (no saved polygons or examples) are left out. Train model uses the selected version’s saved examples plus this image’s reviewed red and green polygons. Later versions’ examples are excluded. To reuse only the original examples, untick Include current image under Training options. Random forests are refitted on these samples; the old weights remain unchanged. GPU measurements are used automatically when available, with CPU fallback. Progress includes reading the examples; Jobs can cancel preparation or fitting. A new model is selected when training completes.')]),status,advanced,deleted);
      root.append(details);
      goodHolder.append(goodReview);
    }else root.append(status);
    const view={root,sync:()=>{
      if(!valid(e)){root.querySelectorAll('button,input,select').forEach(control=>{control.disabled=true;});status.textContent='Loading image…';return;}
      const focused=document.activeElement;
      for(const [element,v] of [[scale,e.values.scale],[minimum,e.values.minimum_um2],[holes,e.values.fill_holes_um2??0],[cut,e.values.cut]])if(element&&element!==focused)element.value=String(v);
      for(const v of sliders){
        for(const element of [v.slider,v.value]){if(element!==focused)element.value=String(v.current());element.disabled=e.read_only||(e.part==='debris'&&!e.values.detectors.includes(v.part));}
      }
      if(detectorChoice){detectorChoice.value=detectorSelection();detectorChoice.disabled=e.read_only;}
      for(const option of scale.options)option.textContent=h.scaleLabel(option.value,e.microns_per_pixel||h.pixelSize?.());
      saved.textContent=e.saveError?'Not saved':e.saving?'Saving…':JSON.stringify(e.saved)!==JSON.stringify(e.values)?'Not saved':'Saved';
      saved.className='chip '+(saved.textContent==='Saved'?'ok':'warn');retry.hidden=!e.saveError;
      const training=!e.busy&&(e.trainingBusy||e.trainingJob||e.lastAction==='training');
      status.textContent=e.saveError||(training?e.trainingMessage:e.message);
      status.title=e.saveError?'':(training?e.trainingDetails:e.details)||'';
      if(detect)detect.disabled=e.busy||e.read_only;
      if(teach){
        remaining.checked=e.values.remaining_good;
        remaining.disabled=e.read_only||e.values.include_current===false;
        include.checked=e.values.include_current!==false;include.disabled=e.read_only||e.trainingBusy||!!e.trainingJob;
        source.value=e.values.training_source||'selected';source.disabled=e.read_only||e.trainingBusy||!!e.trainingJob;
        forget.hidden=source.value!=='library'&&!!e.version;forget.disabled=e.read_only||e.trainingBusy||!!e.trainingJob;
        const drawn=h.exampleState(e.part);
        train.disabled=e.trainingBusy||!!e.trainingJob||e.modelBusy||e.read_only||(include.checked&&drawn.busy);
        train.textContent=e.trainingBusy||e.trainingJob?'Training…':'Train model';
        trainSome.disabled=e.trainingBusy||!!e.trainingJob||e.modelBusy||e.read_only;
        exampleCounts.textContent=drawn.positive+' artifacts · '+drawn.negative+' good';
        marks.forEach((button,i)=>{button.disabled=e.read_only;button.classList.toggle('active',drawn.drawing===(i?'Good tissue':'Artifact'));button.setAttribute('aria-pressed',String(drawn.drawing===(i?'Good tissue':'Artifact')));});
        const base=e.versions.find(v=>v.id===e.version),count=source.value==='selected'&&base?base.images:e.examples.length;
        // The images a model holds: training an image again replaces its examples, so the count grows only with new images.
        library.textContent=source.value==='selected'&&base?'Trained on '+count+' image'+(count===1?'':'s'):count+' reviewed image'+(count===1?'':'s');
        library.title=source.value==='selected'&&base?modelName(e,base)+' was trained on:\n'+(base.image_names||[]).join('\n')+'\n\nTraining an image again replaces its examples; tick other images and use Train on selected to add them.'
          :'The reviewed library:\n'+e.examples.map(x=>x.image).join('\n');
        const signature=JSON.stringify(e.versions.map(v=>v.id));
        if(versions.dataset.versions!==signature){versions.replaceChildren(...e.versions.map(v=>node('option',{value:v.id,text:modelName(e,v),title:v.created+' · '+v.images+' images · '+v.id+(v.parent_version?' · Parent: '+v.parent_version:'')})));versions.dataset.versions=signature;}
        versions.value=e.version;versions.disabled=e.modelBusy||e.busy||e.trainingBusy||e.read_only;
        predict.disabled=e.busy||e.modelBusy||!e.version||e.read_only;
        remove.disabled=e.modelBusy||e.busy||e.trainingBusy||!!e.trainingJob||!e.version||e.read_only;
        deleted.hidden=!e.deleted_versions?.length;
        const archived=JSON.stringify((e.deleted_versions||[]).map(v=>v.id));
        if(deletedList.dataset.versions!==archived){deletedList.replaceChildren(...(e.deleted_versions||[]).map(v=>node('option',{value:v.id,text:modelName(e,v)})));deletedList.dataset.versions=archived;}
        restore.disabled=e.modelBusy||e.trainingBusy||!!e.trainingJob||e.read_only;
      }
    }};e.views.push(view);view.sync();
  }
  async function refreshModels(){
    // A refresh started while a model is being chosen could return the earlier choice: skip it then.
    for(const e of entries.values())if(valid(e)&&e.settings&&!e.modelBusy){
      try{const sequence=e.sequence,result=await get(path(e,'state'));if(sequence!==e.sequence||e.modelBusy)continue;e.examples=result.examples;modelState(e,result);e.microns_per_pixel=result.microns_per_pixel||e.microns_per_pixel;refresh(e);if(e.trainingJob)watchTraining(e);}catch(_){}
    }
  }
  async function panel(current,holder) {
    const tabs=node('div',{class:'segmented',role:'tablist','aria-label':'Artifact type'}),area=node('div');holder.append(tabs,area);
    let render=0,shownImage=image()?.token;
    const show=async part=>{
      const number=++render;selected=part;shownImage=image()?.token;
      tabs.replaceChildren(...tabsShown.map(id=>[id,names[id]]).map(([id,name])=>node('button',{class:'btn small'+(part===id?' active':''),text:name,role:'tab','aria-selected':String(part===id),onclick:()=>show(id)})));
      current.runBar([]);
      area.replaceChildren(node('span',{class:'muted',text:'Loading controls…'}));
      try{
        const e=await load(part);if(!holder.isConnected||number!==render||!valid(e))return;
        const settings=node('section',{class:'studio-sec'}),good=node('section',{class:'studio-sec artifact-good-tissue'}),list=node('section',{class:'studio-sec artifact-polygons','data-save-in-footer':'true'});
        area.replaceChildren(settings,good,list);controls(e,settings,true,good);h.workWithin(settings,part,()=>{e.detected=false;});h.setTarget(e.layer.id);
        current.drawShapes=()=>{if(list.isConnected){h.renderLayer(list,e.layer.id);list.prepend(node('div',{class:'studio-sec-h',text:'Artifact polygons'}));}if(e.exampleList?.isConnected&&e.examplesReady)h.renderLayer(e.exampleList,e.example_layer.id);};current.drawShapes();
        current.saveAnnotationSettings=()=>save(e);current.annotationSettingsPending=()=>JSON.stringify(e.values)!==JSON.stringify(e.saved);
        current.planChanged=()=>{refreshModels();h.refreshClean();};
        const status=node('span',{class:'muted',role:'status'});
        const saveButton=node('button',{class:'btn primary',text:'Save polygons',onclick:async()=>{try{await h.savePolygons(part);}catch(error){status.textContent=error.message;}}});
        // Both kinds together: Blood clots and Image quality, each from its own Work within, one after the other (a job).
        const both=node('input',{type:'checkbox','aria-label':'Blood clots and Image quality together'});
        const subtract=node('button',{class:'btn',text:'Subtract from tissue',onclick:async()=>{
          if(e.subtracting)return;
          if(both.checked){if(window.confirm('Subtract the saved Blood clots and Image quality polygons of this image from its Work within annotation? Earlier versions stay in History.'))batch(subtract,{subtract:true,parts:['clot','debris'],open_image:true});return;}
          e.subtracting=true;subtract.disabled=true;e.subtractionMessage='Saving reviewed polygons…';status.textContent=e.subtractionMessage;
          try{
            await h.flush();
            await h.savePolygons(part);
            if(h.annotationState()[e.layer.id]?.pending)throw Error('Save the reviewed artifact polygons before subtracting.');
            const target=await post(route('/tissue_versions'),{image_token:e.image_token,stage:part});
            e.subtractionMessage='Subtracting and saving…';status.textContent=e.subtractionMessage;
            await h.updateWorkingTissue(target.layer, revision=>postFor(e,'subtract',{
              tissue_token:target.layer.token,tissue_revision:revision,selection:target.selection,
              ...h.reviewed(part)}));
            e.subtractionMessage='Tissue saved. Ctrl+Z undoes; earlier versions are in History.';
          }catch(error){e.subtractionMessage=error.message;}
          finally{e.subtracting=false;current.updatePolygonSave?.();}
        }});
        current.updatePolygonSave=()=>{const v=h.annotationState()[e.layer.id];saveButton.disabled=!v||v.busy||v.loading||e.read_only||e.subtracting;subtract.disabled=e.read_only||e.subtracting||e.busy;status.textContent=v?.error||v?.loadError||e.subtractionMessage||(v?.busy?'Saving…':v?.pending?'Not saved':v?.saved?'Saved':'');};
        // Ticked images: the same model and settings, each image inside its own Work within; one job, reviewable as
        // each image finishes (its A turns green).
        const replace=node('input',{type:'checkbox','aria-label':'Replace saved polygons'});
        const batch=async(button,body)=>{
          const images=body.open_image?[]:h.picked();
          if(!body.open_image&&!images.length){status.textContent='Tick the images first.';return;}
          if(body.find&&part==='clot'&&!e.version){status.textContent='Choose a Blood clots model first.';return;}
          button.disabled=true;
          try{
            await save(e);
            if(body.subtract){await h.flush();if(h.annotationState()[e.layer.id]?.pending)await h.savePolygons(part);}
            const answer=await postFor(e,'batch',{images,version:e.version||'',replace:replace.checked,...body});
            const what=body.parts?.length>1?'Blood clots and Image quality':names[part];
            status.textContent=(body.find?'Finding '+what+' in ':'Subtracting '+what+' from ')+(body.open_image?'this image':images.length+' image'+(images.length===1?'':'s'))+' in the background.';
            if(answer.job)h.jobs();
          }catch(error){status.textContent=error.message;}
          finally{button.disabled=false;}
        };
        const findOne=node('button',{class:'btn',text:'This image',title:'Find them in the open image now, to review before saving',
          onclick:()=>preview(e,part==='clot'||!!e.version)});
        const findSome=node('button',{class:'btn primary',text:'Selected',title:'Find them in each ticked image in the background and save them'});
        findSome.onclick=()=>batch(findSome,{find:true});
        const subtractSome=node('button',{class:'btn',text:'Selected',title:'Subtract each ticked image’s saved polygons from its Work within annotation'});
        subtractSome.onclick=()=>{const what=both.checked?'Blood clots and Image quality':names[part];
          if(window.confirm('Subtract the saved '+what+' polygons of '+h.picked().length+' ticked images from each one’s Work within annotation? Earlier versions stay in History.'))batch(subtractSome,{subtract:true,...(both.checked?{parts:['clot','debris']}:{})});};
        current.runBar([
          node('div',{class:'row wrap'},[node('span',{class:'lbl',text:'Find:'}),node('label',{title:'Off: ticked images that already have saved '+names[part]+' polygons keep them. On: they are found again (the earlier ones stay in History).'},[replace,node('span',{text:' Replace saved'})]),findOne,findSome,
            help('Finds '+names[part]+' with the model chosen above'+(part==='clot'?'':' (or, with no model, the detector settings)')+', inside each image’s Work within. This image: shown now as a preview to review, then Save polygons. Selected: each ticked image in the background; its polygons are saved as they are found, so open a finished image (its A turns green) to review it while the others run.')]),
          node('div',{class:'row wrap'},[saveButton,subtract,subtractSome,node('label',{class:'inline',title:'Subtract both kinds at once: Blood clots and Image quality'},[both,node('span',{text:' Both kinds'})]),help('Save polygons keeps this image’s reviewed polygons. Subtract from tissue cuts them out of the annotation chosen in Work within and saves it for the next step; Selected does the same for each ticked image with its saved polygons. Both kinds: Blood clots and Image quality together, each from its own Work within (in the background). Ctrl+Z undoes; History keeps earlier versions.')]),status]);
        current.updatePolygonSave();
        if(e.trainingJob)watchTraining(e);
      }catch(error){if(holder.isConnected&&number===render)area.replaceChildren(node('div',{class:'row wrap'},[node('span',{class:'hint',text:names[part]+' could not load: '+error.message}),node('button',{class:'btn small',text:'Retry',onclick:()=>show(part)})]));}
    };
    current.imageReady=()=>{if(holder.isConnected&&shownImage!==image()?.token)show(selected);};
    show(selected);
  }
  function renderLayer(layer,body){
    if(!layer.artifact_part)return;
    const e=entry(layer.artifact_part);
    const title=layer.artifact_part==='clot'?'Prediction settings':'Detection settings';
    if(e?.settings){const details=node('details',{class:'artifact-layer-settings'});details.open=!!e.rightOpen;details.ontoggle=()=>{if(details.isConnected)e.rightOpen=details.open;};details.append(node('summary',{text:title}));controls(e,details,false);body.append(details);}
    else body.append(node('button',{class:'btn small',text:title,onclick:async()=>{try{await load(layer.artifact_part);h.rebuild();}catch(error){body.append(node('span',{class:'hint',text:error.message}));}}}));
  }
  return {panel,renderLayer,refreshModels,sync:()=>{for(const e of entries.values())if(e.settings)refresh(e);}};
};
