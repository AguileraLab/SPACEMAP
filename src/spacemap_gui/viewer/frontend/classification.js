/* One review state drives the class list, map and heatmap. */
window.createClassification = h => {
  'use strict';
  const {node,field,choice,help,section,get,post,route}=h;
  const panel = async (current,holder) => {
    const image=h.image();if(!image)return;
    // Another image opened while Classification is shown: the panel is made again for it (its classes, its token).
    current.imageReady=()=>{if(holder.isConnected&&h.image()&&h.image().token!==image.token){holder.replaceChildren();current.runBar([]);h.showRaster('classes',null);panel(current,holder);}};
    const alive=()=>holder.isConnected&&h.image()?.token===image.token;
    const {box,body}=section('Classification',{tip:'MiniBatch K-means groups pixels by marker enrichment inside Work within. Scale is the actual classification resolution. Display colours and thresholds do not change the underlying measurements.'});
    const status=node('p',{class:'muted',role:'status'}),list=node('div',{class:'classification-list'});
    const scale=choice([1,2,4,8,16,32].map(n=>[String(n),h.scaleLabel(n)]),'4','Classification scale');
    const selected=node('input',{type:'checkbox','aria-label':'Selected markers only'});
    const count=node('input',{type:'number',min:'2',max:'40',step:'1',value:'20','aria-label':'Number of classes'});
    const top=node('input',{type:'number',min:'1',max:'100',step:'1',value:'3','aria-label':'Top markers to show'});
    const opacity=node('input',{type:'range',min:'0',max:'100',value:'60','aria-label':'Class map opacity'});
    const texture=node('input',{type:'checkbox','aria-label':'Local texture'}),edges=node('input',{type:'checkbox','aria-label':'Marker edges'});
    const topk=node('input',{type:'checkbox','aria-label':'Emphasize strongest markers'});topk.checked=true;
    const smooth=node('input',{type:'number',min:'0',max:'5',step:'.1',value:'.8','aria-label':'Classification smoothing'});
    const automaticAI=node('input',{type:'checkbox','aria-label':'Suggest names automatically'});
    // Shared classes: a class model fitted on several images; with one chosen, Classify uses its classes (the same
    // numbers and names in every image) instead of finding new ones per image.
    const model=node('select',{'aria-label':'Class model'});let models=[];
    const modelNote=node('p',{class:'muted'});
    const featureOptions=node('details',{class:'studio-options'},[node('summary',{text:'Features'}),
      node('span',{class:'muted',text:'Marker intensity is always used.'}),
      ...[[texture,'Local texture','Is the marker even or patchy around the pixel? Separates smooth areas (stroma) from grainy ones (dense cells). Adds one value per marker, so it weighs as much as the intensity.'],
        [edges,'Marker edges','Is the pixel at a border where a marker changes? Helps outline structures (vessels, glands), but can make borders a class of their own. Adds one value per marker.'],
        [topk,'Emphasize strongest markers','Groups pixels by their few brightest markers only and ignores the rest. Gives cleaner, easier-to-name classes. Recommended.']]
        .map(([input,text,tip])=>node('div',{class:'row'},[node('label',{class:'row'},[input,node('span',{text})]),help(tip)])),
      field('Smoothing',smooth,'Blurs each marker a little first (in pixels at the chosen Scale). 0: every pixel on its own, noisy. 1 to 2: smoother, larger classes.')]);
    body.append(field('Classes from',model,'Each image its own: Classify finds new classes for each image (numbers and names differ between images). A shared model: the classes found once on several images (Shared classes: Selected below); every image classified with it has the same classes, and a name given on one image is its name on all of them. The settings below then come from the model (choose Each image its own to change them for a new model).'),modelNote,
      field('Scale',scale,'Resolution used to measure and classify pixels, and to save their polygons.'),
      field('Markers',node('label',{class:'row'},[selected,node('span',{text:'Selected markers only'})]),
        'Unchecked: all image channels, including CODEX and the three H&E channels. Checked: only the image markers currently turned on in Layers. Display thresholds are ignored.'),
      field('Classes',count,'Number of pixel groups to discover inside the chosen annotation.'),featureOptions,
      field('Top markers',top,'How many enriched markers to show per cluster. The heatmap and Assistant receive the complete profile.'),
      field('Opacity',opacity,'Opacity of the classification overlay.'),
      node('label',{class:'row'},[automaticAI,node('span',{text:'Suggest names automatically'}),help('Requests reviewable naming and merge suggestions after each new classification, using your configured Assistant. It never applies names automatically.')]),status);
    const classes=section('Classes',{tip:'Show a class, change its colour, then type a name and press ✓ or Enter to accept. Names, selections and merges are shared with the heatmap.'});
    classes.body.append(list);holder.append(box,classes.box);
    let result=null,review=null,busy=false,queued=false,reset=false,heat=null,picked=[],suggestions=null,readonly=false;
    let knownOptions='',lastAI='',disposed=false,hiddenIds=[];
    const valid=()=>alive()&&!disposed;
    const opts=()=>({scale:Number(scale.value),n_clusters:Number(count.value),selected_only:selected.checked,
      markers:selected.checked?h.selectedMarkers():[],features:[...(texture.checked?['texture']:[]),...(edges.checked?['edges']:[])],
      top_k:topk.checked,smoothing:Number(smooth.value)});
    const hidden=()=>hiddenIds;
    const settingInputs=()=>[scale,count,selected,texture,edges,topk,smooth];
    const setOptions=o=>{scale.value=String(o.scale);count.value=String(o.n_clusters);selected.checked=o.selected_only;
      texture.checked=o.features.includes('texture');edges.checked=o.features.includes('edges');topk.checked=o.top_k;smooth.value=String(o.smoothing);};
    const drawModels=(found)=>{
      models=found.models||[];
      model.replaceChildren(node('option',{value:'',text:'Each image its own classes'}),...models.map(m=>node('option',{value:m.id,
        text:m.name+' · '+m.n_clusters+' classes · '+m.images+' image'+(m.images===1?'':'s')+(m.scale_by==='slide'?' · per slide':''),title:'Fitted on '+m.images+' images, '+new Date(m.created).toLocaleString()})));
      model.value=found.selected||'';
      const chosen=models.find(m=>m.id===model.value);
      if(chosen)setOptions(chosen.options);
      settingInputs().forEach(input=>input.disabled=!!chosen);
      modelNote.textContent=chosen?'Settings come from the model. Classify uses its classes; names given here apply to every image classified with it.':'';
    };
    model.onchange=async()=>{
      try{drawModels(await post(route('/classification'),{action:'select_model',image_token:image.token,class_model:model.value}));
        status.textContent=model.value?'Shared classes chosen: press Classify (This image or Selected) to use them.':'Each image gets its own classes again.';}
      catch(error){status.textContent=error.message;}
    };
    const classify=node('button',{class:'btn',text:'This image',title:'Classify the open image now',onclick:()=>run('run',true)});
    const batch=node('button',{class:'btn primary',text:'Selected',title:'Classify each ticked image in the background',onclick:()=>classifyPicked()});
    const save=node('button',{class:'btn',text:'This image',title:'Save the reviewed classes of the open image as polygons',onclick:()=>run('save')});
    const saveSome=node('button',{class:'btn primary',text:'Selected',title:'Save the reviewed classes of each ticked image as polygons, in the background',onclick:()=>savePicked()});
    const scaleBy=choice([['image','Scale each image'],['slide','Scale each slide']],'image','Scale within');
    const fitSome=node('button',{class:'btn primary',text:'Selected',title:'Find one set of classes on the ticked images together, then classify each of them with it',onclick:()=>fitPicked()});
    const area=(label,tip)=>{const input=node('input',{type:'number',min:'0',step:'any',value:'0','aria-label':label,title:tip,class:'studio-area-input'});return input;};
    const hole=area('Smallest hole','Holes inside a class smaller than this (µm²) are filled with the class. Tissue holes are kept. 0: keep every hole.');
    const island=area('Smallest island','Pieces of a class smaller than this (µm²) join the class around them. 0: keep every piece.');
    const tidy=()=>({hole_um2:Math.max(0,Number(hole.value)||0),island_um2:Math.max(0,Number(island.value)||0)});
    const askAll=node('button',{class:'btn small',text:'Ask AI',title:'Suggest names and possible merges for all clusters',onclick:()=>ask()});
    const merge=node('button',{class:'btn small',text:'Merge selected',onclick:()=>mergeClasses(picked)});
    classes.body.prepend(node('div',{class:'row wrap'},[merge,askAll]));
    const runStatus=node('p',{class:'muted',role:'status'});
    current.runBar([node('div',{class:'row'},[node('span',{class:'lbl',text:'Classify:'}),classify,batch,
        help('Groups the pixels with the settings above. This image: now, shown at once. Selected: each ticked image in the background, inside its own Work within; open a finished image (its C turns green) to name its classes while the others run. Changing a setting does not classify again until you press one of these.')]),
      node('div',{class:'row'},[node('span',{class:'lbl',text:'Shared classes:'}),scaleBy,fitSome,
        help('Scale each image: every marker is scaled within each image’s own tissue (as Classification does). Scale each slide: within all the ticked images of a slide together (the slide is the name before _roi), so a slide stained brighter than the others is brought to the same levels while an image with only one tissue keeps its level among its slide’s images. Each slide: marker intensity only (no texture or edges).'),
        help('Finds one set of classes on the ticked images together (a fair sample of pixels from each, each image scaled within its own Work within), keeps it as a class model, chooses it above, and classifies every ticked image with it. Class 3 is then the same marker profile in every image, and you name the classes once. Use more classes than for one image (12 to 15), then merge similar ones while naming. Each image also gets checks: pixels that fit no class well, and classes whose profile differs there.')]),
      node('div',{class:'row wrap'},[node('span',{class:'lbl',text:'Save polygons:'}),node('label',{class:'inline'},[node('span',{text:'Smallest hole'}),hole,node('span',{class:'muted',text:'µm²'})]),
        node('label',{class:'inline'},[node('span',{text:'Smallest island'}),island,node('span',{class:'muted',text:'µm²'})]),save,saveSome,
        help('Writes the reviewed classes as polygons (names, merges and colours as shown; excluded classes left out): of the open image, or of each ticked image from its last classification (with shared classes, the model’s names). Smallest island: smaller pieces of a class join the class around them. Smallest hole: smaller holes inside a class are filled. Names and review edits save by themselves.')]),runStatus]);
    const syncButtons=()=>{classify.disabled=busy;batch.disabled=busy;fitSome.disabled=busy||readonly;saveSome.disabled=busy||readonly;save.disabled=busy||readonly||!result;askAll.disabled=busy||!result;merge.disabled=busy||readonly||picked.length<2;};
    const request=async(action,extra={})=>post(route('/classification'),{action,image_token:image.token,options:opts(),tidy:tidy(),
      ...(result?{run_id:result.run_id,revision:result.revision,scope_revision:result.scope_revision,polygon_revision:result.polygon_revision}:{}),
      review,hidden:hidden(),class_model:model.value,...extra});
    function mergeClasses(ids){
      if(busy||readonly||ids.length<2)return;
      const items=result.classes.filter(c=>ids.includes(c.id)),members=items.flatMap(c=>c.members).sort((a,b)=>a-b);
      const first=members[0];
      members.forEach(id=>{if(!(id in review.original_names))review.original_names[id]=review.names[id]||'';});
      review.groups=review.groups.filter(g=>!g.some(i=>members.includes(i)));review.groups.push(members);
      review.names[first]=items.map(c=>c.name).join(' + ').slice(0,120);review.sealed=false;picked=[first];hiddenIds=[];run('review');
    }
    function unmerge(item){
      review.groups=review.groups.filter(g=>!g.includes(item.id));item.members.forEach(id=>{review.names[id]=review.original_names[id]||'';delete review.original_names[id];});picked=[];hiddenIds=[];run('review');
    }
    function accept(id,name){review.names[id]=name.trim();if(suggestions)suggestions.names=suggestions.names.filter(s=>s.id!==id);review.sealed=false;run('review');}
    function pick(ids){picked=ids.map(Number);hiddenIds=picked.length?result.classes.filter(c=>!picked.includes(c.id)).map(c=>c.id):[];draw();paint();}
    // The class map coloured here from its cluster numbers (sent once per run), so showing or picking a class is instant.
    let index=null,shownUrl=null;
    async function loadIndex(found){
      if(!found.index||index?.run_id===found.run_id)return;
      const picture=new Image();picture.src=found.index;await picture.decode();
      const canvas=document.createElement('canvas');canvas.width=picture.width;canvas.height=picture.height;
      const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(picture,0,0);
      index={run_id:found.run_id,canvas,context,source:context.getImageData(0,0,canvas.width,canvas.height)};
    }
    function paint(){
      if(!index||!result||index.run_id!==result.run_id)return false;
      const table=new Uint8ClampedArray(256*4);
      for(const item of result.classes){
        if(item.excluded||hiddenIds.includes(item.id))continue;
        const colour=[1,3,5].map(i=>parseInt(item.colour.slice(i,i+2),16));
        for(const member of item.members){const k=(member+1)*4;table[k]=colour[0];table[k+1]=colour[1];table[k+2]=colour[2];table[k+3]=255;}
      }
      const source=index.source.data,out=new ImageData(index.source.width,index.source.height),target=out.data;
      for(let i=0;i<source.length;i+=4){if(!source[i+3])continue;const k=source[i]*4;target[i]=table[k];target[i+1]=table[k+1];target[i+2]=table[k+2];target[i+3]=table[k+3];}
      index.context.putImageData(out,0,0);
      index.canvas.toBlob(blob=>{if(!blob||!valid())return;const url=URL.createObjectURL(blob),old=shownUrl;shownUrl=url;
        h.showRaster('classes',{url,opacity:Number(opacity.value)/100});if(old)setTimeout(()=>URL.revokeObjectURL(old),10000);});
      return true;
    }
    function draw(){
      if(!valid()||!result)return;
      list.replaceChildren(...[sharedNote(result)].filter(Boolean),...result.classes.map(item=>{
        const tick=node('input',{type:'checkbox','aria-label':'Show class '+item.id});tick.checked=!hiddenIds.includes(item.id);
        tick.onchange=()=>{hiddenIds=tick.checked?hiddenIds.filter(id=>id!==item.id):[...hiddenIds,item.id];picked=result.classes.filter(c=>!hiddenIds.includes(c.id)).map(c=>c.id);draw();paint();};
        const color=node('input',{type:'color',value:item.colour,'aria-label':'Colour of class '+item.id});color.disabled=readonly;
        color.onchange=()=>{review.colors[item.id]=color.value;review.sealed=false;run('review');};
        const markers=item.top_markers.slice(0,Math.max(1,Number(top.value)||3)).join(', ')||'Low enrichment';
        const name=node('input',{type:'text',value:item.name,placeholder:'Class name',maxlength:'120','aria-label':'Name of class '+item.id});name.disabled=readonly;
        const apply=node('button',{class:'btn small ghost',text:'✓',title:'Accept this class name','aria-label':'Accept name of class '+item.id,onclick:()=>accept(item.id,name.value)});apply.disabled=readonly;
        name.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();apply.click();}};
        const ai=node('button',{class:'btn small',text:'Ask AI',onclick:()=>ask([item.id]),title:'Ask the Assistant using this cluster, all marker profiles and the parent annotation'});
        const split=item.members.length>1?node('button',{class:'btn small',text:'↶',title:'Unmerge into original clusters','aria-label':'Unmerge class '+item.id,disabled:review.sealed||readonly,onclick:()=>unmerge(item)}):null;
        const exclude=node('input',{type:'checkbox','aria-label':'Exclude class '+item.id,title:'Exclude this class from saved polygons'});exclude.checked=item.excluded;exclude.disabled=readonly;
        exclude.onchange=()=>{const values=new Set(review.excluded);item.members.forEach(i=>exclude.checked?values.add(i):values.delete(i));review.excluded=Array.from(values);review.sealed=false;run('review');};
        const row=node('div',{class:'classification-row'+(item.excluded?' removed':''),'data-class':String(item.id)},[tick,color,
          node('span',{class:'mono',text:item.members.join('+'),title:item.count.toLocaleString()+' pixels · '+item.percent.toFixed(1)+'%'}),
          node('span',{class:'classification-markers',text:markers,title:item.top_markers.join(', ')}),name,apply,ai,split,exclude]);
        const proposal=suggestions?.names?.find(s=>s.id===item.id);
        if(proposal){name.value=proposal.name;name.classList.add('ai-proposal');name.title='AI suggestion: '+proposal.reason+' Press ✓ to accept.';}
        return row;
      }));
      const name=node('input',{type:'text',placeholder:'Name selected clusters…',maxlength:'120','aria-label':'Heatmap class name'});
      if(picked.length===1)name.value=result.classes.find(c=>c.id===picked[0])?.name||'';
      const suggestion=picked.length===1&&suggestions?.names?.find(p=>p.id===picked[0]);
      if(suggestion){name.value=suggestion.name;name.title=suggestion.reason;name.classList.add('ai-proposal');}
      const rename=node('button',{class:'btn small',text:'✓',title:'Accept name for selected clusters',disabled:readonly||!picked.length,onclick:()=>{picked.forEach(id=>review.names[id]=name.value.trim());if(suggestions)suggestions.names=suggestions.names.filter(p=>!picked.includes(p.id));review.sealed=false;run('review');}});
      name.onkeydown=e=>{if(e.key==='Enter')rename.click();};
      const tools=node('div',{class:'classification-heat-tools'},[node('div',{class:'row'},[name,rename,
        node('button',{class:'btn small',text:'Ask AI',onclick:()=>ask(picked)}),
        node('button',{class:'btn small',text:'Merge',disabled:readonly||picked.length<2,onclick:()=>mergeClasses(picked)})])]);
      const one=picked.length===1&&result.classes.find(c=>c.id===picked[0]);
      if(one?.members?.length>1)tools.append(node('button',{class:'btn small',text:'Unmerge',disabled:review.sealed||readonly,onclick:()=>unmerge(one)}));
      if(suggestions?.merges?.length)tools.append(node('details',{},[node('summary',{text:'Suggested merges'}),...suggestions.merges.map(p=>node('div',{class:'row'},[
        node('span',{class:'muted grow',text:p.ids.join(' + '),title:p.reason}),node('button',{class:'btn small',text:'Merge',disabled:readonly,onclick:()=>mergeClasses(p.ids)})]))]));
      heat=h.heatmap({key:'classes',title:'Classification',profile:result.profile,picked:picked.map(String),tools,
        onAggregation:async aggregation=>{if(busy)return;review.aggregation=aggregation;await run('review');},
        labels:Object.fromEntries(result.classes.map(c=>[String(c.id),c.name])),colours:Object.fromEntries(result.classes.map(c=>[String(c.id),c.colour])),
        onPick:pick,tip:'Every row is a pixel class within the chosen annotation. Ctrl-click several rows to merge them. Names and merges also update the left list.'});
      syncButtons();
    }
    async function ask(ids=[]){
      if(busy||!result)return;
      busy=true;syncButtons();status.textContent='Asking the Assistant…';
      try{
        if(ids.length){picked=ids;hiddenIds=result.classes.filter(c=>!ids.includes(c.id)).map(c=>c.id);draw();paint();await new Promise(r=>setTimeout(r,200));}
        const images=[];try{images.push(h.capture());}catch(_){}
        // The exact map accompanies the live viewer snapshot even while tiles load.
        if(paint())images.push(index.canvas.toDataURL());
        h.aiMessage('Review '+(ids.length?'cluster '+ids.join(', '):'all clusters')+' in '+(h.scopeName()||'the selected annotation')+'.',true);
        const response=await request('suggest',{selected:ids,images,question:'Suggest names for '+(ids.length?'clusters '+ids.join(', '):'all clusters')+' and possible merges; compare all profiles.'});
        if(!valid()||response.run_id!==result.run_id)return;
        suggestions=response.suggestions;
        h.aiMessage(suggestions.say+'\n'+suggestions.names.map(p=>'Cluster '+p.id+': '+p.name+' — '+p.reason).join('\n')+
          (suggestions.images_used?'':'\nThe configured model used marker measurements; it does not support images.'));
        status.textContent='Suggestions ready. Press ✓ beside a name to accept.';draw();
      }catch(error){status.textContent=error.message;h.aiMessage(error.message);}
      finally{busy=false;syncButtons();}
    }
    // The image's last classification in this Work within, as saved (here or by a batch job); nothing is clustered.
    async function show(){
      if(!valid())return;
      if(busy){queued=false;}
      busy=true;syncButtons();status.textContent='Loading the saved classes…';
      try{
        const found=await post(route('/classification'),{action:'load',image_token:image.token,options:opts(),tidy:tidy(),class_model:model.value});
        if(!valid())return;
        if(found.none){result=null;review=null;index=null;picked=[];hiddenIds=[];list.replaceChildren();h.showRaster('classes',null);
          status.textContent='Not classified in this area yet. Check the settings, then press Classify.';return;}
        result=found;review=structuredClone(found.review);knownOptions=JSON.stringify(opts());picked=[];hiddenIds=[];suggestions=null;
        const o=found.options;if(o&&!model.value){scale.value=String(o.scale);count.value=String(o.n_clusters);selected.checked=o.selected_only;
          texture.checked=o.features.includes('texture');edges.checked=o.features.includes('edges');topk.checked=o.top_k;smooth.value=String(o.smoothing);knownOptions=JSON.stringify(opts());}
        try{await loadIndex(found);}catch(_){index=null;}
        if(!valid())return;
        if(!paint())h.showRaster('classes',{url:found.png,opacity:Number(opacity.value)/100});
        status.textContent=found.classes.length+' classes · '+found.pixels.toLocaleString()+' pixels · saved run';
        draw();
      }catch(error){status.textContent=error.message;}
      finally{busy=false;syncButtons();}
    }
    async function classifyPicked(){
      const images=h.picked?.()||[];
      if(!images.length){runStatus.textContent='Tick the images first (tick the header box for all).';return;}
      batch.disabled=true;
      try{
        await post(route('/classification'),{action:'batch',image_token:image.token,images,options:opts(),class_model:model.value});
        runStatus.textContent='Classifying '+images.length+' image'+(images.length===1?'':'s')+(model.value?' with the shared classes':'')+' in the background, one after another. Open a finished image (its C turns green) to name its classes meanwhile.';
        h.jobs?.();
      }catch(error){runStatus.textContent=error.message;}
      finally{batch.disabled=busy;}
    }
    async function fitPicked(){
      const images=h.picked?.()||[];
      if(images.length<2){runStatus.textContent='Tick at least two images (the header box ticks all).';return;}
      fitSome.disabled=true;
      try{
        await post(route('/classification'),{action:'fit',image_token:image.token,images,options:opts(),scale_by:scaleBy.value});
        runStatus.textContent='Finding shared classes on '+images.length+' images, then classifying each with them, in the background. The new model is chosen above when it is ready.';
        h.jobs?.();
      }catch(error){runStatus.textContent=error.message;}
      finally{syncButtons();}
    }
    async function savePicked(){
      const images=h.picked?.()||[];
      if(!images.length){runStatus.textContent='Tick the images first (the header box ticks all).';return;}
      saveSome.disabled=true;
      try{
        await h.flush?.();
        await post(route('/classification'),{action:'save_batch',image_token:image.token,images,tidy:tidy()});
        runStatus.textContent='Saving the classes of '+images.length+' image'+(images.length===1?'':'s')+' as polygons in the background. Earlier versions stay in History.';
        h.jobs?.();
      }catch(error){runStatus.textContent=error.message;}
      finally{syncButtons();}
    }
    // What the shared classes say about this image: whose classes they are, and the checks.
    function sharedNote(found){
      const shared=found?.class_model;
      if(!shared)return model.value?node('p',{class:'hint',text:'This image’s classes are its own. Press Classify: This image to use the shared classes.'}):null;
      const lines=['Shared classes: '+shared.name+'. Names, colours and merges here apply to every image classified with it.'];
      if(shared.far_percent>=2)lines.push(shared.far_percent+'% of this image’s pixels fit no class well: it may have a tissue the other images do not.');
      if(shared.differ.length)lines.push('Differs here from the other images: '+shared.differ.map(d=>d.name).join(', ')+'. Check these classes on this image.');
      return node('div',{class:'classification-shared'},lines.map((text,i)=>node('p',{class:i?'hint':'muted',text})));
    }
    async function run(action='run',fresh=false){
      if(!valid())return;
      if(busy){queued=true;reset ||= fresh;return;}
      busy=true;syncButtons();status.textContent=action==='save'?'Saving polygons…':'Classifying…';
      try{
        await h.flush?.();
        const changed=JSON.stringify(opts())!==knownOptions;
        const resetReview=fresh||changed;
        const found=await request(action,{...(resetReview?{review:null,hidden:[]}:{} )});
        if(!valid())return;
        const newRun=result?.run_id!==found.run_id;
        if(newRun){picked=[];hiddenIds=[];suggestions=null;}
        result=found;review=structuredClone(found.review);knownOptions=JSON.stringify(opts());
        try{await loadIndex(found);}catch(_){index=null;}
        if(!valid())return;
        if(!paint())h.showRaster('classes',{url:found.png,opacity:Number(opacity.value)/100});
        if(action==='run')h.refreshPlan?.();
        if(found.saved_polygons){await h.showPolygons(found.saved_polygons);h.showRaster('classes',null);}
        runStatus.textContent=found.saved_polygons?'Saved '+found.saved_polygons.count+' polygons'+(tidy().hole_um2||tidy().island_um2?' (holes below '+tidy().hole_um2+' µm² filled, islands below '+tidy().island_um2+' µm² joined)':'')+'. Earlier versions are in History.':runStatus.textContent;
        status.textContent=found.saved_polygons?'Saved '+found.saved_polygons.count+' polygons. Earlier versions are in History.':
          found.classes.length+' classes · '+found.pixels.toLocaleString()+' pixels · '+found.seconds+' s'+(found.execution.device==='cuda'?' · GPU':'');
        draw();
        if(action==='run'&&!readonly&&newRun){const saved=await request('review');if(valid()&&result.run_id===saved.run_id){result.revision=saved.revision;}}
      }catch(error){status.textContent=error.message;}
      finally{busy=false;syncButtons();if(queued){queued=false;const fresh=reset;reset=false;run('run',fresh);}else if(automaticAI.checked&&result&&lastAI!==result.run_id){lastAI=result.run_id;ask();}}
    }
    // Settings never classify by themselves: Classify does.
    [scale,count,selected,texture,edges,topk,smooth].forEach(input=>input.onchange=()=>{status.textContent='Settings changed: press Classify (This image or Selected) to use them.';});
    top.onchange=draw;opacity.oninput=()=>h.showRaster('classes',{opacity:Number(opacity.value)/100});
    automaticAI.onchange=()=>{if(automaticAI.checked)ask();};
    syncButtons();
    await h.workWithin(body,'classes',()=>show());
    try{
      const initial=await get(route('/classification'));if(!valid())return;readonly=initial.read_only;
      Array.from(scale.options).forEach(option=>option.textContent=h.scaleLabel(Number(option.value),initial.microns_per_pixel));
      setOptions(initial.options);
      drawModels(initial.class_models||{models:[],selected:''});
      // A shared-classes job may add and choose a model, or classify the open image: show both when it does.
      current.planChanged=async()=>{
        try{
          const before=model.value,last=result?.run_id;
          const again=await get(route('/classification'));if(!valid())return;
          drawModels(again.class_models||{models:[],selected:''});
          if(!busy&&(model.value!==before||(again.run_id&&again.run_id!==last)))await show();
        }catch(_){}
      };
      await show();
    }catch(error){status.textContent=error.message.includes('404')?'This Studio process predates Classification. Start the updated launcher and open its new address. Your AI model can stay running.':error.message;}
  };
  return panel;
};
