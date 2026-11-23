/* Shared-column gland composition review. Analysis is a Studio job; figures and
 * exact cell selections reuse the existing right panel, image viewer and Data. */
window.createGlandPanel = h => {
  'use strict';
  const {node, field, choice, help, section, chip, get, post, route} = h;
  const api=(action,body={})=>post(route('/spatial/gland_'+action),body);
  const number=(value,label,min=0,step=1)=>node('input',{type:'number',value,min,step,'aria-label':label});
  const fmt=v=>v===null||v===undefined?'—':Number(v).toLocaleString(undefined,{maximumSignificantDigits:3});
  const details=(title,tip)=>node('details',{class:'studio-options'},[node('summary',{},[node('span',{text:title}),help(tip)])]);
  const grid=(...children)=>node('div',{class:'gland-grid'},children);
  const kHelp={interior_k:'How many gland composition types to make. K=6 groups glands into six types based on the cell subtypes inside them. These are composition types, not confirmed genetic clones.',exterior_k:'How many surrounding niche types to make. K=4 groups exterior beds into four types based on their immune and stromal cell mix.'};
  const reviewHelp='Tries K=2–8 separately for glands and niches, regardless of the manual K settings. Higher separation scores mean more distinct groups. Review does not change settings or cell labels; use a suggested K, then Fit to apply it.';
  const checkbox=(title,value,change,tip)=>{const tick=node('input',{type:'checkbox','aria-label':title});tick.checked=value;tick.onchange=()=>change(tick.checked);return node('label',{class:'inline'},[tick,node('span',{text:title}),tip?help(tip):null].filter(Boolean));};
  function picker(title,options,selected,change){
    let picked=new Set(selected);const summary=node('summary');
    const box=node('details',{class:'threshold-picker','aria-label':title},[summary]);
    const search=node('input',{type:'search',placeholder:'Find…','aria-label':'Find '+title});
    const list=node('div',{class:'threshold-checks'});
    const update=()=>{summary.textContent=picked.size+' selected';summary.title=[...picked].join(', ');change([...picked]);};
    const redraw=()=>{list.replaceChildren(...options.map(name=>{
      const row=checkbox(name,picked.has(name),on=>{on?picked.add(name):picked.delete(name);update();});
      row.hidden=!name.toLowerCase().includes(search.value.toLowerCase());return row;
    }));};
    search.oninput=redraw;
    box.append(search,node('div',{class:'row'},['All','None'].map(label=>node('button',{class:'btn small',text:label,onclick:()=>{
      options.filter(n=>n.toLowerCase().includes(search.value.toLowerCase())).forEach(n=>label==='All'?picked.add(n):picked.delete(n));update();redraw();
    }}))),list);
    summary.textContent=picked.size+' selected';summary.title=[...picked].join(', ');redraw();return box;
  }
  async function mount(current,pane,actionRow){
    const initial=await get(route('/spatial/gland_settings'));if(!pane.isConnected)return;
    let saved=initial, values=JSON.parse(JSON.stringify(initial.values)), busy=false, generation=0, selectionSequence=0, currentJob='', result=null;
    const silhouetteRequests=new Map();
    const alive=()=>pane.isConnected;
    const said=node('div',{class:'hint',role:'status','aria-live':'polite'}), badge=chip('Saved','ok');
    const mark=()=>{const dirty=JSON.stringify(values)!==JSON.stringify(saved.values);badge.textContent=dirty?'Not saved':'Saved';badge.className='chip '+(dirty?'warn':'ok');};
    const save=async(draft=JSON.parse(JSON.stringify(values)))=>{saved=await api('settings',{values:draft,revision:saved.revision});mark();};
    const config=section('Gland types',{tip:'Group anatomical glands by their cell-subtype proportions, and their exterior beds by cell-type proportions. These are composition types, not proven genetic clones. Small nonempty structures are predicted after smoothing; empty beds stay unassigned.',actions:[badge]});
    config.box.querySelector('.studio-sec-h').append(node('button',{class:'btn small',text:'Save',onclick:()=>save().then(()=>said.textContent='Settings saved.').catch(e=>said.textContent=e.message)}));
    config.box.classList.add('gland-config');
    const cols=initial.columns;
    const column=(key,title,optional=false,tip='')=>{
      const input=choice([...(optional?[['','None']]:[['','Choose a column…']]),...cols.map(c=>[c,c])],values[key],title);
      input.onchange=()=>{values[key]=input.value;mark();};return field(title,input,tip);
    };
    const mappings=details('Annotation columns','A gland ID must be positive, or a nonempty text ID. IDs are local to each image, tissue piece and compartment. Bed ID refers to its gland ID. An optional bed flag selects the cells belonging to that bed. All gland interiors are excluded from exterior counts.');
    const mapRows=node('div'), scopeBox=node('div');
    const drawMappings=()=>{
      mapRows.replaceChildren(...values.regions.map((r,i)=>{
        const name=node('input',{value:r.name,'aria-label':'Compartment name '+(i+1)});
        name.onchange=()=>{const old=r.name;r.name=name.value.trim();values.scope=values.scope.map(v=>v===old?r.name:v);drawScope();mark();};
        const row=grid(field('Compartment',name),node('button',{class:'btn small',text:'Remove',onclick:()=>{values.regions.splice(i,1);values.scope=values.scope.filter(v=>v!==r.name);drawMappings();mark();}}));
        for(const [key,label,optional] of [['roi_col','Gland ID',false],['bed_id_col','Bed ID',true],['bed_col','Bed flag',true]]){
          const input=choice([['',optional?'None':'Choose…'],...cols.map(c=>[c,c])],r[key]||'',label+' '+r.name);
          input.onchange=()=>{r[key]=input.value;mark();};row.append(field(label,input));
        }
        const flag=node('input',{value:r.bed_value??'1','aria-label':'Bed flag value '+r.name});flag.onchange=()=>{r.bed_value=flag.value;mark();};row.append(field('Flag value',flag));
        return row;
      }));drawScope();
    };
    const drawScope=()=>scopeBox.replaceChildren(field('Fit scope',picker('Gland compartments',values.regions.map(r=>r.name),values.scope,v=>{values.scope=v;mark();}), 'Choose one or several compartments. A unified fit uses the same type definitions for all selected compartments.'));
    mappings.append(mapRows,node('button',{class:'btn small',text:'Add compartment',onclick:()=>{const name='Compartment '+(values.regions.length+1);values.regions.push({name,roi_col:'',bed_id_col:'',bed_col:'',bed_value:'1'});values.scope.push(name);drawMappings();mark();}}));
    drawMappings();if(!values.regions.length)mappings.open=true;
    config.body.append(mappings,scopeBox);
    const featureGrid=grid();
    async function featurePanel(key,classes,title){
      const box=node('div'), select=choice([['','Choose a column…'],...cols.map(c=>[c,c])],values[key],title+' column'), list=node('div');
      let request=0;
      const load=async(reset=false)=>{
        const seq=++request;if(!select.value){list.replaceChildren();return;}
        list.textContent='Reading labels…';
        try{const found=await api('options',{column:select.value});if(!alive()||seq!==request)return;
          if(reset||!values[classes].length)values[classes]=found.values.filter(v=>!['none','nan','null','unassigned',''].includes(v.toLowerCase()));
          list.replaceChildren(picker(title+' classes',found.values,values[classes],v=>{values[classes]=v;mark();}));mark();
        }catch(e){if(seq===request)list.textContent=e.message;}
      };
      select.onchange=()=>{values[key]=select.value;values[classes]=[];mark();load(true);};
      box.append(field(title,select,title==='Interior'?'Subtypes of the gland cells. Only selected labels enter its composition.':'Cell types outside all glands, with a matching bed ID. Include stromal/endothelial classes if relevant; changing the selection changes the denominator.'),list);featureGrid.append(box);load();
    }
    featurePanel('interior_col','interior_classes','Interior');featurePanel('exterior_col','exterior_classes','Exterior');config.body.append(featureGrid);
    const roles=details('Cell roles and comparison columns','Image and tissue-piece IDs keep different coordinate spaces separate. Independent sample is a person/animal, not an image or a visit. Visit keeps repeated samples separate; a comparison uses one burden per sample and visit. Comparison/strata columns are optional and are not used to fit types.');
    roles.append(grid(column('image_col','Image ID'),column('piece_col','Tissue piece',true),column('cell_id_col','Cell ID'),column('label_col','Segmentation label',true),column('x_col','X'),column('y_col','Y')));
    const parent=column('parent_col','Parent cells',true,'Optional restriction for interior counts, such as epithelial cells. Gland membership still comes from annotation IDs.');
    const parentValues=node('div');let parentSeq=0;
    const fillParent=async()=>{const seq=++parentSeq;parentValues.replaceChildren();if(!values.parent_col)return;try{const found=await api('options',{column:values.parent_col});if(alive()&&seq===parentSeq)parentValues.append(picker('Parent cell types',found.values,values.parent_values,v=>{values.parent_values=v;mark();}));}catch(e){said.textContent=e.message;}};
    parent.querySelector('select').addEventListener('change',()=>{values.parent_values=[];fillParent();});fillParent();roles.append(grid(parent,parentValues));
    const metadata=grid(column('sample_col','Independent sample',true,'Choose the person/animal ID. Visits are kept separate by Visit; repeated images contribute to that sample.'),column('visit_col','Visit',true),column('group_col','Compare by',true,'Choose the grouping column, such as response or treatment. Groups must be consistent within an independent sample.'),column('stratum_col','Stratify by',true));
    const compareChoices=node('div');let metaSeq=0;
    const fillCompare=async()=>{const seq=++metaSeq;compareChoices.replaceChildren();try{
      for(const [key,labels] of [['group_col',[['group_a','Group A'],['group_b','Group B']]],['visit_col',[['baseline','Before'],['followup','After']]]]){
        if(!values[key])continue;const found=await api('options',{column:values[key]});if(!alive()||seq!==metaSeq)return;
        const row=grid();for(const [setting,label] of labels){const pick=choice([['','Choose…'],...found.values.map(v=>[v,v])],values[setting],label);pick.onchange=()=>{values[setting]=pick.value;mark();};row.append(field(label,pick));}compareChoices.append(row);
      }
    }catch(e){said.textContent=e.message;}};
    metadata.querySelector('[aria-label="Compare by"]').addEventListener('change',()=>{values.group_a='';values.group_b='';mark();fillCompare();});metadata.querySelector('[aria-label="Visit"]').addEventListener('change',()=>{values.baseline='';values.followup='';mark();fillCompare();});const minSamples=number(values.min_samples,'Minimum samples per group',2,1);minSamples.onchange=()=>{values.min_samples=Number(minSamples.value);mark();};roles.append(metadata,compareChoices,field('Samples / group',minSamples,'Minimum independent samples per group for a test. Smaller groups remain visible without a p-value.'));fillCompare();config.body.append(roles);
    const fitting=details('Fit settings','The prior adds this many cells with the pooled selected-cell composition. log1p(cell count) weights each well-sampled structure in fitting and standardization. Small glands are retained for prediction. Auto compares K=2…8 using a reproducible silhouette sample (up to 2,000 structures); it suggests a useful separation, not a proven biological optimum.');fitting.open=true;
    const parameters=grid(), kInputs={};
    for(const [key,title,min,step] of [['min_gland','Gland cells to fit',1,1],['min_bed','Bed cells to fit',1,1],['prior','Prior strength',0,.5],['interior_k','Gland types (K)',1,1],['exterior_k','Niche types (K)',1,1],['seed','Seed',0,1]]){
      const input=number(values[key],title,min,step);input.onchange=()=>{values[key]=Number(input.value);mark();};parameters.append(field(title,input,kHelp[key]||''));if(kHelp[key])kInputs[key]=input;
    }
    fitting.append(parameters,checkbox('Use size-weighted Z-score',values.standardize,v=>{values.standardize=v;mark();},'Balances variation between composition features after smoothing. Z indicates relative enrichment, not biological positivity. Turning this off fits smoothed proportions.'),
      grid(checkbox('Auto gland K',values.auto_k,v=>{values.auto_k=v;mark();},'During Fit, choose the best-scoring tested K for glands instead of the manual Gland types (K).'),checkbox('Auto niche K',values.auto_bed_k,v=>{values.auto_bed_k=v;mark();},'During Fit, choose the best-scoring tested K for niches instead of the manual Niche types (K).')));
    config.body.append(fitting);
    const outputs=details('Output columns','Assignments are saved beside the source tables and appear in Cell data and Data. Original CSV/Parquet files are unchanged; use Jobs to write/export a copy. Gland type labels all cells within its gland; niche labels only assigned exterior cells. Empty beds stay blank.');
    for(const [key,label] of [['output_col','Gland type'],['niche_col','Bed niche']]){const input=node('input',{value:values[key],'aria-label':label+' output'});input.onchange=()=>{values[key]=input.value.trim();mark();};outputs.append(field(label,input));}
    let overwrite=false;outputs.append(checkbox('Overwrite gland results',false,v=>overwrite=v));config.body.append(outputs);
    const results=section('Results',{tip:'Profiles show the fitted type definitions. Explore selects exact cells in the existing image viewer and Data. Comparisons summarise independent samples; they do not treat glands as independent patients.'});
    const runSelect=choice([['','Choose a saved run…']],'','Gland saved run');const body=node('div');results.body.append(runSelect,body);pane.append(config.box,results.box);
    const refreshRuns=async(select)=>{const found=await api('runs');if(!alive())return;const chosen=select||currentJob||new URLSearchParams(location.search).get('gland_run');runSelect.replaceChildren(node('option',{value:'',text:'Choose a saved run…'}),...found.runs.map(r=>node('option',{value:r.id,text:r.label+' · '+r.glands+' glands'})));runSelect.value=chosen||found.runs[0]?.id||'';if(runSelect.value)await loadResult(runSelect.value);};
    runSelect.onchange=()=>loadResult(runSelect.value);
    async function start(images,notice,diagnose=false){
      if(busy)throw Error('A gland run is already starting in this panel.');busy=true;notice.className='hint gland-running';notice.textContent=diagnose?'Reviewing K…':'Fitting gland types…';
      try{
        const submitted=JSON.parse(JSON.stringify(values));await save(submitted);const started=await post(route('/spatial/start'),{kind:'gland_types',values:submitted,images,overwrite,diagnose});h.refreshJobs?.();
        const id=started.job?.id||started.job;
        for(;;){await new Promise(r=>setTimeout(r,2000));if(!alive())return;
          const jobs=(await get(route('/jobs'))).jobs||[];const job=jobs.find(j=>j.id===id);if(!job)continue;
          notice.textContent=job.message||'Working…';
          if(job.status==='done'){notice.className='hint gland-done';notice.textContent=diagnose?'K review ready; no columns changed.':'Gland types ready.';await refreshRuns(id);if(!diagnose)await current.refreshSpatialInputs?.();h.refreshJobs?.();break;}
          if(['failed','cancelled'].includes(job.status))throw Error(job.message||'The run stopped.');
        }
      }finally{busy=false;}
    }
    const runbar=actionRow('Fit:',(images,notice)=>start(images,notice),'Fit on the selected images together, then predict smaller nonempty glands and beds. Results remain linked to their source tables.',said);
    const auto=node('button',{class:'btn small',text:'Review optimal K',title:reviewHelp,onclick:async()=>{
      const images=[...(current.spatialPicked||[])];if(!images.length){said.textContent='Tick the images to use for K review.';return;}
      auto.disabled=true;try{await start(images,said,true);}catch(e){said.className='hint';said.textContent=e.message;}finally{auto.disabled=false;}
    }});runbar.append(auto,help(reviewHelp));current.runBar([runbar,said]);
    async function loadResult(id){
      const seq=++generation;if(!id){body.replaceChildren();return;}currentJob=id;
      try{const found=await api('result',{job_id:id});if(!alive()||seq!==generation)return;result=found;
        const metrics=node('div',{class:'row wrap'},[chip(found.glands.toLocaleString()+' glands'),chip(found.beds.toLocaleString()+' nonempty beds')]);
        for(const kind of ['interior','exterior']){const m=found.models[kind];if(m)metrics.append(chip((found.diagnose?'Reviewed ':'Fitted ')+(kind==='interior'?'gland':'niche')+' K='+m.k,'',m.fitted+' fit structures; '+m.predicted_small+' small predictions. Separation score '+fmt(m.silhouette)+' on '+m.silhouette_n+' structures.'));}
        const tabs=node('div',{class:'segmented',role:'tablist','aria-label':'Gland result view'}), content=node('div');
        const show=async(which)=>{const viewSeq=++generation;content.replaceChildren();tabs.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.view===which));try{
          if(which==='K review')reviewK(content,viewSeq);else if(which==='Fit quality')fitQuality(content,viewSeq);else if(which==='Profiles')await profiles(content,viewSeq);else if(which==='Explore')await explore(content,viewSeq);else await comparisons(content,viewSeq);
        }catch(e){if(viewSeq===generation)content.textContent=e.message;}};
        const hasReview=found.diagnose||Object.values(found.models).some(m=>m?.diagnostics?.length>1);
        for(const label of [...(hasReview?['K review']:[]),'Fit quality','Profiles',...(found.diagnose?[]:['Explore','Compare'])])tabs.append(node('button',{class:'btn small','data-view':label,text:label,onclick:()=>show(label)}));
        const notes=details('Run details','Settings, model diagnostics and source provenance are kept with each run.');
        notes.append(node('div',{class:'hint',text:(found.notes||[]).join(' ')}),node('div',{class:'hint',text:'Selected images: '+found.images.length+' · '+(found.diagnose?'K review only':'Linked columns: '+found.columns.join(', '))}));
        const downloads=node('div',{class:'row wrap'});for(const [file,label] of [['gland_structures.csv','Gland table'],['gland_models.json','Fitted models'],['sample_burdens.csv','Sample burdens'],['clinical_tests.csv','Comparison statistics']])downloads.append(node('button',{class:'btn small',text:label,onclick:()=>download(file)}));
        body.replaceChildren(metrics,tabs,content,notes,downloads);
        const requested=new URLSearchParams(location.search).get('gland_structure');
        if(requested){const url=new URL(location.href);url.searchParams.delete('gland_structure');url.searchParams.delete('gland_run');history.replaceState(null,'',url);await show('Explore');const list=await api('structures',{job_id:id});const row=list.rows.find(r=>r.structure===requested);if(row)await reveal(row,'gland',generation);}
        else show(found.diagnose?'K review':'Profiles');
      }catch(e){if(seq===generation)body.textContent=e.message;}
    }
    async function download(file){try{const response=await fetch(h.apiUrl(route('/spatial/gland_download')),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:currentJob,file})});if(!response.ok)throw Error((await response.json()).error);const url=URL.createObjectURL(await response.blob()), a=node('a',{href:url,download:file});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){said.textContent=e.message;}}
    function reviewK(content,seq){
      content.append(node('div',{class:'hint',text:'Saved K scores · use a suggestion, then Fit to apply it.'}));
      for(const kind of ['interior','exterior']){
        const review=result.k_reviews[kind],title=kind==='interior'?'Gland types · interior':'Niche types · exterior';
        const apply=()=>{
          if(seq!==generation||!alive())return;
          const key=kind==='interior'?'interior_k':'exterior_k',automatic=kind==='interior'?'auto_k':'auto_bed_k';
          values[key]=review.suggested_k;values[automatic]=false;kInputs[key].value=review.suggested_k;
          fitting.querySelector('[aria-label="Auto '+(kind==='interior'?'gland':'niche')+' K"]').checked=false;mark();
          said.className='hint';said.textContent=(kind==='interior'?'Gland':'Niche')+' K set to '+review.suggested_k+'. Press Fit to create results.';
        };
        const open=(reveal=false)=>{
          if(seq!==generation||!alive())return;
          const box=section(title+' · K review',{tip:reviewHelp+' Scores range from −1 to 1. Near zero means overlapping groups. Scores use '+review.scored+' well-sampled structures from this saved run, not necessarily the current controls.'});
          const summary=review.suggested_k?'Suggested K = '+review.suggested_k+' · best tested score '+fmt(review.score):'No K recommendation';
          box.body.append(node('div',{class:'row wrap'},[chip(summary,review.suggested_k?'ok':''),node('button',{class:'btn small',text:'Use K='+review.suggested_k,hidden:!review.suggested_k,onclick:apply}),node('button',{class:'btn small',text:'Save PNG',onclick:()=>download(kind+'_k_review.png')})]));
          const svg=(name,attrs={},text='')=>{const el=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text)el.textContent=text;return el;};
          const chart=svg('svg',{viewBox:'0 0 520 285',class:'gland-k-chart',role:'img','aria-label':title+' K separation scores'});
          chart.append(svg('title',{},summary),svg('desc',{},'One bar per tested K. Green marks the highest score; higher is better. '+review.notes.join(' ')));
          const valid=review.rows.filter(r=>r.silhouette!==null&&Number.isFinite(r.silhouette)),scores=valid.map(r=>r.silhouette);
          const lo=Math.min(0,...scores),hi=Math.max(.05,...scores),pad=Math.max(.04,(hi-lo)*.18),bottom=lo<0?lo-pad:0,top=hi+pad;
          const y=v=>225-(v-bottom)/(top-bottom)*190,step=420/Math.max(1,review.rows.length);
          for(let i=0;i<=4;i++){const v=bottom+(top-bottom)*i/4,py=y(v);chart.append(svg('line',{x1:70,x2:490,y1:py,y2:py,stroke:'currentColor',opacity:.15}),svg('text',{x:62,y:py+4,'text-anchor':'end'},v.toFixed(2)));}
          chart.append(svg('line',{x1:70,x2:490,y1:y(0),y2:y(0),stroke:'currentColor',opacity:.5}));
          review.rows.forEach((r,i)=>{
            const x=70+step*(i+.5),ok=r.silhouette!==null&&Number.isFinite(r.silhouette);
            if(ok){const py=y(r.silhouette),bar=svg('rect',{x:x-step*.3,y:Math.min(py,y(0)),width:step*.6,height:Math.max(1,Math.abs(py-y(0))),fill:r.k===review.suggested_k?'#7cdb97':'#5b8de8','data-k':r.k,'data-score':r.silhouette});bar.append(svg('title',{},'K='+r.k+' · score '+r.silhouette.toFixed(3)));chart.append(bar);}
            chart.append(svg('text',{x,y:ok?(r.silhouette>=0?y(r.silhouette)-8:y(r.silhouette)+16):120,'text-anchor':'middle'},ok?r.silhouette.toFixed(3):'—'),svg('text',{x,y:244,'text-anchor':'middle'},String(r.k)));
          });
          chart.append(svg('text',{x:280,y:274,'text-anchor':'middle'},'Number of types (K)'),svg('text',{x:280,y:20,'text-anchor':'middle'},'Higher score = clearer groups'),svg('text',{transform:'translate(17,133) rotate(-90)','text-anchor':'middle'},'Silhouette score'));
          const plots=node('div',{class:'gland-review-charts'}),distribution=node('div',{class:'gland-silhouette-wrap'});
          plots.append(node('div',{},[chart]),distribution);box.body.append(plots);
          silhouetteChart(distribution,kind,seq,currentJob);
          for(const note of review.notes)box.body.append(node('div',{class:'hint',text:note}));
          h.showFigure('gland-'+kind,box.box);if(reveal)h.revealFigure?.('gland-'+kind);
        };
        content.append(node('div',{class:'row wrap'},[node('button',{class:'btn small',text:title+' chart',onclick:()=>open(true)}),node('button',{class:'btn small',text:'Use '+(kind==='interior'?'gland':'niche')+' K='+review.suggested_k,hidden:!review.suggested_k,onclick:apply})]));open();
      }
    }
    function fitQuality(content,seq){
      for(const kind of ['interior','exterior']){
        const title=kind==='interior'?'Gland types · fit quality':'Niche types · fit quality';
        const open=()=>{if(seq!==generation||!alive())return;const box=section(title,{tip:'Individual silhouette scores for this saved model. Higher means a structure resembles its own type more than other types.'});
          silhouetteChart(box.body,kind,seq,currentJob);h.showFigure('gland-'+kind,box.box);};
        content.append(node('button',{class:'btn small',text:title,onclick:open}));open();
      }
    }
    async function silhouetteChart(host,kind,seq,job){
      const key=job+':'+kind,valid=()=>alive()&&seq===generation&&job===currentJob&&host.isConnected;
      host.append(node('div',{class:'hint',text:'Loading individual scores…',role:'status'}));
      try{
        if(!silhouetteRequests.has(key))silhouetteRequests.set(key,api('silhouette',{job_id:job,kind}).catch(e=>{silhouetteRequests.delete(key);throw e;}));
        const data=await silhouetteRequests.get(key);if(!valid())return;
        const tip='Each dot is one '+(kind==='interior'?'gland':'exterior bed')+'. The thick line covers the middle 50%; the dark tick is the median. Below zero means closer to another type on average. Near zero means overlapping types; a singleton has score zero by convention. Uses the same random sample of up to 2,000 fit structures as the K chart, with equal weight per scored structure. Small predicted structures are excluded. An unsampled type is not scored. This describes the saved K, not an un-applied draft K, and is not accuracy or independent validation.';
        const actions=node('div',{class:'row wrap'},[node('strong',{text:'Individual silhouette scores'}),help(tip)]);
        const status=node('div',{class:'row wrap'},[chip('Saved K='+data.k),chip(data.scored.toLocaleString()+'/'+data.fitted.toLocaleString()+' fit structures scored')]);
        if(data.mean!==null)status.append(chip('Mean '+data.mean.toFixed(3)));
        host.replaceChildren(actions,status);
        if(data.state!=='ok'){host.append(node('div',{class:'hint gland-silhouette-empty',text:data.message}));return;}
        const svg=(name,attrs={},text='')=>{const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;return e;};
        const width=680,left=190,right=540,row=64,top=26,height=top+row*data.groups.length+48;
        const x=v=>left+(v+1)*(right-left)/2;
        const chart=svg('svg',{viewBox:'0 0 '+width+' '+height,class:'gland-silhouette-chart',role:'img','aria-label':(kind==='interior'?'Gland':'Niche')+' individual silhouette scores by type'});
        chart.append(svg('title',{},'Individual silhouette scores at saved K='+data.k),svg('desc',{},tip));
        chart.append(svg('rect',{x:left,y:top-12,width:x(0)-left,height:row*data.groups.length,fill:'#d66060',opacity:.08}));
        chart.append(svg('line',{x1:x(0),x2:x(0),y1:top-12,y2:height-40,stroke:'currentColor','stroke-dasharray':'4 4',opacity:.6}));
        data.groups.forEach((group,i)=>{
          const y=top+row*(i+.5),colour=result.palette[group.name]||'#5b8de8';
          const name=svg('text',{x:left-10,y:y-7,'text-anchor':'end',class:'gland-silhouette-name'});
          const parts=group.name.split(' · '),signature=parts.slice(1).join(' · ');
          name.append(svg('tspan',{x:left-10},parts[0]),svg('tspan',{x:left-10,dy:18},signature.length>25?signature.slice(0,24)+'…':signature),svg('title',{},group.name));chart.append(name);
          chart.append(svg('line',{x1:left,x2:right,y1:y,y2:y,stroke:'currentColor',opacity:.1}));
          const points=data.points.filter(p=>p.type===group.name);
          points.forEach((p,j)=>{const jitter=(((j*73)%101)/100-.5)*22,dot=svg('circle',{cx:x(p.silhouette),cy:y+jitter,r:2.8,fill:colour,opacity:.5,'data-score':p.silhouette,'data-structure':p.structure});
            dot.append(svg('title',{},group.name+' · '+p.image+' · '+p.structure+' · silhouette '+p.silhouette.toFixed(3)));chart.append(dot);});
          if(points.length){const [q1,median,q3]=group.quartiles;chart.append(svg('line',{x1:x(q1),x2:x(q3),y1:y,y2:y,stroke:colour,'stroke-width':7}),svg('line',{x1:x(median),x2:x(median),y1:y-9,y2:y+9,stroke:'#101620','stroke-width':2}));}
          chart.append(svg('text',{x:right+10,y:y-5},'n='+group.scored),svg('text',{x:right+10,y:y+14},points.length?group.negative_percent.toFixed(1)+'% < 0':'Not sampled'));
        });
        for(const v of [-1,-.5,0,.5,1])chart.append(svg('text',{x:x(v),y:height-24,'text-anchor':'middle'},String(v)));
        chart.append(svg('text',{x:(left+right)/2,y:height-3,'text-anchor':'middle'},'Silhouette score'));
        host.append(node('div',{class:'gland-silhouette-scroll'},[chart]),node('div',{class:'row wrap'},[node('span',{class:'hint',text:'Dots: structures · Line: middle 50% · Tick: median'}),
          node('button',{class:'btn small',text:'Save PNG',onclick:()=>{if(valid())download(kind+'_silhouette.png');}}),
          node('button',{class:'btn small',text:'Scores CSV',onclick:()=>{if(valid())download(kind+'_silhouette.csv');}})]));
      }catch(e){if(valid())host.replaceChildren(node('div',{class:'hint',text:e.message}),node('button',{class:'btn small',text:'Retry scores',onclick:()=>{host.replaceChildren();silhouetteChart(host,kind,seq,job);}}));}
    }
    async function profiles(content,seq){
      const profileJob=currentJob;
      for(const kind of ['interior','exterior']){
        const profile=result.profiles[kind];if(!profile){content.append(chip(kind+': no fitted types','warn'));continue;}
        const open=()=>{
          const meaning=kind==='interior'?'A gland is an annotated epithelial structure. Each row is a TYPE of gland with a similar interior cell mix. The percentage in its name is the average measured share of that subtype in the fit glands; individual glands vary.':'A bed is the assigned area outside a gland. Each row is a niche TYPE with a similar surrounding cell mix. Its two named cell types stand out relative to other niches; other cell types can still be present.';
          const box=section(kind==='interior'?'Gland types · interior':'Bed niches · exterior',{tip:meaning+' '+profile.note+' Click a row name to colour its assigned cells in the open image and zoom to a local example. If absent, the image stays unchanged. A K review must be fitted before its labels can be displayed.'});
          const notice=node('div',{class:'hint gland-view-status',role:'status','aria-live':'polite'});
          const picked=new Set();
          const flag=node('button',{class:'btn small',text:'Flag for review',disabled:true,title:'Select types to flag across this run’s images.',onclick:async()=>{flag.disabled=true;try{await window.SpacemapReview.flagResult(profileJob,[...picked],kind);}finally{flag.disabled=!picked.size||result.diagnose;}}});
          const table=node('table',{class:'studio-counts gland-profiles'});table.append(node('tr',{},[node('th',{text:'Type'}),...profile.columns.map(n=>node('th',{text:n}))]));
          const limit=Math.max(.1,...profile.z.flat().map(Math.abs));
          profile.rows.forEach((name,i)=>{const button=node('button',{class:'studio-item-name','data-gland-type':name,'data-gland-kind':kind,text:name,onclick:()=>showType(kind,name,notice,profileJob)});
            const tick=node('input',{type:'checkbox','aria-label':'Select '+name,onchange:()=>{tick.checked?picked.add(name):picked.delete(name);flag.disabled=!picked.size||result.diagnose;}});
            table.append(node('tr',{},[node('td',{},[tick,button]),...profile.columns.map((_,j)=>{
              const z=profile.z[i][j],alpha=Math.abs(z)/limit;
              return node('td',{text:(100*profile.raw[i][j]).toFixed(1)+'%',title:'Relative Z '+z.toFixed(2)+' · measured proportion '+(100*profile.raw[i][j]).toFixed(1)+'%',style:'background:rgba('+(z>0?'190,55,65':'45,110,165')+','+(alpha*.85)+');text-align:center'});
            })]));
          });box.body.append(node('div',{class:'row wrap'},[chip('Z −'+limit.toFixed(1)),node('span',{style:'width:90px;height:10px;background:linear-gradient(to right,#2d6ea5,#222,#be3741)'}),chip('Z +'+limit.toFixed(1)),help(profile.note),flag]),node('div',{class:'studio-pairs-scroll'},[table]),node('button',{class:'btn small',text:'Save PNG',onclick:()=>download(kind+'_profiles.png')}),notice);
          h.showFigure('gland-'+kind,box.box);
        };
        content.append(node('button',{class:'btn small',text:kind==='interior'?'Interior profiles':'Exterior profiles',onclick:open}));open();
      }
    }
    async function showType(kind,name,notice,jobId=currentJob){
      if(jobId!==currentJob||!alive())return;
      const image=h.openRoi(),seq=generation,request=++selectionSequence;
      const current=()=>alive()&&seq===generation&&request===selectionSequence&&currentJob===jobId&&h.openRoi()===image;
      if(result.diagnose){notice.textContent='This is a K review. Press Fit to assign types before showing them on the image.';return;}
      if(!image){notice.textContent='Open an image to view this type.';return;}
      notice.textContent='Showing '+name+'…';
      try{
        let payload;
        const shown=await h.showReviewedCells(async()=>{payload=await api('type_cells',{job_id:jobId,image,kind,name});return payload;},current);
        if(!shown||!current())return;
        if(shown.missing){notice.textContent=shown.message;return;}
        if(payload.bounds)h.zoomBox?.(payload.bounds);
        notice.textContent=name+' · '+shown.current.toLocaleString()+' cells in '+payload.structures+' '+(kind==='interior'?'glands':'beds')+' · '+image;
      }catch(e){if(current())notice.textContent=e.message;}
    }
    async function reveal(row,role,seq,jobId=currentJob){
      if(jobId!==currentJob||!alive())return;
      const request=++selectionSequence;
      const current=()=>alive()&&seq===generation&&request===selectionSequence&&jobId===currentJob&&h.openRoi()===row.image;
      try{if(h.openRoi()!==row.image){await h.openTableImage(row.image);await h.imageSettled?.();}if(!current())return;
        let payload;
        const shown=await h.showReviewedCells(async()=>{payload=await api('cells',{job_id:jobId,structure:row.structure,role});return payload;},current);
        if(shown&&current()){if(payload.bounds)h.zoomBox?.(payload.bounds);said.textContent=payload.name+' · '+shown.current.toLocaleString()+' cells · '+row.image;}
      }catch(e){if(current())said.textContent=e.message;}
    }
    async function explore(content,seq,filter=null){
      const found=await api('structures',{job_id:currentJob});if(seq!==generation||!alive())return;
      let rows=found.rows;const exploreJob=currentJob,checked=new Set();
      const flagStructures=role=>node('button',{class:'btn small',text:role==='gland'?'Flag glands for review':'Flag niches for review',onclick:async()=>{if(!checked.size){info.textContent='Select glands first.';return;}await window.SpacemapReview.flagResult(exploreJob,[...checked],role);}});
      const pick=choice(found.images.map(i=>[i,i]),found.images.includes(h.openRoi())?h.openRoi():found.images[0],'Gland image');
      const search=node('input',{type:'search',placeholder:'Find a gland, type or sample…','aria-label':'Find gland'});
      const tableBox=node('div',{class:'studio-pairs-scroll'}),info=node('div',{class:'hint',role:'status'});let sort=filter?filter.kind+'_distance':'n_interior',ascending=!!filter,mapSequence=0;
      const filtered=()=>rows.filter(r=>r.image===pick.value&&(!filter||r[filter.kind==='interior'?'gland_type':'niche']===filter.name)&&JSON.stringify(r).toLowerCase().includes(search.value.toLowerCase())).sort((a,b)=>{const av=a[sort]??'',bv=b[sort]??'';return(typeof av==='number'&&typeof bv==='number'?av-bv:String(av).localeCompare(String(bv),undefined,{numeric:true}))*(ascending?1:-1);});
      const draw=()=>{const visible=filtered();info.textContent=visible.length.toLocaleString()+' glands';const table=node('table',{class:'studio-counts gland-structures'});
        const headings=[['local_id','Gland'],['piece','Piece'],['compartment','Compartment'],['n_interior','Cells'],['gland_type','Type'],['n_exterior','Bed cells'],['niche','Niche'],['interior_status','Status']];
        const all=node('input',{type:'checkbox','aria-label':'Select listed glands'});all.checked=!!visible.length&&visible.every(r=>checked.has(r.structure));all.indeterminate=visible.some(r=>checked.has(r.structure))&&!all.checked;all.onchange=()=>{visible.forEach(r=>all.checked?checked.add(r.structure):checked.delete(r.structure));draw();};
        table.append(node('tr',{},[node('th',{},[all]),...headings.map(([key,title])=>node('th',{},[node('button',{class:'studio-item-name',text:title+(sort===key?(ascending?' ↑':' ↓'):''),onclick:()=>{ascending=sort===key?!ascending:true;sort=key;draw();}})])),node('th',{text:'Review'})]));
        for(const r of visible.slice(0,500)){const tick=node('input',{type:'checkbox','aria-label':'Select gland '+r.local_id,onchange:()=>{tick.checked?checked.add(r.structure):checked.delete(r.structure);draw();}});tick.checked=checked.has(r.structure);table.append(node('tr',{'data-gland':r.structure},[node('td',{},[tick]),...headings.map(([key])=>node('td',{text:r[key]??'—',title:String(r[key]??'')})),node('td',{},[
          node('button',{class:'btn small',text:'Interior',onclick:()=>reveal(r,'gland',generation,exploreJob)}),node('button',{class:'btn small',text:'Bed',disabled:!r.n_exterior,onclick:()=>reveal(r,'bed',generation,exploreJob)})])]));}
        tableBox.replaceChildren(table);if(visible.length>500)info.textContent+=' · first 500; narrow the search';
      };
      const drawMaps=async()=>{const request=++mapSequence;const data=await api('maps',{job_id:currentJob,image:pick.value});if(request!==mapSequence||seq!==generation||!alive()||!content.isConnected)return;
        const fig=section('Glands and exterior beds · '+data.image,{tip:'Both maps show the same image coordinates. Grey is context or unassigned cells. Click a coloured cell to select its entire gland or bed in the image viewer and Data. Drawing uses a fixed sample of up to 25,000 cells; fitting and selection use all cells.'});
        const panels=grid();for(const [role,title] of [['gland','Interior gland types'],['bed','Exterior niches']]){
          const canvas=node('canvas',{width:600,height:500,'aria-label':title,style:'width:100%;background:#14171d;cursor:crosshair'}),points=data.points;
          const xs=points.map(p=>p.x),ys=points.map(p=>p.y),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),scale=Math.min(560/Math.max(1,x1-x0),460/Math.max(1,y1-y0));
          const ctx=canvas.getContext('2d'),positions=[];for(const p of [...points.filter(p=>!p[role+'_name']),...points.filter(p=>p[role+'_name'])]){const x=20+(p.x-x0)*scale,y=20+(p.y-y0)*scale;ctx.fillStyle=data.palette[p[role+'_name']]||'#666c76';ctx.globalAlpha=p[role+'_name']?.length?1:.25;ctx.fillRect(x,y,2,2);if(p[role+'_id'])positions.push([x,y,p[role+'_id']]);}ctx.globalAlpha=1;
          canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*canvas.width/rect.width,y=(e.clientY-rect.top)*canvas.height/rect.height;let nearest=null,d=100;for(const p of positions){const dist=(p[0]-x)**2+(p[1]-y)**2;if(dist<d){nearest=p;d=dist;}}if(nearest){const row=rows.find(r=>r.structure===nearest[2]);if(row)reveal(row,role,generation,exploreJob);}};
          panels.append(node('div',{},[node('div',{class:'lbl',text:title}),canvas]));
        }
        fig.body.append(panels,chip(data.points.length.toLocaleString()+' displayed / '+data.total.toLocaleString()+' cells'));
        const legend=node('div',{class:'spatial-legend'}),notice=node('div',{class:'hint gland-view-status',role:'status','aria-live':'polite'});
        Object.entries(data.palette).forEach(([name,color])=>{const kind=result.profiles.interior?.rows.includes(name)?'interior':'exterior';legend.append(node('button',{class:'btn small spatial-key',title:'Show this type in the open image',onclick:()=>showType(kind,name,notice,exploreJob)},[node('span',{class:'split-marker-swatch',style:'background:'+color}),node('span',{text:name})]));});fig.body.append(legend,notice);h.showFigure('gland-maps',fig.box);
      };
      pick.onchange=()=>{draw();drawMaps().catch(e=>said.textContent=e.message);};search.oninput=draw;
      const example=node('button',{class:'btn small',text:'View closest fit example',title:'Well-sampled structure closest to its fitted type centre; not a biological ground truth.',onclick:async()=>{
        const eligible=rows.filter(r=>!filter||(r[filter.kind==='interior'?'gland_type':'niche']===filter.name&&r[filter.kind+'_status']==='Fit'));
        const key=(filter?.kind||'interior')+'_distance';eligible.sort((a,b)=>(a[key]??Infinity)-(b[key]??Infinity));const r=eligible[0];if(!r)return;pick.value=r.image;draw();await drawMaps();await reveal(r,filter?.kind==='exterior'?'bed':'gland',seq);
      }});
      content.append(field('Image',pick),search,node('div',{class:'row wrap'},[flagStructures('gland'),flagStructures('bed')]),...(filter?[example]:[]),...(filter?[node('button',{class:'btn small',text:'Clear type filter',onclick:()=>{filter=null;draw();}})]:[]),info,tableBox);draw();await drawMaps();
    }
    async function comparisons(content,seq){
      const compartment=choice([['','All fitted compartments'],...result.values.scope.map(v=>[v,v])],'','Comparison compartment');let split=false,comparison=null,request=0;
      const controls=node('div'),view=choice([],'','Comparison visit'),stratum=choice([['All','All']],'All','Comparison stratum'),tableBox=node('div',{class:'studio-pairs-scroll'}),note=node('div',{class:'hint'});
      const load=async()=>{const token=++request;comparison=await api('compare',{job_id:currentJob,compartment:compartment.value,strata:split});if(!alive()||seq!==generation||token!==request)return;
        note.textContent=comparison.tests.length?'':comparison.notes.join(' ');view.replaceChildren(...[...new Set(comparison.tests.map(r=>r.view))].map(v=>node('option',{value:v,text:v})));stratum.replaceChildren(...[...new Set(comparison.tests.map(r=>r.stratum))].map(v=>node('option',{value:v,text:v})));draw();
      };
      let sort='q',asc=true;
      const draw=()=>{const tests=(comparison?.tests||[]).filter(r=>r.view===view.value&&r.stratum===stratum.value).sort((a,b)=>{if(a[sort]==null)return 1;if(b[sort]==null)return -1;return(typeof a[sort]==='number'?a[sort]-b[sort]:String(a[sort]).localeCompare(String(b[sort])))*(asc?1:-1);});
        const table=node('table',{class:'studio-counts'}),heads=[['feature','Type'],['higher','Higher in'],['n_a','A n'],['n_b','B n'],['median_a','A median'],['median_b','B median'],['mean_a','A mean'],['mean_b','B mean'],['p','p'],['q','q'],['state','State']];table.append(node('tr',{},heads.map(([key,title])=>node('th',{},[node('button',{class:'studio-item-name',text:title,onclick:()=>{asc=sort===key?!asc:true;sort=key;draw();}})]))));
        tests.forEach(r=>table.append(node('tr',{},heads.map(([key])=>node('td',{text:typeof r[key]==='number'?fmt(r[key]):r[key]??'—',title:r.state})))));tableBox.replaceChildren(table);
      };
      const plot=async()=>{try{const token=++request;const found=await api('compare',{job_id:currentJob,compartment:compartment.value,strata:split,view:view.value,stratum:stratum.value,figure:true});if(!alive()||seq!==generation||token!==request)return;const f=section('Sample burdens · '+view.value,{tip:found.notes.join(' ')});f.body.append(node('img',{class:'studio-figure-preview',src:found.figure,alt:'Patient-level burden comparison with p and FDR q values'}),node('a',{class:'btn small',href:found.figure,download:'gland_comparison.png',text:'Save PNG'}));h.showFigure('gland-comparison',f.box);trajectories(found);}catch(e){note.textContent=e.message;}};
      const trajectories=found=>{const rows=found.trajectories||[];if(!rows.length)return;const f=section('Within-sample composition across visits',{tip:'Each sample/visit bar totals 100%. Glands are weighted by log1p(interior cells); niches by log1p(exterior cells). Missing visits stay missing. This is change in composition, not evidence that individual glands were tracked.'});
        for(const kind of ['interior','exterior']){const samples=[...new Set(rows.filter(r=>r.kind===kind&&(stratum.value==='All'||r.stratum===stratum.value)).map(r=>r.sample))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));const holder=node('div',{class:'gland-trajectories'});holder.append(node('div',{class:'lbl',text:kind==='interior'?'Gland types':'Bed niches'}));for(const sample of samples){const visits=[...new Set(rows.filter(r=>r.sample===sample&&r.kind===kind).map(r=>r.visit))].sort((a,b)=>{const order=[result.values.baseline,result.values.followup];const rank=v=>order.includes(v)?order.indexOf(v):2;return rank(a)-rank(b)||a.localeCompare(b,undefined,{numeric:true});});for(const visit of visits){const values=rows.filter(r=>r.sample===sample&&r.visit===visit&&r.kind===kind);holder.append(node('div',{class:'spatial-bar-row'},[node('span',{class:'spatial-bar-label',text:sample+' · '+visit}),node('div',{class:'spatial-bar-track'},[node('div',{class:'spatial-bar',style:'width:100%'},values.filter(v=>v.value>0).map(v=>node('span',{class:'spatial-segment',style:'flex:'+v.value+';background:'+(result.palette[v.feature]||'#888'),title:v.feature+': '+fmt(v.value)+'% ('+v.group+')'})))]) ]));}}f.body.append(holder);}h.showFigure('gland-trajectories',f.box);};
      compartment.onchange=()=>load().catch(e=>note.textContent=e.message);view.onchange=stratum.onchange=()=>{draw();plot();};
      controls.append(field('Compartment',compartment),checkbox('Show strata as well',split,v=>{split=v;load().catch(e=>note.textContent=e.message);},'Compare the same two groups separately within each value of the configured stratification column. FDR includes pooled and stratified comparisons together.'),grid(field('Visit / change',view),field('Stratum',stratum)),node('button',{class:'btn small',text:'Show comparison figures',onclick:plot}));
      content.append(chip('A: '+(result.values.group_a||'Choose a group')+' · B: '+(result.values.group_b||'Choose a group')),controls,tableBox,note);await load();if(comparison?.tests.length)await plot();
    }
    refreshRuns().catch(e=>said.textContent=e.message);
  }
  return {mount};
};
