/* Flow gating: one revisioned workspace, docked or in a separate browser window. */
(() => {
  'use strict';
  const clone=value=>JSON.parse(JSON.stringify(value));
  // getRandomValues also works on Studio's ordinary HTTP network addresses.
  // randomUUID is secure-context-only and crashes before the UI on those URLs.
  const uid=()=>Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');
  const colours=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc949','#af7aa1','#ff9da7'];
  const node=(tag,attrs={},children=[])=>{const e=document.createElement(tag);Object.entries(attrs).forEach(([k,v])=>{
    if(k==='text')e.textContent=v;else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else if(v!==undefined&&v!==false)e.setAttribute(k,v===true?'':v);
  });children.filter(Boolean).forEach(c=>e.append(c));return e;};
  const button=(text,click,attrs={})=>node('button',{type:'button',text,onclick:click,...attrs});
  const input=(value,label,type='text')=>node('input',{type,value,'aria-label':label,...(type==='number'?{step:'any'}:{})});
  const select=(items,value,label)=>{const e=node('select',{'aria-label':label});items.forEach(i=>e.append(node('option',{value:Array.isArray(i)?i[0]:i,text:Array.isArray(i)?i[1]:i})));e.value=value;return e;};
  const field=(label,e)=>node('div',{class:'flow-row'},[node('label',{text:label}),e]);
  const details=(title,...children)=>node('details',{},[node('summary',{text:title}),...children]);
  const fmt=n=>Number(n||0).toLocaleString();
  let plotlyPromise;
  function loadPlotly(url){if(window.Plotly)return Promise.resolve();if(!plotlyPromise)plotlyPromise=new Promise((resolve,reject)=>{
    const failed=()=>{clearTimeout(timer);s.remove();plotlyPromise=null;reject(Error('Plot tools could not load. Retry to reconnect.'));};
    const s=node('script',{src:url,onload:()=>{if(!window.Plotly){failed();return;}clearTimeout(timer);resolve();},onerror:failed});
    const timer=setTimeout(failed,30000);document.head.append(s);
  });return plotlyPromise;}
  function download(value,name,type='application/json'){
    const url=typeof value==='string'&&value.startsWith('data:')?value:URL.createObjectURL(new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type}));
    const a=node('a',{href:url,download:name});a.click();if(url.startsWith('blob:'))setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function mount(root,hooks){
    const {channel,get,post}=hooks, route='/studio/'+encodeURIComponent(channel)+'/flow_gating';
    root.classList.add('flow-app');
    const bus=!hooks.message&&typeof BroadcastChannel!=='undefined'?new BroadcastChannel('spacemap-flow-'+channel):null;
    const sender=uid();let ws,info,activePlot,layoutId,liveContext=null,sourceContext=null,frozen=false,pending=0,disposed=false,saveQueue=Promise.resolve(),sequence=0,pickSequence=0,spatialPick=null;
    const plots=new Map(),collapsed=new Set(),marked=new Set();let studioAppearance=null,selectionAnchor=null,treeDrag=null,statsView=null;
    const status=node('span',{class:'flow-status',role:'status'});
    const sidebar=node('aside',{class:'flow-sidebar'}),work=node('section',{class:'flow-workspace'}),plotArea=node('div',{class:'flow-plots'});
    const divider=node('div',{class:'flow-divider',role:'separator',tabindex:'0','aria-label':'Resize population panel','aria-orientation':'vertical',title:'Drag to resize · double-click to reset'});
    const body=node('div',{class:'flow-body'},[sidebar,divider,work]);let sidebarWidth=290,sidebarDrag=null;
    try{const saved=Number(localStorage.getItem('spacemap-flow-sidebar-width'));if(Number.isFinite(saved)&&saved>=160)sidebarWidth=saved;}catch(e){}
    const sidebarLimit=()=>Math.max(160,Math.floor(body.getBoundingClientRect().width)-388);
    function resizeSidebar(){const width=Math.max(160,Math.min(sidebarLimit(),sidebarWidth));sidebar.style.width=width+'px';sidebar.style.setProperty('--flow-sidebar-width',width+'px');divider.setAttribute('aria-valuemin','160');divider.setAttribute('aria-valuemax',String(sidebarLimit()));divider.setAttribute('aria-valuenow',String(Math.round(width)));}
    function saveSidebar(){sidebarWidth=sidebar.getBoundingClientRect().width;try{localStorage.setItem('spacemap-flow-sidebar-width',String(sidebarWidth));}catch(e){}}
    divider.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();sidebarDrag={pointer:e.pointerId,x:e.clientX,width:sidebar.getBoundingClientRect().width};try{divider.setPointerCapture(e.pointerId);}catch(error){}root.classList.add('flow-resizing-sidebar');divider.focus();};
    divider.onpointermove=e=>{if(!sidebarDrag||e.pointerId!==sidebarDrag.pointer)return;e.preventDefault();sidebarWidth=Math.max(160,Math.min(sidebarLimit(),sidebarDrag.width+e.clientX-sidebarDrag.x));resizeSidebar();};
    const finishSidebar=e=>{if(!sidebarDrag||e.pointerId!==sidebarDrag.pointer)return;if(e.type==='pointercancel'){sidebarWidth=sidebarDrag.width;resizeSidebar();}else saveSidebar();sidebarDrag=null;root.classList.remove('flow-resizing-sidebar');};
    divider.onpointerup=finishSidebar;divider.onpointercancel=finishSidebar;
    divider.ondblclick=()=>{sidebarWidth=290;resizeSidebar();saveSidebar();};
    divider.onkeydown=e=>{if(e.key==='Escape'&&sidebarDrag){e.preventDefault();e.stopPropagation();sidebarWidth=sidebarDrag.width;sidebarDrag=null;root.classList.remove('flow-resizing-sidebar');resizeSidebar();return;}
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();sidebarWidth=e.key==='Home'?160:e.key==='End'?sidebarLimit():sidebar.getBoundingClientRect().width+(e.key==='ArrowRight'?1:-1)*(e.shiftKey?40:10);resizeSidebar();saveSidebar();};
    const sidebarObserver=new ResizeObserver(resizeSidebar);sidebarObserver.observe(body);
    const live=node('input',{type:'checkbox','aria-label':'Live view'});
    const source=select([['cells','Selected cells'],['region','Selected regions'],['image','Current image'],['filters','Filtered Data rows'],['viewport','Current viewport']],'cells','Live source');
    const mode=select([['filter','Filter image'],['highlight','Highlight with context']],'filter','Spatial mode');
    const title=node('strong',{text:'Flow gating'}),trail=node('span',{class:'flow-small'});
    const tool=node('div',{class:'flow-toolbar'},[
      button('☰',()=>{sidebar.hidden=!sidebar.hidden;divider.hidden=sidebar.hidden;}, {'aria-label':'Show populations'}),title,
      button('Plots',()=>showPlots()),button('Layout',()=>showLayout()),button('Stats',()=>showStats()),button('New plot',()=>addPlot(newPlotSpec())),
      button('Undo',()=>command('undo')),button('Redo',()=>command('redo')),
      node('label',{},[live,document.createTextNode(' Live view')]),source,
      button('Freeze view',()=>{frozen=!frozen;tool.querySelector('[data-freeze]').textContent=frozen?'Resume live':'Freeze view';if(!frozen)useContext(sourceContext);setStatus(frozen?'Frozen':'Live');},{'data-freeze':''}),
      button('Reset source',()=>{liveContext=null;sourceContext=null;refreshPlots();}),mode,
      help('Gates use all eligible cells. Scatter can display a sample. Live view narrows plots; it does not rewrite gates or labels.'),status]);
    root.replaceChildren(tool,node('div',{class:'flow-toolbar'},[trail]),body);work.append(plotArea);resizeSidebar();
    const loading=node('p',{class:'flow-empty',role:'status',text:'Opening Flow gating…'});plotArea.append(loading);
    function setStatus(text,error=false){status.textContent=text;status.classList.toggle('flow-error',error);}
    function fail(error){setStatus(error.message||String(error),true);}
    async function call(action,body={}){pending++;setStatus(action==='plot'?'Updating…':action==='save'?'Saving…':'Working…');
      try{return await post(route,{action,revision:ws?.revision,workspace:ws?.id,...body});}finally{pending--;}}
    function notify(type,extra={}){const message={type,sender,...extra};bus?.postMessage(message);hooks.message?.(message);}
    function help(text){return button('?',()=>dialog('Info',node('p',{class:'flow-help-box',text}),[]),{class:'flow-info','aria-label':text,title:text});}
    function dialog(title,content,actions){const d=node('dialog',{},[node('h3',{text:title}),content,node('div',{class:'flow-row flow-dialog-actions'},[
      button('Cancel',()=>d.close()),...actions.map(([name,run,primary])=>button(name,async event=>{const buttons=[...d.querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);event.target.textContent=name==='Delete'?'Deleting…':'Saving…';
        try{await run(d);}catch(e){fail(e);error.textContent=e.message;}finally{buttons.forEach(b=>b.disabled=false);event.target.textContent=name;}},{class:primary?'primary':''}))])]);
      const error=node('p',{class:'flow-error',role:'status'});d.append(error);root.append(d);d.addEventListener('close',()=>d.remove());d.showModal();return d;}
    function menu(event,items){event.preventDefault();root.querySelector('.flow-menu')?.remove();const m=node('div',{class:'flow-menu',role:'menu'},items.map(([name,run])=>button(name,click=>{m.remove();Promise.resolve().then(()=>run(click)).catch(fail);},{role:'menuitem'})));
      root.append(m);m.style.left=Math.min(event.clientX,innerWidth-210)+'px';m.style.top=Math.min(event.clientY,innerHeight-m.offsetHeight-10)+'px';
      setTimeout(()=>document.addEventListener('pointerdown',event=>{if(!m.contains(event.target))m.remove();},{once:true}),0);}
    function updateTrail(){trail.textContent=ws.project+' · '+ws.sources.reduce((n,s)=>n+s.selected_images.length,0)+' images'+(info.n==null?'':' · '+fmt(info.n)+' cells');}
    async function reload(){const request=++sequence;const found=await get(route+'?metadata=1');if(disposed||request!==sequence)return;
      if(ws&&ws.id!==found.workspace.id){statsView?.dispose();statsView=null;plots.forEach(p=>{p.observer.disconnect();Plotly.purge(p.chart);p.card.remove();});plots.clear();marked.clear();treeDrag=null;selectionAnchor=null;liveContext=null;sourceContext=null;activePlot=null;spatialPick=null;pickSequence++;notify('clear-selection');}
      info=found;ws=found.workspace;
      loading.remove();notify('columns',{columns:found.columns});
      live.checked=!!ws.live.enabled;source.value=ws.live.source||'cells';updateTrail();
      renderTree(found.stats);if(!plots.size){if(ws.plots.length)ws.plots.forEach(p=>addPlot(p,false));else addPlot(null,false);}else refreshPlots();
      setStatus('Saved · revision '+ws.revision);}
    function save(changes){saveQueue=saveQueue.catch(()=>{}).then(async()=>{const edits=typeof changes==='function'?changes():changes,found=await call('save',{changes:edits});ws=found.workspace;info.stats=found.stats??info.stats;renderTree(info.stats);setStatus('Saved');notify('changed',{revision:ws.revision});if(edits.gates||edits.root_colour)refreshSpatialSelection();return found;});return saveQueue;}
    async function command(action){try{await saveQueue;const found=await call(action);ws=found.workspace;info.stats=found.stats;renderTree(found.stats);refreshPlots();refreshSpatialSelection();setStatus('Saved');notify('changed',{revision:ws.revision});}catch(e){fail(e);}}
    function populations(){return [{id:'root',name:'Analysis cells',colour:ws.root_colour||'#a8b4c4'},...ws.gates];}
    function population(id){return populations().find(g=>g.id===id)||populations()[0];}
    function populationPath(id){const parts=[],seen=new Set();while(id&&id!=='root'&&!seen.has(id)){seen.add(id);const g=population(id);if(g.id==='root')break;parts.unshift((g.split&&!g.name.startsWith(g.split.column+'_')?g.split.column+': ':'')+g.name);id=g.parent||'root';}return parts.join(' › ')||'Analysis cells';}
    function populationChoices(){return populations().map(g=>[g.id,populationPath(g.id)]);}
    const frequencies=[['count','Count'],['proportion','Proportion'],['percent','Percent'],['density','Probability density'],['peak','Peak normalized']];
    const histogramKinds=['histogram','ridgeline','facets','waterfall'];
    const filledHistogram=spec=>histogramKinds.includes(spec.kind)&&spec.kind!=='waterfall';
    const fillOpacity=style=>Math.max(0,Math.min(1,style.fillOpacity??(.5*(style.opacity??.65))));
    function fillColour(colour,opacity){const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');ctx.fillStyle=colour;ctx.fillRect(0,0,1,1);const [r,g,b]=ctx.getImageData(0,0,1,1).data;return `rgba(${r},${g},${b},${opacity})`;}
    function opacityControl(style,label){const slider=input(Math.round(fillOpacity(style)*100),label,'range');slider.min=0;slider.max=100;slider.step=1;
      const value=node('output',{text:slider.value+'%'});slider.oninput=()=>value.textContent=slider.value+'%';return {slider,box:node('span',{class:'flow-opacity'},[slider,value])};}
    const seriesIdentity=p=>JSON.stringify((p.spec.series?.length?p.spec.series:[{population:p.spec.parent}]).map(s=>[s.population,s.rule||null,!!s.pinned]));
    function seriesOrder(p){const n=p.result.histogram.series.length,saved=p.spec.style.seriesOrderSource===seriesIdentity(p)?p.spec.style.seriesOrder:[],order=[...new Set(saved||[])].filter(i=>Number.isInteger(i)&&i>=0&&i<n);
      return order.concat(Array.from({length:n},(_,i)=>i).filter(i=>!order.includes(i)));}
    function saveAppearance(p){return p.saveAppearance?p.saveAppearance():persistPlots();}
    function redrawAppearance(p){return p.redrawAppearance?p.redrawAppearance():renderPlot(p);}
    function queueAppearance(p){p.appearancePending=true;if(p.appearanceTask)return p.appearanceTask;
      p.appearanceTask=(async()=>{do{p.appearancePending=false;await redrawAppearance(p);}while(p.appearancePending&&!disposed);})().catch(fail).finally(()=>p.appearanceTask=null);return p.appearanceTask;}
    function histogramControls(p){const opacity=opacityControl(p.spec.style,'Histogram fill opacity'),match=node('input',{type:'checkbox','aria-label':'Match Y scales'});
      const facets=node('label',{},[match,document.createTextNode(' Match Y scales')]),box=node('span',{class:'flow-histogram-controls'},[node('span',{text:'Fill'}),opacity.box,facets]);
      opacity.slider.addEventListener('input',()=>{p.spec.style.fillOpacity=Number(opacity.slider.value)/100;queueAppearance(p);});
      opacity.slider.onchange=()=>saveAppearance(p).catch(fail);match.onchange=()=>{p.spec.style.matchY=match.checked;queueAppearance(p);saveAppearance(p).catch(fail);};
      p.syncHistogramControls=()=>{box.hidden=!filledHistogram(p.spec);facets.hidden=p.spec.kind!=='facets';match.checked=!!p.spec.style.matchY;opacity.slider.value=Math.round(fillOpacity(p.spec.style)*100);opacity.box.querySelector('output').textContent=opacity.slider.value+'%';};p.syncHistogramControls();return box;}
    function bindLegendOrder(p){const enabled=histogramKinds.includes(p.spec.kind)&&p.result.histogram.series.length>1;
      const rows=[...p.chart.querySelectorAll('.legend .traces')];
      rows.forEach(row=>{const index=row.__data__?.[0]?.trace?.meta?.flowSeries;row.classList.toggle('flow-legend-item',enabled);const series=p.result.histogram.series[index],threshold=p.result.histogram.threshold;let hint=row.querySelector('title');if(!hint){hint=document.createElementNS('http://www.w3.org/2000/svg','title');row.append(hint);}hint.textContent=series?'Total cells in selected gate: '+fmt(series.n)+(threshold?' | Above threshold ('+threshold.name+' @ '+Number(threshold.bounds[0]).toPrecision(5)+'): '+fmt(series.above_threshold)+' cells ('+Number(series.above_percent).toFixed(1)+'%)':'')+(enabled?' | Drag to reorder · top = front. Alt + arrows also reorder.':''):'';if(!enabled||!Number.isInteger(index)){row.onkeydown=null;row.removeAttribute('tabindex');return;}
        row.dataset.series=String(index);row.setAttribute('tabindex','0');row.setAttribute('role','button');row.setAttribute('aria-label','Reorder '+p.result.histogram.series[index].name);
        let title=row.querySelector('title');if(!title){title=document.createElementNS('http://www.w3.org/2000/svg','title');row.append(title);}title.textContent=hint.textContent;
        row.onkeydown=e=>{if(!e.altKey||!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();e.stopPropagation();const order=seriesOrder(p),from=order.indexOf(index),to=Math.max(0,Math.min(order.length-1,from+(['ArrowUp','ArrowLeft'].includes(e.key)?-1:1)));order.splice(from,1);order.splice(to,0,index);applyLegendOrder(p,order).catch(fail);};
      });
      if(p.legendHooked)return;p.legendHooked=true;
      p.chart.on('plotly_afterplot',()=>bindLegendOrder(p));
      // Capture only a real drag. Ordinary legend clicks retain Plotly's hide/show behaviour.
      for(const type of ['mouseup','click','dblclick'])p.chart.addEventListener(type,e=>{if(p.legendDrag?.moved||Date.now()<(p.legendSuppressUntil||0)){e.preventDefault();e.stopImmediatePropagation();}},true);
      p.chart.addEventListener('pointerdown',event=>{const row=event.target.closest('.flow-legend-item');if(event.button!==0||!row||!histogramKinds.includes(p.spec.kind))return;
        const index=Number(row.dataset.series),order=seriesOrder(p),start={x:event.clientX,y:event.clientY},horizontal=p.spec.style.legendSide==='bottom';
        const drag={moved:false,order};p.legendDrag=drag;let ghost,line;
        const move=e=>{if(e.pointerId!==event.pointerId)return;if(!drag.moved&&Math.hypot(e.clientX-start.x,e.clientY-start.y)<5)return;
          e.preventDefault();e.stopPropagation();if(!drag.moved){drag.moved=true;ghost=node('div',{class:'flow-legend-ghost',text:p.result.histogram.series[index].name});line=node('div',{class:'flow-legend-drop-line'});root.append(ghost,line);row.classList.add('flow-legend-moving');}
          ghost.style.left=(e.clientX+12)+'px';ghost.style.top=(e.clientY+12)+'px';
          const entries=[...p.chart.querySelectorAll('.flow-legend-item')].map(el=>({index:Number(el.dataset.series),rect:el.getBoundingClientRect()}));
          const coordinate=horizontal?e.clientX:e.clientY;let nearest=0,distance=Infinity;entries.forEach(({rect},i)=>{const dx=Math.max(rect.left-e.clientX,0,e.clientX-rect.right),dy=Math.max(rect.top-e.clientY,0,e.clientY-rect.bottom),d=dx*dx+dy*dy;if(d<distance){distance=d;nearest=i;}});
          const target=entries[nearest]?.rect;let at=nearest+(target&&coordinate>(horizontal?target.left+target.width/2:target.top+target.height/2)?1:0);
          const ref=entries[Math.min(at,entries.length-1)]?.rect;if(ref){line.style.cssText=horizontal?`left:${at===entries.length?ref.right:ref.left}px;top:${ref.top}px;width:3px;height:${ref.height}px`:`left:${ref.left}px;top:${at===entries.length?ref.bottom:ref.top}px;width:${ref.width}px;height:3px`;}
          const next=order.filter(i=>i!==index),from=order.indexOf(index);next.splice(at>from?at-1:at,0,index);drag.order=next;};
        const finish=e=>{if(e.type!=='keydown'&&e.pointerId!==event.pointerId)return;if(e.type==='keydown'&&e.key!=='Escape')return;
          const cancelled=e.type==='pointercancel'||e.type==='keydown';if(drag.moved){e.preventDefault();e.stopImmediatePropagation();p.legendSuppressUntil=Date.now()+350;}
          document.removeEventListener('pointermove',move,true);document.removeEventListener('pointerup',finish,true);document.removeEventListener('pointercancel',finish,true);document.removeEventListener('keydown',finish,true);
          ghost?.remove();line?.remove();row.classList.remove('flow-legend-moving');p.legendDrag=null;
          if(drag.moved&&!cancelled)applyLegendOrder(p,drag.order).catch(fail);};
        document.addEventListener('pointermove',move,true);document.addEventListener('pointerup',finish,true);document.addEventListener('pointercancel',finish,true);document.addEventListener('keydown',finish,true);
      },true);
    }
    async function applyLegendOrder(p,order){if(JSON.stringify(order)===JSON.stringify(seriesOrder(p)))return;p.spec.style.seriesOrder=order;p.spec.style.seriesOrderSource=seriesIdentity(p);await queueAppearance(p);await saveAppearance(p);}
    function changeAxisMeasurement(p,axis,column){const spec=clone(p.spec[axis]||{scale:'linear'});spec.column=column;delete spec.fitted;delete spec.pipeline_fitted;delete spec.reference_n;
      for(const step of spec.pipeline||[]){delete step.fitted;if(step.op==='row_zscore'){
        const markers=step.marker_source==='axes'?[column,p.spec[axis==='x'?'y':'x']?.column]:(step.markers||spec.markers||[]);
        step.markers=[...new Set([...markers.filter(Boolean),column])];
      }}
      if(spec.pipeline)delete spec.markers;p.spec[axis]=spec;
    }
    function axisMenu(p,axis,event){
      root.querySelector('.flow-menu')?.remove();activePlot=p;
      const search=input('','Find axis measurement'),list=node('div',{class:'flow-axis-columns'}),m=node('div',{class:'flow-menu flow-axis-menu',role:'menu','aria-label':axis.toUpperCase()+' axis choices'},[node('strong',{text:axis.toUpperCase()+' axis'}),search,list]);
      search.placeholder='Find measurement…';root.append(m);m.style.left=Math.max(8,Math.min(event.clientX||24,innerWidth-300))+'px';m.style.top=Math.max(8,Math.min(event.clientY||100,innerHeight-380))+'px';
      const choose=async(value,frequency=false)=>{m.remove();const hist=['histogram','ridgeline','facets','waterfall'];
        if(frequency){if(!hist.includes(p.spec.kind))p.scatterKind=p.spec.kind;p.spec.kind='histogram';p.spec.frequency=value;if(value==='count')p.spec.patient='';delete p.spec.y_range;p.spec.style.yTitle='';}
        else {changeAxisMeasurement(p,axis,value);delete p.spec[axis+'_range'];p.spec.style[axis+'Title']='';p.controls[axis].setMeasurement(value);
          if(axis==='y'&&hist.includes(p.spec.kind)){p.spec.kind=p.scatterKind||'scatter';delete p.spec.y_range;}}
        p.spec.tool='pan';p.controls.gateTool.value='pan';p.controls.kind.value=p.spec.kind;await refreshPlot(p);await persistPlots();};
      const draw=()=>{const query=search.value.toLowerCase();list.replaceChildren();
        if(axis==='y'){list.append(node('div',{class:'flow-small flow-menu-heading',text:'Histogram'}));for(const [value,label] of frequencies)if(label.toLowerCase().includes(query))list.append(button(label,()=>choose(value,true).catch(fail),{role:'menuitem'}));}
        list.append(node('div',{class:'flow-small flow-menu-heading',text:'Measurements'}));
        info.numeric.filter(c=>c.toLowerCase().includes(query)).forEach(column=>list.append(button(column,()=>choose(column).catch(fail),{role:'menuitem'})));};
      search.oninput=draw;search.onkeydown=e=>{if(e.key==='Escape'){e.stopPropagation();m.remove();}if(e.key==='ArrowDown'){e.preventDefault();list.querySelector('button')?.focus();}};draw();search.focus();
      const dismiss=e=>{if(!m.isConnected||!m.contains(e.target)){m.remove();document.removeEventListener('pointerdown',dismiss);}};setTimeout(()=>document.addEventListener('pointerdown',dismiss),0);
    }
    async function applyTransformation(p,axis,recipe){
      const column=p.spec[axis]?.column||info.numeric[0];let next={column,scale:'linear'};
      if(recipe){const pipeline=clone(recipe.pipeline);pipeline.forEach(step=>{delete step.fitted;if(step.op==='row_zscore'){
          const custom=step.marker_source==='custom'||(!step.marker_source&&(step.markers||recipe.markers)?.length);
          const chosen=custom?(step.markers||recipe.markers||[]):[p.spec.x.column,p.spec.y?.column];
          step.marker_source=custom?'custom':'axes';
          step.markers=[...new Set([...chosen.filter(Boolean),column])];
          if(step.enabled!==false&&step.markers.length<2)throw Error('Z2 needs two measurements. Check its step settings.');
        }});
        next={...next,recipe_id:recipe.id,recipe_version:recipe.version||1,recipe_name:recipe.name,pipeline,reference_population:'root'};}
      p.spec[axis]=next;delete p.spec[axis+'_range'];p.spec.style[axis+'Title']='';p.spec.tool='pan';p.controls.gateTool.value='pan';
      await refreshPlot(p);await persistPlots();
    }
    function transformationMenu(p,axis,event){activePlot=p;
      if(axis==='y'&&histogramKinds.includes(p.spec.kind)){
        menu(event,[...frequencies.map(([value,label])=>[label,async()=>{p.spec.frequency=value;if(value==='count')p.spec.patient='';p.spec.style.yTitle='';delete p.spec.y_range;await refreshPlot(p);await persistPlots();}]),[p.spec.style.yLog?'Linear Y':'Log Y',async()=>{p.spec.style.yLog=!p.spec.style.yLog;await renderPlot(p);await persistPlots();}]]);
        const m=root.querySelector('.flow-menu');m.prepend(node('strong',{class:'flow-menu-heading',text:'Y frequency'}));m.style.top=Math.max(8,Math.min(event.clientY,innerHeight-m.offsetHeight-10))+'px';return;
      }
      const recipes=ws.recipes||[],entries=[['Stored values',()=>applyTransformation(p,axis,null)],...recipes.map(recipe=>[recipe.name,()=>applyTransformation(p,axis,recipe)]),['New transformation…',()=>recipeDialog(p,null)],['Edit transformations…',()=>recipeDialog(p)]];
      menu(event,entries);const m=root.querySelector('.flow-menu');m.classList.add('flow-transform-menu');m.setAttribute('aria-label',axis.toUpperCase()+' transformations');
      [...m.querySelectorAll('button')].slice(0,recipes.length+1).forEach((b,i)=>{const selected=i?((p.spec[axis]?.recipe_id===recipes[i-1].id||(!p.spec[axis]?.recipe_id&&p.spec[axis]?.recipe_name===recipes[i-1].name))&&(p.spec[axis]?.recipe_version||1)===(recipes[i-1].version||1)):!p.spec[axis]?.pipeline&&(!p.spec[axis]?.scale||p.spec[axis].scale==='linear');
        b.setAttribute('role','menuitemradio');b.setAttribute('aria-checked',String(selected));});m.prepend(node('strong',{class:'flow-menu-heading',text:axis.toUpperCase()+' transformation'}));m.style.top=Math.max(8,Math.min(event.clientY,innerHeight-m.offsetHeight-10))+'px';
    }
    function bindAxisTitles(p){for(const axis of ['x','y'])p.chart.querySelectorAll('.'+axis+'title').forEach(title=>{
      title.classList.add('flow-axis-title');title.setAttribute('role','button');title.setAttribute('tabindex','0');title.setAttribute('aria-label','Change '+axis.toUpperCase()+' axis');
      title.setAttribute('title','Click: measurement · Right-click: transformation');
      title.oncontextmenu=e=>{e.stopPropagation();transformationMenu(p,axis,e);};
      title.onclick=e=>{e.stopPropagation();axisMenu(p,axis,e);};title.onkeydown=e=>{if(e.key==='ContextMenu'||e.shiftKey&&e.key==='F10'){e.preventDefault();e.stopPropagation();const r=title.getBoundingClientRect();transformationMenu(p,axis,{clientX:r.left,clientY:r.bottom,preventDefault:()=>{}});}else if(['Enter',' '].includes(e.key)){e.preventDefault();e.stopPropagation();const r=title.getBoundingClientRect();axisMenu(p,axis,{clientX:r.left,clientY:r.bottom});}};
    });}
    function children(id){return ws.gates.filter(g=>(g.parent||'root')===id);}
    function descendants(id){const result=[];for(const g of children(id)){result.push(g.id,...descendants(g.id));}return result;}
    function markRows(){sidebar.querySelectorAll('[data-population]').forEach(row=>{const selected=marked.has(row.dataset.population);row.classList.toggle('marked',selected);row.setAttribute('aria-selected',String(selected));});}
    function chooseRow(id,event){
      if(event.ctrlKey||event.metaKey){marked.has(id)?marked.delete(id):marked.add(id);selectionAnchor=id;markRows();return;}
      if(event.shiftKey&&selectionAnchor){const ids=[...sidebar.querySelectorAll('[data-population]')].map(r=>r.dataset.population),a=ids.indexOf(selectionAnchor),b=ids.indexOf(id);
        if(a>=0&&b>=0){marked.clear();ids.slice(Math.min(a,b),Math.max(a,b)+1).forEach(i=>marked.add(i));markRows();return;}}
      marked.clear();marked.add(id);selectionAnchor=id;pick(id);
    }
    function transferIds(g){return [...(marked.has(g.id)?marked:new Set([g.id]))].filter(id=>id!=='root'&&ws.gates.some(x=>x.id===id));}
    function clearTreeDrag(){treeDrag=null;sidebar.querySelectorAll('[data-drop]').forEach(row=>delete row.dataset.drop);}
    function dragMode(event){return event.shiftKey&&!event.ctrlKey&&!event.metaKey?'move':'copy';}
    function validDrop(ids,target,mode){return ids.length&&(mode==='copy'||!ids.some(id=>id===target||descendants(id).includes(target)));}
    function transferPopulations(ids,target,mode,revision=null){
      saveQueue=saveQueue.catch(()=>{}).then(async()=>{setStatus(mode==='move'?'Moving…':'Copying…');const found=await call('transfer',{populations:ids,target,mode,...(revision===null?{}:{revision})});
        ws=found.workspace;info.stats=found.stats??info.stats;marked.clear();found.selected.forEach(id=>marked.add(id));if(target)collapsed.delete(target);
        renderTree(info.stats);refreshPlots();refreshSpatialSelection();notify('changed',{revision:ws.revision});setStatus(found.affected?(mode==='move'?'Moved':'Copied')+' · Undo available':'Already in this population');return found;
      });return saveQueue;
    }
    function transferDialog(ids,mode){const destinations=populationChoices().filter(([id])=>validDrop(ids,id,mode)),target=select(destinations,'root','Destination population');
      dialog(mode==='move'?'Move populations':'Copy populations',node('div',{},[field('Into',target),node('p',{class:'flow-small',text:ids.length+' selected · includes child populations'})]),[[mode==='move'?'Move':'Copy',async d=>{await transferPopulations(ids,target.value,mode);d.close();},true]]);}
    function treeDragHandlers(tr,g,name){
      name.draggable=g.id!=='root';name.ondragstart=e=>{const ids=transferIds(g);if(!ids.length){e.preventDefault();return;}if(!marked.has(g.id)){marked.clear();marked.add(g.id);markRows();}
        treeDrag={ids,workspace:ws.id,revision:ws.revision};e.dataTransfer.effectAllowed='copyMove';e.dataTransfer.setData('application/x-spacemap-populations',JSON.stringify(treeDrag));e.dataTransfer.setData('text/plain',ids.map(id=>populationPath(id)).join('\n'));setStatus('Drag to copy · Shift to move');};
      name.ondragend=()=>{clearTreeDrag();};
      tr.ondragover=e=>{if(!treeDrag||treeDrag.workspace!==ws.id)return;const mode=dragMode(e),valid=validDrop(treeDrag.ids,g.id,mode);sidebar.querySelectorAll('[data-drop]').forEach(row=>delete row.dataset.drop);
        if(!valid){e.dataTransfer.dropEffect='none';return;}e.preventDefault();e.dataTransfer.dropEffect=mode;tr.dataset.drop=mode;
        setStatus((mode==='move'?'Move':'Copy')+' '+treeDrag.ids.length+' → '+g.name);const r=sidebar.getBoundingClientRect();if(e.clientY<r.top+32)sidebar.scrollTop-=20;else if(e.clientY>r.bottom-32)sidebar.scrollTop+=20;};
      tr.ondragleave=e=>{if(!tr.contains(e.relatedTarget))delete tr.dataset.drop;};
      tr.ondrop=e=>{if(!treeDrag||treeDrag.workspace!==ws.id)return;e.preventDefault();e.stopPropagation();const draft=treeDrag,mode=dragMode(e);clearTreeDrag();
        if(validDrop(draft.ids,g.id,mode))transferPopulations(draft.ids,g.id,mode,draft.revision).catch(fail);};
    }
    async function pick(id){if(statsView?.active()){statsView.addGroups([id]);return;}const version=++pickSequence,workspace=ws.id;ws.active=id;if(activePlot&&activePlot.spec.parent!==id){activePlot.spec.series=[];ws.overlays=[];}renderTree(info.stats);if(activePlot){showPlots();activePlot.spec.parent=id;activePlot.controls.parent.value=id;await refreshPlot(activePlot);}
      if(version!==pickSequence||workspace!==ws.id)return;
      try{const selection=await call('select',{populations:[id]});if(version!==pickSequence||workspace!==ws.id||disposed)return;spatialPick={id,mode:mode.value};notify('selection',{selection,mode:mode.value,table:ws.table});setStatus(fmt(selection.count)+' cells');}catch(e){if(version===pickSequence)fail(e);}}
    async function showPopulation(id){const version=++pickSequence,workspace=ws.id;
      const selection=await call('select',{populations:[id]});if(version!==pickSequence||workspace!==ws.id||disposed)return;
      ws.active=id;renderTree(info.stats);spatialPick={id,mode:'filter'};notify('selection',{selection,mode:'filter',table:ws.table});setStatus('Showing '+population(id).name+' in the open image');}
    async function refreshSpatialSelection(){if(!spatialPick)return;const picked=spatialPick,version=++pickSequence,workspace=ws.id;
      if(picked.id!=='root'&&!ws.gates.some(g=>g.id===picked.id)){spatialPick=null;notify('clear-selection');return;}
      try{const selection=await call('select',{populations:[picked.id]});if(version!==pickSequence||workspace!==ws.id||disposed)return;notify('selection',{selection,mode:picked.mode,table:ws.table});}
      catch(error){if(version===pickSequence)fail(error);}}
    function renderTree(stats=[]){
      info.stats=stats;
      for(const id of marked)if(id!=='root'&&!ws.gates.some(g=>g.id===id))marked.delete(id);
      const search=input('','Find population');search.className='flow-search';search.placeholder='Find population…';
      const table=node('table',{class:'flow-tree',role:'treegrid','aria-label':'Population hierarchy','aria-multiselectable':'true'});
      const rows=()=>{const counts=new Map((info.stats||[]).map(s=>[s.id,s]));table.replaceChildren();function row(g,depth){const kids=children(g.id),s=counts.get(g.id)||{};const match=!search.value||g.name.toLowerCase().includes(search.value.toLowerCase());
        if(match){const overlay=node('input',{type:'checkbox','aria-label':'Overlay '+g.name});overlay.checked=(ws.overlays||[]).includes(g.id);overlay.onchange=()=>{
          const chosen=new Set(ws.overlays);overlay.checked?chosen.add(g.id):chosen.delete(g.id);ws.overlays=[...chosen];
          if(activePlot){activePlot.spec.series=ws.overlays.map(id=>({population:id}));refreshPlot(activePlot);}save({overlays:ws.overlays}).catch(fail);};
          const name=button((g.kind==='filter'?'◇ ':g.kind==='boolean'?'∩ ':'')+g.name,e=>chooseRow(g.id,e),{class:'flow-name',title:populationPath(g.id)+' · Ctrl-click to select · Drag to copy · Shift-drag to move'});
          const expanded=!collapsed.has(g.id)||!!search.value;
          const toggle=kids.length?button(expanded?'−':'+',()=>{expanded?collapsed.add(g.id):collapsed.delete(g.id);rows();},{class:'flow-disclosure','aria-label':(expanded?'Collapse ':'Expand ')+g.name,'aria-expanded':String(expanded)}):node('span',{class:'flow-tree-leaf','aria-hidden':'true'});
          const swatch=input(g.colour,'Colour '+g.name,'color');swatch.className='flow-swatch';swatch.title='Change colour · '+g.name;
          swatch.onchange=()=>{const colour=swatch.value;save(()=>g.id==='root'?{root_colour:colour}:{gates:ws.gates.map(x=>x.id===g.id?{...x,colour}:x)}).then(refreshPlots).catch(fail);};
          const entry=node('div',{class:'flow-tree-entry',style:'padding-left:'+(depth*18)+'px'},[toggle,overlay,swatch,name]);
          const tr=node('tr',{'data-population':g.id,role:'row','aria-level':depth+1,'aria-selected':String(marked.has(g.id)),...(kids.length?{'aria-expanded':String(expanded)}:{}),class:(ws.active===g.id?'active ':'')+(marked.has(g.id)?'marked':''),oncontextmenu:e=>{if(!marked.has(g.id)){marked.clear();marked.add(g.id);markRows();}menu(e,populationActions(g));}},[
            node('td',{},[entry]),
            node('td',{class:'flow-number',text:s.n==null?'…':fmt(s.n)}),node('td',{class:'flow-number',text:g.id==='root'?'':s.n==null?'…':Number(s.percent||0).toFixed(1)+'%',title:fmt(s.evaluable)+' evaluable; % of all parent cells'})]);treeDragHandlers(tr,g,name);table.append(tr);}
        if(!collapsed.has(g.id)||search.value)kids.forEach(k=>row(k,depth+1));}row(population('root'),0);};search.oninput=rows;rows();
      const active=population(marked.size?[...marked][0]:ws.active);const gateActions=node('div',{class:'flow-actions'},populationActions(active).slice(0,4).map(([name,run])=>button(name,()=>Promise.resolve().then(run).catch(fail))));
      gateActions.append(button('More…',e=>menu(e,populationActions(population(marked.size?[...marked][0]:ws.active))),{'aria-label':'Population actions'}));
      sidebar.replaceChildren(search,table,gateActions,node('div',{class:'flow-actions'},[button('Filter population',()=>filterDialog()),button('Boolean',()=>booleanDialog())]),
        details('Phenotype labels',labelsPanel()),details('Workspace',
          button('Sync selected images',()=>command('scope')),button('Export gates',()=>download({version:1,gates:ws.gates,plots:ws.plots},'flow-gates.json')),
          button('Import gates',()=>importGates()),field('Output folder',folderInput()),help('Image scope comes from the checkboxes in Phenotyping. Stored values are the default. Apply labels saves Parquet changes; CSV is exported separately.')));
      if(statsView?.active())statsView.attach();
    }
    function updateStats(stats){info.stats=stats;const counts=new Map(stats.map(s=>[s.id,s]));
      sidebar.querySelectorAll('[data-population]').forEach(row=>{const s=counts.get(row.dataset.population);if(!s)return;const cells=row.querySelectorAll('.flow-number');
        cells[0].textContent=fmt(s.n);cells[1].textContent=row.dataset.population==='root'?'':Number(s.percent||0).toFixed(1)+'%';cells[1].title=fmt(s.evaluable)+' evaluable; % of all parent cells';});}
    function folderInput(){const e=input(ws.output_dir,'Figure output folder');e.onchange=()=>save({output_dir:e.value}).catch(fail);return e;}
    function populationActions(g){const actions=[...(statsView?.active()?[['Add to comparison groups',()=>statsView.addGroups(marked.size?[...marked]:[g.id])]]:[]),['Open population',()=>pick(g.id)],['Show cells on image',()=>showPopulation(g.id)],['New plot',()=>addPlot(newPlotSpec(g.id))],['Select descendants',()=>{ws.overlays=[g.id,...descendants(g.id)];save({overlays:ws.overlays}).then(()=>{if(activePlot){activePlot.spec.series=ws.overlays.map(id=>({population:id}));refreshPlot(activePlot);}});}]];
      actions.push(['Split by…',event=>splitMenu(g,event)]);
      const selected=transferIds(g);if(selected.length)actions.push(['Duplicate'+(selected.length>1?' '+selected.length:''),()=>transferPopulations(selected,null,'copy')],['Copy to…',()=>transferDialog(selected,'copy')],['Move to…',()=>transferDialog(selected,'move')]);
      if(g.id!=='root')actions.push(['Edit / rename…',()=>editGate(g)],
        [g.enabled===false?'Enable':'Disable',()=>save({gates:ws.gates.map(x=>x.id===g.id?{...x,enabled:x.enabled===false}:x)}).then(refreshPlots)],
        [selected.length>1?'Delete selected ('+selected.length+')':'Delete gate',()=>deletePopulations(selected)]);return actions;}
    async function splitMenu(g,event){
      root.querySelector('.flow-menu')?.remove();const search=input('','Split by column'),list=node('div',{class:'flow-split-columns'}),m=node('div',{class:'flow-menu flow-split-menu',role:'menu','aria-label':'Split by column'},[node('strong',{text:'Split '+g.name}),search,list]);
      search.placeholder='Find column…';root.append(m);m.style.left=Math.min(event?.clientX||24,innerWidth-300)+'px';m.style.top=Math.max(8,Math.min(event?.clientY||100,innerHeight-380))+'px';
      const draw=()=>list.replaceChildren(...info.columns.filter(c=>c.toLowerCase().includes(search.value.toLowerCase())).map(column=>button(column,()=>{m.remove();splitColumn(g,column).catch(fail);},{role:'menuitem'})));
      search.oninput=draw;search.onkeydown=e=>{if(e.key==='Escape'){e.stopPropagation();m.remove();}if(e.key==='ArrowDown'){e.preventDefault();list.querySelector('button')?.focus();}};draw();search.focus();
      const dismiss=e=>{if(!m.isConnected){document.removeEventListener('pointerdown',dismiss);return;}if(!m.contains(e.target)){m.remove();document.removeEventListener('pointerdown',dismiss);}};setTimeout(()=>document.addEventListener('pointerdown',dismiss),0);
      try{const found=await get(route+'?metadata=1');if(m.isConnected&&found.workspace.id===ws.id){info.columns=found.columns;info.numeric=found.numeric;draw();}}catch(error){fail(error);}
    }
    function splitColumn(g,column,selection=null,description=null){
      saveQueue=saveQueue.catch(()=>{}).then(async()=>{const colours=studioAppearance?.table===ws.table&&studioAppearance.column===column?studioAppearance.colours:{};
        setStatus('Splitting…');const found=await call('split',{parent:g.id,column,colours,...(selection?{...selection,dataset_revision:description.dataset_revision,revision:description.revision}:{})});
        if(!found.workspace){setStatus(found.kind==='categorical'?'Choose categories':'Choose a split condition');if(found.kind==='categorical')categoricalSplit(g,found);else numericSplit(g,found);return found;}
        ws=found.workspace;info.stats=found.stats;collapsed.delete(g.id);renderTree(found.stats);refreshPlots();refreshSpatialSelection();notify('changed',{revision:ws.revision});setStatus(found.added?found.added+' populations added · Undo available':'Split already exists');return found;
      });return saveQueue;
    }
    function categoricalSplit(g,description){
      const search=input('','Find split category'),summary=node('span',{class:'flow-small','aria-live':'polite'}),list=node('div',{class:'flow-split-categories','aria-label':'Categories to keep'}),choices=[];
      search.className='flow-search';search.placeholder='Find category…';
      const update=()=>{summary.textContent=fmt(choices.filter(c=>c.check.checked).length)+' / '+fmt(choices.length)+' selected';};
      for(const category of description.categories){const check=node('input',{type:'checkbox','aria-label':'Keep '+category.label});check.checked=true;check.onchange=update;
        const row=node('label',{title:category.value??'Empty or missing values'},[check,node('span',{text:category.label}),node('span',{class:'flow-number',text:fmt(category.count),title:'Cells in '+g.name})]);
        choices.push({check,row,category});list.append(row);}
      const empty=node('p',{class:'flow-small',text:'No matching categories.'});empty.hidden=true;
      search.oninput=()=>{const query=search.value.trim().toLowerCase();choices.forEach(c=>c.row.hidden=!c.category.label.toLowerCase().includes(query));empty.hidden=choices.some(c=>!c.row.hidden);};
      const chooseAll=checked=>{choices.forEach(c=>c.check.checked=checked);update();};
      const content=node('div',{},[node('p',{class:'flow-small',text:populationPath(g.id)}),node('p',{text:'Choose categories to keep as subgroups.'}),search,
        node('div',{class:'flow-actions'},[button('All',()=>chooseAll(true)),button('None',()=>chooseAll(false)),summary]),list,empty]);
      dialog('Split '+description.column,content,[['Split',async d=>{const categories=choices.filter(c=>c.check.checked).map(c=>c.category.value);if(!categories.length)throw Error('Choose at least one category.');await splitColumn(g,description.column,{categories},description);d.close();},true]]);update();search.focus();
    }
    function numericSplit(g,description){
      const mode=select([['threshold','Threshold'],['range','Range']],'threshold','Numeric split mode'),cutoff=input(description.median??0,'Split cutoff','number'),low=input(description.minimum??0,'Split minimum','number'),high=input(description.maximum??1,'Split maximum','number');
      [cutoff,low,high].forEach(e=>e.step='any');const threshold=field('Cutoff',cutoff),range=node('div',{},[field('Minimum',low),field('Maximum',high)]),hint=node('p',{class:'flow-small'});
      const draw=()=>{threshold.hidden=mode.value!=='threshold';range.hidden=mode.value!=='range';hint.textContent=mode.value==='threshold'?'Below / at or above the cutoff':'Inside / outside the range (endpoints included)';};mode.onchange=draw;draw();
      const content=node('div',{},[node('p',{class:'flow-small',text:populationPath(g.id)}),field('Split',mode),threshold,range,hint,help('Stored values are used. Numeric columns with fewer than 25 distinct values across Analysis cells split as categories. Missing values form a separate group.')]);
      dialog('Split '+description.column,content,[['Split',async d=>{const condition=mode.value==='threshold'?{mode:'threshold',value:cutoff.value}:{mode:'range',low:low.value,high:high.value};await splitColumn(g,description.column,{condition},description);d.close();},true]]);cutoff.focus();cutoff.select();
    }
    function deleteGate(g){if(g&&g.id!=='root')deletePopulations([g.id]);}
    function deletePopulations(selected){selected=selected.filter(id=>id!=='root'&&ws.gates.some(g=>g.id===id));if(!selected.length)return;
      const ids=new Set(selected);let changed=true;
      while(changed){changed=false;for(const item of ws.gates)if(!ids.has(item.id)&&(ids.has(item.parent)||(item.refs||[]).some(id=>ids.has(id)))){ids.add(item.id);changed=true;}}
      const remove=async()=>{await save({gates:ws.gates.filter(x=>!ids.has(x.id)),active:ids.has(ws.active)?'root':ws.active,overlays:ws.overlays.filter(id=>!ids.has(id))});
        plots.forEach(p=>{if(ids.has(p.selectedGate))p.selectedGate=null;});refreshPlots();setStatus('Gate deleted · Undo to restore');};
      dialog(selected.length>1?'Delete selected populations':'Delete gate',node('p',{text:'Delete '+(selected.length>1?selected.length+' selected populations':population(selected[0]).name)+(ids.size>selected.length?' and '+(ids.size-selected.length)+' dependent populations':'')+'? Undo can restore them.'}),[['Delete',async d=>{await remove();d.close();},true]]);
    }
    function editGate(g,isNew=false){g=clone(g);const name=input(g.name,'Population name'),colour=input(g.colour,'Population colour','color');
      const parent=select(populations().filter(p=>p.id!==g.id&&!descendants(g.id).includes(p.id)).map(p=>[p.id,populationPath(p.id)]),g.parent||'root','Parent population');
      const bounds=input((g.bounds||[]).join(', '),'Gate bounds'),vertices=node('textarea',{'aria-label':'Polygon vertices',rows:3,text:JSON.stringify(g.vertices||[])});
      const rules=g.kind==='filter'?ruleEditor(g.rule):null;
      const content=node('div',{},[field('Name',name),field('Colour',colour),field('Parent',parent),g.bounds||g.vertices?details('Coordinates',g.bounds?field('Bounds',bounds):null,g.vertices?field('Vertices',vertices):null):null,rules]);
      dialog(isNew?'Name gate':'Edit population',content,[['Save',async d=>{g.name=name.value.trim();g.colour=colour.value;g.parent=parent.value;
        if(g.bounds)g.bounds=bounds.value.split(',').map(v=>Number(v.trim()));if(g.vertices)g.vertices=JSON.parse(vertices.value);if(rules)g.rule=rules.read();
        await save({gates:isNew?[...ws.gates,g]:ws.gates.map(x=>x.id===g.id?g:x)});if(activePlot)activePlot.selectedGate=g.id;d.close();refreshPlots();},true]]);name.focus();name.select();}
    function ruleEditor(rule){const rows=node('div'),join=select([['and','All conditions'],['or','Any condition']],'and','Filter join');const controls=[];
      function add(initial={}){const column=select(info.columns,initial.column||info.columns[0],'Filter column'),op=select(['in','not in','>','>=','<','<=','==','missing','present',['blank','Empty / missing'],['nonfinite','Non-finite / missing']],initial.op||'in','Filter operation'),value=input(initial.values?.join(', ')??initial.value??'','Filter values');value.placeholder='Values, separated by commas';
        const datalist=node('datalist',{id:'values-'+uid()});value.setAttribute('list',datalist.id);
        const refresh=async()=>{try{const found=await call('values',{column:column.value});datalist.replaceChildren(...Object.keys(found.values).map(v=>node('option',{value:v})));setStatus('Ready');}catch(e){fail(e);}};column.onchange=refresh;refresh();
        const item={column,op,value};controls.push(item);const row=node('div',{class:'flow-filter-rule'},[column,op,value,datalist,button('×',()=>{controls.splice(controls.indexOf(item),1);row.remove();},{'aria-label':'Remove filter'})]);rows.append(row);}
      if(rule?.rules){join.value=rule.join;rule.rules.forEach(add);}else add(rule||{});
      const box=node('div',{},[join,rows,button('Add condition',()=>add())]);box.read=()=>({join:join.value,rules:controls.map(c=>({column:c.column.value,op:c.op.value,...(['in','not in'].includes(c.op.value)?{values:c.value.value.split(',').map(v=>v.trim())}:['missing','present','blank','nonfinite'].includes(c.op.value)?{}:{value:Number(c.value.value)})}))});return box;}
    function filterDialog(){const name=input('New population','Filtered population name'),rules=ruleEditor(),parent=select(populationChoices(),ws.active||'root','Filter parent');
      dialog('Filter population',node('div',{},[field('Name',name),field('Parent',parent),rules]),[['Create',async d=>{const gate={id:uid(),name:name.value.trim(),colour:colours[ws.gates.length%colours.length],parent:parent.value,kind:'filter',rule:rules.read()};await save({gates:[...ws.gates,gate]});refreshPlots();d.close();},true]]);}
    function booleanDialog(){const name=input('Combined population','Boolean population name'),parent=select(populationChoices(),'root','Boolean parent'),op=select(['and','or','not'],'and','Boolean operation');const checked=[];
      const list=node('div',{class:'flow-checkbox-list'},ws.gates.map(g=>{const c=node('input',{type:'checkbox'});checked.push([c,g.id]);return node('label',{},[c,document.createTextNode(g.name)]);}));
      dialog('Boolean population',node('div',{},[field('Name',name),field('Parent',parent),field('Rule',op),list]),[['Create',async d=>{await save({gates:[...ws.gates,{id:uid(),kind:'boolean',name:name.value,colour:colours[ws.gates.length%colours.length],parent:parent.value,op:op.value,refs:checked.filter(([c])=>c.checked).map(([,id])=>id)}]});refreshPlots();d.close();},true]]);}
    function labelsPanel(){const column=input(ws.output_column||'phenotype_flow','Output phenotype column');const listId='flow-columns-'+uid();column.setAttribute('list',listId);
      const choices=node('datalist',{id:listId},info.columns.map(c=>node('option',{value:c})));const picked=[];
      const list=node('div',{class:'flow-checkbox-list'},ws.gates.map(g=>{const c=node('input',{type:'checkbox','aria-label':'Assign '+g.name});c.checked=ws.overlays.includes(g.id);picked.push([c,g.id]);return node('label',{},[c,document.createTextNode(g.name)]);}));
      const preserve=node('input',{type:'checkbox','aria-label':'Preserve existing labels'});preserve.checked=true;
      const conflict=select([['skip','Leave overlaps unassigned'],['priority','Priority: list order (top first)']],'skip','Overlapping labels');
      const preview=node('div',{class:'flow-label-plan',hidden:true});let receipt=null;
      const read=()=>{const ids=picked.filter(([c])=>c.checked).map(([,id])=>id);return {column:column.value.trim(),populations:ids,preserve:preserve.checked,conflict:conflict.value,...(conflict.value==='priority'?{priority:ids}:{})};};
      const apply=button('Apply labels',async()=>{try{if(!receipt)return;const result=await call('apply_labels',{...read(),receipt:receipt.receipt});ws=result.workspace;preview.textContent=fmt(result.summary.write)+' labels saved in Parquet';receipt=null;apply.disabled=true;notify('labels',{column:column.value,revision:ws.revision});setStatus('Saved');}catch(e){fail(e);}},{class:'primary'});apply.disabled=true;
      const box=node('div',{},[field('Column',column),choices,button('Create column',async()=>{try{const found=await call('create_column',{column:column.value});ws=found.workspace;setStatus('Column ready');notify('labels',{column:column.value,revision:ws.revision});}catch(e){fail(e);}}),list,
        node('label',{},[preserve,document.createTextNode(' Preserve existing labels')]),conflict,
        help('Tick populations that produce labels. Gate names become text in this column. Applies to saved gates across the checked images; Live view does not change this scope. Priority uses the displayed order only when explicitly selected.'),
        button('Preview labels',async()=>{try{receipt=await call('preview_labels',read());preview.hidden=false;preview.textContent=fmt(receipt.write)+' to write · '+fmt(receipt.overlap)+' overlaps · '+fmt(receipt.preserved)+' kept · '+fmt(receipt.unevaluable)+' unevaluable';apply.disabled=false;setStatus('Review label counts');}catch(e){fail(e);}}),preview,apply,
        ws.applied?.stale?node('p',{class:'flow-small',text:'Gates changed · labels need updating'}):null]);
      box.addEventListener('change',()=>{receipt=null;apply.disabled=true;});return box;}
    function defaultPlot(){const numeric=info.numeric;return {id:uid(),name:'Marker plot',parent:ws.active||'root',x:{column:numeric[0],scale:'linear'},y:{column:numeric[1]||numeric[0],scale:'linear'},kind:'scatter',frequency:'percent',bins:80,sample:20000,series:[],style:{font:13,angle:0,width:750,height:590,opacity:.65,line:1.5,title:'',legend:true},tool:'pan'};}
    function newPlotSpec(parent){const spec=activePlot?clone(activePlot.spec):defaultPlot();spec.id=uid();spec.tool='pan';if(parent&&parent!==spec.parent){spec.parent=parent;spec.series=[];}return spec;}
    function searchable(value,label){
      const e=input(value,label),id='flow-columns-'+uid(),list=node('div',{id,class:'flow-measurements',role:'listbox','aria-label':label+' choices',hidden:true});
      e.setAttribute('role','combobox');e.setAttribute('aria-controls',id);e.setAttribute('aria-expanded','false');e.setAttribute('aria-autocomplete','list');e.autocomplete='off';
      e.title='Choose a measurement or type to search';let selected=value,choices=[],at=-1;
      const close=()=>{list.hidden=true;e.setAttribute('aria-expanded','false');e.removeAttribute('aria-activedescendant');};
      const accept=name=>{selected=e.value=name;close();e.dispatchEvent(new Event('change',{bubbles:true}));};
      const highlight=()=>{[...list.children].forEach((b,i)=>{b.setAttribute('aria-selected',String(i===at));if(i===at){e.setAttribute('aria-activedescendant',b.id);b.scrollIntoView({block:'nearest'});}});};
      const show=(query='')=>{choices=info.numeric.filter(n=>n.toLowerCase().includes(query.toLowerCase()));at=-1;
        list.replaceChildren(...choices.map((name,i)=>button(name,()=>accept(name),{id:id+'-'+i,role:'option','aria-selected':'false',tabindex:'-1',onpointerdown:event=>event.preventDefault()})));
        const r=e.getBoundingClientRect();list.style.cssText='left:'+r.left+'px;top:'+r.bottom+'px;min-width:'+Math.max(180,r.width)+'px;max-height:'+Math.max(80,Math.min(280,innerHeight-r.bottom-12))+'px';
        list.hidden=false;e.setAttribute('aria-expanded','true');};
      e.addEventListener('focus',()=>{e.select();show();});e.addEventListener('click',()=>{if(list.hidden){e.select();show();}});
      e.addEventListener('input',()=>show(e.value));
      e.addEventListener('change',()=>{if(info.numeric.includes(e.value))selected=e.value;else e.value=selected;});
      e.addEventListener('blur',()=>{e.value=selected;close();});
      e.addEventListener('keydown',event=>{if(event.key==='Escape'){e.value=selected;close();event.stopPropagation();}
        else if(['ArrowDown','ArrowUp'].includes(event.key)){event.preventDefault();if(list.hidden)show();at=Math.max(0,Math.min(choices.length-1,at+(event.key==='ArrowDown'?1:-1)));highlight();}
        else if(event.key==='Enter'&&!list.hidden){event.preventDefault();if(at>=0)accept(choices[at]);else if(choices.includes(e.value))accept(e.value);}});
      e.setMeasurement=value=>{selected=value;e.value=value;close();};return [e,list];
    }
    function addPlot(spec=null,persist=true){if(!info.numeric.length){setStatus('No numeric measurements found. Check Column roles.',true);return;}
      spec=spec?{...defaultPlot(),...clone(spec)}:defaultPlot();if(plots.has(spec.id))spec.id=uid();spec.style={...defaultPlot().style,...spec.style};
      const available=new Set(populations().map(g=>g.id));if(!available.has(spec.parent))spec.parent='root';spec.series=(spec.series||[]).filter(s=>available.has(s.population));
      const card=node('article',{class:'flow-plot-card','data-plot':spec.id}),chart=node('div',{class:'flow-chart'}),foot=node('div',{class:'flow-plot-foot'});
      card.style.width=(spec.style.width||750)+'px';card.style.height=(spec.style.height||590)+'px';
      const parent=select(populationChoices(),spec.parent,'Plot parent'),[x,xlist]=searchable(spec.x.column,'X measurement'),[y,ylist]=searchable(spec.y?.column||info.numeric[0],'Y measurement');
      const kind=select([['scatter','Scatter'],['density_scatter','Density scatter'],['density','Density raster'],['contour','Contours'],['histogram','Histogram overlay'],['ridgeline','Stacked histograms'],['facets','Histogram facets'],['waterfall','3D waterfall']],spec.kind,'Plot type');
      const gateTool=select([['pan','Pan / inspect'],['rectangle','Rectangle gate'],['polygon','Polygon gate'],['range','Interval gate'],['threshold','Threshold'],['quadrant','Quadrants']],spec.tool||'pan','Gate tool');
      const p={spec,card,chart,foot,controls:{parent,x,y,kind,gateTool},version:0,result:null,rendering:false};plots.set(spec.id,p);activePlot=p;
      const header=node('div',{class:'flow-plot-controls'},[parent,x,xlist,y,ylist,kind,gateTool,button('Threshold ▾',e=>thresholdMenu(p,e)),button('Filters',()=>plotFilters(p)),button('Transform',()=>recipeDialog(p)),button('Settings',()=>plotSettings(p)),button('Compare',()=>seriesDialog(p)),button('Save plot',()=>persistPlots()),button('Expand',()=>{card.classList.toggle('flow-large');Plotly.Plots.resize(chart);}),button('×',()=>removePlot(p),{'aria-label':'Close plot'}),histogramControls(p)]);
      card.append(header,chart,foot);cornerResize(card,()=>{if(window.Plotly&&chart._fullLayout)Plotly.Plots.resize(chart);},()=>{p.spec.style.width=Math.round(card.offsetWidth);p.spec.style.height=Math.round(card.offsetHeight);persistPlots();});
      const observer=new ResizeObserver(()=>{if(chart._fullLayout)Plotly.Plots.resize(chart);});observer.observe(card);p.observer=observer;
      card.addEventListener('pointerdown',()=>{activePlot=p;plots.forEach(q=>q.card.classList.toggle('selected',q===p));});
      card.addEventListener('contextmenu',e=>{if(e.target.closest('input,select,textarea'))return;menu(e,[['Plot settings…',()=>plotSettings(p)],['Transformation recipe…',()=>recipeDialog(p)],['Compare populations…',()=>seriesDialog(p)],['Duplicate plot',()=>addPlot({...clone(p.spec),id:uid()})],['Edit selected gate…',()=>ws.active!=='root'&&editGate(population(ws.active))],['Freeze to layout',()=>freezePlot(p)],['Export figure…',()=>exportDialog(p)],['Close plot',()=>removePlot(p)]]);});
      parent.addEventListener('change',()=>{activePlot=p;pick(parent.value).catch(fail);});
      [x,y,kind].forEach(e=>e.addEventListener('change',()=>{p.spec.parent=parent.value;
        if(p.spec.x.column!==x.value)changeAxisMeasurement(p,'x',x.value);
        if(p.spec.y?.column!==y.value)changeAxisMeasurement(p,'y',y.value);
        p.spec.kind=kind.value;refreshPlot(p);}));
      gateTool.onchange=()=>{p.spec.tool=gateTool.value;if(p.result)renderPlot(p);};plotArea.append(card);showPlots();refreshPlot(p);if(persist)persistPlots();return p;}
    function removePlot(p){p.observer.disconnect();Plotly.purge(p.chart);p.card.remove();plots.delete(p.spec.id);activePlot=[...plots.values()].at(-1);persistPlots();}
    function persistPlots(){return save({plots:[...plots.values()].map(p=>clone(p.spec))}).catch(fail);}
    function cornerResize(card,move,done){for(const corner of ['nw','ne','sw','se']){const handle=node('span',{class:'flow-resize-handle '+corner,'aria-label':'Resize '+corner});card.append(handle);
      handle.onpointerdown=e=>{e.preventDefault();e.stopPropagation();const start={x:e.clientX,y:e.clientY,w:card.offsetWidth,h:card.offsetHeight};handle.setPointerCapture(e.pointerId);
        handle.onpointermove=ev=>{card.style.width=Math.max(360,start.w+(ev.clientX-start.x)*(corner.endsWith('e')?1:-1))+'px';card.style.height=Math.max(300,start.h+(ev.clientY-start.y)*(corner.startsWith('s')?1:-1))+'px';move();};
        handle.onpointerup=()=>{handle.onpointermove=null;done();};};}}
    function showStats(){if(!ws)return;try{if(!statsView)statsView=window.SPACEMAPFlowStats.create({node,button,input,select,field,details,uid,work,sidebar,
      workspace:()=>ws,info:()=>info,populations,plots:()=>[...plots.values()],path:populationPath,call:async(...args)=>{try{const value=await call(...args);setStatus('Ready');return value;}catch(e){fail(e);throw e;}},save,download,live:()=>live.checked,loadPlotly:()=>loadPlotly(hooks.plotly),
      selectCells:async(group,unitRule,datasetRevision,feature={})=>{const rules=[group.rule,unitRule,feature.target_rule].filter(Boolean),selection=await call('select',{populations:[group.population,...(feature.target_population?[feature.target_population]:[])],join:'and',expected_dataset_revision:datasetRevision,...(rules.length?{rule:{join:'and',rules}}:{})});notify('selection',{selection,mode:'filter',table:ws.table});setStatus('Showing '+fmt(selection.count)+' cells in the open image');},
      overlay:result=>{let count=0;for(const p of plots.values()){p.statsOverlay=result?{result,signature:JSON.stringify([ws.gates,ws.sources,ws.thresholds])}:null;if(result&&histogramKinds.includes(p.spec.kind))count++;if(p.result)renderPlot(p);}setStatus(result?'Stats overlay added to matching histograms.':'Stats overlays cleared.');}});statsView.show();}catch(e){fail(e);}}
    function showPlots(){statsView?.hide();work.replaceChildren(plotArea);plots.forEach(p=>{if(p.chart._fullLayout)Plotly.Plots.resize(p.chart);});}
    function refreshPlots(){plots.forEach(p=>{const ids=new Set(populations().map(g=>g.id));if(!ids.has(p.spec.parent))p.spec.parent='root';p.spec.series=(p.spec.series||[]).filter(s=>ids.has(s.population));
      p.controls.parent.replaceChildren(...populations().map(g=>node('option',{value:g.id,text:populationPath(g.id)})));p.controls.parent.value=p.spec.parent;refreshPlot(p);});}
    async function refreshPlot(p){const version=++p.version;
      if(live.checked&&!liveContext){p.foot.textContent='Live · waiting for the selected source';return;}
      if(p.busy){p.queued=true;return;}p.busy=true;p.foot.textContent='Updating…';try{
      await Promise.resolve();await saveQueue;if(version!==p.version||disposed)return;
      const spec=clone(p.spec);if(['histogram','ridgeline','waterfall','facets'].includes(spec.kind))delete spec.y;
      const result=await call('plot',{plot:spec,context:live.checked?liveContext:null});if(version!==p.version||disposed)return;
      p.retries=0;p.result=result;p.spec.x=result.plot.x;if(result.plot.y)p.spec.y=result.plot.y;info.n=result.stats.find(s=>s.id==='root')?.n;updateTrail();updateStats(result.stats);await renderPlot(p);
      const shown=result.points.reduce((n,s)=>n+s.sampled,0),measured=result.points.reduce((n,s)=>n+s.measured,0),invalid=result.histogram.series.reduce((n,s)=>n+s.invalid,0),outside=result.histogram.series.reduce((n,s)=>n+s.outside,0);
      p.foot.textContent=(result.points.length?fmt(shown)+' / '+fmt(measured)+' plotted · ':'')+(invalid?fmt(invalid)+' invalid · ':'')+(outside?fmt(outside)+' outside X limits · ':'')+(result.live_n!==null?fmt(result.live_n)+' in live scope · ':'')+'Exact counts · '+result.elapsed_ms+' ms';
      setStatus(frozen?'Frozen':live.checked?(liveContext?'Live':'Live · awaiting source'):'Ready');
    }catch(e){if(version===p.version){if(e.message.includes('workspace changed')&&(p.retries||0)<2){p.retries=(p.retries||0)+1;await reload();}else{p.foot.textContent=e.message;fail(e);}}}
      finally{p.busy=false;if(p.queued&&!disposed){p.queued=false;refreshPlot(p);}}}
    function plotData(p){const result=p.result,spec=p.spec,hist=result.histogram,kind=spec.kind,style=spec.style||{},traces=[],annotations=[],shapes=[];
      const layout={autosize:true,paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',font:{family:'Arial, sans-serif',size:style.font||13,color:'#223247'},margin:{l:70,r:25,t:style.title?52:30,b:70},
        title:{text:style.title||'',font:{size:(style.font||13)+3}},showlegend:style.legend!==false,legend:{orientation:style.legendSide==='bottom'?'h':'v',traceorder:'normal',groupclick:'toggleitem'},
        xaxis:{title:style.xTitle||spec.x.column+' · '+(spec.x.recipe_name||spec.x.scale||'stored'),tickangle:style.angle||0,zeroline:false},yaxis:{title:style.yTitle||(result.points.length?(spec.y?.column||'')+' · '+(spec.y?.recipe_name||spec.y?.scale||'stored'):{percent:'Cells / bin (%)',proportion:'Proportion / bin',count:'Cells / bin',density:'Probability density',peak:'Peak normalized'}[spec.frequency]),zeroline:false},
        dragmode:'pan',hovermode:'closest',uirevision:p.spec.id+'-'+p.spec.parent+'-'+spec.kind};
      const mid=hist.edges.slice(0,-1).map((v,i)=>(v+hist.edges[i+1])/2),histKinds=histogramKinds,order=histKinds.includes(kind)?seriesOrder(p):hist.series.map((_,i)=>i);
      const displayed=order.map(index=>{const series=hist.series[index];let values=series.values.slice();if(style.smooth){for(let j=0;j<Math.min(10,style.smooth);j++)values=values.map((v,k)=>(values[Math.max(0,k-1)]+2*v+values[Math.min(values.length-1,k+1)])/4);}return {series,index,values};});
      let maxY=0,minPositive=Infinity;for(const {values} of displayed)for(const value of values)if(Number.isFinite(value)&&value>0){maxY=Math.max(maxY,value);minPositive=Math.min(minPositive,value);}
      const matchedRange=style.yLog?(maxY?[Math.log10(minPositive)-.05,Math.log10(maxY)+.05]:[0,1]):[0,maxY?maxY*1.05:1];
      const yLog=style.yLog?'log':'linear';layout.yaxis.type=yLog;
      displayed.forEach(({series,index,values},i)=>{const pop=population(series.population),colour=series.colour||pop.colour,name=series.name||pop.name,opacity=series.opacity??style.opacity??.65;
        const label=name+' · n='+fmt(series.n)+(series.above_percent!=null?' ('+series.above_percent.toFixed(1)+'%)':'')+(series.pinned?' · pinned':'');
        if(histKinds.includes(kind)){
          const common={name:label,uid:'flow-series-'+index,meta:{flowSeries:index},legendrank:i,legendgroup:'flow-series-'+index,line:{color:colour,width:style.line||1.5},opacity:kind==='waterfall'?opacity:1,fillcolor:fillColour(colour,fillOpacity({...style,...(series.opacity!==undefined&&style.fillOpacity===undefined?{opacity:series.opacity}:{})})),hovertemplate:name+'<br>%{x:.4g}<br>%{y:.4g}<extra></extra>'};
          if(kind==='waterfall'){traces.push({...common,type:'scatter3d',mode:'lines',x:mid,y:mid.map(()=>i),z:values,line:{color:colour,width:4}});layout.scene={xaxis:{title:layout.xaxis.title},yaxis:{title:'Population',tickvals:displayed.map((_,j)=>j),ticktext:displayed.map(s=>s.series.name||population(s.series.population).name)},zaxis:{title:layout.yaxis.title},camera:spec.camera||{projection:{type:style.projection||'orthographic'}}};}
          else if(kind==='ridgeline'){const max=Math.max(...values,1e-20),offset=(displayed.length-1-i)*(style.spacing||1.15);traces.push({...common,type:'scatter',mode:'lines',x:mid,y:values.map(v=>v/max+offset),fill:style.fill===false?'none':'tonexty'});layout.yaxis={title:'Offset · peak normalized',tickvals:displayed.map((_,j)=>(displayed.length-1-j)*(style.spacing||1.15)),ticktext:displayed.map(s=>s.series.name||population(s.series.population).name)};}
          else if(kind==='facets'){const axis=i?'y'+(i+1):'y',pad=Math.min(.025,.1/displayed.length);traces.push({...common,type:'scatter',mode:'lines',x:mid,y:values,yaxis:axis,fill:style.fill===false?'none':'tozeroy'});layout['yaxis'+(i?i+1:'')]={domain:[(displayed.length-1-i)/displayed.length+pad,(displayed.length-i)/displayed.length-pad],title:name,type:yLog,autorange:!(style.matchY||spec.y_range),uirevision:String(!!style.matchY)+'-'+yLog,...(style.matchY||spec.y_range?{range:spec.y_range||matchedRange,...(i?{matches:'y'}:{})}:{})};}
          else traces.push({...common,type:'scatter',mode:'lines',x:mid,y:values,fill:style.fill===false?'none':'tozeroy'});
          if(style.median&&series.quantiles&&kind==='histogram'){const q=series.quantiles;shapes.push({type:'rect',xref:'x',yref:'paper',x0:q[0],x1:q[2],y0:0,y1:1,fillcolor:colour,opacity:.1,line:{width:0}}, {type:'line',xref:'x',yref:'paper',x0:q[1],x1:q[1],y0:0,y1:1,line:{color:colour,dash:'dot'}});}
        }else {const point=result.points[index];if(!point)return;
          if(['density','contour'].includes(kind)&&point.density){traces.push({type:kind==='density'?'heatmap':'contour',...point.density,name:label,colorscale:'Viridis',showscale:i===0,opacity,contours:{coloring:'lines'},hovertemplate:'%{z} cells<extra></extra>'});}
          else {let colourValues=colour;if(kind==='density_scatter'&&point.density){const d=point.density,dx=d.x[1]-d.x[0],dy=d.y[1]-d.y[0];colourValues=point.x.map((x,j)=>d.z[Math.max(0,Math.min(95,Math.round((point.y[j]-d.y[0])/dy)))]?.[Math.max(0,Math.min(95,Math.round((x-d.x[0])/dx)))]||0);}
            traces.push({type:'scattergl',mode:'markers',name:label,x:point.x,y:point.y,customdata:point.rows,marker:{color:colourValues,colorscale:'Viridis',size:style.pointSize||3,opacity},hovertemplate:'%{x:.4g}, %{y:.4g}<extra>'+name+'</extra>'});}}
      });
      if(kind==='facets'&&displayed.length){layout.xaxis.anchor=displayed.length===1?'y':'y'+displayed.length;layout.xaxis.side='bottom';layout.xaxis.automargin=true;layout.margin.b=85;}
      const threshold=hist.threshold;
      if(threshold&&!result.gates.some(g=>g.id===threshold.id))shapes.push({type:'line',x0:threshold.bounds[0],x1:threshold.bounds[0],y0:0,y1:1,yref:'paper',line:{color:'#52687e',width:2,dash:'dash'}});
      // SVG paint order runs back to front; legend order runs front to back.
      if(['histogram','ridgeline'].includes(kind))traces.reverse();
      p.shapeGates=[];for(const gate of result.gates){if(gate.kind==='rectangle'||gate.kind==='polygon'){if(!gate.boundary?.length)continue;const parts=gate.boundary.filter(v=>v.every(x=>x!==null));if(parts.length!==gate.boundary.length)continue;
          const same=JSON.stringify(gate.x)===JSON.stringify(result.plot.x)&&JSON.stringify(gate.y)===JSON.stringify(result.plot.y);
          const shape=gate.kind==='rectangle'&&same?{type:'rect',x0:gate.bounds[0],x1:gate.bounds[1],y0:gate.bounds[2],y1:gate.bounds[3]}:{type:'path',path:parts.map((v,i)=>(i?'L':'M')+v.join(',')).join(' ')+' Z'};
          shapes.push({...shape,name:gate.id,line:{color:gate.colour,width:gate.id===p.selectedGate?4:2},editable:false,fillcolor:'rgba(0,0,0,0)'});p.shapeGates[shapes.length-1]=gate;
          annotations.push({x:parts[0][0],y:parts[0][1],text:gate.name,showarrow:false,font:{color:gate.colour,size:style.font||13},yshift:10});
        }else if(['range','threshold'].includes(gate.kind)&&JSON.stringify(gate.x)===JSON.stringify(result.plot.x)){for(const bound of gate.bounds)shapes.push({type:'line',x0:bound,x1:bound,y0:0,y1:1,yref:'paper',line:{color:gate.colour,width:2,dash:'dash'}});}}
      shapes.forEach((s,i)=>{if(!s.name)s.name='decoration-'+i;});
      const overlay=p.statsOverlay;
      if(overlay&&overlay.signature===JSON.stringify([ws.gates,ws.sources,ws.thresholds])&&overlay.result.dataset_revision===result.dataset_revision&&histogramKinds.includes(kind)){
        const report=overlay.result,lines=[],ids=new Set(hist.series.map(s=>s.population)),safe=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
        const features=report.design.features.filter(f=>f.column===spec.x.column&&(f.metric==='positive'?JSON.stringify(f.threshold?.x)===JSON.stringify(spec.x):f.axis?JSON.stringify(f.axis)===JSON.stringify(spec.x):!spec.x.pipeline&&(!spec.x.scale||spec.x.scale==='linear')&&(!spec.x.normalize||spec.x.normalize==='none')));
        for(const f of features){const groups=report.design.groups.filter(g=>hist.series.some(s=>{if(s.population!==g.population)return false;const actual=[spec.rule,s.rule].filter(Boolean),expected=g.rule?.join==='and'?g.rule.rules:g.rule?[g.rule]:[];return JSON.stringify(actual)===JSON.stringify(expected);})),groupIds=new Set(groups.map(g=>g.id));
          for(const g of groups){const summary=report.summaries.find(s=>s.feature===f.id&&s.group===g.id);if(summary)lines.push(safe(g.label)+' · '+safe(f.label)+' = '+(summary.mean==null?'—':Number(summary.mean).toPrecision(4))+' · N='+fmt(summary.N)+' '+safe(report.unit));}
          for(const t of report.tests.filter(t=>t.feature===f.id&&t.family==='comparison'&&groupIds.has(t.a)&&groupIds.has(t.b)))lines.push('p='+(t.p==null?'—':Number(t.p).toPrecision(3))+' · '+(report.correction==='fdr_bh'?'FDR':'adjusted p')+'='+(t.q==null?'—':Number(t.q).toPrecision(3)));
        }
        if(lines.length){layout.margin.t+=lines.length*16+20;annotations.push({xref:'paper',yref:'paper',x:0,y:1,xanchor:'left',yanchor:'bottom',yshift:10,text:lines.join('<br>'),showarrow:false,align:'left',font:{size:11},bgcolor:'#f4f7fb',bordercolor:'#cad3e1',borderpad:5});}
      }
      layout.shapes=shapes;layout.annotations=annotations;
      for(const [key,value] of Object.entries(layout))if(/^[xy]axis\d*$/.test(key))Object.assign(value,{showline:true,mirror:true,linecolor:'#788697',linewidth:1,ticks:'outside'});
      if(spec.x_range)layout.xaxis.range=spec.x_range;if(spec.y_range)layout.yaxis.range=spec.y_range;
      if(kind!=='waterfall'){if(['rectangle','range'].includes(spec.tool))layout.dragmode='drawrect';if(spec.tool==='polygon')layout.dragmode='drawclosedpath';}
      layout.newshape={line:{color:'#d23b55',width:2},fillcolor:'rgba(210,59,85,.08)'};return {traces,layout};}
    async function renderPlot(p){if(!p.result)return;p.syncHistogramControls?.();await loadPlotly(hooks.plotly);p.rendering=true;const {traces,layout}=plotData(p);
      await Plotly.react(p.chart,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToAdd:['drawrect','drawclosedpath','eraseshape'],toImageButtonOptions:{format:'png',filename:'flow-gating',scale:3}});
      p.chart.removeAllListeners?.('plotly_relayout');p.chart.on('plotly_relayout',event=>{if(p.rendering)return;
        if(event['scene.camera'])p.spec.camera=event['scene.camera'];
        drawGateHandles(p);
        if(event.shapes){const shape=event.shapes.find(s=>!s.name);if(shape){const gate={id:uid(),name:'Population '+(ws.gates.length+1),colour:colours[ws.gates.length%colours.length],parent:p.spec.parent,x:clone(p.result.plot.x),...(p.spec.rule?{rule:clone(p.spec.rule)}:{})};
            if(!p.result.plot.y){gate.kind='range';gate.bounds=[Math.min(shape.x0,shape.x1),Math.max(shape.x0,shape.x1)];}
            else {gate.y=clone(p.result.plot.y);if(shape.type==='path'){gate.kind='polygon';const numbers=(shape.path.match(/[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?/gi)||[]).map(Number);gate.vertices=[];for(let i=0;i<numbers.length;i+=2)gate.vertices.push(numbers.slice(i,i+2));}else{gate.kind='rectangle';gate.bounds=[Math.min(shape.x0,shape.x1),Math.max(shape.x0,shape.x1),Math.min(shape.y0,shape.y1),Math.max(shape.y0,shape.y1)];}}
            editGate(gate,true);renderPlot(p);}}
        else {const key=Object.keys(event).find(k=>/^shapes\[\d+\]/.test(k));if(key){const i=Number(key.match(/\d+/)[0]),gate=p.shapeGates[i],shape=p.chart.layout.shapes[i];if(gate&&gate.kind==='rectangle')editGate({...gate,bounds:[shape.x0,shape.x1,shape.y0,shape.y1]});}}
      });
      p.chart.onpointerdown=event=>{if(event.button!==0||!['quadrant','threshold'].includes(p.spec.tool)||p.spec.kind==='waterfall')return;const rect=p.chart.getBoundingClientRect(),fl=p.chart._fullLayout,px=event.clientX-rect.left-fl.xaxis._offset,py=event.clientY-rect.top-fl.yaxis._offset;
        if(px<0||px>fl.xaxis._length||event.clientY-rect.top<fl._size.t||event.clientY-rect.top>fl._size.t+fl._size.h)return;const x=fl.xaxis.p2d(px),y=fl.yaxis.p2d(py);
        if(p.spec.tool==='threshold')editGate({id:uid(),kind:'threshold',name:'High '+p.spec.x.column,colour:colours[ws.gates.length%colours.length],parent:p.spec.parent,x:clone(p.result.plot.x),bounds:[x],above:true,...(p.spec.rule?{rule:clone(p.spec.rule)}:{})},true);
        else if(p.result.plot.y){const gates=[0,1,2,3].map(i=>({id:uid(),kind:'quadrant',name:'Q'+(i+1),colour:colours[(ws.gates.length+i)%colours.length],parent:p.spec.parent,x:clone(p.result.plot.x),y:clone(p.result.plot.y),bounds:[x,y],right:i%2===0,above:i<2,...(p.spec.rule?{rule:clone(p.spec.rule)}:{})}));dialog('Four quadrants',node('p',{text:'Create four non-overlapping populations at this point?'}),[['Create',async d=>{await save({gates:[...ws.gates,...gates]});d.close();refreshPlots();},true]]);}}
      p.chart.oncontextmenu=event=>{const axis=event.target.closest('.xtick,.xtitle')?'x':event.target.closest('.ytick,.ytitle')?'y':null;if(axis){event.stopPropagation();transformationMenu(p,axis,event);return;}const activeThreshold=p.result.histogram.threshold,fl=p.chart._fullLayout,r=p.chart.getBoundingClientRect();if(activeThreshold&&Math.abs(event.clientX-r.left-fl.xaxis._offset-fl.xaxis.d2p(activeThreshold.bounds[0]))<=10&&event.clientY-r.top>=fl._size.t&&event.clientY-r.top<=fl._size.t+fl._size.h){event.stopPropagation();const gate=ws.gates.find(g=>g.id===activeThreshold.id);menu(event,[...thresholdActions(p,activeThreshold),...(gate?[['Edit gate…',()=>editGate(gate)],['Show cells on image',()=>showPopulation(gate.id)],['Delete gate',()=>deleteGate(gate)]]:[])]);return;}const gates=gatesAt(p,event);if(!gates.length)return;event.stopPropagation();
        if(gates.length===1){p.selectedGate=gates[0].id;drawGateHandles(p);}
        menu(event,gates.length===1?[...(gates[0].kind==='threshold'?thresholdActions(p,gates[0]):[]),['Show cells on image',()=>showPopulation(gates[0].id)],['Edit gate…',()=>editGate(population(gates[0].id))],['Delete gate',()=>deleteGate(population(gates[0].id))],['Open population',()=>pick(gates[0].id)]]:
          gates.map(g=>['Show '+g.name+' on image',()=>showPopulation(g.id)]));};
      p.chart.onclick=event=>{if(event.target.closest('.flow-gate-target,.legend')||(p.spec.tool||'pan')!=='pan')return;const gate=gatesAt(p,event)[0];p.selectedGate=gate?.id;activePlot=p;drawGateHandles(p);if(gate)setStatus(gate.name+' · drag handles to resize');};
      drawGateHandles(p);bindAxisTitles(p);bindLegendOrder(p);if(!p.axisHooked){p.chart.on('plotly_afterplot',()=>bindAxisTitles(p));p.axisHooked=true;}p.rendering=false;
    }
    function thresholdActions(p,gate){return [
      ['Apply to all facets in current plot',async()=>{p.spec.threshold_id=gate.id;await persistPlots();await refreshPlot(p);}],
      ['Apply to matching open plots',async()=>{const targets=[...plots.values()].filter(q=>JSON.stringify(q.spec.x)===JSON.stringify(gate.x));targets.forEach(q=>q.spec.threshold_id=gate.id);await persistPlots();await Promise.all(targets.map(refreshPlot));setStatus('Threshold applied to '+targets.length+' plots with the same marker and transformation.');}],
      ['Save as named global threshold…',()=>{const name=input(gate.name,'Threshold name');dialog('Global threshold · '+gate.x.column,field('Name',name),[['Save',async d=>{if(!name.value.trim())throw Error('Enter a threshold name.');const value={id:uid(),name:name.value.trim(),x:clone(gate.x),bounds:[gate.bounds[0]]};await save({thresholds:[...(ws.thresholds||[]),value]});p.spec.threshold_id=value.id;await persistPlots();await refreshPlot(p);d.close();},true]]);}]
    ];}
    function thresholdMenu(p,event){const candidates=[...(ws.thresholds||[]),...ws.gates.filter(g=>g.kind==='threshold')].filter(g=>JSON.stringify(g.x)===JSON.stringify(p.spec.x));const current=candidates.find(g=>g.id===(p.spec.threshold_id||p.result?.histogram.threshold?.id));
      menu(event,[['Draw threshold',()=>{p.spec.tool='threshold';p.controls.gateTool.value='threshold';renderPlot(p);}],...candidates.map(g=>[g.name+' @ '+Number(g.bounds[0]).toPrecision(4),async()=>{p.spec.threshold_id=g.id;await persistPlots();refreshPlot(p);}]),...(current?thresholdActions(p,current):[]),['Hide threshold readout',async()=>{p.spec.threshold_id='none';await persistPlots();refreshPlot(p);}] ]);}
    function drawGateHandles(p){
      if(p.draggingGate)return;
      p.chart.querySelector('.flow-gate-handles')?.remove();const g=ws.gates.find(g=>g.id===p.selectedGate),fl=p.chart._fullLayout;
      if(!g||!fl||!p.result||p.spec.kind==='waterfall'||JSON.stringify(g.x)!==JSON.stringify(p.result.plot.x)||(g.y&&JSON.stringify(g.y)!==JSON.stringify(p.result.plot.y)))return;
      const overlay=node('div',{class:'flow-gate-handles'}),points=[],px=x=>fl.xaxis._offset+fl.xaxis.d2p(x),py=y=>fl.yaxis._offset+fl.yaxis.d2p(y),midY=fl.yaxis._offset+fl.yaxis._length/2;
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),path=document.createElementNS(svg.namespaceURI,'path');svg.classList.add('flow-gate-preview');path.setAttribute('stroke',g.colour);path.setAttribute('fill','none');path.setAttribute('stroke-width','3');path.setAttribute('stroke-dasharray','6 3');svg.append(path);overlay.append(svg);
      const preview=draft=>{let d='';
        if(draft.kind==='rectangle'){const [a,b,c,e]=draft.bounds;d=`M ${px(a)},${py(c)} L ${px(b)},${py(c)} L ${px(b)},${py(e)} L ${px(a)},${py(e)} Z`;}
        else if(draft.kind==='polygon')d=draft.vertices.map(([x,y],i)=>(i?'L':'M')+' '+px(x)+','+py(y)).join(' ')+' Z';
        else if(['range','threshold'].includes(draft.kind))d=draft.bounds.map(x=>`M ${px(x)},${fl.yaxis._offset} V ${fl.yaxis._offset+fl.yaxis._length}`).join(' ');
        else if(draft.kind==='quadrant')d=`M ${px(draft.bounds[0])},${fl.yaxis._offset} V ${fl.yaxis._offset+fl.yaxis._length} M ${fl.xaxis._offset},${py(draft.bounds[1])} H ${fl.xaxis._offset+fl.xaxis._length}`;
        path.setAttribute('d',d);};
      if(g.kind==='rectangle'){const [a,b,c,d]=g.bounds;
        for(const [xi,yi] of [[0,2],[1,2],[1,3],[0,3]])points.push({x:g.bounds[xi],y:g.bounds[yi],cursor:(xi===0)===(yi===2)?'nesw-resize':'nwse-resize',move:(draft,x,y)=>{draft.bounds[xi]=x;draft.bounds[yi]=y;}});
        for(const xi of [0,1])points.push({x:g.bounds[xi],y:(c+d)/2,edge:true,cursor:'ew-resize',height:Math.abs(py(c)-py(d)),move:(draft,x)=>draft.bounds[xi]=x});
        for(const yi of [2,3])points.push({x:(a+b)/2,y:g.bounds[yi],edge:true,cursor:'ns-resize',width:Math.abs(px(a)-px(b)),move:(draft,x,y)=>draft.bounds[yi]=y});
        points.push({x:(a+b)/2,y:(c+d)/2,translate:true,move:(draft,x,y,dx,dy)=>{draft.bounds=g.bounds.map((v,i)=>v+(i<2?dx:dy));}});
      }
      else if(g.kind==='polygon')g.vertices.forEach(([x,y],i)=>points.push({x,y,move:(draft,x,y)=>{draft.vertices[i]=[x,y];}}));
      else if(['range','threshold'].includes(g.kind))g.bounds.forEach((x,i)=>points.push({x,y:null,cursor:'ew-resize',height:fl.yaxis._length,move:(draft,x)=>{draft.bounds[i]=x;}}));
      else if(g.kind==='quadrant')points.push({x:g.bounds[0],y:g.bounds[1],move:(draft,x,y)=>{draft.bounds=[x,y];}});
      if(g.kind==='polygon')points.push({x:g.vertices.reduce((n,v)=>n+v[0],0)/g.vertices.length,y:g.vertices.reduce((n,v)=>n+v[1],0)/g.vertices.length,translate:true,move:(draft,x,y,dx,dy)=>{draft.vertices=g.vertices.map(([a,b])=>[a+dx,b+dy]);}});
      points.forEach((point,i)=>{const left=px(point.x),top=point.y==null?midY:py(point.y);if(!Number.isFinite(left)||!Number.isFinite(top))return;
        const handle=button('',()=>{},{class:'flow-gate-target '+(point.translate?'flow-gate-move':point.edge?'flow-gate-edge':'flow-gate-handle'),'aria-label':(point.translate?'Move '+g.name:'Resize '+g.name+' handle '+(i+1)),title:point.translate?'Drag to move gate':'Drag to resize gate',style:'left:'+left+'px;top:'+top+'px;color:'+g.colour+';cursor:'+(point.cursor||'move')+(point.width?';width:'+Math.max(26,point.width)+'px':'')+(point.height?';height:'+Math.max(26,point.height)+'px':'')});
        handle.onpointerdown=event=>{event.preventDefault();event.stopPropagation();if(event.button!==0)return;if(event.isTrusted)handle.setPointerCapture(event.pointerId);const draft=clone(g),r=p.chart.getBoundingClientRect(),startX=fl.xaxis.p2d(event.clientX-r.left-fl.xaxis._offset),startY=fl.yaxis.p2d(event.clientY-r.top-fl.yaxis._offset);let moved=false;p.draggingGate=true;
          handle.onpointermove=e=>{const x=Math.max(0,Math.min(fl.xaxis._length,e.clientX-r.left-fl.xaxis._offset)),y=Math.max(0,Math.min(fl.yaxis._length,e.clientY-r.top-fl.yaxis._offset));
            const dx=fl.xaxis.p2d(x)-startX,dy=fl.yaxis.p2d(y)-startY;point.move(draft,point.x+dx,(point.y||0)+dy,dx,dy);preview(draft);moved=true;};
          const cleanup=()=>{p.draggingGate=false;handle.onpointermove=null;handle.onpointerup=null;handle.onpointercancel=null;p.cancelGateDrag=null;document.removeEventListener('keydown',cancel);};
          const cancel=e=>{if(e?.key&&e.key!=='Escape')return;e?.preventDefault();e?.stopPropagation();cleanup();drawGateHandles(p);};p.cancelGateDrag=cancel;document.addEventListener('keydown',cancel);
          handle.onpointerup=async e=>{e.stopPropagation();cleanup();if(!moved){drawGateHandles(p);return;}
            if(draft.kind==='rectangle')draft.bounds=[Math.min(...draft.bounds.slice(0,2)),Math.max(...draft.bounds.slice(0,2)),Math.min(...draft.bounds.slice(2)),Math.max(...draft.bounds.slice(2))];
            if(draft.kind==='range')draft.bounds.sort((a,b)=>a-b);
            try{await save({gates:ws.gates.map(item=>item.id===draft.id?draft:item)});refreshPlots();}catch(error){fail(error);drawGateHandles(p);}};
          handle.onpointercancel=cancel;};overlay.append(handle);});
      p.chart.append(overlay);
    }
    function gatesAt(p,event){
      if(!p.result||p.spec.kind==='waterfall')return [];
      const fl=p.chart._fullLayout,r=p.chart.getBoundingClientRect(),x=event.clientX-r.left-fl.xaxis._offset,y=event.clientY-r.top-fl.yaxis._offset;
      if(x<0||x>fl.xaxis._length||event.clientY-r.top<fl._size.t||event.clientY-r.top>fl._size.t+fl._size.h)return [];
      const shape=event.target.closest?.('[data-index]'),direct=shape&&p.shapeGates[Number(shape.dataset.index)];if(direct)return [direct];
      const within=vertices=>{let inside=false;
        for(let i=0,j=vertices.length-1;i<vertices.length;j=i++){
          const a=vertices[j],b=vertices[i],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy||1)));
          if(Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)<=7)return true;
          if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])inside=!inside;
        }return inside;};
      return p.result.gates.filter(g=>{
        if(g.boundary?.length&&g.boundary.every(v=>v.every(Number.isFinite)))return within(g.boundary.map(([a,b])=>[fl.xaxis.d2p(a),fl.yaxis.d2p(b)]));
        if(JSON.stringify(g.x)!==JSON.stringify(p.result.plot.x))return false;
        const bounds=(g.bounds||[]).map(v=>fl.xaxis.d2p(v));
        if(g.kind==='range')return x>=Math.min(...bounds)-7&&x<=Math.max(...bounds)+7;
        if(g.kind==='threshold')return Math.abs(x-bounds[0])<=7||(g.above!==false?x>=bounds[0]:x<=bounds[0]);
        if(g.kind==='quadrant'&&JSON.stringify(g.y)===JSON.stringify(p.result.plot.y))return (g.right?x>=bounds[0]:x<=bounds[0])&&(g.above?y<=fl.yaxis.d2p(g.bounds[1]):y>=fl.yaxis.d2p(g.bounds[1]));
        return false;
      });
    }
    function plotFilters(p){const rules=ruleEditor(p.spec.rule);dialog('Plot filters',rules,[['Clear filters',async d=>{delete p.spec.rule;await refreshPlot(p);await persistPlots();d.close();}],['Apply',async d=>{p.spec.rule=rules.read();await refreshPlot(p);await persistPlots();d.close();},true]]);}
    function axisSettings(spec,label){const column=spec.column,scale=select([['linear','Stored / linear'],['arcsinh','Arcsinh'],['log10','Log10'],...(info.logicle?[['logicle','Logicle']]:[])],spec.scale||'linear',label+' scale');
      const cofactor=input(spec.cofactor??5,label+' cofactor','number'),norm=select([['none','None'],['zscore','Z-score'],['percentile','Percentile scale']],spec.normalize||'none',label+' normalization'),group=select([['','All analysis cells'],...info.columns.map(c=>[c,c])],spec.group||'',label+' normalization group');
      const box=details(label+' transformation',field('Display',scale),field('Cofactor',cofactor),field('Normalize',norm),field('Within',group),help('Fits on fixed analysis cells, not the current child or live region. Z-score uses population SD. Scaling within groups can remove biological differences. Stored columns are never overwritten.'));
      let edited=false;box.addEventListener('change',()=>edited=true);
      if(spec.pipeline)box.prepend(node('p',{class:'flow-small',text:'Recipe: '+(spec.recipe_name||'Custom')+' · use Transform to edit its steps.'}));
      box.read=()=>!edited?clone(spec):({column,scale:scale.value,cofactor:Number(cofactor.value),normalize:norm.value,group:group.value});return box;}
    function recipeDialog(p,selectedId=undefined){
      const ops=[['arcsinh','Arcsinh'],['log10','Log10'],['log1p','Log1p'],...(info.logicle?[['logicle','Logicle']]:[]),['zscore','Z1 · Across cells'],['row_zscore','Z2 · Within cell'],['cdf','CDF'],['nlog','NLog'],['clip','Percentile clipping'],['minmax','Min / Max'],['percentile','Percentile scaling'],['robust_zscore','Robust Z-score'],['relu','Threshold / ReLU'],['sqrt','Square root'],['linear','Scale + offset'],['identity','Unchanged']];
      let editing=null,steps=[],loadedVersion=0;
      const name=input('','Transformation name'),list=node('div',{class:'flow-transform-steps'}),preset=select([['','New transformation'],...(ws.recipes||[]).map(r=>[r.id,r.name])],'','Saved transformations');
      name.placeholder='e.g. Arcsinh + Z1';name.maxLength=100;
      const add=button('+ Add step',()=>{if(steps.length<16){steps.push({op:'arcsinh',cofactor:5});draw();list.lastElementChild?.scrollIntoView({block:'nearest'});}});
      function load(recipe){editing=recipe?.id||null;loadedVersion=recipe?.version||1;name.value=recipe?.name||'';preset.value=editing||'';
        steps=recipe?clone(recipe.pipeline||[]):[{op:'arcsinh',cofactor:5}];
        steps.forEach(step=>{delete step.fitted;if(step.op==='row_zscore'&&!step.marker_source){step.marker_source=(step.markers||recipe?.markers)?.length?'custom':'axes';if(step.marker_source==='custom')step.markers=clone(step.markers||recipe.markers);}});draw();}
      function move(i,direction){const target=i+direction;if(target<0||target>=steps.length)return;[steps[i],steps[target]]=[steps[target],steps[i]];draw();}
      function draw(){list.replaceChildren();add.disabled=steps.length>=16;add.title=add.disabled?'Maximum 16 steps':'';
        if(!steps.length)list.append(node('p',{class:'flow-small',text:'Add your first step.'}));
        steps.forEach((step,i)=>{const enabled=node('input',{type:'checkbox','aria-label':'Enable step '+(i+1)});enabled.checked=step.enabled!==false;
          const card=node('section',{class:'flow-transform-step'+(enabled.checked?'':' disabled'),'data-step':i+1});
          enabled.onchange=()=>{step.enabled=enabled.checked;card.classList.toggle('disabled',!enabled.checked);};
          const op=select(ops,step.op,'Step '+(i+1)+' operation');op.onchange=()=>{steps[i]={op:op.value,enabled:step.enabled!==false,...(step.name?{name:step.name}:{})};draw();};
          const settings=details('Settings'),label=input(step.name||'','Step '+(i+1)+' label');label.placeholder='Optional step label';label.oninput=()=>step.name=label.value;settings.append(field('Label',label));
          const number=(key,title,value,limits={})=>{const e=input(step[key]??value,'Step '+(i+1)+' '+title,'number');Object.entries(limits).forEach(([k,v])=>e.setAttribute(k,String(v)));e.oninput=()=>step[key]=e.value===''?null:Number(e.value);settings.append(field(title,e));};
          if(step.op==='arcsinh'){const method=select([['fixed','Fixed'],['dynamic','Dynamic']],step.dynamic?'dynamic':'fixed','Step '+(i+1)+' cofactor mode');method.onchange=()=>{step.dynamic=method.value==='dynamic';draw();list.children[i].querySelector('details').open=true;};settings.append(field('Cofactor',method));number(step.dynamic?'cofactor_percentile':'cofactor',step.dynamic?'Percentile':'Value',5,step.dynamic?{min:0,max:100}:{min:0});}
          if(step.op==='logicle')for(const [key,value] of Object.entries({t:262144,m:4.5,w:.5,a:0}))number(key,key.toUpperCase(),value);
          if(['clip','percentile'].includes(step.op)){number('low','Low %',1,{min:0,max:100});number('high','High %',99,{min:0,max:100});}
          if(step.op==='relu'){number('threshold','Threshold',0);number('ceiling','Ceiling',1);}
          if(step.op==='linear'){number('multiplier','Multiply',1);number('offset','Add',0);}
          if(['zscore','robust_zscore','clip','percentile','minmax'].includes(step.op)||step.op==='arcsinh'&&step.dynamic){const group=select([['','All Analysis cells'],...info.columns.map(c=>[c,c])],step.group||'','Step '+(i+1)+' grouping');group.onchange=()=>step.group=group.value;settings.append(field('Within',group));}
          if(step.op==='minmax'){const clip=node('input',{type:'checkbox','aria-label':'Step '+(i+1)+' clip to 0–1'});clip.checked=!!step.clip;clip.onchange=()=>step.clip=clip.checked;settings.append(field('Clip 0–1',clip));}
          if(step.op==='row_zscore'){
            const mode=select([['axes','Plot axes'],['custom','Choose markers']],step.marker_source||'axes','Step '+(i+1)+' markers');settings.append(field('Markers',mode));
            mode.onchange=()=>{step.marker_source=mode.value;if(mode.value==='custom'&&!step.markers)step.markers=[...new Set([p.spec.x.column,p.spec.y?.column].filter(Boolean))];draw();list.children[i].querySelector('details').open=true;};
            if(mode.value==='custom'){const chosen=new Set(step.markers||[]),search=input('','Step '+(i+1)+' find marker'),choices=node('div',{class:'flow-checkbox-list'});search.placeholder='Find marker…';
              const drawMarkers=()=>{choices.replaceChildren(...info.numeric.filter(m=>m.toLowerCase().includes(search.value.toLowerCase())).map(m=>{const check=node('input',{type:'checkbox','aria-label':'Step '+(i+1)+' marker '+m});check.checked=chosen.has(m);check.onchange=()=>{check.checked?chosen.add(m):chosen.delete(m);step.markers=[...chosen];};return node('label',{},[check,document.createTextNode(' '+m)]);}));};
              search.oninput=drawMarkers;drawMarkers();settings.append(search,choices);}
            settings.append(node('span',{class:'flow-small',text:'The axis marker is included.'}));
          }
          const up=button('↑',()=>move(i,-1),{'aria-label':'Move step '+(i+1)+' up',title:'Move up'}),down=button('↓',()=>move(i,1),{'aria-label':'Move step '+(i+1)+' down',title:'Move down'});up.disabled=i===0;down.disabled=i===steps.length-1;
          card.append(node('div',{class:'flow-transform-step-head'},[node('span',{class:'flow-step-number',text:String(i+1)}),op,enabled,up,down,button('×',()=>{steps.splice(i,1);draw();},{'aria-label':'Remove step '+(i+1),title:'Remove step'})]),settings);
          card.oncontextmenu=e=>{if(e.target.closest('input,select'))return;menu(e,[['Move up',()=>move(i,-1)],['Move down',()=>move(i,1)],['Duplicate step',()=>{if(steps.length>=16){fail(Error('Use at most 16 steps.'));return;}steps.splice(i+1,0,clone(step));draw();}],['Remove step',()=>{steps.splice(i,1);draw();}]]);};list.append(card);
        });
      }
      preset.onchange=()=>load((ws.recipes||[]).find(r=>r.id===preset.value));
      const body=node('div',{class:'flow-transform-builder'},[node('div',{class:'flow-transform-library'},[preset,button('New',()=>load(null))]),field('Name',name),list,add,node('p',{class:'flow-small',text:'Use saved transformations from an axis’s right-click menu.'})]);
      const current=selectedId===undefined?(ws.recipes||[]).find(r=>r.id===p.spec.x.recipe_id||r.name===p.spec.x.recipe_name):(ws.recipes||[]).find(r=>r.id===selectedId);load(current);
      const d=dialog('Transformations',body,[['Save transformation',async dialog=>{
        const title=name.value.trim();if(!title)throw Error('Name this transformation.');if(!steps.some(s=>s.enabled!==false))throw Error('Add at least one enabled step.');
        for(const [i,step] of steps.entries()){if(step.enabled===false)continue;const problem=message=>{throw Error('Step '+(i+1)+': '+message);};
          for(const [key,value] of Object.entries(step))if(value===null||typeof value==='number'&&!Number.isFinite(value))problem('Enter a valid '+key+'.');
          if(step.op==='arcsinh'&&!step.dynamic&&(step.cofactor??5)<=0)problem('Cofactor must be greater than zero.');
          if(step.op==='arcsinh'&&step.dynamic&&((step.cofactor_percentile??5)<0||(step.cofactor_percentile??5)>100))problem('Use a percentile from 0 to 100.');
          if(['clip','percentile'].includes(step.op)&&!((step.low??1)>=0&&(step.low??1)<(step.high??99)&&(step.high??99)<=100))problem('Use 0 ≤ Low < High ≤ 100.');
          if(step.op==='relu'&&(step.ceiling??1)<=(step.threshold??0))problem('Ceiling must exceed Threshold.');
          if(step.op==='row_zscore'&&step.marker_source==='custom'&&!step.markers?.length)problem('Choose markers or use Plot axes.');
        }
        const id=editing||uid(),pipeline=clone(steps);pipeline.forEach(s=>delete s.fitted);
        await save(()=>{const all=ws.recipes||[],old=all.find(r=>r.id===editing);if(old&&(old.version||1)!==loadedVersion)throw Error('This transformation changed. Reopen it before saving.');
          if(all.some(r=>r.id!==editing&&r.name.trim().toLowerCase()===title.toLowerCase()))throw Error('That name is already used. Choose another name.');
          const recipe={id,name:title,pipeline,version:old?(old.version||1)+1:1};return {recipes:old?all.map(r=>r.id===id?recipe:r):[...all,recipe]};});dialog.close();setStatus('Transformation saved');
      },true]]);d.classList.add('flow-transform-dialog');
    }
    function plotSettings(p){const s=p.spec,style=s.style||{},x=axisSettings(s.x,'X'),y=axisSettings(s.y||{column:info.numeric[0]},'Y'),sample=input(s.sample,'Scatter sample limit','number'),bins=input(s.bins,'Histogram bins','number'),frequency=select(frequencies,s.frequency,'Histogram Y');
      const title=input(style.title||'','Plot title'),font=input(style.font||13,'Text size','number'),angle=input(style.angle||0,'Tick rotation','number'),opacity=input(style.opacity??.65,'Opacity','number'),line=input(style.line||1.5,'Line width','number'),smooth=input(style.smooth||0,'Smoothing','number'),spacing=input(style.spacing||1.15,'Curve spacing','number');
      const fillAlpha=opacityControl(style,'Figure fill opacity'),match=node('input',{type:'checkbox','aria-label':'Match facet Y scales'});match.checked=!!style.matchY;
      const xTitle=input(style.xTitle||'','X axis title'),yTitle=input(style.yTitle||'','Y axis title'),xr=input(s.x_range?.join(', ')||'','X limits'),yr=input(s.y_range?.join(', ')||'','Y limits'),patient=select([['','Pooled cells'],...info.columns.map(c=>[c,'Equal weight: '+c])],s.patient||'','Patient weighting');
      const legend=select([['right','Right'],['bottom','Bottom'],['none','Hidden']],style.legend===false?'none':style.legendSide||'right','Legend position'),projection=select(['orthographic','perspective'],style.projection||'orthographic','3D projection');
      const median=node('input',{type:'checkbox'});median.checked=!!style.median;const fill=node('input',{type:'checkbox'});fill.checked=style.fill!==false;const log=node('input',{type:'checkbox'});log.checked=!!style.yLog;
      const box=node('div',{},[x,y,details('Display and sampling',field('Sample',sample),help('Limits rendered scatter points only. All eligible cells contribute to gates, counts and densities. Use density raster for a complete population view.'),field('Bins',bins),field('Y',frequency),field('Weight',patient),field('Smooth',smooth),field('Spacing',spacing),field('Camera',projection)),
        details('Text and style',field('Title',title),field('X label',xTitle),field('Y label',yTitle),field('Text size',font),field('Tick tilt',angle),filledHistogram(s)?field('Fill opacity',fillAlpha.box):field('Opacity',opacity),field('Line',line),field('Legend',legend),field('Median/IQR',median),field('Fill',fill),field('Log Y',log)),details('Axis limits',s.kind==='facets'&&field('Match Y scales',match),field('X min, max',xr),field('Y min, max',yr))]);
      const numerical=()=>JSON.stringify([s.x,histogramKinds.includes(s.kind)?null:s.y,s.sample,s.bins,s.frequency,s.patient||'',s.x_range||null,s.y_range||null]);
      dialog('Figure settings',box,[['Apply',async d=>{const before=numerical();s.x=x.read();s.y=y.read();s.sample=Number(sample.value);s.bins=Number(bins.value);s.frequency=frequency.value;s.patient=patient.value;
        s.x_range=xr.value?xr.value.split(',').map(Number):null;s.y_range=yr.value?yr.value.split(',').map(Number):null;
        s.style={...style,title:title.value,xTitle:xTitle.value,yTitle:yTitle.value,font:Number(font.value),angle:Number(angle.value),opacity:Number(opacity.value),...(filledHistogram(s)?{fillOpacity:Number(fillAlpha.slider.value)/100}:{}),matchY:match.checked,line:Number(line.value),smooth:Number(smooth.value),spacing:Number(spacing.value),median:median.checked,fill:fill.checked,yLog:log.checked,legend:legend.value!=='none',legendSide:legend.value,projection:projection.value};
        s.camera={...(s.camera||{}),projection:{type:projection.value}};
        if(before===numerical()&&p.result)await renderPlot(p);else await refreshPlot(p);await persistPlots();d.close();},true]]);}
    function seriesDialog(p){const series=clone(p.spec.series?.length?p.spec.series:[{population:p.spec.parent}]);const box=node('div'),rows=node('div'),split=select(info.columns,info.columns[0],'Split comparison by');
      function draw(){rows.replaceChildren();series.forEach((s,i)=>{const pop=select(populationChoices(),s.population,'Series population'),name=input(s.name||population(s.population).name,'Series name'),colour=input(s.colour||population(s.population).colour,'Series colour','color'),pin=node('input',{type:'checkbox','aria-label':'Pin reference'});pin.checked=!!s.pinned;
        pop.onchange=()=>s.population=pop.value;name.onchange=()=>s.name=name.value;colour.onchange=()=>s.colour=colour.value;pin.onchange=()=>s.pinned=pin.checked;
        rows.append(node('div',{class:'flow-series'},[node('div',{class:'flow-row'},[pop,name,colour,node('label',{},[pin,document.createTextNode(' Pin')])]),node('div',{class:'flow-actions'},[
          button('Filters…',()=>{const rules=ruleEditor(s.rule);dialog('Series filters',rules,[['Apply',d=>{s.rule=rules.read();d.close();},true]]);}),button('Duplicate',()=>{series.splice(i+1,0,clone(s));draw();}),button('↑',()=>{if(i){[series[i-1],series[i]]=[series[i],series[i-1]];draw();}}),button('Remove',()=>{series.splice(i,1);draw();})])]));});}
      draw();box.append(rows,button('Add series',()=>{series.push({population:ws.active||'root'});draw();}),field('Split by',split),button('Add groups',async()=>{try{const values=await call('values',{column:split.value});Object.keys(values.values).slice(0,60).forEach((v,i)=>series.push({population:p.spec.parent,name:v,colour:colours[i%colours.length],rule:{column:split.value,op:'in',values:[v]}}));draw();}catch(e){fail(e);}}),help('Each curve has its own parent and filters. Pin keeps a reference independent of Live view. All curves share bins and axis transforms.'));
      dialog('Compare populations',box,[['Apply',async d=>{p.spec.series=series;await refreshPlot(p);await persistPlots();d.close();},true]]);}
    function importGates(){const file=input('','Gate template file','file');file.accept='.json';file.onchange=async()=>{try{const template=JSON.parse(await file.files[0].text());if(template.version!==1||!Array.isArray(template.gates))throw Error('Unsupported gate template.');
        const ids=new Map(template.gates.map(g=>[g.id,uid()]));const gates=template.gates.map(g=>({...g,id:ids.get(g.id),parent:ids.get(g.parent)||'root',refs:g.refs?.map(id=>ids.get(id)||id)}));
        dialog('Import gates',node('p',{text:gates.length+' gates will be evaluated on the checked images. Review their transforms and units before applying labels.'}),[['Import',async d=>{await save({gates:[...ws.gates,...gates]});refreshPlots();d.close();},true]]);
      }catch(e){fail(e);}};file.click();}
    async function freezePlot(p){try{const snapshot=clone(p.result);snapshot.plot=clone(p.spec);let layouts=clone(ws.layouts);let page=layouts.find(l=>l.id===layoutId);
      if(!page){page={id:uid(),name:'Publication page',width:1100,height:800,panels:[]};layouts.push(page);layoutId=page.id;}
      const i=page.panels.length;page.panels.push({id:uid(),x:25+(i%2)*520,y:30+Math.floor(i/2)*400,w:510,h:390,letter:String.fromCharCode(65+i),snapshot,frozen_at:new Date().toISOString()});
      await save({layouts});showLayout();}catch(e){fail(e);}}
    function snapshotStyle(panel,saved){const s=panel.snapshot.plot.style||{},title=input(s.title||'','Panel title'),font=input(s.font||13,'Panel text size','number'),angle=input(s.angle||0,'Panel tick rotation','number'),x=input(s.xTitle||'','Panel X title'),y=input(s.yTitle||'','Panel Y title'),opacity=input(s.opacity??.65,'Panel opacity','number'),legend=select(['right','bottom','none'],s.legend===false?'none':s.legendSide||'right','Panel legend');
      const fillAlpha=opacityControl(s,'Panel fill opacity'),match=node('input',{type:'checkbox','aria-label':'Match panel Y scales'});match.checked=!!s.matchY;
      dialog('Panel appearance',node('div',{},[field('Title',title),field('X label',x),field('Y label',y),field('Text size',font),field('Tick tilt',angle),filledHistogram(panel.snapshot.plot)?field('Fill opacity',fillAlpha.box):field('Opacity',opacity),field('Legend',legend),panel.snapshot.plot.kind==='facets'&&field('Match Y scales',match)]),[['Apply',async d=>{panel.snapshot.plot.style={...s,title:title.value,font:Number(font.value),angle:Number(angle.value),xTitle:x.value,yTitle:y.value,opacity:Number(opacity.value),...(filledHistogram(panel.snapshot.plot)?{fillOpacity:Number(fillAlpha.slider.value)/100}:{}),matchY:match.checked,legend:legend.value!=='none',legendSide:legend.value};await saved();d.close();showLayout();},true]]);}
    function showLayout(){statsView?.hide();const layouts=ws.layouts;if(!layouts.length){work.replaceChildren(node('div',{class:'flow-empty'},[node('p',{text:'Add a plot to start a publication page.'}),button('Add current plot',()=>activePlot&&freezePlot(activePlot))]));return;}
      let layout=layouts.find(l=>l.id===layoutId)||layouts[0];layoutId=layout.id;const choose=select(layouts.map(l=>[l.id,l.name]),layoutId,'Publication page');choose.onchange=()=>{layoutId=choose.value;showLayout();};
      const width=input(layout.width,'Page width','number'),height=input(layout.height,'Page height','number'),page=node('div',{class:'flow-page',style:'width:'+layout.width+'px;height:'+layout.height+'px'}),saved=()=>save(()=>({layouts:ws.layouts.map(item=>item.id===layout.id?clone(layout):clone(item))})).catch(fail);
      const controls=node('div',{class:'flow-layout-controls'},[choose,button('Add plot',()=>activePlot&&freezePlot(activePlot)),button('New page',()=>{const id=uid();ws.layouts.push({id,name:'Page '+(layouts.length+1),width:1100,height:800,panels:[]});layoutId=id;saved().then(showLayout);}),button('Arrange grid',()=>{layout.panels.forEach((p,i)=>{p.x=20+(i%2)*(layout.width-40)/2;p.y=25+Math.floor(i/2)*390;p.w=(layout.width-50)/2;p.h=375;});saved().then(showLayout);}),
        document.createTextNode('Width / height (px at 96 dpi)'),width,height,button('Resize page',()=>{layout.width=Number(width.value);layout.height=Number(height.value);saved().then(showLayout);}),button('Export page…',()=>exportPage(layout,page)),button('Save layout',saved)]);
      work.replaceChildren(controls,page);layout.panels.forEach(panel=>{const item=node('div',{class:'flow-layout-panel',style:`left:${panel.x}px;top:${panel.y}px;width:${panel.w}px;height:${panel.h}px`}),chart=node('div',{class:'flow-chart'});
        const bar=node('div',{class:'flow-panel-bar'},[node('strong',{text:panel.letter||''}),node('span',{text:'Snapshot',class:'flow-small'}),button('Edit',()=>{const x=input(panel.x,'Panel X','number'),y=input(panel.y,'Panel Y','number'),w=input(panel.w,'Panel width','number'),h=input(panel.h,'Panel height','number'),letter=input(panel.letter||'','Panel letter');dialog('Panel position',node('div',{},[field('X',x),field('Y',y),field('Width',w),field('Height',h),field('Letter',letter)]),[['Apply',async d=>{Object.assign(panel,{x:+x.value,y:+y.value,w:+w.value,h:+h.value,letter:letter.value});await saved();d.close();showLayout();},true]]);}),button('×',()=>{layout.panels=layout.panels.filter(p=>p.id!==panel.id);saved().then(showLayout);})]);
        bar.append(button('Style',()=>snapshotStyle(panel,saved)));
        item.append(bar,chart);page.append(item);const p={spec:clone(panel.snapshot.plot),result:panel.snapshot,chart};const {traces,layout:plotLayout}=plotData(p);plotLayout.dragmode='pan';
        p.saveAppearance=()=>{panel.snapshot.plot=clone(p.spec);return saved();};p.redrawAppearance=async()=>{const {traces,layout}=plotData(p);layout.dragmode='pan';await Plotly.react(chart,traces,layout,{responsive:true,displaylogo:false});bindLegendOrder(p);};
        Plotly.newPlot(chart,traces,plotLayout,{responsive:true,displaylogo:false}).then(()=>bindLegendOrder(p));
        const observer=new ResizeObserver(()=>{Plotly.Plots.resize(chart);panel.w=item.offsetWidth;panel.h=item.offsetHeight;});observer.observe(item);
        cornerResize(item,()=>Plotly.Plots.resize(chart),()=>{panel.w=item.offsetWidth;panel.h=item.offsetHeight;saved();});
        bar.onpointerdown=e=>{if(e.target.closest('button'))return;e.preventDefault();const start={x:e.clientX,y:e.clientY,left:panel.x,top:panel.y};bar.setPointerCapture(e.pointerId);bar.onpointermove=ev=>{panel.x=Math.max(0,Math.round((start.left+ev.clientX-start.x)/5)*5);panel.y=Math.max(0,Math.round((start.top+ev.clientY-start.y)/5)*5);item.style.left=panel.x+'px';item.style.top=panel.y+'px';};bar.onpointerup=()=>{bar.onpointermove=null;saved();};};
        item.oncontextmenu=e=>menu(e,[['Duplicate panel',()=>{layout.panels.push({...clone(panel),id:uid(),x:panel.x+20,y:panel.y+20});saved().then(showLayout);}],['Update from current plot',()=>{if(activePlot?.result){panel.snapshot=clone(activePlot.result);panel.snapshot.plot=clone(activePlot.spec);saved().then(showLayout);}}],['Bring to front',()=>{layout.panels=layout.panels.filter(p=>p!==panel).concat(panel);saved().then(showLayout);}],['Remove panel',()=>{layout.panels=layout.panels.filter(p=>p!==panel);saved().then(showLayout);}] ]);
      });}
    async function exportDialog(p){const format=select(['png','svg','pdf','tiff'],'png','Figure format'),name=input(p.spec.style.title||'flow-figure','Figure filename'),dpi=input(300,'Export DPI','number');dialog('Export figure',node('div',{},[field('Name',name),field('Format',format),field('DPI',dpi),help('Saved in the workspace output folder with plot data and a manifest. ScatterGL and 3D traces are rasterized; SVG/PDF axes and text stay vector where supported.')]),[['Export',async d=>{
        const svg=await Plotly.toImage(p.chart,{format:'svg',width:p.chart.offsetWidth,height:p.chart.offsetHeight});const png=format.value!=='svg'&&format.value!=='pdf'?await Plotly.toImage(p.chart,{format:'png',width:p.chart.offsetWidth,height:p.chart.offsetHeight,scale:Number(dpi.value)/96}):null;
        const result=await call('export',{name:name.value,format:format.value,dpi:Number(dpi.value),svg,png,manifest:{plot:p.spec,data:p.result}});setStatus('Saved '+result.file);download(result.download,name.value+'.'+format.value);notify('figure',{file:result.file,preview:png||svg});d.close();},true]]);}
    function exportPage(layout,page){const format=select(['svg','pdf','png','tiff'],'svg','Page format'),dpi=input(300,'Page DPI','number');dialog('Export page',node('div',{},[field('Format',format),field('DPI',dpi)]),[['Export',async d=>{const fragments=[];for(let i=0;i<layout.panels.length;i++){const p=layout.panels[i],chart=page.querySelectorAll('.flow-chart')[i];const url=await Plotly.toImage(chart,{format:'svg',width:p.w,height:p.h-25});
          fragments.push(`<image x="${p.x}" y="${p.y+25}" width="${p.w}" height="${p.h-25}" href="${url.replaceAll('&','&amp;').replaceAll('"','&quot;')}"/><text x="${p.x+4}" y="${p.y+20}" font-family="Arial" font-weight="bold" font-size="18">${(p.letter||'').replaceAll('&','&amp;').replaceAll('<','&lt;')}</text>`);}
        const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${layout.width}" height="${layout.height}"><rect width="100%" height="100%" fill="white"/>${fragments.join('')}</svg>`;
        const result=await call('export',{name:layout.name,format:format.value,dpi:Number(dpi.value),svg:'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg),manifest:{layout}});download(result.download,layout.name+'.'+format.value);setStatus('Saved '+result.file);d.close();
      },true]]);}
    function useContext(context){sourceContext=context;if(!live.checked||frozen)return;liveContext=context;refreshPlots();}
    live.onchange=()=>{ws.live={...ws.live,enabled:live.checked};save({live:ws.live}).catch(fail);if(live.checked){notify('source-request',{source:source.value});useContext(sourceContext);}else refreshPlots();};
    source.onchange=()=>{liveContext=null;sourceContext=null;save({live:{enabled:live.checked,source:source.value}}).catch(fail);notify('source-request',{source:source.value});refreshPlots();};
    mode.onchange=()=>pick(ws.active||'root');
    async function receive(event){const m=event.data||event;if(m.sender===sender)return;if(m.type==='appearance')studioAppearance=m.appearance;if(m.type==='context'&&m.source===source.value)useContext(m.context);
      if(m.type==='changed'&&m.revision!==ws?.revision&&!pending){try{await reload();}catch(e){fail(e);}}}
    if(bus)bus.onmessage=receive;
    const onKey=e=>{if(!root.isConnected||e.target.closest?.('input,textarea,select,[contenteditable],dialog')||root.querySelector('dialog[open]'))return;
      if(e.key==='Escape'&&activePlot?.cancelGateDrag){activePlot.cancelGateDrag(e);return;}
      if(e.key==='Delete'){const selected=[...marked].filter(id=>id!=='root');if(e.target.closest?.('.flow-tree')){if(selected.length){e.preventDefault();deletePopulations(selected);}}else if(selected.length>1){e.preventDefault();deletePopulations(selected);}else if(activePlot?.selectedGate){e.preventDefault();deleteGate(population(activePlot.selectedGate));}else if(selected.length){e.preventDefault();deletePopulations(selected);}}
      else if(e.key==='Escape'){root.querySelector('.flow-menu')?.remove();if(activePlot){activePlot.selectedGate=null;activePlot.spec.tool='pan';activePlot.controls.gateTool.value='pan';renderPlot(activePlot);}}else if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();command(e.shiftKey?'redo':'undo');}};
    document.addEventListener('keydown',onKey);
    let starting=false;const retry=button('Retry',()=>start());retry.hidden=true;tool.append(retry);
    async function start(){if(starting)return;starting=true;retry.hidden=true;
      const controls=[...tool.querySelectorAll('button,input,select')].filter(e=>e!==retry);controls.forEach(e=>e.disabled=true);
      try{setStatus('Opening workspace…');const ready=loadPlotly(hooks.plotly);ready.catch(()=>{});await reload();controls.forEach(e=>e.disabled=!ws);await ready;notify('source-request',{source:source.value});}
      catch(e){fail(e);loading.textContent=e.message;retry.hidden=false;}
      finally{starting=false;controls.forEach(e=>e.disabled=!ws);}}
    await start();
    const timer=setInterval(async()=>{if(disposed||!root.isConnected||!ws||pending||document.hidden)return;try{const found=await get(route+'?status=1');if(found.revision!==ws?.revision||found.id!==ws?.id)await reload();}catch(e){setStatus('Disconnected · '+e.message,true);}},4000);
    return {receive,reload,dispose:()=>{disposed=true;statsView?.dispose();clearInterval(timer);bus?.close();sidebarObserver.disconnect();document.removeEventListener('keydown',onKey);plots.forEach(p=>p.observer.disconnect());}};
  }
  window.SPACEMAPFlow={mount:async(root,hooks)=>{try{return await mount(root,hooks);}catch(error){
    root.replaceChildren(node('div',{class:'flow-empty'},[node('strong',{text:'Flow gating'}),node('p',{role:'alert',text:'Could not open Flow gating: '+error.message}),button('Retry',()=>window.SPACEMAPFlow.mount(root,hooks))]));
    console.error('Flow gating startup failed',error);
  }}};
})();
