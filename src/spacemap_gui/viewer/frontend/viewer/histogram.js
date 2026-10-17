/* SPACEMAP viewer, part 6 of 12: a marker's histogram, to set its range (as in QuPath).
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

// ---------------------------------------------------------------- histogram (as QuPath's Brightness & Contrast)
// Where to measure, besides the whole image, grouped as Layers shows them: the open image, then each imported file;
// in each, every annotation name (all its polygons together, so a tissue in many pieces is one region), a layer's
// whole set when it holds several names, and the polygons picked on the image. Values are JSON for regionParams.
// Returns [value, text] pairs, with [null, group label] starting each group (see regionSelect).
function regionOptions(imageToken) {
  const options = [];
  const groups = new Map();
  shapeLayers().filter((layer) => !imageToken || layer.image_token === imageToken).forEach((layer) => {
    const polygons = (layer.features || []).filter((f) => polygonsOf(f.geometry).length);
    if (!polygons.length || layer.training_part) return;  // training examples are not areas to measure in
    const source = annotationSource(layer);
    if (!groups.has(source)) groups.set(source, []);
    const names = new Map();
    polygons.forEach((f) => { const name = annotationName(layer, f); names.set(name, (names.get(name) || 0) + 1); });
    const list = groups.get(source);
    if (names.size > 1) list.push([JSON.stringify({ shape: layer.token }), layer.name + " (all " + polygons.length + " polygons)"]);
    names.forEach((count, name) => list.push([JSON.stringify({ shape: layer.token, group: name }),
      name + (count > 1 ? " (" + count + " polygons)" : "") + (names.size === 1 && layer.name !== name ? " · " + layer.name : "")]));
  });
  const image = state.layers.find((l) => l.type === "image" && (!imageToken || l.token === imageToken));
  const imageName = String(image ? image.name : "This image").replace(/(\.ome)?\.(tiff?|qptiff|svs)$/i, "");
  groups.forEach((list, source) => { if (list.length) options.push([null, source || imageName], ...list); });
  if (options.length) options.push([null, "Picked on the image"], [JSON.stringify({ selected: true }), "Selected polygons"]);
  return options;
}

// A select from regionOptions' pairs: [null, label] pairs start an optgroup.
function regionSelect(first, options, value, onChange) {
  const input = select(first, value, onChange);
  let group = input;
  options.forEach(([key, text]) => {
    if (key === null) { group = el("optgroup", { label: text }); input.appendChild(group); return; }
    group.appendChild(el("option", { value: key, text }));
  });
  input.value = value;
  return input;
}

// The polygons picked on the image now (click one, Ctrl-click to add more), in one drawn layer.
function pickedShapes() {
  const picks = state.picks;
  const layer = picks && picks.ids && picks.ids.size ? picks.layer : state.selected && state.selected.layer;
  if (!layer || layer.type !== "shapes") return null;
  const ids = new Set(picks && picks.layer === layer ? picks.ids : []);
  if (state.selected && state.selected.layer === layer) ids.add(state.selected.id);
  return ids.size ? { layer: layer, ids: ids } : null;
}

// A region choice as query parameters; "Selected polygons" is read when it is used.
function regionParams(value, imageToken) {
  if (!value) return {};
  let scope = {};
  try { scope = JSON.parse(value); } catch (e) { throw new Error("The saved measurement region is invalid. Choose the area again."); }
  if (scope.group !== undefined) {
    // A name: its polygons in that layer now (the same rule as Layers), so renamed pieces are counted correctly.
    const layer = shapeLayers().find((l) => l.token === scope.shape);
    const ids = layer ? (layer.features || []).filter((f) => annotationName(layer, f) === scope.group).map((f) => f.properties.id) : [];
    if (!ids.length) throw new Error("No polygons are named " + scope.group + " any more. Choose the area again.");
    return { shape: scope.shape, sid: ids.join(",") };
  }
  if (!scope.selected) return scope;
  const picked = pickedShapes();
  if (!picked || picked.layer.image_token !== imageToken) {
    throw new Error("Select polygons on the image first: click one, Ctrl-click to add more.");
  }
  return { shape: picked.layer.token, sid: Array.from(picked.ids).join(",") };
}

function number(value) {
  const v = Number(value);
  return Number.isInteger(v) || Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2);
}

function showHistogram(layer) {
  const holder = layer.histogramHolder;
  if (!holder) return;
  holder.innerHTML = "";
  Object.entries(layer.inputs || {}).forEach(([index, parts]) => parts.row.classList.toggle("picked", Number(index) === layer.histogramFor));
  const channel = layer.channels.find((c) => c.index === layer.histogramFor);
  const parts = channel && (layer.inputs || {})[channel.index];
  if (!channel || !parts) {
    holder.appendChild(el("div", { class: "row muted" }, [el("span", { text: "Histogram" }),
      help("Click a marker's name to see its histogram, set its range there, or rename it.")]));
    return;
  }
  holder.appendChild(el("div", { class: "section-title", text: "Histogram: " + (channel.label || channel.name) }));
  const cells = liveForImage(layer), marker = linkedMarker(layer, channel);
  if (cells && marker) holder.appendChild(resolveHistogramBox(cells, marker, layer, channel));
  else holder.appendChild(histogramBox(layer, channel, parts.low, parts.high));
}

function resolveHistogramBox(cells, marker, image, channel) {
  const box = el("div", { class: "hist" });
  let names = channel.histogramSteps || ["Arcsinh"], histogram = null, dragging = false, requestId = 0, converting = 0;
  const raw = el("input", { type: "number", step: "any", value: marker.threshold });
  const transformed = el("input", { type: "number", step: "any" });
  const steps = el("div", { class: "hint" });
  const status = el("div", { class: "hint", role: "status" });
  const canvas = el("canvas", { width: 300, height: 84, title: "Drag the threshold; values are converted back to raw" });
  const choice = select(["Arcsinh", "Log10", "Z1", "CDF", "NLog"].map((s) => [s, s]), "Arcsinh", () => {});
  const reset = () => { requestId++; channel.histogramSteps = names.slice(); histogram = null; load(); };
  box.appendChild(el("div", { class: "hint", text: "Cell measurements. These options change how the histogram looks. RESOLVE keeps the original values. Combinations that lose precision cannot be used." }));
  box.appendChild(el("div", { class: "row" }, [choice,
    el("button", { class: "btn small", text: "Add step", onclick: () => { if (names.length < 8) { names.push(choice.value); reset(); } } }),
    el("button", { class: "btn small", text: "Remove last", onclick: () => { names.pop(); reset(); } }),
    el("button", { class: "btn small", text: "Raw", onclick: () => { names = []; reset(); } })]));
  box.appendChild(steps);
  box.appendChild(canvas);
  box.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Original value" }), raw]));
  box.appendChild(el("div", { class: "row" }, [el("label", { class: "lbl", text: "Histogram value" }), transformed]));
  const auto = el("button", { class: "btn", text: "Auto from these cells", onclick: async () => {
    const id = ++requestId; auto.disabled = true;
    try {
      const result = await getJSON(query() + "&auto=" + encodeURIComponent(image.autoMethod || "Triangle") + "&" + new URLSearchParams(thresholdOptions(image)));
      if (id !== requestId) return;
      setRaw(result.raw); transformed.value = result.value; draw();
      status.textContent = "Auto used " + (image.autoMethod || "Triangle") + " on the cell measurements shown here.";
    } catch (error) { status.textContent = error.message; }
    finally { auto.disabled = false; }
  } });
  box.appendChild(auto);
  box.appendChild(status);
  const query = () => "/resolve/" + cells.resolve + "/histogram?marker=" + encodeURIComponent(marker.marker) +
    "&transforms=" + encodeURIComponent(names.join(","));
  function draw() {
    const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!histogram) return;
    const maximum = Math.max(...histogram.counts, 1);
    histogram.counts.forEach((count, i) => {
      const height = Math.log1p(count) / Math.log1p(maximum) * (h - 2);
      ctx.fillStyle = channel.color;
      ctx.fillRect(i * w / histogram.counts.length, h - height, Math.max(1, w / histogram.counts.length), height);
    });
    const x = (Number(transformed.value) - histogram.axis[0]) / (histogram.axis[1] - histogram.axis[0]) * w;
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(Math.max(0, Math.min(w, x)), 0); ctx.lineTo(Math.max(0, Math.min(w, x)), h); ctx.stroke();
  }
  function setRaw(value) {
    if (!Number.isFinite(value)) { status.textContent = "Enter a valid number."; return; }
    marker.threshold = value; marker.source = "changed"; raw.value = value;
    if (channel.resolveLinked !== false) setMarkerRange(image, channel, value, markerHigh(channel, value));
    sendLive(cells, { [marker.marker]: value });
  }
  async function convert(value) {
    const id = ++requestId;
    converting++;
    try {
      const result = await getJSON(query() + "&value=" + encodeURIComponent(value));
      if (id !== requestId || cells.live == null || !state.layers.includes(image)) return;
      setRaw(result.raw); transformed.value = value; draw();
    } catch (error) { if (id === requestId) status.textContent = error.message; }
    finally { converting--; }
  }
  async function load() {
    const id = ++requestId;
    steps.textContent = names.length ? names.join(" → ") : "Raw";
    status.textContent = "Reading the cell histogram...";
    try {
      const result = await getJSON(query());
      if (id !== requestId) return;
      histogram = result; transformed.value = result.transformed_threshold; raw.value = marker.threshold;
      status.textContent = result.total.toLocaleString() + " cells; " + result.positive.toLocaleString() + " above the threshold.";
      steps.title = JSON.stringify(result.stages);
      draw();
    } catch (error) { if (id === requestId) { histogram = null; draw(); status.textContent = error.message; } }
  }
  raw.addEventListener("change", () => { requestId++; setRaw(Number(raw.value)); });
  transformed.addEventListener("change", () => convert(Number(transformed.value)));
  const move = (event) => {
    if (!histogram) return;
    const rect = canvas.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const value = histogram.axis[0] + fraction * (histogram.axis[1] - histogram.axis[0]);
    transformed.value = value; draw();
    // At most one conversion every 80 ms; discard replies to earlier drags.
    requestId++;
    debounce("histogram-" + image.token + "-" + channel.index, () => convert(value), 80);
  };
  canvas.addEventListener("pointerdown", (event) => { if (histogram) { dragging = true; canvas.setPointerCapture(event.pointerId); move(event); } });
  canvas.addEventListener("pointermove", (event) => { if (dragging) move(event); });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
  image.redrawHistogram = () => { if (box.isConnected && !dragging) raw.value = marker.threshold; };
  image.refreshConfirmedHistogram = () => { if (box.isConnected && !dragging && !converting) load(); };
  load();
  return box;
}

// Both pixel histogram views use the same axis choices and raw-unit data.
function histogramView(layer) {
  return layer.histogramView ||= {scale:'Linear', y:'log', cofactor:5};
}
function histogramProjection(value, scale, cofactor, inverse=false) {
  if(scale==='Log10')return inverse?10**value:(value>0?Math.log10(value):-Infinity);
  if(scale==='Arcsinh')return inverse?Math.sinh(value)*cofactor:Math.asinh(value/cofactor);
  return value;
}
function histogramDisplayControls(layer, prefix, changed) {
  const settings=histogramView(layer),box=el('div',{class:'histogram-display-controls'});
  let previous=JSON.stringify([settings,!!layer.autoNative]);
  const notify=()=>{
    remember(layer);
    for(const observer of layer.histogramObservers||[]){
      if(observer.box.isConnected)observer.sync();else layer.histogramObservers.delete(observer);
    }
  };
  const scale=select([['Linear','Linear (raw)'],['Log10','Log10'],['Arcsinh','Arcsinh']],settings.scale,v=>{settings.scale=v;notify();});
  scale.setAttribute('aria-label',prefix+' display scale');
  const y=select([['log','Pixel count · log'],['count','Pixel count · linear'],['fraction','Pixel fraction']],settings.y,v=>{settings.y=v;notify();});
  y.setAttribute('aria-label',prefix+' vertical scale');
  const cofactor=el('input',{type:'number',value:settings.cofactor,min:.001,max:1000000,step:'any','aria-label':prefix+' arcsinh cofactor',required:true});
  const cofactorRow=field('Cofactor',cofactor);cofactorRow.hidden=settings.scale!=='Arcsinh';
  cofactor.onchange=()=>{if(!cofactor.checkValidity()){cofactor.value=settings.cofactor;return;}settings.cofactor=Number(cofactor.value);notify();};
  const quality=select([['preview','Preview resolution'],['native','Full resolution']],layer.autoNative?'native':'preview',v=>{layer.autoNative=v==='native';syncThresholdParameterWidgets(layer);notify();});
  quality.setAttribute('aria-label',prefix+' sampling resolution');quality.dataset.thresholdToken=layer.token;quality.dataset.thresholdParameter='native';
  box.append(field('X axis',scale),field('Y axis',y),cofactorRow,field('Pixels',quality));
  (layer.histogramObservers||=new Set()).add({box,sync:()=>{scale.value=settings.scale;y.value=settings.y;cofactor.value=settings.cofactor;cofactorRow.hidden=settings.scale!=='Arcsinh';quality.value=layer.autoNative?'native':'preview';const current=JSON.stringify([settings,!!layer.autoNative]);if(current!==previous){previous=current;changed();}}});
  return box;
}
function histogramViewSummary(layer) {
  const v=histogramView(layer);
  return (v.scale==='Linear'?'Linear X':v.scale+(v.scale==='Arcsinh'?' / '+v.cofactor:'')+' X')+' · '+({log:'log counts',count:'linear counts',fraction:'pixel fraction'}[v.y])+' · '+(layer.autoNative?'full resolution':'preview');
}
function histogramBox(layer, channel, lowInput, highInput) {
  const box=el('div',{class:'hist'}),regions=regionOptions(layer.token);
  if(channel.region===undefined)channel.region=layer.autoScope||'';
  const population=regionSelect([['','Whole image']],regions,channel.region,v=>{channel.region=v;load();});
  population.setAttribute('aria-label','Image histogram population');
  const add=(value,label)=>{const existing=Array.from(population.options).find(o=>o.value===value);if(existing)existing.textContent=label;else population.append(el('option',{value,text:label}));};
  if(channel.region&&!Array.from(population.options).some(o=>o.value===channel.region))add(channel.region,'Current annotation selection');
  for(const annotation of thresholdAnnotations(layer)){
    add(JSON.stringify({annotation:annotation.token}),annotation.name+': all areas');
    annotationLegend(annotation).then(legend=>{for(const item of legend.items)add(JSON.stringify({annotation:annotation.token,class:item.value}),annotation.name+': '+item.value);population.value=channel.region;}).catch(error=>{stats.textContent='Annotation choices: '+error.message;});
  }
  population.value=channel.region;
  box.append(field('Measure in',population));
  const options=el('details',{class:'histogram-options'},[el('summary',{text:'Histogram display'})]);
  options.append(histogramDisplayControls(layer,'Image histogram',()=>load()));box.append(options);
  const canvas=el('canvas',{width:400,height:110}),caption=el('div',{class:'hint'}),stats=el('div',{class:'hint',text:'Measuring…'});
  box.append(canvas,caption,stats);
  box.append(el('div',{class:'hint',text:'Drag the white lines to set the black and full-colour levels. Upper brightness changes only the display.'}));
  const rename=el('input',{type:'text',value:channel.label||channel.name});
  rename.onchange=()=>{channel.label=rename.value.trim()||channel.name;remember(layer);};box.append(field('Name',rename));
  let histogram=null,dragging=null,request=0;
  const xOf=v=>Math.max(0,Math.min(canvas.width,(histogramProjection(v,histogram.scale,histogram.cofactor)-histogram.axis[0])/(histogram.axis[1]-histogram.axis[0])*canvas.width));
  function draw(){
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.fillStyle='#0f1115';ctx.fillRect(0,0,w,h);if(!histogram)return;
    const mode=histogramView(layer).y, counts=histogram.counts, top=Math.max(1,...counts);
    ctx.fillStyle=channel.color;
    counts.forEach((n,i)=>{const height=(mode==='log'?Math.log1p(n)/Math.log1p(top):n/top)*(h-6);ctx.fillRect(i*w/counts.length,h-height,Math.max(1,w/counts.length),height);});
    const lowX=xOf(channel.low),highX=xOf(channel.high);
    ctx.fillStyle='rgba(0,0,0,.55)';ctx.fillRect(0,0,lowX,h);ctx.fillRect(highX,0,w-highX,h);
    ctx.strokeStyle='#fff';ctx.lineWidth=2;[lowX,highX].forEach(x=>{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();});
    caption.textContent=histogramViewSummary(layer)+' · full intensity range';
  }
  function move(event){
    const rect=canvas.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
    const value=histogramProjection(histogram.axis[0]+fraction*(histogram.axis[1]-histogram.axis[0]),histogram.scale,histogram.cofactor,true);
    const low=dragging==='low'?Math.min(value,channel.high-1):channel.low,high=dragging==='high'?Math.max(value,channel.low+1):channel.high;
    setMarkerRange(layer,channel,low,high);lowInput.value=channel.low;highInput.value=channel.high;draw();
  }
  layer.redrawHistogram=draw;
  canvas.onpointerdown=e=>{if(!histogram)return;const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*canvas.width/rect.width;dragging=Math.abs(x-xOf(channel.low))<=Math.abs(x-xOf(channel.high))?'low':'high';canvas.setPointerCapture(e.pointerId);move(e);};
  canvas.onpointermove=e=>{if(dragging)move(e);};
  canvas.onpointerup=canvas.onpointercancel=()=>{dragging=null;};
  async function load(){
    const id=++request;stats.textContent='Measuring…';
    try{
      const v=histogramView(layer),result=await getJSON('/image/'+layer.token+'/histogram?'+new URLSearchParams({ch:channel.index,scale:v.scale,cofactor:v.cofactor,native:layer.autoNative?1:0,...regionParams(channel.region,layer.token)}));
      if(id!==request)return;histogram=result;
      if(channel.low===undefined)channel.low=result.suggested[0];
      if(channel.high===undefined)channel.high=Math.max(result.suggested[1],channel.low+1);
      lowInput.value=channel.low;highInput.value=channel.high;
      stats.textContent=number(result.min)+'–'+number(result.max)+' raw · median '+number(result.median)+' · '+result.pixels.toLocaleString()+' pixels'+(result.downsample>1?' at 1/'+number(result.downsample)+' resolution':'')+(result.omitted?' · '+result.omitted.toLocaleString()+' nonpositive pixels omitted on Log10':'');
      draw();
    }catch(error){if(id===request)stats.textContent=error.message;}
  }
  load();return box;
}

// Background's comparison figure. Estimation is server-side in raw units;
// scale changes only project the histogram and cut-off coordinates.
function clearThresholdPreview() {
  for (const layer of state.layers) if (layer.thresholdPreview) {
    delete layer.thresholdPreview; refreshLayer(layer);
  }
}

function renderThresholdBenchmark(holder) {
  let image=null,controller=null,generation=0,disposed=false,timer;
  const sync=()=>{
    if(disposed)return;
    const active=state.layers.find(l=>l.type==='image'&&!l.info?.rgb&&l.channels);
    if(active===image&&holder.childNodes.length)return;
    controller?.abort();clearTimeout(timer);generation++;image=active;holder.replaceChildren();
    if(!image){holder.append(el('p',{class:'muted',text:'Open a marker image to compare thresholds.'}));return;}
    build(image);
  };
  function build(layer){
    const format=value=>Number(value).toLocaleString(undefined,{maximumSignificantDigits:6});
    const methods=layer.info.auto_methods||['Triangle'];
    const colours=['#56b4e9','#e69f00','#009e73','#cc79a7','#f0e442','#0072b2','#d55e00','#aa88ee','#8dd3c7','#ffffff'];
    const methodsChosen=new Set(['Triangle','Otsu','Li','Yen']);
    const picked=new Set(layer.channels.filter(c=>c.visible).slice(0,1).map(c=>c.index));
    if(!picked.size)picked.add(layer.channels[0].index);
    let results=[],originals=null,originalOptions=null,lastApplied=null,selectedChannel=String([...picked][0]);
    const note=el('div',{class:'hint threshold-status',role:'status',text:'Compare thresholds, then click a cutoff to inspect it.'});
    const parameterRows=new Map(),pickers=[];
    function picker(label,items,chosen,searchable){
      const summary=el('summary'),details=el('details',{class:'threshold-picker','aria-label':label+' selector'},[summary]);pickers.push(details);
      const grid=el('div',{class:'threshold-checks','aria-label':'Benchmark '+label.toLowerCase()});
      const update=()=>{const names=items.filter(i=>chosen.has(i.value)).map(i=>i.name);summary.textContent=names.length===1?names[0]:names.length+' selected'+(names.length?' · '+names.slice(0,2).join(', ')+(names.length>2?'…':''):'');summary.title=names.join(', ');};
      if(searchable){const search=el('input',{type:'search',placeholder:'Find a marker…','aria-label':'Find benchmark channel'});search.oninput=()=>{for(const row of grid.children)row.hidden=!row.textContent.toLowerCase().includes(search.value.toLowerCase());};details.append(search);}
      for(const item of items){
        const check=el('input',{type:'checkbox','aria-label':(searchable?'Benchmark ':'Compare ')+item.name});check.checked=chosen.has(item.value);
        check.onchange=()=>{if(check.checked&&searchable&&chosen.size>=8&&!chosen.has(item.value)){check.checked=false;note.textContent='Compare up to 8 channels at once.';return;}if(check.checked)chosen.add(item.value);else chosen.delete(item.value);update();for(const [method,row]of parameterRows)row.hidden=!methodsChosen.has(method);};
        grid.append(el('label',{title:item.name},[check,el('span',{text:item.name})]));
      }
      details.append(grid,el('button',{class:'btn small',text:'Done',onclick:()=>{details.open=false;summary.focus();}}));
      details.addEventListener('toggle',()=>{if(details.open)for(const other of pickers)if(other!==details)other.open=false;});
      details.onkeydown=e=>{if(e.key==='Escape'){details.open=false;summary.focus();}};update();return field(label,details);
    }
    const selections=el('div',{class:'threshold-selections'},[
      picker('Channels',layer.channels.map(c=>({name:c.name,value:c.index})),picked,true),
      picker('Methods',methods.map(name=>({name,value:name})),methodsChosen,false)
    ]);
    const parameters=el('div',{class:'threshold-parameters'});
    for(const [key,label,method,initial]of [['percentile','Percentile','Percentile',95],['manual','Manual cutoff','Manual',0]]){
      const input=el('input',{type:'number',step:'any',value:layer.autoParameters?.[key]??initial,'aria-label':'Benchmark '+label,required:true});
      if(key==='percentile'){input.min=0;input.max=100;}
      input.dataset.thresholdToken=layer.token;input.dataset.thresholdParameter=key;
      input.onchange=()=>{if(!input.checkValidity()||!Number.isFinite(Number(input.value))){input.value=layer.autoParameters?.[key]??initial;note.textContent='Enter a finite cutoff, or a percentile from 0 to 100.';return;}(layer.autoParameters||={})[key]=Number(input.value);remember(layer);syncThresholdParameterWidgets(layer);};
      const row=field(label,input);row.hidden=!methodsChosen.has(method);parameterRows.set(method,row);parameters.append(row);
    }
    const scope=regionSelect([['','Whole image']],regionOptions(layer.token),layer.autoScope||'',v=>{layer.autoScope=v;remember(layer);});scope.setAttribute('aria-label','Benchmark annotation');
    const addScope=(value,label)=>{if(!Array.from(scope.options).some(o=>o.value===value))scope.append(el('option',{value,text:label}));};
    for(const annotation of thresholdAnnotations(layer)){
      addScope(JSON.stringify({annotation:annotation.token}),annotation.name+': all areas');
      annotationLegend(annotation).then(legend=>{if(disposed||image!==layer)return;for(const item of legend.items)addScope(JSON.stringify({annotation:annotation.token,class:item.value}),annotation.name+': '+item.value);scope.value=layer.autoScope||'';}).catch(error=>{note.textContent='Annotation classes: '+error.message;});
    }
    if(layer.autoScope)addScope(layer.autoScope,'Current annotation selection');scope.value=layer.autoScope||'';
    const viewSummary=el('summary',{text:'Plot settings · '+histogramViewSummary(layer)});
    const options=el('details',{class:'threshold-options'},[viewSummary,histogramDisplayControls(layer,'Benchmark',()=>redraw()),
      el('p',{class:'hint',text:'These display settings also apply to Image Layer histograms. Log10 omits nonpositive pixels from the plot only. Full range; no bright-tail clipping.'})]);
    const inspection=select([['overlay','Threshold overlay'],['mask','Binary mask'],['off','Normal image']],'overlay',v=>{
      if(v==='off'){delete layer.thresholdPreview;refreshLayer(layer);}else if(lastApplied){layer.thresholdPreview={...lastApplied,mode:v};refreshLayer(layer);}
    });inspection.setAttribute('aria-label','Threshold inspection display');
    const plots=el('div',{class:'threshold-plots'});
    const resultPick=select([],selectedChannel,v=>{selectedChannel=v;showResult();});resultPick.setAttribute('aria-label','View compared channel');
    const resultRow=field('View channel',resultPick);resultRow.hidden=true;
    const undo=el('button',{class:'btn small',text:'Restore previous levels',disabled:true,onclick:()=>{
      if(originals)for(const row of originals){row.channel.low=row.low;row.channel.high=row.high;if(Number.isFinite(row.low))setMarkerRange(layer,row.channel,row.low,row.high);}
      if(originalOptions){layer.autoMethod=originalOptions.method;layer.autoParameters=originalOptions.parameters;layer.autoScope=originalOptions.scope;}
      originals=null;originalOptions=null;lastApplied=null;delete layer.thresholdPreview;remember(layer);refreshLayer(layer);buildPanel();undo.disabled=true;note.textContent='Previous levels restored.';plots.querySelectorAll('[aria-pressed]').forEach(b=>b.setAttribute('aria-pressed','false'));
    }});
    const clear=el('button',{class:'btn small',text:'Clear preview',onclick:()=>{delete layer.thresholdPreview;refreshLayer(layer);inspection.value='off';}});
    const run=el('button',{class:'btn small primary',text:'Compare thresholds',onclick:()=>evaluate()});
    const cancel=el('button',{class:'btn small',text:'Cancel',onclick:()=>{controller?.abort();generation++;run.disabled=false;cancel.hidden=true;note.textContent='Comparison cancelled.';}});cancel.hidden=true;
    const actions=el('details',{class:'threshold-actions'},[el('summary',{text:'Preview options'}),field('Inspect as',inspection),el('div',{class:'row wrap'},[undo,clear])]);
    holder.append(el('div',{class:'threshold-heading'},[el('strong',{text:'Threshold benchmark'}),help('Compare raw pixel cutoffs. Click a line or table value to preview it. Solid = whole image; dashed = annotation. Export thresholds in Background to reuse the chosen lower levels in RESOLVE.')]),
      selections,parameters,field('Compare in',scope,'Compare the whole image with a selected annotation category. Both populations use shared intensity bins.'),options,
      el('div',{class:'row wrap'},[run,cancel]),note,resultRow,plots,actions);
    function redraw(){
      viewSummary.textContent='Plot settings · '+histogramViewSummary(layer);
      // Y changes are client-side; scale, cofactor and sampling reuse backend pixel caches.
      if(results.length&&results.every(({result})=>result.scale===histogramView(layer).scale&&result.cofactor===histogramView(layer).cofactor&&result.native===!!layer.autoNative)&&!run.disabled){showResult();return;}
      if(results.length||run.disabled){clearTimeout(timer);timer=setTimeout(evaluate,200);}
    }
    async function evaluate(){
      if(disposed||image!==layer)return;
      if(!picked.size||!methodsChosen.size){note.textContent='Choose 1–8 channels and at least one method.';return;}
      for(const picker of pickers)picker.open=false;
      clearTimeout(timer);controller?.abort();controller=new AbortController();const signal=controller.signal,id=++generation;
      run.disabled=true;cancel.hidden=false;results=[];plots.replaceChildren();resultPick.replaceChildren();resultRow.hidden=true;
      const chosen=[...picked],view=histogramView(layer),scopeValue=scope.value,scopeLabel=scope.selectedOptions[0]?.textContent||'Whole image';
      const query={methods:[...methodsChosen].join(','),scale:view.scale,cofactor:view.cofactor,...thresholdOptions(layer)};
      try{
        Object.assign(query,regionParams(scopeValue,layer.token));
        for(const ch of chosen){
          const channel=layer.channels.find(c=>c.index===ch);note.textContent='Comparing '+channel.name+'… '+(results.length+1)+' / '+chosen.length;
          const response=await fetch(api('/image/'+layer.token+'/threshold_benchmark?'+new URLSearchParams({...query,ch})),{signal,credentials:'same-origin'});
          const result=await response.json();if(!response.ok)throw Error(result.error||response.status);
          if(id!==generation||disposed||image!==layer)return;
          results.push({channel,result,scopeValue,scopeLabel});resultPick.append(el('option',{value:String(ch),text:channel.name}));
        }
        if(!results.some(r=>String(r.channel.index)===selectedChannel))selectedChannel=String(results[0].channel.index);
        resultPick.value=selectedChannel;resultRow.hidden=results.length<2;showResult();
        note.textContent='Compared '+results.length+' channel'+(results.length===1?'':'s')+'. Click a cutoff below to apply it.';
      }catch(error){if(id===generation)note.textContent=error.name==='AbortError'?'Comparison cancelled.':error.message;}
      finally{if(id===generation){run.disabled=false;cancel.hidden=true;}}
    }
    function apply(entry,gate,colour,population){
      const {channel,scopeValue,result}=entry;
      if(disposed||image!==layer||!state.layers.includes(layer)||!Number.isFinite(gate.threshold))return;
      if(!originals){originals=layer.channels.map(channel=>({channel,low:channel.low,high:channel.high}));originalOptions={method:layer.autoMethod,parameters:{...layer.autoParameters},scope:layer.autoScope};}
      layer.autoMethod=gate.method;layer.autoParameters={...gate.parameters};layer.autoScope=population==='Whole image'?'':scopeValue;
      setMarkerRange(layer,channel,gate.threshold,markerHigh(channel,gate.threshold));
      lastApplied={channel:channel.index,threshold:gate.threshold,colour:colour.replace('#',''),method:gate.method,population};
      if(inspection.value!=='off')layer.thresholdPreview={...lastApplied,mode:inspection.value};
      refreshLayer(layer);buildPanel();undo.disabled=false;
      note.textContent='Applied '+channel.name+' · '+gate.method+' · '+population+' · '+format(gate.threshold)+' raw.';
      showResult();
    }
    function showResult(){
      plots.replaceChildren();const entry=results.find(r=>String(r.channel.index)===selectedChannel);if(!entry)return;
      const {channel,result,scopeLabel}=entry;
      const wrap=el('section',{class:'threshold-chart'}),ns='http://www.w3.org/2000/svg';
      const svg=(tag,attrs)=>{const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs||{})){if(k==='text')e.textContent=v;else e.setAttribute(k,v);}return e;};
      const plot=svg('svg',{viewBox:'0 0 620 245',role:'group','aria-label':channel.name+' intensity distribution'});
      const [low,high]=result.axis,x=v=>58+(v-low)/(high-low)*548,mode=histogramView(layer).y;
      const amount=(n,p)=>mode==='fraction'?n/Math.max(1,p.pixels-p.omitted):n;
      const maximum=Math.max(1e-12,...result.populations.flatMap(p=>p.counts.map(n=>amount(n,p))));
      const y=value=>201-170*(mode==='log'?Math.log1p(value)/Math.log1p(maximum):value/maximum);
      const tick=value=>mode==='fraction'?(value*100).toPrecision(2)+'%':value>=1000000?(value/1000000).toPrecision(2)+'M':value>=1000?(value/1000).toPrecision(2)+'k':Math.round(value);
      for(let i=0;i<=4;i++){const value=low+(high-low)*i/4;plot.append(svg('text',{x:x(value),y:221,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'currentColor','font-size':11,text:Number(value.toPrecision(4))}));}
      plot.append(svg('path',{d:'M58 27V201H606',stroke:'#8c94a4',fill:'none'}),svg('text',{x:8,y:17,fill:'currentColor','font-size':11,text:mode==='fraction'?'Pixel fraction':mode==='log'?'Pixel count (log)':'Pixel count'}),svg('text',{x:332,y:242,'text-anchor':'middle',fill:'currentColor','font-size':11,text:result.scale+(result.scale==='Arcsinh'?' / '+result.cofactor:result.scale==='Linear'?' intensity (raw)':' intensity')}));
      for(const fraction of [0,.5,1]){const value=mode==='log'?Math.expm1(fraction*Math.log1p(maximum)):fraction*maximum;plot.append(svg('text',{x:52,y:y(value)+4,'text-anchor':'end',fill:'currentColor','font-size':10,text:tick(value)}));}
      const curveLegend=el('div',{class:'threshold-populations'});
      result.populations.forEach((population,index)=>{
        let d='M58,201';population.counts.forEach((n,i)=>{d+='L'+x(result.edges[i])+','+y(amount(n,population))+'H'+x(result.edges[i+1]);});d+='V201Z';
        const colour=index?'#d5c5f8':channel.color||'#aaa';
        plot.append(svg('path',{d,stroke:colour,fill:colour,'fill-opacity':index?.05:.15,'stroke-width':1,'stroke-dasharray':index?'5 3':''}));
        const item=el('span',{text:index?'Annotation':'Whole image',title:index?scopeLabel:'Whole image'});item.style.borderBottom='2px '+(index?'dashed':'solid')+' '+colour;curveLegend.append(item);
      });
      const table=el('table',{class:'threshold-results'}),head=el('tr',{},[el('th',{text:'Method'})]);
      for(const p of result.populations)head.append(el('th',{text:p.name+' cutoff',title:p.name==='Annotation'?scopeLabel:'Whole image'}));table.append(el('thead',{},[head]));
      const body=el('tbody');table.append(body);
      for(const method of result.populations[0].thresholds.map(g=>g.method)){
        const colour=colours[methods.indexOf(method)%colours.length],label=el('th',{scope:'row'},[el('span',{class:'threshold-method-swatch'}),el('span',{text:method})]);label.firstChild.style.background=colour;
        const row=el('tr',{},[label]);
        result.populations.forEach((population,index)=>{
          const gate=population.thresholds.find(g=>g.method===method),active=lastApplied?.channel===channel.index&&lastApplied?.method===method&&lastApplied?.population===population.name;
          const button=el('button',{class:'btn small threshold-cutoff','aria-pressed':String(active),'aria-label':channel.name+' '+population.name+' '+method+' cutoff '+(gate.error?'unavailable':format(gate.threshold))});
          button.dataset.method=method;button.dataset.channel=channel.index;button.dataset.population=population.name;button.disabled=!!gate.error;
          button.append(el('span',{text:gate.error?'Unavailable':format(gate.threshold)}));
          if(!gate.error)button.append(el('small',{class:'muted',text:(100*gate.positive_fraction).toFixed(1)+'% above'}));
          button.title=gate.error||('Raw cutoff '+gate.threshold+'. Click to apply to the image.');button.onclick=()=>apply(entry,gate,colour,population.name);row.append(el('td',{},[button]));
          if(!gate.error&&gate.coordinate!==null&&gate.coordinate>=low&&gate.coordinate<=high){
            const xx=x(gate.coordinate),line=svg('line',{x1:xx,x2:xx,y1:27,y2:201,stroke:colour,'stroke-width':active?3:1.5,'stroke-dasharray':index?'5 3':''});
            const hit=svg('line',{x1:xx,x2:xx,y1:27,y2:201,stroke:'transparent','stroke-width':10,tabindex:0,role:'button','aria-label':button.getAttribute('aria-label')});
            hit.style.cursor='pointer';hit.onclick=button.onclick;hit.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();button.click();}};plot.append(line,hit);
          }else if(!gate.error)button.title+=' Outside this plot axis; the table still applies the exact raw value.';
        });body.append(row);
      }
      const metadata=el('details',{class:'threshold-measurements'},[el('summary',{text:'Measurement details'})]);
      for(const p of result.populations)metadata.append(el('p',{class:'hint',text:(p.name==='Annotation'?scopeLabel:p.name)+' · '+p.pixels.toLocaleString()+' pixels · '+(p.downsample===1?'full resolution':'1/'+number(p.downsample)+' resolution')+(p.omitted?' · '+p.omitted.toLocaleString()+' nonpositive pixels omitted on '+result.scale:'')}));
      metadata.append(el('p',{class:'hint',text:'256 shared bins across the full intensity range. Cutoffs are raw values; “above” counts all pixels strictly above the cutoff. Preview and full-resolution populations can give different cutoffs.'}));
      wrap.append(el('strong',{text:channel.name}),curveLegend,plot,el('div',{class:'hint',text:'Cutoffs in raw units · click a value to preview'}),table,metadata);plots.append(wrap);
    }
  }
  sync();return {sync,sources:annotation=>{
    if(disposed)return;const layer=state.layers.find(l=>l.type==='image'&&l.token===annotation?.image_token);
    if(!layer||layer.thresholdAnnotations?.[0]?.token===annotation.token)return;layer.thresholdAnnotations=[annotation];
    if(image===layer){image=null;sync();buildPanel();}
  },dispose:()=>{disposed=true;controller?.abort();generation++;clearTimeout(timer);}};
}
