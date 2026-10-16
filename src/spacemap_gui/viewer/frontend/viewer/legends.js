/* SPACEMAP viewer, part 11 of 12: colour columns, legends and the labelling card.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

async function columnsFor(layer) {
  if (!layer.columns) {
    const result = await getJSON("/cells/" + layer.token + "/columns?t=" + encodeURIComponent(layer.table || ""));
    layer.columns = result.columns || [];
    layer.columnKinds = result.kinds || {};
  }
  return layer.columns;
}

function sortColumns(columns) {
  const channels = new Set(((state.base && state.base.info.channels) || []).map(String));
  const priority = (c) => (/pheno|cluster|consensus|ref|manual|neigh|^cn|zone|bed|class/i.test(c) ? 0 : channels.has(c) ? 1 : 2);
  return columns.slice().sort((a, b) => priority(a) - priority(b) || a.localeCompare(b));
}

function annotationLegend(layer) {
  if (!layer.legendRequest) {
    layer.legendRequest = state.whenCellsReady.then(() => {
      if (!state.layers.includes(layer) && !state.layers.some(image => image.thresholdAnnotations?.includes(layer))) return { items: [] };
      return getJSON("/annotations/" + layer.token + "/legend");
    }).catch((error) => {
      layer.legendRequest = null;
      throw error;
    });
  }
  return layer.legendRequest;
}

function legendInto(layer, body, path, onKind) {
  const box = el("div", { class: "legend", text: "Loading legend..." });
  body.appendChild(box);
  const hide = Array.from(layer.hidden).join("\u001f");
  const request = layer.type === "annotations" ? annotationLegend(layer) :
    getJSON(path + (path.includes("?") ? "&" : "?") + "hide=" + encodeURIComponent(hide));
  request.then((legend) => {
    box.textContent = "";
    if (onKind) onKind(legend.kind);
    if (legend.kind === "numeric") {
      box.appendChild(el("div", { class: "gradient", style: "background: linear-gradient(90deg," + gradientStops(legend.colormap) + ")" }));
      box.appendChild(el("div", { class: "range" }, [el("span", { text: Number(legend.min).toFixed(1) }), el("span", { text: Number(legend.max).toFixed(1) })]));
      return;
    }
    if ((legend.items || []).length > 1) {
      const all = allBox("All", (on) => {
        layer.hidden = on ? new Set() : new Set((legend.items || []).map((i) => i.value));
        bump(layer);
      });
      all.sync((legend.items || []).map((i) => !layer.hidden.has(i.value)));
      box.appendChild(all.row);
    }
    (legend.items || []).forEach((item) => {
      const row = el("div", { class: "item" + (layer.hidden.has(item.value) ? " hidden" : ""), title: "Click to show or hide" }, [
        el("span", { class: "swatch", style: "background:" + item.color }),
        el("span", { text: item.value }),
        el("span", { class: "count", text: Number(item.count).toLocaleString() }),
      ]);
      row.addEventListener("click", () => {
        if (layer.hidden.has(item.value)) layer.hidden.delete(item.value); else layer.hidden.add(item.value);
        bump(layer);
      });
      box.appendChild(row);
    });
    if (!(legend.items || []).length) box.appendChild(el("div", { class: "hint", text: "No values for this ROI." }));
  }).catch((error) => { box.textContent = error.message; });
}

function gradientStops(name) {
  const maps = {
    viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"], magma: ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
    inferno: ["#000004", "#56106e", "#bb3754", "#f98e09", "#fcffa4"], plasma: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
    cividis: ["#00224e", "#434e6c", "#7d7c78", "#bcaf6f", "#fee838"], Reds: ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
    Blues: ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    turbo: ["#30123b", "#28bbec", "#a4fc3c", "#fb7e21", "#7a0403"], hot: ["#0b0000", "#b20000", "#ff6700", "#ffe34d", "#ffffff"],
  };
  return (maps[name] || maps.viridis).join(",");
}

function referenceChoice(input, choices, label) {
  const wrapper = el('div', {class: 'reference-choice'});
  const menu = el('div', {id: choices.id + '-menu', class: 'reference-choice-menu', role: 'listbox',
    'aria-label': label + ' choices', hidden: true});
  input.removeAttribute('list');
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-controls', menu.id);
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('autocomplete', 'off');
  let active = -1, filtering = false;
  const close = () => { menu.hidden = true; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant'); active = -1; };
  const pick = value => {
    input.value = value;
    input.dispatchEvent(new Event('input', {bubbles: true}));
    close();
    input.dispatchEvent(new Event('change', {bubbles: true}));
    if (input.isConnected) input.focus({preventScroll: true});
  };
  const fill = () => {
    menu.replaceChildren(); active = -1; input.removeAttribute('aria-activedescendant');
    const query = filtering ? input.value.toLocaleLowerCase() : '';
    const values = Array.from(new Set(Array.from(choices.options, option => option.value)))
      .filter(value => value.toLocaleLowerCase().includes(query)).sort((a,b) => a.localeCompare(b));
    if (!values.length) menu.appendChild(el('div', {class: 'hint', text: 'No saved match. You can use your typed name.'}));
    values.forEach((value, index) => {
      const option = el('button', {class: 'reference-option', type: 'button', text: value, tabindex: '-1',
        id: menu.id + '-' + index, role: 'option', 'aria-selected': String(value === input.value), onclick: () => pick(value)});
      const colour=Array.from(choices.options).find(item=>item.value===value)?.dataset.color;
      if(colour)option.prepend(el('span',{class:'swatch',style:'display:inline-block;margin-right:7px;background:'+colour}));
      option.addEventListener('pointerdown', event => event.preventDefault());
      menu.appendChild(option);
    });
  };
  const open = () => { fill(); menu.hidden = false; input.setAttribute('aria-expanded', 'true'); };
  input.addEventListener('click', () => { filtering = false; open(); });
  input.addEventListener('input', () => { filtering = true; open(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') { close(); event.stopPropagation(); }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (menu.hidden) { filtering = false; open(); }
      const options = Array.from(menu.querySelectorAll('[role=option]'));
      if (!options.length) return;
      active = (active + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
      options.forEach((option, index) => option.classList.toggle('active', index === active));
      input.setAttribute('aria-activedescendant', options[active].id);
      options[active].scrollIntoView({block: 'nearest'});
    }
    if (event.key === 'Enter') {
      event.preventDefault(); event.stopPropagation();
      const option = menu.querySelectorAll('[role=option]')[active];
      if (!menu.hidden && option) pick(option.textContent);
      else { close(); input.dispatchEvent(new Event('change', {bubbles: true})); }
    }
  });
  wrapper.addEventListener('focusout', event => { if (!wrapper.contains(event.relatedTarget)) close(); });
  choices.refreshChoices = () => { if (!menu.hidden) fill(); };
  wrapper.append(input, el('span', {class: 'reference-arrow', text: '▾', 'aria-hidden': 'true'}), choices, menu);
  return wrapper;
}

function manualReferenceRoute() {
  return '/studio/' + encodeURIComponent(new URLSearchParams(location.search).get('session')) + '/manual_reference';
}

function cellPhenotypeMenu() {
  const inspected=state.inspectedCell;
  if(!state.studio || !inspected || state.tool!=='off')return null;
  const {layer,result}=inspected, table=layer.table, column=(state.labelColumn || 'phenotype_ref1').trim();
  const labels=(result.labels || [result.label]).slice();
  if(!labels.length || !labels.every(Number.isInteger))return null;
  const dataMatches=()=>state.manualReferenceSource!=='data' || (dataView.selected.size===labels.length &&
    dataView.pickedRoi===dataLayerRoi(layer) && dataView.pickedLabels?.length===labels.length && labels.every(label=>dataView.pickedLabels.includes(label)));
  if(!dataMatches())return el('div',{class:'ctx-title',text:'Data rows selected. Use Manual labels to assign those rows, or select cells in Inspect for quick assignment.'});
  const box=el('div',{class:'cell-phenotype-menu'}), suggested=el('div'), list=el('div',{class:'ctx-phenotype-list'}),
    status=el('div',{class:'ctx-title',role:'status',text:'Loading phenotypes…'}),
    search=el('input',{class:'ctx-phenotype-search',placeholder:'Find or type a phenotype','aria-label':'Find phenotype to assign',maxlength:128});
  let info, busy=false;
  const current=()=>state.layers.includes(layer)&&layer.table===table&&state.inspectedCell===inspected&&dataMatches()&&
    (state.labelColumn || 'phenotype_ref1').trim()===column;
  const choose=async value=>{
    if(!info || busy || !value.trim())return;
    if(!current()){closeContextMenu();message('The selected cells changed. Open the menu again.',true);return;}
    busy=true;closeContextMenu();
    try{
      await flushLabels();
      if(labelSaves.error || !current())throw Error(labelSaves.error || 'The selection changed. Open the menu again.');
      const found=await postJSON(manualReferenceRoute(),{action:'assign',table:info.table,revision:info.revision,
        column,phenotype:value.trim(),selection:{cells:labels,roi:dataLayerRoi(layer)},request_id:newSaveId()});
      if(!current()){message(found.message);return;}
      state.manualReferenceEdit=found.request_id;state.labelColumn=column;state.labelName=value.trim();
      layer.columns=null;layer.referenceOptionsRequest=null;layer.live=null;layer.color_by=column;layer.hidden=new Set();
      if(layer.mode==='outline')layer.mode='both';
      const members=result.members || [result];members.forEach(member=>{member.details={...member.details,[column]:value.trim()};});
      if(result.members)result.details=selectedCellResult(members).details;
      bump(layer);
      if(current())showCellCard(layer,result);
      document.dispatchEvent(new CustomEvent('spacemap-reference-updated',{detail:{column}}));
      document.querySelectorAll('.manual-reference-card').forEach(card=>card.refreshReference?.());
      if(dataView.holder?.isConnected){dataView.columns=null;await dataLoad(true);}
      message(found.message+' Undo is in Manual labels.');
    }catch(error){message(error.message,true);}
  };
  const row=(name,probability)=>el('button',{type:'button',role:'menuitem','data-phenotype':name,
    disabled:info?null:'',
    class:Number.isFinite(probability)?'ctx-phenotype-suggestion':'',
    title:Number.isFinite(probability)?'Model support, not certainty. Assigns this name to all '+labels.length+' selected cells.':'Assign to all '+labels.length+' selected cells.',
    onclick:()=>choose(name)},[
      el('span',{class:'ctx-swatch',style:'background:'+(info?.preset_colours?.[name] || '#888888')}),
      el('span',{text:name}),Number.isFinite(probability)?el('span',{class:'ctx-probability',text:(100*probability).toFixed(1)+'%'}):null].filter(Boolean));
  box.refresh=()=>{
    if(!box.isConnected || !current())return;
    const ready=layer.suggestionsOn&&result.suggestions?.status==='ready'&&
      result.suggestionSettings===JSON.stringify(suggestionSettings(layer));
    const choices=ready?(result.suggestions.choices || []).filter(item=>Number.isFinite(item.probability)).slice().sort((a,b)=>b.probability-a.probability):[];
    const used=new Set(choices.map(item=>item.phenotype));
    suggested.replaceChildren(...(choices.length?[
      el('div',{class:'ctx-title',text:labels.length>1?'Suggested · average scores'+(result.suggestions.scored<result.suggestions.total?' · '+result.suggestions.scored+'/'+result.suggestions.total+' scored':''):'Suggested'}),
      ...choices.map(choice=>row(choice.phenotype,choice.probability))]:[]));
    list.replaceChildren(...(info?.phenotypes || []).filter(name=>!used.has(name)&&name.toLowerCase().includes(search.value.toLowerCase())).map(name=>row(name)));
    status.textContent=info?'':status.textContent;
    box.closest('#ctxmenu')?.reposition?.();
  };
  search.oninput=()=>box.refresh();
  search.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();event.stopPropagation();choose(search.value);} };
  box.append(el('div',{class:'ctx-title',text:'Assign to '+labels.length+' '+(labels.length===1?'cell':'cells')+' · '+column}),suggested,search,list,status);
  if(table!==layer.combinedTable){status.textContent='Choose the combined cell table in Phenotyping to label this selection.';search.disabled=true;return box;}
  const cached=document.querySelector('.manual-reference-card')?.referenceInfo;
  if(cached?.table===dataView.combined?.path&&cached.column===column)info=cached;
  queueMicrotask(()=>box.refresh());
  getJSON(manualReferenceRoute()+'?column='+encodeURIComponent(column)).then(found=>{
    if(!box.isConnected||!current())return;
    if(found.table!==dataView.combined?.path)throw Error('The combined table changed. Select the cells again.');
    info=found;box.refresh();
  }).catch(error=>{status.textContent=error.message;});
  return box;
}

async function phenotypePresetEditor(updated) {
  const dialog=el('dialog',{class:'phenotype-preset-editor','aria-label':'Phenotype preset'});
  const rows=el('div',{class:'phenotype-preset-rows'}), status=el('div',{class:'hint',role:'status'});
  const close=()=>{dialog.close();dialog.remove();};
  const add=item=>{
    const row=el('div',{class:'row'}), name=el('input',{value:item.name,placeholder:'Phenotype name','aria-label':'Preset phenotype name'}),
      colour=el('input',{type:'color',value:item.color || '#888888','aria-label':'Preset phenotype colour'});
    colour.dataset.auto = item.color ? 'false' : 'true';
    colour.dataset.autoValue = colour.value;
    colour.addEventListener('input', () => { colour.dataset.auto = 'false'; });
    colour.addEventListener('change', () => { colour.dataset.auto = 'false'; });
    name.addEventListener('change', async () => {
      if (colour.dataset.auto !== 'true' || !name.value.trim()) return;
      const value = name.value.trim();
      try {
        const found = await postJSON(manualReferenceRoute().replace('/manual_reference','/colours'), {names:[value]});
        if (name.value.trim() === value && colour.dataset.auto === 'true') {
          colour.value = found.mapping[value]; colour.dataset.autoValue = colour.value;
        }
      } catch(error) { status.textContent = error.message; }
    });
    row.append(colour,name,el('button',{class:'btn small',text:'×','aria-label':'Remove phenotype from preset',onclick:()=>row.remove()}));rows.append(row);
  };
  const route=manualReferenceRoute().replace('/manual_reference','/phenotype_presets');
  let revision;
  const save=el('button',{class:'btn primary',text:'Save preset',disabled:true,onclick:async()=>{
    save.disabled=true;
    try{
      const entries=Array.from(rows.children).map(row=>{
        const colour=row.querySelector('input[type=color]');
        return {name:row.querySelector('[aria-label="Preset phenotype name"]').value.trim(),
          color:colour.dataset.auto === 'true' && colour.value === colour.dataset.autoValue ? null : colour.value};
      });
      await postJSON(route,{entries,revision});
      document.dispatchEvent(new CustomEvent('spacemap-phenotype-presets'));
      await updated?.();close();
    }catch(error){status.textContent=error.message;save.disabled=false;}
  }});
  dialog.append(el('div',{class:'row'},[el('strong',{class:'grow',text:'Phenotype preset'}),el('button',{class:'btn small',text:'Close',onclick:close})]),
    el('p',{class:'hint',text:'One colour per phenotype across all methods and projects. New names get a distinct colour; click its swatch to change it. This list does not rename labelled cells.'}),
    rows,el('button',{class:'btn small',text:'Add phenotype',onclick:()=>add({name:''})}),status,
    el('div',{class:'row'},[save,el('button',{class:'btn',text:'Cancel',onclick:close})]));
  dialog.addEventListener('cancel',()=>dialog.remove());document.body.append(dialog);dialog.showModal();
  try{const found=await getJSON(route);revision=found.revision;found.entries.forEach(add);save.disabled=false;}
  catch(error){status.textContent=error.message;}
}

function referenceFeatureControls() {
  const summary=el('summary',{text:'Features · loading…'}), box=el('details',{class:'reference-training-features'},[summary]);
  const status=el('div',{class:'hint',role:'status'}), list=el('div',{class:'reference-feature-list'});
  const search=el('input',{placeholder:'Find a feature…','aria-label':'Find model feature'});
  const set=el('select',{'aria-label':'Use model feature marker set'},[el('option',{value:'',text:'Use a marker set…'})]);
  const missing=el('input',{type:'checkbox','aria-label':'Use available model features'});
  const save=el('button',{class:'btn small',text:'Save features',disabled:true});
  let found=null,chosen=new Set(),dirty=false;
  const update=()=>{
    summary.textContent='Features · '+chosen.size+' selected';save.disabled=!dirty;
    list.querySelectorAll('input[type=checkbox]').forEach(input=>{input.checked=chosen.has(input.value);});
  };
  const changed=()=>{dirty=true;status.textContent='Unsaved feature changes';update();};
  const draw=()=>{
    list.replaceChildren();
    for(const group of ['Markers','Morphology','H&E','Other numeric']){
      const entries=(found?.features || []).filter(item=>item.group===group&&item.name.toLowerCase().includes(search.value.toLowerCase()));
      if(!entries.length)continue;
      const head=el('div',{class:'row wrap'},[el('strong',{text:group})]);
      for(const [label,on] of [['All',true],['None',false]])head.append(el('button',{class:'btn small',text:label,onclick:()=>{
        entries.forEach(item=>on?chosen.add(item.name):chosen.delete(item.name));changed();
      }}));
      list.append(head);
      const rows=el('div',{class:'reference-feature-grid'});
      entries.forEach(item=>{
        const input=el('input',{type:'checkbox',value:item.name,'aria-label':'Train with '+item.name,checked:chosen.has(item.name)?'':null});
        input.onchange=()=>{input.checked?chosen.add(item.name):chosen.delete(item.name);changed();};
        rows.append(el('label',{class:'inline-check'},[input,el('span',{text:item.name})]));
      });list.append(rows);
    }
    update();
  };
  search.oninput=draw;missing.onchange=changed;
  set.onchange=()=>{
    const entry=found?.sets?.[Number(set.value)];if(set.value===''||!entry)return;
    const names=new Set(entry.markers);
    (found.features || []).filter(item=>item.group==='Markers').forEach(item=>names.has(item.name)?chosen.add(item.name):chosen.delete(item.name));
    changed();
  };
  box.saveIfChanged=async()=>{
    if(!dirty)return;
    save.disabled=true;
    const value=await postJSON(manualReferenceRoute().replace('/manual_reference','/reference_features'),{features:[...chosen],allow_missing:missing.checked});
    found=value;chosen=new Set(value.selected);dirty=false;status.textContent='Saved for the next training update.';update();
    if(value.auto_reference)document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:value.auto_reference}));
  };
  save.onclick=async()=>{try{await box.saveIfChanged();}catch(error){status.textContent=error.message;update();}};
  box.append(el('div',{class:'row wrap'},[search,set]),list,
    el('label',{class:'inline-check'},[missing,el('span',{text:'Use available features when some are missing'}),help('Refit from saved training examples using only the features measured in the new cell or dataset. Missing values are not zero. Fewer features may reduce reliability; no model can classify every dataset accurately.')]),
    el('div',{class:'row wrap'},[save,help('These saved table columns are used by manual and Auto-train, independently of Top markers and clustering. Area and other morphology features are optional. H&E uses saved per-cell measurements, not image tiles. Keep compatible feature names, scales and units across datasets; clinical data and IDs are not marker measurements.')]),status);
  getJSON(manualReferenceRoute().replace('/manual_reference','/reference_features')).then(value=>{
    if(!box.isConnected)return;
    found=value;chosen=new Set(value.selected);missing.checked=value.allow_missing;
    value.sets.forEach((entry,index)=>set.append(el('option',{value:String(index),text:entry.name})));
    status.textContent=value.missing.length?'Missing columns: '+value.missing.join(', '):'';draw();
  }).catch(error=>{summary.textContent='Features';status.textContent=error.message;});
  return box;
}

function manualReferenceSelections(layer) {
  const items = [], picked = state.manualReferencePick;
  if (picked?.layer === layer && picked.table === layer?.table && layer?.table === layer?.combinedTable && picked.labels.length)
    items.push({id:'viewer', count:picked.labels.length, text:'Viewer cells', table:dataView.combined?.path,
      selection:{cells:picked.labels, roi:dataLayerRoi(layer)}});
  if (dataView.source?.combined) {
    if (dataView.selected.size) items.push({id:'data', count:dataView.selected.size, text:'Selected Data rows', table:dataView.combined?.path,
      selection:{rows:Array.from(dataView.selected)}});
    if (!dataView.loading && dataView.queryReady && dataView.result?.matched) {
      const body = dataView.resultBody, filters = body.filters.slice();
      if (body.rows.this_image && body.rows.column && body.rows.value !== undefined) filters.push({col:body.rows.column,values:[String(body.rows.value)]});
      items.push({id:'filtered', count:dataView.result.matched, text:'Filtered Data rows', table:dataView.combined?.path, selection:{filters}});
    }
  }
  if (state.spec?.selection?.id && state.spec.selection.count) items.push({id:'rule', text:'Marker rule', count:state.spec.selection.count,
    selection:{token:state.spec.selection.id}});
  return items;
}

document.addEventListener('spacemap-cells-picked', event => {
  const {layer, labels, fromData} = event.detail || {};
  if (!fromData) {
    state.manualReferencePick = {layer,table:layer?.table,labels:labels || []};
    state.manualReferenceSource = 'viewer';
  }
  document.querySelectorAll('.manual-reference-card').forEach(card => card.refreshSelection?.());
});

function manualReferenceCard(layer) {
  const card = el('div', {class:'layer manual-reference-card'}), body = el('div', {class:'layer-body'});
  const status = el('div', {class:'hint',role:'status'}), counts = el('div', {class:'reference-counts muted'});
  const column = el('input', {value:state.labelColumn || 'phenotype_ref1',placeholder:'Choose or type a column', 'aria-label':'Save labels in column'});
  const columnList = el('datalist', {id:'manual-reference-columns'}); column.setAttribute('list',columnList.id);
  const name = el('input', {value:state.labelName || '',placeholder:'Choose or type a phenotype','aria-label':'Phenotype name to assign'});
  const names = el('datalist', {id:'manual-reference-names'}); name.setAttribute('list',names.id);
  const scope = el('select', {'aria-label':'Cells to label'});
  let info = null, busy = false, loadSequence = 0;
  state.labelColumn = column.value;
  name.oninput = () => { state.labelName = name.value; };
  const update = async () => {
    if (!card.isConnected) return;
    const sequence = ++loadSequence;
    try {
      const found = await getJSON(manualReferenceRoute() + '?column=' + encodeURIComponent(column.value.trim()));
      if (!card.isConnected || sequence !== loadSequence) return;
      if(info?.writing.state==='writing'&&found.writing.state!=='writing')
        status.textContent=found.writing.state==='done'?'Labels and columns written into the dataset.':found.writing.message || 'The write stopped. Please try again.';
      info = found;
      card.referenceInfo=found;
      const learned=learn.value;
      learn.replaceChildren(el('option',{value:'',text:'Manual label column'}),...found.columns.map(value=>el('option',{value,text:value})));
      if(Array.from(learn.options).some(option=>option.value===learned))learn.value=learned;
      train.textContent=learn.value?'Train now':'Train from this reference';
      columnList.replaceChildren(...found.references.map(value => el('option',{value})));
      names.replaceChildren(...found.phenotypes.map(value => el('option',{value,'data-color':found.preset_colours?.[value] || ''})));
      columnList.refreshChoices?.(); names.refreshChoices?.();
      counts.replaceChildren(...(found.counts.groups || []).map(([label,n]) => el('button',{class:'btn small count-chip',text:label+' · '+n.toLocaleString(),onclick:()=>{name.value=label;state.labelName=label;}})));
      if (!counts.childNodes.length) counts.textContent = 'No manual labels yet.';
      save.disabled = found.writing.state === 'writing' || !(found.pending_labels || found.pending_columns.length);
      save.textContent = found.writing.state === 'writing' ? 'Writing…' : 'Write into table';
      const previousPrediction=compare.value;
      compare.replaceChildren(...found.columns.filter(value=>value!==column.value).map(value=>el('option',{value,text:value})));
      if(Array.from(compare.options).some(option=>option.value===previousPrediction)) compare.value=previousPrediction;
      evaluation.querySelector('button').disabled=!compare.value || !found.references.includes(column.value.trim());
      card.refreshSelection();
      if (found.writing.state === 'writing') setTimeout(update,1500);
    } catch(error) { status.textContent=error.message; }
  };
  const refresh = async () => {
    if (layer) {
      layer.columns=null;layer.referenceOptionsRequest=null;layer.live=null;layer.color_by=state.labelColumn;layer.hidden=new Set();
      if (layer.mode==='outline') layer.mode='both';
      bump(layer);
    }
    if (dataView.holder?.isConnected) { dataView.columns=null; await dataLoad(true); }
    document.dispatchEvent(new CustomEvent('spacemap-reference-updated',{detail:{column:state.labelColumn}}));
    await update();
  };
  card.refreshReference=update;
  column.onchange = () => { state.labelColumn=column.value.trim(); if(layer) layer.learnColumn=state.labelColumn; update(); if(auto.checked&&!learn.value)saveAuto(); };
  const create = el('button',{class:'btn small',text:'Create', 'aria-label':'Create reference column',onclick:async()=>{
    create.disabled=true;
    try { state.labelColumn=column.value.trim(); await postJSON(manualReferenceRoute(),{action:'create',column:state.labelColumn}); status.textContent='Column ready. Select cells and assign a phenotype.'; await refresh(); }
    catch(error){status.textContent=error.message;}finally{create.disabled=false;}
  }});
  const show=el('button',{class:'btn small',text:'Show',disabled:!layer,onclick:()=>{if(layer){state.labelColumn=column.value.trim();layer.color_by=state.labelColumn;layer.live=null;layer.hidden=new Set();if(layer.mode==='outline')layer.mode='both';bump(layer);}}});
  const assign = async clear => {
    const selected=manualReferenceSelections(layer).find(item=>item.id===scope.value);
    if(!selected || !info || busy) return;
    if(selected.table && selected.table!==info.table){status.textContent='The selection belongs to another table. Select the cells again.';return;}
    const phenotype=clear ? '' : name.value.trim();
    if(!clear && !phenotype){status.textContent='Choose or enter a phenotype first.';name.focus();return;}
    busy=true;card.refreshSelection();status.textContent='Saving labels…';
    try {
      await flushLabels();
      const result=await postJSON(manualReferenceRoute(),{action:'assign',table:info.table,revision:info.revision,
        column:column.value.trim(),phenotype,selection:selected.selection,request_id:newSaveId()});
      state.manualReferenceEdit=result.request_id;state.labelColumn=result.column;status.textContent=result.message;
      await refresh();
    }catch(error){status.textContent=error.message;}finally{busy=false;card.refreshSelection();}
  };
  const apply=el('button',{class:'btn primary',text:'Assign phenotype',onclick:()=>assign(false)});
  const clear=el('button',{class:'btn small',text:'Clear labels',onclick:()=>assign(true)});
  scope.onchange=()=>{state.manualReferenceSource=scope.value;card.refreshSelection();};
  card.refreshSelection=()=>{
    const selections=manualReferenceSelections(layer), previous=state.manualReferenceSource || scope.value;
    scope.replaceChildren(...selections.map(item=>el('option',{value:item.id,text:item.text+' · '+item.count.toLocaleString()})));
    scope.value=selections.some(item=>item.id===previous)?previous:selections.find(item=>item.id!=='filtered')?.id || '';
    if(!scope.value) scope.prepend(el('option',{value:'',text:'Select cells first'}));
    const chosen=selections.find(item=>item.id===scope.value);
    apply.disabled=clear.disabled=busy || !info || !chosen;
    apply.textContent=chosen?'Assign '+chosen.count.toLocaleString()+' cells':'Assign phenotype';
  };
  const undoButton=el('button',{class:'btn small',text:'Undo',onclick:async()=>{
    try { if(state.manualReferenceEdit){await postJSON(manualReferenceRoute(),{action:'undo',request_id:state.manualReferenceEdit});state.manualReferenceEdit=null;status.textContent='Last assignment undone.';await refresh();}else await undo(); }
    catch(error){status.textContent=error.message;}
  }});
  const save=el('button',{class:'btn small',text:'Write into table',onclick:async()=>{
    save.disabled=true;
    try{await flushLabels();await postJSON(manualReferenceRoute(),{action:'write'});if(info)info.writing={state:'writing'};status.textContent='Writing saved labels and result columns into the dataset…';await update();}
    catch(error){status.textContent=error.message;save.disabled=false;}
  }});
  card.append(el('div',{class:'layer-head'},[el('span',{class:'name',text:'Manual labels'}),help('Choose or create a reference column. Select cells, choose a phenotype and Assign. Labels are saved beside the dataset immediately and used by Studio. Write into table also puts saved changes into the Parquet or CSV file.')]),body);
  body.append(field('Column',el('div',{class:'inline'},[referenceChoice(column,columnList,'column'),create,show]),'Use any new column name, such as phenotype_ref1. Each reference column is independent.'),
    field('Phenotype',el('div',{class:'inline'},[referenceChoice(name,names,'phenotype'),el('button',{class:'btn small',text:'Edit list…','aria-label':'Edit phenotype preset',onclick:()=>phenotypePresetEditor(update)})]),'Choose a saved phenotype or type a new one. Edit list saves names and colours for all your projects; existing cell labels are never renamed by editing this list.'),counts,
    field('Selection',scope,'Ctrl-click cells in Inspect mode, select Data rows with Ctrl/Shift, or use Data filters or a marker rule. The count is the number that will be labelled.'),
    el('div',{class:'row wrap'},[apply,clear,undoButton,save]),status);
  if(layer){
    const tools=el('div',{class:'seg'});
    [['off','Inspect'],['click','Click to label'],['box','Box to label']].forEach(([key,label])=>tools.append(el('button',{class:state.tool===key?'on':'',text:label,onclick:()=>{state.manualReferenceEdit=null;setTool(key);}})));
    body.append(field('Tool',tools,'Inspect selects cells first (Ctrl adds). Click to label and Box to label assign the current phenotype directly; Shift clears.'));
    body.append(el('div',{class:'row save-row'},[el('span',{class:'chip ok label-save-status',role:'status',text:'Saved'}),
      el('button',{class:'btn small label-save-now',text:'Save now',hidden:true,onclick:()=>flushLabels().then(refreshLabelColours)})]),
      el('div',{class:'hint label-save-error',role:'alert',hidden:true}));
    queueMicrotask(updateLabelStatus);
  }
  const modelBox=el('details',{class:'reference-model-training'},[el('summary',{text:'Model training'})]), modelBody=el('div');modelBox.append(modelBody);
  const features=referenceFeatureControls();
  const checked=el('input',{type:'checkbox','aria-label':'Only checked training cells'});
  const learn=el('select',{'aria-label':'Learn model from column'},[el('option',{value:'',text:'Manual label column'})]);
  const trainingColumn=()=>learn.value || column.value.trim();
  const auto=el('input',{type:'checkbox','aria-label':'Auto-train suggestions'}), autoStatus=el('span',{class:'hint',role:'status'});
  let autoBusy=false;
  card.applyAuto=value=>{
    if(autoBusy || !card.isConnected || (info?.table&&value.table&&info.table!==value.table))return;
    auto.checked=!!value.enabled;
    const short={ready:'Ready',training:'Updating…',waiting:'Waiting',off:'Off'}[value.state];
    autoStatus.textContent=value.enabled ? (value.column?value.column+' · ':'')+(short || value.message || '') : '';
    autoStatus.title=value.message || '';
    if(value.enabled&&['waiting','error'].includes(value.state)&&/five|phenotype|usable|failed|stopped|missing|changed/i.test(value.message || ''))autoStatus.textContent=value.message;
  };
  const saveAuto=async()=>{
    autoBusy=true;auto.disabled=true;
    try{if(auto.checked){await flushLabels();await features.saveIfChanged();}const found=await postJSON(manualReferenceRoute().replace('/manual_reference','/auto_reference'),
      {enabled:auto.checked,column:trainingColumn(),checked_only:checked.checked});
      autoBusy=false;card.applyAuto(found);document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:found}));}
    catch(error){auto.checked=false;autoStatus.textContent=error.message;}
    finally{autoBusy=false;auto.disabled=false;}
  };
  auto.onchange=saveAuto;
  learn.onchange=()=>{if(auto.checked)saveAuto();};
  checked.onchange=()=>{if(auto.checked)saveAuto();};
  body.append(el('div',{class:'row wrap'},[el('label',{class:'inline-check'},[auto,el('span',{text:'Auto-train'})]),
    help('Updates suggestions from Learn from after you pause labelling. Keeps one automatic model per table and training column, replacing it only after a successful update. Manual training saves separate models. Starts with two phenotypes and five usable cells each; updates at most once a minute. It never assigns labels on its own. Improvement is not guaranteed.'),autoStatus]));
  getJSON(manualReferenceRoute().replace('/manual_reference','/auto_reference')).then(found=>{
    card.applyAuto(found);checked.checked=!!found.checked_only;
    if(found.column&&found.column!==column.value.trim()){
      if(!Array.from(learn.options).some(option=>option.value===found.column))learn.append(el('option',{value:found.column,text:found.column}));
      learn.value=found.column;
      train.textContent='Train now';
    }
    document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:found}));
  }).catch(error=>{autoStatus.textContent=error.message;});
  const train=el('button',{class:'btn',text:'Train from this reference',onclick:async()=>{
    train.disabled=true;
    try{await flushLabels();await features.saveIfChanged();const result=await runCommand('train_reference_model',{reference_column:trainingColumn(),checked_only:checked.checked},{confirmed:true});status.textContent=result.message;state.studio?.showResult(result);}
    catch(error){status.textContent=error.message;}finally{train.disabled=false;}
  }});
  const predict=el('button',{class:'btn small',text:'Predict other cells…',onclick:()=>window.dispatchEvent(new CustomEvent('studio-reference-method',{
    detail:{method:'mr_ml',column:trainingColumn(),model_id:layer?.suggestionModelId}}))});
  learn.addEventListener('change',()=>{train.textContent=learn.value?'Train now':'Train from this reference';});
  modelBody.append(field('Learn from',learn,'Use your manual labels or any existing phenotype column. Learning another method copies its labels; its agreement score is not an independent accuracy check. This choice never changes the column where you save manual labels.'),
    features,
    el('div',{class:'row wrap'},[train,predict,help('Train saves Random Forest and Extra Trees models using Learn from and the selected features. At least two phenotypes with five complete examples each are needed. Auto-train uses this same feature list. Predict other cells opens MR-ML with your manual reference selected.')]),
    el('label',{class:'inline-check'},[checked,el('span',{text:'Only cells marked checked'}),help('On: train only on cells you approved with Mark checked. Off: use all labelled cells. Needs review cells are excluded either way.') ]));
  const folder=el('input',{'aria-label':'Reference model library folder',placeholder:'Model library folder'});
  const libraryStatus=el('div',{class:'hint',role:'status'});
  modelBody.append(field('Models folder',el('div',{class:'inline'},[folder,el('button',{class:'btn small',text:'Use',onclick:async()=>{
    try{await postJSON(manualReferenceRoute().replace('/manual_reference','/model_library'),{action:'location',path:folder.value});libraryStatus.textContent='Model folder updated.';await card.refreshModels();}
    catch(error){libraryStatus.textContent=error.message;}
  }})]),'The shared model library. Compatible models appear in Saved below. Partial panels are refitted using the shared markers; check their performance on manual cells in the new project.'),libraryStatus);
  const sharedModels=el('div',{class:'row wrap'}), suggestions=el('div');
  modelBody.append(sharedModels,suggestions);
  card.refreshModels=async trained=>{
    if(!modelBox.open){modelBox.loaded=false;return;}
    modelBox.loaded=true;
    try{const library=await getJSON(manualReferenceRoute().replace('/manual_reference','/model_library'));folder.value=library.location;folder.disabled=library.fixed;
      const mine=library.project.filter(item=>item.kind==='phenotyping');
      sharedModels.replaceChildren();
      mine.filter(item=>!library.library.some(saved=>saved.kind==='phenotyping'&&saved.id===item.id)).forEach(item=>sharedModels.append(el('button',{class:'btn small',text:'Share '+item.name,onclick:async()=>{try{await postJSON(manualReferenceRoute().replace('/manual_reference','/model_library'),{action:'share',kind:'phenotyping',id:item.id});libraryStatus.textContent='Model copied to the library.';await card.refreshModels();}catch(error){libraryStatus.textContent=error.message;}}})));
      if(layer && trained && mine.length) {layer.suggestionMode='saved_reference';layer.suggestionModelId=null;}
    }catch(error){libraryStatus.textContent=error.message;}
    suggestions.replaceChildren();
    if(layer){layer.suggestionMode ||= 'examples';layer.learnColumn=column.value.trim();suggestionControls(layer,suggestions);}
  };
  modelBox.addEventListener('toggle',async()=>{
    state.manualReferenceModelsOpen=modelBox.open;
    if(modelBox.open&&!modelBox.loaded)await card.refreshModels();
  });
  modelBox.open=!!state.manualReferenceModelsOpen;
  body.append(modelBox);
  const compare=el('select',{'aria-label':'Prediction column to check'}), score=el('div',{role:'status'});
  const evaluation=el('details',{},[el('summary',{text:'Check agreement (F1)'}),field('Compare',compare,'Choose the named phenotype output from Leiden, SOM, Vortex, MR-ML or another method. Only cells with a manual reference label are compared.'),
    el('button',{class:'btn small',text:'Calculate F1',disabled:true,onclick:async event=>{
      event.target.disabled=true;score.textContent='Comparing labelled cells…';
      try{const result=await postJSON(manualReferenceRoute(),{action:'evaluate',reference:column.value.trim(),method:compare.value});
        score.replaceChildren(el('div',{class:'row wrap'},[chip(result.cells.toLocaleString()+' labelled cells'),chip('Macro F1 '+result.macro_f1.toFixed(3)),help(result.note)]),
          el('table',{class:'studio-counts'},[el('tr',{},['Phenotype','Cells','Precision','Recall','F1'].map(text=>el('th',{text}))),
            ...result.phenotypes.map(row=>el('tr',{},[row.phenotype,row.cells,row.precision.toFixed(3),row.recall.toFixed(3),row.f1.toFixed(3)].map(text=>el('td',{text:String(text)}))))]));
      }catch(error){score.textContent=error.message;}finally{event.target.disabled=false;}
    }}),score]);
  body.append(evaluation);if(layer)referenceReviewControls(layer,body);card.refreshSelection();queueMicrotask(update);
  return card;
}

function labellingCard(layer) {
  const card = el("div", { class: "layer" });
  card.appendChild(el("div", { class: "layer-head" }, [el("span", { class: "name", text: "Label cells (Manual Reference)" }),
    help("Name cells from what you see. Clicks save together in the background and colours update after a short pause. " +
      "Write them into the CSV from the file panel ('Write pending changes into the CSV').")]));
  const body = el("div", { class: "layer-body" });
  if (!state.labelColumn) state.labelColumn = (layer.label_columns || [])[0] || "phenotype_ref1";
  const column = el("input", { type: "text", value: state.labelColumn, list: "label-columns" });
  const columnList = el("datalist", { id: "label-columns" });
  const availableColumns = new Set([...(layer.label_columns || []), ...(layer.columns || []).filter(c => /^phenotype_/i.test(c))]);
  availableColumns.forEach((c) => columnList.appendChild(el("option", { value: c })));
  column.setAttribute('aria-label', 'Save labels in column');
  column.placeholder = 'Choose or type a column';
  column.addEventListener("change", () => {
    const selected = column.value.trim();
    if (!selected) { column.value = state.labelColumn; message("Choose or name a reference column.", true); return; }
    // Manual labels belong in a reference column. An existing column that is not
    // one (a method's result such as phenotype_leiden) is only used when confirmed.
    const references = new Set([...(layer.label_columns || []), ...(layer.referenceColumns || [])]);
    const existing = availableColumns.has(selected) || (layer.columns || []).includes(selected);
    if (existing && !references.has(selected) && !window.confirm('"' + selected + '" holds another method\'s results, ' +
        'not manual labels. Labels you assign will change it. Use it anyway?')) {
      column.value = state.labelColumn; return;
    }
    state.labelColumn = selected;
    layer.label_columns = Array.from(new Set((layer.label_columns || []).concat([selected])));
    if (layer.live) layer.referencePhenotypes = layer.live.phenotypes.slice();
    layer.live = null; layer.color_by = selected; layer.hidden = new Set();
    if (layer.mode === "outline") layer.mode = "both";
    if (!layer.visible) setVisible(layer, true);
    if (state.inspectedCell?.layer === layer) inspectReviewedCell(layer, state.inspectedCell.result.label);
    bump(layer);
  });
  const showColumn = el("button", { class: "btn small", type: "button", text: "Show", title: "Colour the cells by this column",
    onclick: () => { layer.color_by = state.labelColumn; if (layer.mode === "outline") layer.mode = "both"; if (!layer.visible) setVisible(layer, true); bump(layer); } });
  body.appendChild(field("Column", el("div", { class: "inline" }, [referenceChoice(column, columnList, 'column'), showColumn]),
    "The column your labels are saved in. Click to see saved columns or type a new name. " +
    "Columns named ref, reference or manual hold manual labels; choosing another method's column asks first."));
  const name = el("input", { type: "text", value: state.labelName || "", placeholder: "e.g. CD8 T cell", list: "label-names" });
  name.setAttribute('aria-label', 'Phenotype name to assign');
  const names = el("datalist", { id: "label-names" });
  name.addEventListener("input", () => { state.labelName = name.value; });
  const phenotypeTipText = "The name given to the cells you click. Click to see saved names or type a new one. Click a count below to reuse its name.";
  const phenotypeTip = help(phenotypeTipText);
  const setPhenotypeTip = extra => { const text = phenotypeTipText + (extra ? " " + extra : ""); phenotypeTip.dataset.tip = text; phenotypeTip.setAttribute("aria-label", text); };
  body.appendChild(el("div", { class: "row field" }, [el("label", { class: "lbl", text: "Phenotype" }), referenceChoice(name, names, 'phenotype'), phenotypeTip]));
  const referenceCounts = el("div", { class: "reference-counts muted", text: "Reading saved labels..." });
  const referenceOptions = new Set((layer.live && layer.live.phenotypes) || layer.referencePhenotypes || []);
  const presetNote = el('div', {class: 'hint', role: 'status'});
  const retryChoices = el('button', {class: 'btn small', type: 'button', text: 'Retry choices', hidden: true});
  const optionsKey = layer.table;
  let choicesRequest = 0;
  const loadChoices = async (retry = false) => {
    const sequence = ++choicesRequest;
    retryChoices.hidden = true; presetNote.textContent = '';
    if (retry || !layer.referenceOptionsRequest || layer.referenceOptionsTable !== optionsKey) {
      layer.referenceOptionsTable = optionsKey;
      const request = state.whenCellsReady.then(() =>
        getJSON('/cells/' + layer.token + '/reference-options?t=' + encodeURIComponent(optionsKey)));
      layer.referenceOptionsRequest = request;
      request.catch(() => { if (layer.referenceOptionsRequest === request) layer.referenceOptionsRequest = null; });
    }
    try {
      const result = await layer.referenceOptionsRequest;
      if (sequence !== choicesRequest || layer.table !== optionsKey) return;
      layer.referenceColumns = result.reference_columns || [];
      result.columns.forEach(c => {
        if (!availableColumns.has(c)) { availableColumns.add(c); columnList.appendChild(el('option', {value: c})); }
      });
      result.phenotypes.forEach(value => {
        if (!referenceOptions.has(value)) { referenceOptions.add(value); names.appendChild(el('option', {value})); }
      });
      columnList.refreshChoices(); names.refreshChoices();
      setPhenotypeTip(result.sources.length ? 'Names also include phenotype_ml_consensus from: ' + result.sources.join(', ') + '.' : '');
      presetNote.textContent = result.unavailable?.length ? 'Could not read names from: ' + result.unavailable.join(', ') + '.' : '';
    } catch (error) {
      if (sequence !== choicesRequest || layer.table !== optionsKey) return;
      presetNote.textContent = error.message === 'HTTP 404' ?
        'The extra choices could not be loaded. Try Retry choices. If this continues, restart SPACEMAP and reopen the viewer.' :
        'Could not load extra choices: ' + error.message;
      retryChoices.hidden = false;
    }
  };
  retryChoices.addEventListener('click', () => loadChoices(true));
  loadChoices();
  // Column choices come from the existing table route too, so they remain
  // available even when an older service cannot supply consensus presets.
  columnsFor(layer).then(columns => {
    if (layer.table !== optionsKey) return;
    columns.filter(c => /^phenotype_/i.test(c)).forEach(c => {
      if (!availableColumns.has(c)) { availableColumns.add(c); columnList.appendChild(el('option', {value: c})); }
    });
    columnList.refreshChoices();
  }).catch(() => {});
  if (layer.resolve && !layer.referencePanelPromise) {
    layer.referencePanelPromise = getJSON("/resolve/" + layer.resolve + "/panel").then((result) => {
      layer.referencePhenotypes = (result.panel.table || []).map((row) => row.Phenotype).filter(Boolean);
      return layer.referencePhenotypes;
    }).catch(() => []);
  }
  if (layer.referencePanelPromise) layer.referencePanelPromise.then((phenotypes) => {
    phenotypes.forEach((value) => {
      if (!referenceOptions.has(value)) { referenceOptions.add(value); names.appendChild(el("option", { value: value })); }
    });
    names.refreshChoices();
  });
  layer.refreshReferenceCounts = () => getJSON("/cells/" + layer.token + "/legend?t=" + encodeURIComponent(layer.table) + "&col=" + encodeURIComponent(state.labelColumn))
    .then((legend) => {
      referenceCounts.textContent = "";
      names.textContent = "";
      const items = legend.items || [];
      items.forEach((item) => {
        referenceOptions.add(item.value);
        referenceCounts.appendChild(el("button", { class: "btn small count-chip", text: item.value + " · " + Number(item.count).toLocaleString(),
          title: "Use this phenotype for the next assignment", onclick: () => { state.labelName = item.value; name.value = item.value; } }));
      });
      referenceOptions.forEach((value) => names.appendChild(el("option", { value: value })));
      names.refreshChoices();
      if (!items.length) referenceCounts.textContent = "No labels in this column for this ROI yet.";
    }).catch((error) => { referenceCounts.textContent = error.message; });
  layer.refreshReferenceCounts();
  body.append(referenceCounts, presetNote, retryChoices);
  const tools = el("div", { class: "seg" });
  [["off", "Inspect"], ["click", "Click cells"], ["box", "Draw box"]].forEach(([key, label]) => {
    const button = el("button", { type: "button", text: label, class: state.tool === key ? "on" : "" });
    button.addEventListener("click", () => setTool(key));
    tools.appendChild(button);
  });
  body.appendChild(field("Tool", tools, "Inspect: click a cell to see its markers. Click cells: click to give it the name. " +
    "Draw box: drag to name every cell inside. Hold Shift to clear instead."));
  body.appendChild(el('div', {class: 'row save-row'}, [
    el('span', {class: 'chip ok label-save-status', role: 'status', text: 'Saved'}),
    el('button', {class: 'btn small label-save-now', text: 'Save now', hidden: true, onclick: () => flushLabels().then(refreshLabelColours)}),
    el('button', {class: 'btn small label-discard', text: 'Discard waiting', hidden: true,
      title: 'Remove waiting changes; labels already saved remain saved', onclick: () => {
        if (labelSaves.busy) return;
        labelSaves.queue.forEach(item => {
          try { localStorage.removeItem(draftPrefix + item.id); } catch (_) {}
          item.layer.columns = null; labelSaves.changed.add(item.layer);
        });
        labelSaves.queue = []; labelSaves.error = null;
        updateLabelStatus(); refreshLabelColours();
        message('Waiting changes removed. Labels already saved are kept.');
      }}),
    el('span', {class: 'spacer'}),
    el('button', {class: 'btn small', text: 'Undo', title: 'Remove the last waiting click, or undo the last saved group', onclick: undo}),
    el('button', {class: 'btn small', text: '\u21bb', title: 'Refresh colours', 'aria-label': 'Refresh colours', onclick: async () => {
      await flushLabels(); labelSaves.changed.add(layer); refreshLabelColours();
    }}),
  ]));
  body.appendChild(el('div', {class: 'hint label-save-error', role: 'alert', hidden: true}));
  queueMicrotask(updateLabelStatus);
  suggestionControls(layer, body);
  referenceReviewControls(layer, body);
  card.appendChild(body);
  return card;
}

function referenceReviewControls(layer, body) {
  const review = el('details', {class: 'reference-review'});
  review.open = !!layer.reviewOpen;
  review.appendChild(el('summary', {text: 'Review assigned cells'}));
  const reviewBody = el('div');
  const mode = select([['recent', 'Recent edits'], ['all', 'All assigned'], ['flagged', 'Needs review']], layer.reviewMode || 'recent', value => {
    layer.reviewMode = value; layer.reviewOffset = 0; layer.refreshReview();
  });
  const reviewList = el('div', {class: 'review-list'});
  const previous = el('button', {class: 'btn small', text: 'Previous', onclick: () => {
    layer.reviewOffset = Math.max(0, (layer.reviewOffset || 0)-30); layer.refreshReview();
  }});
  const next = el('button', {class: 'btn small', text: 'Next', onclick: () => {
    layer.reviewOffset = (layer.reviewOffset || 0)+30; layer.refreshReview();
  }});
  let reviewRequest = 0;
  layer.refreshReview = async () => {
    if (!review.open) return;
    const sequence = ++reviewRequest;
    reviewList.textContent = 'Reading cells...';
    try {
      const result = await getJSON('/cells/' + layer.token + '/review?t=' + encodeURIComponent(layer.table) +
        '&column=' + encodeURIComponent(state.labelColumn) + '&mode=' + (layer.reviewMode || 'recent') + '&offset=' + (layer.reviewOffset || 0));
      if (sequence !== reviewRequest) return;
      reviewList.textContent = '';
      result.items.forEach(item => {
        const button = el('button', {class: 'btn review-cell', text: (item.flagged ? '⚑ ' : '') + 'Cell ' + item.label + ' · ' + (item.value || 'No label'),
          onclick: () => inspectReviewedCell(layer, item.label).catch(error => message(error.message, true))});
        button.disabled = !!item.missing;
        reviewList.appendChild(button);
      });
      if (!result.items.length) reviewList.textContent = (layer.reviewMode || 'recent') === 'recent' ?
        'No recent viewer edits. Choose All assigned to review existing labels.' : 'No cells in this list.';
      previous.disabled = result.offset === 0;
      next.disabled = result.offset + result.items.length >= result.total;
      reviewList.appendChild(el('div', {class: 'hint', text: Number(result.total).toLocaleString() + ' cells in this list'}));
    } catch (error) { reviewList.textContent = error.message; }
  };
  reviewBody.appendChild(el('div', {class: 'row'}, [mode,
    help('Click a cell to return to it. Mark doubtful cells as Needs review; this keeps their phenotype unchanged.')]));
  reviewBody.appendChild(reviewList);
  reviewBody.appendChild(el('div', {class: 'row'}, [previous, next]));
  review.appendChild(reviewBody);
  review.addEventListener('toggle', () => { layer.reviewOpen = review.open; if (review.open) layer.refreshReview(); });
  if (review.open) layer.refreshReview();
  body.appendChild(review);
}

// Refresh image fills, masks and legends when the shared phenotype palette changes.
document.addEventListener('spacemap-phenotype-presets', () => {
  state.layers.filter(layer => layer.type === 'cells').forEach(layer => { layer.legendRequest = null; bump(layer); });
});
