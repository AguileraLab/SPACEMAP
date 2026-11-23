/* SPACEMAP viewer, part 11 of 12: colour columns, legends and the labelling card.
 * The viewer's parts share their names and load in order (see index.html). */
"use strict";

async function columnsFor(layer) {
  const table=layer.table, revision=layer.columnRevision||0;
  if (!layer.columns || layer.columnsTable!==table) {
    const result = await getJSON("/cells/" + layer.token + "/columns?t=" + encodeURIComponent(table || ""));
    if(layer.table!==table || (layer.columnRevision||0)!==revision)return columnsFor(layer);
    layer.columns = result.columns || [];
    layer.columnsTable=table;
    layer.columnKinds = result.kinds || {};
  }
  return layer.columns.filter(name=>!(layer.stagedRemovedColumns||[]).includes(name));
}

function invalidateCellColumns(layer) {
  layer.columnRevision=(layer.columnRevision||0)+1;
  layer.columns=null;layer.columnKinds=null;layer.referenceOptionsRequest=null;
}

function sortColumns(columns) {
  const channels = new Set(((state.base && state.base.info.channels) || []).map(String));
  const priority = (c) => (/pheno|cluster|consensus|ref|manual|neigh|^cn|zone|bed|class/i.test(c) ? 0 : channels.has(c) ? 1 : 2);
  return columns.slice().sort((a, b) => priority(a) - priority(b) || a.localeCompare(b));
}

function applyAnnotationIdentity(layer, display) {
  if(!display)return;
  Object.assign(layer,display);
  if(!display.name)return;
  // Repair older saved views too, without replacing their geometry or controls.
  for(const branch of document.querySelectorAll('[data-annotation-token]'))if(branch.dataset.annotationToken===layer.token){
    branch.dataset.annotationSource=layer.name;
    const title=branch.querySelector(':scope > summary .name');if(title)title.textContent=layer.name;
    const origin=branch.querySelector(':scope > summary .annotation-source-kind');if(origin)origin.textContent='Copy';
  }
}
function annotationLegend(layer) {
  if (!layer.legendRequest || layer.legendToken !== layer.token) {
    const token = layer.token, controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    // Class metadata is independent of tile rendering. A transparent, removed
    // or stalled cell layer must not leave the source controls waiting forever.
    const request = Promise.resolve().then(() => {
      if (!state.layers.includes(layer) && !state.layers.some(image => image.thresholdAnnotations?.includes(layer))) return { items: [] };
      return getJSON("/annotations/" + token + "/legend", {signal:controller.signal}).then(found=>{
        if(layer.legendToken===token){applyAnnotationIdentity(layer,found.display);layer.annotationLegend=found;finishAnnotationDisplayNames(layer,found);}
        return found;
      });
    }).catch((error) => {
      if (layer.legendRequest === request) layer.legendRequest = null;
      if (controller.signal.aborted) throw new Error("The annotation classes took too long to load.");
      throw error;
    }).finally(() => clearTimeout(timeout));
    layer.legendToken = token;
    layer.legendRequest = request;
  }
  return layer.legendRequest;
}

function annotationLegendError(box, error, retry) {
  box.replaceChildren(el('div', {class:'row'}, [el('span', {text:error.message}),
    el('button', {class:'btn small', text:'Retry', onclick:retry})]));
}

function legendInto(layer, body, path, onKind) {
  const box = el("div", { class: "legend", text: "Loading legend..." });
  body.appendChild(box);
  const hide = Array.from(layer.hidden).join("\u001f");
  const request = layer.type === "annotations" ? annotationLegend(layer) :
    getJSON(path + (path.includes("?") ? "&" : "?") + "hide=" + encodeURIComponent(hide));
  request.then((legend) => {
    if(!box.isConnected)return;
    box.textContent = "";
    if (onKind) onKind(legend.kind);
    if (legend.kind === "numeric") {
      box.appendChild(el("div", { class: "gradient", style: "background: linear-gradient(90deg," + gradientStops(legend.colormap) + ")" }));
      box.appendChild(el("div", { class: "range" }, [el("span", { text: Number(legend.min).toFixed(1) }), el("span", { text: Number(legend.max).toFixed(1) })]));
      return;
    }
    if((state.studio||state.reviewer) && layer.type==='cells') { cellGroupsLegend(layer,box,legend); return; }
    const syncData = () => {
      if (layer.type !== 'cells' || layer.table !== layer.combinedTable) return;
      const filtered = (legend.items || []).some(item => layer.hidden.has(item.value));
      const values = filtered ? (legend.items || []).filter(item => !layer.hidden.has(item.value)).map(item => item.value) : [];
      document.dispatchEvent(new CustomEvent('spacemap-groups-picked',
        {detail:{layer,column:layer.color_by,values,filtered}}));
    };
    if ((legend.items || []).length > 1) {
      const all = allBox("All", (on) => {
        claimCellView(layer,"Viewer groups");
        layer.hidden = on ? new Set() : new Set((legend.items || []).map((i) => i.value));
        bump(layer);
        syncData();
      });
      all.sync((legend.items || []).map((i) => !layer.hidden.has(i.value)));
      box.appendChild(all.row);
    }
    (legend.items || []).forEach((item) => {
      const name = cellGroupAppearance(layer)?.names?.[item.value];
      const row = el("div", { class: "item" + (layer.hidden.has(item.value) ? " hidden" : ""), title: "Click to show or hide" }, [
        el("span", { class: "swatch", style: "background:" + item.color }),
        el("span", { text: name && name !== item.value ? name + ' (' + item.value + ')' : item.value }),
        el("span", { class: "count", text: Number(item.count).toLocaleString() }),
      ]);
      row.addEventListener("click", () => {
        claimCellView(layer,"Viewer groups");
        if (layer.hidden.has(item.value)) layer.hidden.delete(item.value); else layer.hidden.add(item.value);
        bump(layer);
        syncData();
      });
      box.appendChild(row);
    });
    if (!(legend.items || []).length) box.appendChild(el("div", { class: "hint", text: "No values for this ROI." }));
  }).catch((error) => {
    if (!box.isConnected) return;
    if (layer.type === 'annotations') annotationLegendError(box, error, () => {
      box.remove(); legendInto(layer, body, path, onKind);
    });
    else box.textContent = error.message;
  });
}

// Search and select-all affect only matching groups; colours remain per table/column.
function cellGroupsLegend(layer,box,legend) {
  const items=legend.items||[],key=JSON.stringify([layer.table,layer.color_by]);
  if(!items.length){box.append(el('div',{class:'hint',text:'No values for this image.'}));return;}
  const searches=layer.legendSearch||(layer.legendSearch={});
  const search=el('input',{type:'search',value:searches[key]||'',placeholder:'Find a group…','aria-label':'Find cell groups'});
  const matched=()=>items.filter(i=>String(i.value).toLocaleLowerCase().includes(search.value.toLocaleLowerCase()));
  const syncData=()=>{
    const filtered=items.some(i=>layer.hidden.has(i.value));
    document.dispatchEvent(new CustomEvent('spacemap-groups-picked',{detail:{layer,column:layer.color_by,
      values:filtered?items.filter(i=>!layer.hidden.has(i.value)).map(i=>i.value):[],filtered}}));
  };
  const all=allBox('All matching',on=>{
    claimCellView(layer,'Viewer groups');matched().forEach(i=>{if(on)layer.hidden.delete(i.value);else layer.hidden.add(i.value);});bump(layer);syncData();
  });
  const list=el('div',{class:'cell-group-list'}),rows=[];
  items.forEach(item=>{
    const name=cellGroupAppearance(layer)?.names?.[item.value];
    const check=el('input',{type:'checkbox','aria-label':'Show group '+item.value});check.checked=!layer.hidden.has(item.value);
    const colour=el('input',{type:'color',value:item.color,'aria-label':'Colour for '+item.value,class:'swatch'});
    colour.onclick=e=>e.stopPropagation();colour.onchange=()=>{
      const maps=layer.groupColours||(layer.groupColours={});(maps[key]||(maps[key]={}))[item.value]=colour.value;bump(layer);
    };
    const row=el('div',{class:'item'+(check.checked?'':' hidden')},[check,colour,
      el('span',{class:'cell-group-name',text:name&&name!==item.value?name+' ('+item.value+')':item.value,title:String(item.value)}),
      el('span',{class:'count',text:Number(item.count).toLocaleString()})]);
    const toggle=()=>{claimCellView(layer,'Viewer groups');if(check.checked)layer.hidden.delete(item.value);else layer.hidden.add(item.value);bump(layer);syncData();};
    check.onclick=e=>e.stopPropagation();check.onchange=toggle;
    row.onclick=()=>{check.checked=!check.checked;toggle();};list.append(row);rows.push([item,row]);
  });
  const sync=()=>{const matches=new Set(matched());rows.forEach(([item,row])=>row.hidden=!matches.has(item));all.sync([...matches].map(i=>!layer.hidden.has(i.value)));};
  search.oninput=()=>{searches[key]=search.value;sync();};sync();box.append(el('div',{class:'row cell-group-search'},[all.row,search]),list);
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
      layer.columns=null;layer.referenceOptionsRequest=null;rememberReferenceOverlay(layer,column);
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
  box.append(el('button',{class:'ctx-action',text:'Flag for review',onclick:async()=>{if(!info||!current())return;closeContextMenu();await window.SpacemapReview.flagCells({table:info.table,column,selection:{cells:labels,roi:dataLayerRoi(layer)}});}}));
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

function referencePhenotypeSwitches() {
  const box=el('details',{class:'reference-phenotype-switches studio-options'});
  const title=el('summary',{},[el('span',{text:'Phenotypes · Train / Predict'}),help(
    'Project-wide switches for Manual reference models and MR-ML. Train off: leave this label out of future training, including saved examples. Predict off: an existing model cannot choose this label. Remaining confidence scores are not increased. Manual labels stay unchanged. The Refined low-confidence label (for example Others) is separate and can still be used. Existing predictions stay as saved until you rerun.')]);
  const search=el('input',{type:'search',placeholder:'Find a phenotype…','aria-label':'Find model phenotype'});
  const status=el('div',{class:'hint',role:'status'}), body=el('tbody');
  const head=el('tr',{},[el('th',{text:'Phenotype'})]);
  let setup=null, extra=[], legacyTraining=[], busy=false, sequence=0, saveError=null;
  const route=()=>manualReferenceRoute().replace('/manual_reference','/reference_setup');
  const keys=['excluded_training_phenotypes','excluded_prediction_phenotypes'];
  const headers=keys.map((key,i)=>{
    const label=i?'Predict':'Train', all=el('input',{type:'checkbox','aria-label':label+' all matching phenotypes'});
    head.append(el('th',{},[el('label',{class:'inline-check'},[all,el('span',{text:label})])]));
    all.onchange=()=>change(key,visible(),all.checked);return all;
  });
  const names=()=>[...new Set([...extra,...(setup?.models||[]).flatMap(m=>m.classes||[]),...keys.flatMap(k=>setup?.[k]||[])])].filter(Boolean).sort((a,b)=>a.localeCompare(b));
  const visible=()=>names().filter(n=>n.toLocaleLowerCase().includes(search.value.toLocaleLowerCase()));
  const excluded=key=>key==='excluded_training_phenotypes'&&!setup?.phenotype_policy_configured?legacyTraining:(setup?.[key]||[]);
  const blocked=(key,name)=>excluded(key).some(v=>v.toLocaleLowerCase()===name.toLocaleLowerCase());
  const draw=()=>{
    const shown=visible();body.replaceChildren();
    shown.forEach(name=>{
      const row=el('tr',{},[el('td',{text:name})]);
      keys.forEach((key,i)=>{
        const input=el('input',{type:'checkbox','aria-label':(i?'Predict ':'Train ')+name});
        input.checked=!blocked(key,name);input.disabled=busy||!setup;
        input.onchange=()=>change(key,[name],input.checked);row.append(el('td',{},[input]));
      });body.append(row);
    });
    headers.forEach((input,i)=>{const n=shown.filter(v=>!blocked(keys[i],v)).length;
      input.checked=shown.length>0&&n===shown.length;input.indeterminate=n>0&&n<shown.length;input.disabled=busy||!setup||!shown.length;});
  };
  const change=async(key,changed,on)=>{
    if(busy||!setup)return;
    const selected=new Set(changed.map(n=>n.toLocaleLowerCase()));
    const value=excluded(key).filter(n=>!selected.has(n.toLocaleLowerCase()));
    if(!on)value.push(...changed);
    let finish;box.pending=new Promise(resolve=>{finish=resolve;});
    busy=true;saveError=null;status.textContent='Saving…';draw();
    try{
      const result=await postJSON(route(),{action:'phenotypes',...Object.fromEntries(keys.map(k=>[k,excluded(k)])),[key]:value});
      box.apply(result);document.dispatchEvent(new CustomEvent('spacemap-reference-setup',{detail:result}));
      if(result.auto_reference)document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:result.auto_reference}));
      status.textContent='Saved';
    }catch(error){saveError=error;status.textContent=error.message;}finally{busy=false;draw();finish();}
  };
  box.apply=value=>{++sequence;setup=value;draw();};
  box.setPhenotypes=value=>{extra=value.map(v=>String(v));draw();};
  box.setLegacyTraining=value=>{legacyTraining=value;draw();};
  box.flush=async()=>{await box.pending;if(saveError)throw saveError;};
  box.refresh=async()=>{const token=++sequence;try{const value=await getJSON(route());if(token===sequence)box.apply(value);}
    catch(error){if(token===sequence)status.textContent=error.message;}};
  box.addEventListener('toggle',()=>{if(box.open&&!setup)box.refresh();});
  search.oninput=draw;
  box.append(title,search,el('div',{class:'reference-phenotype-scroll'},[
    el('table',{class:'reference-phenotype-table'},[el('thead',{},[head]),body])]),status);
  return box;
}

function referenceSetupControls(layer) {
  const box=el('div',{class:'reference-setup'}), status=el('div',{class:'hint',role:'status'});
  const choice=el('select',{'aria-label':'Phenotyping model to load'});
  const percent=el('input',{type:'number',min:'50',max:'90',step:'1',value:'70','aria-label':'Training percentage'});
  const name=el('input',{type:'text',maxlength:'120',placeholder:'Automatic name','aria-label':'New model name'});
  const replay=el('input',{type:'checkbox','aria-label':'Include saved training examples'});
  const remainder=el('span',{class:'muted'}), info=el('div',{class:'hint'});
  const warning=el('div',{class:'hint',role:'status'});
  const path=el('input',{'aria-label':'Saved phenotyping model file',readonly:'',type:'text'});
  const storage=el('div',{class:'hint reference-model-storage',role:'status'});
  const advanced=el('details',{class:'reference-training-details'},[el('summary',{text:'Details & options'})]);
  const copy=el('button',{class:'btn small',text:'Copy path',onclick:async()=>{
    try{await navigator.clipboard.writeText(path.value);status.textContent='Path copied. Copy this whole folder to share the model.';}
    catch(_){path.focus();path.select();status.textContent='Press Ctrl+C to copy the path.';}
  }});
  const files=field('Model folder',el('div',{class:'inline'},[path,copy]),'Copy this whole folder to share the model, including its training examples.');
  const route=()=>manualReferenceRoute().replace('/manual_reference','/reference_setup');
  let setup, refreshSequence=0;
  box.phenotypes=referencePhenotypeSwitches();
  const selected=()=>setup?.models.find(model=>model.key===choice.value);
  const describe=()=>{
    const item=selected(), bits=item?[item.classes.length?item.classes.length+' phenotypes':'Phenotypes listed after Use',item.features+' features']:['New Random Forest + Extra Trees'];
    if(item?.reusable_examples>0)bits.push(item.reusable_examples.toLocaleString()+' saved examples');
    if(item&&!item.compatible)bits.push(item.missing.length===item.features?'No shared features':'Needs all trained features; retrain to use here');
    else if(item?.missing.length)bits.push(item.missing.length+' features missing');
    if(item?.unused_features?.length)bits.push('Uses shared features');
    if(item?.needs_examples?.length)bits.push('New labels skipped (need 5 cells): '+item.needs_examples.join(', '));
    info.textContent=bits.join(' · ');
    info.title=item?.classes.join(', ')||'';
    warning.textContent=item&&!item.compatible?'Model unavailable with these features. See Details & options.':
      item?.needs_examples?.length?'Need 5 labels each: '+item.needs_examples.join(', '):'';
    const loaded=setup?.models.find(model=>model.id===setup.base_model_id);
    path.value=loaded?.shared_path||loaded?.path||'';path.title=path.value;files.hidden=!loaded;
    share.hidden=!loaded||!!loaded.shared_path;share.disabled=!loaded;
    storage.textContent=loaded?.shared_path?'Saved in Models folder · '+formatModelTime(loaded.trained):
      loaded?'Loaded model · new versions save to Models folder':'New versions save to Models folder';
    storage.title=loaded?.shared_path||setup?.folder||'';
    metrics.textContent=loaded?.validation_held_out ? loaded.training_cells+' training / '+loaded.validation_cells+' validation'+
      (Number.isFinite(loaded.f1)?' · Validation F1 '+loaded.f1.toFixed(3):'') : '';
    classes.textContent=loaded?.classes.length?'Phenotypes: '+loaded.classes.join(', '):'';
    replay.disabled=!setup?.base_model_id;
  };
  box.apply=found=>{
    ++refreshSequence;
    setup=found;
    box.phenotypes.apply(found);
    choice.replaceChildren(el('option',{value:'',text:'New model · Random Forest + Extra Trees'}),...found.models.map(item=>el('option',{
      value:item.key,text:referenceModelLabel(item)})));
    choice.value=found.models.find(item=>item.id===found.base_model_id)?.key||'';
    percent.value=found.training_percent;remainder.textContent='% training · '+(100-found.training_percent)+'% validation';
    if(document.activeElement!==name)name.value=found.model_name||'';
    replay.checked=found.include_saved_examples!==false;
    box.dataset.modelId=found.base_model_id||'';describe();
    box.closest('.manual-reference-card')?.describeTraining?.();
  };
  const save=async body=>{
    const found=await postJSON(route(),{...body,timezone:Intl.DateTimeFormat().resolvedOptions().timeZone});box.apply(found);
    document.dispatchEvent(new CustomEvent('spacemap-reference-setup',{detail:found}));
    if(found.auto_reference)document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:found.auto_reference}));
    box.closest('.manual-reference-card')?.describeTraining?.();
    return found;
  };
  const load=el('button',{class:'btn small',text:'Use',onclick:async()=>{
    load.disabled=true;
    try{const found=await save({action:'load',key:choice.value});const item=found.models.find(model=>model.id===found.base_model_id);
      status.textContent='';
      if(layer){layer.suggestionMode='saved_reference';layer.suggestionModelId=item?.id||null;layer.loadedSetupModelId=item?.id||null;layer.suggestionsOn=!!item?.compatible;
        rememberReferenceChoice(layer);saveView();
        if(state.inspectedCell?.layer===layer)requestCellProfile(layer,state.inspectedCell.result,++state.cardSeq);}
      await box.closest('.manual-reference-card')?.refreshModels?.();
    }catch(error){status.textContent=error.message;}finally{load.disabled=false;}
  }});
  choice.onchange=()=>{describe();status.textContent='Press Use to activate this model.';};
  box.saveTraining=async()=>{
    await box.phenotypes.flush();
    if(!setup)return;
    const active=setup.models.find(item=>item.id===setup.base_model_id)?.key||'';
    if(choice.value!==active)throw Error('Press Use to activate the selected model first.');
    if(name.value.trim()!==setup.model_name || replay.checked!==setup.include_saved_examples || setup.model_timezone!==Intl.DateTimeFormat().resolvedOptions().timeZone)
      await save({action:'training',model_name:name.value,include_saved_examples:replay.checked});
  };
  name.onchange=replay.onchange=()=>box.saveTraining().catch(error=>{status.textContent=error.message;});
  percent.onchange=async()=>{
    percent.disabled=true;
    try{await save({action:'split',training_percent:Number(percent.value)});status.textContent='';}
    catch(error){status.textContent=error.message;percent.value=setup?.training_percent||70;}
    finally{percent.disabled=false;}
  };
  const share=el('button',{class:'btn small',text:'Save a copy in Models folder',disabled:true,onclick:async()=>{
    share.disabled=true;
    try{await save({action:'share'});status.textContent='Copied to '+setup.folder;}
    catch(error){status.textContent=error.message;describe();}
  }});
  const metrics=el('div',{class:'hint'}),classes=el('div',{class:'hint'});
  box.refresh=async()=>{const token=++refreshSequence, found=await getJSON(route());if(token===refreshSequence)box.apply(found);};
  advanced.append(info,storage,files,field('Version name',name,'Optional. Each update gets its own date and time automatically.'),
    el('label',{class:'inline-check'},[replay,el('span',{text:'Keep previous training examples'}),help('On: old examples plus your new labels. Off: only the chosen images. Combined training uses features measured in both sets. Validation labels stay out.')]),metrics,classes,share);
  box.advanced=advanced;
  box.split=field('Split',el('div',{class:'inline reference-split'},[percent,remainder]),'This share of your labelled cells trains the model. The rest are reserved for assessment.');
  box.trainingDescription=()=>setup?.base_model_id&&replay.checked?'Previous examples + new labels':'New labels only';
  box.append(field('Model',el('div',{class:'inline'},[choice,load]),'Use loads a model for suggestions and further training. Each training run saves a dated version in Models and keeps the original. Only load models from people you trust.'),warning,status);
  box.refresh().catch(error=>{status.textContent=error.message;});
  return box;
}

document.addEventListener('spacemap-reference-setup',event=>{
  document.querySelectorAll('.reference-setup').forEach(box=>box.apply?.(event.detail));
  document.querySelectorAll('.reference-phenotype-switches').forEach(box=>box.apply?.(event.detail));
});

function referenceTrainingControls(layer, trainingColumn, features, checked, setup, autoControls) {
  const box=el('div',{class:'reference-training-actions'});
  const status=el('div',{class:'hint',role:'status','data-reference-training-status':''});
  const route=()=>manualReferenceRoute().replace('/manual_reference','/reference_collection');
  const context=route();
  let count=0, loading=true, busy=false, sequence=0;
  const sync=()=>{
    current.disabled=busy||!layer;
    current.title=layer?'Train on the open image’s labels.':'Open an image to train on its labels.';
    selected.disabled=busy||loading||!count;
    selected.dataset.imageCount=count;
    selected.title=loading?'Loading image selection…':count?'Train on '+count+' selected images.':'Select images in the table above.';
  };
  const load=async()=>{
    const request=++sequence;
    const found=await getJSON(route());
    if(request!==sequence)return;
    count=found.selected_images.length;loading=false;sync();
  };
  document.addEventListener('spacemap-phenotyping-images',()=>{if(box.isConnected)load().catch(error=>{status.textContent=error.message;});});
  const poll=async run=>{
    if(!box.isConnected||state.referenceTraining!==run)return;
    if(!run.running){busy=false;status.textContent=run.message;sync();return;}
    if(!run.id){setTimeout(()=>poll(run),1200);return;}
    try{
      const {job}=await getJSON(context.replace('/reference_collection','/jobs/'+run.id));
      if(!box.isConnected||state.referenceTraining!==run)return;
      if(['starting','running','stopping'].includes(job.status)){
        status.textContent=run.message;status.title=job.message||'';
        setTimeout(()=>poll(run),1200);return;
      }
      busy=false;run.running=false;status.title='';
      const exportError=job.result?.models?.find(model=>model.export_error)?.export_error;
      run.message=job.status==='done'?(exportError?'Trained, but saving failed: '+exportError:'Training complete. Saved in Models.'):
        job.error||job.message||'Training stopped. See Jobs.';
      status.textContent=run.message;sync();
      if(job.status==='done')box.closest('.manual-reference-card')?.refreshModels?.(job.result);
    }catch(error){
      status.textContent='Cannot check training progress. See Jobs.';status.title=error.message;
      setTimeout(()=>poll(run),3000);
    }
  };
  const start=async scope=>{
    busy=true;sync();status.textContent=scope==='current'?'Training current image…':'Training selected images…';status.title='';
    const run=state.referenceTraining={context,id:null,running:true,message:status.textContent};
    try{
      await state.studio?.flushImageSelection?.();await flushLabels();await features.saveIfChanged();await setup.saveTraining();
      const settings={reference_column:trainingColumn(),checked_only:checked.checked};
      const result=scope==='current'?await runCommand('train_reference_model',settings,{confirmed:true}):
        await postJSON(route(),{action:'train',...settings,request_id:newSaveId()});
      const job=result.job||result.data?.job;
      state.studio?.showResult(scope==='current'?result:{status:'done',message:result.message,data:{job}});
      if(!job?.id)throw Error(result.message||'Training did not start. See Jobs.');
      run.id=job.id;
      poll(run);
    }catch(error){busy=false;run.running=false;run.message=error.message;status.textContent=run.message;sync();}
  };
  const current=el('button',{class:'btn','data-reference-train':'',text:'Current image',onclick:()=>start('current')});
  const selected=el('button',{class:'btn','data-reference-train-selected':'',text:'Selected images',onclick:()=>start('selected')});
  const trainingHelp=help('');
  box.describe=()=>{
    const text='Labels: '+trainingColumn()+'. '+setup.trainingDescription()+'. Current image uses the open image. Selected images uses the checkboxes in the table above. A new phenotype needs 5 usable labels.';
    trainingHelp.setAttribute('data-tip',text);trainingHelp.setAttribute('aria-label',text);
  };
  box.append(field('Train on',el('div',{class:'inline reference-training-buttons'},[current,selected])),
    field('',autoControls),status);
  box.firstChild.append(trainingHelp);
  box.describe();sync();
  queueMicrotask(()=>{
    load().catch(error=>{status.textContent=error.message;});
    const run=state.referenceTraining;
    if(run?.context===context){busy=run.running;status.textContent=run.message;sync();if(run.running)poll(run);}
  });
  return box;
}

function manualReferenceSelections(layer) {
  const items = [], picked = state.manualReferencePick;
  // The inspector may belong to a newly linked/rebuilt layer, while this card
  // was created before that layer loaded. Segmentation IDs stay the same when
  // the visible measurement table changes; use the checked phenotyping link.
  const selectedLayer=picked && state.layers.find(item=>item.type==='cells'&&item.token===picked.layer?.token);
  if (selectedLayer?.combinedTable && picked.labels.length &&
      (!picked.targetTable || picked.targetTable===dataView.combined?.path))
    items.push({id:'viewer', count:picked.labels.length, text:'Viewer cells', table:dataView.combined?.path,
      selection:{cells:picked.labels, roi:selectedLayer.table_roles?.[selectedLayer.combinedTable]?.roi || selectedLayer.roi}});
  if (dataView.source?.combined) {
    if (dataView.selected.size) items.push({id:'data', count:dataView.selected.size, text:'Selected Data rows', table:dataView.combined?.path,
      selection:{rows:Array.from(dataView.selected)}});
    if (!dataView.loading && dataView.queryReady && dataView.result?.matched) {
      const body = dataView.resultBody, filters = body.filters.slice();
      if (body.rows.this_image && body.rows.column && body.rows.value !== undefined) filters.push({col:body.rows.column,values:[String(body.rows.value)]});
      items.push({id:'filtered', count:dataView.result.matched, text:'Filtered Data rows', table:dataView.combined?.path,
        selection:dataView.flowSelection?{flow_token:dataView.flowSelection,view_filters:filters}:{filters}});
    }
  }
  if (state.spec?.selection?.id && state.spec.selection.count) items.push({id:'rule', text:'Marker rule', count:state.spec.selection.count,
    selection:{token:state.spec.selection.id}});
  return items;
}

document.addEventListener('spacemap-cells-picked', event => {
  const {layer, labels, fromData} = event.detail || {};
  if (!fromData) {
    state.manualReferencePick = {layer,table:layer?.table,targetTable:dataView.combined?.path,labels:labels || []};
    state.manualReferenceSource = 'viewer';
  }
  document.querySelectorAll('.manual-reference-card').forEach(card => card.refreshSelection?.());
});

document.addEventListener('spacemap-reference-updated',event=>{
  const detail=event.detail || {};
  document.querySelectorAll('.manual-reference-card').forEach(card=>{
    if(!card.applySavedLabels?.(detail)&&!detail.localRefresh){
      clearTimeout(card.labelSummaryTimer);
      card.labelSummaryTimer=setTimeout(()=>{if(card.isConnected)card.refreshReference?.();},800);
    }
  });
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
      setupControls.phenotypes.setPhenotypes([...(found.phenotypes||[]),...(found.counts.groups||[]).map(([name])=>name)]);
      const learned=learn.value;
      learn.replaceChildren(el('option',{value:'',text:'Manual labels · '+column.value.trim()}),
        el('optgroup',{label:'Phenotype labels'},(found.training_columns||found.columns).filter(value=>value!==column.value.trim()).map(value=>el('option',{value,text:value}))),
        ...((found.other_training_columns||[]).length ? [el('optgroup',{label:'Other table labels — check contents'},found.other_training_columns.map(value=>el('option',{
          value,text:value+(value==='bed'?' · annotation region':''),title:'The model will learn the categories in this column.'})))] : []));
      if(Array.from(learn.options).some(option=>option.value===learned))learn.value=learned;
      card.describeTraining?.();
      columnList.replaceChildren(...found.references.map(value => el('option',{value})));
      names.replaceChildren(...found.phenotypes.map(value => el('option',{value,'data-color':found.preset_colours?.[value] || ''})));
      columnList.refreshChoices?.(); names.refreshChoices?.();
      counts.replaceChildren(...(found.counts.groups || []).map(([label,n]) => el('button',{class:'btn small count-chip',text:label+' · '+n.toLocaleString(),onclick:()=>{name.value=label;state.labelName=label;}})));
      if (!counts.childNodes.length) counts.textContent = 'No manual labels yet.';
      save.disabled = found.writing.state === 'writing' || !(found.pending_labels || found.pending_columns.length);
      save.textContent = found.writing.state === 'writing' ? 'Writing…' : 'Write into table';
      card.refreshSelection();
      if (found.writing.state === 'writing') setTimeout(update,1500);
    } catch(error) { status.textContent=error.message; }
  };
  const refresh = async () => {
    if (layer) {
      layer.columns=null;layer.referenceOptionsRequest=null;rememberReferenceOverlay(layer,state.labelColumn);
      bump(layer);
    }
    if (dataView.holder?.isConnected) { dataView.columns=null; await dataLoad(true); }
    document.dispatchEvent(new CustomEvent('spacemap-reference-updated',{detail:{column:state.labelColumn,localRefresh:true}}));
    await update();
  };
  card.refreshReference=update;
  card.applySavedLabels=detail=>{
    if(!info || detail.table!==info.table || detail.column!==column.value.trim() || !detail.labels)return false;
    if((info.counts.groups||[]).length>=200)return false;
    ++loadSequence; // an older in-flight summary must not replace acknowledged edits
    const groups=new Map(info.counts.groups || []);
    const clean=value=>value==null?'':String(value).trim();
    detail.labels.forEach(label=>{
      const before=clean(detail.previous[String(label)]),after=clean(detail.values[String(label)]);
      if(before===after)return;
      if(before)groups.set(before,Math.max(0,(groups.get(before)||0)-1));
      if(after)groups.set(after,(groups.get(after)||0)+1);
    });
    info.counts.groups=[...groups].filter(([,count])=>count>0).sort((a,b)=>b[1]-a[1]);
    setupControls.phenotypes.setPhenotypes([...(info.phenotypes||[]),...info.counts.groups.map(([name])=>name)]);
    counts.replaceChildren(...info.counts.groups.map(([label,n])=>el('button',{class:'btn small count-chip',text:label+' · '+n.toLocaleString(),onclick:()=>{name.value=label;state.labelName=label;}})));
    if(!counts.childNodes.length)counts.textContent='No manual labels yet.';
    for(const [label] of info.counts.groups)if(![...names.options].some(option=>option.value===label))names.append(el('option',{value:label}));
    if(!info.references.includes(detail.column)){
      info.references.push(detail.column);columnList.append(el('option',{value:detail.column}));
    }
    save.disabled=info.writing.state==='writing';
    return true;
  };
  column.onchange = () => { state.labelColumn=column.value.trim(); learn.value=''; card.describeTraining?.(); if(layer) layer.learnColumn=state.labelColumn; update(); if(auto.checked)saveAuto(); };
  const create = el('button',{class:'btn small',text:'Create', 'aria-label':'Create reference column',onclick:async()=>{
    create.disabled=true;
    try { state.labelColumn=column.value.trim(); await postJSON(manualReferenceRoute(),{action:'create',column:state.labelColumn}); status.textContent='Column ready. Select cells and assign a phenotype.'; await refresh(); }
    catch(error){status.textContent=error.message;}finally{create.disabled=false;}
  }});
  const show=el('button',{class:'btn small',text:'Show',disabled:layer?null:'',title:'Show assigned colours and keep unassigned cell outlines',onclick:()=>{if(layer){state.labelColumn=column.value.trim();rememberReferenceOverlay(layer,state.labelColumn);layer.referenceOverlayEnabled=true;setCellColour(layer,'');}}});
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
  const flag=el('button',{class:'btn small',text:'Flag for review',onclick:async()=>{const chosen=manualReferenceSelections(layer).find(item=>item.id===scope.value);if(chosen&&info){flag.disabled=true;try{await window.SpacemapReview.flagCells({table:info.table,column:column.value.trim(),selection:chosen.selection});}finally{card.refreshSelection();}}}});
  scope.onchange=()=>{state.manualReferenceSource=scope.value;card.refreshSelection();};
  card.refreshSelection=()=>{
    const selections=manualReferenceSelections(layer), previous=state.manualReferenceSource || scope.value;
    scope.replaceChildren(...selections.map(item=>el('option',{value:item.id,text:item.text+' · '+item.count.toLocaleString()})));
    scope.value=selections.some(item=>item.id===previous)?previous:selections.find(item=>item.id!=='filtered')?.id || '';
    if(!scope.value) scope.prepend(el('option',{value:'',text:'Select cells first'}));
    const chosen=selections.find(item=>item.id===scope.value);
    apply.disabled=clear.disabled=flag.disabled=busy || !info || !chosen;
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
    el('div',{class:'row wrap'},[apply,clear,flag,undoButton,save]),status);
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
    autoStatus.textContent=value.enabled&&value.state==='training'?'Auto-training current image…':'';
    autoStatus.title=value.message || '';
    if(value.enabled&&['waiting','error'].includes(value.state)&&/five|phenotype|usable|failed|stopped|missing|changed/i.test(value.message || ''))autoStatus.textContent=value.message;
  };
  const saveAuto=async()=>{
    autoBusy=true;auto.disabled=true;
    try{if(auto.checked){await flushLabels();await features.saveIfChanged();await setupControls.saveTraining();}const found=await postJSON(manualReferenceRoute().replace('/manual_reference','/auto_reference'),
      {enabled:auto.checked,column:trainingColumn(),checked_only:checked.checked});
      autoBusy=false;card.applyAuto(found);document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:found}));}
    catch(error){auto.checked=false;autoStatus.textContent=error.message;}
    finally{autoBusy=false;auto.disabled=false;}
  };
  auto.onchange=saveAuto;
  let phenotypeSequence=0;
  const loadTrainingPhenotypes=async()=>{
    const token=++phenotypeSequence;
    try{const found=await getJSON(manualReferenceRoute()+'?column='+encodeURIComponent(trainingColumn()));
      if(card.isConnected&&token===phenotypeSequence)setupControls.phenotypes.setPhenotypes(
        [...(found.phenotypes||[]),...(found.counts.groups||[]).map(([name])=>name)]);
    }catch(error){status.textContent=error.message;}
  };
  learn.onchange=()=>{loadTrainingPhenotypes();if(auto.checked)saveAuto();};
  checked.onchange=()=>{if(auto.checked)saveAuto();};
  const autoControls=el('label',{class:'inline-check'},[auto,el('span',{text:'Auto-train this image'}),
    help('Train this image after you pause labelling, at most once a minute. Uses the same features, split and saved-example setting as manual training.')]);
  getJSON(manualReferenceRoute().replace('/manual_reference','/auto_reference')).then(found=>{
    card.applyAuto(found);checked.checked=!!found.checked_only;
    // A previous Auto-train column must never override the visible manual
    // label column. An active auto run is rebound to the visible choice.
    if(found.enabled&&found.column!==trainingColumn())saveAuto();
    card.describeTraining?.();
    document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:found}));
  }).catch(error=>{autoStatus.textContent=error.message;});
  const predict=el('button',{class:'btn small',text:'Predict other cells…',onclick:()=>window.dispatchEvent(new CustomEvent('studio-reference-method',{
    detail:{method:'mr_ml',column:trainingColumn(),model_id:layer?.suggestionModelId}}))});
  learn.addEventListener('change',()=>{card.describeTraining?.();});
  const setupControls=referenceSetupControls(layer);
  const trainingControls=referenceTrainingControls(layer,trainingColumn,features,checked,setupControls,autoControls);
  card.describeTraining=()=>trainingControls.describe();
  setupControls.advanced.append(field('Training labels',learn,'Normally your manual label column above. Choose another column only to learn its category names.'),
    el('label',{class:'inline-check'},[checked,el('span',{text:'Use checked labels only'}),help('Leave off to use your manual labels. Turn on only if you also use Mark checked to approve labels. Cells marked Needs review are excluded.')]));
  setupControls.advanced.append(el('div',{class:'row'},[predict]));
  setupControls.phenotypes.addEventListener('toggle',()=>{if(setupControls.phenotypes.open)loadTrainingPhenotypes();});
  modelBody.append(setupControls,features,setupControls.phenotypes,setupControls.split,trainingControls,autoStatus,setupControls.advanced);
  const suggestions=el('div');modelBody.append(suggestions);
  card.refreshModels=async trained=>{
    if(!modelBox.open){modelBox.loaded=false;return;}
    modelBox.loaded=true;
    try{await setupControls.refresh();}catch(error){status.textContent=error.message;}
    if(trained?.models?.some(m=>m.export_error))status.textContent=trained.models.find(m=>m.export_error).export_error;
    suggestions.replaceChildren();
    if(layer){layer.suggestionMode='saved_reference';layer.learnColumn=column.value.trim();suggestionControls(layer,suggestions);}
  };
  modelBox.addEventListener('toggle',async()=>{
    state.manualReferenceModelsOpen=modelBox.open;
    if(modelBox.open&&!modelBox.loaded)await card.refreshModels();
  });
  modelBox.open=!!state.manualReferenceModelsOpen;
  body.append(modelBox);
  body.append(el('div',{class:'row'},[el('button',{class:'btn small',text:'Compare predictions (F1)…',onclick:()=>window.dispatchEvent(new CustomEvent('studio-reference-assessment'))}),
    help('Compare predicted phenotypes with your manual labels. Choose the methods and reference cells in Visualization.')]));
  body.append(modelReviewControls());
  if(layer)referenceReviewControls(layer,body);card.refreshSelection();queueMicrotask(update);
  return card;
}

function modelReviewControls() {
  // Viewer/model refreshes must not discard an open inspection or its job.
  const context=manualReferenceRoute();
  if(state.modelReviewPanel?.context===context)return state.modelReviewPanel.panel;
  const panel=el('details',{class:'reference-model-review'},[el('summary',{text:'Models review and edit'})]);
  state.modelReviewPanel={context,panel};
  const body=el('div'), choices=el('div',{class:'model-review-choices'}), results=el('div'), edits=el('div',{hidden:true});
  const status=el('p',{class:'hint',role:'status','data-model-review-status':''});
  const route=()=>manualReferenceRoute().replace('/manual_reference','/model_review');
  let checked=[], inspected=null, busy=false, loaded=false, folder=null;
  const name=el('input',{type:'text',placeholder:'Automatic name',maxlength:100,'aria-label':'New model name'});
  const features=el('div',{class:'model-review-features'}), groups=el('div');
  const featureBox=el('details',{},[el('summary',{text:'Features to keep'}),features,
    el('p',{class:'hint',text:'Untick a feature to leave it out of the new fit. Correct wrong measurements in the source table before training again.'})]);
  const groupBox=el('details',{},[el('summary',{text:'Group phenotypes (optional)'}),
    el('p',{class:'hint',text:'Give types the same output name to group them. Detailed types stay in the model.'}),groups]);
  const keep=el('input',{type:'checkbox',checked:true,'aria-label':'Keep detailed model predictions'});
  const buttons=[];
  const button=(text,action)=>{const b=el('button',{class:'btn small',text,onclick:()=>act(action)});buttons.push([b,action]);return b;};
  const sync=()=>buttons.forEach(([b,action])=>{
    b.disabled=busy || !checked.length || (action!=='inspect'&&!inspected) ||
      (action==='merge'&&!inspected?.can_merge) || (['copy','refit'].includes(action)&&checked.length!==1) ||
      (['refit','validate'].includes(action)&&!!inspected?.issues?.length);
  });
  const render=found=>{
    results.replaceChildren();
    const table=el('table',{class:'model-review-table'});
    table.append(el('thead',{},[el('tr',{},['Model','Inputs','Phenotypes','Training','Validation F1'].map(text=>el('th',{text})))]));
    const rows=el('tbody');table.append(rows);
    found.models.forEach(m=>{
      const more=el('details',{},[el('summary',{text:m.classes.length+' types'}),el('span',{text:m.classes.join(', ')})]);
      rows.append(el('tr',{},[el('td',{text:m.name+' · '+m.method,title:formatModelTime(m.trained)}),
        el('td',{text:String(m.features.length),title:m.features.join(', ')}),el('td',{},[more]),
        el('td',{text:Number(m.training_cells).toLocaleString()}),el('td',{text:m.validation?.cells?m.validation.f1.toFixed(2)+' · '+m.validation.cells+' cells':'Not checked'})]));
    });
    results.append(table);
    const overlap=el('details',{},[el('summary',{text:found.shared_features.length+' shared features · '+found.all_features.length+' total'}),
      el('p',{class:'hint',text:'Shared: '+(found.shared_features.join(', ')||'None')})]);
    found.models.forEach(m=>overlap.append(el('p',{class:'hint',text:m.name+' only: '+(m.unique_features.join(', ')||'None')})));
    results.append(overlap);
    if(found.models.length>1)results.append(el('p',{class:'hint',text:'Shared feature names must use comparable measurement scales.'}));
    found.issues.forEach(text=>results.append(el('p',{class:'hint',text})));
    if(found.validation_cells)results.append(el('p',{class:'hint',text:found.validation_cells+' saved validation cells never used by the selected models for training.'}));
    if(found.reused_validation_cells)results.append(el('p',{class:'hint',text:found.reused_validation_cells+' cells excluded from validation: another selected model used them for training.'}));
    if(found.duplicate_training_cells>0&&!found.issues.length)results.append(el('p',{class:'hint',text:found.duplicate_training_cells+' repeated examples counted once.'}));
  };
  const figures=found=>window.dispatchEvent(new CustomEvent('studio-model-figures',{detail:found}));
  const load=async()=>{
    const found=await getJSON(route());loaded=true;choices.replaceChildren();
    if(folder!==found.folder){checked=[];inspected=null;results.replaceChildren();edits.hidden=true;}
    folder=found.folder;
    checked=checked.filter(key=>found.models.some(m=>m.key===key));
    found.models.forEach(m=>{
      const input=el('input',{type:'checkbox','aria-label':'Review model '+(m.name||m.method),value:m.key});input.checked=checked.includes(m.key);
      input.onchange=()=>{checked=[...choices.querySelectorAll('input:checked')].map(b=>b.value);inspected=null;results.replaceChildren();edits.hidden=true;sync();};
      choices.append(el('label',{class:'inline-check'},[input,el('span',{text:(m.name||m.column)+' · '+m.method+' · '+formatModelTime(m.trained)})]));
    });
    if(!found.models.length)choices.textContent='No models in the Models folder.';
    sync();
  };
  const poll=async id=>{
    if(!panel.isConnected)return;
    try{
      const {job}=await getJSON(route().replace('/model_review','/jobs/'+id));
      if(['starting','running','stopping'].includes(job.status)){status.textContent=job.message||'Working…';setTimeout(()=>poll(id),1200);return;}
      busy=false;state.modelReviewJob=null;
      if(job.status!=='done')throw Error(job.error||job.message||'Model operation failed. See Jobs.');
      const found=job.result;
      if(found.models?.[0]?.validation){render(found);figures(found);status.textContent='Compared on the same saved validation cells. Check performance in your project, too.';}
      else {
        status.textContent=found.message||'Saved a new version in Models.';
        document.dispatchEvent(new CustomEvent('spacemap-reference-trained',{detail:job}));
        const setup=await getJSON(route().replace('/model_review','/reference_setup'));
        document.dispatchEvent(new CustomEvent('spacemap-reference-setup',{detail:setup}));
        if(found.review)figures(found.review);
        inspected=null;edits.hidden=true;await load();
      }
    }catch(error){busy=false;status.textContent=error.message;}finally{sync();}
  };
  const act=async action=>{
    busy=true;sync();status.textContent=action==='inspect'?'Reading models…':'Starting…';
    try{
      const mapping={};groups.querySelectorAll('input').forEach(input=>{if(input.value.trim()&&input.value.trim()!==input.dataset.phenotype)mapping[input.dataset.phenotype]=input.value.trim();});
      const found=await postJSON(route(),{action,keys:checked,name:name.value.trim(),phenotype_groups:mapping,
        features:action==='refit'||action==='merge'?[...features.querySelectorAll('input:checked')].map(i=>i.value):undefined,
        keep_detailed_predictions:keep.checked,timezone:Intl.DateTimeFormat().resolvedOptions().timeZone,request_id:newSaveId()});
      if(action==='inspect'){
        inspected=found;render(found);features.replaceChildren();groups.replaceChildren();
        found.merge_features.forEach(f=>features.append(el('label',{class:'inline-check'},[el('input',{type:'checkbox',
          checked:found.models.length!==1||found.models[0].features.includes(f),value:f,'aria-label':'Keep model feature '+f}),el('span',{text:f})])));
        const mapping=found.models.length===1?found.models[0].phenotype_groups:{};
        keep.checked=found.models.length!==1||found.models[0].keep_detailed_predictions!==false;
        [...new Set(found.models.flatMap(m=>m.classes))].sort().forEach(p=>groups.append(field(p,el('input',{
          type:'text',value:mapping[p]||'',placeholder:p,maxlength:60,'data-phenotype':p,'aria-label':'Group model phenotype '+p}))));
        edits.hidden=false;status.textContent='Review the heatmaps in Figures. Originals stay unchanged.';figures(found);busy=false;sync();
      }else{
        state.modelReviewJob=found.job.id;
        state.studio?.showResult({status:'done',message:found.message,data:{job:found.job}});
        poll(found.job.id);
      }
    }catch(error){busy=false;status.textContent=error.message;sync();}
  };
  edits.append(field('New model name',name),featureBox,groupBox,
    el('label',{class:'inline-check'},[keep,el('span',{text:'Keep detailed predictions in a second column'})]),
    el('div',{class:'row wrap'},[button('Save custom copy','copy'),button('Retrain selected model','refit'),button('Merge selected models','merge')]),
    el('p',{class:'hint',text:'Retrain and Merge use saved examples and the checked features. A dated version saves in Models.'}));
  body.append(el('p',{class:'hint',text:'Select models from the Models folder above.'}),choices,
    el('div',{class:'row wrap'},[button('Inspect models','inspect'),button('Validate models','validate'),
      el('button',{class:'btn small',text:'Refresh list',onclick:()=>load().catch(e=>status.textContent=e.message)})]),results,edits,status);
  panel.append(body);sync();
  panel.addEventListener('toggle',()=>{if(panel.open&&!loaded){load().catch(e=>status.textContent=e.message);if(state.modelReviewJob){busy=true;poll(state.modelReviewJob);}}});
  document.addEventListener('spacemap-reference-setup',event=>{if(event.detail?.folder!==folder){loaded=false;inspected=null;results.replaceChildren();edits.hidden=true;}if(panel.isConnected&&panel.open)load().catch(e=>status.textContent=e.message);});
  return panel;
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
    if (state.inspectedCell?.layer === layer) inspectReviewedCell(layer, state.inspectedCell.result.label);
  });
  const showColumn = el("button", { class: "btn small", type: "button", text: "Show", title: "Show assigned colours and keep unassigned cell outlines",
    onclick: () => { rememberReferenceOverlay(layer,state.labelColumn);layer.referenceOverlayEnabled=true;setCellColour(layer,''); } });
  body.appendChild(field("Column", el("div", { class: "inline" }, [referenceChoice(column, columnList, 'column'), showColumn]),
    "The column your labels are saved in. Click to see saved columns or type a new name. " +
    "Columns named ref, reference or manual hold manual labels; choosing another method's column asks first."));
  body.appendChild(referenceOverlayControl(layer));
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
