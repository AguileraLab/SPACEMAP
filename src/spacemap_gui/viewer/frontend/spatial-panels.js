/* Spatial analysis (Density, Triads, Feature screening, where each group is) and Visualization (Heatmap,
 * Proportions, UMAP, Method agreement) on the combined table. Every result is linked to the image: a click shows
 * the cells, a density map, a tissue outline or the triangles of a triad on the open image. */
window.createSpatialPanels = h => {
  'use strict';
  const {node, field, choice, help, section, chip, get, post, route} = h;
  const api = (action, body = {}) => post(route('/spatial/' + action), body);
  let infoRequest = null;
  let tlsDraft = null; // one unsaved boundary review, retained while browsing images/tabs
  const info = () => (infoRequest ||= api('info').catch(error => { infoRequest = null; throw error; }));
  const prefer = (options, wanted) => options.find(o => wanted.includes(o)) || options[0] || '';
  const columnPick = (options, value, label) => choice(options.map(c => [c, c]), value, label);
  const number = (value, label, step = 'any', min = '0') => node('input', {type:'number', value:String(value), step, min, 'aria-label':label, class:'studio-area-input'});
  const fmt = v => v === null || v === undefined || Number.isNaN(v) ? '–' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number(v.toPrecision(3)).toString();
  const paletteFor = async names => (await post(route('/colours'), {names})).mapping;


  // A job, followed here until it ends: its progress in *said*, then its result.
  async function runJob(kind, body, said) {
    const started = await api('start', {kind, ...body});
    const id = started.job?.id || started.job;
    h.refreshJobs?.();
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const jobs = (await get(route('/jobs'))).jobs || [];
      const job = jobs.find(item => item.id === id);
      if (!job) continue;
      if (job.status === 'done') { said.textContent = 'Done.'; return job.result; }
      if (['failed', 'cancelled'].includes(job.status)) throw Error(job.message || 'The analysis stopped.');
      said.textContent = job.message || 'Working…';
    }
  }
  const lastResult = async kind => {
    const jobs = (await get(route('/jobs'))).jobs || [];
    return jobs.find(job => job.kind === kind && job.status === 'done' && job.result)?.result || null;
  };
  const showOnImage = async (roi, then) => {
    if (roi && h.openRoi() !== roi) { await h.openTableImage(roi); await h.imageSettled?.(); }
    if (then) await then();
  };

  // Stacked bars: one bar per row (an image), a segment per group; a click opens the image showing that group.
  async function barsFigure({key, title, tip, rows, groups, counts, column, onPick}) {
    const palette = await paletteFor(groups);
    const {box, body} = section(title, {tip});
    const share = node('label', {class:'inline'}, [node('input', {type:'checkbox', checked:''}), node('span', {text:' Shares (%)'})]);
    const holder = node('div', {class:'spatial-bars'});
    const draw = () => {
      const percent = share.querySelector('input').checked;
      const largest = Math.max(...counts.map(r => r.reduce((a, b) => a + b, 0)), 1);
      holder.replaceChildren(...rows.map((row, i) => {
        const total = counts[i].reduce((a, b) => a + b, 0) || 1;
        const bar = node('div', {class:'spatial-bar', style:'width:' + (percent ? 100 : Math.max(2, 100 * total / largest)) + '%'},
          groups.map((group, j) => counts[i][j] ? node('span', {class:'spatial-segment', style:'flex:' + counts[i][j] + ';background:' + palette[group],
            title:group + ' · ' + counts[i][j].toLocaleString() + ' cells (' + (100 * counts[i][j] / total).toFixed(1) + '%) in ' + row + '. Click to show them.',
            onclick:() => onPick(row, group)}) : null).filter(Boolean));
        return node('div', {class:'spatial-bar-row'}, [node('button', {type:'button', class:'studio-item-name studio-file spatial-bar-label', text:row, title:'Open ' + row + ' coloured by ' + column,
          onclick:() => onPick(row, null)}), node('div', {class:'spatial-bar-track'}, [bar]), node('span', {class:'muted mono', text:total.toLocaleString()})]);
      }));
    };
    share.querySelector('input').onchange = draw;
    const legend = node('div', {class:'spatial-legend'}, groups.map(group => node('button', {type:'button', class:'spatial-key', title:'Show ' + group + ' on the open image',
      onclick:() => onPick(null, group)}, [node('span', {class:'split-marker-swatch', style:'background:' + palette[group]}), node('span', {text:group})])));
    body.append(node('div', {class:'row wrap'}, [share, help('Each bar is an image; each colour a group of ' + column + '. Click a colour to open that image showing only those cells; click a name to open it coloured by every group.')]), legend, holder);
    draw(); h.showFigure(key, box);
  }

  // ------------------------------------------------------------ where each group is (Neighbourhoods, any column)
  async function whereFigure(column, images, key = 'where', title) {
    const found = await api('composition', {column, images});
    await barsFigure({key, title:title || column + ' in each image', tip:'How many cells of each group each image has.', rows:found.images, groups:found.groups,
      counts:found.counts, column, onPick:(roi, group) => showOnImage(roi, () => h.showCellGroups(column, group ? [group] : [], found.groups))});
  }

  // ------------------------------------------------------------ TLS candidates
  async function tlsCandidates(current, pane, actionRow) {
    const [known, initial] = await Promise.all([api('info'), get(route('/spatial/tls_settings'))]);
    if (!pane.isConnected) return;
    let saved = initial;
    const values = {...saved.values, b_labels:[...saved.values.b_labels], t_labels:[...saved.values.t_labels]};
    if (!values.phenotype_col) values.phenotype_col = ['phenotype_ml_mr', 'phenotype_ml_consensus'].find(c => known.groups.includes(c)) || known.groups[0] || '';
    if (!known.columns.includes(values.roi_col)) values.roi_col = '';
    const status = node('p', {class:'muted', role:'status'}), badge = chip('Saved', 'ok');
    const settings = section('TLS candidate settings', {tip:'Exploratory B-cell aggregates with nearby T cells, not confirmed or mature TLS. Connected B-cell cores must meet Minimum B. Only T cells within the maximum gap of a core cell are added. Each contributes a disc of radius half the gap; holes remain empty. Each image and tissue region is analysed separately.', actions:[badge]});
    const mark = () => { const dirty = JSON.stringify(values) !== JSON.stringify(saved.values); badge.textContent = dirty ? 'Not saved' : 'Saved'; badge.className = 'chip ' + (dirty ? 'warn' : 'ok'); };
    let writing = Promise.resolve();
    const save = async () => {
      if (mppMode.value === 'manual' && !(Number.isFinite(values.pixel_um) && values.pixel_um > 0)) throw Error('Enter a positive MPP or choose Auto.');
      const sent = JSON.parse(JSON.stringify(values));
      writing = writing.catch(() => {}).then(async () => {
        saved = await api('tls_settings', {changes:sent, revision:saved.revision}); mark(); return saved;
      });
      return writing;
    };
    settings.box.querySelector('.studio-sec-h').append(node('button', {class:'btn small', text:'Save', onclick:() => save().then(() => {status.textContent = 'Settings saved.';}).catch(e => {status.textContent = e.message;})}));
    const column = columnPick(known.groups, values.phenotype_col, 'TLS phenotype column');
    const region = choice([['', 'Whole image'], ...known.columns.map(c => [c,c])], values.roi_col, 'TLS tissue region');
    region.onchange = () => { values.roi_col = region.value; mark(); };
    const labelBody = node('div');
    let labelGeneration = 0;
    const fillLabels = async () => {
      const generation = ++labelGeneration;
      const found = await api('tls_labels', {column:column.value});
      if (!pane.isConnected || generation !== labelGeneration) return;
      const pickers = [];
      labelBody.replaceChildren(...[['b_labels','B cells'], ['t_labels','T cells']].map(([key, label]) => {
        const summary = node('summary'), picker = node('details', {class:'threshold-picker', 'aria-label':'TLS '+label+' selector'}, [summary]);
        pickers.push(picker);
        const update = () => { summary.textContent = values[key].join(', ') || 'Choose phenotypes…'; summary.title = summary.textContent; };
        const list = node('div', {class:'threshold-checks'}, [...new Set([...found.labels, ...values[key]])].map(name => {
          const tick = node('input', {type:'checkbox', 'aria-label':label + ': ' + name}); tick.checked = values[key].includes(name);
          tick.onchange = () => { values[key] = tick.checked ? [...values[key], name] : values[key].filter(n => n !== name); update(); mark(); };
          return node('label', {title:name}, [tick, node('span', {text:name})]);
        }));
        const search = node('input', {type:'search', placeholder:'Find a phenotype…', 'aria-label':'Find TLS '+label});
        search.oninput = () => { for (const row of list.children) row.hidden = !row.textContent.toLowerCase().includes(search.value.toLowerCase()); };
        picker.append(search, list, node('button', {class:'btn small', text:'Done', onclick:() => {picker.open=false;summary.focus();}}));
        summary.addEventListener('click', () => { if (!picker.open) for (const other of pickers) if (other !== picker) other.open=false; });
        picker.onkeydown = e => { if (e.key === 'Escape') {picker.open=false;summary.focus();} };
        update(); return field(label, picker);
      }));
    };
    column.onchange = () => { values.phenotype_col = column.value; mark(); fillLabels().catch(e => {status.textContent = e.message;}); };
    const numeric = (key, label, min = '0', step = 'any') => {
      const input = number(values[key], label, step, min);
      input.onchange = () => { values[key] = Number(input.value); mark(); };
      return input;
    };
    let mppByImage = {};
    const mppMode = choice([['auto','Auto'],['manual','Manual']], values.pixel_um > 0 ? 'manual' : 'auto', 'TLS MPP mode');
    const mpp = number(values.pixel_um || '', 'TLS MPP'), mppSource = node('span', {class:'muted', 'aria-label':'TLS MPP source'});
    const updateMPP = () => {
      const image = h.openRoi(), entry = mppByImage[image];
      mpp.readOnly = mppMode.value === 'auto';
      if (mpp.readOnly) {
        mpp.value = entry?.mpp || ''; mpp.placeholder = 'Unavailable';
        mppSource.textContent = entry?.mpp ? (entry.source === 'image' ? 'Image' : 'Project') + ' · µm/px' : 'No calibration';
      } else mppSource.textContent = 'All selected images · µm/px';
      mppSource.title = 'Auto uses each image’s own MPP.\n' + Object.entries(mppByImage).map(([name,e]) => name+': '+(e.mpp ? e.mpp+' µm/px ('+e.source+')' : 'unavailable')).join('\n');
    };
    const loadMPP = async () => { try { mppByImage = (await api('tls_mpp')).images; if (pane.isConnected) updateMPP(); }
      catch (error) { if (pane.isConnected) mppSource.textContent = error.message; } };
    const mppReady = loadMPP();
    mppMode.onchange = () => { values.pixel_um = mppMode.value === 'auto' ? 0 : Number(mpp.value) || 0; updateMPP(); mark(); };
    mpp.onchange = () => { if (mppMode.value === 'manual') { values.pixel_um=Number(mpp.value);mark(); } };
    const output = node('input', {type:'text', value:values.output_col, 'aria-label':'TLS output prefix'});
    const overwrite = node('input', {type:'checkbox', 'aria-label':'Overwrite TLS results'});
    output.oninput = () => { values.output_col = output.value.trim(); overwrite.checked = false; mark(); };
    const advanced = node('details', {class:'studio-options'}, [node('summary', {text:'Advanced'}),
      field('Trim edge (µm)', numeric('edge_trim','TLS edge trim'), 'Insets the detected boundary after B/T connections are found. Zero keeps the half-gap margin. Must be less than half the Maximum gap; trimming can split a candidate or remove edge cells, so all count and density filters are checked again.'),
      field('Minimum B', numeric('min_b','TLS minimum B cells','1','1')),
      field('Minimum T', numeric('min_t','TLS minimum T cells','1','1')),
      field('B/T fraction', numeric('min_fraction','TLS minimum B/T fraction'), 'Minimum B + T fraction of all cells inside the candidate, from 0 to 1.'),
      field('B/T density', numeric('min_density','TLS minimum density'), 'Minimum B + T cells per mm². Zero leaves density as a review measurement.')]);
    const outNote = node('span', {class:'muted'});
    const updateNote = () => { outNote.textContent = [values.output_col, ...['_id','_size','_distance'].map(s => values.output_col+s)].join(' · '); };
    output.addEventListener('input', updateNote); updateNote();
    settings.body.append(field('Phenotypes', column), labelBody, field('By region', region, 'IDs restart in each image / tissue region. Missing and zero region IDs are excluded.'),
      field('Maximum gap', numeric('max_gap','TLS maximum gap','1'), 'µm between connected B-cell centres, and from an added T cell to the B core. T-cell chains do not extend the search. Larger gaps remain separate.'),
      field('MPP', node('div', {class:'inline studio-tls-mpp'}, [mppMode,mpp,mppSource]), 'Microns per pixel: Auto reads each image’s metadata, then the project value. The value shown is for the open image. Manual overrides every selected image. Cell coordinates must be in image pixels.'),
      field('Minimum B + T', numeric('min_cells','TLS minimum lymphocytes','2','1'), 'Minimum number of B and T cells combined. Advanced also sets separate B and T minima.'), advanced,
      field('Save as', output, 'Use TLS or TLS_ followed by a short name. Size is boundary area in µm²; distance is in µm.'),
      node('label', {class:'row'}, [overwrite,node('span', {text:'Overwrite existing results'}),help('Replaces all four output columns. Rows outside the selected images or valid tissue are blank. Untick this to keep existing outputs.')]), outNote,
      node('p', {class:'muted', text:'Exploratory starting settings. Review boundaries and vary the gap before interpreting candidates.'}));
    const results = section('TLS candidates found', {tip:'Each row is one connected structure. IDs repeat across tissue regions: image + region + ID identifies a structure. Click Review to open its boundary and cells. Size is polygon area, not cell count. Density is B + T cells per mm².'});
    pane.append(settings.box, results.box);
    let result = null, imagePick = null, rows = null, visibility = null, selectedKey = null, request = 0, navigating = false, imageReady = true;
    const view = current.tlsView ||= {visible:true};
    const shapes = new Map();
    let editing = null, editButton = null, savingEdit = false;
    const editBox = node('div', {class:'studio-options studio-tls-editor', hidden:true}), editNote=node('p',{class:'muted'});
    const trimAmount=number(3,'TLS draft trim','any','0');
    const safely = act => async () => {try{await act();}catch(error){status.textContent=error.message;}};
    const captureEdit = () => {
      const draft = editing && h.tlsEditor('read');
      if(draft && !draft.busy)editing.features=draft.features;
      return draft;
    };
    const updateEditor = () => {
      const busy=savingEdit||tlsDraft?.saving||tlsDraft?.busy;
      if(editButton){editButton.textContent=tlsDraft?'Resume edits':'Edit boundaries';editButton.hidden=!!editing;editButton.disabled=!!busy;}
      editBox.hidden=!tlsDraft;
      editNote.textContent=tlsDraft ? tlsDraft.image+' · region '+tlsDraft.region+(editing?' · Shift add · Alt subtract · Ctrl+Z undo':' · draft kept') : '';
      for(const control of editBox.querySelectorAll('button,input'))control.disabled=!!busy||(!editing&&control!==discardEdit);
    };
    const pauseEdit = () => {captureEdit();editing=null;h.tlsEditor('close');updateEditor();};
    const discardEdit=node('button',{class:'btn small',text:'Discard edits',onclick:safely(async()=>{
      if(savingEdit||tlsDraft?.saving||tlsDraft?.busy)return;pauseEdit();tlsDraft=null;updateEditor();await show(h.openRoi());
    })});
    const saveEdit=node('button',{class:'btn primary small',text:'Save edits',onclick:safely(async()=>{
      if(!editing || savingEdit)return;
      if(captureEdit()?.busy)throw Error('Wait for the boundary edit to finish.');
      const draft=editing;
      savingEdit=true;draft.saving=true;h.tlsEditor('freeze',{value:true});updateEditor();
      try{
        const started=await api('tls_save',draft), id=started.job?.id||started.job;
        h.refreshJobs?.();status.textContent='Saving boundaries and recalculating TLS columns…';
        for(;;){
          await new Promise(resolve=>setTimeout(resolve,2000));
          const job=((await get(route('/jobs'))).jobs||[]).find(j=>j.id===id);
          if(job?.status==='done'){
            if(tlsDraft===draft)tlsDraft=null;
            if(editing===draft){editing=null;h.tlsEditor('close');}shapes.clear();drawResult({...job.result,job_id:id});
            status.textContent='Edits saved · membership, IDs, area and distances updated.';break;
          }
          if(job && ['failed','cancelled'].includes(job.status))throw Error(job.message||'Stopped. Your draft is kept.');
        }
      }finally{savingEdit=false;draft.saving=false;if(editing===draft)h.tlsEditor('freeze',{value:false});updateEditor();}
    })});
    editBox.append(editNote,node('div',{class:'row wrap'},[
      node('button',{class:'btn small',text:'Undo',onclick:safely(()=>h.tlsEditor('undo'))}),
      node('button',{class:'btn small',text:'Redo',onclick:safely(()=>h.tlsEditor('redo'))}),
      node('button',{class:'btn small',text:'Merge',title:'Click two candidates in the same tissue region. The Annotation toolbar sets the maximum bridging gap.',onclick:safely(()=>h.tlsEditor('merge'))}),
      trimAmount,node('span',{class:'muted',text:'µm'}),node('button',{class:'btn small',text:'Trim edge',title:'Shrink this region’s draft boundaries by the given µm; Undo restores them.',onclick:safely(async()=>{
        if(!editing||captureEdit()?.busy)throw Error('Wait for the boundary edit to finish.');
        const draft=editing;
        draft.busy=true;h.tlsEditor('freeze',{value:true});updateEditor();
        try{
          const answer=await api('tls_trim',{...draft,amount:Number(trimAmount.value)});
          if(editing!==draft||h.openRoi()!==draft.image)return;
          h.tlsEditor('freeze',{value:false});h.tlsEditor('replace',answer);captureEdit();
        }finally{draft.busy=false;if(editing===draft)h.tlsEditor('freeze',{value:false});updateEditor();}
      })}),saveEdit,discardEdit,
      help('Save edits replaces the four TLS columns only within this image / tissue region. Touching boundaries merge, disconnected pieces receive separate local IDs. Boundaries are clipped to tissue and this region’s cell support. Detection count filters are not reapplied to a manual review. The original run and boundaries remain in Jobs; candidates are still not confirmed TLS.') ]));
    const beginEdit = safely(async()=>{
      if(!result)return;
      if(tlsDraft && tlsDraft.job_id!==result.job_id)throw Error('TLS results changed. Discard the old draft before editing this run.');
      if(tlsDraft){await show(tlsDraft.image,null,true);}
      else{
        const selected=result.structures.find(s=>s.key===selectedKey)||result.structures.find(s=>s.image===h.openRoi());
        if(!selected)throw Error('Select a TLS to edit.');
        const draft=await api('tls_edit',{job_id:result.job_id,image:selected.image,region:selected.region});
        if(!pane.isConnected||h.openRoi()!==draft.image)return;
        tlsDraft=draft;
      }
      editing=tlsDraft;
      h.tlsEditor('open',{...editing,selectedKey});updateEditor();
      await show(editing.image,null,true);
    });
    const clear = () => { h.showPreview('tls-candidates', null); h.showPreview('tls-selected', null); };
    const syncVisibility = () => { if (visibility) { visibility.textContent = view.visible ? 'Hide TLS' : 'Show TLS'; visibility.setAttribute('aria-pressed', String(view.visible)); } };
    const drawRows = () => {
      if (!result || !rows || !imagePick) return;
      const structures = result.structures.filter(s => s.image === imagePick.value);
      rows.replaceChildren(structures.length ? node('table', {class:'studio-counts'}, [node('tr', {}, ['Region / ID','B / T','Area (µm²)','B/T per mm²',''].map(text => node('th',{text}))),
        ...structures.map(s => node('tr', {'data-tls':s.key, class:s.key===selectedKey ? 'selected' : '', onclick:()=>show(s.image,s,true)}, [node('td',{text:s.region+' / '+s.id+(s.reviewed?' · edited':'')}),node('td',{text:s.b_cells+' / '+s.t_cells}),
          node('td',{text:fmt(s.area_um2)}),node('td',{text:fmt(s.density_per_mm2)}),node('td',{},[node('button',{class:'btn small',text:'Review',title:'Zoom to this TLS; keep every boundary visible',onclick:e=>{e.stopPropagation();show(s.image,s,true);}})])]))])
        : node('p', {class:'muted', text:!Object.hasOwn(result.counts,imagePick.value) ? 'This image is not in this TLS run.' : result.measured[imagePick.value] ? 'No candidates passed these settings.' : 'No cells with valid coordinates and tissue regions were analysed.'}));
    };
    const setReviewImage = image => {
      if (!imagePick || !image) return;
      if (imagePick.value !== image) selectedKey = null;
      if (![...imagePick.options].some(o=>o.value===image)) imagePick.append(node('option',{value:image,text:image+' · not in this run'}));
      imagePick.value=image; drawRows();
    };
    const show = async (image, structure = null, forceShow = false) => {
      if (!result || !image || !pane.isConnected) return;
      const ticket = ++request, run = result;
      if(editing && (editing.image!==image || structure && editing.region!==structure.region))pauseEdit();
      if (forceShow) view.visible = true;
      setReviewImage(image); selectedKey = structure?.key || null; drawRows(); syncVisibility();
      try {
        navigating = true;
        await showOnImage(image, async () => {
          if (ticket !== request || !pane.isConnected) return;
          if (h.openRoi() !== image) throw Error('Open the image and its cell layer before reviewing this candidate.');
          updateMPP();
          if (!view.visible || !Object.hasOwn(run.counts,image)) {clear();return;}
          const cacheKey = run.job_id + ':' + image;
          const found = shapes.get(cacheKey) || await api('tls_shapes', {job_id:run.job_id, image});
          if (ticket !== request || !pane.isConnected || h.openRoi() !== image || !view.visible) return;
          shapes.set(cacheKey,found);
          h.showPreview('tls-candidates', {outlines:found.features.filter(f=>!editing||f.properties.region!==editing.region).flatMap(f => rings(f.geometry)), colour:'#00dcc8', kind:'example'});
          const selected = structure ? found.features.filter(f=>f.properties.key===structure.key) : [];
          h.showPreview('tls-selected', !editing && selected.length ? {outlines:selected.flatMap(f=>rings(f.geometry)),colour:'#ffd166',kind:'example'} : null);
          if(editing && structure)h.tlsEditor('focus',{key:structure.key});
          // Selecting a structure changes the view, not the set of TLS cells.
          await h.filterCells(editing ? [] : [{col:run.column,lo:1,hi:1}], run.column);
          if (ticket !== request || h.openRoi() !== image || !pane.isConnected) return;
          if (selected.length) {
            const points = selected.flatMap(f => rings(f.geometry)).flat();
            const bounds = points.reduce((b,p) => [Math.min(b[0],p[0]),Math.min(b[1],p[1]),Math.max(b[2],p[0]),Math.max(b[3],p[1])],[Infinity,Infinity,-Infinity,-Infinity]);
            const [x,y] = bounds, w = bounds[2]-x, hgt = bounds[3]-y;
            h.zoomBox?.([x-w*.1,y-hgt*.1,w*1.2,hgt*1.2]);
          }
        });
        if (ticket === request) status.textContent = image + (structure ? ' · region '+structure.region+' · TLS '+structure.id : view.visible ? ' · all candidates' : ' · TLS hidden');
      } catch (error) { if(ticket===request)status.textContent = error.message; }
      finally { if(ticket===request){navigating = false;if(imageReady && h.openRoi() && h.openRoi()!==image)syncOpenImage();} }
    };
    const syncOpenImage = (ready = true) => {
      if (!pane.isConnected) return;
      imageReady = ready;
      if(!ready && editing)pauseEdit();
      updateMPP(); const image = h.openRoi();
      if (!image || !result) return;
      setReviewImage(image);
      if (!ready || navigating) return;
      show(image, result.structures.find(s=>s.key===selectedKey) || null);
    };
    current.tlsReview = {syncOpenImage, selectImage: image => show(image,null,true), changed:()=>{captureEdit();updateEditor();},
      refresh:async()=>{
        if(savingEdit)return;
        await mppReady;
        const latest=(await api('tls_result')).result;
        if(!pane.isConnected||!latest||latest.job_id===result?.job_id)return;
        pauseEdit();
        if(tlsDraft?.saving&&latest.parent_job===tlsDraft.job_id)tlsDraft=null;
        shapes.clear();drawResult(latest);
      },
      dispose:() => {++request;pauseEdit();clear();}};
    const drawResult = found => {
      result = found;
      if (!pane.isConnected) return;
      if (!result) { results.body.replaceChildren(node('p', {class:'muted', text:'No TLS candidate run yet.'})); return; }
      imagePick = choice(Object.keys(result.counts).map(name => [name, name+' · '+result.counts[name]+' candidates']),
        Object.hasOwn(result.counts, h.openRoi()) ? h.openRoi() : result.images[0], 'TLS review image');
      rows = node('div', {class:'studio-samples'});
      imagePick.onchange = () => show(imagePick.value);
      visibility = node('button',{class:'btn small', 'aria-label':'TLS boundary visibility', title:'Show or hide every candidate boundary in this image',onclick:()=>{
        view.visible = !view.visible; syncVisibility();
        if(view.visible)show(imagePick.value);else{++request;navigating=false;pauseEdit();clear();h.filterCells([],result.column);}
      }});
      syncVisibility();
      editButton=node('button',{class:'btn small',text:'Edit boundaries',onclick:beginEdit});
      results.body.replaceChildren(node('p', {class:'muted', text:result.total+' candidates · '+result.images.length+' images · '+result.column}), field('Image', imagePick),
        node('div',{class:'row wrap'},[visibility,editButton,
          ...['geojson','csv'].map(format=>node('a',{class:'btn small',text:format==='geojson'?'GeoJSON':'Summary CSV',href:route('/spatial/tls_download')+'?job_id='+encodeURIComponent(result.job_id)+'&format='+format}))]), editBox, rows);
      updateEditor();
      if (result.warnings.length) results.body.append(node('details',{class:'studio-options'},[node('summary',{text:'Tissue support · '+result.warnings.length+' notices'}),
        node('p',{class:'muted',text:'Where tissue annotations are unavailable, the boundary and clear-line check use cell-derived tissue support. Unseen gaps cannot be guaranteed.'}),
        ...result.warnings.map(text=>node('p',{class:'muted',text}))]));
      drawRows();
      syncOpenImage();
    };
    current.runBar([actionRow('Find TLS:', async (images, note) => {
      if(tlsDraft)throw Error('Save or discard the boundary draft before running detection again.');
      await save();
      const started = await api('start', {kind:'tls_candidates', settings_revision:saved.revision, overwrite:overwrite.checked, images});
      overwrite.checked = false; h.refreshJobs?.();
      note.textContent = 'Finding candidates in the background. Results stay in Jobs.';
      const id = started.job?.id || started.job;
      for (;;) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        const jobs = (await get(route('/jobs'))).jobs || [], job = jobs.find(j => j.id === id);
        if (!job) break;
        if (job.status === 'done') { drawResult({...job.result,job_id:id}); note.textContent = job.result.total+' candidates saved.'; break; }
        if (['failed','cancelled'].includes(job.status)) throw Error(job.message || 'Stopped.');
        if (!pane.isConnected) break;
      }
    }, 'Runs on this image or the ticked images. Saves membership, local ID, area and signed boundary distance to the combined dataset.', status),status]);
    mark();
    await Promise.all([mppReady, fillLabels()]);
    drawResult((await api('tls_result')).result);
  }

  // ------------------------------------------------------------ Density
  async function density(current, pane, actionRow) {
    const known = await info();
    if (!pane.isConnected) return;
    const settings = section('Cells per mm²', {tip:'Classic Tab 8’s density: each image’s cells of each group divided by the tissue area, the area being the outline of the cells themselves (each cell widened by the spacing of its neighbours; holes smaller than the size given are filled). The outline used is drawn on the image.'});
    const column = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus', 'phenotype_leiden']), 'Density group column');
    const pixel = number(0.5083, 'Pixel size in µm'), tight = number(99, 'Outline spacing percentile'), holes = number(10000, 'Fill holes smaller than (px²)', '100'), least = number(20, 'Smallest image or region (cells)', '1');
    settings.body.append(field('Groups', column, 'The column whose groups are counted (a phenotype, a neighbourhood…).'),
      field('Pixel size', pixel, 'µm per pixel of the cell positions.'), field('Outline', tight, 'Percentile of the distance between neighbouring cells used to widen each cell (99: generous; lower: tighter).'),
      field('Fill holes', holes, 'Holes in the outline smaller than this (pixels²) count as tissue.'), field('Smallest', least, 'Images with fewer cells are left out.'));
    const results = section('Density found', {tip:'Rows are images, columns groups; the colour shows how dense a group is in an image compared with the other images (log scale). Click an image to open it with its tissue outline; choose a group to map it.'});
    const map = section('Density map on the image', {tip:'Where a group’s cells are crowded in the open image: a heat map of the group’s cells (the Layers density map). Radius: how far each cell spreads its weight.'});
    const mapGroup = node('select', {'aria-label':'Map group'}), radius = number(40, 'Map radius in µm', '5', '5'), mapSaid = node('p', {class:'muted', role:'status'});
    const fillGroups = async () => { try { const found = await api('composition', {column:column.value}); mapGroup.replaceChildren(...found.groups.map(g => node('option', {value:g, text:g}))); } catch (_) { /* table changing */ } };
    column.onchange = fillGroups; fillGroups();
    map.body.append(field('Group', mapGroup), field('Radius', radius, 'In µm (as Layers).'), node('div', {class:'row'}, [node('button', {class:'btn small primary', text:'Show on the open image', onclick:async () => {
      try { const answer = await h.runCommand('density', {column:column.value, phenotype:mapGroup.value, radius:Number(radius.value) || 40}, {confirmed:true}); h.showResult(answer); mapSaid.textContent = answer.message; }
      catch (error) { mapSaid.textContent = error.message; }
    }})]), mapSaid);
    const byDistance = section('By distance from an annotation', {tip:'Uses Cell tagging’s distance columns (µm from an annotation’s edge; negative inside). For each group: how much more (warm) or less (blue) it is found in each band than overall. Click a group to show it on the image; choose a band to show only its cells.'});
    const distance = columnPick(known.distances, known.distances[0] || '', 'Distance column'), step = number(25, 'Band width in µm', '5', '1'), limit = number(300, 'Bands up to µm', '25', '25');
    const inside = node('input', {type:'checkbox', checked:'', 'aria-label':'Include inside the annotation'});
    const band = node('select', {'aria-label':'Distance band'}), distSaid = node('p', {class:'muted', role:'status'});
    let profileResult = null;
    const showDistance = async images => {
      if (!distance.value) { distSaid.textContent = 'No distance column: tag the cells with an annotation distance first (Cell tagging).'; return; }
      distSaid.textContent = 'Counting…';
      profileResult = await api('distance', {column:column.value, distance:distance.value, step:Number(step.value), limit:Number(limit.value), inside:inside.checked, images});
      band.replaceChildren(node('option', {value:'', text:'Every band'}), ...profileResult.bands.map((b, i) => node('option', {value:String(i), text:b + ' µm (' + profileResult.totals[i].toLocaleString() + ' cells)'})));
      let picked = [];
      h.heatmapFigure({key:'distance', title:column.value + ' by ' + distance.value + ' (µm)', profile:profileResult.profile,
        tip:'Warm: the group is more common in that band than overall; blue: less. Hover for the share of the band’s cells. Click a group to show it; Ctrl-click for several.',
        onPick:groups => { picked = groups; applyBand(picked); }});
      distSaid.textContent = 'Bands of ' + step.value + ' µm from the annotation edge (' + profileResult.totals.reduce((a, b) => a + b, 0).toLocaleString() + ' cells).';
    };
    const applyBand = async (groups = []) => {
      const filters = [];
      if (band.value !== '' && profileResult) { const i = Number(band.value); filters.push({col:distance.value, lo:profileResult.edges[i], hi:profileResult.edges[i + 1]}); }
      if (groups.length) filters.push({col:column.value, values:groups});
      const shown = await h.filterCells(filters, column.value);
      distSaid.textContent = !shown ? 'Open an image whose cells are in the combined table to see them.' : filters.length ? 'The image shows ' + (groups.join(', ') || 'every group') + (band.value !== '' ? ' in ' + band.selectedOptions[0].textContent : '') + '.' : 'The image shows every cell.';
    };
    band.onchange = () => applyBand();
    byDistance.body.append(field('Distance', distance, 'A distance column from Cell tagging (µm; negative inside the annotation).'),
      field('Bands', node('div', {class:'inline'}, [step, node('span', {class:'muted', text:'µm, up to'}), limit, node('span', {class:'muted', text:'µm'})])),
      node('label', {class:'row'}, [inside, node('span', {text:'Include inside the annotation (negative distances)'})]),
      field('Show band', band, 'Draw only the cells of this band on the image (with the groups picked in the heatmap).'), distSaid);
    pane.append(settings.box, results.box, map.box, byDistance.box);
    const said = node('p', {class:'muted', role:'status'});
    const drawResult = result => {
      results.body.replaceChildren(node('div', {class:'row wrap'}, [chip(result.units.length + ' images'), chip(result.groups.length + ' groups'), chip(result.column)]),
        node('p', {class:'muted', text:'Saved in ' + result.folder.split('/').slice(-3).join('/') + ' (density_per_unit.csv, tissue outlines).'}));
      h.heatmapFigure({key:'density', title:'Cells per mm² (' + result.column + ')', profile:result.profile,
        tip:'Each row an image, each column a group: colour compares the image’s density with the other images; hover for cells per mm². Click an image to open it with its tissue outline.',
        onPick:async rows => {
          const unit = rows[rows.length - 1]; if (!unit) { h.showPreview('density-outline', null); return; }
          const roi = result.images[result.units.indexOf(unit)];
          await showOnImage(roi, async () => {
            try { const outline = await api('outline', {folder:result.folder, image:roi});
              h.showPreview('density-outline', {outlines:outline.features.flatMap(f => rings(f.geometry)), colour:'#ffd166', kind:'example'});
              said.textContent = roi + ': tissue area ' + fmt(result.areas_mm2[result.units.indexOf(unit)]) + ' mm² (the yellow outline).'; }
            catch (error) { said.textContent = error.message; }
          });
        }});
    };
    lastResult('spatial_density').then(result => { if (result && pane.isConnected) drawResult(result); }).catch(() => {});
    current.runBar([
      actionRow('Density:', async (images, note) => { note.textContent = 'Counting cells and drawing tissue outlines…';
        drawResult(await runJob('spatial_density', {column:column.value, pixel_um:Number(pixel.value), percentile:Number(tight.value), min_hole_px2:Number(holes.value), min_cells:Number(least.value), images}, note)); },
        'Cells per mm² of each group, for the open image or the ticked ones (all: tick the header box), in the background. Each image’s outline is kept to draw on it.', said),
      actionRow('By distance:', async (images, note) => { await showDistance(images); note.textContent = distSaid.textContent; },
        'Each group’s share in bands of distance from the annotation, for the open image or the ticked images together (quick).', said), said]);
  }
  const rings = geometry => geometry.type === 'Polygon' ? geometry.coordinates : geometry.type === 'MultiPolygon' ? geometry.coordinates.flat() : [];

  // ------------------------------------------------------------ Triads
  async function triads(current, pane, actionRow, savedPanel) {
    const known = await info();
    if (!pane.isConnected) return;
    const settings = section('Triad', {tip:'Classic Tab 8’s triads: three cells of phenotypes A, B and C each within the distance of the others. The test compares how many such triangles there are with the same cells after shuffling their phenotypes (within each image). Each triangle is kept to draw on the image.'});
    const column = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus', 'phenotype_leiden']), 'Triad group column');
    const corners = ['A', 'B', 'C'].map(letter => node('select', {'aria-label':'Phenotype ' + letter}));
    const fill = async () => { try { const found = await api('composition', {column:column.value});
      corners.forEach((select, i) => { select.replaceChildren(...found.groups.map(g => node('option', {value:g, text:g}))); select.value = found.groups[i] || found.groups[0]; }); } catch (_) { /* later */ } };
    column.onchange = fill; fill();
    const near = number(30, 'Triad distance up to µm', '1', '1'), far = number(0, 'Triad distance at least µm', '1'), pixel = number(0.5083, 'Triad pixel size in µm');
    const shuffles = number(200, 'Triad shuffles', '10', '10'), least = number(20, 'Triad smallest image', '1', '1');
    settings.body.append(field('Groups', column), field('A · B · C', node('div', {class:'inline'}, corners), 'A is the anchor: each triangle is counted from an A cell.'),
      field('Distance', node('div', {class:'inline'}, [far, node('span', {class:'muted', text:'to'}), near, node('span', {class:'muted', text:'µm'})]), 'Each pair of the three cells is at least and at most this far apart (centre to centre).'),
      field('Pixel size', pixel, 'µm per pixel of the cell positions.'), field('Shuffles', shuffles, 'How many times the phenotypes are shuffled to see what chance gives.'),
      field('Smallest', least, 'Images with fewer cells are left out.'));
    const results = section('Triads found', {tip:'How many triangles there are, how many chance gives, and each image’s count. Click an image to open it with its triangles drawn and the triad density map.'});
    pane.append(settings.box, results.box);
    if (savedPanel) { const saved = section('Saved earlier (classic Tab 8)', {tip:'Triad tables and feature screening runs classic Tab 8 saved in this project.'}); pane.append(saved.box); savedPanel(saved.body); }
    const said = node('p', {class:'muted', role:'status'});
    const drawResult = result => {
      const s = result.summary;
      const hist = result.null.counts.length ? histogram(result.null, s.observed_total_count) : null;
      results.body.replaceChildren(
        node('div', {class:'row wrap'}, [chip(result.triad.join(' · '), 'ok'), chip(fmt(s.observed_total_count) + ' found'), chip(fmt(s.expected_mean_count) + ' by chance'),
          chip('z ' + fmt(s.z_score), s.z_score > 2 ? 'ok' : s.z_score < -2 ? 'warn' : ''), chip('p ' + fmt(s.empirical_p_value_enrichment))]),
        node('p', {class:'muted', text:s.z_score > 2 ? 'These three are found together more often than chance.' : s.z_score < -2 ? 'These three are found together less often than chance (they avoid each other).' : 'Not different from chance.'}),
        ...(hist ? [hist] : []),
        node('table', {class:'studio-counts'}, [node('tr', {}, [node('th', {text:'Image'}), node('th', {text:'Cells'}), node('th', {text:'Triangles'})]),
          ...result.per_image.map(row => node('tr', {}, [node('td', {}, [node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:'Open with its triangles',
            onclick:() => showTriads(result, row.image)})]), node('td', {class:'mono', text:row.n_cells.toLocaleString()}), node('td', {class:'mono', text:row.observed_triad_count.toLocaleString()})]))]));
    };
    const showTriads = (result, roi) => showOnImage(roi, async () => {
      try {
        const shapes = await api('triad_shapes', {folder:result.folder, image:roi});
        h.showPreview('triads', {outlines:shapes.rings, colour:'#00e5ff', kind:'example'});
        await h.filterCells([{col:result.column, values:result.triad}], result.column);
        said.textContent = roi + ': ' + shapes.count.toLocaleString() + ' triangles drawn (cyan), with only ' + result.triad.join(', ') + ' cells shown.';
        const answer = await h.runCommand('density', {table:result.table_name, column:'triad_label', phenotype:result.triad.join('::'), radius:60}, {confirmed:true});
        h.showResult(answer);
      } catch (error) { said.textContent = error.message; }
    });
    lastResult('spatial_triads').then(result => { if (result && pane.isConnected) drawResult(result); }).catch(() => {});
    current.runBar([actionRow('Find triads:', async (images, note) => {
      const triad = corners.map(c => c.value);
      if (new Set(triad).size < 2) { note.textContent = 'Choose at least two different phenotypes.'; return; }
      note.textContent = 'Counting triangles and shuffling…';
      drawResult(await runJob('spatial_triads', {images, values:{phenotype_col:column.value, triad, min_um:Number(far.value), max_um:Number(near.value), pixel_um:Number(pixel.value),
        n_permutations:Number(shuffles.value), min_cells:Number(least.value), batch_col:'batch_id', roi_col:''}}, note));
    }, 'Counts the triangles in the open image or the ticked images (all: tick the header box) and tests them against shuffled phenotypes, in the background.', said), said]);
  }
  function histogram(found, observed) {
    const width = 300, height = 70, max = Math.max(...found.counts, 1), edges = found.edges, low = Math.min(edges[0], observed), high = Math.max(edges[edges.length - 1], observed);
    const x = v => 4 + (width - 8) * (v - low) / Math.max(high - low, 1e-9);
    const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + (height + 14)); svg.setAttribute('class', 'spatial-hist'); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Triangles by chance (bars) and found (line)');
    found.counts.forEach((count, i) => { const r = document.createElementNS(ns, 'rect'); const x0 = x(edges[i]), x1 = x(edges[i + 1]);
      r.setAttribute('x', x0); r.setAttribute('width', Math.max(1, x1 - x0 - 1)); r.setAttribute('y', height - height * count / max); r.setAttribute('height', height * count / max); r.setAttribute('class', 'spatial-hist-bar'); svg.append(r); });
    const line = document.createElementNS(ns, 'line'); line.setAttribute('x1', x(observed)); line.setAttribute('x2', x(observed)); line.setAttribute('y1', 0); line.setAttribute('y2', height); line.setAttribute('class', 'spatial-hist-observed'); svg.append(line);
    const text = document.createElementNS(ns, 'text'); text.setAttribute('x', 4); text.setAttribute('y', height + 12); text.textContent = 'Bars: by chance · line: found'; text.setAttribute('class', 'spatial-hist-text'); svg.append(text);
    return svg;
  }

  // ------------------------------------------------------------ Feature screening
  async function screening(current, pane, actionRow, savedPanel) {
    const known = await info();
    if (!pane.isConnected) return;
    const settings = section('Feature screening', {tip:'Classic Tab 8’s feature screening: spatial features of each patient (densities, proportions, ratios, neighbours, pairs and triads of the chosen phenotypes, within growing distances of an annotation) tested against an outcome. The run is saved with its tables and figures where classic saves it.'});
    const all = known.columns;
    const patient = columnPick(all, prefer(all, ['patient', 'Patient', 'patient_id']), 'Patient column');
    const groups = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus']), 'Screening group column');
    const distance = columnPick(known.distances, known.distances[0] || '', 'Screening distance column');
    const kind = choice([['survival', 'Survival (time and event)'], ['binary', 'Two groups (response…)']], 'survival', 'Outcome kind');
    const time = columnPick(all, prefer(all, ['os_months', 'OS_months', 'time']), 'Outcome time column'), event = columnPick(all, prefer(all, ['os_event', 'OS_event', 'event']), 'Outcome event column');
    const censored = node('input', {type:'checkbox', 'aria-label':'1 means censored'});
    const group = columnPick(all, prefer(all, ['response', 'Response']), 'Outcome group column'), positive = node('input', {type:'text', placeholder:'e.g. responder', 'aria-label':'Positive value'});
    const pixel = number(0.5083, 'Screening pixel size in µm');
    const survivalRows = [field('Time', time), field('Event', event), node('label', {class:'row'}, [censored, node('span', {text:'1 in the event column means censored (not an event)'})])];
    const binaryRows = [field('Group', group), field('Positive', positive, 'The value of the group column that counts as the outcome (e.g. responder).')];
    const sync = () => { survivalRows.forEach(r => r.hidden = kind.value !== 'survival'); binaryRows.forEach(r => r.hidden = kind.value === 'survival'); };
    kind.onchange = sync;
    settings.body.append(field('Patient', patient, 'The column naming each cell’s patient (the outcome is per patient).'), field('Groups', groups), field('Distance', distance, 'Cell tagging’s distance from an annotation (µm); the zones grow along it.'),
      field('Outcome', kind), ...survivalRows, ...binaryRows, field('Pixel size', pixel));
    sync();
    const results = section('Strongest findings', {tip:'The features most linked to the outcome. Click one to show its cells on the open image: the phenotypes it is made of, inside its zone (its distance from the annotation).'});
    pane.append(settings.box, results.box);
    if (savedPanel) { const saved = section('Saved runs and figures', {tip:'Feature screening runs saved in this project (here or in classic Tab 8), with their figures.'}); pane.append(saved.box); savedPanel(saved.body); }
    const said = node('p', {class:'muted', role:'status'});
    const detail = node('div');
    const drawResult = result => {
      results.body.replaceChildren(node('div', {class:'row wrap'}, [chip(result.run, 'ok'), chip('best zone ' + fmt(result.selected_width) + ' µm'), chip(result.hits.length + ' findings')]),
        node('div', {class:'studio-samples'}, [node('table', {class:'studio-counts'}, [node('tr', {}, ['Feature', 'Zone (µm)', 'p', 'q', 'Effect', 'Patients'].map(k => node('th', {text:k}))),
          ...result.hits.map(hit => { const row = node('tr', {class:'spatial-hit', title:'Show on the open image, and the images where it is highest and lowest'},
            [hit.feature, hit.zone, fmt(hit.p), fmt(hit.q), fmt(hit.effect), String(hit.patients)].map(text => node('td', {text})));
            row.onclick = () => showHit(result, hit); return row; })])]), detail);
    };
    const showHit = async (result, hit) => {
      const names = hit.phenotypes || [];
      const width = Number(hit.width ?? result.selected_width);
      const open = roi => showOnImage(roi, () => showHit(result, hit));
      const list = (title, items) => node('div', {class:'row wrap'}, [node('span', {class:'lbl', text:title}),
        ...items.map(([roi, value]) => node('button', {type:'button', class:'btn small', text:roi + ' · ' + fmt(value), title:'Open ' + roi + ' showing this feature’s cells', onclick:() => open(roi)}))]);
      detail.replaceChildren(node('p', {class:'muted', text:hit.feature + ' (' + hit.family + ') within ' + hit.zone + ' µm.'}), list('Highest', hit.highest || []), list('Lowest', hit.lowest || []));
      const filters = [...(names.length ? [{col:result.phenotype_column, values:names}] : []), ...(Number.isFinite(width) ? [{col:result.distance_column, lo:0, hi:width}] : [])];
      const shown = await h.filterCells(filters, result.phenotype_column);
      said.textContent = shown ? 'The image shows ' + (names.join(', ') || 'every group') + (Number.isFinite(width) ? ' within ' + fmt(width) + ' µm of the annotation' : '') + '.' : 'Open an image whose cells are in the combined table.';
    };
    lastResult('spatial_screening').then(result => { if (result && pane.isConnected) drawResult(result); }).catch(() => {});
    current.runBar([actionRow('Screen:', async (images, note) => {
      const endpoint = kind.value === 'survival' ? {type:'survival', time_col:time.value, censor_col:event.value, censored_flag:censored.checked}
        : {type:'binary', group_col:group.value, positive_value:positive.value.trim()};
      note.textContent = 'Screening (several minutes on many patients)…';
      drawResult(await runJob('spatial_screening', {images, values:{patient_col:patient.value, phenotype_col:groups.value, distance_col:distance.value, endpoint, pixel_um:Number(pixel.value)}}, note));
    }, 'Screens the patients of the ticked images (all: tick the header box; the open image alone has one patient, too few) in the background.', said), said]);
  }

  // ------------------------------------------------------------ Visualization: Heatmap, Proportions, UMAP, Method agreement
  function scopeChoice() { return choice([['all', 'All images'], ['open', 'The open image']], 'all', 'Images shown'); }
  async function scopeImages(scope) {
    if (scope.value !== 'open') return [];
    const roi = h.openRoi();
    if (!roi) throw Error('Open an image of the combined table first.');
    return [roi];
  }
  async function heatmap(current, pane) {
    const known = await info();
    const box = section('Heatmap', {tip:'Each group’s mean marker values (arcsinh, z-scored across the cells). Click a group to show its cells on the open image; Ctrl-click for several.'});
    const column = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus']), 'Heatmap group column'), scope = scopeChoice();
    const markers = node('select', {multiple:'', size:'6', 'aria-label':'Heatmap markers'}, known.markers.map(m => node('option', {value:m, text:m, selected:known.markers.indexOf(m) < 30 ? '' : null})));
    const said = node('p', {class:'muted', role:'status'});
    box.body.append(field('Groups', column), field('Markers', markers, 'Ctrl-click to choose; the first 30 are chosen.'), field('Images', scope), said);
    pane.append(box.box);
    current.runBar([node('div', {class:'row'}, [node('span', {class:'lbl', text:'Heatmap:'}), node('button', {class:'btn primary', text:'Draw', onclick:async () => {
      said.textContent = 'Averaging…';
      try {
        const found = await api('profile', {column:column.value, markers:Array.from(markers.selectedOptions).map(o => o.value), images:await scopeImages(scope)});
        h.heatmapFigure({key:'viz-heatmap', title:found.column + ' · mean markers', profile:found.profile, onPick:async groups => {
          const shown = await h.showCellGroups(found.column, groups, found.profile.rows);
          said.textContent = shown ? (groups.length ? 'Showing ' + groups.join(', ') + ' on the image.' : 'Showing every group.') : 'Open an image whose cells are in the combined table.';
        }, tip:'Rows: groups (cells: ' + Object.values(found.cells).reduce((a, b) => a + b, 0).toLocaleString() + ' sampled). Click a row to show those cells on the image.'});
        said.textContent = 'Drawn beside the image.';
      } catch (error) { said.textContent = error.message; }
    }}), help('Draws the heatmap beside the image, linked to it: a click on a group shows its cells.')]), said]);
  }
  async function proportions(current, pane) {
    const known = await info();
    const box = section('Proportions', {tip:'The cells of each group in each image, as stacked bars. Click a colour to open that image showing only those cells.'});
    const column = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus']), 'Proportion group column');
    const said = node('p', {class:'muted', role:'status'});
    box.body.append(field('Groups', column), said);
    pane.append(box.box);
    current.runBar([node('div', {class:'row'}, [node('span', {class:'lbl', text:'Proportions:'}), node('button', {class:'btn primary', text:'Draw', onclick:async () => {
      said.textContent = 'Counting…';
      try { await whereFigure(column.value, [], 'viz-proportions', column.value + ' in each image'); said.textContent = 'Drawn beside the image.'; }
      catch (error) { said.textContent = error.message; }
    }}), help('Stacked bars per image, beside the image and linked to it.')]), said]);
  }
  async function umapPanel(current, pane) {
    const known = await info();
    const box = section('UMAP', {tip:'A map of cells by their markers: similar cells lie close together. A fair sample of each image’s cells (the same number from each). Colour by any group column; drag a box to select cells and show them on the image; click a point to open its image at that cell.'});
    const saved = choice([['', 'Make a new one'], ...known.umaps.map(u => [u.name, u.name.replace('umap_', '') + ' · ' + (u.cells || 0).toLocaleString() + ' cells'])], known.umaps[0]?.name || '', 'Saved UMAP');
    const column = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus']), 'UMAP colour column');
    const perImage = number(2000, 'Cells per image', '100', '100');
    const markers = node('select', {multiple:'', size:'6', 'aria-label':'UMAP markers'}, known.markers.map(m => node('option', {value:m, text:m, selected:known.markers.indexOf(m) < 30 ? '' : null})));
    const said = node('p', {class:'muted', role:'status'});
    box.body.append(field('UMAP', saved), field('Colour by', column), field('Cells per image', perImage), field('Markers', markers, 'Ctrl-click to choose; the first 30 are chosen.'), said);
    pane.append(box.box);
    const drawPoints = async name => {
      said.textContent = 'Reading the points…';
      const points = await api('umap_points', {name, column:column.value});
      await scatterFigure(points, said);
      said.textContent = points.x.length.toLocaleString() + ' cells. Drag a box on the map to select; click a point to open its cell.';
    };
    column.onchange = () => { if (saved.value) drawPoints(saved.value).catch(error => { said.textContent = error.message; }); };
    saved.onchange = column.onchange;
    if (saved.value) drawPoints(saved.value).catch(error => { said.textContent = error.message; });
    current.runBar([node('div', {class:'row'}, [node('span', {class:'lbl', text:'UMAP:'}), node('button', {class:'btn primary', text:'Make', onclick:async () => {
      try {
        const result = await runJob('spatial_umap', {markers:Array.from(markers.selectedOptions).map(o => o.value), per_image:Number(perImage.value)}, said);
        saved.append(node('option', {value:result.name, text:result.name.replace('umap_', '') + ' · ' + result.cells.toLocaleString() + ' cells'})); saved.value = result.name;
        await drawPoints(result.name);
      } catch (error) { said.textContent = error.message; }
    }}), help('Makes a UMAP of the cells of every image in the background (about a minute for 50,000 cells) and draws it beside the image.')]), said]);
  }
  async function scatterFigure(points, said) {
    const groups = points.groups ? [...new Set(points.groups)] : [];
    const palette = await paletteFor(groups);
    const {box, body} = section('UMAP · ' + (points.column || 'cells'), {tip:'Drag a box to select cells; they are shown on the open image (those of that image). Click a point to open its image at the cell.'});
    const canvas = node('canvas', {class:'spatial-umap', width:'520', height:'420', 'aria-label':'UMAP'});
    const legend = node('div', {class:'spatial-legend'}, groups.slice(0, 40).map(g => node('span', {class:'spatial-key'}, [node('span', {class:'split-marker-swatch', style:'background:' + palette[g]}), node('span', {text:g})])));
    const note = node('p', {class:'muted', role:'status'});
    body.append(canvas, legend, note);
    h.showFigure('viz-umap', box);
    const xs = points.x, ys = points.y, n = xs.length;
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const px = i => 10 + (canvas.width - 20) * (xs[i] - x0) / Math.max(x1 - x0, 1e-9), py = i => canvas.height - 10 - (canvas.height - 20) * (ys[i] - y0) / Math.max(y1 - y0, 1e-9);
    let chosen = new Set(), drag = null;
    const draw = () => {
      const g = canvas.getContext('2d'); g.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < n; i++) { g.fillStyle = chosen.size && !chosen.has(i) ? 'rgba(128,128,128,.25)' : (points.groups ? palette[points.groups[i]] : '#6aa9ff'); g.fillRect(px(i) - 1, py(i) - 1, 2.5, 2.5); }
      if (drag) { g.strokeStyle = '#fff'; g.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0); }
    };
    const at = event => { const r = canvas.getBoundingClientRect(); return [(event.clientX - r.left) * canvas.width / r.width, (event.clientY - r.top) * canvas.height / r.height]; };
    canvas.onpointerdown = event => { const [x, y] = at(event); drag = {x0:x, y0:y, x1:x, y1:y}; try { canvas.setPointerCapture(event.pointerId); } catch (_) { /* a synthetic event */ } };
    canvas.onpointermove = event => { if (!drag) return; [drag.x1, drag.y1] = at(event); draw(); };
    canvas.onpointerup = async () => {
      const box = drag; drag = null;
      if (!box) return;
      const [a, b, c, d] = [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1), Math.min(box.y0, box.y1), Math.max(box.y0, box.y1)];
      if (b - a < 3 && d - c < 3) {  // a click: the nearest point's cell
        let best = -1, dist = 64;
        for (let i = 0; i < n; i++) { const q = (px(i) - a) ** 2 + (py(i) - c) ** 2; if (q < dist) { dist = q; best = i; } }
        chosen = new Set(); draw();
        if (best < 0) { await h.filterCells([]); note.textContent = 'Selection cleared.'; return; }
        const roi = points.image[best], label = points.label[best];
        await showOnImage(roi, async () => { try { h.showResult(await h.runCommand('zoom_to_cell', {cell:label}, {confirmed:true})); await h.filterCells([{col:'label', values:[String(label)]}]); } catch (error) { note.textContent = error.message; } });
        note.textContent = 'Cell ' + label + ' of ' + roi + (points.groups ? ' (' + points.groups[best] + ')' : '') + '.';
        return;
      }
      chosen = new Set(); for (let i = 0; i < n; i++) { const x = px(i), y = py(i); if (x >= a && x <= b && y >= c && y <= d) chosen.add(i); }
      draw();
      const roi = h.openRoi(), here = [...chosen].filter(i => points.image[i] === roi).map(i => String(points.label[i]));
      const byImage = {}; chosen.forEach(i => { byImage[points.image[i]] = (byImage[points.image[i]] || 0) + 1; });
      const top = Object.entries(byImage).sort((p, q) => q[1] - p[1]).slice(0, 5).map(([k, v]) => k + ' ' + v).join(', ');
      if (here.length) await h.filterCells([{col:'label', values:here}]);
      note.textContent = chosen.size.toLocaleString() + ' cells selected (' + top + ')' + (roi ? '; ' + here.length + ' of them in the open image, shown on it.' : '. Open one of these images to see them.');
    };
    draw();
  }
  async function agreement(current, pane) {
    const known = await info();
    const box = section('Method agreement', {tip:'The F1 paper figures’ measures: a phenotyping method compared cell by cell with a reference (a manual or trusted column). Precision, recall and F1 per phenotype; the table shows where each reference phenotype went. Click a reference phenotype to show its cells on the image coloured by the method (mismatches stand out).'});
    const reference = columnPick(known.groups, prefer(known.groups, ['phenotype_ref1']), 'Reference column');
    const method = columnPick(known.groups, prefer(known.groups, ['phenotype_ml_consensus']), 'Method column'), scope = scopeChoice();
    const said = node('p', {class:'muted', role:'status'}), table = node('div', {class:'studio-samples'});
    box.body.append(field('Reference', reference), field('Method', method), field('Images', scope), said, table);
    pane.append(box.box);
    current.runBar([node('div', {class:'row'}, [node('span', {class:'lbl', text:'Agreement:'}), node('button', {class:'btn primary', text:'Compare', onclick:async () => {
      said.textContent = 'Comparing…';
      try {
        const found = await api('agreement', {reference:reference.value, method:method.value, images:await scopeImages(scope)});
        said.textContent = found.cells.toLocaleString() + ' cells with both · accuracy ' + (100 * found.accuracy).toFixed(1) + '% · mean F1 ' + found.macro_f1.toFixed(3);
        table.replaceChildren(node('table', {class:'studio-counts'}, [node('tr', {}, ['Phenotype', 'Cells', 'Precision', 'Recall', 'F1'].map(t => node('th', {text:t}))),
          ...found.phenotypes.map(p => node('tr', {}, [p.phenotype, p.cells.toLocaleString(), p.precision.toFixed(3), p.recall.toFixed(3), p.f1.toFixed(3)].map(t => node('td', {text:t}))))]));
        h.heatmapFigure({key:'viz-agreement', title:found.reference + ' (rows) → ' + found.method + ' (columns), % of row', profile:{rows:found.confusion.rows, columns:found.confusion.columns,
          values:found.confusion.values.map(r => r.map(v => (v - 50) / 20)), raw:found.confusion.values},
          tip:'Each row: where the cells of that reference phenotype went (hover for the %). Click a row to show those cells on the image coloured by the method.',
          onPick:async rows => {
            const shown = await h.filterCells(rows.length ? [{col:found.reference, values:rows}] : [], found.method);
            said.textContent = shown ? (rows.length ? 'Cells called ' + rows.join(', ') + ' in ' + found.reference + ', coloured by ' + found.method + '.' : 'Every cell, coloured by ' + found.method + '.') : 'Open an image whose cells are in the combined table.';
          }});
      } catch (error) { said.textContent = error.message; }
    }}), help('Compares the two columns and draws the table beside the image, linked to it.')]), said]);
  }

  return {tlsCandidates, density, triads, screening, whereFigure, heatmap, proportions, umap:umapPanel, agreement, info};
};
