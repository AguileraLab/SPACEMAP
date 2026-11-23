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
  async function runJob(kind, body, said, onProgress) {
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
      if(onProgress){const progress=await api('screening_progress',{job_id:id});if(progress.result)onProgress(progress.result);}
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
    const reviewPicked=new Set();
    const flagTLS=node('button',{class:'btn small',text:'Flag for review',disabled:true,onclick:async()=>{flagTLS.disabled=true;try{await window.SpacemapReview.flagResult(result.job_id,[...reviewPicked]);}finally{flagTLS.disabled=!reviewPicked.size;}}});
    const drawRows = () => {
      if (!result || !rows || !imagePick) return;
      const structures = result.structures.filter(s => s.image === imagePick.value);
      rows.replaceChildren(structures.length ? node('table', {class:'studio-counts'}, [node('tr', {}, ['','Region / ID','B / T','Area (µm²)','B/T per mm²',''].map(text => node('th',{text}))),
        ...structures.map(s => node('tr', {'data-tls':s.key, class:s.key===selectedKey ? 'selected' : '', onclick:()=>show(s.image,s,true)}, [node('td',{},[node('input',{type:'checkbox','aria-label':'Select TLS '+s.key,checked:reviewPicked.has(s.key),onclick:e=>e.stopPropagation(),onchange:e=>{e.target.checked?reviewPicked.add(s.key):reviewPicked.delete(s.key);flagTLS.disabled=!reviewPicked.size;}})]),node('td',{text:s.region+' / '+s.id+(s.reviewed?' · edited':'')}),node('td',{text:s.b_cells+' / '+s.t_cells}),
          node('td',{text:fmt(s.area_um2)}),node('td',{text:fmt(s.density_per_mm2)}),node('td',{},[node('button',{class:'btn small',text:'Review',title:'Zoom to this TLS; keep every boundary visible',onclick:e=>{e.stopPropagation();show(s.image,s,true);}})])]))])
        : node('p', {class:'muted', text:!Object.hasOwn(result.counts,imagePick.value) ? 'This image is not in this TLS run.' : result.measured[imagePick.value] ? 'No candidates passed these settings.' : 'No cells with valid coordinates and tissue regions were analysed.'}));
      if(structures.length){
        const all=node('input',{type:'checkbox','aria-label':'Select all TLS candidates in this image',onchange:()=>{structures.forEach(s=>all.checked?reviewPicked.add(s.key):reviewPicked.delete(s.key));drawRows();}});
        all.checked=structures.every(s=>reviewPicked.has(s.key));all.indeterminate=!all.checked&&structures.some(s=>reviewPicked.has(s.key));
        rows.querySelector('th').append(all);flagTLS.disabled=!reviewPicked.size;rows.append(flagTLS);
      }
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
    const tips={'Cell types':'Select the cell types whose features you want to measure. Other types remain in the all-cell denominator and spatial graph.',
      'Image ID':'The column identifying each image. Coordinates from different images are never mixed.',
      'X':'The horizontal cell-centroid coordinate column.','Y':'The vertical cell-centroid coordinate column.',
      'Coordinates':'Units already stored in the X/Y columns. Pixel coordinates are converted using the image calibration.',
      'Annotations':'Folder containing one corrected GeoJSON per image. These files are read only.',
      'Distance from':'Exact annotation class whose boundary defines the distance zones, for example a target tissue compartment.',
      'Out of':'The denominator for the percentage. Choose all zone cells or a named population containing the target types.',
      'Group A':'The reference group. Positive effects mean higher values in Group B.',
      'Group B':'The group compared with A. Both groups must contain enough independent biological samples.',
      'Outcome':'A numeric value per independent sample. Tests whether it rises or falls with the measured feature.',
      'Time':'Positive follow-up duration, using one consistent time unit.',
      'Event':'A column coded 1 for an event and 0 for censoring; tick the option below only if this coding is reversed.',
      'Cells / zone':'Skip a zone with fewer cells than this. An unmeasurable zone stays missing; it is not treated as zero.',
      'Samples / group':'Minimum independent samples with valid measurements needed in each group for a test.',
      'q cutoff':'Threshold for highlighting findings after correction for the complete search. Default 0.05.',
      'Permutations':'Number of group-label shuffles for the permutation group test, paired test or numeric-outcome test. Separate from spatial random-label shuffles.',
      'Seed':'Starting seed for reproducible randomization with the same settings, data and compute backend.',
      'Show':'Filter the saved findings and their figures by feature type. This does not change or rerun the analysis.'};
    const field=(label,control,tip)=>h.field(label,control,tip||tips[label]);
    const [known, initial] = await Promise.all([api('info'), get(route('/spatial/screening_settings'))]);
    if (!pane.isConnected) return;
    let saved = initial, v = structuredClone(initial.values), generation = 0, showDesign = null;
    const said = node('p', {class:'muted', role:'status'}), badge = chip('Saved', 'ok');let invalidatePlan=()=>{};
    const mark = () => {const dirty=JSON.stringify(v)!==JSON.stringify(saved.values);badge.textContent=dirty?'Not saved':'Saved';badge.className='chip '+(dirty?'warn':'ok');showDesign?.();invalidatePlan();};
    const save = async () => {const sent=structuredClone(v);saved=await api('screening_settings',{values:sent,revision:saved.revision});mark();};
    const fail = error => {if(pane.isConnected)said.textContent=error.message;};
    const pair = (...children) => node('div',{class:'screening-grid'},children);
    const details = (title,...children) => node('details',{class:'studio-options'},[node('summary',{text:title}),...children]);
    const bind = (object,key,control,numeric=false) => {control.onchange=()=>{object[key]=numeric?Number(control.value):control.value;mark();};return control;};
    const pick = (key,label,optional=false) => bind(v,key,choice([['',optional?'None':'Choose…'],...known.columns.map(c=>[c,c])],v[key],label));
    const select = (object,key,options,label) => bind(object,key,choice(options,object[key],label));
    const num = (object,key,label,min='0',step='any') => bind(object,key,number(object[key],label,step,min),true);
    const text = (object,key,label) => bind(object,key,node('input',{type:'text',value:object[key],'aria-label':label}));
    const tick = (object,key,label) => {const input=node('input',{type:'checkbox','aria-label':label});input.checked=object[key];input.onchange=()=>{object[key]=input.checked;mark();};return node('label',{class:'row'},[input,node('span',{text:label})]);};
    const chooser = (names,selected,set,label) => {
      const summary=node('summary'), box=node('details',{class:'threshold-picker','aria-label':label},[summary]);
      const search=node('input',{type:'search',placeholder:'Find…','aria-label':'Find '+label}), all=node('input',{type:'checkbox','aria-label':'All matching '+label});
      const list=node('div',{class:'studio-markers'});
      const chosen=new Set(selected);
      const update=()=>{summary.textContent=chosen.size+' selected';const shown=[...list.children].filter(r=>!r.hidden);all.checked=!!shown.length&&shown.every(r=>r.querySelector('input').checked);all.indeterminate=shown.some(r=>r.querySelector('input').checked)&&!all.checked;};
      names.forEach(name=>{const cb=node('input',{type:'checkbox','aria-label':label+': '+name});cb.checked=chosen.has(name);cb.onchange=()=>{cb.checked?chosen.add(name):chosen.delete(name);set([...chosen]);update();mark();};list.append(node('label',{title:name},[cb,node('span',{text:name})]));});
      search.oninput=()=>{[...list.children].forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(search.value.toLowerCase()));update();};
      all.onchange=()=>{[...list.children].filter(r=>!r.hidden).forEach(r=>{const cb=r.querySelector('input');cb.checked=all.checked;all.checked?chosen.add(r.textContent):chosen.delete(r.textContent);});set([...chosen]);update();mark();};
      box.append(node('div',{class:'row'},[all,search]),list);update();return box;
    };
    const study=section('1. Cells and regions',{actions:[badge,node('button',{class:'btn small',text:'Save',onclick:()=>save().catch(fail)})],tip:'Tick images above, choose cell types, then choose where to measure. Filters can restrict the scan to a tissue compartment. Other cell types remain available as neighbours and denominators.'});
    const labels=node('div'), phenotype=pick('phenotype_col','Screening cell type column');
    let labelsGeneration=0;
    const loadLabels=async()=>{const g=++labelsGeneration;labels.replaceChildren();if(!v.phenotype_col)return;const r=await api('tls_labels',{column:v.phenotype_col});if(!pane.isConnected||g!==labelsGeneration)return;labels.append(field('Cell types',chooser(r.labels,v.phenotypes,x=>v.phenotypes=x,'Screening cell types')));};
    phenotype.onchange=()=>{v.phenotype_col=phenotype.value;v.phenotypes=[];mark();loadLabels().catch(fail);};
    study.body.append(field('Cell-type column',phenotype,'The column containing cell labels, such as epithelial, T cell or fibroblast. Choose the types to investigate below.'),labels);
    const filters=node('div');
    const drawFilters=()=>{filters.replaceChildren(...v.filters.map((rule,index)=>{const col=choice([['','Column…'],...known.columns.map(c=>[c,c])],rule.column,'Screening filter column '+index),holder=node('span');
      let request=0;const fill=async()=>{const token=++request;holder.replaceChildren();if(!rule.column)return;const r=await api('tls_labels',{column:rule.column});if(token!==request||!pane.isConnected)return;holder.append(chooser(r.labels,rule.values,x=>rule.values=x,'Screening region values '+index));};
      col.onchange=()=>{rule.column=col.value;rule.values=[];mark();fill().catch(fail);};fill().catch(fail);
      return node('div',{class:'row'},[col,holder,node('button',{class:'btn small',text:'×',title:'Remove region filter',onclick:()=>{v.filters.splice(index,1);mark();drawFilters();}})]);}));};
    study.body.append(details('Region filters',filters,node('button',{class:'btn small',text:'Add filter',onclick:()=>{v.filters.push({column:'',values:[]});mark();drawFilters();}})));
    drawFilters();loadLabels().catch(fail);
    const geometry=pick('region_col','Screening separate regions',true);
    study.body.append(details('Coordinates and units',pair(field('Image ID',pick('image_col','Screening image column')),field('Separate pieces',geometry,'Optional region IDs; neighbours cannot connect between different pieces. Filter unwanted IDs explicitly.')),
      pair(field('X',pick('x_col','Screening X column')),field('Y',pick('y_col','Screening Y column'))),
      pair(field('Coordinates',select(v,'coordinate_unit',[['px','Pixels'],['um','µm']],'Screening coordinate units')),field('µm / pixel',num(v,'pixel_um','Screening pixel size'),'0 = use each image’s saved calibration. A positive value applies to every selected image.'))));
    const zones={body:node('div')};
    const source=select(v,'zone_source',[['distance','Saved distance column'],['polygons','Annotation polygons'],['whole','Whole filtered region']],'Screening zone source');
    const distance=pick('distance_col','Screening distance column');
    const distanceUnit=select(v,'distance_unit',[['','Choose units…'],['px','Pixels'],['um','µm']],'Screening distance units');
    distance.onchange=()=>{v.distance_col=distance.value;const u=initial.units?.[distance.value];if(['px','pixels','pixel'].includes(u))v.distance_unit='px';else if(['um','µm','micrometers','micrometres'].includes(u))v.distance_unit='um';else v.distance_unit='';distanceUnit.value=v.distance_unit;mark();};
    const distrow=pair(field('Distance',distance,'The saved distance from a region boundary, for example a tumour annotation. Negative values are inside; positive values are outside.'),field('Stored unit',distanceUnit,'The unit of values already in that column. Scan distances below are always µm; pixel distances are converted using the image calibration.'));
    const poly= node('div',{},[field('Annotations',text(v,'annotation_folder','Screening annotation folder')),field('File pattern',text(v,'annotation_pattern','Screening annotation pattern'),'One exact file per image, e.g. {image}.geojson. Sources are not combined across folders.'),pair(field('Sampled region',text(v,'study_class','Screening sampled-region class'),'The area actually imaged and eligible for analysis; holes are excluded. For a bed-restricted density choose the bed polygon class.'),field('Distance from',text(v,'target_class','Screening distance-target class')))]);
    const widths=node('input',{type:'text',value:v.widths.join(', '),'aria-label':'Screening zone widths'});widths.onchange=()=>{v.widths=widths.value.split(/[ ,;]+/).filter(Boolean).map(Number);mark();};
    const zoneOptions=node('div',{},[pair(field('Zones',select(v,'zone_mode',[['cumulative','Growing zones'],['rings','Separate rings']],'Screening zone mode'),'50, 100 gives 0–50 and 0–100 µm with Growing zones, or 0–50 and 50–100 µm with Separate rings.'),field('Side',select(v,'side',[['outside','Outside'],['inside','Inside'],['both','Both sides']],'Screening zone side'),'Choose the outside, inside, or both sides of the region boundary. Zero-distance boundary cells enter the first zone.')),field('Distances (µm)',widths,'Increasing distances to scan, separated by commas. Choose a useful range before looking at the results; every distance adds statistical comparisons.')]);
    const syncZones=()=>{distrow.hidden=v.zone_source!=='distance';poly.hidden=v.zone_source!=='polygons';zoneOptions.hidden=v.zone_source==='whole';};
    source.onchange=()=>{v.zone_source=source.value;mark();syncZones();};syncZones();
    zones.body.append(field('Where to measure',source,'Use a saved distance column, calculate distances and areas from annotation polygons, or measure the whole filtered region.'),distrow,poly,zoneOptions);
    study.body.append(zones.body);
    const features=section('2. What to measure',{tip:'Select one or all seven. Open Settings beside a feature to adjust it. Start with the measurements relevant to your question: every added type and distance increases the number of tests.'});
    const familyNames={density:'Density',proportion:'Proportion',ratio:'Ratios',diversity:'Diversity',nn_distance:'Neighbour distance',dyad:'Pairs (dyads)',triad:'Triplets (triads)'};
    const diversityNames={shannon:'Shannon',simpson:'Simpson',richness:'Number of types',evenness:'Evenness'};
    const familyList=node('div'),allFeatures=node('input',{type:'checkbox','aria-label':'All seven features'});
    const drawFamilies=()=>{allFeatures.checked=v.families.length===7;allFeatures.indeterminate=!!v.families.length&&v.families.length<7;familyList.replaceChildren(...Object.entries(familyNames).map(([key,label])=>{
      const cb=node('input',{type:'checkbox','aria-label':'Screen '+label});cb.checked=v.families.includes(key);
      const opts=node('div',{class:'screening-feature-options'}),opt=v.options[key];opts.hidden=!cb.checked;
      cb.onchange=()=>{v.families=cb.checked?[...v.families,key]:v.families.filter(f=>f!==key);mark();drawFamilies();};
      if(key==='density'){
        const area=select(opt,'area_source',[['polygons','Annotation polygons (measured)'],['estimate','Cell footprint (Classic estimate)'],['csv','Area file (CSV)']],'Screening density area source');
        const file=field('Area file',text(v,'area_table','Screening zone area CSV'),'Optional alternative to polygons. One row per sample, group, image, region and zone (0-based), with area_mm2. Use All for unused group/region. The area must match the measured zone, including holes and exclusions.');
        const estimate=field('Neighbour percentile',num(opt,'area_percentile','Screening area percentile','1'),'This chooses the size of circles drawn around cell centres to estimate occupied area. 99.9 uses almost the largest nearest-neighbour gap, not 99.9% accuracy. Larger circles can include empty space and lower the estimated density. These circles are not clipped to the bed polygon or distance-zone boundary; use measured polygons for that area.');
        const updateArea=()=>{file.hidden=opt.area_source!=='csv';estimate.hidden=opt.area_source!=='estimate';};
        area.onchange=()=>{opt.area_source=area.value;if(area.value==='polygons'){v.zone_source='polygons';source.value='polygons';syncZones();}mark();updateArea();};updateArea();
        opts.append(field('Area',area,'Density = cells divided by the area they occupy. Annotation polygons measure the sampled zone directly: set Where to measure to Annotation polygons. Classic estimate needs no area file. Polygon zones always use their measured area.'),file,estimate);
      }
      if(key==='proportion'){
        const denominator=select(opt,'denominator',[['all','All cells in the zone'],['selected','Chosen denominator types']],'Screening proportion denominator');
        const values=text(opt,'labels','Screening denominator labels');values.value=opt.labels.join(', ');values.placeholder='Comma-separated cell types';values.hidden=opt.denominator!=='selected';
        values.onchange=()=>{opt.labels=values.value.split(',').map(s=>s.trim()).filter(Boolean);mark();};denominator.onchange=()=>{opt.denominator=denominator.value;values.hidden=denominator.value!=='selected';mark();};
        opts.append(field('Out of',denominator),values);
      }
      if(key==='ratio')opts.append(pair(field('Pseudocount',num(opt,'pseudocount','Screening ratio pseudocount'),'Added to both counts. Zero leaves undefined denominators missing.'),tick(opt,'log2','Log2 ratios')));
      if(key==='diversity'){
        const chosen=new Set(opt.metrics||[opt.metric||'shannon']),all=node('input',{type:'checkbox','aria-label':'All four diversity measures'}),list=node('div',{class:'studio-markers'});
        const sync=()=>{all.checked=chosen.size===4;all.indeterminate=chosen.size>0&&chosen.size<4;};
        Object.entries(diversityNames).forEach(([metric,name])=>{const input=node('input',{type:'checkbox','aria-label':'Screen diversity '+name});input.checked=chosen.has(metric);
          input.onchange=()=>{input.checked?chosen.add(metric):chosen.delete(metric);opt.metrics=Object.keys(diversityNames).filter(m=>chosen.has(m));sync();mark();};
          list.append(node('label',{},[input,node('span',{text:name})]));});
        all.onchange=()=>{opt.metrics=all.checked?Object.keys(diversityNames):[];chosen.clear();opt.metrics.forEach(m=>chosen.add(m));[...list.querySelectorAll('input')].forEach(input=>input.checked=all.checked);sync();mark();};sync();
        opts.append(node('div',{class:'row'},[node('label',{class:'row'},[all,node('span',{text:'All four'})]),help('Select any or all measures. Shannon describes the mixture; Simpson is the chance that two independent draws have different types; Number of types counts the types found; Evenness describes how balanced they are. Only the selected cell types enter this mixture. Every selected measure and zone is included in the statistical correction.')]),list);
      }
      if(key==='nn_distance')opts.append(pair(field('Neighbours',num(opt,'k','Screening neighbour count','1','1'),'Mean distance from each source cell to this many nearest target cells. 1 means the closest target.'),field('Targets in',select(opt,'targets',[['zone','Same zone'],['region','Whole filtered region']],'Screening neighbour target scope'),'Same zone measures distances among cells in that zone. Whole filtered region also considers targets outside the zone. No target gives a missing value.')),node('div',{class:'row'},[tick(opt,'same_type','Include same-type neighbours'),help('Also measure each type’s distance to its own type, excluding the source cell itself.')]));
      if(['dyad','triad'].includes(key)){
        const metric=select(opt,'metric',[['zscore','Beyond chance (Classic Z-score)'],['count','Number found'],['fraction','% of all pairs / triplets'],['enrichment','Log2 observed / expected']],'Screening '+key+' measure');
        const pseudo=field('Pseudocount',num(opt,'pseudocount','Screening '+key+' pseudocount'),'Used only for log2 observed / expected. Added to observed and expected counts to stabilize rare motifs.');pseudo.hidden=opt.metric!=='enrichment';
        metric.onchange=()=>{opt.metric=metric.value;pseudo.hidden=opt.metric!=='enrichment';mark();};
        opts.append(pair(field('Min (µm)',num(opt,'min_um','Screening '+key+' minimum'),'Exclude pairs closer than this distance.'),field('Max (µm)',num(opt,'max_um','Screening '+key+' maximum'),'Maximum distance between two cells. For triplets all three pairs must be within the min–max range.')),
          field('Measure',metric,'Beyond chance uses Classic’s random-label shuffle method: positions and type counts stay fixed while labels are shuffled within each zone. Positive Z means more co-occurrence than expected. This Z is a feature value; the p and q in Results compare biological samples. It does not prove direct contact or a three-way mechanism.'),
          node('div',{class:'row'},[tick(opt,'same_type','Include repeated cell types'),help('Allow combinations such as A–A or A–A–B as well as combinations of different cell types.')]),pseudo);
      }
      const settings=details('Settings',opts);settings.hidden=!cb.checked;
      const purpose={density:'Cells per mm². Requires a matching zone area; measured polygons are preferred.',proportion:'Percentage of cells of each selected type in the zone.',ratio:'Relative numbers of two cell types, for example suppressor cells versus effector cells.',diversity:'How many cell types are present and how evenly represented they are.',nn_distance:'How far one cell type is from another, in µm.',dyad:'Nearby pairs of cells. Count them or ask whether they occur more often than under random labels.',triad:'Nearby groups of three cells with all three connections. Count them or compare with random labels.'};
      return node('div',{class:'screening-feature'},[node('div',{class:'row'},[node('label',{class:'row'},[cb,node('strong',{text:label})]),help(purpose[key])]),settings]);
    }));};
    allFeatures.onchange=()=>{v.families=allFeatures.checked?Object.keys(familyNames):[];mark();drawFamilies();};
    features.body.append(node('label',{class:'row'},[allFeatures,node('span',{text:'All seven'})]),familyList);drawFamilies();
    features.body.append(details('Advanced: random-label comparison',pair(field('Shuffles',num(v,'motif_permutations','Screening motif permutations','99','1'),'Used for Beyond chance pairs and triplets. More shuffles improve precision and take longer; 1,000 is Classic’s starting setting.'),field('Minimum found',num(v,'motif_min_count','Screening motif minimum count','0','1'),'Classic excludes Z-scores with fewer than this many observed pairs/triplets, or with no null variation. These stay missing, never zero.')),node('div',{class:'row'},[tick(v,'motif_use_gpu','Use GPU when available'),help('Uses the same CUDA permutation engine as Classic, with CPU fallback. The actual backend is saved in Run details. CPU and GPU use different random sequences, so Monte Carlo Z-scores can differ slightly.')])));
    const compare=section('3. What to compare',{tip:'Each dot is one independent biological sample. Several images from the same sample are summarized first. Every selected feature and distance is included in the multiple-testing correction.'});
    compare.body.append(field('Independent sample ID',pick('sample_col','Screening sample column'),'Choose the independent person, animal or specimen. Two visits from one person still count as ONE person. A pooled comparison summarizes their eligible images together. Use Split comparisons to study each visit separately, matched groups to compare visits within a person, or Also compare changes to compare before/after changes between groups. Do not use a cell ID or give repeated visits new independent IDs.'));
    const e=v.endpoint,mode=select(e,'type',[['binary','Two groups'],['continuous','Numeric outcome'],['survival','Survival'],['describe','Describe only']],'Screening comparison');
    const groupCol=bind(e,'group_col',choice([['','Choose…'],...known.columns.map(c=>[c,c])],e.group_col,'Screening comparison column'));
    const groupValues=node('div');let groupGeneration=0;
    const loadGroups=async()=>{const g=++groupGeneration;groupValues.replaceChildren();if(!e.group_col)return;const r=await api('tls_labels',{column:e.group_col});if(g!==groupGeneration||!pane.isConnected)return;
      groupValues.append(pair(field('Group A',select(e,'group_a',[['','Choose…'],...r.labels.map(s=>[s,s])],'Screening group A')),field('Group B',select(e,'group_b',[['','Choose…'],...r.labels.map(s=>[s,s])],'Screening group B'))));};
    groupCol.onchange=()=>{e.group_col=groupCol.value;e.group_a='';e.group_b='';mark();loadGroups().catch(fail);};
    const binary=node('div',{},[field('Compare by',groupCol,'Choose the column containing the two groups you want to compare, such as treated and untreated. Then select Group A and Group B. Positive effects mean higher values in B.'),groupValues,node('div',{class:'row'},[tick(e,'paired','Match groups within each sample'),help('Tick only when the same biological samples occur in both groups, such as before and after treatment. Only samples with both measurements are compared. Leave unticked for separate groups of people or specimens.')])]);
    const ecol=(key,label)=>bind(e,key,choice([['','Choose…'],...known.columns.map(c=>[c,c])],e[key],label));
    const continuous=field('Outcome',ecol('value_col','Screening numeric outcome'));
    const survival=node('div',{},[pair(field('Time',ecol('time_col','Screening survival time')),field('Event',ecol('event_col','Screening survival event'))),tick(e,'censored','1 means censored'),help('Continuous-feature Cox model; no optimized high/low cutoff. At least 10 samples and 5 events are needed. Review proportional-hazards diagnostics in the results CSV.')]);
    const syncCompare=()=>{binary.hidden=e.type!=='binary';continuous.hidden=e.type!=='continuous';survival.hidden=e.type!=='survival';};mode.onchange=()=>{e.type=mode.value;mark();syncCompare();};syncCompare();loadGroups().catch(fail);
    compare.body.append(field('Comparison',mode,'Compare two groups, test association with a numeric outcome, analyse survival, or describe measurements without significance tests.'),binary,continuous,survival,details('Advanced: coverage and statistics',
      pair(field('Cells / zone',num(v,'min_cells','Screening minimum cells','1','1')),field('Samples / group',num(v,'min_samples','Screening minimum samples','2','1'))),
      field('Image summary',select(v,'aggregation',[['median','Median within sample'],['mean','Mean within sample']],'Screening sample aggregation'),'Each image/piece is measured separately. Images are equally weighted within the sample; this is not pooled tissue density.'),
      field('Group test',select(v,'group_test',[['mannwhitney','Mann–Whitney (Classic)'],['permutation','Mann–Whitney permutation']],'Screening group test'),'Independent two-group comparisons. Classic uses the two-sided Mann–Whitney test. The permutation alternative handles small cohorts and ties by shuffling group labels. Matched groups use paired sign flips.'),
      field('Correct together',select(v,'correction_scope',[['family','Each feature type · all zones'],['global','Whole scan · all types and zones'],['classic','Each feature type and zone · Classic']],'Screening correction scope'),'Choose before interpreting the results. Each feature type keeps Density separate from Proportion, etc., and includes all its scanned distances. Whole scan combines every type and distance. Classic corrects each type separately at each distance, excluding untestable comparisons; it does not adjust for searching across distances or additional comparisons. Per-feature-type and whole-scan scopes include all additional comparisons. Exports retain all three q-values; the displayed q follows this choice.'),
      field('Correction',select(v,'correction',[['fdr_bh','FDR (Benjamini–Hochberg)'],['fdr_by','FDR (arbitrary dependence)'],['bonferroni','Bonferroni']],'Screening multiplicity correction'),'p tests one comparison. q adjusts across the comparisons in Correct together. q=1 is possible; it does not prove that groups are identical. Nested zones are dependent; BY or Bonferroni are more conservative.'),
      pair(field('q cutoff',num(v,'alpha','Screening q cutoff','0.0001')),field('Permutations',num(v,'permutations','Screening comparison permutations','999','1'))),field('Seed',num(v,'seed','Screening random seed','0','1'))));
    const resampling=select(v,'resampling',[['off','Off'],['subsample','Sample subsets · Classic stability'],['bootstrap','Bootstrap · with replacement']],'Screening resampling'),resampleSettings=node('div');
    const fraction=field('Fraction kept',num(v,'sample_fraction','Screening subset fraction','.5','.05'),'0.8 keeps 80% of biological samples per repeat, without replacement. This uses Classic’s subset sampler. Repeated images from a sample stay together.');
    resampleSettings.append(pair(field('Repeats',num(v,'resamples','Screening resampling repeats','20','1')),fraction),field('Nearby zones',num(v,'nearby_zones','Screening nearby zones','0','1'),'Recovery requires q below your cutoff and the same effect direction. Exact recovery uses this distance; nearby recovery also allows this many adjacent scanned distances. This measures sensitivity to sample selection, not independent validation.'));
    const syncResampling=()=>{resampleSettings.hidden=v.resampling==='off';fraction.hidden=v.resampling!=='subsample';};resampling.onchange=()=>{v.resampling=resampling.value;mark();syncResampling();};syncResampling();
    compare.body.append(details('Stability and uncertainty',field('Resampling',resampling,'Classic’s stability check uses subsets without replacement. Bootstrap draws whole biological samples with replacement and also estimates a 95% interval for the effect. For one comparison, independent groups are resampled separately. Multiple comparisons share one draw of whole independent samples, keeping visits and overlapping subgroups together. Spatial graphs are reused. Intervals are pointwise and do not validate a feature chosen from this scan.'),resampleSettings));
    const extras=node('div'),planBox=node('div',{class:'studio-samples'});invalidatePlan=()=>planBox.replaceChildren();
    const drawExtras=()=>{extras.replaceChildren();v.extra_comparisons.forEach((item,index)=>{
      const col=select(item,'group_col',[['','Choose…'],...known.columns.map(n=>[n,n])],'Additional comparison column '+(index+1)),pairs=node('div');let serial=0;
      const load=async()=>{const token=++serial;pairs.replaceChildren();if(!item.group_col)return;const found=await api('tls_labels',{column:item.group_col});if(token!==serial||!pairs.isConnected)return;
        pairs.append(pair(field('Group A',select(item,'group_a',[['','Choose…'],...found.labels.map(n=>[n,n])],'Additional group A '+(index+1))),field('Group B',select(item,'group_b',[['','Choose…'],...found.labels.map(n=>[n,n])],'Additional group B '+(index+1)))));};
      col.onchange=()=>{item.group_col=col.value;item.group_a='';item.group_b='';mark();load().catch(fail);};
      const all=tick(item,'all_pairs','All pairs in comparison '+(index+1));all.querySelector('input').onchange=ev=>{item.all_pairs=ev.target.checked;pairs.hidden=item.all_pairs;mark();};pairs.hidden=item.all_pairs;
      const box=details('Comparison '+(index+1),field('Compare by',col),node('div',{class:'row wrap'},[all,tick(item,'paired','Match within samples in comparison '+(index+1)),help('All pairs tests every pair of values in this chosen column (up to 12 values). Match within samples only when the same independent IDs are measured in both groups. Groups with too few independent samples are listed as skipped.'),node('button',{class:'btn small',text:'Remove',onclick:()=>{v.extra_comparisons.splice(index,1);mark();drawExtras();}})]),pairs);box.open=true;extras.append(box);load().catch(fail);
    });};
    drawExtras();
    const change=v.change,changeFields=node('div'),visitColumn=select(change,'column',[['','Choose…'],...known.columns.map(n=>[n,n])],'Screening repeated observation column'),visits=node('div');let visitGeneration=0;
    const loadVisits=async()=>{const token=++visitGeneration;visits.replaceChildren();if(!change.column)return;const found=await api('tls_labels',{column:change.column});if(token!==visitGeneration||!pane.isConnected)return;visits.append(pair(field('Before',select(change,'before',[['','Choose…'],...found.labels.map(n=>[n,n])],'Screening before observation')),field('After',select(change,'after',[['','Choose…'],...found.labels.map(n=>[n,n])],'Screening after observation'))));};
    visitColumn.onchange=()=>{change.column=visitColumn.value;change.before='';change.after='';mark();loadVisits().catch(fail);};
    const changeTick=tick(change,'enabled','Also compare changes');changeTick.querySelector('input').onchange=ev=>{change.enabled=ev.target.checked;changeFields.hidden=!change.enabled;mark();};changeFields.hidden=!change.enabled;
    changeFields.append(field('Observation',visitColumn,'Choose the column identifying repeated observations. For each independent sample, summarize each observation first, then subtract before from after. Compare these changes between the primary independent groups. Both observations are required; this does not match individual cells across images.'),visits);loadVisits().catch(fail);
    const builder=details('More comparisons',field('Split comparisons by',chooser(known.columns,v.split_by,value=>v.split_by=value,'Screening splitting columns'),'Select up to three grouping columns. The primary comparison is repeated within each observed group. For example, selecting visit and treatment can compare outcomes within each visit, each treatment, and their combinations. IDs and continuous measurements are not suitable splitting columns.'),
      pair(field('Combine',select(v,'split_depth',[[1,'One column at a time'],[2,'Up to two columns together'],[3,'Up to three columns together']],'Screening splitting depth')),tick(v,'include_overall','Include pooled comparison')),
      extras,node('button',{class:'btn small',text:'Add comparison',onclick:()=>{if(v.extra_comparisons.length>=5){fail(Error('Add up to five comparison columns.'));return;}v.extra_comparisons.push({group_col:'',group_a:'',group_b:'',all_pairs:true,paired:false});mark();drawExtras();}}),
      node('div',{class:'row'},[changeTick,help('Adds a comparison of after-minus-before changes between the primary groups. A group difference at one visit but not another does not itself establish a difference in change. Splitting by the observation column is omitted for this change comparison.')]),changeFields,
      node('button',{class:'btn small',text:'Preview comparisons',onclick:async()=>{try{const snapshot=JSON.stringify(v),images=[...(current.spatialPicked||[])];const found=await api('screening_plan',{values:structuredClone(v),images});if(!pane.isConnected||snapshot!==JSON.stringify(v))return;planBox.replaceChildren(node('table',{class:'studio-counts'},[node('thead',{},[node('tr',{},['Comparison','Samples','State'].map(t=>node('th',{text:t})))]),node('tbody',{},found.comparisons.map(p=>node('tr',{title:p.reason||'Counts before cell-zone coverage checks'},[p.label,Object.entries(p.samples).map(([k,n])=>k+': '+(typeof n==='object'?Object.entries(n).map(([g,count])=>g+' '+count).join(', '):n)).join(' · '),p.reason||p.status].map(t=>node('td',{text:t})))))]));}catch(error){fail(error);}}}),help('Uses the images ticked above. Review this declared list before running. Whole-scan and per-feature-type correction include all generated comparisons and zones. Classic scope corrects each comparison and distance separately. Small subgroups stay untestable; no search can guarantee a significant finding.'),planBox);
    builder.querySelector('[aria-label="Screening splitting depth"]').onchange=ev=>{v.split_depth=Number(ev.target.value);mark();};
    compare.body.insertBefore(builder,compare.body.querySelector('details'));
    const results=section('Results',{tip:'Effects and adjusted q values over the complete search. No winning distance is called optimal. Tables include excluded and untestable sample-feature measurements. Click a finding to review its figure and source cells.'});
    const resultBody=node('div'),runPicker=choice([['','Choose a saved run…']],'','Screening saved run');
    const refreshRuns=async()=>{const found=await api('screening_runs');if(!pane.isConnected)return;const selected=latestResult?.run||runPicker.value;runPicker.replaceChildren(node('option',{value:'',text:'Choose a saved run…'}),...found.runs.map(r=>node('option',{value:r.run,text:r.run+' · '+r.samples+' samples'})));runPicker.value=selected;};
    const download=(result,name,label)=>node('a',{class:'btn small',text:label,href:h.apiUrl(route('/spatial/screening_file'))+'?run='+encodeURIComponent(result.run)+'&comparison='+encodeURIComponent(result.comparison_id||'')+'&name='+encodeURIComponent(name),target:'_blank'});
    const zoneLabel=(c,j)=>c.zone_source==='whole'?'Whole region':(c.zone_mode==='rings'&&j?c.widths[j-1]:0)+'–'+c.widths[j]+' µm';
    const scopeNames={global:'Whole scan',family:'Each feature type · all zones',classic:'Each feature type and zone · Classic'};
    const presentation=window.spacemapScreeningResults;
    const directionBadge=(endpoint,effect,palette)=>{const found=presentation.direction(endpoint,effect);return node('span',{class:'screening-direction',title:found.label},[...(found.group!==null?[node('span',{class:'screening-group-dot',style:'background:'+presentation.colour(found.group,endpoint,palette),'aria-hidden':'true'})]:[]),node('span',{text:found.label})]);};
    const view={run:'',comparison:'',panelKey:'',family:'',feature:'',zone:null,filter:'all',search:'',page:0,sort:'q',descending:false};let latestResult=null,latestRevision=null,displayGeneration=0,findingGeneration=0,comparisonGeneration=0;
    const inspect=async(result,row,image=h.openRoi(),zoom=false,observation={})=>{
      const token=++displayGeneration;
      try{
        if(image!==h.openRoi()){await showOnImage(image);if(token!==displayGeneration)return;}
        let payload;
        const shown=await h.showReviewedCells(async()=>{payload=await api('screening_cells',{run:result.run,comparison:result.comparison_id||'',image,feature:row.feature,zone:row.zone,...observation});return payload;},()=>pane.isConnected&&token===displayGeneration&&image===h.openRoi());
        if(!shown)return;
        h.showPreview('screening-zone',payload.geometry?{outlines:rings(payload.geometry),colour:'#ffd166',kind:'example'}:null);
        if(zoom&&payload.bounds)h.zoomBox?.(payload.bounds);
        said.textContent=shown.current.toLocaleString()+' context cells · '+image+' · '+zoneLabel(result.settings,row.zone);
      }catch(error){fail(error);}
    };
    let stabilityGeneration=0,stabilitySelection=0,closeStability=()=>{};
    const openStability=async result=>{
      closeStability();
      const session=++stabilityGeneration,box=node('div',{class:'screening-stability'}),status=node('span',{role:'status',text:'Loading saved tests…'}),controls=node('div'),grid=node('div',{class:'screening-stability-scroll'}),pager=node('div',{class:'row wrap'}),legend=node('div',{class:'row wrap'});
      let large=null,spot=null,observer=null;
      const expand=node('button',{class:'btn small ghost',text:'Large view','aria-label':'Expand stability matrix'});
      const closeLarge=()=>{observer?.disconnect();observer=null;if(large){if(spot?.isConnected)spot.replaceWith(box);else box.remove();large.remove();large=null;spot=null;expand.textContent='Large view';}};
      closeStability=closeLarge;
      expand.onclick=()=>{if(large){closeLarge();return;}spot=node('div',{class:'studio-figure-placeholder'});box.before(spot);large=node('div',{class:'studio-figure-large',role:'dialog','aria-label':'Feature stability across distances'});large.append(box);document.getElementById('stage').append(large);expand.textContent='Back to the pane';expand.focus();large.onkeydown=ev=>{if(ev.key==='Escape'){ev.preventDefault();closeLarge();expand.focus();}};observer=new MutationObserver(()=>{if(!spot?.isConnected)closeLarge();});observer.observe(spot.parentNode,{childList:true});};
      const options={run:result.run,comparison:'',family:'',search:'',only_fdr:false,offset:0,limit:20};let request=0,searchTimer;
      const link=(extension,label)=>node('a',{class:'btn small',text:label,target:'_blank',href:h.apiUrl(route('/spatial/screening_stability_file'))+'?'+new URLSearchParams({...options,format:extension})});
      const openCell=async(row,cell)=>{const intent=++stabilitySelection,runGeneration=generation,token=++comparisonGeneration;findingGeneration++;displayGeneration++;view.comparison=row.comparison;
        try{const found=await api('screening_result',{run:result.run,comparison:row.comparison});
          if(!found.hits.some(r=>r.feature===row.feature&&r.zone===cell.zone))found.hits.push(await api('screening_finding',{run:result.run,comparison:row.comparison,feature:row.feature,zone:cell.zone}));
          if(token!==comparisonGeneration||intent!==stabilitySelection||runGeneration!==generation||session!==stabilityGeneration||!pane.isConnected)return;
          Object.assign(view,{run:result.run,comparison:row.comparison,panelKey:result.run+'/'+row.comparison,family:row.family,feature:row.feature,zone:cell.zone,initialized:true,filter:'all',page:0,search:''});
          closeLarge();latestRevision=null;drawResult(found);h.revealFigure?.('screening-'+row.family);
        }catch(error){if(token===comparisonGeneration)fail(error);}
      };
      const render=found=>{
        status.textContent=found.total+' features · '+found.scope+' · '+(found.resampling==='bootstrap'?'Bootstrap recovery':found.resampling==='subsample'?'Subset recovery':'Not resampled');
        if(!controls.childElementCount){
          const comparison=choice([['','All comparisons'],...found.comparisons.map(p=>[p.id,p.label])],options.comparison,'Stability comparison'),family=choice([['','All feature types'],...Object.entries(familyNames)],options.family,'Stability feature type'),search=node('input',{type:'search',placeholder:'Find a feature or comparison…','aria-label':'Find stability features'}),filter=choice([['all','All features'],['fdr','With FDR-significant zones']],'all','Stability significance filter'),limit=choice([[10,'10 rows'],[20,'20 rows'],[40,'40 rows']],20,'Stability rows per page');
          const change=()=>{options.offset=0;load();};comparison.onchange=()=>{options.comparison=comparison.value;change();};family.onchange=()=>{options.family=family.value;change();};filter.onchange=()=>{options.only_fdr=filter.value==='fdr';change();};limit.onchange=()=>{options.limit=Number(limit.value);change();};search.oninput=()=>{options.search=search.value;options.offset=0;clearTimeout(searchTimer);request++;searchTimer=setTimeout(load,250);};
          controls.append(node('div',{class:'row wrap'},[comparison,family,filter,limit]),search);
        }
        legend.replaceChildren(node('span',{text:'Cell: higher group · exact recovery (%)'}),...Object.entries(found.states).map(([key,[label,colour,ink]])=>node('span',{class:'chip',style:'background:'+colour+';color:'+ink,text:label})),help('Cell text names the group with the higher estimated value, or the larger after-minus-before change. Survival arrows mean higher/lower event hazard, not a causal protective effect. Positive/Negative refer to numeric associations. The percentage is exact recovery at this distance, not nearby recovery. — means unavailable; 0% is a measured zero recovery. Hover or focus a cell for p, q, sample counts and testable repeats.'));
        const header=node('tr',{},[node('th',{text:'Rank · feature / comparison'}),...found.zones.map(z=>node('th',{text:z.label}))]);
        grid.replaceChildren(node('table',{class:'screening-stability-matrix'},[node('thead',{},[header]),node('tbody',{},found.rows.map(row=>{
          const title=row.name+'\n'+row.supported_zones+'/'+row.total_zones+' FDR zones in the same direction'+(row.mixed_fdr_directions?' · FDR directions change across zones':'');
          return node('tr',{},[node('th',{scope:'row',title},[node('strong',{text:row.rank+'. '+familyNames[row.family]+' · '+row.label}),node('span',{text:row.comparison_label}),node('span',{class:'muted',text:row.supported_zones+'/'+row.total_zones+' FDR zones'+(row.mixed_fdr_directions?' · mixed directions':'')})]),...found.zones.map(zone=>{
            const cell=row.cells.find(c=>c.zone===zone.zone);if(!cell)return node('td',{text:'—',class:'stability-missing'});
            const [state,colour,ink]=found.states[cell.state],recovery=cell.recovery===null?'—':Math.round(cell.recovery*100)+'%';
            const tip=row.name+'\n'+zone.label+' · '+cell.direction+'\n'+state+' · effect '+fmt(cell.effect)+' · p='+fmt(cell.p)+' · q='+fmt(cell.q)+'\n'+cell.samples+' independent samples'+(cell.n_a!==null?' · A '+cell.n_a+' / B '+cell.n_b:'')+'\nExact recovery '+recovery+' · nearby '+(cell.nearby_recovery===null?'—':Math.round(cell.nearby_recovery*100)+'%')+(cell.repeats!==null?' · '+cell.testable_repeats+'/'+cell.repeats+' testable repeats':' · resampling unavailable')+(cell.ci_low!==null&&cell.ci_low!==undefined?'\nEffect 95% bootstrap interval: '+fmt(cell.ci_low)+' to '+fmt(cell.ci_high):'')+'\n'+cell.status;
            return node('td',{},[node('button',{class:'screening-stability-cell','data-screening-comparison':row.comparison,'data-screening-feature':row.feature,'data-screening-zone':cell.zone,style:'background:'+colour+';color:'+ink,title:tip,'aria-label':tip,disabled:cell.state==='missing'?'':null,onclick:()=>openCell(row,cell)},[node('span',{class:'screening-direction'},[...(cell.group_colour?[node('span',{class:'screening-group-dot',style:'background:'+cell.group_colour,'aria-hidden':'true'})]:[]),node('span',{text:cell.short})]),node('strong',{text:recovery})])]);
          })]);
        }))]));
        if(!found.rows.length)grid.append(node('p',{text:'No features match these filters.'}));
        pager.replaceChildren(node('span',{text:(found.zone_source==='whole'?'Whole region':(found.zone_mode==='rings'?'Rings':'Cumulative zones')+' · '+found.side)+' · page '+(Math.floor(found.offset/found.limit)+1)+'/'+Math.max(1,Math.ceil(found.total/found.limit))}),node('button',{class:'btn small',text:'Previous',disabled:found.offset===0?'':null,onclick:()=>{options.offset=Math.max(0,found.offset-found.limit);load();}}),node('button',{class:'btn small',text:'Next',disabled:found.offset+found.limit>=found.total?'':null,onclick:()=>{options.offset=found.offset+found.limit;load();}}),link('png','PNG'),link('svg','SVG'),link('pdf','PDF'),link('csv','All rows CSV'));
      };
      const load=async()=>{const token=++request;status.textContent='Loading saved tests…';try{const found=await api('screening_stability',structuredClone(options));if(token!==request||session!==stabilityGeneration||!pane.isConnected)return;options.offset=found.offset;render(found);}catch(error){if(token===request)status.textContent=error.message;}};
      box.append(node('div',{class:'row wrap'},[node('strong',{text:'Feature stability across distances'}),expand,help('Top rows have the most FDR-significant distances with the same direction, then the highest median exact recovery in those distances, then smallest q and p. This is a transparent priority list, not a measure of biological importance or independent validation. Cumulative zones overlap. The saved q-values are used unchanged; the matrix does not rerun or recorrect the scan.')]),status,controls,legend,grid,pager);
      h.showFigure('screening-stability',box);h.revealFigure?.('screening-stability');await load();
    };
    const drawResult=result=>{
      if(!pane.isConnected||result.version!==2||latestRevision===result.revision)return;
      if(result.comparisons&&view.run===result.run&&view.comparison&&view.comparison!==result.comparison_id){const wanted=view.comparison,token=++comparisonGeneration;api('screening_result',{run:result.run,comparison:wanted}).then(found=>{if(token===comparisonGeneration&&wanted===view.comparison&&pane.isConnected)drawResult(found);}).catch(fail);return;}
      latestRevision=result.revision;latestResult=result;
      if([...runPicker.options].some(o=>o.value===result.run))runPicker.value=result.run;
      const panelKey=result.run+'/'+(result.comparison_id||'');
      if(view.panelKey!==panelKey){view.panelKey=panelKey;view.run=result.run;view.comparison=result.comparison_id||'';view.family='';view.feature='';view.zone=null;view.initialized=false;view.page=0;view.filter='all';view.search='';findingGeneration++;displayGeneration++;}
      const c=result.settings,allTests=result.tests||result.hits,families=c.families.filter(f=>allTests.some(r=>r.family===f));
      if(!view.initialized&&families.length){view.initialized=true;view.family=families[0];}
      const which=choice([['','All feature types'],...families.map(f=>[f,familyNames[f]])],view.family,'Screening result feature type'),table=node('div',{class:'studio-samples'}),detail=node('div');
      const showFigure=family=>{
        const rows=result.hits.filter(r=>r.family===family&&(view.filter==='all'||(!result.partial&&(view.filter==='fdr'?r.q!==null&&r.q<c.alpha:r.p!==null&&r.p<.05)))), chosen=rows.find(r=>r.feature===view.feature&&r.zone===view.zone)||rows[0];if(!chosen){h.showFigure('screening-'+family,node('div',{text:familyNames[family]+' · No findings match this filter.'}));return;}
        const featurePicker=choice(rows.map((r,i)=>[String(i),r.label+' · '+zoneLabel(c,r.zone)]),String(rows.indexOf(chosen)),'Screening interactive finding');
        const plot=node('div'),publication=node('div'),visitReview=node('div',{class:'row wrap','aria-label':'Review repeated observations'}),status=node('span',{class:result.partial?'chip warn':'chip ok',text:result.partial?'Preview · incomplete':'Complete'});
        const drawPlot=()=>{
          const row=rows[Number(featurePicker.value)];view.feature=row.feature;view.zone=row.zone;
          const svgNS='http://www.w3.org/2000/svg',svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox','0 0 620 270');svg.setAttribute('role','img');svg.setAttribute('aria-label','Sample measurements; click a dot to inspect its image');svg.style.cssText='width:100%;min-height:180px;color:inherit';
          const el=(name,attrs,text)=>{const e=document.createElementNS(svgNS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;};
          const obs=row.observations||[],groups=c.endpoint.type==='binary'?[c.endpoint.group_a,c.endpoint.group_b].filter(g=>obs.some(r=>r.group===g)):[...new Set(obs.map(r=>r.group))],vals=obs.map(r=>r.value),lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1,spread=hi-lo||Math.max(1,Math.abs(hi)*.1),y=v=>210-160*(v-lo+.1*spread)/(spread*1.2);
          const metric=family==='diversity'?JSON.parse(row.feature)[1]:null;
          const unit=family==='density'?(c.options.density?.area_source==='estimate'&&c.zone_source!=='polygons'?'Cells / estimated mm²':'Cells / mm²'):family==='proportion'?'% of denominator cells':family==='ratio'?(c.options.ratio.log2?'Log2 count ratio':'Count ratio'):family==='diversity'?diversityNames[metric]:family==='nn_distance'?'Mean distance (µm)':c.options[family].metric==='fraction'?'% of geometric motifs':c.options[family].metric==='zscore'?'Random-label Z-score (Classic)':c.options[family].metric==='count'?'Number found':'Log2 observed / expected';
          svg.append(el('text',{x:72,y:14,fill:'currentColor','font-size':11},(c.endpoint.change_col?'Change in ':'')+unit));
          svg.append(el('line',{x1:70,y1:20,x2:70,y2:218,stroke:'currentColor'}),el('line',{x1:70,y1:218,x2:600,y2:218,stroke:'currentColor'}));
          for(let j=0;j<=4;j++){const value=lo+spread*j/4;svg.append(el('text',{x:62,y:y(value)+4,'text-anchor':'end',fill:'currentColor','font-size':11},fmt(value)));}
          groups.forEach((group,i)=>svg.append(el('text',{x:100+(i+.5)*470/Math.max(1,groups.length),y:242,'text-anchor':'middle',fill:'currentColor','font-size':12},group)));
          obs.forEach((point,i)=>{const x=100+(groups.indexOf(point.group)+.5)*470/Math.max(1,groups.length)+((i*31)%17-8),dot=el('circle',{cx:x,cy:y(point.value),r:5,fill:presentation.colour(point.group,c.endpoint,result.palette),tabindex:0,role:'button','aria-label':point.sample+' · '+point.group+' · '+fmt(point.value)});dot.style.cursor='pointer';dot.append(el('title',{},point.sample+' · '+point.group+' · '+fmt(point.value)+' · click to inspect'));
            const act=()=>{visitReview.replaceChildren();if(point.visits)visitReview.append(node('span',{text:point.sample+' · '+point.group}),...point.visits.map(visit=>node('button',{class:'btn small',text:visit.name+' · '+fmt(visit.value),title:'Open this observation; the plotted value is after minus before',onclick:()=>inspect(result,row,visit.images.includes(h.openRoi())?h.openRoi():visit.images[0],true,{sample:point.sample,group:point.group,occasion:visit.name})})));const image=point.images.includes(h.openRoi())?h.openRoi():point.images[0];if(image)inspect(result,row,image,true,{sample:point.sample,group:point.group,occasion:point.visits?.find(v=>v.images.includes(image))?.name||''});};dot.onclick=act;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act();}};svg.append(dot);});
          visitReview.replaceChildren();plot.replaceChildren(statLine(row),svg,visitReview,node('div',{class:'row wrap'},[node('span',{text:'Each dot: '+c.sample_col}),help('Click a sample dot to open a representative image, show its context cells in the viewer and Data, and zoom to the zone. Multiple images are summarized within each sample. For a change comparison, the dot is after minus before; clicking it exposes buttons for both observations. '+(result.partial?'Coverage is incomplete; q values wait for the complete scan.':'')),node('button',{class:'btn small',text:'Show context cells',onclick:()=>inspect(result,row)})]));
          showDetail(row);
          const fig=result.figures.find(f=>f.family===family&&(!f.metric||f.metric===metric));
          const wasOpen=publication.querySelector('details')?.open;publication.replaceChildren();
          if(fig){const stem=fig.file.replace(/\.png$/i,'');const exported=details('Family overview figure',help('This saved overview summarizes the family. Its lower panels identify one finding in their title. The live sample plot above follows your selected finding; the saved overview does not change when you select another row.'),node('img',{style:'max-width:100%;background:white',alt:fig.name+' screening results',src:h.apiUrl(route('/spatial/screening_file'))+'?run='+encodeURIComponent(result.run)+'&comparison='+encodeURIComponent(result.comparison_id||'')+'&name='+encodeURIComponent('figures/'+fig.file)}),node('div',{class:'row wrap'},['png','svg','pdf'].map(ext=>download(result,'figures/'+stem+'.'+ext,ext.toUpperCase()))));exported.open=!!wasOpen;publication.append(exported);}
        };
        featurePicker.onchange=()=>{stabilitySelection++;drawPlot();drawRows();h.revealFigure?.('screening-'+family);};
        const box=node('div',{},[node('div',{class:'row wrap'},[node('strong',{text:familyNames[family]}),status]),featurePicker,plot,publication]);
        h.showFigure('screening-'+family,box);drawPlot();
      };
      const statLine=row=>{
        const recovery=presentation.recovery(row),percent=value=>presentation.finite(value)===null?'—':fmt(value*100)+'%';
        const resampled=['bootstrap','subsample'].includes(row.resampling);
        return node('div',{class:'screening-statistics row wrap'},[
          directionBadge(c.endpoint,row.effect,result.palette),
          node('span',{text:c.endpoint.type==='binary'?c.endpoint.group_b+' − '+c.endpoint.group_a+(c.endpoint.change_col?' change ('+c.endpoint.change_after+' − '+c.endpoint.change_before+')':'')+': '+fmt(row.effect)+' · '+c.endpoint.group_a+' n='+row.n_a+' · '+c.endpoint.group_b+' n='+row.n_b:'Effect: '+fmt(row.effect)+' · n='+row.n_samples}),
          node('span',{text:'p='+fmt(row.p)+' · q='+(result.partial?'Pending':fmt(row.q))}),
          ...(resampled?[node('span',{text:(row.resampling==='bootstrap'?'Bootstrap recovery ':'Subset recovery ')+percent(recovery)+' · nearby '+percent(row.nearby_recovery)+' · '+row.valid_resamples+'/'+row.resamples+' testable repeats'})]:[]),
          ...(row.resampling==='bootstrap'?[node('span',{text:presentation.finite(row.ci_low)!==null&&presentation.finite(row.ci_high)!==null?'Effect 95% bootstrap interval: '+fmt(row.ci_low)+' to '+fmt(row.ci_high):'95% bootstrap interval unavailable',title:'Requires enough testable resamples; absent intervals are not zero uncertainty.'})]:[]),
          help((row.test||'Descriptive')+'. q scope: '+scopeNames[c.correction_scope||'global']+'. '+(row.correction_n||result.hypotheses)+' comparisons in this correction group. Recovery means significant again, with the same direction; repeats with insufficient samples count as failures. Bootstrap intervals are pointwise, not adjusted for selecting findings. '+(c.endpoint.paired?'n counts matched samples.':''))]);
      };
      const showDetail=row=>{detail.replaceChildren(node('strong',{text:familyNames[row.family]+' · '+row.label+' · '+zoneLabel(c,row.zone)}),statLine(row),
        ...[['Higher',row.highest],['Lower',row.lowest]].filter(([,items])=>items.length).map(([name,items])=>node('div',{class:'row wrap'},[node('span',{text:name}),...items.map(([image,value])=>node('button',{class:'btn small',text:image+' · '+fmt(value),title:'Open this image and show the selected feature’s zone',onclick:()=>inspect(result,row,image,true)}))])),
        node('button',{class:'btn small',text:'Show context cells',onclick:()=>inspect(result,row)}),help('Context cells are the target cell types in the exact selected zone. This does not assert that each displayed cell participates in a pair or triplet.'));
      };
      const show=async row=>{
        stabilitySelection++;const token=++findingGeneration;
        try{
          if(!result.hits.some(r=>r.feature===row.feature&&r.zone===row.zone)){
            h.showFigure('screening-'+row.family,node('div',{role:'status',text:'Loading '+row.label+' · '+zoneLabel(c,row.zone)+'…'}));h.revealFigure?.('screening-'+row.family);
            const found=await api('screening_finding',{run:result.run,comparison:result.comparison_id||'',feature:row.feature,zone:row.zone});
            if(token!==findingGeneration||latestResult!==result||!pane.isConnected)return;result.hits.push(found);
          }
          view.feature=row.feature;view.zone=row.zone;drawRows();showFigure(row.family);h.revealFigure?.('screening-'+row.family);
        }catch(error){if(token===findingGeneration){h.showFigure('screening-'+row.family,node('div',{role:'alert',text:error.message}));h.revealFigure?.('screening-'+row.family);fail(error);}}
      };
      const filter=choice([['all','All findings'],['fdr','FDR significant'],['nominal','Raw p < 0.05']],view.filter,'Screening significance filter'),search=node('input',{type:'search',placeholder:'Find a feature…','aria-label':'Find screening results'}),pager=node('div',{class:'row wrap'});search.value=view.search;
      const columns=[['name','Feature'],['zone','Zone'],['direction','Direction'],['effect','Effect'],['p','p'],['q','q'],['recovery','Recovery'],['n_samples','Samples'],['status','State']];
      const drawRows=()=>{
        const matching=allTests.filter(r=>(!which.value||r.family===which.value)&&(!search.value||(familyNames[r.family]+' '+r.label).toLowerCase().includes(search.value.toLowerCase()))&&(filter.value==='all'||(!result.partial&&(filter.value==='fdr'?r.q!==null&&r.q<c.alpha:r.p!==null&&r.p<.05))));
        const hits=presentation.sortRows(matching,view.sort,view.descending,c.endpoint,familyNames),pages=Math.max(1,Math.ceil(hits.length/100));view.page=Math.min(view.page,pages-1);
        const page=hits.slice(view.page*100,(view.page+1)*100);
        const header=node('tr',{},columns.map(([key,label])=>node('th',{scope:'col','aria-sort':view.sort===key?(view.descending?'descending':'ascending'):'none'},[node('button',{type:'button',class:'screening-sort','aria-label':'Sort screening results by '+label,text:label+(view.sort===key?(view.descending?' ▾':' ▴'):' ↕'),title:'Sort every matching finding by '+label+'; click again to reverse',onclick:()=>{view.descending=view.sort===key?!view.descending:['recovery','n_samples'].includes(key);view.sort=key;view.page=0;drawRows();}})])));
        table.replaceChildren(node('table',{class:'studio-counts screening-results-table','aria-label':'Screening findings'},[node('thead',{},[header]),node('tbody',{},page.map(r=>{
          const selected=view.feature===r.feature&&view.zone===r.zone,recovery=presentation.recovery(r),method=r.resampling==='bootstrap'?'Bootstrap':r.resampling==='subsample'?'Sample subsets':'Not resampled';
          const tr=node('tr',{tabindex:0,class:'spatial-hit'+(selected?' selected':''),'aria-selected':String(selected),'data-feature':r.feature,'data-zone':r.zone,title:'Open '+r.label+' · '+zoneLabel(c,r.zone)+' in Figures on the right',onclick:()=>show(r),onkeydown:ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();show(r);}}});
          tr.append(node('td',{text:(which.value?'':familyNames[r.family]+' · ')+r.label}),node('td',{text:zoneLabel(c,r.zone)}),node('td',{},[directionBadge(c.endpoint,r.effect,result.palette)]),node('td',{text:fmt(r.effect)}),node('td',{text:fmt(r.p)}),node('td',{text:result.partial?'Pending':fmt(r.q)}),node('td',{text:recovery===null?'—':Math.round(recovery*100)+'%',title:method+(recovery===null?' · recovery unavailable':' · exact recovery at this distance; '+r.valid_resamples+'/'+r.resamples+' testable repeats')}),node('td',{text:fmt(r.n_samples)}),node('td',{text:r.status||'—'}));return tr;
        }))]));
        pager.replaceChildren(node('span',{text:hits.length+' findings · page '+(view.page+1)+'/'+pages}),node('button',{class:'btn small',text:'Previous',disabled:view.page===0?'':null,onclick:()=>{view.page--;drawRows();}}),node('button',{class:'btn small',text:'Next',disabled:view.page>=pages-1?'':null,onclick:()=>{view.page++;drawRows();}}));
      };
      filter.onchange=()=>{stabilitySelection++;view.filter=filter.value;view.page=0;findingGeneration++;detail.replaceChildren();drawRows();if(view.family)showFigure(view.family);};search.oninput=()=>{view.search=search.value;view.page=0;findingGeneration++;drawRows();};
      which.onchange=()=>{stabilitySelection++;view.family=which.value;view.feature='';view.zone=null;view.page=0;findingGeneration++;detail.replaceChildren();drawRows();if(which.value){showFigure(which.value);h.revealFigure?.('screening-'+which.value);}};drawRows();
      const design=node('div',{class:'screening-run-design'}),difference=node('span',{class:'chip warn'});
      showDesign=()=>{difference.textContent='Controls differ from this saved run';difference.hidden=JSON.stringify(v)===JSON.stringify(result.scan_settings||c);};showDesign();
      design.append(node('div',{class:'row wrap'},[node('span',{text:'Saved run · Sample ID: '+c.sample_col+' · '+(c.endpoint.type==='binary'?c.endpoint.group_a+' vs '+c.endpoint.group_b:c.endpoint.type)}),difference]),
        ...(c.sample_col===c.image_col?[node('div',{class:'studio-warning',text:'This run counted images as independent samples. Use a biological sample ID if several images came from the same person or specimen.'})]:[]));
      const comparisons=node('div');
      if(result.comparisons){const pickComparison=choice(result.comparisons.map(p=>[p.id,p.label+' · '+p.status]),result.comparison_id,'Screening result comparison'),findComparison=node('input',{type:'search',placeholder:'Find a comparison…','aria-label':'Find screening comparisons'});
        const fill=()=>{const query=findComparison.value.toLowerCase();pickComparison.replaceChildren(...result.comparisons.filter(p=>p.id===result.comparison_id||p.label.toLowerCase().includes(query)).map(p=>node('option',{value:p.id,text:p.label+' · '+p.status,disabled:['ready','skipped'].includes(p.status)?'':null,title:p.reason||p.label})));pickComparison.value=result.comparison_id;};findComparison.oninput=fill;fill();
        pickComparison.onchange=async()=>{const wanted=pickComparison.value,token=++comparisonGeneration;view.comparison=wanted;findingGeneration++;displayGeneration++;try{const found=await api('screening_result',{run:result.run,comparison:wanted});if(token===comparisonGeneration&&view.comparison===wanted)drawResult(found);}catch(error){if(token===comparisonGeneration)fail(error);}};
        comparisons.append(field('Comparison',pickComparison,'This run contains several declared comparisons. The sample count and figures below refer to the selected comparison; the full-scan exports include every comparison, including skipped ones.'),...(result.comparisons.length>8?[findComparison]:[]),...(result.scan_hypotheses?[node('div',{class:'row wrap'},[download(result,'tables/all_comparisons.csv','All comparisons'),download(result,'tables/all_significant_features.csv','All FDR findings')])]:[]));}
      resultBody.replaceChildren(design,comparisons,node('div',{class:'row wrap'},[chip(result.partial?'Preview · '+result.message:'Complete',result.partial?'warn':'ok'),chip(result.samples+' samples'),...(result.partial?[]:[chip(result.tested+' tested / '+result.hypotheses),chip(result.significant+' q < '+c.alpha)]),help('p is for one comparison; q follows the saved correction scope. All findings includes every comparison, with search and pages. FDR significant filters by the saved cutoff; its export includes every qualifying feature and distance, not only the plotted shortlist.')]),
        details('Run settings and notes',node('div',{text:'Sample ID: '+c.sample_col+' · Image ID: '+c.image_col+' · '+c.aggregation+' across images per sample'}),node('div',{text:'Tests: '+[...new Set(result.hits.map(r=>r.test))].join(', ')+' · Correction: '+c.correction}),...(result.warnings||[]).map(w=>node('div',{text:w})),download(result,'design.json','Full run settings')),
        node('div',{class:'row wrap'},[node('span',{text:'q scope: '+scopeNames[c.correction_scope||'global']})]),
        node('div',{class:'row wrap screening-result-actions'},[node('button',{class:'btn small',text:'Selected feature',onclick:()=>{const row=allTests.find(r=>r.feature===view.feature&&r.zone===view.zone)||allTests.find(r=>!view.family||r.family===view.family);if(row)show(row);}}),node('button',{class:'btn small primary',text:'Stability across distances',disabled:result.partial?'':null,onclick:()=>openStability(result)}),chip(c.resampling==='bootstrap'?'Bootstrap':c.resampling==='subsample'?'Sample subsets':'Not resampled'),help('Click a table header to sort every matching result, including later pages. Click a row for its plot on the right. Stability across distances shows group direction and exact resampling recovery in every cell. Bootstrap also provides pointwise effect intervals in the selected-feature plot.')]),
        pair(field('Show',which),field('Findings',filter)),search,table,pager,detail,
        ...(result.partial?[]:[node('div',{class:'row wrap'},[download(result,'tables/significant_features.csv','FDR significant'),download(result,'tables/screening_results_all.csv','All tests'),download(result,'tables/sample_features.csv','Sample values'),download(result,'tables/region_features.csv','Image values'),download(result,'design.json','Run details')])]));
      if(view.family&&families.includes(view.family))showFigure(view.family);
    };
    runPicker.onchange=async()=>{const token=++generation;if(!runPicker.value)return;try{const r=await api('screening_result',{run:runPicker.value});if(token===generation)drawResult(r);}catch(err){fail(err);}};
    [study.box,features.box,compare.box,results.box].forEach(box=>box.classList.add('screening-section'));
    results.body.append(runPicker,resultBody);pane.append(study.box,features.box,compare.box,results.box);refreshRuns().catch(fail);
    // Reattach to a running scan after changing tabs; do not wait for its final result.
    (async()=>{const token=generation,jobs=(await get(route('/jobs'))).jobs||[];
      const active=jobs.find(j=>j.kind==='spatial_screening'&&['running','starting'].includes(j.status));if(!active)return;
      while(pane.isConnected&&token===generation){const progress=await api('screening_progress',{job_id:active.id});
        if(progress.result&&pane.isConnected&&token===generation&&progress.result.table.split('/').pop()===known.table)drawResult(progress.result);
        if(['done','failed','cancelled'].includes(progress.status))break;await new Promise(r=>setTimeout(r,2500));}
    })().catch(fail);
    current.runBar([actionRow('Screen:',async(images,note)=>{const submitted=structuredClone(v);await save();const token=++generation;note.textContent='Preparing the spatial scan…';const found=await runJob('spatial_screening',{images,values:submitted},note,result=>{if(pane.isConnected&&token===generation)drawResult(result);});if(pane.isConnected&&token===generation){drawResult(found);refreshRuns().catch(fail);}},'Scans only the ticked images using the selected cells, regions and feature settings. Original cell data is read-only.',said),said]);
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
    const endpoint=route('/phenotype_assessment'), known=await get(endpoint);
    if(!pane.isConnected)return;
    const box=section('Phenotype assessment', {tip:'Compare saved phenotype columns with your reference labels. Methods are scored on the same cells; missing predictions count as unassigned.'});
    const reference=columnPick(known.columns,prefer(known.references.length?known.references:known.columns,['phenotype_ref1']),'Assessment reference');
    const subset=choice([['validation','Validation cells'],['training','Training cells'],['both','Training + validation'],['all','All labelled cells']],'all','Assessment cells');
    const split=choice([], '', 'Assessment saved split');
    const splitRow=field('Saved split',split,'The original cells used for this training run. Comparing never changes the split.');
    const methods=node('details',{open:''},[node('summary',{text:'Methods'})]), methodList=node('div',{class:'studio-method-list'});
    methods.append(methodList);
    const selected=new Set(known.defaults), images=new Set(known.tables.map(item=>item.path));
    const imageList=node('details',{},[node('summary',{text:'Images · '+images.size+' selected'})]);
    known.tables.forEach(item=>{
      const pick=node('input',{type:'checkbox','aria-label':'Assess '+item.image});pick.checked=true;
      pick.onchange=()=>{pick.checked?images.add(item.path):images.delete(item.path);imageList.querySelector('summary').textContent='Images · '+images.size+' selected';};
      imageList.append(node('label',{class:'inline-check'},[pick,node('span',{text:item.image})]));
    });
    const drawMethods=()=>{
      const available=[...known.methods,...known.columns.filter(c=>!known.methods.includes(c))].filter(c=>c!==reference.value);
      selected.delete(reference.value);
      methodList.replaceChildren(...available.map(column=>{
        const pick=node('input',{type:'checkbox','aria-label':'Assess method '+column,'data-assessment-method':column});pick.checked=selected.has(column);
        pick.onchange=()=>{pick.checked?selected.add(column):selected.delete(column);};
        return node('label',{class:'inline-check'},[pick,node('span',{text:column})]);
      }));
    };
    const drawSplits=()=>{
      const available=known.splits.filter(item=>item.reference===reference.value), previous=split.value;
      split.replaceChildren(...available.map(item=>node('option',{value:item.id,text:
        new Date(item.trained).toLocaleString()+' · '+item.training_cells+' train / '+item.validation_cells+' validation'})));
      if(available.some(item=>item.id===previous))split.value=previous;
      [...subset.options].forEach(option=>{option.disabled=option.value!=='all'&&!available.length;});
      subset.value=available.length?'validation':'all';splitRow.hidden=subset.value==='all';drawMethods();
    };
    reference.onchange=drawSplits;subset.onchange=()=>{splitRow.hidden=subset.value==='all';};drawSplits();
    const exclusions=node('input',{value:'','aria-label':'Excluded reference phenotypes',placeholder:'None'});
    const advanced=node('details',{},[node('summary',{text:'Options'}),field('Exclude labels',exclusions,'Comma-separated phenotype names. Leave blank to keep all labelled phenotypes.')]);
    const said=node('p',{class:'hint',role:'status'});
    box.body.append(field('Reference',reference),field('Cells',subset,
      'Validation uses the saved holdout. Training and Both can assess methods that did not learn from these labels. All labelled cells also includes labels added later.'),splitRow,methods,imageList,advanced);
    pane.append(box.box);
    const show=found=>{
      const figure=section('Phenotype assessment', {tip:'Classic F1 measures and figures. Save a figure or download the scores and exact cell list.'});
      const kind=choice(Object.entries(found.plots),'comparison','Assessment figure');
      const method=choice(Object.entries(found.methods),Object.keys(found.methods)[0],'Confusion method');
      const format=choice([['png','PNG'],['svg','SVG'],['pdf','PDF']],'png','Assessment figure format');
      const image=node('img',{class:'studio-figure-preview',alt:'Phenotyping method assessment'}), notice=node('p',{class:'hint',role:'status'});
      const summary=node('table',{class:'studio-counts'},[node('tr',{},['Method','Cells','Macro F1','Micro F1','Weighted F1','Coverage'].map(text=>node('th',{text}))),
        ...found.summary.map(item=>node('tr',{},[item.method,item.n_test.toLocaleString(),item.macro_f1.toFixed(3),item.micro_f1.toFixed(3),item.weighted_f1.toFixed(3),(100*item.prediction_coverage).toFixed(1)+'%'].map(text=>node('td',{text})) ))]);
      let sequence=0;
      const render=async()=>{
        const asked=++sequence;method.hidden=kind.value!=='confusion';notice.textContent='Drawing…';
        try{
          const picture=await post(endpoint,{action:'figure',id:found.id,kind:kind.value,method:method.value,options:{format:'png',dpi:120}});
          if(asked!==sequence)return;
          image.src='data:'+picture.type+';base64,'+picture.data;notice.textContent='';
        }catch(error){if(asked===sequence){image.removeAttribute('src');notice.textContent=error.message;}}
      };
      kind.onchange=render;method.onchange=render;
      const download=async(button,action)=>{
        button.disabled=true;
        try{h.downloadFile(await post(endpoint,{action,id:found.id,kind:kind.value,method:method.value,options:{format:format.value,dpi:300}}));}
        catch(error){notice.textContent=error.message;}finally{button.disabled=false;}
      };
      figure.body.append(node('div',{class:'row wrap'},[chip(found.subset),chip(found.cells.toLocaleString()+' cells'),help(found.note)]),summary,
        ...found.notices.map(text=>node('p',{class:'hint',text})),field('Figure',kind),method,image,notice,
        node('div',{class:'row wrap'},[format,node('button',{class:'btn small',text:'Save figure',onclick:event=>download(event.currentTarget,'figure')}),
          node('button',{class:'btn small',text:'Download scores',onclick:event=>download(event.currentTarget,'export')})]));
      h.showFigure('viz-agreement',figure.box);render();
    };
    const compare=node('button',{class:'btn primary',text:'Compare',onclick:async()=>{
      if(!images.size){said.textContent='Select at least one image.';return;}
      compare.disabled=true;said.textContent='Comparing…';
      try{
        const found=await post(endpoint,{reference:reference.value,methods:[...selected],subset:subset.value,
          split_id:split.value,tables:[...images],exclude_labels:exclusions.value.split(',').map(s=>s.trim()).filter(Boolean)});
        if(!pane.isConnected)return;
        said.textContent=found.cells.toLocaleString()+' cells · '+found.summary.length+' methods';show(found);
      }catch(error){said.textContent=error.message;}finally{compare.disabled=false;}
    }});
    current.runBar([node('div',{class:'row'},[node('span',{class:'lbl',text:'Agreement:'}),compare,
      help('For model validation, use predictions made without copying manual labels. In MR-ML, turn off Keep existing reference labels. Training cells only give an independent check for methods that never used them.')]),said]);
  }

  return {tlsCandidates, density, triads, screening, whereFigure, heatmap, proportions, umap:umapPanel, agreement, info,
    glands:(current,pane,actionRow)=>window.createGlandPanel(h).mount(current,pane,actionRow)};
};
