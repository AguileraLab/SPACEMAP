/* SPACEMAP Studio: a step rail, the chosen step's tools, the image, and on the
   right the step's outputs, the layers and the background jobs. Actions go
   through the shared command bus or a job; classic tabs share the settings. */
(() => {
  'use strict';
  // The code's date (the page asks for it with ?v=<file date>): shown in the header, so a stale copy is easy to spot.
  const BUILD = (() => { const stamp = Number((document.currentScript?.src.match(/[?&]v=(\d+)/) || [])[1]);
    return stamp ? new Date(stamp * 1000).toLocaleString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}) : 'unversioned'; })();
  const node = (tag, attrs = {}, children = []) => {
    const element = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value === undefined || value === null || value === false) return;
      if (key === 'text') element.textContent = value;
      else if (key.startsWith('on')) element.addEventListener(key.slice(2), value);
      else element.setAttribute(key, value === true ? '' : value);
    });
    children.forEach(child => child && element.appendChild(child));
    return element;
  };
  // Fills an element, leaving out the parts a panel does not show (null or false).
  const fill = (element, ...items) => element.replaceChildren(...items.filter(Boolean));
  // Building blocks shared by every step (the viewer has the same three).
  const help = text => node('span', {class:'help', tabindex:'0', role:'note', 'aria-label':text, 'data-tip':text, text:'?'});
  const setTip = (tip, text) => { tip.dataset.tip = text; tip.setAttribute('aria-label', text); };
  // A tick box heading a list's tick column: ticks or clears every row (half-ticked when only some are).
  const pickAll = (keys, picked, redraw) => {
    const box = node('input', {type:'checkbox', 'aria-label':'Select all', title:'Select or clear all'});
    box.sync = () => { const some = keys.filter(key => picked.has(key)).length; box.checked = !!keys.length && some === keys.length; box.indeterminate = some > 0 && some < keys.length; };
    box.addEventListener('change', () => { keys.forEach(key => box.checked ? picked.add(key) : picked.delete(key)); redraw(); });
    box.sync(); return box;
  };
  // A step's list while its jobs run: reloaded as they progress (at most every 6 s, and once when they end), so each row's
  // state turns green as it is done, and the row being worked on is marked. Job messages start with "<row name>:".
  const ACTIVE_JOBS = ['starting', 'running', 'stopping'];
  let lastJobs = [];
  const followJobs = (current, kinds, reload) => {
    let last = 0, timer = null;
    const active = jobs => jobs.filter(job => kinds.includes(job.kind) && ACTIVE_JOBS.includes(job.status));
    let wasActive = active(lastJobs).length > 0;
    current.workingOn = new Set(active(lastJobs).map(job => String(job.message || '').split(':')[0]));
    current.jobsChanged = jobs => {
      const mine = active(jobs);
      current.workingOn = new Set(mine.map(job => String(job.message || '').split(':')[0]));
      const ended = wasActive && !mine.length; wasActive = mine.length > 0;
      if (!mine.length && !ended) return;
      const run = async () => {
        timer = null;
        if (!current.body.isConnected) return;
        if (current.body.querySelector('.studio-pairs select:focus')) { timer = setTimeout(run, 2000); return; }  // not while choosing
        last = Date.now();
        const top = current.body.querySelector('.studio-pairs-scroll')?.scrollTop;
        await reload();
        const scroll = current.body.querySelector('.studio-pairs-scroll'); if (scroll && top) scroll.scrollTop = top;
      };
      clearTimeout(timer); timer = setTimeout(run, ended ? 0 : Math.max(0, 6000 - (Date.now() - last)));
    };
  };
  const workingChip = (current, name) => current.workingOn?.has(name) ? chip('Working…', 'run', 'A job is working on it now') : null;
  const chip = (text, kind = '', title) => node('span', {class:'chip' + (kind ? ' ' + kind : ''), text, title});
  const field = (label, control, tip) => node('div', {class:'row field'}, [node('label', {class:'lbl', text:label}), control, tip ? help(tip) : null]);
  const section = (title, {tip, actions = [], id, cls} = {}) => {
    const body = node('div', {class:'studio-sec-body'});
    const box = node('section', {class:'studio-sec' + (cls ? ' ' + cls : ''), id}, [
      node('div', {class:'studio-sec-h'}, [node('span', {text:title}), tip ? help(tip) : null, node('span', {class:'grow'}), ...actions]), body]);
    return {box, body};
  };
  const scaleLabel = (step, mpp) => {
    const factor = Number(step), size = Number(mpp);
    const name = factor === 1 ? 'Full' : '1/' + factor;
    return Number.isFinite(size) && size > 0 && Number.isFinite(factor) && factor > 0
      ? name + ' · ' + (size * factor).toFixed(2) + ' µm/px' : name + ' · pixel size unavailable';
  };
  // A list with a few first options, then named groups ([[group, [[value, text], ...]], ...]); empty groups left out.
  const groupedChoice = (first, groups, value, label) => {
    const select = node('select', {'aria-label':label});
    first.forEach(([key, text]) => select.append(node('option', {value:key, text})));
    groups.filter(([, items]) => items.length).forEach(([group, items]) => select.append(node('optgroup', {label:group},
      items.map(([key, text]) => node('option', {value:key, text})))));
    select.value = value;
    return select;
  };
  const choice = (options, value, label) => {
    const input = node('select', {'aria-label':label});
    options.forEach(([key, text]) => input.append(node('option', {value:key, text})));
    input.value = value;
    return input;
  };
  const STATE_KIND = {done:'ok', partial:'warn', running:'run', failed:'error', needs_review:'warn', not_run:''};
  const ITEMS = {tab0:'Project', tab1:'Image pairs', tab2:'Images', tab3:'Images', tab4:'ROIs', tab5:'ROIs', tagging:'Cell tables', tab6:'Tables',
    tab7:'ROIs', tab8:'Tables', tab9:'Tables'};
  const JOB_NAMES = {resolve_all:'RESOLVE · all ROIs', mr_ml:'MR-ML', train_reference_model:'Reference models',
    subtract_background:'Background removal', extract_rois:'ROI extraction', segment_images:'Segmentation',
    register_pairs:'Registration', detect_tissue:'Tissue detection', clean_tissue:'Tissue clean-up',
    train_artifact_model:'Artifact model training', detect_artifacts:'Artifact detection', artifact_batch:'Artifacts · ticked images', subtract_annotations:'Subtract annotations', remove_small_pieces:'Remove small pieces', classify_shared:'Shared classes', save_class_polygons:'Class polygons', subtract_artifacts:'Subtract artifacts',
    export_classes:'Class export', import_annotations:'GeoJSON import', export_annotations:'Annotation export', apply_pieces:'Tissue pieces', make_zones:'Zones', leiden:'Leiden', name_clusters:'Names', neighbourhoods:'Neighbourhoods', train_cellpose:'Cellpose training', score_cellpose:'Cellpose scores',
    correct_batches:'Batch correction', som:'SOM', vortex:'VORTEX', export_table:'CSV copy', consensus_ml:'Consensus ML', subphenotype:'Sub-phenotyping', cell_interactions:'Cell–cell interactions', cell_tagging:'Cell tagging', tls_candidates:'TLS candidates'};
  // Tab 7's own methods, in its order, plus manual reference labelling.
  const METHODS = [
    {id:'leiden', name:'Leiden clustering', studio:true, tip:'Groups cells by their markers. Review and name the resulting clusters.'},
    {id:'som', name:'SOM clustering', studio:true, tip:'Groups similar cells on a map, then merges them into clusters for review and naming.'},
    {id:'vortex', name:'VORTEX clustering', studio:true, tip:'Groups cells using similarities in their markers. Review and name the resulting clusters.'},
    {id:'resolve', name:'RESOLVE (threshold)', studio:true, tip:'Define positive marker signals, review phenotype rules, then apply classic RESOLVE to the combined cell table.'},
    {id:'tools', name:'Tools', tip:'Heatmaps and marker composition of phenotypes.'},
    {id:'mr_ml', name:'MR-ML', studio:true, tip:'Learns from a reference column and marker or morphology measurements. Additional method votes are optional.'},
    {id:'consensus', name:'Consensus ML', studio:true, tip:'Uses cells with agreement between methods to train a model that labels all cells.'},
    {id:'subphenotyping', name:'Sub-phenotyping', studio:true, tip:'Finds finer groups within selected phenotypes using markers that vary between their cells.'},
    {id:'reference', name:'Manual reference', studio:true, tip:'Name cells yourself (click, box or a marker rule) into a reference ' +
      'column such as phenotype_ref1. Suggestions and least-sure navigation help.'},
  ];

  window.startSpacemapStudio = hooks => {
    const {session, get:rawGet, post:rawPost, stream, openImage, runCommand:rawCommand, applyLevels, captureView, renderResolve, refreshResolve, renderLabels,
      renderBackground, backgroundLevels, renderRois, renderShapeLayer, setDrawTarget, flushShapes, saveTissuePolygons, activeTissueId, chooseTissue, ensureArtifactLayer, saveArtifactPolygons, reviewedArtifact, drawArtifactExamples, artifactExampleState, finishArtifactExamples, artifactNeedsReview, annotationState, reloadAnnotations, showTissuePreview, ensureTissueLayer, clearTissuePreview, tissueForPieces, applyPiecePreview, zoomAnnotationBox, showNeckReview, approveNeck, viewBounds, imageMarkers, showSegPreview, showPreview, showRaster, drawExample, saveFigure,
      heCompare, detectLevels, cellsState, viewSummary, colourCells, previewCells, setBackgroundScope, showCellGroups, setMarkerRoles, editCells, readCells, dropCells, pickPoint, imageSize,
      setSelectionTool} = hooks;
    const route = path => '/studio/' + encodeURIComponent(session) + path;
    // Something happening behind the screen: the page waiting on the server ("Loading…", after a moment, so quick
    // requests do not flicker), or background jobs running. Shown beside the title; a click opens Jobs.
    let waiting = 0, jobsRunning = 0, waitTimer = null;
    const busy = node('button', {type:'button', class:'studio-busy', hidden:true, title:'Work in progress', onclick:() => showTab('jobs')},
      [node('span', {class:'studio-spinner', 'aria-hidden':'true'}), node('span', {class:'studio-busy-text'})]);
    const updateBusy = () => {
      const text = waiting && waitTimer === 'shown' ? 'Loading…' : jobsRunning ? jobsRunning + (jobsRunning === 1 ? ' job running' : ' jobs running') : '';
      busy.querySelector('.studio-busy-text').textContent = text; busy.hidden = !text;
    };
    const tracked = request => async (...args) => {
      const quiet = typeof args[0] === 'string' && /\/(jobs|overview)(\?|$)/.test(args[0]);  // regular polling
      if (quiet) return request(...args);
      waiting += 1;
      if (!waitTimer) waitTimer = setTimeout(() => { waitTimer = 'shown'; updateBusy(); }, 300);
      try { return await request(...args); }
      finally {
        waiting -= 1;
        if (!waiting) { if (waitTimer !== 'shown') clearTimeout(waitTimer); waitTimer = null; }
        updateBusy();
      }
    };
    const get = tracked(rawGet), post = tracked(rawPost);
    const runCommand = tracked(rawCommand);
    document.body.classList.add('studio');
    if (/^[0-9a-f]{32}$/.test(session)) {
      try { localStorage.setItem('spacemap-viewer-browser', session); } catch (_) {}
    }
    const remember = (key, value) => { try { localStorage.setItem('spacemap-studio-' + key, value); } catch (_) {} };
    const recall = (key, fallback) => { try { return localStorage.getItem('spacemap-studio-' + key) || fallback; } catch (_) { return fallback; } };
    let context, images = null, imagesRequest, selectionTools, heatmapTools, currentSpec, frame = null, matches = [];
    let selected = new URLSearchParams(location.search).get('step'), method = recall('method', 'resolve');

    // ------------------------------------------------------------ header
    const header = node('header', {id:'studio-header'});
    const toggle = node('button', {class:'btn', text:'Steps', title:'Show or hide the step names', 'aria-expanded':'true', onclick:() => {
      const hidden = document.body.classList.toggle('studio-hide-steps');
      toggle.setAttribute('aria-expanded', String(!hidden));
    }});
    const brand = node('div', {class:'studio-brand', title:'Studio\u2019s code: ' + BUILD}, [node('strong', {text:'SPACEMAP'}), node('span', {text:'Studio'}),
      node('span', {class:'muted studio-build', text:BUILD})]);
    const projectName = node('strong', {text:'Connecting…'});
    const projectMenu = node('details', {class:'studio-image-list studio-project'});
    const imageFilter = node('input', {type:'search', placeholder:'Filter images', 'aria-label':'Filter project images'});
    const list = node('div', {class:'studio-image-rows'});
    projectMenu.append(node('summary', {title:'Project images'}, [node('span', {class:'muted', text:'Project'}), projectName]),
      node('div', {class:'studio-menu'}, [imageFilter, list, node('button', {type:'button', class:'btn small', text:'Other projects',
        title:'Switch to another project (the Project step)', onclick:() => { projectMenu.open = false; chooseStep('tab0'); }})]));
    const search = node('input', {type:'search', placeholder:'Find a step or image, or ask…  Ctrl+K', 'aria-label':'Find a step or image',
      autocomplete:'off', 'aria-controls':'studio-matches', 'aria-expanded':'false'});
    const menu = node('div', {id:'studio-matches', role:'group', 'aria-label':'Matching steps and images', hidden:true});
    const finder = node('div', {class:'studio-finder'}, [search, menu]);
    const askButton = node('button', {class:'btn', text:'Ask', title:'Ask the assistant to update this view', onclick:() => ask()});
    const jobsBadge = node('span', {class:'badge idle', text:'0'});
    const jobsButton = node('button', {class:'btn', title:'Background jobs', onclick:() => showTab('jobs')}, [node('span', {text:'Jobs'}), jobsBadge]);
    const classic = node('a', {class:'btn', text:'Classic SPACEMAP', target:'spacemap-classic', rel:'noopener'});
    const assistant = node('a', {class:'btn', text:'AI assistant', target:'spacemap-classic', rel:'noopener'});
    header.append(toggle, brand, busy, projectMenu, finder, askButton, node('span', {class:'grow'}), jobsButton, assistant, classic);

    // ------------------------------------------------------------ rail, tools, right side
    const rail = node('aside', {id:'studio-steps', 'aria-label':'Analysis steps'});
    const steps = node('nav', {'aria-label':'Workflow'});
    rail.append(steps, node('div', {class:'studio-legend'}, [['done', 'Done'], ['partial', 'Partly done'], ['none', 'Not run']]
      .map(([state, text]) => node('span', {}, [node('i', {class:'studio-dot', 'data-state':state}), node('span', {text})]))));
    const tools = node('section', {id:'studio-tools', 'aria-label':'Step tools'});
    const right = node('aside', {id:'studio-right'});
    const tabs = node('div', {class:'studio-tabs', role:'tablist', 'aria-label':'Outputs, layers and jobs'});
    const outputsPane = node('div', {id:'studio-outputs', class:'studio-pane', role:'tabpanel'});
    const panel = document.getElementById('panel');
    panel.classList.add('studio-pane'); panel.setAttribute('role', 'tabpanel');
    const jobsPane = node('div', {id:'studio-jobs', class:'studio-pane', role:'tabpanel'});
    const tabBadge = node('span', {class:'badge idle', text:'0'});
    const assistantPane = node('div', {id:'studio-assistant', class:'studio-pane', role:'tabpanel'});
    const TABS = {outputs:[outputsPane, 'Outputs'], layers:[panel, 'Layers'], jobs:[jobsPane, 'Jobs'], assistant:[assistantPane, 'Assistant']};
    const tabButtons = {};
    Object.entries(TABS).forEach(([key, [pane, label]]) => {
      tabButtons[key] = node('button', {type:'button', role:'tab', 'data-tab':key, 'aria-controls':pane.id, onclick:() => showTab(key)},
        [node('span', {text:label}), key === 'jobs' ? tabBadge : null]);
      tabs.append(tabButtons[key]);
    });
    let onAssistantShown = null;
    let onJobsShown = null;  // set by Jobs below: Saved results are read when the tab is shown
    let openAiSetup = () => {};  // opens AI setup in the Assistant tab (defined with the assistant below)
    let explorerNow = null;  // the cluster explorer on screen (run, picks, names typed), told to the assistant  // set by the assistant below: its conversation loads when first shown
    const showTab = key => {
      if (!TABS[key]) key = 'layers';
      if (key === 'assistant' && onAssistantShown) onAssistantShown();
      if (key === 'jobs' && onJobsShown) onJobsShown();
      Object.entries(TABS).forEach(([name, [pane]]) => {
        pane.hidden = name !== key;
        tabButtons[name].setAttribute('aria-selected', String(name === key));
      });
      tabButtons[key].classList.remove('fresh');
      remember('tab', key);
    };
    right.append(tabs, outputsPane, panel, jobsPane, assistantPane);
    const app = document.getElementById('app');
    app.prepend(header, rail, tools); app.append(right);

    // The right-hand pane: drag its left edge to widen it towards the image; figures (heatmaps) sit in a dock at
    // its foot, below whichever tab is open, and the divider above them moves up or down. Sizes are remembered.
    const figures = node('div', {id:'studio-figures'});
    const dockHandle = node('div', {class:'studio-resize-y', role:'separator', 'aria-orientation':'horizontal', tabindex:'0',
      'aria-label':'Drag to give the figures more or less room', title:'Drag up or down'});
    const dock = node('section', {id:'studio-dock', 'aria-label':'Figures', hidden:true});
    const hideDock = node('button', {class:'btn small ghost', text:'Hide', title:'Hide the figures until the next one', onclick:() => { dock.hidden = true; }});
    dock.append(dockHandle, node('div', {class:'studio-dock-head'}, [node('span', {text:'Figures'}),
      help('Heatmaps from the step’s results. Click a row to show it on the image. Drag the line above to resize.'), node('span', {class:'grow'}), hideDock]), figures);
    right.append(dock);
    const widthHandle = node('div', {class:'studio-resize-x', role:'separator', 'aria-orientation':'vertical', tabindex:'0',
      'aria-label':'Drag to widen or narrow the right-hand pane', title:'Drag left or right'});
    right.prepend(widthHandle);
    const setWidth = width => {
      const px = Math.round(Math.max(260, Math.min(window.innerWidth * 0.6, width)));
      app.style.setProperty('--studio-right', px + 'px'); return px;
    };
    const setDock = height => {
      const px = Math.round(Math.max(90, Math.min(right.clientHeight * 0.8 || 600, height)));
      dock.style.height = px + 'px'; return px;
    };
    if (Number(recall('right-width', 0))) setWidth(Number(recall('right-width', 0)));
    if (Number(recall('dock-height', 0))) dock.style.height = Number(recall('dock-height', 0)) + 'px';
    const drag = (handle, move, save) => {
      handle.addEventListener('pointerdown', event => {
        event.preventDefault(); document.body.classList.add('studio-resizing');
        try { handle.setPointerCapture(event.pointerId); } catch (_) { /* keeps following without capture */ }
        const follow = moved => move(moved);
        const stop = () => {
          handle.removeEventListener('pointermove', follow); handle.removeEventListener('pointerup', stop); handle.removeEventListener('pointercancel', stop);
          document.body.classList.remove('studio-resizing'); save(); window.dispatchEvent(new Event('resize'));
        };
        handle.addEventListener('pointermove', follow); handle.addEventListener('pointerup', stop); handle.addEventListener('pointercancel', stop);
      });
    };
    drag(widthHandle, event => setWidth(window.innerWidth - event.clientX), () => remember('right-width', right.getBoundingClientRect().width));
    drag(dockHandle, event => setDock(right.getBoundingClientRect().bottom - event.clientY), () => remember('dock-height', dock.getBoundingClientRect().height));
    widthHandle.addEventListener('keydown', event => {
      const step = {ArrowLeft:40, ArrowRight:-40}[event.key]; if (!step) return;
      event.preventDefault(); remember('right-width', setWidth(right.getBoundingClientRect().width + step)); window.dispatchEvent(new Event('resize'));
    });
    dockHandle.addEventListener('keydown', event => {
      const step = {ArrowUp:40, ArrowDown:-40}[event.key]; if (!step) return;
      event.preventDefault(); remember('dock-height', setDock(dock.getBoundingClientRect().height + step));
    });
    // The two left panels (the steps, and the step's tools) widen or narrow the same way, from their right edge.
    [['steps', rail, 140, 360], ['tools', tools, 240, 0.45]].forEach(([key, pane, least, most]) => {
      const handle = node('div', {class:'studio-resize-x at-right', role:'separator', 'aria-orientation':'vertical', tabindex:'0',
        'aria-label':'Drag to widen or narrow the ' + (key === 'steps' ? 'steps' : 'tools') + ' panel', title:'Drag left or right'});
      pane.append(handle);
      const set = width => {
        const px = Math.round(Math.max(least, Math.min(most < 1 ? window.innerWidth * most : most, width)));
        app.style.setProperty('--studio-' + key, px + 'px'); return px;
      };
      if (Number(recall(key + '-width', 0))) set(Number(recall(key + '-width', 0)));
      drag(handle, event => set(event.clientX - pane.getBoundingClientRect().left), () => remember(key + '-width', pane.getBoundingClientRect().width));
      handle.addEventListener('keydown', event => {
        const step = {ArrowLeft:-40, ArrowRight:40}[event.key]; if (!step) return;
        event.preventDefault(); remember(key + '-width', set(pane.getBoundingClientRect().width + step)); window.dispatchEvent(new Event('resize'));
      });
    });
    showTab(recall('tab', 'layers'));

    const empty = node('div', {id:'studio-empty'}, [node('h2', {text:'Choose an image to begin'}),
      node('p', {text:'Your image and cell tools stay here as you move through the steps.'}),
      node('button', {class:'btn primary', text:'Browse images', onclick:() => { projectMenu.open = true; imageFilter.focus(); }})]);
    document.getElementById('stage').append(empty);

    // Outputs: what the last action returned, then the chosen step's saved outputs.
    const notice = node('div', {id:'studio-action-status', role:'status'}, [node('p', {class:'muted', text:'Answers and action results appear here.'})]);
    const describe = node('button', {class:'btn small', text:'Describe view', title:'Send a small preview of this view to your local image model', onclick:async () => {
      describe.disabled = true; notice.textContent = 'The local model is looking at this view…';
      try { showResult(await post(route('/describe'), {image:captureView()})); }
      catch (error) { notice.textContent = error.message; }
      finally { describe.disabled = false; }
    }});
    const results = section('Results', {tip:'What your last question or action returned.', actions:[describe]});
    results.body.append(notice);
    const stepOutputs = node('div', {id:'studio-step-outputs'});
    const outputGroups = new Map();
    const expandedOutputs = new Set();
    const rememberOutputs = () => stepOutputs.querySelectorAll('[data-output-key]').forEach(group => outputGroups.set(group.dataset.outputKey, group.open));
    outputsPane.append(results.box, stepOutputs);
    // Figures a step's tools put beside the image (heatmaps) go in the dock; one per key, cleared when the step
    // or part changes.
    const showFigure = (key, box) => {
      const old = figures.querySelector('[data-figure="' + key + '"]');
      box.dataset.figure = key;
      if (old) old.replaceWith(box); else figures.append(box);
      dock.hidden = false;
    };
    const clearFigures = () => { figures.replaceChildren(); dock.hidden = true; };

    // A clickable heatmap: rows are groups (clusters, neighbourhoods, classes), columns markers or phenotypes,
    // coloured by how much a group stands out from the others. Clicking a row picks it (Shift or Ctrl adds more)
    // and onPick(rows) shows the picked groups on the image; no pick shows them all.
    const SVG = 'http://www.w3.org/2000/svg';
    const svg = (tag, attrs = {}, children = []) => {
      const element = document.createElementNS(SVG, tag);
      Object.entries(attrs).forEach(([key, value]) => { if (key === 'text') element.textContent = value; else if (value !== undefined && value !== null) element.setAttribute(key, value); });
      children.forEach(child => child && element.appendChild(child));
      return element;
    };
    const diverging = (value, palette='blue-orange') => {
      const t = Math.max(-1, Math.min(1, (Number(value) || 0) / 2.5));
      const mix = (a, b, f) => a.map((c, i) => Math.round(c + (b[i] - c) * f));
      const palettes={'blue-orange':[[59,111,182],[42,47,58],[232,116,59]],'blue-red':[[49,130,189],[245,245,245],[203,24,29]],
        viridis:[[68,1,84],[33,145,140],[253,231,37]],magma:[[0,0,4],[183,55,121],[252,253,191]]};
      const [low,middle,high]=palettes[palette]||palettes['blue-orange'];
      return 'rgb(' + (t < 0 ? mix(middle, low, -t) : mix(middle, high, t)).join(',') + ')';
    };
    // Publication figures (studio/figures.py): the server draws them as classic Tab 7 does; the browser shows and downloads.
    const downloadFile = found => {
      const bytes = Uint8Array.from(atob(found.data), ch => ch.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], {type:found.type}));
      const link = node('a', {href:url, download:found.name}); document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      return url;
    };
    const figureOptions = ({dot = false, axes = true, summary = false} = {}) => {
      const format = choice([['png', 'PNG'], ['svg', 'SVG (editable)'], ['pdf', 'PDF']], recall('figure-format', 'png'), 'Figure file type');
      const dpi = node('input', {type:'number', min:'72', max:'600', step:'50', value:recall('figure-dpi', '300'), 'aria-label':'Figure resolution (dpi)'});
      const cmap = choice(dot ? [['viridis', 'Viridis'], ['magma', 'Magma'], ['plasma', 'Plasma'], ['Reds', 'Reds'], ['Blues', 'Blues']] :
        [['RdBu_r', 'Blue–red'], ['coolwarm', 'Cool–warm'], ['bwr', 'Blue–white–red'], ['viridis', 'Viridis'], ['magma', 'Magma']], dot ? 'viridis' : 'RdBu_r', 'Figure colours');
      const clusterX = node('input', {type:'checkbox', 'aria-label':'Cluster figure X axis'}); clusterX.checked = true;
      const clusterY = node('input', {type:'checkbox', 'aria-label':'Cluster figure Y axis'}); clusterY.checked = true;
      const flip = node('input', {type:'checkbox', 'aria-label':'Groups as columns'});
      const aggregation=choice([['median','Median'],['mean','Mean']], 'median', 'Figure summary');
      const box = node('div', {class:'studio-figure-options'}, [
        node('div', {class:'row wrap'}, [format, node('label', {class:'row'}, [node('span', {class:'muted', text:'dpi'}), dpi]), cmap]),
        dot ? null : node('div', {class:'row wrap'}, [
          summary ? node('label',{class:'row'},[node('span',{text:'Summary'}),aggregation,help('Median shows the typical cell and is less affected by a few bright cells. Mean averages all cells and matches classic’s cluster summaries.')]) : null,
          axes ? node('label', {class:'row'}, [clusterX, node('span', {text:'Cluster X'})]) : null,
          axes ? node('label', {class:'row'}, [clusterY, node('span', {text:'Cluster Y'})]) : null,
          node('label', {class:'row'}, [flip, node('span', {text:'Groups as columns'})]),
          axes ? help('Cluster X reorders the horizontal axis; Cluster Y reorders the vertical axis. Choose either, both, or neither. This only groups similar profiles for display.') : null].filter(Boolean))].filter(Boolean));
      const read = () => {
        remember('figure-format', format.value); remember('figure-dpi', dpi.value);
        return {format:format.value, dpi:Math.round(Number(dpi.value) || 300), cmap:cmap.value,
          cluster_x:axes && clusterX.checked, cluster_y:axes && clusterY.checked,
          ...(summary ? {aggregation:aggregation.value} : {}), groups_on:flip.checked ? 'columns' : 'rows'};
      };
      return {box, read};
    };

    // The heatmap fills the width it is given (drag the pane wider, or use Large view over the image) and is
    // redrawn in place by update(): names, colours, notes (a text per row, right of the colours) and picks.
    const heatmapFigure = ({key, title, tip, profile, labels = {}, colours = {}, notes = {}, onPick, picked = [], tools = null, onAggregation = null, onDisplay = null}) => {
      const {rows, columns} = profile;
      const composition = key === 'neighbourhoods' && !!profile.raw;
      const display = composition ? choice([['share','Share (%)'],['relative','Relative (z-score)']],recall('heatmap-values:'+key,'share'),'Neighbourhood heatmap values') : null;
      const isShare = () => display?.value === 'share';
      let values = isShare() ? profile.raw.map(row => row.map(v => 100*v)) : profile.values;
      const scaleLabel = () => isShare() ? 'Share (%)' : 'Z-score';
      const scaleTicks = () => isShare() ? [100,50,0] : [2.5,0,-2.5];
      const tickLabel = v => isShare() ? String(v) : (v>0?'+':'')+v;
      const heatColour = (value, palette) => diverging(isShare() ? value/20-2.5 : value,palette);
      const caption = () => composition ? (isShare() ? 'Mean share of each phenotype in the local windows (%)' : 'Mean shares compared across neighbourhood types (z-score)') : profile.note || '';
      const view = {labels, colours, notes};
      const chosen = new Set(picked.filter(row => rows.includes(row)));
      const {box, body} = section(title, {tip});
      const head = box.querySelector('.studio-sec-h'), count = chip('');
      head.insertBefore(count, head.querySelector('.grow'));
      const holder = node('div', {class:'studio-heatmap-scroll'});
      const oldOrder=recall('heatmap-order:'+key,'original');
      const order=choice([['original','Cluster number'],['name','Name']],oldOrder==='similarity'?'original':oldOrder,'Heatmap row order');
      const clusterX=node('input',{type:'checkbox','aria-label':'Cluster heatmap X axis'});
      const clusterY=node('input',{type:'checkbox','aria-label':'Cluster heatmap Y axis'});
      clusterX.checked=recall('heatmap-cluster-x:'+key,oldOrder==='similarity');
      clusterY.checked=recall('heatmap-cluster-y:'+key,oldOrder==='similarity');
      order.disabled=clusterY.checked;
      const palette=choice([['blue-orange','Blue–orange'],['blue-red','Blue–red'],['viridis','Viridis'],['magma','Magma']],recall('heatmap-palette:'+key,composition?'viridis':'blue-orange'),'Heatmap colours');
      const heatControls=node('div',{class:'row wrap heatmap-controls'},[node('span',{class:'muted',text:'Sort'}),order,palette,
        node('label',{class:'row'},[clusterX,node('span',{text:'Cluster X'})]),
        node('label',{class:'row'},[clusterY,node('span',{text:'Cluster Y'})]),
        help('Cluster X groups similar columns; Cluster Y groups similar rows. Choose either, both, or neither. Sorting by name is available when Y clustering is off. Ctrl, ⌘ or Shift-click adds rows to the selection. ' +
          (composition ? 'Share shows mean local phenotype percentages, from 0 to 100. Relative shows each phenotype compared across neighbourhood types, from −2.5 to +2.5 z-score; it is not fold enrichment. The list on the left follows this choice. ' : 'Colours show z-scores from −2.5 to +2.5, not fold enrichment. ') + 'Display choices do not change classification.')]);
      if (display) heatControls.prepend(display);
      if (onAggregation) {
        const methods=profile.available_summaries || Object.keys(profile.summaries || {});
        if (methods.length) {
          const summary=choice(methods.map(m=>[m,m==='median'?'Median':'Mean']),profile.aggregation || 'median','Heatmap summary');
          summary.onchange=async()=>{
            summary.disabled=true;const expanded=!!large;if(expanded)closeLarge();
            try{
              await onAggregation(summary.value);
              if(expanded)figures.querySelector('[data-figure="'+key+'"] button[title="Show this figure over the image, with more room"]')?.click();
            }catch(error){summary.value=profile.aggregation || 'median';showResult({message:error.message});}
            finally{summary.disabled=false;}
          };
          heatControls.prepend(node('span',{class:'muted',text:'Summary'}),summary,
            help('Median (default): shows a typical cell or pixel and is less affected by a few very bright ones. Mean: averages all values and gives bright cells or pixels more influence. Use Mean to match classic phenotyping. This changes the heatmap and marker ranking, not cluster membership.'));
        }
      }
      const similarityOrder=matrix=>{
        if(matrix.length<2)return matrix.map((_,i)=>i);
        let groups=matrix.map((v,i)=>({ids:[i],v}));
        const distance=(a,b)=>a.reduce((s,v,i)=>s+(v-b[i])**2,0);
        // Ward merge cost; saved SciPy orders take precedence for clustering runs.
        while(groups.length>1){let best=Infinity,pair=[0,1];
          for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){const d=distance(groups[i].v,groups[j].v)*groups[i].ids.length*groups[j].ids.length/(groups[i].ids.length+groups[j].ids.length);if(d<best){best=d;pair=[i,j];}}
          const [i,j]=pair,a=groups[i],b=groups[j],size=a.ids.length+b.ids.length;
          const merged={ids:a.ids.concat(b.ids),v:a.v.map((v,k)=>(v*a.ids.length+b.v[k]*b.ids.length)/size)};
          groups.splice(j,1);groups.splice(i,1,merged);
        }return groups[0].ids;
      };
      let computedRows,computedCols;
      const ordered=()=>{
        let ri=rows.map((_,i)=>i),ci=columns.map((_,i)=>i);
        if(order.value==='name')ri.sort((a,b)=>String(view.labels[rows[a]]||rows[a]).localeCompare(String(view.labels[rows[b]]||rows[b]),undefined,{numeric:true}));
        if(clusterY.checked)ri=profile.row_order||(computedRows ||= similarityOrder(values));
        if(clusterX.checked)ci=profile.column_order||(computedCols ||= similarityOrder(columns.map((_,i)=>values.map(row=>row[i]))));
        return {ri,ci};
      };
      let figure = null, lines = [], drawnWidth = 0, large = null;
      const textMeasure=document.createElement('canvas').getContext('2d');
      const textWidth=text=>{textMeasure.font=(large?'11':'10')+'px sans-serif';return textMeasure.measureText(String(text)).width;};
      const sync = () => {
        lines.forEach(line => line.classList.toggle('picked', chosen.has(line.dataset.row)));
        figure?.classList.toggle('picking', chosen.size > 0);
        count.textContent = chosen.size ? chosen.size + ' picked' : 'All shown'; count.className = 'chip' + (chosen.size ? ' ok' : '');
      };
      const pick = (row, event) => {
        if (!(event.shiftKey || event.ctrlKey || event.metaKey)) { const only = chosen.has(row) && chosen.size === 1; chosen.clear(); if (!only) chosen.add(row); }
        else chosen.has(row) ? chosen.delete(row) : chosen.add(row);
        sync(); onPick(Array.from(chosen));
      };
      const draw = () => {
        const {ri,ci}=ordered();
        const room = Math.max(260, holder.clientWidth || 320);
        const swatches = rows.some(row => view.colours[row]);
        const label = row => String(view.labels[row] || row);
        const left = Math.min(220, Math.max(96, 6.6 * Math.min(Math.max(...rows.map(row => label(row).length), 4), 30) + (swatches ? 18 : 6)));
        const noteText = Object.values(view.notes).map(String);
        const noteWidth = noteText.length ? Math.max(80, Math.max(...noteText.map(textWidth))) + 12 : 0;
        const legendWidth=76;
        const cell = Math.max(14, Math.min(large ? 72 : 42, Math.floor((room - left - noteWidth - legendWidth - 10) / Math.max(columns.length, 1)))), tall = large ? 20 : 16;
        const markerWidth=Math.max(...columns.map(textWidth),0);
        const top = 12 + Math.ceil(markerWidth*Math.sin(Math.PI/3));
        const scaleX=left+columns.length*cell+noteWidth+18, scaleH=Math.max(56,Math.min(180,rows.length*tall-24));
        const width = Math.max(scaleX+legendWidth,left+columns.length*cell+markerWidth*.5+12), height = top + Math.max(rows.length*tall,scaleH+24) + 8;
        figure = svg('svg', {width, height, style:'width:' + width + 'px;height:' + height + 'px', class:'studio-heatmap', role:'img', 'aria-label':title});
        ci.forEach((source, column) => {const name=columns[source];figure.append(svg('text', {x:left + column * cell + cell / 2, y:top - 4, class:'column',
          transform:'rotate(-60 ' + (left + column * cell + cell / 2) + ' ' + (top - 4) + ')', text:name}));});
        const fitLabel = text => {
          const available=left-(swatches?22:8);
          if(textWidth(text)<=available)return text;
          let n=text.length;while(n>0&&textWidth(text.slice(0,n)+'…')>available)n--;
          return text.slice(0,n)+'…';
        };
        lines = ri.map((sourceIndex, index) => {const row=rows[sourceIndex];
          const y = top + index * tall, text = label(row);
          const line = svg('g', {class:'row', 'data-row':row, tabindex:'0', role:'button', 'aria-label':'Show ' + text + ' on the image'});
          line.append(svg('rect', {x:0, y, width, height:tall, class:'band'}));
          if (view.colours[row]) line.append(svg('rect', {x:2, y:y + 3, width:10, height:10, fill:view.colours[row], rx:2}));
          line.append(svg('text', {x:swatches ? 17 : 2, y:y + tall - 4, class:'label', text:fitLabel(text)}, [svg('title', {text})]));
          ci.forEach((sourceColumn, column) => {const name=columns[sourceColumn];
            const value = values[sourceIndex][sourceColumn], raw = profile.raw ? profile.raw[sourceIndex][sourceColumn] : null;
            const detail = composition ? 'mean share ' + Number(raw*100).toFixed(1) + '%; relative z-score ' + Number(profile.values[sourceIndex][sourceColumn]).toFixed(2) :
              (raw !== null ? (profile.raw_label || 'Summary value') + ' ' + Number(raw).toFixed(3) + '; ' : '') + 'z-score ' + (value >= 0 ? '+' : '') + Number(value).toFixed(2);
            line.append(svg('rect', {x:left + column * cell, y:y + 1, width:cell - 1, height:tall - 2, fill:heatColour(value,palette.value)},
              [svg('title', {text:text + ' · ' + name + ': ' + detail})]));
          });
          if (view.notes[row]) line.append(svg('text', {x:left + columns.length * cell + 6, y:y + tall - 4, class:'note', text:String(view.notes[row])}));
          line.addEventListener('click', event => pick(row, event));
          line.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pick(row, event); } });
          figure.append(line);
          return line;
        });
        const gradientId='heatmap-scale-'+key.replace(/[^a-z0-9]/gi,'-');
        const gradient=svg('linearGradient',{id:gradientId,x1:'0',y1:'1',x2:'0',y2:'0'},
          Array.from({length:21},(_,i)=>svg('stop',{offset:(i*5)+'%','stop-color':diverging(-2.5+i*.25,palette.value)})));
        figure.append(svg('defs',{},[gradient]),svg('text',{x:scaleX,y:top+9,class:'scale-label',text:scaleLabel()}),
          svg('rect',{x:scaleX,y:top+18,width:10,height:scaleH,fill:'url(#'+gradientId+')',class:'heatmap-scale'}));
        scaleTicks().forEach((v,i)=>figure.append(svg('text',{x:scaleX+15,y:top+22+i*scaleH/2,text:tickLabel(v)})));
        holder.replaceChildren(figure); drawnWidth = room; sync();
      };
      // Large view: the figure over the image, with room for every column; Close puts it back.
      const expand = node('button', {class:'btn small ghost', text:'Large view', title:'Show this figure over the image, with more room'});
      expand.addEventListener('click', () => {
        if (large) { closeLarge(); return; }
        const stage = document.getElementById('stage');
        large = node('div', {class:'studio-figure-large', role:'dialog', 'aria-label':title});
        box.before(node('div', {class:'studio-figure-placeholder', 'data-for':key}));
        large.append(box); stage.append(large); expand.textContent = 'Back to the pane';
        requestAnimationFrame(draw);
      });
      const closeLarge = () => {
        const spot = figures.querySelector('.studio-figure-placeholder[data-for="' + key + '"]');
        if (spot) spot.replaceWith(box); else figures.append(box);
        large?.remove(); large = null; expand.textContent = 'Large view'; requestAnimationFrame(draw);
      };
      head.append(expand);
      // Save figure: this heatmap as a publication figure (clustered, the names shown), PNG, SVG or PDF.
      const saving = figureOptions({axes:false});
      order.onchange=()=>{remember('heatmap-order:'+key,order.value);draw();};
      clusterX.onchange=()=>{remember('heatmap-cluster-x:'+key,clusterX.checked);draw();};
      clusterY.onchange=()=>{remember('heatmap-cluster-y:'+key,clusterY.checked);order.disabled=clusterY.checked;draw();};
      palette.onchange=()=>{remember('heatmap-palette:'+key,palette.value);draw();};
      const saveNote = node('span', {class:'muted'});
      const saveFigure = node('button', {class:'btn small', text:'Save figure', onclick:async () => {
        saveFigure.disabled = true; saveNote.textContent = 'Drawing…';
        try {
          const {ri,ci}=ordered();
          const found = await post(route('/figure'), {title, options:{...saving.read(),cluster_x:false,cluster_y:false,vmin:isShare()?0:-2.5,vmax:isShare()?100:2.5,value_label:scaleLabel(),cmap:({'blue-orange':'coolwarm','blue-red':'RdBu_r',viridis:'viridis',magma:'magma'})[palette.value]},
            matrix:{rows:ri.map(i => view.labels[rows[i]] || rows[i]), columns:ci.map(i=>columns[i]), values:ri.map(i=>ci.map(j=>values[i][j]))}});
          downloadFile(found); saveNote.textContent = 'Saved ' + found.name + '.';
        } catch (error) { saveNote.textContent = error.message; }
        finally { saveFigure.disabled = false; }
      }});
      const savePanel = node('details', {class:'studio-options'}, [node('summary', {text:'Figure'}), saving.box, node('div', {class:'row'}, [saveFigure, saveNote])]);
      const all = node('button', {class:'btn small', text:'Show all', onclick:() => { chosen.clear(); sync(); onPick([]); }});
      const captionNode = node('span', {class:'muted grow', text:caption()});
      if (display) display.onchange = () => {
        remember('heatmap-values:'+key,display.value);
        values = isShare() ? profile.raw.map(row => row.map(v => 100*v)) : profile.values;
        computedRows = computedCols = null;
        captionNode.textContent = caption(); onDisplay?.(display.value); draw();
      };
      onDisplay?.(display?.value || 'relative');
      body.append(...[heatControls,tools, holder, node('div', {class:'row'}, [all, captionNode]), savePanel].filter(Boolean));
      showFigure(key, box);
      draw();
      const watcher = new ResizeObserver(() => {
        if (!box.isConnected) { watcher.disconnect(); return; }
        if (Math.abs((holder.clientWidth || 0) - drawnWidth) > 12) draw();
      });
      watcher.observe(holder);
      const capture = () => {
        const {ri,ci} = ordered(), cell = 22, left = 200, rowHeight = 20;
        const canvas = document.createElement('canvas'),ctx=canvas.getContext('2d');ctx.font='11px sans-serif';
        const top=40+Math.ceil(Math.max(...columns.map(c=>ctx.measureText(String(c)).width),0)*Math.sin(Math.PI/3));
        canvas.width = left + columns.length*cell + 90; canvas.height = top + Math.max(rows.length*rowHeight,130) + 36;
        ctx.fillStyle = '#151820'; ctx.fillRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 14px sans-serif'; ctx.fillText(title,8,20); ctx.font = '11px sans-serif';
        ci.forEach((source,col)=>{ctx.save();ctx.translate(left+col*cell+12,top-6);ctx.rotate(-Math.PI/3);ctx.fillText(columns[source],0,0);ctx.restore();});
        ri.forEach((source,row)=>{
          const y=top+row*rowHeight;
          if(chosen.has(rows[source])){ctx.fillStyle='#31476c';ctx.fillRect(0,y,canvas.width,rowHeight);}
          ctx.fillStyle='#fff';ctx.fillText(String(view.labels[rows[source]]||rows[source]),5,y+14,left-10);
          ci.forEach((column,col)=>{ctx.fillStyle=heatColour(values[source][column],palette.value);ctx.fillRect(left+col*cell,y+1,cell-1,rowHeight-2);});
        });
        const sx=left+columns.length*cell+18,sh=Math.min(180,Math.max(80,rows.length*rowHeight-24));
        const gradient=ctx.createLinearGradient(0,top+18+sh,0,top+18);
        for(let i=0;i<=20;i++)gradient.addColorStop(i/20,diverging(-2.5+i*.25,palette.value));
        ctx.fillStyle=gradient;ctx.fillRect(sx,top+18,10,sh);ctx.fillStyle='#ddd';ctx.fillText(scaleLabel(),sx,top+8);
        scaleTicks().forEach((v,i)=>ctx.fillText(tickLabel(v),sx+15,top+22+i*sh/2));
        ctx.fillStyle='#ddd';ctx.fillText(caption() || 'Marker profiles compared across clusters',8,canvas.height-12,canvas.width-16);
        return canvas.toDataURL('image/png');
      };
      return {box, capture, picked:() => Array.from(chosen), update:changes => {
        Object.assign(view, changes);
        if (changes.picked) { chosen.clear(); changes.picked.filter(row => rows.includes(row)).forEach(row => chosen.add(row)); }
        draw();
      }};
    };

    // ------------------------------------------------------------ the assistant
    // A chat beside the image: it knows the step, the image and the project, answers in plain words or proposes one
    // checked action (Do it runs it), and keeps its conversations and memories locally.
    const chatState = {conversation:recall('chat', ''), busy:false};
    const modelLine = node('div', {class:'row studio-chat-model'});
    const thread = node('div', {class:'studio-chat', role:'log', 'aria-live':'polite'});
    const quick = node('div', {class:'row wrap studio-chat-quick'});
    const chatInput = node('textarea', {rows:'2', placeholder:'Ask about this view, or ask Studio to do something… (paste a screenshot here)', 'aria-label':'Message the assistant'});
    // Pictures for the assistant: the current view (when ticked), and screenshots pasted or attached. They go with the
    // next question only, to a model that can see images; the server checks and re-encodes them and never keeps them.
    const attached = [];
    const attachStrip = node('div', {class:'studio-attachments'});
    const drawAttached = () => attachStrip.replaceChildren(...attached.map((url, index) => node('span', {class:'studio-attachment'}, [
      node('img', {src:url, alt:'Attached image ' + (index + 1)}),
      node('button', {type:'button', class:'btn small ghost', text:'×', 'aria-label':'Remove image ' + (index + 1), onclick:() => { attached.splice(index, 1); drawAttached(); }})])));
    const addPicture = file => {
      if (!file || !/^image\/(png|jpeg|webp)$/.test(file.type)) return;
      if (attached.length >= 3) { addMessage('note', 'At most three pictures per question (the view counts as a fourth).'); return; }
      const reader = new FileReader();
      reader.onload = () => { attached.push(String(reader.result)); drawAttached(); };
      reader.readAsDataURL(file);
    };
    chatInput.addEventListener('paste', event => {
      let files = Array.from(event.clipboardData?.items || []).filter(item => item.kind === 'file' && item.type.startsWith('image/')).map(item => item.getAsFile());
      if (!files.length) files = Array.from(event.clipboardData?.files || []).filter(file => file.type.startsWith('image/'));
      if (files.length) { event.preventDefault(); files.forEach(addPicture); }
    });
    const pickPictures = node('input', {type:'file', accept:'image/png,image/jpeg,image/webp', multiple:true, hidden:true, 'aria-label':'Attach images'});
    pickPictures.addEventListener('change', () => { Array.from(pickPictures.files).forEach(addPicture); pickPictures.value = ''; });
    const attachButton = node('button', {type:'button', class:'btn small ghost', text:'Attach image', onclick:() => pickPictures.click()});
    // The whole Studio page as a picture: the browser asks which tab to share, one frame is taken, sharing stops.
    const capturePage = node('button', {type:'button', class:'btn small ghost', text:'Capture page', title:'Attach a picture of the whole Studio page (your browser asks you to choose this tab)',
      onclick:async () => {
        if (!navigator.mediaDevices?.getDisplayMedia) { addMessage('note', 'This browser cannot capture the page; attach a screenshot instead.'); return; }
        let stream;
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({video:{displaySurface:'browser'}, audio:false, preferCurrentTab:true, selfBrowserSurface:'include'});
          const video = node('video', {muted:true, playsinline:true}); video.srcObject = stream; await video.play();
          await new Promise(resolve => setTimeout(resolve, 350));
          const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
          const canvas = document.createElement('canvas'); canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
          canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
          if (attached.length >= 3) attached.pop();
          attached.push(canvas.toDataURL('image/png')); drawAttached();
        } catch (error) { if (error.name !== 'NotAllowedError') addMessage('note', 'The page could not be captured: ' + error.message); }
        finally { stream?.getTracks().forEach(track => track.stop()); }
      }});
    const pageContext = () => ({
      step:(context?.steps || []).find(item => item.id === selected)?.name || selected || '',
      phenotyping_method:selected === 'tab7' ? method : undefined,
      explorer:explorerNow ? explorerNow() : undefined,
      right_pane_tab:recall('tab', ''),
    });
    const withView = node('input', {type:'checkbox', 'aria-label':'Show the assistant the view'}); withView.checked = recall('chat-view', '1') === '1';
    withView.addEventListener('change', () => remember('chat-view', withView.checked ? '1' : '0'));
    const sendButton = node('button', {class:'btn primary', text:'Send'});
    const stopChat = node('button', {class:'btn', text:'Stop', hidden:true, 'aria-label':'Stop assistant reply', onclick:() => chatState.abort?.abort()});
    const memoryBox = node('details', {class:'studio-memories'});
    // AI setup, here instead of classic's page: on or off, local models (download, use, start, stop), the online
    // model (used only when chosen), memory mode, stopping with SPACEMAP, and servers earlier sessions left running.
    const setupBox = node('section', {class:'studio-ai-setup', hidden:true, 'aria-label':'AI setup'});
    let setupTimer = null;
    const drawSetup = async value => {
      clearTimeout(setupTimer);
      if (!value) {
        if (!setupBox.childElementCount) setupBox.append(node('p', {class:'muted', text:'Reading the AI setup…'}));
        try { value = await get(route('/ai')); } catch (error) { setupBox.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      }
      const status = node('p', {class:'muted', role:'status'});
      const act = async (action, body = {}, done = '') => {
        status.textContent = 'Working…';
        try { await drawSetup(await post(route('/ai/' + action), body)); setupStatus(done); loadChat(); }
        catch (error) { status.textContent = error.message; }
      };
      const setupStatus = text => { const line = setupBox.querySelector('[role=status]'); if (line) line.textContent = text; };
      const on = node('input', {type:'checkbox', 'aria-label':'Assistant on'}); on.checked = value.enabled; on.disabled = value.pinned;
      on.addEventListener('change', () => act('enable', {on:on.checked}, on.checked ? 'The assistant is on.' : 'The assistant is off.'));
      const runtime = value.runtime;
      const rows = value.models.map(model => {
        const buttons = [];
        let state = model.installed ? 'Downloaded' : 'Not downloaded', kind = '';
        if (model.downloading) {
          state = 'Downloading ' + model.percent.toFixed(0) + '%'; kind = 'warn';
          buttons.push(node('button', {class:'btn small ghost', text:'Cancel', onclick:() => act('cancel', {model:model.id}, 'Cancelled; Download resumes it.')}));
        } else if (!model.installed) {
          buttons.push(node('button', {class:'btn small', text:'Download', title:'About ' + model.size_gb + ' GB', onclick:() => act('download', {model:model.id}, 'Downloading ' + model.name + '…')}));
          if (model.failed) state = 'Download failed';
        } else if (!model.active) {
          buttons.push(node('button', {class:'btn small', text:'Use', onclick:() => act('use', {model:model.id}, model.name + ' will answer.')}));
        } else if (runtime.ready) {
          state = 'In use · ready'; kind = 'ok';
          buttons.push(node('button', {class:'btn small ghost', text:'Stop', onclick:() => act('stop', {}, 'Stopped.')}));
        } else if (runtime.running) {
          state = 'In use · loading'; kind = 'warn';
          buttons.push(node('button', {class:'btn small ghost', text:'Stop', onclick:() => act('stop', {}, 'Stopped.')}));
        } else {
          state = 'In use · stopped';
          buttons.push(node('button', {class:'btn small primary', text:'Start', onclick:() => act('start', {}, 'Starting ' + model.name + ' (about a minute on a GPU)…')}));
        }
        return node('div', {class:'studio-folder'}, [node('span', {class:'grow', text:model.name + ' · ' + model.size_gb + ' GB', title:model.failed || model.why || ''}),
          model.fits ? null : chip('May be slow here', 'warn', model.why), chip(state, kind), ...buttons]);
      });
      const precision = choice(value.precisions, value.precision, 'GPU memory mode');
      precision.addEventListener('change', () => act('settings', {precision:precision.value}, 'Used the next time the model starts.'));
      const withSpacemap = node('input', {type:'checkbox', 'aria-label':'Stop the model when SPACEMAP stops'}); withSpacemap.checked = value.stop_with_spacemap;
      withSpacemap.addEventListener('change', () => act('settings', {stop_with_spacemap:withSpacemap.checked}, 'Used the next time the model starts.'));
      const left = value.leftovers;
      const online = value.online;
      const useOnline = node('input', {type:'checkbox', 'aria-label':'Answer with an online model'}); useOnline.checked = online.chosen;
      const onlineModel = choice(online.models, online.model, 'Online model');
      const onlineDetails = (online.details || {})[online.model] || {};
      const checked = (online.checks || {})[online.model];
      const checkOnline = node('button', {class:'btn small', text:'Check connection', disabled:!online.key_saved,
        onclick:async () => {
          checkOnline.disabled = true; checkOnline.textContent = 'Checking…';
          await act('check_online', {model:onlineModel.value}, 'Connection check finished.');
          checkOnline.disabled = false; checkOnline.textContent = 'Check connection';
        }});
      const key = node('input', {type:'password', autocomplete:'off', placeholder:online.key_saved ? 'Saved (type a new one to replace it)' : 'NVIDIA API key', 'aria-label':'NVIDIA API key'});
      const saveOnline = () => act('online', {use:useOnline.checked, model:onlineModel.value, key:key.value.trim() || undefined},
        useOnline.checked ? 'Answers come from the online model.' : 'Answers come from the local model.');
      useOnline.addEventListener('change', saveOnline); onlineModel.addEventListener('change', saveOnline);
      setupBox.replaceChildren(...[
        node('div', {class:'row'}, [node('label', {class:'row grow'}, [on, node('b', {text:'Assistant on'})]), node('span', {class:'muted', text:value.hardware})]),
        value.missing.length ? node('div', {class:'studio-import'}, [node('p', {class:'hint', text:'The local model needs ' + value.missing.join(', ') + '. Install into SPACEMAP’s environment, then restart SPACEMAP:'}),
          node('code', {class:'mono', text:value.install})]) : null,
        node('p', {class:'lbl', text:'Local model (answers stay on this computer)'}), ...rows,
        left.length ? node('div', {class:'studio-import'}, [node('p', {class:'hint', text:left.length + ' model server' + (left.length === 1 ? '' : 's') +
            ' left running by earlier SPACEMAP sessions on this node (port ' + left.map(item => item.port).join(', ') + '), holding memory and ports.'}),
          node('button', {class:'btn small', text:'Stop ' + (left.length === 1 ? 'it' : 'them'), onclick:() => act('stop_leftovers', {}, 'Stopped.')})]) : null,
        field('GPU memory', precision, 'Less memory loads bigger models on smaller GPUs, a little less exactly. Used the next time the model starts.'),
        node('label', {class:'row'}, [withSpacemap, node('span', {class:'grow', text:'Stop the model when SPACEMAP stops'}),
          help('On: the model server stops when SPACEMAP stops, so it never keeps its port and memory. Off: it keeps running, so the next start is instant.')]),
        runtime.port ? node('p', {class:'muted', text:runtime.message + ' Port ' + runtime.port + '.'}) : node('p', {class:'muted', text:runtime.message}),
        node('details', {class:'studio-options', open:online.chosen || undefined}, [node('summary', {text:'Online model instead (optional)'}),
          node('p', {class:'muted', text:'Off unless you choose it. Online models run on NVIDIA’s servers: your question and the recent chat leave this computer.'}),
          node('label', {class:'row'}, [useOnline, node('span', {text:'Answer with the online model'})]), field('Model', onlineModel),
          node('div', {class:'row'}, [chip(onlineDetails.images ? 'Text + images' : 'Text only'), checkOnline,
            help((onlineDetails.description || '') + ' The connection check sends a tiny test message. It checks availability and reply speed, not biological accuracy. Speed varies with server load.')]),
          checked ? node('p', {class:checked.ok ? 'muted' : 'warn', text:checked.message + ' · ' + checked.seconds + ' s',
            title:'Checked ' + checked.checked_at + '. This result describes that request only.'}) : null,
          node('div', {class:'row'}, [key, node('button', {class:'btn small', text:'Save key', onclick:() => { if (key.value.trim()) saveOnline(); }})]),
          online.key_saved ? node('div', {class:'row'}, [chip('Key saved', 'ok'), node('button', {class:'btn small ghost', text:'Forget key',
            onclick:() => act('forget_key', {}, 'The key is deleted; answers come from the local model.')})]) : null]),
        status].filter(Boolean));
      if (!setupBox.hidden && (value.models.some(model => model.downloading) || (runtime.running && !runtime.ready))) setupTimer = setTimeout(() => drawSetup(), 3000);
    };
    openAiSetup = () => { showTab('assistant'); if (setupBox.hidden) setupButton.click(); };
    const setupButton = node('button', {class:'btn small', text:'AI setup', 'aria-expanded':'false', onclick:() => {
      setupBox.hidden = !setupBox.hidden; setupButton.setAttribute('aria-expanded', String(!setupBox.hidden));
      if (!setupBox.hidden) drawSetup(); else clearTimeout(setupTimer);
    }});
    assistantPane.append(modelLine, setupBox, thread, quick, attachStrip, node('div', {class:'studio-chat-input'}, [chatInput, sendButton, stopChat]),
      node('div', {class:'row studio-chat-tools'}, [node('label', {class:'row grow'}, [withView, node('span', {text:'Show it the view'})]),
        help('Sends a picture of what the viewer shows, and what is on screen (markers, colours, zoom), with your question, to a model that can see images.'),
        capturePage, attachButton, pickPictures]), memoryBox);
    const bubble = (role, text) => node('div', {class:'studio-bubble ' + role}, [node('p', {text})]);
    const actionCard = (action, context = {}) => {
      if (context.execution?.args && context.execution.name === action.name) action = {...action,args:context.execution.args};
      const place = node('div', {class:'studio-action-result'});
      // The plan, shown before Do it: every cluster's name (editable) for naming; the arguments otherwise.
      let plan = null, edits = null;
      if (action.name === 'name_clusters' && action.args?.names) {
        edits = {};
        plan = node('table', {class:'studio-counts studio-plan'}, [node('tr', {}, [node('th', {text:'Cluster'}), node('th', {text:'Name (edit before Do it)'})]),
          ...Object.entries(action.args.names).map(([cluster, name]) => {
            const input = node('input', {type:'text', value:name, maxlength:'60', 'aria-label':'Name of ' + cluster}); edits[cluster] = input;
            return node('tr', {}, [node('td', {text:cluster}), node('td', {}, [input])]);
          })]);
      } else if (action.args && Object.keys(action.args).length) {
        plan = node('p', {class:'muted mono', text:Object.entries(action.args).map(([key, value]) => key + ': ' + (typeof value === 'object' ? JSON.stringify(value) : value)).join(' · ').slice(0, 400)});
      }
      const go = async () => {
        run.disabled = true;
        const command = edits ? {...action, args:{...action.args, names:Object.fromEntries(Object.entries(edits).map(([cluster, input]) => [cluster, input.value.trim()]).filter(([, name]) => name))}} : action;
        if (edits) {
          const clear = [...(action.args.clear || []), ...Object.entries(edits).filter(([,input])=>!input.value.trim()).map(([cluster])=>cluster)];
          if (clear.length) command.args.clear = clear;
          if (!Object.keys(command.args.names).length) delete command.args.names;
        }
        try {
          const result = await perform(command, {confirmed:true, stay:true,
            ...(context.message_id ? {assistant:{conversation:context.conversation,message:context.message_id,index:context.index || 0}} : {})}, place);
          if (result?.status !== 'done') throw Error(result?.message || 'The change did not run.');
          if (result?.data?.settings) window.dispatchEvent(new CustomEvent('studio-settings-changed', {detail:result.data.settings}));
          if (result.data?.job) {
            const job = await waitForJob(result.data.job.id, message => { place.textContent = message; }, card);
            await refreshJobs();
            if (job.status !== 'done') throw Error(job.message || 'The job did not finish.');
            place.textContent = job.result?.column ? 'Saved in ' + job.result.column + '.' : job.message || 'Done.';
            if (command.name === 'name_clusters') {
              window.dispatchEvent(new CustomEvent('studio-cluster-names', {detail:{job_id:command.args.job_id,names:job.result.names || command.args.names || {},replace:true,phase:'saved',column:job.result.column}}));
              await hooks.refreshPhenotypingLinks?.();
              await colourCells(job.result.column);
              if (frame?.colour) { await loadColours(frame); frame.colour.value = job.result.column; }
            }
          }
          run.textContent = 'Done'; card.completed = true;
        } catch (error) {
          place.replaceChildren(node('p', {class:'hint', text:error.message}));
          if (command.name === 'name_clusters') window.dispatchEvent(new CustomEvent('studio-cluster-names', {detail:{job_id:command.args.job_id,phase:'failed',message:error.message}}));
        }
        finally { run.disabled = !!card.completed; }
      };
      const run = node('button', {class:'btn primary small', text:'Do it', onclick:go});
      const card = node('div', {class:'studio-action'}, [node('span', {class:'mono', text:action.name.replace(/_/g, ' ')}), run, plan, place].filter(Boolean));
      card.run = go;
      if (context.execution) {
        place.textContent = context.execution.status === 'done' ? 'Saved' + (context.execution.column ? ' in ' + context.execution.column : '') + '.' : context.execution.message;
        run.disabled = true; run.textContent = context.execution.status === 'done' ? 'Done' : 'See Jobs';
      }
      return card;
    };
    // A plan: several checked actions in order, each ticked; Do it (the confirmation) runs the ticked ones one after
    // another, waiting for each job to finish, and stops at the first that does not.
    const stepText = step => step.name.replace(/_/g, ' ') + (step.args && Object.keys(step.args).length ? ': ' + Object.entries(step.args)
      .map(([key, value]) => key + ' ' + (typeof value === 'object' ? JSON.stringify(value) : value)).join(' · ').slice(0, 300) : '');
    const waitForJob = async (id, onMessage, card) => {
      for (;;) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        if (!card.isConnected) return {status:'stopped', message:'No longer followed here; see Jobs.'};
        let job;
        try { job = (await get(route('/jobs/' + id))).job; } catch (error) { onMessage(error.message); continue; }
        if (!['starting', 'running', 'stopping'].includes(job.status)) return job;
        onMessage(job.message || 'Running…');
      }
    };
    const planCard = (steps, context = {}) => {
      const note = node('p', {class:'muted', role:'status'});
      const rows = steps.map((step, index) => {
        const tick = node('input', {type:'checkbox', 'aria-label':'Include step ' + (index + 1)}); tick.checked = true;
        const state = node('span', {class:'muted studio-plan-state'});
        const line = node('li', {class:'studio-plan-step'}, [tick, node('span', {class:'grow mono', text:stepText(step)}),
          step.changes_project ? chip('changes project', 'warn', 'Saves into your project (a new column, file or settings); earlier results are kept') : null, state].filter(Boolean));
        return {step, tick, state, line};
      });
      const set = (row, text, kind = '') => { row.state.textContent = text; row.state.className = 'studio-plan-state ' + (kind || 'muted'); };
      const card = node('div', {class:'studio-action studio-plan-card'});
      const go = async () => {
        run.disabled = true; rows.forEach(row => { row.tick.disabled = true; set(row, row.tick.checked ? 'waiting' : 'skipped'); });
        let stopped = false;
        for (const row of rows) {
          if (!row.tick.checked) continue;
          if (stopped) { set(row, 'not run'); continue; }
          set(row, 'running…', 'run');
          try {
            const result = await runCommand(row.step.name, row.step.args, {confirmed:true,
              ...(context.message_id ? {assistant:{conversation:context.conversation,message:context.message_id,index:rows.indexOf(row)}} : {})});
            showResult(result);
            if (result.status !== 'done') throw new Error(result.message || 'It did not run.');
            if (result.data?.settings) window.dispatchEvent(new CustomEvent('studio-settings-changed', {detail:result.data.settings}));
            if (result.data?.job) {
              await refreshJobs();
              const job = await waitForJob(result.data.job.id, message => set(row, message, 'run'), card);
              await refreshJobs();
              if (job.status !== 'done') throw new Error(job.message || 'The job stopped.');
              set(row, 'done: ' + (job.message || ''), 'ok');
              if (row.step.name === 'name_clusters') {
                window.dispatchEvent(new CustomEvent('studio-cluster-names', {detail:{job_id:row.step.args.job_id,names:job.result.names || row.step.args.names || {},replace:true,phase:'saved',column:job.result.column}}));
                await hooks.refreshPhenotypingLinks?.(); await colourCells(job.result.column);
                if (frame?.colour) { await loadColours(frame); frame.colour.value = job.result.column; }
              }
            } else set(row, 'done: ' + (result.message || ''), 'ok');
          } catch (error) { set(row, 'stopped: ' + error.message, 'hint'); stopped = true; }
        }
        note.textContent = stopped ? 'The plan stopped; the steps before it are kept. Ask again from there.' : 'Every ticked step ran.';
        run.textContent = 'Run again'; run.disabled = false; rows.forEach(row => { row.tick.disabled = false; });
      };
      const run = node('button', {class:'btn primary small', text:'Do it', title:'Runs the ticked steps in order', onclick:go});
      card.append(node('span', {class:'mono', text:'plan · ' + steps.length + ' steps'}), run, node('ol', {class:'studio-plan-steps'}, rows.map(row => row.line)), note);
      card.run = go;
      return card;
    };
    const addMessage = (role, text, action, plan, context = {}) => {
      const item = bubble(role, text);
      if (context.cluster_review) {
        const review = context.cluster_review;
        item.append(node('p',{class:'muted studio-cluster-evidence',text:'Cluster evidence · ' + review.clusters + ' groups · ' + review.markers + ' markers' +
          (review.images ? ' · ' + review.images + ' images' : ' · measurements only')}));
      }
      if (action) item.append(actionCard(action, context));
      if (plan && plan.length) item.append(planCard(plan, context));
      if (context.investigation?.length) {
        const records = context.investigation, cells = records.flatMap(record => record.data?.cells || []);
        const evidence = node('details', {class:'studio-options studio-investigation'}, [node('summary', {text:
          cells.length ? 'Evidence · ' + cells.length + ' cell inspections · ' + new Set(cells.map(cell => cell.roi)).size + ' ROIs' : 'Evidence · ' + records.length + (records.length === 1 ? ' check' : ' checks')})]);
        evidence.append(node('p', {class:'muted',text:cells.length ? 'Samples may miss rare or mixed cells. Display brightness is not marker positivity.' : 'Cluster summaries can hide mixed cells. Display brightness is not marker positivity.'}));
        records.forEach(record => {
          const data = record.data;
          if (record.error) { evidence.append(node('p', {class:'hint',text:record.error})); return; }
          if (data?.cells) {
            evidence.append(node('p', {class:'muted',text:(data.sample_size || 0) + ' inspected' +
              (data.matched_cells != null ? ' from ' + data.matched_cells.toLocaleString() + ' matching cells' : '') +
              (record.images_available_to_model ? ' · images sent to the model' : ' · measurements only for this model')}));
            data.cells.forEach(cell => {
              const row = node('details', {}, [node('summary', {text:'Cell ' + (cell.cell_id ?? cell.local_label ?? cell.row) + ' · ' + cell.roi})]);
              if (cell.crop?.image && cell.local_label != null) row.append(node('button', {class:'btn small',text:'Show cell',onclick:async () => {
                try { await perform({name:'zoom_to_cell',args:{cell:cell.local_label,image:cell.crop.image}}, {stay:true}); }
                catch (error) { showResult({message:error.message}); }
              }}));
              if (cell.image_note) row.append(node('p',{class:'muted',text:cell.image_note}));
              if (cell.crop) row.append(node('p',{class:'muted',text:cell.crop.locator}));
              row.append(node('table',{class:'studio-counts'}, Object.entries(cell.measurements || {}).map(([name,value]) =>
                node('tr',{},[node('td',{text:name}),node('td',{text:value == null ? '—' : String(value)})]))));
              evidence.append(row);
            });
          } else if (data?.summaries) {
            evidence.append(node('p',{class:'muted',text:'Whole-group summaries: ' + data.summaries.map(value=>value.column).join(', ')}));
          } else if (record.tool === 'cluster_profiles' && data?.profile) {
            evidence.append(node('p',{class:'muted',text:'Read ' + data.clusters.length + ' cluster profiles · ' + data.profile.columns.length + ' markers · ' + data.profile.note}));
          } else if (record.tool === 'cluster_reference' && data) {
            evidence.append(node('p',{class:'muted',text:'Compared with ' + data.labelled.toLocaleString() + ' labelled cells in ' + data.reference_column + '.'}));
          } else if (record.tool === 'read_settings' && data) {
            evidence.append(node('p',{class:'muted',text:'Read saved ' + data.method.replaceAll('_',' ') + ' settings.'}));
          } else if (data?.columns) evidence.append(node('p',{class:'muted',text:'Read ' + data.rows.toLocaleString() + ' cells and the available columns and images.'}));
        });
        (context.evidence_images || []).forEach((url,index) => evidence.append(node('img',{src:url,alt:'Inspected cell marker tiles ' + (index+1),style:'width:100%;height:auto;margin-top:8px'})));
        item.append(evidence);
      }
      if (context.timings?.model_calls) {
        const timing = context.timings, seconds = value => ((value || 0)/1000).toFixed(1) + ' s';
        item.append(node('span', {class:'muted',text:seconds(timing.total_ms),title:
          'Studio context: ' + seconds(timing.context_ms) + '\nModel setup: ' + seconds(timing.model_setup_ms) +
          '\nCollecting evidence: ' + seconds(timing.evidence_ms) + '\nModel responses: ' + seconds(timing.model_ms) +
          (timing.first_token_ms != null ? '\nModel first text: ' + seconds(timing.first_token_ms) : '') + '\nModel calls: ' + timing.model_calls}));
      }
      thread.append(item); thread.scrollTop = thread.scrollHeight;
      return item;
    };
    const drawMemories = list => {
      const add = node('input', {type:'text', maxlength:'1000', placeholder:'Something the assistant should always know', 'aria-label':'New memory'});
      const keep = node('button', {class:'btn small', text:'Remember', onclick:async () => {
        try { drawMemories((await post(route('/memory'), {content:add.value})).memories); } catch (error) { addMessage('note', error.message); }
      }});
      memoryBox.replaceChildren(node('summary', {text:'Memory (' + list.length + ')'}),
        ...list.map(item => node('div', {class:'studio-folder'}, [node('span', {class:'grow', text:item.content, title:item.status}),
          node('button', {class:'btn small ghost', text:'Forget', onclick:async () => {
            try { drawMemories((await post(route('/memory'), {forget:item.id})).memories); } catch (error) { addMessage('note', error.message); }
          }})])),
        node('div', {class:'row'}, [add, keep]),
        node('p', {class:'muted', text:'Kept on this computer; classic SPACEMAP’s assistant uses the same memory.'}));
    };
    const loadChat = async () => {
      let found;
      try { found = await get(route('/chat') + '?conversation=' + encodeURIComponent(chatState.conversation)); }
      catch (error) { modelLine.replaceChildren(node('span', {class:'hint', text:error.message})); return; }
      const model = found.model; chatState.model = model;
      modelLine.replaceChildren(chip(!model.enabled ? 'Common requests' : model.provider === 'online' ? 'Online' : 'Local', model.enabled ? 'ok' : ''),
        node('span', {class:'muted grow', text:model.enabled ? (model.model || 'no model chosen') + (model.provider === 'local' && !model.running ? ' · starts when you ask' : '') :
          'Set up a model for any wording'}),
        node('button', {class:'btn small ghost', text:'New chat', onclick:() => { chatState.conversation = ''; remember('chat', ''); thread.replaceChildren(); greet(); }}),
        setupButton);
      thread.replaceChildren();
      found.messages.forEach(item => {
        const shown = addMessage(item.role, item.content, item.role === 'assistant' ? item.action : null, item.role === 'assistant' ? item.plan : null, {...item,conversation:chatState.conversation});
        if (item.images) shown.append(node('p', {class:'muted', text:'(' + item.images + ' image' + (item.images === 1 ? '' : 's') + ', not kept)'}));
      });
      if (!found.messages.length) greet();
      drawMemories(found.memories);
    };
    const greet = () => addMessage('assistant', 'Ask about what you see, or ask me to do something here: “show markers CD3 and CD8”, ' +
      '“count cells”, “open sample 1”, “label these clusters”. Explicit requests run here; suggestions fill the name boxes for review. ' +
      'You can ask to change or clear names. Progress and saved results appear in this conversation.');
    const send = async (text, evidence = null, mode = 'auto', clusterReview = null) => {
      text = (text || '').trim();
      if (!text || chatState.busy) return;
      showTab('assistant'); await chatState.loading;  // the thread is drawn before the new message joins it
      if (chatState.busy) { if (evidence) throw Error('Wait for the Assistant’s current reply, then ask again.'); return; }
      chatState.busy = true; sendButton.disabled = true;
      const controller = new AbortController(); chatState.abort = controller; stopChat.hidden = false;
      // What goes along: what the viewer shows (always, as words), a picture of it (when ticked and the model sees
      // images), and the pictures attached.
      let view = null, pictures = evidence ? evidence.images.slice() : attached.splice(0);
      if (clusterReview?.image && chatState.model?.images) pictures.unshift(clusterReview.image);
      if (clusterReview && !chatState.model?.images) addMessage('note','This model receives the cluster measurements. An image-capable model also receives the heatmap and cell tiles.');
      try { view = viewSummary(); } catch (_) { view = null; }
      if (evidence) view = {...view, cell_inspection:evidence.context};
      if (!evidence && withView.checked && chatState.model?.images && view?.image) { try { pictures = [captureView(), ...pictures]; } catch (_) { /* no image yet */ } }
      if (evidence && !chatState.model?.images) {
        pictures=[];view.cell_inspection={...view.cell_inspection,images_available_to_model:false};
        addMessage('note','This model cannot read images; the cell review uses measurements and phenotype context. Choose an image-capable model in AI setup to include marker tiles.');
      }
      pictures = pictures.slice(0,4);
      drawAttached();
      const mine = addMessage('user', text); chatInput.value = '';
      if (pictures.length) mine.append(node('div', {class:'studio-attachments'}, pictures.map((url, index) => node('span', {class:'studio-attachment'},
        [node('img', {src:url, alt:'Sent image ' + (index + 1)})]))));
      const waiting = addMessage('note', pictures.length ? 'Looking…' : 'Thinking…');
      const startedAt = performance.now(); let waitingText = pictures.length ? 'Looking…' : 'Thinking…';
      const waitingTimer = setInterval(() => { if (waiting.isConnected) waiting.querySelector('p').textContent = waitingText + ' · ' + Math.floor((performance.now()-startedAt)/1000) + ' s'; }, 1000);
      let live = null, written = '';
      try {
        let page = null; try { page = pageContext(); } catch (_) { page = null; }
        const body = {text, conversation:chatState.conversation, step:selected || '', view, page, mode, images:pictures.length ? pictures.slice(0,4) : undefined,
          ...(clusterReview ? {cluster_review:{job_id:clusterReview.job_id,clusters:clusterReview.clusters}} : {})};
        // The answer shows while it is written; an action or plan (JSON) is only shown once it is checked.
        const found = stream ? await stream(route('/chat'), {...body, stream:true}, event => {
          if (event.status) {
            written = ''; live?.remove(); live = null;
            if (!waiting.isConnected) thread.append(waiting);
            waitingText = event.status; waiting.querySelector('p').textContent = waitingText; thread.scrollTop = thread.scrollHeight; return;
          }
          if (!event.text) return;
          written += event.text;
          const shown = written.trimStart(), text = /^[{`]/.test(shown) ? 'Preparing an action…' : shown;
          if (!live) { waiting.remove(); live = addMessage('assistant', text); } else live.querySelector('p').textContent = text;
          thread.scrollTop = thread.scrollHeight;
        }, {signal:controller.signal}) : await post(route('/chat'), body);
        if (controller.signal.aborted) return;
        stopChat.hidden = true; chatState.abort = null;
        chatState.conversation = found.conversation; remember('chat', found.conversation);
        waiting.remove(); live?.remove();
        const said = addMessage('assistant', found.say, evidence ? null : found.action, evidence ? null : found.plan, found);
        if (!evidence) {
          for (const command of (found.action ? [found.action] : found.plan || [])) if (command.name === 'name_clusters')
            window.dispatchEvent(new CustomEvent('studio-cluster-names', {detail:{...command.args,phase:found.runs_now ? 'saving' : 'draft'}}));
          if ((found.action || found.plan?.length) && found.runs_now) await said.querySelector('.studio-action').run();
        }
        (found.remember || []).forEach(content => thread.append(node('div', {class:'studio-bubble note'}, [node('span', {text:'Remember “' + content + '”? '}),
          node('button', {class:'btn small', text:'Remember', onclick:async event => {
            try { drawMemories((await post(route('/memory'), {content})).memories); event.target.closest('.studio-bubble').remove(); } catch (error) { addMessage('note', error.message); }
          }})])));
      } catch (error) { waiting.remove(); live?.remove(); addMessage('note', error.name === 'AbortError' ? 'Stopped. No new labels were applied.' : error.message); }
      finally { clearInterval(waitingTimer); waiting.remove(); chatState.busy = false; chatState.abort = null; stopChat.hidden = true; sendButton.disabled = false; thread.scrollTop = thread.scrollHeight; }
    };
    [['What am I seeing?', 'What am I seeing in this view?'], ['What next?', 'What should I do next in this step?'],
      ['How do I…', 'How do I ']].forEach(([label, text]) => quick.append(node('button', {class:'btn small ghost', text:label, onclick:() => {
        if (text.endsWith(' ')) { chatInput.value = text; chatInput.focus(); } else send(text);
      }})));
    sendButton.addEventListener('click', () => send(chatInput.value));
    chatInput.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(chatInput.value); } });
    let chatLoaded = false;
    onAssistantShown = () => { if (!chatLoaded) { chatLoaded = true; chatState.loading = loadChat(); } };

    // ------------------------------------------------------------ the step frame
    // Every step: a head (name, (?), classic link), Items from saved files, the
    // step's own tools, a run bar; its outputs go to the right-hand Outputs tab.
    let stepRequest = 0;
    const chooseStep = async id => {
      const step = context?.steps.find(item => item.id === id); if (!step) return;
      const request = ++stepRequest;
      if (tools.contains(document.activeElement)) document.activeElement.blur();
      if (frame?.annotationSettingsPending?.()) {
        try { await frame.saveAnnotationSettings(); } catch (_) { return; }
        if (request !== stepRequest) return;
      }
      frame?.clearAnnotationAutosave?.();
      frame?.thresholdBenchmark?.dispose?.(); hooks.clearThresholdPreview?.();
      frame?.tlsReview?.dispose?.();
      selected = id; selectionTools = null;
      const url = new URL(location.href); url.searchParams.set('step', id); history.replaceState(null, '', url);
      steps.querySelectorAll('button').forEach(button => button.setAttribute('aria-current', button.dataset.step === id ? 'step' : 'false'));
      const link = node('a', {class:'btn small', text:'Open Tab ' + id.replace('tab', ''), href:step.classic, target:'spacemap-classic',
        rel:'noopener', title:'Open this step in classic SPACEMAP; settings are shared'});
      const body = node('div', {class:'studio-step-body'});
      const bar = node('div', {class:'studio-runbar', hidden:true});
      tools.replaceChildren(node('div', {class:'studio-step-head'}, [node('h2', {text:step.name}), help(step.description), link]), body, bar, ...tools.querySelectorAll(':scope > .studio-resize-x'));
      showPreview(null); showRaster('classes', null); clearFigures(); if (setMarkerRoles && id !== 'tab5') setMarkerRoles(null);
      if (dropCells) dropCells(TRAINING_LAYER);  // a tile being corrected is left behind with its step
      rememberOutputs(); stepOutputs.replaceChildren();
      frame = {id, step, body, bar, runBar:(rows, note) => {
        bar.replaceChildren(...[...rows, note ? node('p', {class:'muted', text:note}) : null].filter(Boolean));
        bar.hidden = !rows.length;
      }};
      if (!['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tagging', 'tab6', 'tab7', 'tab8', 'tab9'].includes(id)) workList(frame);
      (TOOLS[id] || classicOnly)(frame);
      if (FOLDER_STEPS.has(id)) folderSection(frame);
      menu.hidden = true; search.setAttribute('aria-expanded', 'false');
    };

    // The assistant changed a step's saved settings: that step, when shown, is drawn again from them.
    window.addEventListener('studio-settings-changed', event => {
      const step = {batch_correction:'tab6', neighbourhoods:'tab8', interactions:'tab8'}[event.detail?.method] || 'tab7';
      if (selected === step && tools.isConnected) chooseStep(step);
    });

    const workList = current => {
      const count = chip('…'), again = node('button', {class:'btn small ghost', text:'↻', title:'Read the list again', 'aria-label':'Refresh list'});
      const {box, body} = section(ITEMS[current.id], {cls:'studio-work-list', actions:[count, again],
        tip:'States come from saved files and jobs. An output still needs your quality review. Click a name to open its image.'});
      current.body.append(box);
      const read = async () => {
        body.replaceChildren(node('p', {class:'muted', text:'Checking saved files…'}));
        try {
          const value = await get(route('/step/' + current.id));
          if (!body.isConnected) return;
          const done = value.items.filter(item => item.state === 'done').length;
          const partly = value.items.filter(item => item.state === 'partial').length;
          count.textContent = value.items.length ? done + ' / ' + value.items.length + (partly ? ' · ' + partly + ' partly' : '') : '0';
          count.className = 'chip' + (partly ? ' warn' : done ? ' ok' : '');
          count.title = done + ' of ' + value.items.length + ' done' + (partly ? ', ' + partly + ' partly done' : '');
          body.replaceChildren();
          if (value.last_run && ['failed', 'error', 'cancelled', 'interrupted'].includes(value.last_run.status))
            body.append(node('p', {class:'hint warn', text:'Last run: ' + value.last_run.status + '. ' + value.last_run.message}));
          const filter = node('input', {type:'search', placeholder:'Filter', 'aria-label':'Filter work list'});
          const rows = node('div', {class:'studio-work-rows'});
          if (value.items.length > 8) body.append(filter);
          body.append(rows);
          const draw = () => {
            rows.replaceChildren();
            const found = value.items.filter(item => item.name.toLowerCase().includes(filter.value.toLowerCase()));
            found.slice(0, 200).forEach(item => {
              const name = item.image ? node('button', {type:'button', class:'studio-item-name', text:item.name, title:'Open ' + item.image.name,
                onclick:() => chooseImage(item.image)}) : node('span', {class:'studio-item-name', text:item.name});
              rows.append(node('div', {class:'studio-work-row', 'data-state':item.state}, [name, chip(item.state_label, STATE_KIND[item.state] || '')]));
            });
            if (!found.length) rows.append(node('p', {class:'muted', text:'Nothing found for this step yet.'}));
            if (found.length > 200 || value.truncated) rows.append(node('p', {class:'muted', text:'Showing the first 200. Narrow the filter.'}));
          };
          filter.addEventListener('input', draw); draw();
          showOutputs(current, value);
        } catch (error) { body.replaceChildren(node('p', {class:'hint', text:error.message})); }
      };
      again.addEventListener('click', read); read();
      current.refreshItems = read;
    };

    // Where a step reads and writes, changed in the step: the folder in use, and a new one beside it.
    const FOLDER_STEPS = new Set(['tab8', 'tab9']);  // earlier steps show their folders at the top
    const folderSection = current => {
      const {box, body} = section('Folders', {tip:'Where this step reads from and writes to, shared with classic SPACEMAP. Choose ' +
        'another of the project\u2019s folders, or type a new one (inside the project) on the right; a new output folder is made ' +
        'when the step first writes there.'});
      current.body.append(box);
      const note = node('p', {class:'muted', role:'status'});
      const draw = found => {
        const rows = found.folders.map(row => {
          const role = chip(row.role === 'in' ? 'Reads from' : 'Writes to', row.role === 'in' ? '' : 'ok');
          if (!row.editable) return node('div', {class:'row field studio-folder'}, [node('label', {class:'lbl', text:row.label}), role,
            node('span', {class:'studio-file grow', text:row.folder, title:row.folder}), help(row.note)]);
          const known = found.choices.includes(row.folder) ? found.choices : [row.folder, ...found.choices];
          const pick = choice(known.map(item => [item, item]), row.folder, row.label + ' folder');
          const other = node('input', {type:'text', placeholder:'or a new folder', maxlength:'200', 'aria-label':'New ' + row.label.toLowerCase() + ' folder'});
          const save = async value => {
            try {
              draw(await post(route('/folders/' + current.id), {label:row.label, folder:value}));
              note.textContent = row.label + ': ' + value + '. Saved; classic SPACEMAP uses the same folder.';
              current.refreshItems?.();
            } catch (error) { note.textContent = error.message; }
          };
          pick.addEventListener('change', () => save(pick.value));
          other.addEventListener('change', () => { if (other.value.trim()) save(other.value.trim()); });
          return node('div', {class:'row field studio-folder'}, [node('label', {class:'lbl', text:row.label}), role,
            node('div', {class:'inline grow'}, [pick, other]), row.exists ? null : chip('New', 'warn', 'Made when the step first writes there')]);
        });
        fill(body, ...rows, note);
      };
      get(route('/folders/' + current.id)).then(draw).catch(error => fill(body, node('p', {class:'hint', text:error.message})));
    };

    const showOutputs = (current, value) => {
      if (frame !== current) return;
      rememberOutputs(); stepOutputs.replaceChildren();
      if (current.id === 'tab7') {
        const methods = Array.from(new Set(value.items.flatMap(item => item.methods || []))).sort();
        if (methods.length) {
          const {box, body} = section('Phenotype columns', {actions:[chip(String(methods.length))],
            tip:'Result columns of the cell table with calls for these ROIs. Colour the cells by one in the Phenotyping step.'});
          body.append(node('div', {class:'studio-chips'}, methods.map(name => chip(name))));
          stepOutputs.append(box);
        }
      }
      const kinds = {};
      value.outputs.forEach(item => (kinds[item.kind] ||= []).push(item.name));
      if (value.outputs.length) stepOutputs.append(node('p', {class:'hint', text:'Saved files for this step. Use Layers to show or hide results on the image.'}));
      Object.entries(kinds).forEach(([kind, names]) => {
        const key = current.id + ':' + kind;
        const box = node('details', {class:'layer-group studio-output-group', 'data-output-kind':kind, 'data-output-key':key});
        box.open = !!outputGroups.get(key);
        box.addEventListener('toggle', () => { if (box.isConnected) outputGroups.set(key, box.open); });
        const body = node('div', {class:'studio-output-files'});
        const row = name => node('div', {class:'studio-out-row', text:name, title:name});
        const count = expandedOutputs.has(key) ? names.length : 12;
        names.slice(0, count).forEach(name => body.append(row(name)));
        if (names.length > count) body.append(node('button', {class:'btn small', text:'Show all ' + names.length,
          onclick:event => { expandedOutputs.add(key); event.currentTarget.remove(); names.slice(count).forEach(name => body.append(row(name))); }}));
        box.append(node('summary', {}, [node('span', {class:'name', text:kind}), chip(names.length.toLocaleString())]), body);
        stepOutputs.append(box);
      });
      if (!stepOutputs.children.length) {
        const {box, body} = section('Outputs'); body.append(node('p', {class:'muted', text:'Nothing saved by this step yet.'}));
        stepOutputs.append(box);
      }
    };

    const classicOnly = current => {
      const {box, body} = section('Tools', {tip:'Studio controls for this step are being built (see the checklist). ' +
        'Until then it runs in classic SPACEMAP, with the same settings and files.'});
      body.append(node('div', {class:'row'}, [chip('Runs in classic for now', 'warn'),
        node('a', {class:'btn small', text:'Open Tab ' + current.id.replace('tab', ''), href:current.step.classic, target:'spacemap-classic', rel:'noopener'})]));
      current.body.append(box);
    };

    const TOOLS = {};
    // ------------------------------------------------------------ Project
    const DOT = {done:'done', partial:'partial', needs_review:'partial', running:'partial'};
    // The model library: its folder, this project's models (Share) and the library's (Import), by kind.
    const modelLibrary = async holder => {
      const note = node('p', {class:'muted', role:'status'});
      let data;
      const describe = item => {
        const d = item.details || {}, bits = [];
        if (d.artifact) bits.push(d.artifact);
        if (d.markers) bits.push((Array.isArray(d.markers) ? d.markers.length : Object.keys(d.markers).length) + ' markers');
        if (d.examples) bits.push(d.examples + ' examples');
        if (d.images) bits.push(d.images + ' images');
        if (Number.isFinite(d.held_out_balanced_accuracy)) bits.push('held-out accuracy ' + Math.round(100 * d.held_out_balanced_accuracy) + '%');
        if (d.marker_importance) {
          const total = Object.values(d.marker_importance).reduce((a, b) => a + b, 0) || 1;
          bits.push('relies on ' + Object.entries(d.marker_importance).sort((a, b) => b[1] - a[1]).slice(0, 3)
            .map(([name, weight]) => name + ' ' + Math.round(100 * weight / total) + '%').join(', '));
        }
        if (d.adaptable) bits.push('adapts to missing markers');
        if (d.he_backup) bits.push('H&E backup when markers are missing');
        if (item.source_project) bits.push('from ' + item.source_project);
        else if (d.imported_from?.project) bits.push('imported from ' + d.imported_from.project);
        if (item.created) bits.push(String(item.created).slice(0, 10));
        return bits.join(' · ');
      };
      const act = async (body, busy) => {
        note.textContent = busy;
        try { data = await post(route('/model_library'), body); draw(); const d = data.done;
          note.textContent = body.action === 'share' ? d.name + ' is in the library.' : body.action === 'import' ? d.name + ' is now in this project (' + data.kinds[d.kind] + ').' : 'Library folder changed.'; }
        catch (error) { note.textContent = error.message; }
      };
      const list = (items, button) => {
        if (!items.length) return [node('p', {class:'muted', text:'None yet.'})];
        return Object.keys(data.kinds).flatMap(kind => {
          const own = items.filter(item => item.kind === kind);
          return own.length ? [node('div', {class:'studio-sub', text:data.kinds[kind]}), ...own.map(item => node('div', {class:'row model-row'}, [
            node('span', {class:'grow model-name', text:item.name, title:describe(item)}), node('span', {class:'muted model-about', text:describe(item)}), button(item)]))] : [];
        });
      };
      const draw = () => {
        const folder = node('input', {type:'text', value:data.location, maxlength:'1024', class:'grow', 'aria-label':'Model library folder', disabled:data.fixed ? '' : null});
        folder.addEventListener('change', () => { if (folder.value.trim() && folder.value.trim() !== data.location) act({action:'location', path:folder.value.trim()}, 'Changing the library folder…'); });
        const shared = new Set(data.library.map(e => e.kind + '/' + e.id));
        fill(holder,
          field('Library', folder, data.fixed ? 'Set by SPACEMAP_MODEL_LIBRARY, so it cannot be changed here.' : 'The folder holding the shared models. Type a full path and press Enter (a shared lab folder works too).'),
          node('div', {class:'studio-sec-h studio-sub'}, [node('span', {text:'In this project'})]),
          ...list(data.project, item => shared.has(item.kind + '/' + item.id) ? chip('In the library', 'ok') :
            node('button', {class:'btn small', text:'Share', 'aria-label':'Share ' + item.name, onclick:() => act({action:'share', kind:item.kind, id:item.id}, 'Copying ' + item.name + ' into the library…')})),
          node('div', {class:'studio-sec-h studio-sub'}, [node('span', {text:'In the library'})]),
          ...list(data.library, item => data.project.some(own => own.kind === item.kind && own.id === item.id) ? chip('In this project', 'ok') :
            node('button', {class:'btn small', text:'Import', 'aria-label':'Import ' + item.name, onclick:() => act({action:'import', kind:item.kind, entry:item.entry}, 'Copying ' + item.name + ' into this project…')})),
          note);
      };
      holder.append(node('p', {class:'muted', text:'Reading the model library…'}));
      try { data = await get(route('/model_library')); if (holder.isConnected) draw(); }
      catch (error) { fill(holder, node('p', {class:'hint', text:error.message})); }
    };
    // The open project at a glance, the other projects to switch to, each sample's progress, and the project's tools
    // (folders, the model library). Settings that belong to a step stay in that step; classic Tab 0 keeps the rest.
    const goStep = id => document.querySelector('[data-step="' + id + '"]')?.click();
    TOOLS.tab0 = current => {
      const about = section('Project', {tip:'The project this window works in: its folder, when its settings were last saved, ' +
        'and its notes. Every step reads and writes inside it.'});
      const others = section('Projects', {tip:'The projects SPACEMAP knows (the same list as classic SPACEMAP). Click one ' +
        'to switch: Studio opens it at once, and classic SPACEMAP follows the next time its page refreshes (it saves the ' +
        'project it had open first). Nothing in a project changes by opening it.'});
      const samples = section('Samples', {tip:'Each sample in every step, from the files on disk: green done, amber partly ' +
        'done, grey not run, a dash when the step has no file for it. Click a dot or a step number to go to that step.'});
      const folders = section('Folders', {tip:'Where each step reads (in) and writes (out), and how many files are there. ' +
        'Change a folder in its step. An output folder is made when its step first runs.'});
      const library = section('Model library', {tip:'Trained models (Cellpose segmentation, artifact models, phenotyping ' +
        'reference models) shared between projects. Share copies one of this project’s models into the library; Import copies a ' +
        'library model into this project, where its step lists it. The library is one folder: keep it in your SPACEMAP folder, or ' +
        'point it at a shared lab folder so other people and computers use the same models. Nothing is moved or deleted.'});
      const note = node('p', {class:'muted', role:'status'});
      about.body.append(node('p', {class:'muted', text:'Reading the project…'}));
      current.body.append(about.box, others.box, samples.box, folders.box, library.box);
      current.runBar([]);
      const tell = text => { note.textContent = text; };
      let list = null;
      const act = async (body, doing) => {
        tell(doing);
        try {
          const answer = await post(route('/projects'), body);
          if (body.action === 'open' || body.action === 'create') { tell(''); await refresh(); return; }  // the step is drawn again
          list = answer; drawProjects();
          if (body.action === 'notes') drawAbout();
          tell(body.action === 'forget' ? body.name + ' is off the list (its folder is unchanged).' : 'Notes saved.');
        } catch (error) { tell(error.message); }
      };
      // ---- the open project
      let status = null;
      const drawAbout = () => {
        if (!list?.open) {
          fill(about.body, node('p', {class:'hint', text:'No project is open. Choose one below, open its folder, or make a new one.'}));
          return;
        }
        const project = status?.project, notes = list.notes || {};
        const text = (key, label, long) => {
          const input = node(long ? 'textarea' : 'input', {type:long ? null : 'text', rows:long ? '2' : null, class:'grow',
            maxlength:long ? '4000' : '200', placeholder:label, 'aria-label':'Project ' + label.toLowerCase()});
          input.value = notes[key] || '';
          input.addEventListener('change', () => { if (input.value.trim() !== (notes[key] || '')) act({action:'notes', changes:{[key]:input.value}}, 'Saving…'); });
          return input;
        };
        const facts = [project ? 'Saved ' + (project.saved || 'never') + ' · ' + project.size_kb + ' kB' : null,
          notes.created ? 'Made ' + notes.created : null].filter(Boolean).join(' · ');
        const credit = [notes.pi, notes.institution].filter(Boolean).join(' · ');
        fill(about.body,
          node('div', {class:'row'}, [node('strong', {class:'grow studio-project-title', text:list.open.split('/').pop(), title:list.open}),
            status ? chip(status.samples.length + ' sample' + (status.samples.length === 1 ? '' : 's')) : null,
            project?.read_only ? chip('View only', 'warn') : null]),
          node('div', {class:'muted studio-value', text:list.open, title:list.open}),
          facts ? node('div', {class:'muted studio-value', text:facts}) : null,
          notes.description ? node('p', {class:'studio-project-description', text:notes.description}) : null,
          node('details', {class:'studio-options'}, [node('summary', {text:'Notes' + (credit ? ' · ' + credit : '')}),
            field('About', text('description', 'Description', true), 'A short description saved with the project.'),
            field('PI', text('pi', 'PI')), field('Institution', text('institution', 'Institution'))]));
      };
      // ---- the other projects
      const filter = node('input', {type:'search', placeholder:'Filter projects', 'aria-label':'Filter projects'});
      const rows = node('div', {class:'studio-pairs-scroll studio-projects'});
      const folder = node('input', {type:'text', maxlength:'4096', class:'grow', placeholder:'/full/path/to/the/project',
        'aria-label':'Project folder to open'});
      folder.addEventListener('keydown', event => { if (event.key === 'Enter') openFolder.click(); });
      const openFolder = node('button', {class:'btn small', text:'Open', onclick:() => folder.value.trim() ?
        act({action:'open', folder:folder.value.trim()}, 'Opening ' + folder.value.trim() + '…') : tell('Type the project folder first.')});
      const newName = node('input', {type:'text', maxlength:'80', class:'grow', placeholder:'Name (letters, digits, - and _)', 'aria-label':'New project name'});
      const newParent = node('input', {type:'text', maxlength:'4096', class:'grow', 'aria-label':'Folder for the new project'});
      const create = node('button', {class:'btn small', text:'Create', onclick:() =>
        act({action:'create', name:newName.value.trim(), parent:newParent.value.trim()}, 'Making ' + newName.value.trim() + '…')});
      const drawProjects = () => {
        const query = filter.value.trim().toLowerCase();
        const shown = list.projects.filter(item => !query || (item.name + ' ' + item.folder).toLowerCase().includes(query));
        filter.hidden = list.projects.length < 8;
        if (!newParent.value) newParent.value = list.new_parent;
        rows.replaceChildren(shown.length ? node('table', {class:'studio-counts studio-pairs'}, [
          node('tr', {}, ['Project', 'Saved', ''].map(text => node('th', {text}))),
          ...shown.map(item => node('tr', {class:item.open ? 'selected' : null, 'data-project':item.name}, [
            node('td', {}, [item.missing ? node('span', {class:'studio-item-name muted', text:item.name, title:item.name + '\n' + item.folder})
              : node('button', {type:'button', class:'studio-item-name', text:item.name, title:item.name + '\n' + item.folder, disabled:item.open || null,
                onclick:() => act({action:'open', name:item.name}, 'Opening ' + item.name + '…')})]),
            node('td', {class:'muted', text:(item.saved || '').slice(0, 10), title:item.saved ? 'Settings saved ' + item.saved : null}),
            node('td', {}, [item.open ? chip('Open', 'ok') : item.missing ? node('span', {class:'row'}, [chip('Not found', 'warn', 'The folder is gone: ' + item.folder),
              node('button', {class:'btn small ghost', text:'Remove', title:'Take it off the list (nothing is deleted)', 'aria-label':'Remove ' + item.name + ' from the list',
                onclick:() => act({action:'forget', name:item.name}, 'Taking ' + item.name + ' off the list…')})]) : null])]))])
          : node('p', {class:'muted', text:list.projects.length ? 'No project matches.' : 'No projects yet.'}));
      };
      filter.addEventListener('input', () => drawProjects());
      others.body.append(filter, rows,
        field('Folder', node('div', {class:'inline grow'}, [folder, openFolder]), 'Open a project that is not in the list (a colleague’s, say) from its folder, the one holding <name>.json. It is added to the list.'),
        node('details', {class:'studio-options'}, [node('summary', {text:'New project'}),
          field('Name', newName, 'Letters, digits, - and _ only. The project gets the usual data and outputs folders, and opens at once.'),
          field('In', node('div', {class:'inline grow'}, [newParent, create]), 'The folder to make it in (a "project" folder is made inside it unless it is one).')]),
        note);
      // ---- samples and folders
      const sampleFilter = node('input', {type:'search', placeholder:'Filter samples', 'aria-label':'Filter samples'});
      const table = node('div', {class:'studio-matrix'});
      const drawSamples = () => {
        const query = sampleFilter.value.trim().toLowerCase(), steps = status.steps;
        const shown = status.samples.filter(item => !query || item.name.toLowerCase().includes(query));
        const grid = 'minmax(0, 1fr) repeat(' + steps.length + ', 22px)';
        const line = (cls, cells) => { const row = node('div', {class:'studio-matrix-row' + cls}, cells); row.style.gridTemplateColumns = grid; return row; };
        table.replaceChildren(line(' head', [node('span', {class:'grow', text:shown.length + ' of ' + status.samples.length}),
            ...steps.map((step, index) => node('button', {type:'button', class:'studio-matrix-step', text:String(index + 1), title:'Go to ' + step.name,
              onclick:() => goStep(step.id)}))]),
          ...shown.slice(0, 400).map(item => line('', [node('span', {class:'grow', text:item.name, title:item.name}),
            ...steps.map(step => {
              const cell = item.steps[step.id];
              return cell ? node('button', {type:'button', class:'studio-matrix-cell', title:step.name + ': ' + cell.label + ' (click to go there)',
                onclick:() => goStep(step.id)}, [node('i', {class:'studio-dot', 'data-state':DOT[cell.state] || 'none'})])
                : node('span', {class:'muted studio-matrix-step', text:'–', title:step.name + ': no file for this sample'});
            })])));
      };
      sampleFilter.addEventListener('input', drawSamples);
      const drawFolders = () => {
        const missing = status.folders.filter(item => !item.exists && item.kind === 'in').length;
        fill(folders.body, node('details', {class:'studio-options'}, [
          node('summary', {text:status.folders.length + ' folders' + (missing ? ' · ' + missing + ' input folder' + (missing === 1 ? '' : 's') + ' missing' : '')}),
          ...status.folders.map(item => node('div', {class:'studio-folder'}, [
            node('button', {type:'button', class:'studio-step-number', text:String(status.steps.findIndex(step => step.id === item.step) + 1), title:'Go to ' + item.step_name,
              onclick:() => goStep(item.step)}),
            node('span', {class:'grow', text:item.label + (item.kind === 'in' ? ' (in)' : ' (out)'), title:item.path}),
            item.exists ? node('span', {class:'muted', text:item.files + (item.more ? '+' : '') + ' file' + (item.files === 1 ? '' : 's')})
              : chip('Missing', item.kind === 'in' ? 'warn' : '', item.path)]))]));
      };
      (async () => {
        try { list = await get(route('/projects')); }
        catch (error) { fill(about.body, node('p', {class:'hint', text:error.message})); return; }
        if (!about.box.isConnected) return;
        drawProjects(); drawAbout();
        if (!list.open) { samples.box.hidden = folders.box.hidden = library.box.hidden = true; return; }
        const libraryBody = node('div');
        fill(library.body, node('details', {class:'studio-options'}, [node('summary', {text:'Share or import trained models'}), libraryBody]));
        modelLibrary(libraryBody);
        samples.body.append(node('p', {class:'muted', text:'Reading each step…'}));
        try { status = await get(route('/project_status')); }
        catch (error) { fill(samples.body, node('p', {class:'hint', text:error.message})); return; }
        if (!about.box.isConnected || !status.project) return;
        drawAbout();
        fill(samples.body, sampleFilter, table); drawSamples(); drawFolders();
      })();
    };

    // ------------------------------------------------------------ Spatial analysis
    // Cell-cell interactions (Tab 8's permutation test) as a job; its z-scores as a clickable heatmap beside the image.
    const INTERACTION_KEYS = ['phenotype_col', 'batch_col', 'roi_col', 'radius_um', 'pixel_um', 'n_permutations', 'min_cells'];
    const interactionsPanel = async (body, current) => {
      let saved;
      try { saved = await get(route('/interaction_settings')); } catch (error) { body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!body.isConnected) return;
      const values = JSON.parse(JSON.stringify(saved.values));
      if (!values.phenotype_col && saved.phenotypes.length) values.phenotype_col = saved.phenotypes[0];
      const changes = () => Object.fromEntries(INTERACTION_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const pick = (key, options, label) => { const input = choice(options, values[key], label); input.addEventListener('change', () => { values[key] = input.value; }); return input; };
      const number = (key, label, step, min, max) => {
        const input = node('input', {type:'number', step, min:String(min), max:String(max), value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const n = Number(input.value); if (Number.isFinite(n) && n >= min && n <= max) values[key] = step === '1' ? Math.round(n) : n; else input.value = String(values[key]); });
        return input;
      };
      const none = [['', 'None']], result = node('div'), status = node('p', {class:'muted', role:'status'});
      const draw = found => {
        const r = found.result, note = node('p', {class:'muted', role:'status'});
        const show = async groups => {
          if (!(await showCellGroups(r.column, groups, r.profile.rows))) { note.textContent = 'Open an image whose cells are in this table to see them.'; return; }
          note.textContent = groups.length ? 'Showing ' + groups.join(' and ') + ' on the image.' : 'Showing every phenotype.';
        };
        result.replaceChildren(node('div', {class:'row wrap'}, [chip(r.phenotypes.length + ' phenotypes'), chip(r.meta.radius_um + ' µm'),
            chip(r.meta.n_regions_used + ' regions'), chip(r.meta.n_permutations + ' shuffles')]),
          node('p', {class:'lbl', text:'Strongest pairs (click to see both on the image)'}),
          node('div', {class:'studio-measures'}, r.pairs.slice(0, 12).map(pair => node('button', {type:'button', class:'studio-measure',
            title:pair.observed.toLocaleString() + ' neighbouring pairs; ' + pair.expected.toLocaleString() + ' expected by chance · p ' + pair.p.toPrecision(2),
            onclick:() => show(pair.a === pair.b ? [pair.a] : [pair.a, pair.b])}, [
            node('span', {class:'grow', text:pair.a + ' – ' + pair.b}),
            node('span', {class:'mono' + (pair.z < 0 ? ' warn-text' : ''), text:(pair.z > 0 ? '+' : '') + pair.z.toFixed(1)})]))), note);
        heatmapFigure({key:'interactions', title:'Cell–cell interactions (z-score)', profile:r.profile, onPick:show,
          tip:'Each square: how much more (warm) or less (blue) often the row and column phenotypes are neighbours within ' + r.meta.radius_um +
            ' µm than by chance. Click rows to show those phenotypes on the image (Shift or Ctrl for several).'});
      };
      const latest = async () => {
        const jobs = (await get(route('/jobs'))).jobs || [];
        const going = jobs.find(job => job.kind === 'cell_interactions' && ['starting', 'running'].includes(job.status));
        const done = jobs.find(job => job.kind === 'cell_interactions' && job.status === 'done' && job.result?.profile);
        return {going, done};
      };
      const follow = async () => {
        for (;;) {
          if (!body.isConnected) return;
          const {going, done} = await latest();
          if (going) { status.textContent = going.message || 'Testing…'; await new Promise(resolve => setTimeout(resolve, 2500)); continue; }
          if (done) { draw(done); status.textContent = ''; } else result.replaceChildren(node('p', {class:'muted', text:'No test yet.'}));
          return;
        }
      };
      // Run by the step's bottom bar: the images named, or every image (none named).
      const test = async (images, said) => {
        if (Object.keys(changes()).length) saved = {...saved, ...(await post(route('/interaction_settings'), {changes:changes(), revision:saved.revision}))};
        const answer = await runCommand('run_cell_interactions', images.length ? {images} : {}, {confirmed:true});
        showResult(answer); said.textContent = answer.message;
        if (answer.status === 'done') { await refreshJobs(); await follow(); }
      };
      body.append(
        field('Phenotypes', pick('phenotype_col', saved.phenotypes.map(name => [name, name]), 'Interaction phenotype column'), 'The phenotype column whose pairs are tested.'),
        field('Radius', number('radius_um', 'Radius in µm', '1', 1, 300), 'Two cells closer than this (µm, centre to centre) are neighbours.'),
        field('Pixel size', number('pixel_um', 'Pixel size in µm', 'any', 0.01, 100), 'Micrometres per pixel. Use the image calibration.'),
        field('Shuffles', number('n_permutations', 'Shuffles', '1', 50, 5000), 'How many times the phenotypes are shuffled to see what chance gives (default 500).'),
        field('By slide', pick('batch_col', none.concat(saved.columns.map(name => [name, name])), 'Interaction slide column'), 'Pairs and shuffles stay within each slide.'),
        field('By region', pick('roi_col', none.concat(saved.columns.map(name => [name, name])), 'Interaction region column'), 'And within each tissue region, when there is a column for it.'),
        field('Smallest', number('min_cells', 'Smallest region for interactions', '1', 5, 5000), 'Regions with fewer cells are left out (default 20).'),
        status, result);
      follow();
      return test;
    };
    // What classic Tab 8 saved: triad tables (a type drawn on the image as a density map) and feature screening runs.
    const tab8SavedPanel = async (body, run) => {
      let found;
      body.replaceChildren(node('p', {class:'muted', text:'Reading the saved results…'}));
      try { found = await get(route('/tab8_results') + (run ? '?run=' + encodeURIComponent(run) : '')); }
      catch (error) { body.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      const note = node('p', {class:'muted', role:'status'});
      const triads = found.triads.length ? found.triads.map(table => node('details', {class:'studio-options', open:found.triads.length === 1 || undefined}, [
        node('summary', {text:table.name.replace('Triads / ', '') + ' · ' + table.triads.toLocaleString() + ' triads'}),
        node('div', {class:'studio-measures'}, table.types.slice(0, 15).map(type => node('button', {type:'button', class:'studio-measure', title:'Draw where ' + type.label + ' is, on the open image',
          onclick:async () => {
            try { const answer = await runCommand('density', {table:table.name, column:'triad_label', phenotype:type.label, radius:40}); showResult(answer); note.textContent = answer.message; }
            catch (error) { note.textContent = error.message; }
          }}, [node('span', {class:'grow', text:type.label.replace(/::/g, ' · ')}), node('span', {class:'mono', text:type.count.toLocaleString()})])))]))
        : [node('p', {class:'muted', text:'No triads saved yet (classic Tab 8, Triads).'})];
      const screening = found.screening;
      const parts = [node('p', {class:'lbl', text:'Triads'}), ...triads, node('p', {class:'lbl', text:'Feature screening'})];
      if (!screening.runs.length) parts.push(node('p', {class:'muted', text:'No feature screening run saved yet (classic Tab 8, Feature screening).'}));
      else {
        const pickRun = choice(screening.runs.map(name => [name, name.replace('feature_screening_', '')]), screening.run, 'Screening run');
        pickRun.addEventListener('change', () => tab8SavedPanel(body, pickRun.value));
        const cell = value => value === null || value === undefined ? '' : typeof value === 'number' ? (Math.abs(value) < 0.001 && value !== 0 ? value.toExponential(1) : Number(value.toPrecision(3)).toString()) : String(value);
        const columns = Object.keys(screening.hits[0] || {});
        parts.push(field('Run', pickRun), node('div', {class:'row wrap'}, [chip(screening.tested.toLocaleString() + ' tests'),
            screening.significant !== null ? chip(screening.significant + ' with q < 0.05', screening.significant ? 'ok' : '') : null].filter(Boolean)),
          node('div', {class:'studio-samples'}, [node('table', {class:'studio-counts'}, [node('tr', {}, columns.map(name => node('th', {text:name.replace(/_/g, ' ')}))),
            ...screening.hits.map(row => node('tr', {}, columns.map(name => node('td', {text:cell(row[name])}))))])]),
          node('div', {class:'studio-thumbs'}, screening.figures.map((figure, index) => node('button', {type:'button', class:'studio-thumb', title:'Show ' + figure.name + ' beside the image',
            onclick:() => { const {box, body:inside} = section(figure.name, {}); inside.append(node('img', {src:figure.image, alt:figure.name, class:'studio-figure-preview'})); showFigure('screening-' + index, box); }},
            [node('img', {src:figure.image, alt:figure.name}), node('span', {text:figure.name})]))));
      }
      body.replaceChildren(...parts, note);
    };
    const NEIGHBOURHOOD_KEYS = ['phenotype_col', 'batch_col', 'roi_col', 'k_neighbors', 'n_neighborhoods', 'min_cells', 'normalize', 'output_col'];
    // Spatial analysis in the Studio layout: the combined table's images (tick, open, which runs cover them), the
    // methods as tabs, and the bottom bar's This image | Selected. A run on every image: tick the header box.
    const SPATIAL_PARTS = [
      {id:'neighbourhoods', name:'Neighbourhoods', tip:'Groups cells by the mix of nearby phenotypes ' +
        'within the same image and region.'},
      {id:'tls', name:'TLS candidates', tip:'Connected B/T aggregates with a maximum gap, area, density and signed boundary distances. Candidates for review, not confirmed TLS.'},
      {id:'interactions', name:'Cell–cell interactions', tip:'Tests which phenotypes are neighbours more, or less, ' +
        'often than if the phenotypes were shuffled among the same cells, within each slide and region.'},
      {id:'density', name:'Density', tip:'Cells per mm² of each group in each image (the tissue outline drawn on the image), density maps on the image, and each group by distance from an annotation.'},
      {id:'triads', name:'Triads', tip:'Three phenotypes found together (each within a distance of the others) more or less often than chance; the triangles are drawn on the image.'},
      {id:'screening', name:'Feature screening', tip:'Test spatial features against an outcome and review findings on the image.'}];
    // Spatial analysis and Visualization results beside the image and on it (spatial-panels.js), made once.
    let spatialPanels = null;
    const spatial = () => spatialPanels ||= window.createSpatialPanels({node, field, choice, help, section, chip, get, post, route, runCommand, showResult,
      heatmapFigure, showFigure, showCellGroups, filterCells:(...a) => hooks.filterCells(...a), showPreview, openTableImage, imageSettled:(...a) => hooks.imageSettled?.(...a),
      openRoi:() => (currentSpec?.layers || []).find(l => l.type === 'cells' && l.roi)?.roi || null, zoomBox:zoomAnnotationBox, refreshJobs,
      tlsEditor:(...args)=>hooks.tlsEditor(...args)});
    // The image of a combined-table image name (batch_id): opened in the viewer.
    const openTableImage = async roi => {
      let list = [];
      try { list = await loadImages(); } catch (_) { /* said below */ }
      const image = list.find(item => item.sample === roi) || list.find(item => item.name === roi || item.name.startsWith(roi + '.'));
      if (image) await chooseImage(image); else showResult({message:'No image of ' + roi + ' in this project.'});
    };
    // The combined-table image name of the image open in the viewer.
    const openTableName = rois => {
      const own = (currentSpec?.layers || []).find(l => l.type === 'cells' && rois.includes(l.roi))?.roi;
      if (own) return own;
      const name = currentSpec?.layers?.find(l => l.type === 'image')?.name || '';
      return rois.filter(roi => name === roi || name.startsWith(roi + '.') || name.startsWith(roi + '_')).sort((a, b) => b.length - a.length)[0] || null;
    };
    TOOLS.tab8 = current => {
      let part = recall('spatial-part', 'neighbourhoods');
      const picked = current.spatialPicked = current.spatialPicked || new Set();
      const openTLS = async roi => {
        if (part !== 'tls') {part='tls';remember('spatial-part',part);draw();}
        try {await current.tlsReady;await current.tlsReview?.selectImage(roi);}
        catch (error) {showResult({message:error.message});}
      };
      current.imageChanged = () => { if(part==='tls')current.tlsReview?.syncOpenImage(false); };
      current.imageReady = () => { if(part==='tls')current.tlsReview?.syncOpenImage(); };
      const list = section('Images', {tip:'The combined table’s images (every image’s cells, as Batch correction made it) with their cells. ' +
        'Click a name to open it; tick images to run on them with Selected (the header box ticks all). The chips show neighbourhood coverage and candidate counts from the latest TLS run for this table.'});
      list.body.append(node('p', {class:'muted', text:'Reading the combined table…'}));
      let overview = null;
      const drawList = () => {
        if (!list.box.isConnected) return;
        if (!overview.table) { fill(list.body, node('p', {class:'hint', text:'No combined table yet: make it in Batch correction.'})); return; }
        const rois = overview.rois.map(item => item.roi);
        const head = pickAll(rois, picked, drawList);
        const runs = Object.entries(overview.results || {});
        const rows = overview.rois.map(item => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + item.roi}); tick.checked = picked.has(item.roi);
          tick.addEventListener('change', () => { tick.checked ? picked.add(item.roi) : picked.delete(item.roi); head.sync(); });
          const covered = runs.filter(([, counts]) => counts[item.roi]).map(([column]) => column.replace(/_cn$/, ''));
          return node('tr', {'data-roi':item.roi}, [node('td', {}, [tick]),
            node('td', {}, [node('button', {type:'button', class:'studio-item-name studio-file', text:item.roi, title:'Open ' + item.roi, onclick:() => openTableImage(item.roi)})]),
            node('td', {class:'mono', text:item.cells.toLocaleString()}),
            node('td', {}, [workingChip(current, item.roi), covered.length ? chip(covered.length === 1 ? 'Neighbourhoods' : covered.length + ' neighbourhood runs', 'ok',
              'Neighbourhood columns with this image’s cells: ' + covered.join(', ')) : null,
              Object.hasOwn(overview.tls?.counts || {}, item.roi) ? node('button', {type:'button', class:'chip' + (overview.tls.measured[item.roi] ? ' ok' : ''),
                text:overview.tls.measured[item.roi] ? overview.tls.counts[item.roi] + ' TLS candidates' : 'TLS: no eligible cells',
                title:'Open '+item.roi+' and review all TLS candidates ('+overview.tls.column+')', 'aria-label':'Show TLS in '+item.roi, onclick:()=>openTLS(item.roi)}) : null,
              !covered.length && !Object.hasOwn(overview.tls?.counts || {}, item.roi) ? chip('Not run') : null].filter(Boolean))]);
        });
        fill(list.body,
          node('div', {class:'row wrap'}, [chip(overview.table, 'ok', 'The combined table'), chip(overview.cells.toLocaleString() + ' cells'),
            chip(overview.rois.length + ' images'), chip(runs.length + ' neighbourhood run' + (runs.length === 1 ? '' : 's'))]),
          node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'}, [
            node('tr', {}, [node('th', {}, [head]), node('th', {text:'Image'}), node('th', {text:'Cells'}), node('th', {text:'State'})]), ...rows])]));
      };
      const loadList = async () => {
        try { overview = await get(route('/phenotyping_overview') + '?results=1'); drawList(); }
        catch (error) { fill(list.body, node('p', {class:'hint', text:error.message})); }
      };
      followJobs(current, ['neighbourhoods', 'cell_interactions', 'tls_candidates'], async()=>{
        await loadList();await current.tlsReview?.refresh?.();
      });
      loadList();
      // The images a run is for: the open one, or the ticked ones (every image when all are ticked).
      const chosen = (which, said) => {
        const rois = (overview?.rois || []).map(item => item.roi);
        if (which === 'open') {
          const roi = openTableName(rois);
          if (!roi) { said.textContent = 'Open an image of the combined table first (click its name in the list).'; return null; }
          return [roi];
        }
        const ticked = rois.filter(roi => picked.has(roi));
        if (!ticked.length) { said.textContent = 'Tick the images first (the header box ticks all).'; return null; }
        return ticked.length === rois.length ? [] : ticked;
      };
      const actionRow = (label, act, tip, said) => {
        const run = (button, which) => async () => {
          const images = chosen(which, said);
          if (!images) return;
          button.disabled = true; said.textContent = '';
          try { await act(images, said); } catch (error) { said.textContent = error.message; }
          finally { button.disabled = false; }
        };
        const one = node('button', {class:'btn', text:'This image', title:'The image open in the viewer'});
        const some = node('button', {class:'btn primary', text:'Selected', title:'The ticked images, together (tick the header box for all)'});
        one.addEventListener('click', run(one, 'open')); some.addEventListener('click', run(some, 'picked'));
        return node('div', {class:'row'}, [node('span', {class:'lbl', text:label}), one, some, help(tip)]);
      };
      const tabs = node('div', {class:'segmented studio-stage-tabs', role:'tablist', 'aria-label':'Spatial analysis method'});
      const tip = help('');
      const holder = node('div', {class:'studio-method'});
      current.body.append(freshnessNote(['combined']), list.box, node('div', {class:'row studio-stage-row'}, [tabs, tip]), holder);
      const draw = () => {
        const chosenPart = SPATIAL_PARTS.find(item => item.id === part) || SPATIAL_PARTS[0];
        part = chosenPart.id; setTip(tip, chosenPart.tip);
        tabs.replaceChildren(...SPATIAL_PARTS.map(item => node('button', {type:'button', class:'btn small' + (item.id === part ? ' active' : ''), role:'tab',
          'aria-selected':String(item.id === part), text:item.name, onclick:() => { part = item.id; remember('spatial-part', part); draw(); }})));
        const pane = node('div', {class:'studio-method-pane'});
        current.saveAnnotationSettings?.().catch(() => {}); current.clearAnnotationAutosave?.();
        current.tlsReview?.dispose();current.tlsReview=null;
        holder.replaceChildren(pane); current.runBar([]); clearFigures(); showPreview('tls-candidates', null);
        if (part === 'neighbourhoods') neighbourhoodsPanel(current, pane, actionRow);
        else if (part === 'tls') current.tlsReady = spatial().tlsCandidates(current, pane, actionRow).catch(error => pane.append(node('p', {class:'hint', text:error.message})));
        else if (part === 'density') spatial().density(current, pane, actionRow).catch(error => pane.append(node('p', {class:'hint', text:error.message})));
        else if (part === 'triads') spatial().triads(current, pane, actionRow, tab8SavedPanel).catch(error => pane.append(node('p', {class:'hint', text:error.message})));
        else if (part === 'screening') spatial().screening(current, pane, actionRow, tab8SavedPanel).catch(error => pane.append(node('p', {class:'hint', text:error.message})));
        else if (part === 'interactions') {
          const contacts = section('Cell–cell interactions', {tip:chosenPart.tip + ' Settings are shared with classic Tab 8.'});
          pane.append(contacts.box);
          const said = node('p', {class:'muted', role:'status'});
          interactionsPanel(contacts.body, current).then(test => {
            if (!test || !pane.isConnected) return;
            current.runBar([actionRow('Test:', test, 'Tests the pairs with the settings above, in the background: on the open image, or on the ticked images together. ' +
              'The result (a heatmap beside the image) stays with the job; nothing in the table changes.', said), said]);
          });
        }
      };
      draw();
    };
    const neighbourhoodsPanel = async (current, pane, actionRow) => {
      const state = current.neighbourhoods || (current.neighbourhoods = {});
      const settings = section('Neighbourhood settings', {tip:'Group cells by the phenotype mix around them. Settings save automatically.'});
      const where = node('button', {class:'btn small', text:'Where in each image', title:'Stacked bars: the cells of each neighbourhood in each image, beside the image; a click shows them on it', onclick:async () => {
        try {
          const jobs = (await get(route('/jobs'))).jobs || [];
          const last = jobs.find(job => job.kind === 'neighbourhoods' && job.status === 'done' && job.result?.column);
          if (!last) { showResult({message:'Find neighbourhoods first.'}); return; }
          await spatial().whereFigure(last.result.column, last.result.images || [], 'where', 'Neighbourhoods in each image');
        } catch (error) { showResult({message:error.message}); }
      }});
      const results = section('Neighbourhoods found', {tip:'The neighbourhoods of the last run, with their size and main phenotypes. ' +
        'Show them on the image, name each, and Save names: the Save as column holds each cell’s neighbourhood name. Further name edits update that column. Where in each image: the neighbourhoods of each image as bars, linked to it.', actions:[where]});
      pane.append(settings.box, results.box);
      let saved;
      try { saved = await get(route('/neighbourhood_settings')); }
      catch (error) { settings.body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!pane.isConnected) return;
      if (!state.values) state.values = JSON.parse(JSON.stringify(saved.values));
      const values = state.values;
      const dirty = chip('Saved', 'ok');
      const status = node('p', {class:'muted', role:'status'});
      const changes = () => Object.fromEntries(NEIGHBOURHOOD_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); status.textContent = ''; };
      const pick = (key, options, label) => {
        const input = choice(options, values[key], label);
        input.addEventListener('change', () => { values[key] = input.value; mark(); });
        return input;
      };
      const whole = (key, label, min, max) => {
        const input = node('input', {type:'number', min:String(min), max:String(max), step:'1', value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const n = Math.round(Number(input.value)); if (n >= min && n <= max) { values[key] = n; mark(); } else input.value = String(values[key]); });
        return input;
      };
      const none = [['', 'None']];
      if (!values.phenotype_col && saved.phenotypes.length) values.phenotype_col = saved.phenotypes[0];
      const scale = node('input', {type:'checkbox', 'aria-label':'Z-score the mixes'}); scale.checked = !!values.normalize;
      scale.addEventListener('change', () => { values.normalize = scale.checked; mark(); });
      const output = node('input', {type:'text', value:values.output_col, 'aria-label':'Neighbourhood result column'});
      ['input', 'change'].forEach(event => output.addEventListener(event, () => { values.output_col = output.value.trim(); mark(); }));
      const overwrite = overwriteOption([output], 'Replace the Save as column and its _cn groups when the run succeeds. Previous names reset to the new CN groups. This replaces the whole output columns; cells outside the selected images are left blank.');
      settings.body.append(
        field('Phenotypes', pick('phenotype_col', saved.phenotypes.map(name => [name, name]), 'Phenotype column'), 'The phenotype column whose mix makes a neighbourhood.'),
        field('Neighbours', whole('k_neighbors', 'Neighbours', 3, 100), 'How many nearest cells make each cell’s neighbourhood.'),
        field('Groups', whole('n_neighborhoods', 'Neighbourhoods', 2, 50), 'How many neighbourhoods to find (default 10).'),
        field('By slide', pick('batch_col', none.concat(saved.columns.map(name => [name, name])), 'Slide column'), 'Neighbours are found within each slide.'),
        field('By region', pick('roi_col', none.concat(saved.columns.map(name => [name, name])), 'Region column'), 'Neighbours are also kept within each tissue region, when there is a column for it.'),
        field('Smallest', whole('min_cells', 'Smallest region', 5, 5000), 'Regions with fewer cells are left out (default 20).'),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Z-score'}), scale, node('span', {class:'grow'}),
          help('Scales each phenotype’s share before grouping, so rare phenotypes count.')]),
        field('Save as', output, 'The column for the names; the neighbourhood groups use the same name with _cn. Choose a new name to keep earlier results, or tick Overwrite existing results.'),
        overwrite.row);
      const write = async () => {
        const sent = {...values};
        saved = {...saved, ...(await post(route('/neighbourhood_settings'), {changes:changes(), revision:saved.revision}))};
        // Keep edits made while the request was saving; the next save picks them up.
        Object.keys(sent).forEach(key => { if (values[key] === sent[key]) values[key] = saved.values[key]; });
        mark();
      };
      const save = headerSave(settings.box, dirty, write, status, current, settings.body, () => Object.keys(changes()).length > 0, mark);
      mark();
      const follow = groupList({results, holder:settings.box, state, kind:'neighbourhoods', noun:'Name', running:'Finding neighbourhoods', none:'No neighbourhoods yet.'});
      const find = async (images, said) => {
        await save();
        const answer = await runCommand('run_neighbourhoods', {...(images.length ? {images} : {}),
          settings_revision:saved.revision, overwrite:overwrite.input.checked}, {confirmed:true});
        showResult(answer); said.textContent = answer.message;
        if (answer.status === 'done') { overwrite.input.checked = false; await refreshJobs(); follow(); }
      };
      const said = node('p', {class:'muted', role:'status'});
      settings.body.append(status);
      current.runBar([actionRow('Find:', find, 'Finds the neighbourhoods with the settings above, in the background: in the open image, or in the ticked images ' +
        'together (tick the header box for all). The groups are saved in the combined table; the other images’ cells get none. Existing results change only when Overwrite existing results is ticked.', said), said]);
      follow();
    };

    // ------------------------------------------------------------ Visualization
    // Visualization: the figure of the open (or each ticked) image, and figures of the combined table beside the image,
    // each linked to it (Heatmap, Proportions, UMAP, Method agreement).
    const VIZ_PARTS = [
      {id:'figure', name:'Figure', tip:'What the image shows, saved as a PNG with a title, legend and scale bar: the open image or each ticked one.'},
      {id:'heatmap', name:'Heatmap', tip:'Each group’s mean marker values; a click on a group shows its cells on the image.'},
      {id:'proportions', name:'Proportions', tip:'The cells of each group in each image as stacked bars; a click opens that image showing those cells.'},
      {id:'umap', name:'UMAP', tip:'A map of cells by their markers; select points to see those cells on the image.'},
      {id:'agreement', name:'Method agreement', tip:'A phenotyping method against a reference column (the F1 paper figures’ measures); a click shows the cells on the image.'}];
    TOOLS.tab9 = current => {
      let part = recall('viz-part', 'figure');
      const tabs = node('div', {class:'segmented studio-stage-tabs', role:'tablist', 'aria-label':'Visualization'});
      const tip = help('');
      const holder = node('div', {class:'studio-method'});
      current.body.append(node('div', {class:'row studio-stage-row'}, [tabs, tip]), holder);
      const draw = () => {
        const chosen = VIZ_PARTS.find(item => item.id === part) || VIZ_PARTS[0];
        part = chosen.id; setTip(tip, chosen.tip);
        tabs.replaceChildren(...VIZ_PARTS.map(item => node('button', {type:'button', class:'btn small' + (item.id === part ? ' active' : ''), role:'tab',
          'aria-selected':String(item.id === part), text:item.name, onclick:() => { part = item.id; remember('viz-part', part); draw(); }})));
        const pane = node('div', {class:'studio-method-pane'});
        holder.replaceChildren(pane); current.runBar([]); clearFigures();
        const staged = new Proxy(current, {get:(target, key) => key === 'runBar' ? (rows, note) => { if (pane.isConnected) target.runBar(rows, note); } : key === 'body' ? pane : target[key]});
        if (part === 'figure') figurePanel(staged);
        else spatial()[part](staged, pane).catch(error => pane.append(node('p', {class:'hint', text:error.message})));
      };
      draw();
    };
    const figurePanel = current => {
      const state = current.figure || (current.figure = {title:'', legend:true, scaleBar:true, full:false});
      const picked = current.figurePicked = current.figurePicked || new Set();
      const list = section('Images', {tip:'The project’s images. Click a name to open it and set up its view (markers and colours in Layers); ' +
        'tick images to save a figure of each with Selected (the header box ticks all).'});
      list.body.append(node('p', {class:'muted', text:'Reading the images…'}));
      const figure = section('Figure', {tip:'Saves what the image shows (markers, cells coloured by a column, drawn shapes) as a PNG ' +
        'figure, with a title, a legend of what the colours mean and the scale bar. Each image is drawn with its own saved view settings.'});
      const others = section('Other figures', {tip:'Heatmaps, proportions, UMAPs and the paper figures run in classic Tab 9 for now.'});
      others.body.append(node('div', {class:'row'}, [chip('Classic Tab 9', 'warn'), node('a', {class:'btn small', text:'Open Tab 9',
        href:current.step.classic, target:'spacemap-classic', rel:'noopener'})]));
      let images = [];
      const drawList = () => {
        if (!list.box.isConnected) return;
        const keys = images.map(image => image.group + ' | ' + image.name);
        const head = pickAll(keys, picked, drawList);
        const rows = images.map((image, index) => {
          const key = keys[index];
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + image.name}); tick.checked = picked.has(key);
          tick.addEventListener('change', () => { tick.checked ? picked.add(key) : picked.delete(key); head.sync(); });
          return node('tr', {'data-image':image.name}, [node('td', {}, [tick]),
            node('td', {}, [node('button', {type:'button', class:'studio-item-name studio-file', text:image.name, title:'Open ' + image.name, onclick:() => chooseImage(image)})]),
            node('td', {class:'muted', text:image.group})]);
        });
        fill(list.body, node('div', {class:'row wrap'}, [chip(images.length + ' images')]),
          images.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'}, [
            node('tr', {}, [node('th', {}, [head]), node('th', {text:'Image'}), node('th', {text:'Folder'})]), ...rows])])
            : node('p', {class:'muted', text:'No images found yet.'}));
      };
      loadImages().then(found => { images = found || []; drawList(); }).catch(error => fill(list.body, node('p', {class:'hint', text:error.message})));
      const title = node('input', {type:'text', maxlength:'200', value:state.title, placeholder:'No title', 'aria-label':'Figure title'});
      title.addEventListener('input', () => { state.title = title.value; });
      const tick = (key, label) => { const input = node('input', {type:'checkbox', 'aria-label':label}); input.checked = !!state[key];
        input.addEventListener('change', () => { state[key] = input.checked; }); return input; };
      const size = choice([['view', 'As on screen'], ['full', 'Full resolution (up to 8,192 px)']], state.full ? 'full' : 'view', 'Figure size');
      size.addEventListener('change', () => { state.full = size.value === 'full'; });
      figure.body.append(field('Title', title, 'Printed above the image. With Selected, each figure’s title also names its image.'),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Legend'}), tick('legend', 'Add the legend'), node('span', {class:'grow'}),
          help('The visible markers with their colours, and the column the cells are coloured by.')]),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Scale bar'}), tick('scaleBar', 'Add the scale bar'), node('span', {class:'grow'})]),
        field('Size', size, 'On screen is quick; full resolution reads the tiles of the visible area (each ticked image is shown whole).'));
      current.body.append(list.box, figure.box, others.box);
      const note = node('p', {class:'muted', role:'status'});
      const options = name => ({title:[state.title.trim(), name].filter(Boolean).join(' · '), legend:state.legend, scaleBar:state.scaleBar, full:state.full});
      const one = node('button', {class:'btn', text:'This image', title:'The image open in the viewer, as it is shown'});
      const some = node('button', {class:'btn primary', text:'Selected', title:'Each ticked image in turn: opened, then saved'});
      one.addEventListener('click', async () => {
        one.disabled = true; note.textContent = 'Making the figure…';
        try {
          const made = await saveFigure(options(''));
          note.textContent = made ? 'Saved a ' + made.width + ' × ' + made.height + ' px figure.' : 'Open an image first.';
        } catch (error) { note.textContent = error.message; }
        finally { one.disabled = false; }
      });
      some.addEventListener('click', async () => {
        const chosen = images.filter(image => picked.has(image.group + ' | ' + image.name));
        if (!chosen.length) { note.textContent = 'Tick the images first (the header box ticks all).'; return; }
        some.disabled = one.disabled = true;
        let done = 0;
        try {
          for (const image of chosen) {
            if (!some.isConnected) return;
            note.textContent = 'Figure ' + (done + 1) + ' of ' + chosen.length + ': opening ' + image.name + '…';
            await chooseImage(image);
            await hooks.imageSettled?.();
            const made = await saveFigure(options(image.name));
            if (made) done += 1;
          }
          note.textContent = 'Saved ' + done + ' figure' + (done === 1 ? '' : 's') + (done < chosen.length ? ' (' + (chosen.length - done) + ' could not be drawn)' : '') + '.';
        } catch (error) { note.textContent = error.message; }
        finally { some.disabled = one.disabled = false; }
      });
      current.runBar([node('div', {class:'row'}, [node('span', {class:'lbl', text:'Save figure:'}), one, some,
        help('Saves a PNG of the open image as it is shown, or of each ticked image in turn (each opened with its own saved view). The browser downloads them; nothing is written to the project.')]), note]);
    };

    // ------------------------------------------------------------ Cell tagging
    // Tab 5's standalone spatial cell tagging: each cell gets columns from the annotation it sits in. Settings are
    // Tab 5's seg_tag_* keys (classic Tab 5 shows the same) and save as they change; the tagging runs as a job.
    const TAG = {mode:'seg_tag_mode', suffix:'seg_tag_csv_suffix', file:'seg_tag_merged_file', batch:'seg_tag_batch_column',
      override:'seg_tag_geojson_override', matches:'seg_tag_geojson_matches', outSuffix:'seg_tag_output_suffix', x:'seg_tag_x_column',
      y:'seg_tag_y_column', tissue:'seg_tag_tissue_label', rules:'seg_tag_rules', beds:'seg_tag_bed_labels', edge:'seg_tag_edge_enabled',
      edgeRef:'seg_tag_edge_reference', edgeBuffer:'seg_tag_edge_buffer', edgeColumn:'seg_tag_edge_column', minBed:'seg_tag_min_cells_per_bed_target',
      clean:'seg_tag_require_clean_path', overwrite:'seg_tag_overwrite_columns', gpu:'seg_tag_use_gpu_distances',
      csvDir:'seg_tag_csv_dir', mergedDir:'seg_tag_merged_dir', outDir:'seg_tag_output_dir', geojsonDir:'seg_tag_geojson_dir'};
    const PER_ROI = 'One CSV per image', COMBINED = 'One merged CSV for several images';
    // Results older than what they were made from, said where they are used (tagging, correction, phenotyping).
    const freshnessNote = kinds => {
      const box = node('div', {class:'studio-freshness', hidden:''});
      get(route('/freshness')).then(found => {
        const lines = [];
        const list = names => names.slice(0, 6).join(', ') + (names.length > 6 ? ' and ' + (names.length - 6) + ' more' : '');
        if (kinds.includes('tagging') && found.tagging.length) lines.push('Tagged before their latest segmentation: ' + list(found.tagging) +
          '. Check whether segmentation changed before reusing these tagged tables.');
        if (kinds.includes('combined') && found.combined.length) lines.push('The combined table' + (found.table ? ' (' + found.table + ')' : '') +
          ' has changed or missing recorded inputs: ' + list(found.combined) + '. Review its saved run before reusing it.');
        if (kinds.includes('combined') && found.unverified) {
          box.append(node('details', {}, [node('summary', {text:'Active table: ' + found.table, title:found.combined_path}),
            node('p', {class:'muted', text:'This table has no recorded input provenance. Its age does not establish a link to the current images or cell tables. Select the intended unified table or use the new result after correction.'})]));
          box.hidden = false;
          if (!lines.length) box.classList.add('info');
        }
        if (!lines.length) return;
        box.append(...lines.map(text => node('p', {class:'hint', text})));
        box.hidden = false;
      }).catch(() => {});
      return box;
    };

    TOOLS.tagging = current => {
      // Cell tagging in the Studio design: the cell tables with their annotations (the Annotation step's export by
      // default), the tissue pieces, the annotations each cell is tagged with (columns named by the user), beds, a
      // preview on the image, then Tag for the open image or the ticked tables. The engine is classic Tab 5's.
      const place = section('Folders and tables', {tip:'One row per cell table (what Segmentation wrote). Each is tagged with its ' +
        'image’s annotations: by default the GeoJSON the Annotation step exported (its final names), or a file picked in the row. ' +
        'Tagged tables are new files in the output folder; the originals are never changed.'});
      const compute = section('Compute', {tip:'CPU cores and optional GPU distances for new tagging runs. Changing these settings does not change a running job.'});
      const pieces = section('Tissue pieces', {tip:'The parent annotation (usually tissue). Each separate polygon of it is one tissue piece, ' +
        'numbered by size (largest = 1). Every other column is measured inside the cell’s own piece, never across glass to another one.'});
      const marks = section('Annotations', {tip:'For each annotation, the columns each cell gets. Tick a column to add it; its name can be changed. ' +
        'Inside: 1 when the cell is in it, else 0. Piece: which polygon of it holds the cell. Size: that polygon’s area. Distance: how far the ' +
        'cell is from its edge, negative inside, positive outside, measured inside the cell’s tissue piece.'});
      const beds = section('Beds', {tip:'A bed gives each cell to the nearest of the chosen annotations, inside its own tissue piece: a cell inside ' +
        'Tumor goes to Tumor; a cell in between goes to whichever edge is closer. Annotation pieces holding fewer cells than Smallest are left out, ' +
        'so a few stray pixels cannot claim cells. The bed column holds the winner’s name; <name>_bed is 1 for its cells and <name>_bed_id ' +
        'says which of its pieces.'});
      const trial = section('Preview on the image', {tip:'Tags the open image’s cells in memory with the settings above and colours them by ' +
        'any new column: distances as a colour scale, pieces, beds and inside/outside as colours. Nothing is written. In view only is quicker on a large image.'});
      current.body.append(freshnessNote(['tagging']), place.box, compute.box, pieces.box, marks.box, beds.box, trial.box);
      const note = node('p', {class:'muted', role:'status'}), trialNote = node('p', {class:'muted', role:'status'});
      const picked = new Set();
      let data = null, previewed = null, openTable = null, inView = true;
      const load = async () => {
        try { data = await get(route('/tagging_settings')); if (place.box.isConnected) draw(); }
        catch (error) { fill(place.body, node('p', {class:'hint', text:error.message})); }
      };
      const save = async (change, said) => {
        try {
          data = await post(route('/tagging_settings'), {changes:change, revision:data.revision});
          note.textContent = said || 'Saved; classic Tab 5’s Cell tagging uses the same settings.'; draw();
          if (previewed) trialNote.textContent = 'Settings changed: press Preview again to see them on the image.';
        } catch (error) { note.textContent = error.message; await load(); }
      };
      const names = () => ({...(data.values.seg_tag_column_names || {})});
      const defaultName = (label, kind) => ({inside:label, roi:label + '_roi', roi_size:label + '_roi_size', distance:label + '_distance',
        bed:label + '_bed', bed_id:label + '_bed_id'})[kind].replace(/[^A-Za-z0-9_.]+/g, '_');
      const nameOf = (label, kind) => names()[label + '|' + kind] || defaultName(label, kind);
      const nameBox = (label, kind) => {
        const input = node('input', {type:'text', value:nameOf(label, kind), maxlength:'64', class:'grow', 'aria-label':'Column name ' + label + ' ' + kind,
          title:'The column’s name in the tagged table'});
        input.addEventListener('change', () => {
          const value = input.value.trim(), all = names();
          if (!value || value === defaultName(label, kind)) delete all[label + '|' + kind]; else all[label + '|' + kind] = value;
          save({seg_tag_column_names:all}, label + ': column named ' + (value || defaultName(label, kind)) + '.');
        });
        return input;
      };
      const flagKey = {inside:'Binary (0/1)', roi:'ROI ID', roi_size:'ROI Size', distance:'Distance'};
      const rules = () => data.rules.map(rule => ({...rule}));
      const setFlag = (label, kind, on) => {
        const all = rules(), rule = all.find(r => r['Annotation Label'] === label);
        if (!rule) return;
        rule[flagKey[kind]] = on; save({seg_tag_rules:all});
      };
      // One column: a tick, and its name when ticked.
      const columnRow = (label, kind, text, tip) => {
        const rule = data.rules.find(r => r['Annotation Label'] === label) || {};
        const on = !!rule[flagKey[kind]];
        const tick = node('input', {type:'checkbox', 'aria-label':label + ' ' + text}); tick.checked = on;
        tick.addEventListener('change', () => setFlag(label, kind, tick.checked));
        return node('div', {class:'row field studio-tag-column'}, [node('label', {class:'inline-check'}, [tick, node('span', {text})]),
          on ? nameBox(label, kind) : node('span', {class:'grow muted', text:nameOf(label, kind)}), help(tip)]);
      };
      const units = () => data.values.seg_tag_units === 'px' ? 'px' : 'µm';
      const openRow = async row => {
        if (!row.image) { note.textContent = 'No image for ' + row.source + ' in the Segmentation step’s images folder.'; return; }
        openTable = row.source;
        place.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
        place.body.querySelector('tr[data-source="' + CSS.escape(row.source) + '"]')?.classList.add('selected');
        note.textContent = 'Opening ' + row.image + '…';
        try { const answer = await runCommand('open_segmentation_image', {image:row.image}); showResult(answer); note.textContent = answer.message; }
        catch (error) { note.textContent = error.message; }
      };
      const draw = () => {
        const v = data.values, found = data.found, combined = v[TAG.mode] === COMBINED;
        // ---- tables and their annotations
        const head = pickAll(found.rows.map(row => row.source), picked, draw);
        const rows = found.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.source}); tick.checked = picked.has(row.source);
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.source) : picked.delete(row.source); head.sync(); });
          const origin = {export:'Export', imported:'Imported', folder:'By name'}[row.origin] || 'None';
          const first = row.chosen ? 'As Annotations from says' : origin + ': ' + (row.annotation || 'none');
          // This image's imported GeoJSON first, then the Annotations folder's files.
          const file = groupedChoice([['', first]], [['Imported for this image', row.imported || []],
            ['Annotations folder', found.geojson_files.map(name => [name, name])]].filter(([, items]) => items.length), row.chosen, 'Annotation for ' + row.source);
          file.title = 'Left as it is, the table follows Annotations from (above); or pick its own file';
          file.addEventListener('change', () => {
            const matches = {...(v[TAG.matches] || {})};
            if (file.value) matches[row.source] = file.value; else delete matches[row.source];
            save({[TAG.matches]:matches});
          });
          const done = data.tagged['*'] || data.tagged[row.source];
          const state = node('span', {}, [workingChip(current, row.source), done ? chip('Tagged', 'ok', done) :
            row.annotation ? chip('Not tagged') : chip('No annotation', 'warn', 'Export the image’s annotations in the Annotation step, or pick a file')]);
          const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.sample, title:row.source + (row.image ? ' · open ' + row.image : ''),
            onclick:() => openRow(row)});
          const line = node('tr', {'data-source':row.source, class:row.source === openTable ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
          fill(line, node('td', {}, [tick]), node('td', {}, [name]), node('td', {}, [file]), node('td', {}, [state]));
          return line;
        });
        // Where every table's annotations come from (a table can still pick its own file).
        const sourceNow = v.seg_tag_annotation_source || (v.seg_tag_use_annotation_export === false ? 'folder' : 'export');
        const annotationSource = choice([['export', 'The Annotation step’s export'], ['imported', 'The GeoJSON imported for each image'],
          ['folder', 'The Annotations folder (by name)']], sourceNow, 'Annotations from');
        annotationSource.addEventListener('change', () => save({seg_tag_annotation_source:annotationSource.value}, 'Annotations from: ' +
          annotationSource.options[annotationSource.selectedIndex].text + '.'));
        const columnChoice = (label, key, value) => {
          const select = choice([['', '(choose)'], ...found.columns.map(n => [n, n])], value || '', label);
          select.addEventListener('change', () => save({[key]:select.value}));
          return select;
        };
        // A folder is its whole path, typed (inside the project or not); Enter or leaving the box saves it.
        const pathBox = (label, kind, key, tip, disabled) => {
          const input = node('input', {type:'text', value:(data.paths || {})[kind] || data.folders[kind] || '', maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
          input.disabled = !!disabled;
          input.addEventListener('change', () => { const value = input.value.trim(); if (value && value !== (data.paths || {})[kind]) save({[key]:value}, label + ' folder saved.'); });
          return field(label, input, tip);
        };
        const layout = node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Tables'}),
          ...[[PER_ROI, 'One per image'], [COMBINED, 'One merged table']].map(([value, text]) => {
            const radio = node('input', {type:'radio', name:'tag-layout', value, 'aria-label':text}); radio.checked = v[TAG.mode] === value;
            radio.addEventListener('change', () => { if (radio.checked) save({[TAG.mode]:value}, text + '.'); });
            return node('label', {class:'inline-check'}, [radio, node('span', {text})]);
          }), node('span', {class:'grow'}),
          help('One per image: a CSV or Parquet file per image, as Segmentation writes them. One merged table: every image’s cells in one CSV or Parquet ' +
            'file, with a column naming each cell’s image (batch_id, say); the tagged table is written as one file too.')]);
        const follow = node('input', {type:'checkbox', 'aria-label':'Follow Segmentation'}); follow.checked = v.seg_tag_follow_segmentation !== false;
        follow.addEventListener('change', () => save({seg_tag_follow_segmentation:follow.checked},
          follow.checked ? 'Tables: Segmentation’s output, followed when it changes.' : 'Tables: type the folder.'));
        const suffix = node('input', {type:'text', value:v[TAG.outSuffix], maxlength:'64', 'aria-label':'Tagged name ending'});
        suffix.addEventListener('change', () => { if (suffix.value.trim()) save({[TAG.outSuffix]:suffix.value.trim()}); else note.textContent = 'The name ending cannot be empty.'; });
        const tablesFile = combined ? choice([['', '(choose)'], ...(found.tables || []).map(n => [n, n])], found.table || '', 'Merged table') : null;
        if (tablesFile) tablesFile.addEventListener('change', () => save({[TAG.file]:tablesFile.value}));
        const done = found.rows.filter(row => data.tagged['*'] || data.tagged[row.source]).length;
        const matched = found.rows.filter(row => row.annotation).length;
        fill(place.body,
          layout,
          combined ? pathBox('Folder', 'merged', TAG.mergedDir, 'The folder holding the merged table.') :
            node('div', {}, [pathBox('Folder', 'csv', TAG.csvDir, 'The folder with one cell table per image (CSV or Parquet). With Follow Segmentation, the Segmentation step’s Output.', follow.checked),
              node('div', {class:'row field'}, [node('label', {class:'lbl'}), node('label', {class:'inline-check'}, [follow, node('span', {text:'Follow Segmentation'})]),
                help('On: the tables are what the Segmentation step writes, in its Output folder, even after it changes. Off: type the folder.')])]),
          combined ? field('Table', tablesFile, 'The merged CSV or Parquet file in that folder.') : null,
          combined ? field('Image column', columnChoice('Image column', TAG.batch, found.batch_column), 'The column naming each cell’s image (batch_id, say): its values are the rows below.') : null,
          field('Annotations from', annotationSource, 'Export: each image’s file from the Annotation step’s Export (its rules: kept inside, subtracted, ' +
            'renamed), else the folder’s file by name. Imported: the GeoJSON imported for each image (Annotation > Export > Imported files), as it is. ' +
            'Folder: the file matching each table’s name in the Annotations folder below. A table can pick its own file in its row.'),
          pathBox('Annotations', 'geojson', TAG.geojsonDir, 'Annotation GeoJSON files matched to the tables by name (or picked in each row).'),
          pathBox('Output', 'output', TAG.outDir, 'Where the tagged tables are written: <table><ending>, in the format of the table (CSV or Parquet). The originals are kept.'),
          node('div', {class:'row'}, [chip(found.rows.length + (combined ? ' images' : ' tables')), chip(matched + ' with annotations', matched === found.rows.length ? 'ok' : 'warn'),
            chip(done + ' tagged', done ? 'ok' : '')]),
          found.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs studio-seg-list'},
            [node('tr', {}, [node('th', {}, [head]), ...['Image', 'Annotations', 'State'].map(text => node('th', {text}))]), ...rows])])
            : node('p', {class:'hint', text:combined ? 'Choose the merged table and its image column above.' : 'No cell tables in this folder: type the folder above.'}),
          node('details', {class:'studio-options'}, [node('summary', {text:'Table columns'}),
            field('X, Y', node('div', {class:'inline'}, [columnChoice('X column', TAG.x, found.coordinates.x), columnChoice('Y column', TAG.y, found.coordinates.y)]),
              'Each cell’s position in full-resolution image pixels (found by itself when named x_centroid, y_centroid…).'),
            field('Ending', suffix, 'Added to each tagged table’s name, so it never replaces the original.')]),
          note);
        const gpu = node('input', {type:'checkbox', 'aria-label':'GPU distances'}); gpu.checked = !!v[TAG.gpu];
        gpu.addEventListener('change', () => save({[TAG.gpu]:gpu.checked}));
        const cpu = node('input', {type:'number', min:'0', max:'256', step:'1', value:String(v.seg_tag_cpu_workers || 0), 'aria-label':'Tagging CPU cores'});
        cpu.addEventListener('change', () => { const n = Number(cpu.value); if (Number.isInteger(n) && n >= 0 && n <= 256) save({seg_tag_cpu_workers:n}); });
        fill(compute.body,
          field('CPU cores', cpu, '0: Auto, up to 8 available cores. More cores can speed up gap-aware distances; results use the same geometry. Counts are limited to your allocation. File loading and geometry preparation also take time.'),
          node('div', {class:'row field'}, [node('label', {class:'inline-check'}, [gpu, node('span', {text:'GPU distances'})]), node('span', {class:'grow'}),
            help('CUDA can accelerate large ordinary boundary-distance calculations. Gap-aware distances and beds use CPU workers; this switch does not move them to the GPU.')]),
          node('p', {class:'muted', text:'0 = Auto · Applies to new runs'}));
        // ---- tissue pieces
        const classes = found.classes;
        if (!classes.length) {
          [pieces, marks, beds].forEach(part => fill(part.body, node('p', {class:'hint', text:'The annotations appear once the tables have annotation files ' +
            '(export them in the Annotation step).'})));
          drawTrial(); return;
        }
        const parent = classes.includes(v[TAG.tissue]) ? v[TAG.tissue] : classes[0];
        const parentPick = choice(classes.map(n => [n, n]), parent, 'Parent annotation');
        parentPick.addEventListener('change', () => save({[TAG.tissue]:parentPick.value, [TAG.edgeRef]:parentPick.value}));
        const unitPick = choice([['um', 'µm (µm² for sizes)'], ['px', 'pixels']], v.seg_tag_units || 'um', 'Units');
        unitPick.addEventListener('change', () => save({seg_tag_units:unitPick.value}));
        const edge = node('input', {type:'checkbox', 'aria-label':'Edge cells'}); edge.checked = !!v[TAG.edge];
        edge.addEventListener('change', () => save({[TAG.edge]:edge.checked, [TAG.edgeRef]:parent}));
        const microns = data.microns || 0;
        const edgeDistance = node('input', {type:'number', min:'0', step:'1', 'aria-label':'Edge distance',
          value:String(Math.round((units() === 'µm' && microns ? v[TAG.edgeBuffer] * microns : v[TAG.edgeBuffer]) * 10) / 10)});
        edgeDistance.addEventListener('change', () => { const n = Number(edgeDistance.value); if (n >= 0) save({[TAG.edgeBuffer]:units() === 'µm' && microns ? n / microns : n}); });
        const edgeName = node('input', {type:'text', value:v[TAG.edgeColumn], maxlength:'64', class:'grow', 'aria-label':'Edge column name'});
        edgeName.addEventListener('change', () => { if (edgeName.value.trim()) save({[TAG.edgeColumn]:edgeName.value.trim()}); });
        fill(pieces.body,
          field('Parent', parentPick, 'The annotation whose separate polygons are the tissue pieces (usually tissue).'),
          columnRow(parent, 'inside', 'Inside', '1 when the cell is inside a tissue piece, else 0.'),
          columnRow(parent, 'roi', 'Piece', 'Which tissue piece the cell is in (1 = the largest); 0 outside every piece. Classic calls it tissue_roi.'),
          columnRow(parent, 'roi_size', 'Size', 'The area of the cell’s tissue piece, in ' + (units() === 'µm' ? 'µm²' : 'square pixels') + '.'),
          field('Units', unitPick, 'Distances in µm and sizes in µm² use each image’s pixel size' + (microns ? ' (here ' + microns + ' µm per pixel)' : '') +
            '; pixels keeps the image’s own units, as classic tagging does.'),
          node('div', {class:'row field studio-tag-column'}, [node('label', {class:'inline-check'}, [edge, node('span', {text:'Edge cells'})]),
            v[TAG.edge] ? node('div', {class:'inline grow'}, [node('span', {class:'muted', text:'within'}), edgeDistance, node('span', {class:'muted', text:units()}), edgeName]) : node('span', {class:'grow'}),
            help('A 0/1 column: 1 for cells this close to the edge of their tissue piece.')]));
        // ---- the other annotations
        const others = classes.filter(n => n !== parent);
        const open = current.tagOpen || (current.tagOpen = {});
        const cards = others.map(label => {
          const rule = data.rules.find(r => r['Annotation Label'] === label) || {};
          const count = ['Binary (0/1)', 'ROI ID', 'ROI Size', 'Distance'].filter(key => rule[key]).length;
          const card = node('details', {class:'studio-options studio-tag-class', 'data-class':label});
          card.open = open[label] ?? count > 0;
          card.addEventListener('toggle', () => { open[label] = card.open; });
          card.append(node('summary', {text:label + (count ? ' · ' + count + ' column' + (count === 1 ? '' : 's') : '')}),
            columnRow(label, 'inside', 'Inside', '1 when the cell is inside ' + label + ', else 0.'),
            columnRow(label, 'roi', 'Piece', 'Which polygon of ' + label + ' holds the cell, numbered by size within each tissue piece; 0 outside.'),
            columnRow(label, 'roi_size', 'Size', 'The area of that polygon, in ' + (units() === 'µm' ? 'µm²' : 'square pixels') + '.'),
            columnRow(label, 'distance', 'Distance', 'How far the cell is from the edge of ' + label + ', in ' + units() + ': negative inside, positive ' +
              'outside, measured inside the cell’s tissue piece.'));
          return card;
        });
        const clean = node('input', {type:'checkbox', 'aria-label':'Straight lines inside the piece'}); clean.checked = !!v[TAG.clean];
        clean.addEventListener('change', () => save({[TAG.clean]:clean.checked}));
        fill(marks.body, ...(cards.length ? cards : [node('p', {class:'muted', text:'No other annotations in the files: only tissue pieces can be tagged.'})]),
          node('div', {class:'row field'}, [node('label', {class:'inline-check'}, [clean, node('span', {text:'Distances stay inside the piece'})]), node('span', {class:'grow'}),
            help('On (recommended): a distance is only written when the straight line from the cell to the annotation stays inside the cell’s ' +
              'tissue piece, so it never crosses glass, another piece, or the mouth of a U-shaped piece; other cells are left blank. Off: the ' +
              'straight-line distance within the piece, as older runs did.')]));
        // ---- beds
        const chosen = new Set(v[TAG.beds] || []);
        const bedTicks = node('div', {class:'studio-markers'}, others.map(label => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Bed ' + label}); tick.checked = chosen.has(label);
          tick.addEventListener('change', () => {
            tick.checked ? chosen.add(label) : chosen.delete(label);
            if (chosen.size === 1) { note.textContent = 'Tick a second annotation: a bed compares at least two.'; return; }
            save({[TAG.beds]:Array.from(chosen)});
          });
          return node('label', {}, [tick, node('span', {text:label})]);
        }));
        const minBed = node('input', {type:'number', min:'1', max:'100000', step:'1', value:String(v[TAG.minBed]), 'aria-label':'Smallest bed'});
        minBed.addEventListener('change', () => { const n = Math.round(Number(minBed.value)); if (n >= 1) save({[TAG.minBed]:n}); });
        const bedColumn = node('input', {type:'text', value:v.seg_tag_bed_column ?? 'bed', maxlength:'64', class:'grow', 'aria-label':'Bed column name'});
        bedColumn.addEventListener('change', () => save({seg_tag_bed_column:bedColumn.value.trim()}));
        const bedNames = Array.from(chosen).flatMap(label => [field(label + ' bed', nameBox(label, 'bed'), '1 for the cells given to ' + label + ', 0 for the others, blank outside the tissue.'),
          field(label + ' piece', nameBox(label, 'bed_id'), 'Which polygon of ' + label + ' the cell was given to.')]);
        fill(beds.body, bedTicks,
          chosen.size >= 2 ? field('Bed column', bedColumn, 'One column with the name of the annotation each cell was given to (empty: none).') : null,
          chosen.size >= 2 ? field('Smallest', minBed, 'An annotation polygon needs at least this many cells to take part (10 to start).') : null,
          chosen.size >= 2 ? node('details', {class:'studio-options'}, [node('summary', {text:'Column names'}), ...bedNames]) : null,
          chosen.size < 2 ? node('p', {class:'muted', text:'Tick two or more annotations to give each cell to the nearest of them.'}) : null);
        drawTrial();
        syncBar();
      };
      // ---- preview
      const drawTrial = () => {
        const go = node('button', {class:'btn primary', text:previewed ? 'Preview again' : 'Preview this image'});
        const view = node('input', {type:'checkbox', 'aria-label':'Cells in view only'}); view.checked = inView;
        view.addEventListener('change', () => { inView = view.checked; });
        const pick = previewed ? choice(previewed.columns.map(name => [name, name]), previewed.shown && previewed.columns.includes(previewed.shown) ? previewed.shown : previewed.columns[0], 'Colour cells by') : null;
        const summary = node('p', {class:'muted'});
        const number = value => value.toLocaleString(undefined, {maximumFractionDigits:1});
        const describe = item => item.kind === 'numeric' ? 'from ' + number(item.min) + ' to ' + number(item.max) + ' (middle ' + number(item.median) + ')' +
          (item.blank ? ', ' + item.blank.toLocaleString() + ' blank' : '') :
          item.counts.map(([value, count]) => value + ': ' + count.toLocaleString()).join(' · ') + (item.blank ? ' · blank: ' + item.blank.toLocaleString() : '');
        const drawSummary = () => { if (pick) summary.textContent = pick.value + ': ' + describe(previewed.summary[pick.value]); };
        go.addEventListener('click', async () => {
          go.disabled = true; trialNote.textContent = 'Tagging ' + (inView ? 'the cells in view' : 'this image’s cells') + ' in memory…';
          try {
            const area = inView ? viewBounds() : null;
            previewed = await post(route('/tagging_preview'), area ? {bounds:area} : {});
            trialNote.textContent = previewed.cells.toLocaleString() + (previewed.of > previewed.cells ? ' of ' + previewed.of.toLocaleString() : '') +
              ' cells of ' + previewed.table + ' tagged with ' + previewed.annotation + (previewed.warnings.length ? '. ' + previewed.warnings.join(' ') : '. Nothing is written.');
            drawTrial();
          } catch (error) { trialNote.textContent = error.message; }
          finally { go.disabled = false; }
        });
        const clear = node('button', {class:'btn small ghost', text:'Clear', hidden:!previewed, onclick:async () => {
          try { await post(route('/tagging_preview'), {clear:true}); } catch (_) {}
          previewed = null; previewCells(null); trialNote.textContent = ''; drawTrial();
        }});
        if (pick) pick.addEventListener('change', () => { previewed.shown = pick.value; previewCells(previewed.prefix + pick.value); drawSummary(); });
        fill(trial.body, node('div', {class:'row'}, [go, node('label', {class:'inline-check', title:'Only the cells in view: much quicker on a large image'},
            [view, node('span', {text:'In view only'})]), clear]),
          pick ? field('Colour by', pick, 'The image’s cells take this new column’s colours: distances as a scale, the rest as categories. The legend is in Layers (Cells).') : null,
          summary, trialNote);
        drawSummary();
        if (pick) previewCells(previewed.prefix + pick.value);
      };
      // ---- bottom: Tag [Again] This image | Selected
      const status = node('p', {class:'muted', role:'status'});
      const planned = chip('');
      const syncBar = () => { const list = data?.columns_added || []; planned.textContent = list.length + ' column' + (list.length === 1 ? '' : 's');
        planned.title = list.join(', ') || 'Tick at least one column to add'; planned.className = 'chip' + (list.length ? ' ok' : ' warn'); };
      const tagRun = (button, which) => async () => {
        button.disabled = true;
        try {
          let tables = null;
          if (which === 'open') {
            const image = (currentSpec?.layers || []).find(l => l.type === 'image')?.name;
            const row = data.found.rows.find(r => r.image && r.image === image) || data.found.rows.find(r => r.source === openTable);
            if (!row) throw Error('Open an image whose cells are in the list first.');
            tables = [row.source];
          } else {
            tables = Array.from(picked);
            if (!tables.length) throw Error('Tick the tables first.');
          }
          const answer = await runCommand('tag_cells', {tables}, {confirmed:true});
          showResult(answer); status.textContent = answer.message;
          if (answer.data?.job) { await refreshJobs(); showTab('jobs'); }
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const one = node('button', {class:'btn', text:'This image'}), many = node('button', {class:'btn primary', text:'Selected'});
      one.addEventListener('click', tagRun(one, 'open')); many.addEventListener('click', tagRun(many, 'picked'));
      current.runBar([node('div', {class:'row wrap'}, [node('span', {class:'lbl', text:'Tag:'}), planned, one, many,
        help('Tags the cells with the columns above, as a job: this image’s table, or the ticked ones (tick the header box for all). ' +
          'Tagged tables are new files; the originals are kept. Batch correction reads the tagged tables.')]), status]);
      followJobs(current, ['cell_tagging', 'import_annotations'], load);
      load();
    };

    // ------------------------------------------------------------ Batch correction
    // Samples: one row per ROI with its slide (from the ROI step) and the study's own columns, linked from a CSV.
    const samplesPanel = (body, changed = () => {}) => {
      const summary = node('div', {class:'row wrap'}), grid = node('div', {class:'studio-samples'}), status = node('p', {class:'muted', role:'status'});
      const columnsRow = node('div', {class:'row wrap'}), importBox = node('div');
      const file = node('input', {type:'file', accept:'.csv,.tsv,.txt', 'aria-label':'Your table (CSV)', hidden:true});
      const pickFile = node('button', {class:'btn small', text:'Add your table…', title:'Import metadata: choose matching columns and preview the join.'});
      pickFile.addEventListener('click', () => file.click());
      const addColumn = node('button', {class:'btn small', text:'Add column', onclick:() => {
        const name = node('input', {type:'text', placeholder:'Column name', maxlength:'64', 'aria-label':'New sample column'});
        const save = async () => { if (!name.value.trim()) return; await act('add', {column:name.value.trim()}, 'Column added. Enter its values in the table.'); };
        name.addEventListener('keydown', event => { if (event.key === 'Enter') save(); });
        importBox.replaceChildren(node('div', {class:'row'}, [name, node('button', {class:'btn small', text:'Add', onclick:save}),
          node('button', {class:'btn small ghost', text:'Cancel', onclick:() => importBox.replaceChildren()})])); name.focus();
      }});
      body.append(summary, grid, columnsRow, node('div', {class:'row wrap'}, [pickFile, addColumn, file,
        help('Slides come from the ROI step. Use one metadata row per ROI if that ROI has one patient and timepoint. If an ROI mixes samples, match a sample ID already recorded for each cell. Match by slide only for values shared by every sample on that slide. All imported columns are saved in new correction outputs, whether or not you tick them in Keep biological differences.')]), importBox, status);
      let found = null;
      const act = async (action, payload, message) => {
        try { draw(await post(route('/samples/' + action), payload)); if (action === 'add') importBox.replaceChildren(); status.textContent = message || ''; await changed(); }
        catch (error) { status.textContent = error.message; }
      };
      const draw = value => {
        found = value;
        const slides = new Set(found.rows.map(row => row.slide)), guessed = found.rows.filter(row => row._found === 'name').length;
        summary.replaceChildren(...[chip(found.rows.length + ' ROIs'), chip(slides.size + ' slides', 'ok'),
          guessed ? chip(guessed + ' slide' + (guessed === 1 ? '' : 's') + ' from names only', 'warn', 'These ROIs are in no ROI file; their slide is read from the name. Check them.') : null,
          node('span', {class:'muted grow', text:found.saved ? 'Saved in ' + found.file : 'Not saved yet: slides from the ROI step'})].filter(Boolean));
        const extra = found.columns.filter(name => !['roi', 'slide'].includes(name));
        const head = node('tr', {}, ['ROI', 'Slide', ...extra].map(text => node('th', {text})));
        const rows = found.rows.map(row => node('tr', {}, [
          node('td', {text:row.roi, title:row._table || 'no cell table yet'}),
          ...['slide', ...extra].map(column => {
            const input = node('input', {type:'text', value:row[column] || '', 'aria-label':column + ' of ' + row.roi,
              class:column === 'slide' && row._found === 'name' ? 'guess' : '', title:column === 'slide' ?
                (row._suggested ? 'The ROI step says ' + row._suggested : row._source ? 'Parent slide from ' + row._source : 'Check the parent slide') :
                row[column] === 'Varies by cell' ? 'Imported values differ between cells. Editing this value sets it for every cell of this ROI.' : ''});
            input.addEventListener('change', () => act('edit', {roi:row.roi, column, value:input.value}, row.roi + ': ' + column + ' saved.'));
            return node('td', {}, [input]);
          })]));
        grid.replaceChildren(node('table', {class:'studio-counts'}, [head, ...rows]));
        columnsRow.replaceChildren(...extra.map(name => node('span', {class:'chip'}, [node('span', {text:name}),
          node('button', {class:'btn small ghost', text:'Rename', 'aria-label':'Rename ' + name, onclick:() => {
            const input = node('input', {type:'text', value:name, 'aria-label':'New name for ' + name});
            const done = () => { if (input.value.trim() && input.value.trim() !== name) act('rename', {column:name, name:input.value.trim()}, name + ' renamed.'); else draw(found); };
            input.addEventListener('keydown', event => { if (event.key === 'Enter') done(); if (event.key === 'Escape') draw(found); });
            columnsRow.replaceChildren(input, node('button', {class:'btn small', text:'Save', onclick:done})); input.focus();
          }}),
          node('button', {class:'btn small ghost', text:'×', title:'Remove ' + name, 'aria-label':'Remove ' + name, onclick:() => act('drop', {column:name}, name + ' removed.')})])));
      };
      // Your table: the link column and level are suggested; choose the columns and their names; preview, then add.
      file.addEventListener('change', async () => {
        const chosen = file.files[0]; file.value = '';
        if (!chosen) return;
        let text;
        try { text = await chosen.text(); } catch (error) { status.textContent = error.message; return; }
        let guess;
        try { guess = await post(route('/samples/suggest'), {text}); } catch (error) { status.textContent = error.message; return; }
        const linkColumn = choice(guess.columns.map(item => [item.name, item.name]), guess.link_column || guess.columns[0].name, 'Link column');
        const linkTo = choice(guess.targets.map(item => [item.value, item.label]), guess.link_to, 'Match in dataset');
        const picks = guess.columns.map(item => {
          const keep = node('input', {type:'checkbox', 'aria-label':'Add ' + item.name, checked:true});
          const name = node('input', {type:'text', value:item.name, 'aria-label':'Name for ' + item.name});
          return {item, keep, name, row:node('div', {class:'row'}, [keep, name, node('span', {class:'muted grow', text:item.examples.join(', ')})])};
        });
        const result = node('div');
        const payload = () => ({text, link_column:linkColumn.value, link_to:linkTo.value,
          columns:Object.fromEntries(picks.filter(pick => pick.keep.checked && pick.item.name !== linkColumn.value).map(pick => [pick.item.name, pick.name.value.trim()]))});
        const showPicks = () => picks.forEach(pick => { pick.row.hidden = pick.item.name === linkColumn.value; });
        linkColumn.addEventListener('change', showPicks); showPicks();
        const preview = node('button', {class:'btn small', text:'Preview', onclick:async () => {
          try {
            const seen = await post(route('/samples/preview'), payload());
            const table = node('table', {class:'studio-counts'}, [node('tr', {}, Object.keys(seen.preview[0] || {}).map(text => node('th', {text}))),
              ...seen.preview.map(row => node('tr', {}, Object.values(row).map(text => node('td', {text:String(text)}))))]);
            result.replaceChildren(node('p', {class:seen.missing.length ? 'hint' : 'muted', text:seen.matched + ' of ' + seen.rois + ' ROIs matched' +
                (seen.matched_cells != null ? ' · ' + seen.matched_cells.toLocaleString() + ' of ' + seen.cells.toLocaleString() + ' cells matched' : '') +
                (seen.missing.length ? '; no match for ' + seen.missing.slice(0, 6).join(', ') + (seen.missing.length > 6 ? '…' : '') : '') +
                (seen.unused.length ? '. Rows of your table not used: ' + seen.unused.slice(0, 6).join(', ') + (seen.unused.length > 6 ? '…' : '') : '') +
                (seen.replaced.length ? '. Replaces: ' + seen.replaced.join(', ') : '') + '.'}), node('div', {class:'studio-samples'}, [table]), add);
          } catch (error) { result.replaceChildren(node('p', {class:'hint', text:error.message})); }
        }});
        const add = node('button', {class:'btn primary small', text:'Add these columns', onclick:async () => {
          try { const done = await post(route('/samples/link'), payload()); draw(done.table); importBox.replaceChildren(); status.textContent = 'Added to ' + done.matched + ' ROIs.'; await changed(); }
          catch (error) { status.textContent = error.message; }
        }});
        importBox.replaceChildren(node('div', {class:'studio-import'}, [
          node('p', {class:'muted', text:chosen.name + ': ' + guess.rows + ' rows' + (guess.matches ? ', ' + guess.matches + ' names match the project' : ', no names match yet')}),
          field('Your column', linkColumn, 'Choose the ID column in your metadata table. Each ID must occur once. Example: roi contains S1_roi_1. If a patient has several timepoints, use an ID for each patient-and-timepoint sample, not patient alone.'),
          field('Match in dataset', linkTo, 'Choose where those same IDs occur in Studio. Example: roi → ROI name, or sample_id → Cell tables · sample_id. Slide ID gives the same values to every cell on that slide, so do not use it for patient or timepoint when a slide contains several. Preview the matches before adding.'),
          node('p', {class:'lbl', text:'Columns to add (you can rename them)'}), ...picks.map(pick => pick.row),
          node('div', {class:'row'}, [preview, node('button', {class:'btn small ghost', text:'Cancel', onclick:() => importBox.replaceChildren()})]), result]));
      });
      (async () => {
        try { draw(await get(route('/samples'))); } catch (error) { status.textContent = error.message; }
      })();
    };

    // Batch correction in the Studio design: the cell tables (one row per ROI, ticked to take part), the settings,
    // then Correct: Selected. Tab 6's steps run as a job, between the chosen units (the slide by default).
    const correctionPanel = (list, body, current, reload) => {
      const status = node('p', {class:'muted', role:'status'}), result = node('div');
      const run = node('button', {class:'btn primary', text:'Selected', disabled:true});
      const savedChip = chip('Saved', 'ok');
      body.closest('.studio-sec')?.querySelector('.studio-sec-h').append(savedChip);
      let running = false;
      list.append(node('p', {class:'muted', text:'Reading the cell tables…'}));
      const picked = current.correctionPicked || (current.correctionPicked = new Set());
      (async () => {
        let plan;
        try { plan = await get(route('/correction')); } catch (error) { list.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
        if (!list.isConnected) return;
        if (!current.correctionPickedOnce) { current.correctionPickedOnce = true; plan.rows.filter(row => row.match === 'matched').forEach(row => picked.add(row.roi)); }
        let openRoi = '';
        // ---- the list
        const openRow = async row => {
          openRoi = row.roi; drawList();
          let images = [];
          try { images = await loadImages(); } catch (_) { /* said below */ }
          const image = images.find(item => item.path === row.image_path);
          if (image) chooseImage(image);
          else try {
            const added = await post('/session/' + encodeURIComponent(session) + '/add-image', {path:row.image_path});
            chooseImage(added.images.find(item => item.id === added.added));
          }
          catch (error) { status.textContent = error.message; }
        };
        const counts = node('div', {class:'row wrap'});
        const tableBox = node('div', {class:'studio-pairs-scroll studio-correction-matches'});
        const foldersBox = node('div');
        const drawFolders = () => fill(foldersBox, ...[['images', 'Images'], ['input', 'Cell tables'], ['output', 'Output']].map(([kind, label]) => {
          const input = node('input', {type:'text', value:plan.folders[kind], maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
          const save = async () => {
            if (!input.value.trim() || input.value.trim() === plan.folders[kind]) return;
            input.disabled = true; status.textContent = 'Reading the new ' + label.toLowerCase() + ' folder…';
            try { await post(route('/correction/folders'), {folders:{[kind]:input.value.trim()}}); current.correctionPickedOnce = false; picked.clear(); reload(); }
            catch (error) { status.textContent = error.message; input.disabled = false; }
          };
          input.addEventListener('change', save);
          input.addEventListener('keydown', event => { if (event.key === 'Enter') input.blur(); });
          return field(label, input, kind === 'images' ? 'ROI image directory. These images define the rows below; each is matched to its measurement table. Press Enter to refresh.' : kind === 'input' ? 'Segmentation or Cell tagging measurements. CSV and Parquet names are matched to the ROI images; known tagging suffixes are removed. Verified Parquet caches are used for faster reads.' :
            'New combined Parquet tables and their run settings are saved here. Existing results are kept. Use Jobs → Saved results to make a CSV copy.');
        }));
        const drawList = () => {
          const available = new Set(plan.rows.filter(row => row.match === 'matched').map(row => row.roi));
          Array.from(picked).forEach(roi => { if (!available.has(roi)) picked.delete(roi); });
          const head = pickAll(Array.from(available), picked, drawList);
          const unitOf = row => unit.value === '(use batch column as-is)' ? row.roi : String(row.values?.[unit.value] ?? '');
          const selectedRows = plan.rows.filter(row => picked.has(row.roi));
          const units = new Set(selectedRows.map(unitOf).filter(Boolean));
          run.disabled = running || picked.size < 2 || units.size < 2 || selectedRows.some(row => !unitOf(row)) || chosen.size < 2;
          const showUnit = unit.value !== '(use batch column as-is)';
          fill(counts, chip(plan.rows.length + ' images'), chip(available.size + ' matched'), chip(picked.size + ' ticked'),
            chip(plan.rows.filter(row => row.corrected).length + ' corrected', plan.rows.some(row => row.corrected) ? 'ok' : ''),
            chip(units.size + ' ' + (unit.value === '(use batch column as-is)' ? 'ROI' : unit.value) + (units.size === 1 ? '' : 's') + ' to compare', units.size >= 2 ? '' : 'warn',
              'The correction removes differences between these; it needs at least two'),
            plan.last ? chip('Active result', '', plan.last.name + ' · ' + plan.last.rois + ' ROIs, between ' + (plan.last.unit === '(use batch column as-is)' ? 'ROIs' : plan.last.unit + 's')) : null);
          const rows = plan.rows.map(row => {
            const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.roi}); tick.checked = picked.has(row.roi);
            tick.disabled = row.match !== 'matched';
            tick.addEventListener('change', () => { tick.checked ? picked.add(row.roi) : picked.delete(row.roi); drawList(); });
            const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:row.image_path + (row.path ? '\nMatched table: ' + row.path : '') + '\nOpen image', onclick:() => openRow(row)});
            const stateLabel = {saved:'✓ Corrected', stale:'Inputs changed', legacy:'Saved · unverified', excluded:'No cells retained', not_run:'Not run', missing:'No table', ambiguous:'Multiple tables'}[row.status] || 'Not run';
            const state = node('span', {}, [workingChip(current, row.roi),
              chip(stateLabel, row.corrected ? 'ok' : ['stale', 'legacy', 'excluded', 'missing', 'ambiguous'].includes(row.status) ? 'warn' : '',
                row.result ? row.result + (row.corrected_cells != null ? ' · ' + row.corrected_cells.toLocaleString() + ' cells with output values' : '') +
                  '. Saved results use that run’s settings; green means saved, not scientifically validated.' : 'No saved correction result for this table'),
              row.tagged ? chip('Tagged', 'ok', 'From the Cell tagging output folder') : null]);
            const line = node('tr', {'data-roi':row.roi, class:openRoi === row.roi ? 'selected' : ''});
            line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
            fill(line, node('td', {}, [tick]), node('td', {}, [name]),
              node('td', {class:'muted correction-table-file', text:row.table || '–', title:row.candidates.join('\n')}),
              showUnit ? node('td', {class:'muted', text:unitOf(row) || '–'}) : null, node('td', {}, [state]));
            return line;
          });
          tableBox.replaceChildren(plan.rows.length ? node('table', {class:'studio-counts studio-pairs'}, [node('tr', {}, [node('th', {}, [head]), node('th', {text:'ROI image'}),
            node('th', {text:'Cell table', class:'correction-table-file'}), showUnit ? node('th', {text:unit.value === 'slide' ? 'Slide ID' : unit.value, style:'width:20%'}) : null, node('th', {text:'State'})]), ...rows]) : node('p', {class:'muted', text:'No ROI images in this folder. Choose their directory above.'}));
        };
        drawFolders();
        fill(list, foldersBox, counts, tableBox,
          plan.missing.length ? node('p', {class:'muted', text:'No cell table yet for ' + plan.missing.slice(0, 6).join(', ') + (plan.missing.length > 6 ? '…' : '') + '.'}) : null);
        // ---- the settings
        const settings = {...plan.settings}, chosen = new Set(plan.markers), keep = new Set(settings.covariates || []);
        const unit = choice(plan.units.map(item => [item.column, (item.column === '(use batch column as-is)' ? 'ROI' : item.name) + ' (' + item.units + ')']), plan.unit, 'Correct between');
        unit.addEventListener('change', drawList);
        drawList();
        const method = choice([['cycombine', 'cyCombine'], ['monotonic', 'Monotonic (keeps each unit’s cell order)'],
          ['per_cluster', 'Earlier cyCombine approximation']], settings.correction_mode, 'Method');
        const methodNote = help('');
        const onlyOthers = [];  // settings the exact cyCombine does not use
        const describeMethod = () => {
          const tip = {
            cycombine:'Python port of cyCombine: group similar cells with an SOM, then apply ComBat within each group. Output values are arcsinh-transformed.',
            monotonic:'The batch effect is estimated within the same SOM groups, then applied as one shift and scale per unit and marker, so no cell changes place within its unit.',
            per_cluster:'SPACEMAP’s earlier version of cyCombine (another SOM and ComBat library). Choose it only to reproduce an earlier run.',
          }[method.value];
          methodNote.dataset.tip = tip; methodNote.setAttribute('aria-label', tip);
          onlyOthers.forEach(row => { row.hidden = method.value === 'cycombine'; });
        };
        method.addEventListener('change', describeMethod); describeMethod();
        const markerBox = node('details', {class:'studio-borders'}), markerSummary = node('summary'), markerList = node('div', {class:'studio-markers'});
        const describeMarkers = () => { markerSummary.textContent = chosen.size + ' markers selected'; markerSummary.title = Array.from(chosen).join(', '); drawList(); };
        plan.common.forEach(name => {
          const input = node('input', {type:'checkbox', 'aria-label':'Correct ' + name}); input.checked = chosen.has(name);
          input.addEventListener('change', () => { if (input.checked) chosen.add(name); else chosen.delete(name); describeMarkers(); });
          markerList.append(node('label', {}, [input, node('span', {text:name, title:name})]));
        });
        describeMarkers(); markerBox.append(markerSummary, markerList);
        const protect = node('div', {class:'row wrap'});
        const biologyHelp = name => {
          const key = name.toLowerCase().replace(/[ -]/g, '_');
          if (/^(arm|treatment|treatment_group|condition)$/.test(key)) return 'Treatment group. Tick to account for treatment differences you want to keep. Each treatment-and-timepoint group should be represented on more than one technical batch.';
          if (/^(timepoint|time_point|visit)$/.test(key)) return 'When the sample was collected, such as before or after treatment. Tick to account for these differences. Check that treatment-and-timepoint groups occur across batches.';
          if (/^(patient|patient_id|patientid)$/.test(key)) return 'Patient identity. Leave off for an initial treatment/timepoint correction. Consider ticking only when patient samples span technical batches and you need to account for patient differences. Patient ID is still saved when unticked.';
          if (/^(response|outcome)$/.test(key)) return 'Clinical outcome. Keep this column for later comparisons. Tick only if your analysis plan also uses response in the correction and the response groups overlap across batches; it is optional.';
          if (/^(slide|slide_id|slideid|batch|batch_id|batchid)$/.test(key)) return 'Usually a technical ID. Choose the technical batch under Correct between. Leave its ID off here: the same difference cannot be both removed and preserved.';
          return 'Tick only if this column describes biological groups you want the correction to account for. Those groups need overlap across technical batches. The column is still saved when unticked.';
        };
        const drawProtect = () => fill(protect, ...(plan.covariate_options.length ? plan.covariate_options.map(name => {
          const tip = biologyHelp(name);
          const input = node('input', {type:'checkbox', 'aria-label':'Keep biological differences in ' + name, 'aria-description':tip}); input.checked = keep.has(name);
          input.addEventListener('change', () => { if (input.checked) keep.add(name); else keep.delete(name); });
          return node('label', {class:'chip', title:tip}, [input, node('span', {text:name})]);
        }) : [node('span', {class:'muted', text:'Add columns in Samples.'})]));
        drawProtect();
        const setupHelp = node('details', {class:'studio-options'}, [node('summary', {text:'How to assign the data'}),
          node('ol', {class:'muted'}, [
            'Choose Images (ROI images), Cell tables (Segmentation or Cell tagging results), and Output above. Check each image has a matching table; tick the ROIs to include.',
            'Open Samples. Check the parent slides from the ROI step. Add your table with patient, arm (treatment), timepoint and response. For one patient/timepoint per ROI, choose Your column: roi → Match in dataset: ROI name.',
            'If one ROI mixes samples, match a unique sample ID already present in its cell table instead, such as sample_id → Cell tables · sample_id. Patient ID alone cannot distinguish different timepoints. Do not assign patient data by slide when that slide contains several patients.',
            'Preview the import, check the matches and values, then Add these columns. A manual value in Samples applies to every cell in that ROI; Varies by cell means the import kept different values for different cells.',
            'Correct between selects the technical difference to remove: choose slide if each slide is a separate staining/acquisition batch, or your staining-run column if several slides share a batch.',
            'For a treatment/timepoint study, start with arm and timepoint under Keep biological differences only when the treatment-and-timepoint groups overlap across batches. Leave slide_id and batch_id off. Patient and response are optional; hover over their boxes for guidance. More ticks are not automatically better.',
            'Choose cyCombine and the raw marker measurements to correct. Click Correct: Selected. Review Distribution checks and Run notes, then Use this table in Studio when you want the new output used downstream.',
          ].map(text => node('li', {text}))),
          node('p', {class:'muted', text:'All imported metadata columns are saved, even when unticked. These boxes ask the model to account for biological differences; they do not guarantee those differences are preserved. Several ticks combine into one group, for example treatment A + before treatment. Those groups need enough support across batches.'})]);
        const number = (key, label, step, min, max) => {
          const input = node('input', {type:'number', step, min, max, value:String(settings[key]), 'aria-label':label});
          input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= min && value <= max) settings[key] = step === '1' ? Math.round(value) : value; else input.value = String(settings[key]); });
          return input;
        };
        const suffix = node('input', {type:'text', value:settings.corrected_suffix || '_corr', maxlength:'31', 'aria-label':'Corrected column suffix'});
        suffix.addEventListener('change', () => { if (/^_[A-Za-z0-9_]{1,30}$/.test(suffix.value.trim())) settings.corrected_suffix = suffix.value.trim(); else suffix.value = settings.corrected_suffix || '_corr'; });
        const parametric = node('input', {type:'checkbox', 'aria-label':'Parametric ComBat'}); parametric.checked = !!settings.parametric;
        parametric.addEventListener('change', () => { settings.parametric = parametric.checked; });
        const norm = choice([['scale', 'Z-score within each batch'], ['rank', 'Rank within each batch'], ['none', 'None']], settings.norm_method || 'scale', 'SOM normalization');
        norm.addEventListener('change', () => { settings.norm_method = norm.value; });
        const cyOnly = [field('SOM passes', number('som_rlen', 'SOM passes', '1', 1, 1000), 'Passes over all cells when training the cyCombine SOM. Default 10.'),
          field('Seed', number('cycombine_seed', 'SOM seed', '1', 0, 2147483647), 'Random seed for the cyCombine SOM. Default 473; kept with each saved result.')];
        const advanced = node('details', {class:'studio-options'}, [node('summary', {text:'Advanced'}),
          field('Corrected names', suffix, 'Each corrected marker is saved beside its raw one with this ending: CD45 and CD45' + (settings.corrected_suffix || '_corr') +
            '. Corrected values are in arcsinh space, not raw intensities. The selected cofactor is saved with each result.'),
          field('SOM normalization', norm, 'For grouping cells only. Z-score suits similar acquisitions; ranks can help with stronger batch effects. ComBat still corrects the original arcsinh values.'),
          field('SOM grid', number('som_grid_dim', 'SOM grid size', '1', 2, 50), 'The SOM is N × N groups of similar cells (cyCombine uses 8).'),
          ...cyOnly,
          ...(onlyOthers.push(field('SOM steps', number('som_iterations', 'SOM iterations', '1', 1000, 500000), 'Training steps of the SOM (Monotonic and the earlier approximation; cyCombine makes 10 passes over the cells).'),
            field('Fewest cells', number('min_cells_per_cluster', 'Fewest cells per group', '1', 1, 500), 'Groups with fewer cells of a unit are not used to estimate its shift (Monotonic and the earlier approximation).')), onlyOthers),
          field('Arcsinh cofactor', number('arcsinh_cofactor', 'Arcsinh cofactor', 'any', 0.01, 65535), 'Input must be raw measured intensities. Applies arcsinh(value / cofactor) once. The paper uses 5 for CyTOF; there is no universal imaging cofactor. Review marker distributions for your assay.'),
          node('label', {class:'row'}, [parametric, node('span', {text:'Parametric ComBat (faster)'})])]);
        const syncCy = () => cyOnly.forEach(row => { row.hidden = method.value !== 'cycombine'; });
        method.addEventListener('change', syncCy); syncCy();
        describeMethod();  // now that its settings exist, hide the ones cyCombine does not use
        body.replaceChildren(
          field('Correct between', unit, 'Which technical difference should be removed? Choose slide when each slide is a separate staining/acquisition batch. If several slides share a run, use that run’s batch_id instead. Do not choose patient, treatment or timepoint to remove technical effects. ROI names identify images; they are not automatically technical batches.'),
          field('Method', node('div', {class:'inline grow'}, [method, methodNote])),
          node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Markers'}), markerBox, help('Choose the measured marker columns to correct.')]),
          node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('span', {class:'lbl', text:'Keep biological differences'}),
            help('Which biological differences should the model try to keep? For example, tick arm and timepoint to account for treatment and collection time. Unticked columns are still saved. The groups must overlap across technical batches: if all before-treatment samples are on one slide and all after-treatment samples on another, the model cannot tell time from slide. Multiple ticks form combined groups; some cell groups may lack enough support, reported in Run notes.')]), protect]),
          setupHelp, advanced, node('details', {class:'studio-options'}, [node('summary', {text:'Cell filters (shared with classic)'}),
            node('p', {class:'muted', text:'Area: ' + plan.filters.area_col + ', percentiles ' + plan.filters.area_min_pct + '–' + plan.filters.area_max_pct +
              '. Marker outliers: ' + plan.filters.outlier_low + '–' + plan.filters.outlier_high + ' in most markers. Tissue: ' + plan.filters.tissue_col +
              (plan.filters.tissue_col !== 'None' ? ' = ' + plan.filters.tissue_val : '') + '. Post-correction QC: ' + (plan.filters.enable_post_qc ? 'on' : 'off') +
              '. Edit these in classic Batch correction. Excluded cells stay in the output with blank corrected values.'})]));
        const changes = () => ({...settings, correction_mode:method.value, batch_unit_col:unit.value, covariates:Array.from(keep)});
        let saveTimer = null, saving = Promise.resolve(), revision = 0;
        const persist = () => {
          clearTimeout(saveTimer); const version = ++revision, payload = {changes:changes(), markers:Array.from(chosen)};
          savedChip.textContent = 'Saving…';
          saving = saving.catch(() => {}).then(() => post(route('/correction/save'), payload)).then(() => {
            if (version === revision) { savedChip.textContent = 'Saved'; savedChip.className = 'chip ok'; }
          }).catch(error => { savedChip.textContent = 'Not saved'; savedChip.className = 'chip warn'; status.textContent = error.message; throw error; });
          return saving;
        };
        const scheduleSave = () => { savedChip.textContent = 'Not saved'; savedChip.className = 'chip warn'; clearTimeout(saveTimer); saveTimer = setTimeout(() => persist().catch(() => {}), 300); };
        body.addEventListener('change', () => { drawList(); scheduleSave(); });
        const refreshPlan = async () => {
          const next = await get(route('/correction'));
          if (!list.isConnected) return;
          const unitsChanged = JSON.stringify(plan.units) !== JSON.stringify(next.units);
          const covarsChanged = JSON.stringify(plan.covariate_options) !== JSON.stringify(next.covariate_options);
          plan = next;
          if (unitsChanged) {
            const selectedUnit = unit.value;
            fill(unit, ...plan.units.map(item => node('option', {value:item.column, text:(item.column === '(use batch column as-is)' ? 'ROI' : item.name) + ' (' + item.units + ')'})));
            unit.value = plan.units.some(item => item.column === selectedUnit) ? selectedUnit : plan.unit;
          }
          if (covarsChanged) {
            const removed = Array.from(keep).filter(name => !plan.covariate_options.includes(name));
            removed.forEach(name => keep.delete(name)); drawProtect();
            if (removed.length) scheduleSave();
          }
          drawList();
          get(route('/step/tab6')).then(value => showOutputs(current, value)).catch(() => {});
        };
        current.refreshItems = refreshPlan;
        followJobs(current, ['correct_batches'], refreshPlan);
        get(route('/step/tab6')).then(value => showOutputs(current, value)).catch(() => {});
        // ---- the bottom bar
        run.addEventListener('click', async () => {
          if (picked.size < 2) { status.textContent = 'Tick at least two tables (tick the header box for all).'; return; }
          if (!chosen.size) { status.textContent = 'Choose the markers to correct.'; return; }
          running = true; run.disabled = true;
          try {
            await persist();
            const {job} = await post(route('/correction/run'), {changes:changes(), markers:Array.from(chosen), rois:plan.rows.map(row => row.roi).filter(roi => picked.has(roi))});
            status.textContent = 'Correcting ' + picked.size + ' tables in the background (settings saved in Tab 6)…'; await refreshJobs();
            for (;;) {
              await new Promise(resolve => setTimeout(resolve, 2000));
              if (!body.isConnected) return;
              const now = ((await get(route('/jobs'))).jobs || []).find(item => item.id === job.id);
              if (now && /running|starting/.test(now.status)) { status.textContent = now.message || status.textContent; continue; }
              if (!now || now.status !== 'done') { status.textContent = now?.message || 'The correction stopped.'; break; }
              const done = now.result;
              const use = node('button', {class:'btn primary small', text:'Use this table in Studio', onclick:async () => {
                try { const used = await post(route('/correction/use'), {job_id:job.id}); status.textContent = used.table + ' is Studio’s combined table now.'; reload(); }
                catch (error) { status.textContent = error.message; }
              }});
              result.replaceChildren(node('div', {class:'studio-import'}, [
                node('div', {class:'row wrap'}, [chip('Pooled EMD reduction ' + done.emd_reduction.toFixed(2)),
                  chip('Relative MAD change ' + done.mad.toFixed(2), '', 'Change in pooled batch spread. This does not validate biological preservation.'),
                  chip(done.units.length + ' ' + (done.unit === '(use batch column as-is)' ? 'ROIs' : done.unit + 's'))]),
                node('p', {class:'muted', text:
                  done.corrected_cells.toLocaleString() + ' of ' + done.cells.toLocaleString() + ' cells corrected (' + done.markers.length + ' markers), saved as ' + done.name + '.'}),
                done.warnings?.length ? node('details', {class:'studio-options'}, [node('summary', {text:'Run notes (' + done.warnings.length + ')'}),
                  ...done.warnings.map(text => node('p', {class:'hint', text}))]) : null,
                node('div', {class:'row'}, [use])]));
              await refreshPlan();
              status.textContent = 'Done.';
              break;
            }
          } catch (error) { status.textContent = error.message; }
          finally { running = false; drawList(); }
        });
        current.runBar([node('div', {class:'row wrap'}, [node('span', {class:'lbl', text:'Correct:'}), run,
          help('Corrects the ticked tables together, as a job (tick the header box for all). Batch correction compares tables with each other, ' +
            'so it needs at least two units to compare and has no single-image run. Writes a new combined table beside Tab 6’s; nothing is replaced.')]),
          result, status]);
      })();
    };

    // Each marker's distribution in each unit (slide or ROI), on one before/after scale.
    // Curves can differ for biological as well as technical reasons. Clicking a unit in the legend picks it out.
    const histogramFigure = found => {
      const hist = found.histograms, names = Object.keys(hist?.markers || {});
      if (!names.length) return null;
      const weakest = found.per_marker.slice().sort((a, b) => a.emd_reduction - b.emd_reduction)[0]?.marker;
      const marker = choice(names.map(name => [name, name]), names.includes(weakest) ? weakest : names[0], 'Histogram marker');
      const picked = new Set(), plots = node('div', {class:'studio-histograms'}), legend = node('div', {class:'row wrap studio-hist-legend'});
      const colourOf = index => PALETTE[index % PALETTE.length];
      const W = 300, H = 150, L = 8, B = 24, T = 20;
      const panel = (title, rows, range, top) => {
        const x = i => L + (W - 2 * L) * i / hist.bins, y = v => T + (H - T - B) * (1 - v / top);
        return svg('svg', {viewBox:'0 0 ' + W + ' ' + H, class:'studio-hist', role:'img', 'aria-label':title + ': ' + marker.value}, [
          svg('text', {x:L, y:13, class:'studio-hist-title', text:title}),
          svg('line', {x1:L, x2:W - L, y1:H - B, y2:H - B, class:'studio-hist-axis'}),
          ...rows.map((row, index) => svg('polyline', {points:row.map((v, i) => x(i + 0.5).toFixed(1) + ',' + y(v).toFixed(1)).join(' '),
            fill:'none', stroke:colourOf(index), 'stroke-width':picked.has(hist.units[index]) ? '2.4' : '1.3',
            opacity:picked.size && !picked.has(hist.units[index]) ? '0.12' : '0.9'}, [svg('title', {text:hist.units[index]})])),
          svg('text', {x:L, y:H - 7, class:'studio-hist-tick', text:range[0].toFixed(1)}),
          svg('text', {x:W - L, y:H - 7, 'text-anchor':'end', class:'studio-hist-tick', text:range[1].toFixed(1)}),
          svg('text', {x:W / 2, y:H - 7, 'text-anchor':'middle', class:'studio-hist-tick', text:'arcsinh(' + marker.value + ' / ' + found.cofactor + ')'})]);
      };
      const draw = () => {
        const item = hist.markers[marker.value], top = Math.max(1e-6, ...item.before.flat(), ...item.after.flat());
        plots.replaceChildren(panel('As measured', item.before, item.range, top), panel('Corrected', item.after, item.range, top));
        legend.replaceChildren(...hist.units.map((unit, index) => node('button', {type:'button', class:'studio-hist-unit' + (picked.has(unit) ? ' picked' : ''),
          'aria-pressed':String(picked.has(unit)), title:'Pick out ' + unit, onclick:() => { if (picked.has(unit)) picked.delete(unit); else picked.add(unit); draw(); }},
          [node('span', {class:'studio-hist-swatch', style:'background:' + colourOf(index)}), node('span', {text:unit})])));
      };
      marker.addEventListener('change', draw);
      // Save figure: both panels as one SVG, with their styles, for a report.
      const save = node('button', {class:'btn small ghost', text:'Save figure', onclick:() => {
        const both = svg('svg', {xmlns:SVG, viewBox:'0 0 ' + 2 * W + ' ' + H, width:String(4 * W), height:String(2 * H)}, [
          svg('style', {text:'text{font:11px sans-serif;fill:#333}.studio-hist-title{font-weight:600}.studio-hist-axis{stroke:#999}'}),
          svg('rect', {width:String(2 * W), height:String(H), fill:'#fff'})]);
        plots.querySelectorAll('svg').forEach((one, index) => both.append(svg('g', {transform:'translate(' + index * W + ',0)'}, Array.from(one.childNodes).map(child => child.cloneNode(true)))));
        const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(both)], {type:'image/svg+xml'}));
        const link = node('a', {href:url, download:'batch_histograms_' + marker.value + '.svg'}); document.body.append(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      }});
      draw();
      const {box, body} = section('Marker distributions', {actions:[save], tip:'Each line is one ' + (found.measured_by === found.batch_column ? 'ROI' : found.measured_by) +
        '’s cells for the marker: the share of its sampled cells at each value (arcsinh), as measured and corrected, on the same scale. ' +
        'Biological differences in cell composition can remain after correction. Click a name below to pick it out; click a marker in Distribution checks to show it here.'});
      body.append(field('Marker', marker), plots, legend);
      box.setMarker = name => { if (hist.markers[name]) { marker.value = name; draw(); } };
      return box;
    };

    TOOLS.tab6 = current => {
      const tablesSec = section('Folders and tables', {tip:'One row per ROI: its cell table and its slide. ' +
        'Ticked tables are corrected together; all are ticked at first. Click a name to open its image.'});
      const correct = section('Settings', {tip:'Choose the technical batch to correct between and the markers to correct. ' +
        'Keep biological differences tells the model which sample groups to account for. Open How to assign the data for the steps. Settings are shared with classic.'});
      const about = section('Combined table', {tip:'The combined table Studio uses: every ROI’s cells in one table, with each marker ' +
        'as measured and corrected (<marker>_corr, or the ending chosen in Advanced).'});
      const checks = section('Distribution checks', {tip:'Pooled diagnostics on approximately 300 cells per ROI, comparing the same cells before and after. ' +
        'EMD reduction measures closer pooled distributions. Relative MAD change measures pooled spread changes. These differ from cyCombine’s ' +
        'population-level benchmark scores and cannot establish preservation of biology. Click a marker to colour the cells ' +
        'by it, corrected (it is added to Cells and maps in Layers), and to show its histograms.'});
      const sheet = section('Samples', {tip:'One row per ROI. The slide each ROI was cut from comes from the ROI step. ' +
        'Add patient, treatment, timepoint and response by ROI or a sample ID recorded in the cell tables. ' +
        'Match by slide only for values shared by every sample on that slide. All imported columns are saved in new combined outputs, even when unticked under Keep biological differences.'});
      // Keep optional review panels compact; their headings retain the shared help controls.
      [about, checks, sheet].forEach(part => {
        const heading = part.box.querySelector('.studio-sec-h');
        heading.style.cssText = 'display:inline-flex; width:calc(100% - 18px); margin-bottom:0; vertical-align:middle';
        const details = node('details', {class:'studio-options'}, [node('summary', {}, [heading]), part.body]);
        part.box.replaceChildren(details);
      });
      let samplesLoaded = false;
      sheet.box.querySelector('details').addEventListener('toggle', () => {
        if (sheet.box.querySelector('details').open && !samplesLoaded) { samplesLoaded = true; samplesPanel(sheet.body, () => current.refreshItems?.()); }
      });
      correctionPanel(tablesSec.body, correct.body, current, () => document.querySelector('[data-step=tab6]')?.click());
      current.body.append(tablesSec.box, freshnessNote(['tagging', 'combined']), correct.box, about.box, checks.box, sheet.box);
      about.body.append(node('p', {class:'muted', text:'Reading the saved result…'}));
      let checksLoaded = false;
      const loadChecks = async () => {
        if (checksLoaded) return; checksLoaded = true;
        let found;
        try { found = await get(route('/batch_checks')); }
        catch (error) { checksLoaded = false; about.body.replaceChildren(node('p', {class:'hint', text:error.message})); checks.body.replaceChildren(node('p', {class:'muted', text:error.message})); return; }
        if (!about.box.isConnected) return;
        const pair = (label, value) => node('div', {class:'row field'}, [node('label', {class:'lbl', text:label}), node('span', {class:'grow studio-value', text:value, title:value})]);
        about.body.replaceChildren(pair('Table', found.table), pair('Cells', found.cells.toLocaleString() + ' in ' + found.batches.length + ' batches'),
          pair('Batches by', found.batch_column),
          pair('Measured between', found.measured_by === found.batch_column ? 'ROIs' : found.measured_by + 's (' + new Set(Object.values(found.unit_of)).size + ')'),
          pair('Method', found.method + ' · arcsinh cofactor ' + found.cofactor),
          pair('Corrected', found.markers.length + ' markers'));
        let distributions = null;
        const paint = marker => { if (cellsState()) colourCells(marker + (found.suffix || '_corr')); else showResult({message:'Open an image with its cells to colour them by ' + marker + '.'}); };
        const markerRows = found.per_marker.slice().sort((a, b) => a.emd_reduction - b.emd_reduction).map(item => node('button',
          {type:'button', class:'studio-measure', title:'Colour the cells by ' + item.marker + ', corrected, and show its histograms',
            onclick:() => { paint(item.marker); distributions?.setMarker(item.marker); }}, [
          node('span', {class:'grow', text:item.marker}),
          node('span', {class:'studio-bar'}, [node('div', {style:'width:' + Math.round(100 * Math.max(0, item.emd_reduction)) + '%'})]),
          node('span', {class:'mono', text:item.emd_reduction.toFixed(2)}),
          node('span', {class:'mono', text:item.mad.toFixed(2)})]));
        checks.body.replaceChildren(node('div', {class:'row wrap'}, [chip('Pooled EMD reduction ' + found.overall.emd_reduction.toFixed(2)),
          chip('Relative MAD change ' + found.overall.mad.toFixed(2))]),
          node('div', {class:'studio-measure head'}, [node('span', {class:'grow', text:'Marker'}), node('span', {text:'Pooled EMD'}), node('span', {text:''}), node('span', {text:'Δ MAD'})]),
          node('div', {class:'studio-measures'}, markerRows),
          node('p', {class:'muted', text:'Sampled ' + found.sampled.toLocaleString() + ' cells · ' + found.ms + ' ms. Batch maps before and after are beside the image; click a batch to open it.'}));
        // Each marker compared across the batches; clicking a batch opens it (a batch is an ROI).
        const baseline = found.heatmap.before;
        const compared = matrix => {
          const columns = baseline[0] ? baseline[0].length : 0;
          const stats = Array.from({length:columns}, (_, column) => {
            const values = baseline.map(row => row[column]).filter(Number.isFinite);
            const mean = values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
            const spread = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(values.length, 1)) || 1;
            return [mean, spread];
          });
          return matrix.map(row => row.map((value, column) => Number.isFinite(value) ? (value - stats[column][0]) / stats[column][1] : 0));
        };
        const openBatch = async picked => {
          if (!picked.length) return;
          const name = picked[picked.length - 1];
          let list = [];
          try { list = await loadImages(); } catch (_) { /* the message below says it */ }
          const image = list.find(item => item.sample === name) || list.find(item => item.name === name || item.name.startsWith(name + '.'));
          if (image) chooseImage(image); else showResult({message:'No image of ' + name + ' in this project.'});
        };
        [['after', 'Batches after correction'], ['before', 'Batches before correction']].forEach(([when, title]) => heatmapFigure({
          key:'batches-' + when, title, onPick:openBatch,
          profile:{rows:found.heatmap.batches, columns:found.heatmap.markers, values:compared(found.heatmap[when]), raw:found.heatmap[when],
            note:'Marker medians; before and after use the same pre-correction centering and scale'},
          tip:'Each row is an ROI, each column a marker. Both panels use the pre-correction scale. Biological composition can cause real differences between rows; identical rows are not a validation target. Click an ROI to open it.'}));
        distributions = histogramFigure(found);
        if (distributions) showFigure('batches-histograms', distributions);
      };
      [about, checks].forEach(part => part.box.querySelector('details').addEventListener('toggle', () => {
        if (part.box.querySelector('details').open) loadChecks();
      }));
    };

    // ------------------------------------------------------------ Registration
    TOOLS.tab1 = current => {
      const compare = section('Compare on the image', {tip:'For a registered image (sample_merged): show the H&E over the ' +
        'CODEX markers and move the swipe line; nuclei should sit on the H&E\u2019s dark nuclei.'});
      const holder = node('div', {id:'studio-compare'});
      compare.body.append(holder);
      const draw = () => {
        const now = heCompare();
        if (!now) { holder.replaceChildren(node('p', {class:'muted', text:'Click a pair in the list to see its H&E over its CODEX, or open a registered image (sample_merged) to check it.'})); return; }
        const show = node('input', {type:'checkbox', 'aria-label':'Show the H&E'}); show.checked = now.visible;
        show.addEventListener('change', () => heCompare({visible:show.checked}));
        const opacity = node('input', {type:'range', min:'0', max:'100', value:String(Math.round(now.opacity * 100)), 'aria-label':'H&E opacity'});
        opacity.addEventListener('input', () => { heCompare({opacity:opacity.value / 100, visible:true}); show.checked = true; });
        const swipe = node('input', {type:'range', min:'0', max:'100', value:String(Math.round(now.swipe * 100)), 'aria-label':'Swipe'});
        swipe.addEventListener('input', () => { heCompare({swipe:swipe.value / 100, visible:true}); show.checked = true; });
        const scoreNote = node('p', {class:'muted', role:'status'});
        const score = node('button', {class:'btn small', text:'Check the alignment', onclick:async () => {
          score.disabled = true; scoreNote.textContent = 'Comparing DAPI with the H&E nuclei…';
          try {
            const found = await get(route('/registration_quality'));
            scoreNote.className = found.verdict === 'good' ? 'muted' : 'hint';
            scoreNote.textContent = (found.verdict === 'good' ? 'Looks aligned: ' : 'Worth a look: ') + 'match ' + found.correlation.toFixed(2) +
              ', best match ' + found.shift_um.toFixed(1) + ' µm away.';
          } catch (error) { scoreNote.className = 'hint'; scoreNote.textContent = error.message; }
          finally { score.disabled = false; }
        }});
        const checker = choice([['0', 'Off'], ['64', 'Small squares'], ['256', 'Medium squares'], ['1024', 'Large squares']], String(now.checker || 0), 'Checkerboard');
        checker.addEventListener('change', () => { heCompare({checker:Number(checker.value), visible:true}); show.checked = true; if (Number(checker.value)) swipe.value = '100'; });
        // A pair's raw H&E over its CODEX: turn it, scale it by the pixel sizes, move it; its turn is saved with the pair.
        const align = now.align;
        let alignRows = [];
        if (align) {
          const turn = node('input', {type:'range', min:'-180', max:'180', step:'0.5', value:String(align.rotation), 'aria-label':'Turn the H&E', class:'grow'});
          const degrees = node('input', {type:'number', min:'-180', max:'180', step:'0.5', value:String(align.rotation), 'aria-label':'H&E turn in degrees'});
          let saving;
          const setTurn = value => {
            heCompare({rotation:value, visible:true}); show.checked = true; turn.value = degrees.value = String(value);
            clearTimeout(saving);
            saving = setTimeout(() => post(route('/registration_rotation'), {sample:align.sample, rotation:Number(value)})
              .then(() => { scoreNote.textContent = 'Rotation ' + value + '° saved for ' + align.sample + '; used when it is registered.'; })
              .catch(error => { scoreNote.textContent = error.message; }), 600);
          };
          turn.addEventListener('input', () => setTurn(Number(turn.value)));
          degrees.addEventListener('change', () => setTurn(Number(degrees.value) || 0));
          const quarter = step => node('button', {class:'btn small ghost', text:(step > 0 ? '+' : '') + step + '°', onclick:() => setTurn(((Number(turn.value) + step + 540) % 360) - 180)});
          const scaled = node('input', {type:'checkbox', 'aria-label':'Scale the H&E by the pixel sizes'}); scaled.checked = align.scaled;
          scaled.addEventListener('change', () => heCompare({scaled:scaled.checked}));
          const move = node('input', {type:'checkbox', 'aria-label':'Move the H&E'}); move.checked = now.move;
          move.addEventListener('change', () => heCompare({move:move.checked}));
          const reset = node('button', {class:'btn small ghost', text:'Start again', onclick:() => { heCompare({reset:true, move:false}); setTurn(0); draw(); }});
          // Quick align: palom's coarse matching on small copies places the H&E in seconds (registration then refines it).
          const quick = node('button', {class:'btn small', text:'Quick align', title:'Place the H&E on the CODEX from small copies of both (seconds)', onclick:async () => {
            quick.disabled = true; scoreNote.textContent = 'Matching small copies of the H&E and the CODEX…';
            try {
              const found = await post(route('/registration_quick_align'), {sample:align.sample});
              heCompare({place:{x:found.x, y:found.y, rotation:found.rotation, zoom:found.zoom}, visible:true});
              setTurn(found.rotation);
              scoreNote.textContent = 'Placed: turned ' + found.rotation + '°' + (found.pixel_sizes_ok ? '.' :
                '; its size is ' + found.zoom + '× what the pixel sizes say: check them in Markers.');
            } catch (error) { scoreNote.textContent = error.message; }
            finally { quick.disabled = false; }
          }});
          // Rotate CODEX: the whole view (CODEX and H&E together), the same slider, box and ±90° as the H&E.
          const viewTurn = Math.round(((now.codexRotation || 0) + 540) % 360 - 180);
          const codexSlider = node('input', {type:'range', min:'-180', max:'180', step:'1', value:String(viewTurn), 'aria-label':'Rotate the CODEX view', class:'grow'});
          const codexDegrees = node('input', {type:'number', min:'-180', max:'180', step:'1', value:String(viewTurn), 'aria-label':'CODEX view rotation in degrees'});
          const setView = value => { const turned = ((Number(value) || 0) + 540) % 360 - 180; heCompare({codexRotation:turned}); codexSlider.value = codexDegrees.value = String(turned); };
          codexSlider.addEventListener('input', () => setView(codexSlider.value));
          codexDegrees.addEventListener('change', () => setView(codexDegrees.value));
          const viewQuarter = step => node('button', {class:'btn small ghost', text:(step > 0 ? '+' : '') + step + '°', onclick:() => setView(Number(codexSlider.value) + step)});
          alignRows = [
            field('Rotate H&E', node('div', {class:'inline grow'}, [quarter(-90), turn, degrees, quarter(90)]), 'Rotate the H&E to match the CODEX (any angle). Saved with the pair and used as the H&E\u2019s rotation when it is registered.'),
            field('Rotate CODEX', node('div', {class:'inline grow'}, [viewQuarter(-90), codexSlider, codexDegrees, viewQuarter(90)]),
              'Rotates the whole view (CODEX and H&E together), to look at it the usual way round. The registered image keeps the CODEX as it is.'),
            field('Scale', node('div', {class:'inline'}, [scaled, node('span', {class:'muted', text:'by the pixel sizes (' + align.he_mpp + ' → ' + align.codex_mpp + ' µm/px)'})]),
              'Shows the H&E at the CODEX\u2019s scale, from the two pixel sizes (Markers). Untick to see it at its own pixel size.'),
            field('Quick align', node('div', {class:'inline'}, [quick, help('Finds how the H&E is turned, sized and shifted against the CODEX (their anchor marker and nuclei) ' +
              'on small copies, in seconds, and places it. Its turn is saved with the pair; registering then aligns it exactly.')]), ''),
            field('Move H&E', node('div', {class:'inline'}, [move, node('span', {class:'muted', text:'drag on the image, or arrow keys (Shift: further)'}), reset]),
              'While ticked, dragging moves the H&E instead of the view. Palom does the exact alignment when registering; this checks the pair and its turn.'),
          ];
        }
        fill(holder, field('H&E', node('div', {class:'inline'}, [show, opacity]), 'Tick to show the H&E; the slider sets how much it covers the markers.'),
          ...alignRows,
          field('Swipe', swipe, 'The H&E shows left of the line and the CODEX markers right of it.'),
          field('Checkerboard', checker, 'H&E and CODEX in alternating squares: along each square’s edge, tissue should continue without a jump.'),
          align ? scoreNote : node('div', {class:'row'}, [score, help('Compares the DAPI nuclei with the H&E nuclei on a small copy of the image, inside the tissue: ' +
            'how alike they are (1 is a perfect match) and how far one would have to move to match best (near 0 µm when the registration is right).')]), align ? null : scoreNote);
      };
      current.drawCompare = draw;
      const settings = section('Settings for new pairs', {tip:'The number of keypoints palom matches. Each pair\u2019s H&E rotation is set in ' +
        'Compare on the image, and its anchor marker in Markers.'});
      const keypoints = node('input', {type:'number', min:'500', max:'100000', step:'500', value:'10000', 'aria-label':'Keypoints'});
      settings.body.append(field('Keypoints', keypoints, 'Image features used for registration. More features can improve matching but take longer.'));
      // The pairs: Tab 1's folders, naming rule and per-image choice, shared with classic Tab 1.
      const pairing = section('Pairs', {tip:'Which H&E image is registered to which CODEX image. Tab 1 finds pairs by name: it ' +
        'strips the naming rule’s prefix and suffix from each file name and pairs what remains (sample_1_codex and sample_1_he both become ' +
        'sample_1). When names do not match, pick the H&E for a CODEX image by hand, or Skip it. Saved as Tab 1’s choices; classic Tab 1 uses them too.'});
      const picked = new Set();
      let found = null, chosenPair = null, addingCodex = false;
      const pairNote = node('p', {class:'muted', role:'status'});
      // One save at a time: a change made while another saves waits for it, so it carries the new revision.
      let pairSaving = Promise.resolve();
      const savePairs = changes => (pairSaving = pairSaving.then(() => savePairsNow(changes)));
      const savePairsNow = async changes => {
        const folder = changes.folders || changes.extra_folders;
        pairNote.textContent = folder ? 'Reading the new folder…' : 'Saving…';
        pairing.box.classList.add('studio-working');
        try {
          const answer = await post(route('/registration_pairs'), {changes, revision:found.revision});
          drawPairs(answer);
          pairNote.textContent = (folder ? 'Folder changed. ' : 'Saved; classic Tab 1 uses the same pairs. ') +
            answer.rows.filter(row => row.he).length + ' pairs · ' + answer.rows.filter(row => row.registered).length + ' registered.';
        } catch (error) { pairNote.textContent = error.message; await loadPairs(); }
        finally { pairing.box.classList.remove('studio-working'); }
      };
      // A folder is its whole path, typed as in classic Tab 1 (inside the project or not).
      const folderBox = (kind, label, tip) => {
        const input = node('input', {type:'text', value:found.folders[kind], maxlength:'1024', class:'grow', 'aria-label':(kind === 'codex' ? 'CODEX' : label) + ' folder'});
        input.addEventListener('change', () => { if (input.value.trim() && input.value.trim() !== found.folders[kind]) savePairs({folders:{[kind]:input.value.trim()}}); });
        if (kind !== 'codex') return field(label, input, tip);
        const plus = node('button', {class:'btn small ghost', text:'+', 'aria-label':'Add a CODEX folder',
          title:'Add another CODEX folder, to register several CODEX images of the same sample together (each aligned to CODEX (1) by its anchor)',
          onclick:() => { addingCodex = true; drawPairs(found); pairing.body.querySelector('[aria-label="Add CODEX folder"]')?.focus(); }});
        return field(label, node('div', {class:'inline grow'}, [input, plus]), tip);
      };
      // More CODEX folders (rare): each is a column, CODEX (2), (3)..., aligned to CODEX (1) by its anchor.
      const extraBoxes = () => {
        const folders = found.extra_folders;
        const save = list => savePairs({extra_folders:list});
        const boxes = folders.map((folder, k) => {
          const input = node('input', {type:'text', value:folder, maxlength:'1024', class:'grow', 'aria-label':'CODEX (' + (k + 2) + ') folder'});
          input.addEventListener('change', () => { const list = [...folders]; list[k] = input.value.trim(); if (list[k]) save(list); });
          const remove = node('button', {class:'btn small ghost', text:'✕', title:'Remove this CODEX folder', onclick:() => save(folders.filter((_, j) => j !== k))});
          return field('CODEX (' + (k + 2) + ')', node('div', {class:'inline grow'}, [input, remove]), 'Another CODEX folder: its image for each sample is aligned to CODEX (1) and merged in.');
        });
        if (!addingCodex) return boxes;
        const add = node('input', {type:'text', maxlength:'1024', placeholder:'Path of the next CODEX folder', 'aria-label':'Add CODEX folder', class:'grow'});
        add.addEventListener('change', () => { if (add.value.trim()) { addingCodex = false; save([...folders, add.value.trim()]); } });
        const cancel = node('button', {class:'btn small ghost', text:'✕', title:'Cancel', onclick:() => { addingCodex = false; drawPairs(found); }});
        return [...boxes, field('CODEX (' + (folders.length + 2) + ')', node('div', {class:'inline grow'}, [add, cancel]),
          'Another CODEX folder: its image for each sample is aligned to CODEX (1) and merged in.')];
      };
      const drawPairs = answer => {
        found = answer;
        const many = found.extra_folders.length > 0;
        if (!chosenPair || !found.rows.some(row => row.sample === chosenPair)) {
          chosenPair = found.rows[0]?.sample || null;
          setTimeout(drawMarkers);  // the first pair's layers, once the list is there
        }
        const rule = Object.fromEntries(Object.entries(found.rule).map(([key, value]) => {
          const input = node('input', {type:'text', value, maxlength:'64', 'aria-label':key.replace('tab1_batch_', '').replace('_', ' ')});
          input.addEventListener('change', () => savePairs({[key]:input.value}));
          return [key, input];
        }));
        const he = found.he_files;
        const head = pickAll(found.rows.filter(row => row.he).map(row => row.sample), picked, () => drawPairs(found));
        const rows = found.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.sample}); tick.checked = picked.has(row.sample);
          tick.disabled = !row.he;
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.sample) : picked.delete(row.sample); head.sync(); });
          const auto = row.auto ? 'Auto: ' + row.auto : 'Auto: no match';
          const partner = choice([['<Auto>', auto], ['<Skip>', 'Skip this CODEX image'], ...he.map(name => [name, name])], row.choice, 'H&E for ' + row.codex);
          partner.addEventListener('change', () => savePairs({choices:{[row.codex]:partner.value}}));
          const state = workingChip(current, row.sample) || (row.duplicate ? chip('Same sample as another', 'warn', 'Another CODEX image gives the sample name ' + row.sample +
              ', so both would write ' + row.output + '. Skip one, or change the naming rule.') :
            row.registered ? chip('Registered', 'ok', row.output) : row.he ? chip('Not registered', '', row.output) :
            chip(row.choice === '<Skip>' ? 'Skipped' : 'No H&E', 'warn', row.follows_rule ? 'No H&E has this sample name: pick one.' :
              'This name does not follow the CODEX naming rule: pick its H&E, or change the rule.'));
          // A name opens its image in the viewer (with its markers in Layers).
          const select = () => {
            if (chosenPair === row.sample) return;
            chosenPair = row.sample; pairing.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
            pairing.body.querySelector('tr[data-sample="' + CSS.escape(row.sample) + '"]')?.classList.add('selected'); drawMarkers();
          };
          // The row: this pair's CODEX with its H&E over it, to align by eye (Compare on the image).
          const compare = async () => {
            select();
            if (!row.he) { pairNote.textContent = row.sample + ' has no H&E: pick one first.'; return; }
            pairNote.textContent = 'Opening ' + row.sample + '…';
            try { const answer = await runCommand('compare_pair', {sample:row.sample}); showResult(answer); pairNote.textContent = answer.message; }
            catch (error) { pairNote.textContent = error.message; }
          };
          // A name selects its pair (its layers in Markers) and opens its image.
          const opener = (text, image, what) => image ? node('button', {type:'button', class:'studio-item-name studio-file', text, title:'Open ' + what,
            onclick:() => { select(); chooseImage(image); }}) : node('span', {class:'studio-file', text, title:text});
          const matches = row.registered ? node('button', {class:'btn small ghost', text:'Matches', title:'Show the points palom matched ' +
            'between CODEX and H&E for this pair, joined by lines', onclick:() => showMatches(row.sample)}) : null;
          const line = node('tr', {'data-sample':row.sample, class:row.sample === chosenPair ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) compare(); });
          fill(line, node('td', {}, [tick]),
            node('td', {}, [opener(row.sample, row.registered && row.images.merged, 'the registered image ' + row.output)]),
            node('td', {}, [opener(row.codex, row.images.codex, 'the CODEX image')]),
            ...row.extra.map((item, k) => {
              const pick = choice([['<Auto>', item.auto ? 'Auto: ' + item.auto : 'Auto: no match'], ['<Skip>', 'None'],
                ...found.extra_files[k].map(name => [name, name])], item.choice, 'CODEX (' + (k + 2) + ') for ' + row.codex);
              pick.addEventListener('change', () => savePairs({extra_choices:{[row.codex]:row.extra.map((other, j) => j === k ? pick.value : other.choice)}}));
              return node('td', {}, [node('div', {class:'inline'}, [item.file ? opener(item.file, row.extra_images[k], 'this CODEX image') : null, pick])]);
            }),
            node('td', {}, [node('div', {class:'inline'}, [row.he ? opener(row.he, row.images.he, 'the H&E image') : null, partner])]),
            node('td', {}, [state, row.edited ? node('span', {class:'studio-edited', text:' ✎', title:'This pair\u2019s layers were edited in Markers'}) : null]),
            node('td', {}, [matches]));
          return line;
        });
        const leftovers = (names, text) => names.length ? node('p', {class:'hint', text:text + names.join(', ')}) : null;
        fill(pairing.body,
          folderBox('codex', many ? 'CODEX (1)' : 'CODEX', 'The folder with the CODEX images (Tab 1’s CODEX directory). Type its path and press Enter.'),
          ...extraBoxes(),
          folderBox('he', 'H&E', 'The folder with the H&E images (Tab 1’s H&E directory). Type its path and press Enter.'),
          folderBox('registered', 'Results', 'Where each pair’s merged image (sample_merged.ome.tiff) is written; a new folder is made.'),
          field('CODEX name', node('div', {class:'inline'}, [rule.tab1_batch_codex_prefix, rule.tab1_batch_codex_suffix]),
            'Text removed from the start (prefix) and end (suffix) of each CODEX file name before matching; e.g. CODEX_ and _codex.'),
          field('H&E name', node('div', {class:'inline'}, [rule.tab1_batch_he_prefix, rule.tab1_batch_he_suffix]),
            'The same for the H&E names; e.g. suffix -HE pairs CRC02-HE with CRC02. Empty accepts every file the CODEX rule does not.'),
          node('div', {class:'row'}, [chip(found.rows.filter(row => row.he).length + ' pairs'), chip(found.rows.filter(row => row.registered).length + ' registered', 'ok')]),
          found.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'}, [node('tr', {}, [node('th', {}, [head]), ...['Sample', many ? 'CODEX (1)' : 'CODEX', ...found.extra_folders.map((_, k) => 'CODEX (' + (k + 2) + ')'), 'H&E', 'State', ''].map(text => node('th', {text}))]), ...rows])])
            : node('p', {class:'muted', text:found.found.codex ? 'No CODEX image follows the naming rule.' : 'The CODEX folder is not there: choose it above.'}),
          leftovers(found.unpaired_he, 'H&E images without a CODEX image: '),
          leftovers(found.ignored_codex, 'In the CODEX folder but taken as H&E images by the naming rule: '),
          leftovers(found.ignored_he, 'In the H&E folder but taken as CODEX images by the naming rule: '),
          pairNote);
      };
      // The coarse alignment's matched points (the Registration notebook's figure), beside the image.
      const showMatches = async sample => {
        const {box, body} = section('Matched points · ' + sample, {tip:'palom\u2019s coarse alignment: points it found on the ' +
          'CODEX reference (left) and on the H&E (right); each line joins a pair it kept. Many lines running in parallel mean a ' +
          'sound alignment; few or crossing lines mean it is worth checking the pair on the image.'});
        body.append(node('p', {class:'muted', text:'Loading…'}));
        showFigure('registration-matches', box);
        try {
          const found = await get(route('/registration_figure?sample=' + encodeURIComponent(sample)));
          body.replaceChildren(found.image ? node('img', {src:found.image, alt:'Matched points for ' + sample, class:'studio-figure-preview'}) :
            node('p', {class:'muted', text:'No figure for this pair: it is saved when a pair is registered with this version ' +
              '(Register picked again makes one).'}));
        } catch (error) { body.replaceChildren(node('p', {class:'hint', text:error.message})); }
      };
      const loadPairs = async () => {
        try { drawPairs(await get(route('/registration_pairs'))); }
        catch (error) { pairing.body.replaceChildren(node('p', {class:'muted', text:error.message})); }
      };
      followJobs(current, ['register_pairs'], loadPairs);
      // New marker names and colours for the registered images (every pair), as Tab 1's channel table sets them.
      // The chosen pair's layers in its registered image: each CODEX image's markers (keep, anchor, name, colour), then the H&E's.
      const markerBox = section('Markers', {tip:'The layers the chosen pair’s registered image gets. Untick a marker to leave it out, ' +
        'type a new name or pick a colour. The anchor is the marker each CODEX image is aligned by (DAPI by default); it is used for the ' +
        'alignment even when left out of the image. Click another pair in the list to see its layers; Save keeps this pair’s choices.'});
      let markersRequest = 0;
      const drawMarkers = async () => {
        if (!chosenPair) { fill(markerBox.body, node('p', {class:'muted', text:'Choose a pair in the list.'})); return; }
        const request = ++markersRequest;
        fill(markerBox.body, node('p', {class:'muted', text:'Reading ' + chosenPair + '’s layers…'}));
        let data;
        try { data = await get(route('/registration_markers?sample=' + encodeURIComponent(chosenPair))); }
        catch (error) { if (request === markersRequest) fill(markerBox.body, node('p', {class:'muted', text:error.message})); return; }
        if (request !== markersRequest) return;
        const note = node('p', {class:'muted', role:'status'}), warn = node('p', {class:'hint'});
        // Pixel sizes (µm per pixel): read from each file, or typed when a file has none or a wrong one.
        const magnification = mpp => mpp ? ' ≈ ' + Math.round(10 / mpp) + '×' : '';
        const mppBox = kind => {
          const item = data.mpp[kind];
          const input = node('input', {type:'number', min:'0.01', max:'50', step:'0.0001', value:String(item.used), 'aria-label':(kind === 'he' ? 'H&E' : 'CODEX') + ' pixel size'});
          const hint = node('span', {class:'muted', text:magnification(item.used) + (item.saved ? ' (typed)' : item.detected ? ' (from the file)' : ' (not in the file: Tab 1\u2019s default)')});
          input.addEventListener('input', () => { hint.textContent = magnification(Number(input.value)) + (Number(input.value) === item.detected ? ' (from the file)' : ' (typed)'); });
          return {kind, item, input, box:node('div', {class:'inline'}, [input, hint])};
        };
        const sizes = [mppBox('codex'), mppBox('he')];
        const panels = data.images.map((image, number) => {
          const isHe = image.label === 'H&E';
          const group = 'anchor-' + number;
          const rows = image.channels.map(channel => {
            const keep = node('input', {type:'checkbox', 'aria-label':'Keep ' + channel.name + ' (' + image.label + ')'}); keep.checked = channel.keep;
            const anchor = isHe ? null : node('input', {type:'radio', name:group, 'aria-label':'Anchor ' + channel.name + ' (' + image.label + ')', title:'Align by this marker'});
            if (anchor) anchor.checked = channel.name === image.anchor;
            const colour = node('input', {type:'color', value:channel.new_color || channel.color, 'aria-label':'Colour of ' + channel.name + ' (' + image.label + ')'});
            // The name is the box: type over it to rename the layer (empty goes back to the file's name).
            const name = node('input', {type:'text', value:channel.rename || channel.name, placeholder:channel.name, maxlength:'64', class:'grow',
              'aria-label':'Name of ' + channel.name + ' (' + image.label + ')', title:'In the file: ' + channel.name + '. Type to rename.'});
            [keep, colour, name].forEach(input => input.addEventListener(input === name ? 'input' : 'change', check));
            return {channel, keep, anchor, colour, name, row:node('div', {class:'row studio-roi-marker'}, [keep, anchor, colour, name,
              node('span', {class:'level-meta studio-meta', text:channel.dye, title:'Dye'}), node('span', {class:'level-meta studio-meta', text:channel.exposure, title:'Exposure (ms)'})])};
          });
          const all = node('input', {type:'checkbox', 'aria-label':'Select all of ' + image.label}); all.checked = rows.every(r => r.keep.checked);
          all.addEventListener('change', () => { rows.forEach(r => { r.keep.checked = all.checked; }); check(); });
          const head = node('div', {class:'row'}, [all, node('strong', {text:image.label}), node('span', {class:'muted grow studio-file', text:' ' + (image.file || ''), title:image.file || ''}),
            isHe ? null : node('span', {class:'muted', text:'Anchor ◉ · Dye · Exposure'})]);
          return {image, rows, box:node('div', {class:'studio-marker-panel'}, [head, node('div', {class:'studio-roi-markers'}, rows.map(r => r.row))])};
        });
        // Two kept layers with the same final name (a second DAPI, say): flag them to leave out or rename.
        function check() {
          const seen = {};
          panels.forEach(panel => panel.rows.forEach(r => { if (r.keep.checked) { const final = r.name.value.trim() || r.channel.name; (seen[final] ||= []).push(panel.image.label); } }));
          const twice = Object.entries(seen).filter(([, where]) => where.length > 1);
          warn.textContent = twice.length ? 'Same name twice: ' + twice.map(([name, where]) => name + ' (' + where.join(', ') + ')').join('; ') +
            '. Untick one or rename it.' : '';
        }
        check();
        const save = node('button', {class:'btn small', text:'Save ' + data.sample + '’s layers', onclick:async () => {
          const edits = side => ({
            keep:side.rows.filter(r => r.keep.checked).map(r => r.channel.index),
            rename:Object.fromEntries(side.rows.filter(r => r.name.value.trim() && r.name.value.trim() !== r.channel.name).map(r => [r.channel.index, r.name.value.trim()])),
            colors:Object.fromEntries(side.rows.filter(r => r.colour.value.toLowerCase() !== r.channel.color.toLowerCase()).map(r => [r.channel.index, r.colour.value])),
          });
          const codex = panels.filter(panel => panel.image.label !== 'H&E');
          const he = panels.find(panel => panel.image.label === 'H&E');
          try {
            const mpp = Object.fromEntries(sizes.filter(s => Number(s.input.value) > 0 && Number(s.input.value) !== s.item.detected).map(s => [s.kind, Number(s.input.value)]));
            await post(route('/registration_markers'), {sample:data.sample, edits:{mpp,
              images:codex.map(panel => ({...edits(panel), anchor:(panel.rows.find(r => r.anchor?.checked) || panel.rows[0]).channel.name})),
              he:edits(he)}});
            note.textContent = 'Saved for ' + data.sample + '; used when it is registered (again).';
            await loadPairs();
          } catch (error) { note.textContent = error.message; }
        }});
        fill(markerBox.body, node('p', {class:'muted', text:data.sample + (data.edited ? ' · edited ✎' : '')}),
          field('CODEX µm/px', sizes[0].box, 'The CODEX image\u2019s pixel size, read from its file. The H&E is scaled to it before aligning.'),
          field('H&E µm/px', sizes[1].box, 'The H&E image\u2019s pixel size, read from its file (0.25 µm/px is about 40×, 0.5 about 20×). Type it when the file has none or a wrong one.'),
          ...panels.map(panel => panel.box), warn,
          node('div', {class:'row'}, [save]), note);
      };
      current.body.append(pairing.box, compare.box, settings.box, markerBox.box);
      drawMarkers();
      current.drawCompare = () => { draw(); };
      current.drawCompare();
      loadPairs();
      const status = node('p', {class:'muted', role:'status'});
      const register = (button, scope, again = false) => async () => {
        const samples = Array.from(picked);
        if (scope === 'pairs' && !samples.length) { status.textContent = 'Tick the pairs to register in the Pairs list.'; return; }
        if (again && !window.confirm('Register ' + samples.join(', ') + ' again? Each earlier merged image is moved into the ' +
          'previous folder beside it, not deleted.')) return;
        button.disabled = true;
        try {
          const args = {scope, keypoints:Math.round(Number(keypoints.value) || 10000)};  // each pair's rotation is saved with it
          if (scope === 'pairs') Object.assign(args, {samples, again});
          const answer = await runCommand('register_pairs', args, {confirmed:true});
          showResult(answer); status.textContent = answer.message;
          if (answer.status === 'done') { picked.clear(); await refreshJobs(); showTab('jobs'); loadPairs(); }
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const one = node('button', {class:'btn', text:'Register this pair', title:'The pair of the open image'});
      const some = node('button', {class:'btn', text:'Register picked', title:'The ticked pairs not yet registered'});
      const again = node('button', {class:'btn ghost', text:'Register picked again', title:'Redo the ticked pairs; earlier results move into previous/'});
      const all = node('button', {class:'btn primary', text:'Register all', title:'Every pair not yet registered'});
      one.addEventListener('click', register(one, 'image')); some.addEventListener('click', register(some, 'pairs'));
      again.addEventListener('click', register(again, 'pairs', true)); all.addEventListener('click', register(all, 'all'));
      current.runBar([node('div', {class:'row'}, [one, some, again, all]), status],
        'Runs Tab 1\u2019s registration (palom) as a job; registered pairs are kept unless registered again. Open the merged image here to check it.');
    };

    // ------------------------------------------------------------ Background
    TOOLS.tab2 = current => {
      // Tab 2's two routes in one step: blank cycles subtracted from the markers they serve, and/or each marker's
      // background level (Auto, on the whole image or inside an annotation class); then kept markers, names, colours.
      const place = section('Folders and images', {tip:'Where the images are (registered, ROIs or raw QPTIFF), where their tissue ' +
        'annotations are (GeoJSON, optional), and where threshold tables and background-removed images are written. Click an image ' +
        'to open it with its annotation. ⚑ marks images with blank cycles.'});
      const method = section('Background of the open image', {tip:'Auto measures each marker’s background with Tab 2’s methods, ' +
        'on the whole image or inside the chosen annotation class (the same controls as the image card in Layers). Per marker: keep ' +
        'it in the new image, its colour and new name, its dye and exposure, the blank cycle subtracted from it (paired by dye, as ' +
        'classic Tab 2 does) and its level. Blank cycles are left out of the new image unless ticked.'});
      const holder = node('div', {id:'studio-background'});
      method.box.id='studio-background-detection';
      method.body.append(holder);
      const benchmark=node('div',{class:'studio-threshold-benchmark'});
      showFigure('threshold-benchmark',benchmark);
      current.thresholdBenchmark=hooks.renderThresholdBenchmark?.(benchmark);
      current.body.append(place.box, method.box);
      const note = node('p', {class:'muted', role:'status'});
      const picked = new Set();
      let plan = null, openImage = null, openAnnotation = null, markers = null, saving;
      const saveStep = async changes => {
        const folder = changes.folders ? Object.keys(changes.folders)[0] : null;
        note.textContent = folder ? 'Reading the new ' + folder + ' folder…' : 'Saving…';
        place.box.classList.add('studio-working');
        try {
          plan = await post(route('/background_images'), {changes}); drawImages();
          const matched = plan.rows.filter(row => row.annotation).length;
          note.textContent = (folder ? 'Folder changed. ' : 'Saved with the project. ') + plan.rows.length + ' images · ' + matched + ' with an annotation' +
            (plan.rows.some(row => row.blanks.length) ? ' · ' + plan.rows.filter(row => row.blanks.length).length + ' with blank cycles' : '') + '.';
        }
        catch (error) { note.textContent = error.message; }
        finally { place.box.classList.remove('studio-working'); }
      };
      const folderBox = (kind, label, tip) => {
        const input = node('input', {type:'text', value:plan.folders[kind], maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
        input.addEventListener('change', () => { if (input.value.trim() && input.value.trim() !== plan.folders[kind]) saveStep({folders:{[kind]:input.value.trim()}}); });
        return field(label, input, tip);
      };
      const openRow = async row => {
        openImage = row.image;
        place.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
        place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"]')?.classList.add('selected');
        note.textContent = 'Opening ' + row.image + '…';
        try {
          const answer = await runCommand('open_background_image', {image:row.image});
          showResult(answer); note.textContent = answer.message;
          openAnnotation = answer.data?.annotation || null;
          if (answer.data) setBackgroundScope(answer.data.annotation, answer.data.class);
        } catch (error) { note.textContent = error.message; }
      };
      // One class for many images: the ticked ones (all when none is ticked) that have it in their annotation.
      const classAll = () => {
        const names = [...new Set(plan.rows.flatMap(row => row.classes))].sort((a, b) => a.localeCompare(b));
        const select = choice([['', 'Class (set all)'], ...names.map(name => [name, name])], '', 'Class of the ticked images');
        select.title = 'Measure the background inside this class in the ticked images (all when none is ticked) that have it';
        select.addEventListener('change', () => {
          if (!select.value) return;
          const targets = plan.rows.filter(row => (!picked.size || picked.has(row.image)) && row.classes.includes(select.value));
          if (!targets.length) { note.textContent = 'No ticked image has ' + select.value + ' in its annotation.'; return; }
          saveStep({classes:Object.fromEntries(targets.map(row => [row.image, select.value]))});
          note.textContent = targets.length + ' image' + (targets.length === 1 ? '' : 's') + ': inside ' + select.value + '.';
        });
        return select;
      };
      const drawImages = () => {
        const head = pickAll(plan.rows.map(row => row.image), picked, drawImages);
        const rows = plan.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.image}); tick.checked = picked.has(row.image);
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.image) : picked.delete(row.image); head.sync(); });
          const file = choice([['', row.chosen ? 'By name' : 'By name: ' + (row.annotation || 'none')], ...plan.geojson_files.map(name => [name, name])],
            row.chosen, 'Annotation for ' + row.image);
          file.addEventListener('change', () => saveStep({matches:{[row.image]:file.value}}));
          const cls = row.classes.length ? choice(row.classes.map(name => [name, name]), row.class, 'Class for ' + row.image) : null;
          if (cls) cls.addEventListener('change', () => { saveStep({classes:{[row.image]:cls.value}}); if (openImage === row.image && openAnnotation) setBackgroundScope(openAnnotation, cls.value); });
          const state = node('span', {}, [workingChip(current, row.image), row.detected ? chip('Detected', '', 'Levels kept in the project') : null, row.levels ? chip('CSV', 'ok', 'Thresholds exported') : null,
            row.removed ? chip('Image', 'ok', 'Background-removed image written') : null, row.detected || row.levels || row.removed ? null : chip('Not run', '')]);
          const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:'Open ' + row.image, onclick:() => openRow(row)});
          const line = node('tr', {'data-image':row.image, class:row.image === openImage ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
          fill(line, node('td', {}, [tick]), node('td', {}, [name, row.blanks.length ? node('span', {class:'studio-edited', text:' ⚑', title:'Blank cycles: ' + row.blanks.join(', ') + '. They can be subtracted from the markers (Background of the open image).'}) : null]),
            node('td', {}, [file]), node('td', {}, [cls || node('span', {class:'muted', text:row.annotation ? '' : '—'})]),
            node('td', {}, [state, row.edited ? node('span', {class:'studio-edited', text:' ✎', title:'Markers edited'}) : null]));
          return line;
        });
        fill(place.body,
          folderBox('images', 'Images', 'The images to work on: registered, ROIs, or raw QPTIFF. Type the folder’s path and press Enter.'),
          folderBox('annotations', 'Annotations', 'Tissue annotations (GeoJSON) matched to the images by name, to measure the background inside a class (for example tissue). Optional.'),
          folderBox('tables', 'Thresholds', 'Where each image’s threshold table is saved (Tab 2’s format; RESOLVE reads it).'),
          folderBox('output', 'Output', 'Where background-removed images are written (<image>_bg_subtracted.ome.tiff).'),
          node('div', {class:'row'}, [chip(plan.rows.length + ' images'), chip(plan.rows.filter(row => row.blanks.length).length + ' with blanks', 'warn'),
            chip(plan.rows.filter(row => row.removed).length + ' removed', 'ok')]),
          plan.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'},
            [node('tr', {}, [node('th', {}, [head]), node('th', {text:'Image'}), node('th', {text:'Annotation'}), node('th', {}, [classAll()]), node('th', {text:'State'})]), ...rows])])
            : node('p', {class:'muted', text:'No image in this folder: type the images folder above.'}),
          note);
      };
      // The open image's markers (from the images folder) with the saved choices; levels are the viewer's.
      const loadMarkers = async (opened = false) => {
        if (!openImage) {
          // An image opened another way (Layers, another step): its markers when it is in the images folder, else its saved levels.
          let open = null;
          try { open = await get(route('/background_levels')); } catch (_) {}
          if (!plan) await planReady;  // the images list may still be on its way
          if (open && (plan?.rows || []).some(row => row.image === open.image)) openImage = open.image;
          else {
            current.saved = open || null;
            if (holder.isConnected) renderBackground(holder, current.saved); return;
          }
        }
        try {
          const found = await get(route('/background_images?image=' + encodeURIComponent(openImage)));
          plan = {...found, markers:undefined}; markers = found.markers; drawImages();
          current.thresholdBenchmark?.sources?.(found.measurement_annotation);
        } catch (error) { markers = null; note.textContent = error.message; }
        current.saved = markers ? {levels:markers.levels, table:markers.table, markers, onConfig:config => {
          clearTimeout(saving);
          saving = setTimeout(() => post(route('/background_images'), {changes:{image:openImage, markers:config}})
            .then(() => { note.textContent = 'Markers saved for ' + openImage + '.'; }).catch(error => { note.textContent = error.message; }), 500);
        }} : null;
        if (holder.isConnected) renderBackground(holder, current.saved);
      };
      current.readSaved = () => { openImage = null; return loadMarkers(true); };  // the image now open, however it was opened
      followJobs(current, ['detect_background', 'export_background', 'remove_background'],
        () => get(route('/background_images')).then(found => { plan = found; drawImages(); }).catch(() => {}));
      const planReady = get(route('/background_images')).then(found => { plan = found; drawImages(); loadMarkers(true); })
        .catch(error => fill(place.body, node('p', {class:'hint', text:error.message})));
      renderBackground(holder, null);
      const status = node('p', {class:'muted', role:'status'});
      const whichOpen = async () => { if (!openImage) await loadMarkers(); return openImage; };
      const chosen = async which => {
        if (which === 'open') { await whichOpen(); return openImage ? [openImage] : []; }
        return Array.from(picked);
      };
      const jobStarted = async result => { showResult(result); status.textContent = result.message; if (result.status === 'done') { await refreshJobs(); showTab('jobs'); } };
      // Detect: the viewer's Auto on the open image (it shows and keeps the levels); a job for the other ticked images.
      const detect = (button, which) => async () => {
        const images = await chosen(which);
        if (!images.length) { status.textContent = which === 'open' ? 'Open an image first.' : 'Tick the images first.'; return; }
        const method = backgroundLevels()?.method || 'Triangle';
        const others = images.filter(name => name !== openImage);
        button.disabled = true;
        try {
          if (images.includes(openImage)) { status.textContent = 'Detecting on ' + openImage + '…'; await detectLevels?.(); status.textContent = 'Levels detected on ' + openImage + ' (' + method + '), kept in the project.'; }
          if (others.length) await jobStarted(await runCommand('detect_background_levels', {images:others, method, inside:!!backgroundLevels()?.inside, ...backgroundLevels()?.parameters, native:!!backgroundLevels()?.parameters?.native}, {confirmed:true}));
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const asCsv = node('input', {type:'checkbox', 'aria-label':'Export thresholds as CSV'}); asCsv.checked = true;
      const asImage = node('input', {type:'checkbox', 'aria-label':'Export background-removed images as OME-TIFF'});
      // The open image's thresholds, as shown, saved at once (Undo takes them back).
      const saveOpenCsv = async () => {
        const value = backgroundLevels();
        if (!value || !Object.keys(value.levels).length) { status.textContent = 'Set a level for at least one marker first (Detect, Auto, or type it).'; return false; }
        const kept = current.saved?.levels;
        const differ = kept ? Object.keys(kept).filter(name => !(name in value.levels) || Math.abs(value.levels[name] - kept[name]) > 0.05) : [];
        if (differ.length && !window.confirm('Replace the exported levels in ' + current.saved.table + '? ' + differ.length + ' marker' +
            (differ.length === 1 ? '' : 's') + ' differ from it (' + differ.slice(0, 4).join(', ') + (differ.length > 4 ? '…' : '') + '). You can undo this.')) return false;
        const result = await runCommand('save_background_levels', {levels:value.levels, method:value.method}, {confirmed:true});
        showResult(result); status.textContent = result.message; await loadMarkers(); return true;
      };
      const exportFor = (button, which) => async () => {
        const images = await chosen(which);
        if (!images.length) { status.textContent = which === 'open' ? 'Open an image first.' : 'Tick the images first.'; return; }
        if (!asCsv.checked && !asImage.checked) { status.textContent = 'Tick Thresholds (CSV) and/or Image (OME-TIFF).'; return; }
        const others = images.filter(name => name !== openImage), withOpen = images.includes(openImage);
        const replaced = asCsv.checked ? others.filter(name => (plan?.rows || []).some(row => row.image === name && row.levels)) : [];
        if (replaced.length && !window.confirm('Replace the exported thresholds of ' + replaced.length + ' image' + (replaced.length === 1 ? '' : 's') +
            ' (' + replaced.slice(0, 3).join(', ') + (replaced.length > 3 ? '…' : '') + ')?')) return;
        button.disabled = true;
        try {
          if (withOpen && asCsv.checked && !(await saveOpenCsv()) && !others.length && !asImage.checked) return;
          if (withOpen && asImage.checked) {
            const value = backgroundLevels();
            await jobStarted(await runCommand('export_background', {images:[openImage], csv:false, write_image:true, open:openImage,
              levels:value?.levels || {}, method:value?.method || ''}, {confirmed:true}));
          }
          if (others.length) await jobStarted(await runCommand('export_background', {images:others, csv:asCsv.checked, write_image:asImage.checked}, {confirmed:true}));
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const buttons = (run, tip) => ['This image', 'Selected'].map((text, k) => {
        const button = node('button', {class:'btn' + (k ? ' primary' : ''), text, title:tip[k]});
        button.addEventListener('click', run(button, k ? 'picked' : 'open')); return button;
      });
      current.runBar([
        node('div', {class:'row'}, [node('span', {class:'lbl', text:'Detect'}),
          ...buttons(detect, ['Auto all markers on the open image', 'Each ticked image, inside the Class in its row (or the whole image)']),
          help('Measures every marker’s background (not the blank cycles) with the method in the Auto row above the markers, inside ' +
            'each image’s annotation Class when Auto is set to an annotation, else on the whole image. The levels are kept in the ' +
            'project only (they show as the markers’ minimums when you open the image); check them, then Export.')]),
        node('div', {class:'row'}, [node('span', {class:'lbl', text:'Export'}), node('label', {}, [asCsv, node('span', {text:' Thresholds (CSV)'})]),
          node('label', {}, [asImage, node('span', {text:' Image (OME-TIFF)'})]),
          ...buttons(exportFor, ['The open image, with its levels as shown', 'Each ticked image, with its levels kept in the project']),
          help('Thresholds writes each image’s levels as Tab 2’s threshold table (RESOLVE reads it). Image writes a new ' +
            'image with the blank cycles and the levels removed, and the markers kept, renamed and coloured as chosen. The originals stay.')]),
        status]);
      current.sync = () => {};
    };

    // ------------------------------------------------------------ Regions of interest
    TOOLS.tab3 = current => {
      // Tab 3 in the Studio scheme: the images with their ROI files, the open image's ROIs and markers, TMA cores; then
      // Detect cores and Extract for the open image or the ticked ones.
      const place = section('Folders and images', {tip:'Where the images are (registered or background-removed), where their ROI ' +
        'shapes are (GeoJSON; Tab 3’s file for each image, found by name, or any file picked in its row), and where the cut-out ' +
        'ROI images are written (<image>_<ROI name>.ome.tiff). Click an image to open it with its ROIs.'});
      const shown = section('ROIs of the open image', {tip:'Draw rectangles, polygons or freehand shapes on the image and name them; ' +
        'they are saved into the image’s ROI file as you draw. Each name becomes the cut-out image’s name. Detected TMA cores ' +
        'appear here too, to check, rename or delete before extracting.'});
      const holder = node('div', {id:'studio-rois'});
      shown.body.append(holder);
      const markerBox = section('Markers', {tip:'Which of the open image’s markers go into its cut-out ROIs, with the name and ' +
        'colour each gets there; dye and exposure as the file records them. Saved for this image as you change them (where classic ' +
        'Tab 3 keeps its choice), and used by Extract.'});
      const coreBox = section('TMA cores', {tip:'Settings for Detect cores (classic Tab 3’s detector): it looks for round cores on a ' +
        '700 px wide picture of the DAPI and names them by row and column (A1, A2… B1…). Only for tissue microarrays.'});
      current.body.append(place.box, shown.box, markerBox.box, coreBox.box);
      const note = node('p', {class:'muted', role:'status'});
      const picked = new Set();
      let plan = null, openImage = null, saving, markersRequest = 0;
      const saveStep = async changes => {
        const folder = changes.folders ? Object.keys(changes.folders)[0] : null;
        note.textContent = folder ? 'Reading the new ' + folder + ' folder…' : 'Saving…';
        place.box.classList.add('studio-working');
        try {
          plan = await post(route('/roi_images'), {changes}); drawImages(); drawCores();
          note.textContent = (folder ? 'Folder changed. ' : 'Saved with the project. ') + plan.rows.length + ' images · ' +
            plan.rows.filter(row => row.rois).length + ' with ROIs.';
        }
        catch (error) { note.textContent = error.message; }
        finally { place.box.classList.remove('studio-working'); }
      };
      const loadPlan = async () => { try { plan = await get(route('/roi_images')); drawImages(); drawCores(); } catch (error) { fill(place.body, node('p', {class:'hint', text:error.message})); } };
      const folderBox = (kind, label, tip) => {
        const input = node('input', {type:'text', value:plan.folders[kind], maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
        input.addEventListener('change', () => { if (input.value.trim() && input.value.trim() !== plan.folders[kind]) saveStep({folders:{[kind]:input.value.trim()}}); });
        return field(label, input, tip);
      };
      const openRow = async row => {
        openImage = row.image;
        place.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
        place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"]')?.classList.add('selected');
        note.textContent = 'Opening ' + row.image + '…';
        try { const answer = await runCommand('open_roi_image', {image:row.image}); showResult(answer); note.textContent = answer.message; }
        catch (error) { note.textContent = error.message; }
      };
      const drawImages = () => {
        const head = pickAll(plan.rows.map(row => row.image), picked, drawImages);
        const rows = plan.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.image}); tick.checked = picked.has(row.image);
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.image) : picked.delete(row.image); head.sync(); });
          const file = choice([['', 'By name: ' + row.by_name], ...plan.geojson_files.filter(name => name !== row.by_name).map(name => [name, name])],
            row.chosen, 'ROI file for ' + row.image);
          file.title = 'The GeoJSON with this image’s ROIs';
          file.addEventListener('change', () => saveStep({files:{[row.image]:file.value}}).then(() => { if (openImage === row.image) openRow(row); }));
          const state = node('span', {}, [workingChip(current, row.image), chip(row.rois ? row.rois + ' ROI' + (row.rois === 1 ? '' : 's') : 'No ROIs', row.rois ? '' : 'warn'),
            row.rois ? chip(row.extracted + ' extracted', row.extracted === row.rois ? 'ok' : '') : null]);
          const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:'Open ' + row.image + ' with its ROIs', onclick:() => openRow(row)});
          const line = node('tr', {'data-image':row.image, class:row.image === openImage ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
          fill(line, node('td', {}, [tick]), node('td', {}, [name]), node('td', {}, [file]),
            node('td', {}, [state, row.edited ? node('span', {class:'studio-edited', text:' ✎', title:'Markers chosen for this image'}) : null]));
          return line;
        });
        fill(place.body,
          folderBox('images', 'Images', 'The images to cut ROIs from: registered or background-removed. Type the folder’s path and press Enter.'),
          folderBox('shapes', 'ROI shapes', 'The GeoJSON files with the ROIs (Tab 3’s boundaries folder). Drawn and detected ROIs are saved here.'),
          folderBox('output', 'Output', 'Where the cut-out ROI images are written (<image>_<ROI name>.ome.tiff); the folder is made if needed.'),
          node('div', {class:'row'}, [chip(plan.rows.length + ' images'), chip(plan.rows.filter(row => row.rois).length + ' with ROIs'),
            chip(plan.rows.filter(row => row.rois && row.extracted === row.rois).length + ' extracted', 'ok')]),
          plan.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'},
            [node('tr', {}, [node('th', {}, [head]), ...['Image', 'ROI file', 'State'].map(text => node('th', {text}))]), ...rows])])
            : node('p', {class:'muted', text:'No image in this folder: type the images folder above.'}),
          note);
      };
      // TMA core settings, saved with the project as they change.
      const drawCores = () => {
        const tma = plan.tma;
        const number = (key, label, attrs, tip) => {
          const input = node('input', {type:'number', value:String(tma[key]), 'aria-label':label, ...attrs});
          input.addEventListener('change', () => saveStep({tma:{...plan.tma, [key]:Number(input.value)}}));
          return field(label, input, tip);
        };
        const format = choice([['A-Z', 'A1, A2… B1…'], ['1-N', 'R1_C1, R1_C2…']], tma.format, 'Row names');
        format.addEventListener('change', () => saveStep({tma:{...plan.tma, format:format.value}}));
        const prefix = node('input', {type:'text', value:tma.prefix, maxlength:'32', 'aria-label':'Core name prefix'});
        prefix.addEventListener('change', () => saveStep({tma:{...plan.tma, prefix:prefix.value}}));
        fill(coreBox.body,
          number('min', 'Smallest core (px)', {min:'5', max:'700', step:'5'}, 'On the 700 px wide picture. Smaller spots are not cores.'),
          number('max', 'Largest core (px)', {min:'5', max:'700', step:'5'}, 'On the 700 px wide picture. Larger areas are not cores.'),
          number('threshold', 'Brightness cut', {min:'0', max:'255', step:'1'}, '0 chooses it automatically (Otsu); otherwise 1 to 255 on the picture.'),
          number('gap', 'Row gap', {min:'0.1', max:'2', step:'0.1'}, 'How far apart two cores must be, top to bottom, relative to a core’s height, to start a new row.'),
          field('Row names', format, 'Letters for rows (A1, B1…) or numbers (R1_C1, R2_C1…).'),
          field('Prefix', prefix, 'Optional text at the start of every core’s name.'));
      };
      // The open image's markers: keep (all on top), colour, name (the box), dye, exposure.
      const drawMarkers = async () => {
        const request = ++markersRequest;
        let outside = null;  // an image open in the viewer but not in the images folder: its markers all the same
        if (!openImage) {
          try { const open = await get(route('/roi_markers')); if (!plan) await planReady; if ((plan?.rows || []).some(row => row.image === open.image)) openImage = open.image; else outside = open; } catch (_) {}
          if (request !== markersRequest) return;
          if (!openImage && !outside) { fill(markerBox.body, node('p', {class:'muted', text:'Open an image from the list to choose its markers.'})); return; }
          if (openImage) drawImages();
        }
        let found = outside;
        if (!outside) {
          try { found = await get(route('/roi_images?image=' + encodeURIComponent(openImage))); }
          catch (error) { if (request === markersRequest) fill(markerBox.body, node('p', {class:'muted', text:error.message})); return; }
          if (request !== markersRequest) return;
          plan = {...found, markers:undefined}; drawImages();
        }
        const image = outside ? outside.image : openImage;
        const status = node('p', {class:'muted', role:'status'}), warn = node('p', {class:'hint'});
        const rows = found.markers.map(item => {
          const keep = node('input', {type:'checkbox', 'aria-label':'Extract ' + item.name}); keep.checked = item.keep;
          const colour = node('input', {type:'color', value:item.new_color || item.color, 'aria-label':'Colour of ' + item.name});
          const name = node('input', {type:'text', value:item.rename || item.name, placeholder:item.name, maxlength:'64', class:'grow',
            'aria-label':'Name of ' + item.name, title:'In the file: ' + item.name + '. Type to rename.'});
          return {item, keep, colour, name, row:node('div', {class:'row studio-roi-marker'}, [keep, colour, name,
            node('span', {class:'level-meta studio-meta', text:item.dye, title:'Dye'}), node('span', {class:'level-meta studio-meta', text:item.exposure, title:'Exposure (ms)'})])};
        });
        const all = node('input', {type:'checkbox', 'aria-label':'Extract all markers'});
        const sync = () => { const some = rows.filter(r => r.keep.checked).length; all.checked = some === rows.length; all.indeterminate = some > 0 && some < rows.length;
          summary.textContent = some + ' of ' + rows.length + ' markers'; };
        const summary = chip('');
        const save = () => {
          const seen = {};
          rows.forEach(r => { if (r.keep.checked) { const final = r.name.value.trim() || r.item.name; seen[final] = (seen[final] || 0) + 1; } });
          const twice = Object.keys(seen).filter(key => seen[key] > 1);
          warn.textContent = twice.length ? 'Same name twice: ' + twice.join(', ') + '. Untick one or rename it.' : '';
          if (twice.length || !rows.some(r => r.keep.checked)) { if (!twice.length) warn.textContent = 'Keep at least one marker.'; return; }
          clearTimeout(saving);
          const chosen = {
            keep:rows.filter(r => r.keep.checked).map(r => r.item.name),
            rename:Object.fromEntries(rows.filter(r => r.name.value.trim() && r.name.value.trim() !== r.item.name).map(r => [r.item.name, r.name.value.trim()])),
            colors:Object.fromEntries(rows.filter(r => r.colour.value.toLowerCase() !== r.item.color.toLowerCase()).map(r => [r.item.name, r.colour.value])),
          };
          saving = setTimeout(() => (outside ? post(route('/roi_markers'), chosen) : post(route('/roi_images'), {image, markers:chosen})).then(() => { status.textContent = 'Markers saved for ' + image + '; Extract uses them.'; }).catch(error => { status.textContent = error.message; }), 500);
        };
        rows.forEach(r => { r.keep.addEventListener('change', () => { sync(); save(); }); r.colour.addEventListener('change', save); r.name.addEventListener('change', save); });
        all.addEventListener('change', () => { rows.forEach(r => { r.keep.checked = all.checked; }); sync(); save(); });
        fill(markerBox.body, node('div', {class:'row'}, [all, node('strong', {text:image, class:'studio-file grow'}), summary,
          node('span', {class:'muted', text:'Dye · Exposure'})]),
          node('div', {class:'studio-roi-markers'}, rows.map(r => r.row)), warn, status);
        sync();
      };
      const planReady = loadPlan();
      followJobs(current, ['extract_rois', 'detect_roi_cores'], loadPlan);
      const draw = () => {
        if (renderRois(holder)) return;
        const open = node('button', {class:'btn small', text:'Show its ROIs', onclick:async () => {
          open.disabled = true;
          try { const result = await runCommand('draw_rois', {}); showResult(result); note.textContent = result.message; draw(); }
          catch (error) { note.textContent = error.message; }
          finally { open.disabled = false; }
        }});
        holder.replaceChildren(node('div', {class:'row'}, [node('span', {class:'muted grow', text:'Click an image in the list to open it with its ROIs.'}), open]));
      };
      current.drawRois = draw; draw();
      const markersBefore = current.refreshMarkers;
      current.refreshMarkers = () => { markersBefore?.(); openImage = null; drawMarkers(); };
      drawMarkers();
      // Bottom: Detect cores and Extract, for the open image or the ticked ones.
      const status = node('p', {class:'muted', role:'status'});
      const chosen = async which => {
        if (which === 'open') { if (!openImage) await drawMarkers(); return openImage ? [openImage] : []; }
        return Array.from(picked);
      };
      const jobStarted = async result => { showResult(result); status.textContent = result.message; if (result.data?.job) { await refreshJobs(); showTab('jobs'); } };
      const detect = (button, which) => async () => {
        const images = await chosen(which);
        if (!images.length) { status.textContent = which === 'open' ? 'Open an image first.' : 'Tick the images first.'; return; }
        const replaced = images.filter(name => (plan?.rows || []).some(row => row.image === name && row.rois));
        if (replaced.length && !window.confirm('Replace the ROIs of ' + replaced.slice(0, 3).join(', ') + (replaced.length > 3 ? '…' : '') +
            ' with the detected cores? Their ROI files are rewritten.')) return;
        button.disabled = true; status.textContent = 'Looking for cores…';
        try { await jobStarted(await runCommand('detect_roi_cores', {images, open:openImage || ''}, {confirmed:true})); await loadPlan(); }
        catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const whole = node('input', {type:'checkbox', 'aria-label':'Keep the whole box around each ROI'});
      const extract = (button, which) => async () => {
        const images = await chosen(which);
        if (!images.length && which === 'open' && holder.querySelector('.shape-list')) {  // open another way: its ROIs as shown
          button.disabled = true;
          try { await jobStarted(await runCommand('extract_rois', {scope:'image', keep_outside:whole.checked}, {confirmed:true})); }
          catch (error) { status.textContent = error.message; }
          finally { button.disabled = false; }
          return;
        }
        if (!images.length) { status.textContent = which === 'open' ? 'Open an image first.' : 'Tick the images first.'; return; }
        button.disabled = true;
        try { await jobStarted(await runCommand('extract_roi_images', {images, keep_outside:whole.checked}, {confirmed:true})); }
        catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const buttons = (run, tip) => ['This image', 'Selected'].map((text, k) => {
        const button = node('button', {class:'btn' + (k ? ' primary' : ''), text, title:tip[k]});
        button.addEventListener('click', run(button, k ? 'picked' : 'open')); return button;
      });
      current.runBar([
        node('div', {class:'row'}, [node('span', {class:'lbl', text:'Detect cores'}),
          ...buttons(detect, ['The open image', 'Each ticked image']),
          help('For tissue microarrays: finds the round cores with the TMA core settings and saves them as the image’s ROIs ' +
            '(its ROI file is rewritten). Check them on the image, rename or delete any, then Extract.')]),
        node('div', {class:'row'}, [node('span', {class:'lbl', text:'Extract'}), node('label', {}, [whole, node('span', {text:' Whole box'})]),
          ...buttons(extract, ['The open image’s ROIs', 'The ROIs of each ticked image']),
          help('Cuts each ROI out as its own OME-TIFF, with the markers chosen above and all of the parent image’s details ' +
            '(pyramid, pixel size, names, colours, dye, exposure, where it was cut from). Outside the ROI’s shape is set to 0, ' +
            'unless Whole box is ticked. ROIs already extracted are kept; the originals stay.')]),
        status]);
    };

    // ------------------------------------------------------------ Annotation
    // Annotation in the Studio scheme: the images with a mark per stage done, then the stages as tabs, each with its
    // settings (live on the open image) and its actions for the open image or the ticked ones.
    const ANNOTATION_STAGES = [
      {id:'tissue', name:'Tissue', letter:'T', tip:'Finds the tissue (H&E layers and/or markers) and names it. The later stages build on it.'},
      {id:'artifacts', name:'Artifacts', letter:'A', done:'clean', tip:'Train Blood clots and Image quality (blur, bright spots and lint) from reviewed ' +
        'red/green examples using available H&E and markers. Review predictions, save polygons, then subtract from tissue.'},
      {id:'pieces', name:'Pieces', letter:'P', tip:'Splits the clean tissue into connected pieces, cuts narrow necks and drops pieces ' +
        'too small; each piece is one tissue region (tissue_1, tissue_2…).'},
      {id:'classes', name:'Classification', letter:'C', tip:'Groups the pixels by their markers into classes, names them into domains.'},
      {id:'regions', name:'Marker regions', letter:'R', tip:'Regions where chosen markers are above a level (all of them), as polygons to review and name.'},
      {id:'zones', name:'Zones', letter:'Z', tip:'Pools domains and draws rings inside and outside chosen classes.'},
      {id:'export', name:'Export', letter:'E', tip:'Writes one GeoJSON per image. Import your own GeoJSON files (each goes to its image), choose what to keep inside, ' +
        'what to export and what to subtract once, and export the open image or all ticked images by the same rules.'},
    ];
    const ANNOTATION_JOBS = ['classify_images', 'detect_tissue', 'clean_tissue', 'detect_artifacts', 'artifact_batch', 'subtract_annotations', 'classify_shared', 'save_class_polygons', 'remove_small_pieces', 'subtract_artifacts', 'apply_pieces', 'export_classes', 'make_zones', 'marker_regions', 'export_annotations', 'import_annotations', 'train_artifact_model'];
    TOOLS.tab4 = current => {
      let part = recall('annotation-part', 'tissue');
      if (!ANNOTATION_STAGES.some(stage => stage.id === part)) part = ANNOTATION_STAGES.some(stage => stage.id === ({clot:'artifacts', blur:'artifacts', bright:'artifacts', clean:'artifacts'})[part]) ? 'artifacts' : 'tissue';
      const place = section('Folders and images', {tip:'Where the images are and where the annotations are written. Click an image to ' +
        'open it with its tissue and artifacts. The letters show saved results for each image (green): T tissue, A artifacts, ' +
        'P pieces, C classification, R marker regions, Z zones, E exported. Review artifacts, then use Subtract from tissue to clean it.'});
      const note = node('p', {class:'muted', role:'status'});
      const picked = current.annotationPicked = new Set();
      let plan = null, openImage = null;
      const saveStep = async changes => {
        const folder = changes.folders ? Object.keys(changes.folders)[0] : null;
        note.textContent = folder ? 'Reading the new ' + folder + ' folder…' : 'Saving…';
        place.box.classList.add('studio-working');
        try { plan = await post(route('/annotation_images'), {changes}); current.annotationPlan = plan; drawImages(); note.textContent = (folder ? 'Folder changed. ' : 'Saved with the project. ') + plan.rows.length + ' images.'; }
        catch (error) { note.textContent = error.message; }
        finally { place.box.classList.remove('studio-working'); }
      };
      current.saveAnnotationStep = saveStep;
      const loadPlan = async () => {
        try { plan = await get(route('/annotation_images')); current.annotationPlan = plan; drawImages(); current.planChanged?.(); await reloadAnnotations?.(); }
        catch (error) { fill(place.body, node('p', {class:'hint', text:error.message})); }
      };
      const folderBox = (kind, label, tip) => {
        const input = node('input', {type:'text', value:plan.folders[kind], maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
        input.addEventListener('change', () => { if (input.value.trim() && input.value.trim() !== plan.folders[kind]) saveStep({folders:{[kind]:input.value.trim()}}); });
        return field(label, input, tip);
      };
      current.refreshAnnotationPlan = loadPlan;
      const openRow = async row => {
        openImage = row.image;
        place.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
        place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"]')?.classList.add('selected');
        note.textContent = 'Opening ' + row.image + '…';
        try { await flushShapes?.(); const answer = await runCommand('open_annotation_image', {image:row.image}); showResult(answer); note.textContent = answer.message; }
        catch (error) { note.textContent = error.message; }
      };
      // A letter opens its image at that stage and shows the stage's polygons (all artifact kinds for A).
      const STAGE_LAYERS = {
        tissue:l => l.id === 'tissue-regions' || l.id === activeTissueId?.(), pieces:l => l.id === 'tissue-regions' || l.id === activeTissueId?.(),
        clean:l => (/^artifact-/.test(l.id) && !/^artifact-examples-/.test(l.id)) || l.id === 'drawn-artifacts',
        classes:l => l.id === 'pixel-classes' || l.annotation_group === 'classes',
        regions:l => l.id === 'marker-regions' || l.annotation_group === 'regions', zones:l => l.annotation_group === 'zones'};
      const goToStage = async (row, stage) => {
        const shownImage = currentSpec?.layers?.find(l => l.type === 'image')?.name;
        if (row.image !== openImage || shownImage !== row.image) await openRow(row);
        const target = stage.id === 'pieces' ? 'tissue' : stage.id;
        if (target !== part) await changeStage(target);
        const test = STAGE_LAYERS[stage.done || stage.id];
        if (!test || !row.done[stage.done || stage.id]) return;
        const shown = hooks.showLayersWhere?.(test) || 0;
        note.textContent = row.image + ' · ' + stage.name + (shown ? ': ' + shown + ' layer' + (shown === 1 ? '' : 's') + ' shown.' : '.');
      };
      // I: GeoJSON files imported for the image; a click opens it with them shown.
      const importedLetter = row => {
        const files = row.imported || [];
        const letter = node('b', {'data-annotation-stage':'imported', class:files.length ? 'done' : '', text:'I', role:'button', tabindex:'0',
          title:files.length ? 'Imported: ' + files.join(', ') + '. Click to see them on the image.' : 'No imported GeoJSON. Add them in Export > Imported files.'});
        const go = async event => {
          event.stopPropagation();
          const shownImage = currentSpec?.layers?.find(l => l.type === 'image')?.name;
          if (row.image !== openImage || shownImage !== row.image) await openRow(row);
          if (part !== 'export') await changeStage('export');
          if (!files.length) return;
          const shown = hooks.showLayersWhere?.(l => l.annotation_group === 'imported') || 0;
          note.textContent = row.image + ' · imported: ' + files.join(', ') + (shown ? ' (shown in Layers).' : '.');
        };
        letter.addEventListener('click', go);
        letter.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); go(event); } });
        return letter;
      };
      const strip = row => node('span', {class:'studio-stages-done'}, [importedLetter(row), ...ANNOTATION_STAGES.map(stage => {
        const done = row.done[stage.done || stage.id];
        const classified = !done && stage.id === 'classes' && row.classified;  // classes to review, no polygons yet
        const letter = node('b', {'data-annotation-stage':stage.id, class:done ? 'done' : classified ? 'classified' : '', text:stage.letter, role:'button', tabindex:'0',
          title:stage.name + (done ? ': saved result' : classified ? ': classified; name the classes, then Save polygons' : ': not saved') + '. Click to open it here.'});
        const go = event => { event.stopPropagation(); goToStage(row, stage); };
        letter.addEventListener('click', go);
        letter.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); go(event); } });
        return letter;
      })]);
      current.updateAnnotationStatus = () => {
        if (!plan || !place.body.isConnected) return;
        const tissue = annotationState?.()['tissue-regions'];
        const image = currentSpec?.layers?.find(l => l.type === 'image' && l.token === tissue?.image_token)?.name;
        for (const row of plan.rows) {
          const badge = place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"] [data-annotation-stage="tissue"]');
          if (!badge) continue;
          let done = row.done.tissue, waiting = false, title = done ? 'Tissue: saved result' : 'Tissue: not saved';
          if (row.image === image && tissue && !tissue.loading) {
            if (tissue.saved) row.done.tissue = true;
            waiting = !!(tissue.pending || tissue.busy || tissue.error || tissue.loadError);
            done = tissue.saved && !waiting;
            title = 'Tissue: ' + (tissue.loadError ? 'could not check saved polygons' : tissue.error ? 'save not confirmed' :
              tissue.busy ? 'saving edits' : tissue.preview ? 'preview not saved' : tissue.pending ? 'edits not saved' :
              done ? 'polygons saved' : 'not saved');
          }
          badge.classList.toggle('done', !!done); badge.classList.toggle('waiting', waiting);
          badge.title = title; badge.setAttribute('aria-label', title);
        }
        // Artifacts of the open image: amber while a prediction or edits are not saved (A turns green once saved).
        const states = annotationState?.() || {};
        const artifacts = Object.entries(states).filter(([id]) => /^artifact-(?!examples-)/.test(id)).map(([, value]) => value)
          .filter(value => value.image_token === currentSpec?.layers?.find(l => l.type === 'image')?.token);
        const row = plan.rows.find(item => item.image === image);
        const badge = row && place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"] [data-annotation-stage="artifacts"]');
        if (badge) {
          const unsaved = artifacts.some(value => value.preview || value.pending);
          const saved = artifacts.some(value => value.saved && !value.preview);
          if (saved) row.done.clean = true;
          badge.classList.toggle('waiting', unsaved); badge.classList.toggle('done', !unsaved && !!row.done.clean);
          badge.title = 'Artifacts: ' + (unsaved ? 'prediction or edits not saved: press Save polygons' : row.done.clean ? 'saved result' : 'not saved') + '. Click to open it here.';
        }
        const count = place.body.querySelector('[data-tissue-count]');
        if (count) count.textContent = plan.rows.filter(row => row.done.tissue).length + ' with tissue';
      };
      // Sub-directory: the step's own files (tissue, artifacts, pieces, classes, regions, History) in a folder of the
      // output; the export stays in the output folder. An existing project's files move only when asked.
      const subfolderRow = () => {
        const chosen = plan.subfolder || {on:false, path:'./working_files', pending:0};
        const on = node('input', {type:'checkbox', 'aria-label':'Sub-directory'}); on.checked = chosen.on;
        const path = node('input', {type:'text', value:chosen.path, maxlength:'200', class:'grow', 'aria-label':'Sub-directory path', placeholder:'./working_files'});
        const send = async (move = false) => {
          if (move && !window.confirm('Move ' + chosen.pending + ' files and folders of this step from ' + plan.folders.output + ' into ' + path.value.trim() + '? ' +
            'Their History and every reference to them go along. Classic Tab 4 reads the output folder itself, so it will not see them there.')) { on.checked = chosen.on; return; }
          note.textContent = move ? 'Moving the files…' : 'Saving…';
          try {
            plan = await post(route('/annotation_images'), {changes:{subfolder:{on:on.checked, path:path.value.trim(), move}}});
            current.annotationPlan = plan; drawImages();
            note.textContent = move ? plan.moved + ' files and folders moved into ' + plan.working + '.' : 'The step’s files are kept in ' + plan.working + '.';
            if (move) { const image = currentSpec?.layers?.find(l => l.type === 'image')?.name; if (image) showResult(await runCommand('open_annotation_image', {image})); }
          } catch (error) { note.textContent = error.message; on.checked = chosen.on; }
        };
        on.addEventListener('change', () => send(on.checked && chosen.pending > 0 && !chosen.on));
        path.addEventListener('change', () => send(false));
        const move = chosen.pending && !chosen.on ? node('button', {class:'btn small', text:'Move ' + chosen.pending + ' into it',
          title:'This project’s files are in the output folder itself: move them into the sub-directory', onclick:() => { on.checked = true; send(true); }}) : null;
        return field('Sub-directory', node('div', {class:'inline grow'}, [on, path, move].filter(Boolean)),
          'On: tissue, artifacts, pieces, classification, marker regions and their History are kept in this folder of the output (./name), ' +
          'so the output folder itself holds only the exports. Off: everything in the output folder, as classic Tab 4 does. A project made before ' +
          'this keeps its files where they are until you press Move.');
      };
      const drawImages = () => {
        const head = pickAll(plan.rows.map(row => row.image), picked, drawImages);
        const rows = plan.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.image}); tick.checked = picked.has(row.image);
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.image) : picked.delete(row.image); head.sync(); });
          const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:'Open ' + row.image + ' with its annotations', onclick:() => openRow(row)});
          const line = node('tr', {'data-image':row.image, class:row.image === openImage ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
          fill(line, node('td', {}, [tick]), node('td', {}, [name, row.drawn || row.edited ? node('span', {class:'studio-edited', text:' ✎', title:'Annotations edited by hand'}) : null]),
            node('td', {}, [workingChip(current, row.image) || workingChip(current, row.sample), strip(row)]));
          return line;
        });
        const count = stage => plan.rows.filter(row => row.done[stage]).length;
        fill(place.body,
          folderBox('images', 'Images', 'The images to annotate (ROIs, or background-removed images). Type the folder’s path and press Enter.'),
          folderBox('output', 'Output', 'One folder for the whole step: the exported GeoJSON files go here, and the working files (tissue, artifacts, pieces, classes, regions) in its Sub-directory. Studio reads its saved results from here: pointing it at a new folder starts with none (the letters turn grey).'),
          subfolderRow(),
          node('div', {class:'row'}, [chip(plan.rows.length + ' images'), node('span', {class:'chip '+(count('tissue')?'ok':''), 'data-tissue-count':'', text:count('tissue') + ' with tissue'}),
            chip(count('export') + ' exported', count('export') ? 'ok' : '')]),
          plan.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs'},
            [node('tr', {}, [node('th', {}, [head]), ...['Image', 'Stages'].map(text => node('th', {text}))]), ...rows])])
            : node('p', {class:'muted', text:'No image in this folder: type the images folder above.'}),
          note);
        current.updateAnnotationStatus();
      };
      followJobs(current, ANNOTATION_JOBS, loadPlan);
      loadPlan();
      // The stages, as tabs.
      const tabs = node('div', {class:'segmented studio-stage-tabs', role:'tablist', 'aria-label':'Annotation stage'});
      const tip = help('');
      const holder = node('div', {class:'studio-method'});
      // Delete results: the shown stage's saved results, for the open image or each ticked image, set aside (recoverable).
      const removal = node('div', {class:'row wrap studio-delete-results', hidden:true});
      const deleteButton = node('button', {type:'button', class:'btn small ghost', text:'Delete results…',
        title:'Remove this stage’s saved results (its letter turns grey), to start it again'});
      deleteButton.addEventListener('click', () => {
        const stage = ANNOTATION_STAGES.find(item => item.id === part) || ANNOTATION_STAGES[0];
        const count = (current.annotationPicked || new Set()).size;
        const said = node('span', {class:'muted', role:'status'});
        const run = async (images, button) => {
          const which = images ? count + ' ticked image' + (count === 1 ? '' : 's') : 'the open image';
          if (!window.confirm('Delete the ' + stage.name + ' results of ' + which + '? The files are set aside in outputs/studio/deleted.')) return;
          button.disabled = true;
          try {
            await flushShapes?.();
            const found = await post(route('/annotation_export'), {action:'delete_results', stage:stage.id, ...(images ? {images:Array.from(current.annotationPicked)} : {})});
            said.textContent = found.files ? found.files + ' file' + (found.files === 1 ? '' : 's') + ' set aside in outputs/studio/deleted.' : 'There were no ' + stage.name + ' results to delete.';
            await loadPlan();
            const image = (currentSpec?.layers || []).find(l => l.type === 'image')?.name;
            if (found.open && found.files && image) showResult(await runCommand('open_annotation_image', {image}));
          } catch (error) { said.textContent = error.message; }
          finally { button.disabled = false; }
        };
        const one = node('button', {type:'button', class:'btn small', text:'This image'});
        one.addEventListener('click', () => run(false, one));
        const some = node('button', {type:'button', class:'btn small', text:'Ticked images (' + count + ')', disabled:count ? null : ''});
        some.addEventListener('click', () => run(true, some));
        const close = node('button', {type:'button', class:'btn small ghost', text:'Close'});
        close.addEventListener('click', () => { removal.hidden = true; });
        removal.replaceChildren(node('span', {class:'lbl', text:'Delete ' + stage.name + ' results of:'}), one, some, close,
          help('Removes what this stage saved (for Classification also its run; for Export the exported files), so its letter turns grey ' +
            'and you can run it again. Settings stay. Files are set aside in outputs/studio/deleted, to recover by hand.'), said);
        removal.hidden = false;
      });
      current.body.append(place.box, node('div', {class:'row studio-stage-row'}, [tabs, tip, deleteButton]), removal, holder);
      let stageRequest = 0;
      const changeStage = async next => {
        const request = ++stageRequest;
        if (holder.contains(document.activeElement)) document.activeElement.blur();
        if (current.annotationSettingsPending?.()) {
          try { await current.saveAnnotationSettings(); } catch (_) { return; }
          if (request !== stageRequest || !holder.isConnected) return;
        }
        part = next; remember('annotation-part', part); removal.hidden = true; draw();
      };
      const draw = () => {
        const chosen = ANNOTATION_STAGES.find(item => item.id === part) || ANNOTATION_STAGES[0];
        part = chosen.id; setTip(tip, chosen.tip);
        if (part === 'pieces') part = 'tissue';
        tabs.replaceChildren(...ANNOTATION_STAGES.filter(stage => stage.id !== 'pieces').map(stage => {
          const button = node('button', {type:'button', class:'btn small' + (stage.id === part ? ' active' : ''), role:'tab', 'aria-selected':String(stage.id === part),
            text:stage.name, onclick:() => changeStage(stage.id)});
          return button;
        }));
        current.clearAnnotationAutosave?.();
        const pane = node('div', {class:'studio-method-pane'});
        holder.replaceChildren(pane); current.runBar([]); showPreview(null); showRaster('classes', null); clearFigures();
        current.drawShapes = null; current.planChanged = null; current.updatePolygonSave = null; current.refreshDifference = null;
        current.saveAnnotationSettings = null; current.annotationSettingsPending = null;
        setDrawTarget?.(({tissue:activeTissueId?.() || 'tissue-regions', pieces:activeTissueId?.() || 'tissue-regions', artifacts:'artifact-loading', regions:'marker-regions'})[part] || null);
        // A stage left before it finished loading must not set the bottom bar or the step's hooks over the stage
        // now shown: the panel sees the step only while its pane is on the page.
        const staged = new Proxy(current, {
          get:(target, key) => key === 'runBar' ? (rows, note) => { if (pane.isConnected) target.runBar(rows, note); } : target[key],
          set:(target, key, value) => { if (pane.isConnected) target[key] = value; return true; }});
        if (part === 'tissue') tissuePanel(staged, pane);
        else if (part === 'artifacts') artifactPanel(staged, pane);
        else if (part === 'classes') classesPanel(staged, pane);
        else if (part === 'zones') zonesPanel(staged, pane);
        else if (part === 'regions') regionsPanel(staged, pane);
        else if(part==='export') exportPanel(staged,pane);
      };
      draw();
    };
    // Imported GeoJSON for the Annotation step's images, from any step that uses annotations (Export, Segmentation):
    // a folder imported at once (each file to the image it names, in the background), and the imported files listed
    // across all images to take off the ones no longer wanted.
    const importsBox=(onChange)=>{
      const said=node('span',{class:'muted',role:'status'});
      const folder=node('input',{type:'text',maxlength:'1024',class:'grow',placeholder:'Folder of GeoJSON files','aria-label':'Folder of GeoJSON files to import'});
      const go=node('button',{class:'btn small',text:'Import folder',onclick:async()=>{
        if(!folder.value.trim()){said.textContent='Type the folder first.';return;}
        go.disabled=true;
        try{await post(route('/annotation_imports'),{action:'folder',folder:folder.value.trim()});
          said.textContent='Importing in the background: each file goes to the image it names; files already imported are skipped.';refreshJobs();showTab('jobs');}
        catch(e){said.textContent=e.message;}finally{go.disabled=false;}}});
      folder.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();go.click();}};
      const list=node('div',{class:'annotation-imports'});
      const filter=node('input',{type:'search',placeholder:'Filter by image or file','aria-label':'Filter imported files'});
      let files=[];
      const ticked=new Set();
      const draw=()=>{
        const text=filter.value.trim().toLowerCase();
        const shown=files.filter(f=>!text||(f.image+' '+f.file).toLowerCase().includes(text));
        const all=node('input',{type:'checkbox','aria-label':'Tick every listed file'});
        all.checked=shown.length&&shown.every(f=>ticked.has(f.path));
        all.onchange=()=>{shown.forEach(f=>all.checked?ticked.add(f.path):ticked.delete(f.path));draw();};
        list.replaceChildren(node('label',{class:'row annotation-import-row head'},[all,node('span',{class:'grow',text:shown.length+' of '+files.length+' files'})]),
          ...shown.map(f=>{const box=node('input',{type:'checkbox','aria-label':'Tick '+f.file+' of '+f.image});box.checked=ticked.has(f.path);
            box.onchange=()=>{box.checked?ticked.add(f.path):ticked.delete(f.path);};
            return node('label',{class:'row annotation-import-row',title:f.names.join(', ')},[box,node('span',{class:'grow',text:f.image.replace(/(\.ome)?\.(tiff?|qptiff)$/i,'')+' · '+f.file}),
              node('span',{class:'muted',text:f.kind==='imported'?f.polygons+' polygons':'found in the project'})]);}));
      };
      filter.oninput=draw;
      const remove=node('button',{class:'btn small',text:'Remove ticked',onclick:async()=>{
        const items=files.filter(f=>ticked.has(f.path)).map(f=>({image:f.image,path:f.path}));
        if(!items.length){said.textContent='Tick the files to remove.';return;}
        if(!window.confirm('Take '+items.length+' imported file'+(items.length===1?'':'s')+' off their images? Studio’s copies are kept in outputs/studio/deleted; your own files are not touched.'))return;
        remove.disabled=true;
        try{const value=await post(route('/annotation_imports'),{action:'remove',items});ticked.clear();
          said.textContent=value.removed+' removed. Open an image again to see it without them.';await load();onChange?.();}
        catch(e){said.textContent=e.message;}finally{remove.disabled=false;}}});
      const load=async()=>{try{files=(await get(route('/annotation_imports'))).files;draw();}catch(e){said.textContent=e.message;}};
      const manage=node('details',{class:'studio-options'},[node('summary',{text:'Manage imported files'}),node('div',{class:'row'},[filter,remove]),list]);
      manage.addEventListener('toggle',()=>{if(manage.open)load();});
      return node('div',{},[field('From a folder',node('div',{class:'inline grow'},[folder,go]),
        'Every GeoJSON in the folder goes to the image whose name it contains (sample_1_annotations.geojson to sample_1). Files already imported for their image are skipped.'),
        manage,said]);
    };
    const exportPanel=async(current,pane)=>{
      // Export in three parts: Imported files (each GeoJSON matched to its image), What to export (one set of rules for
      // every image: Keep inside, and per annotation Export / Subtract / export name and colour) and File names. The
      // annotation files themselves never change here.
      const status=node('span',{class:'muted',role:'status'});
      const imageLayer=()=>(currentSpec?.layers||[]).find(l=>l.type==='image');
      const downloadFile=(value,name)=>{const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/geo+json'}));
        node('a',{href:url,download:name}).click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
      let settings=null,rows=[],scope=null;
      const ruleLabel=rule=>{const at=String(rule).indexOf(':');return at<0?String(rule):rule.slice(at+1)+' ('+rule.slice(0,at)+')';};

      // 1. Imported files: matched by name; the ones matching no image wait here for their image.
      const imports=section('Imported files',{tip:'Add GeoJSON files made elsewhere (QuPath, an older export), one per image. Each goes to the image whose name ' +
        'it contains (sample_1_annotations.geojson to sample_1; small spelling differences are fine). A file matching no image is listed here: choose its image. ' +
        'Studio keeps its own copy; your files are never changed. The I letter in the images list marks images with one: click it to see them.'});
      const picker=node('input',{type:'file',accept:'.geojson,.json',multiple:'',hidden:''});
      const waiting=node('div',{class:'annotation-unmatched'});
      const importStatus=node('span',{class:'muted',role:'status'});
      const showWaiting=()=>{
        const list=hooks.unmatchedImports?.()||[];
        waiting.replaceChildren(...list.map(item=>{
          const choose=node('select',{'aria-label':'Image for '+item.name},[node('option',{value:'',text:'Choose its image…'}),
            ...(item.images||[]).map(name=>node('option',{value:name,text:name}))]);
          choose.onchange=async()=>{if(!choose.value)return;
            try{const found=await post(route('/annotation_export'),{action:'import_matched',name:item.name,features:item.features,image:choose.value});
              hooks.forgetUnmatched?.(item);if(found.open&&found.layer)await hooks.showShapesLayer?.(found.layer);
              importStatus.textContent=item.name+' → '+found.matched+'.';showWaiting();await current.refreshAnnotationPlan?.();await load();}
            catch(e){importStatus.textContent=e.message;}};
          return node('div',{class:'row annotation-unmatched-row'},[node('span',{class:'grow',text:item.name}),choose]);}));
        if(list.length)waiting.prepend(node('p',{class:'muted',text:list.length+' file'+(list.length===1?'':'s')+' matched no image:'}));
      };
      picker.onchange=async()=>{const chosen=Array.from(picker.files||[]);picker.value='';if(!chosen.length)return;
        importStatus.textContent='Importing '+chosen.length+' file'+(chosen.length===1?'':'s')+'…';
        try{const found=await hooks.importMatched?.(chosen);
          importStatus.textContent=found?(found.matched.length+' imported'+(found.matched.length?': '+found.matched.join(', '):'')+'.'):'';}
        catch(e){importStatus.textContent=e.message;}
        showWaiting();await current.refreshAnnotationPlan?.();await load();};
      const onUnmatched=()=>{if(!pane.isConnected){document.removeEventListener('spacemap-imports-unmatched',onUnmatched);return;}showWaiting();};
      document.addEventListener('spacemap-imports-unmatched',onUnmatched);
      imports.body.append(node('div',{class:'row'},[node('button',{class:'btn small',text:'Import GeoJSON files…',onclick:()=>picker.click()}),picker,
        node('span',{class:'muted',text:'or drop them on the image'})]),importsBox(()=>{current.refreshAnnotationPlan?.();load();}),waiting,importStatus);
      showWaiting();

      // 2. What to export: one set of rules for every image.
      const rules=section('What to export',{tip:'One set of rules for every image. Scan lists the annotations of the ticked images (all images when none is ' +
        'ticked): local ones were made in Studio, imported ones came from your GeoJSON files. Keep inside: every exported polygon is cut to this ' +
        'annotation (e.g. tissue (imported)). Export: write it to the file. Subtract: cut it out of the others (e.g. Blood clots). Type a name or ' +
        'pick a colour to change them in the exported file only. An image without some of these annotations simply goes without them.'});
      const inside=node('select',{'aria-label':'Keep the exported annotations inside'});
      const table=node('div',{class:'annotation-rules'});
      const scanNote=node('span',{class:'muted'});
      const saveRules=async changes=>{try{settings=(await post(route('/annotation_export'),{action:'settings',changes})).settings;
          hooks.refreshExportTicks?.();drawRules();}
        catch(e){status.textContent=e.message;}};
      const rule=key=>(settings?.rules||{})[key]||{};
      const drawRules=()=>{
        const chosen=settings?.inside||'';
        if(document.activeElement!==smallest)smallest.value=String(settings?.smallest_um2||0);smallest.disabled=!chosen;
        const options=[['','Nothing (no cutting)'],...rows.filter(r=>r.role!=='artifact').map(r=>[r.rule,ruleLabel(r.rule)])];
        if(chosen&&!options.some(([v])=>v===chosen))options.push([chosen,ruleLabel(chosen)+' (not in these images)']);
        inside.replaceChildren(...options.map(([value,text])=>node('option',{value,text})));inside.value=chosen;
        const head=node('div',{class:'annotation-rule head'},[node('span',{text:'Export'}),node('span',{text:'Subtract'}),node('span'),
          node('span',{text:'Annotation'}),node('span',{text:'From'}),node('span',{text:'Images'})]);
        table.replaceChildren(head,...rows.map(row=>{
          const r=rule(row.rule),container=row.rule===chosen;
          const tick=(what,text)=>{const box=node('input',{type:'checkbox','aria-label':text+' '+ruleLabel(row.rule)});box.checked=!!r[what];
            box.onchange=()=>saveRules({rules:{[row.rule]:{[what]:box.checked}}});return box;};
          const subtract=tick('subtract','Subtract');if(container){subtract.disabled=true;subtract.title='The Keep inside annotation';}
          const colour=node('input',{type:'color',value:r.colour||row.colour||'#ffd600','aria-label':'Export colour of '+ruleLabel(row.rule)});
          colour.onchange=()=>saveRules({rules:{[row.rule]:{colour:colour.value}}});
          const name=node('input',{type:'text',value:r.name||'',placeholder:row.name,maxlength:'60','aria-label':'Export name of '+ruleLabel(row.rule),
            title:'The name in the exported file (empty: '+row.name+')'});
          name.onchange=()=>saveRules({rules:{[row.rule]:{name:name.value.trim()}}});
          name.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();name.blur();}};
          return node('div',{class:'annotation-rule'+(container?' container':''),'data-rule':row.rule},[tick('export','Export'),subtract,colour,name,
            node('span',{class:'chip '+(row.origin==='imported'?'warn':'ok'),text:row.origin}),
            node('span',{class:'muted',text:row.images.length+(scope?' of '+scope.length:''),title:row.images.join(', ')+' · '+row.count+' polygons'})]);}));
        if(!rows.length)table.append(node('p',{class:'muted',text:'No annotations found. Open or tick images, then Scan.'}));
      };
      inside.onchange=()=>saveRules({inside:inside.value});
      const load=async(images)=>{
        const ticked=images||scope||[imageLayer()?.name].filter(Boolean);
        try{const [value,found]=await Promise.all([get(route('/annotation_export')),ticked.length?post(route('/annotation_export'),{action:'summary',images:ticked}):{names:[]}]);
          if(!pane.isConnected)return;
          settings=value.settings;rows=found.names;showNaming(value);drawRules();
          scanNote.textContent=scope?'Annotations of '+scope.length+' image'+(scope.length===1?'':'s')+'.':'Annotations of the open image. Scan to include the ticked images.';}
        catch(e){status.textContent=e.message;}
      };
      const scan=node('button',{class:'btn small',text:'Scan images',title:'List the annotations of the ticked images (all images when none is ticked)',onclick:async()=>{
        let images=Array.from(current.annotationPicked||[]);
        if(!images.length)images=Array.from(document.querySelectorAll('#studio-tools tr[data-image]')).map(tr=>tr.dataset.image);
        scan.disabled=true;scanNote.textContent='Reading '+images.length+' images…';
        try{scope=images;await load(images);}finally{scan.disabled=false;}}});
      const smallest=node('input',{type:'number',min:'0',step:'100',class:'small','aria-label':'Drop pieces smaller than (µm²)'});
      smallest.onchange=()=>{const value=Math.max(0,Number(smallest.value)||0);smallest.value=String(value);saveRules({smallest_um2:value});};
      rules.body.append(node('div',{class:'row'},[scan,scanNote]),field('Keep inside',inside,'Every exported polygon is cut to this annotation; nothing (no cutting) keeps them as they are.'),
        field('Drop pieces',node('div',{class:'inline'},[node('span',{class:'muted',text:'smaller than'}),smallest,node('span',{class:'muted',text:'µm²'})]),
          'Pieces of the Keep inside annotation smaller than this (after Subtract) are dropped, with every polygon inside them, as Remove small pieces does in Tissue. 0 keeps every piece.'),table);
      // Follow Layers: a rename, colour or Export tick there shows here at once.
      let timer=null;
      const follow=()=>{if(!pane.isConnected){document.removeEventListener('spacemap-annotations-changed',follow);return;}clearTimeout(timer);timer=setTimeout(()=>load(),400);};
      document.addEventListener('spacemap-annotations-changed',follow);

      // 3. File names.
      const naming=section('File names',{tip:'Each image’s file is <prefix><image name><suffix>.geojson in the Output folder chosen at the top of the step; ' +
        'with no prefix or suffix, the image’s own name. Exported files are never read back as annotations.'});
      const prefix=node('input',{type:'text',maxlength:'40','aria-label':'File name prefix',placeholder:'prefix'});
      const suffix=node('input',{type:'text',maxlength:'40','aria-label':'File name suffix',placeholder:'_suffix'});
      const folder=node('span',{class:'muted grow','aria-label':'Export folder'});
      const example=node('span',{class:'muted'});
      const merge=node('input',{type:'checkbox','aria-label':'Merge polygons with the same name'});merge.checked=true;
      const showNaming=value=>{if(document.activeElement!==prefix)prefix.value=value.settings.prefix;if(document.activeElement!==suffix)suffix.value=value.settings.suffix;
        folder.textContent=value.folder;example.textContent='e.g. '+value.file_name;merge.checked=value.settings.merge!==false;};
      const saveNaming=async changes=>{try{const value=await post(route('/annotation_export'),{action:'settings',changes});
        settings=value.settings;showNaming(value);status.textContent='Saved: files go to '+value.folder+'.';}catch(e){status.textContent=e.message;load();}};
      prefix.onchange=()=>saveNaming({prefix:prefix.value.trim()});suffix.onchange=()=>saveNaming({suffix:suffix.value.trim()});
      merge.onchange=()=>saveNaming({merge:merge.checked});
      naming.body.append(field('Name',node('div',{class:'inline grow'},[prefix,node('span',{class:'muted',text:'<image name>'}),suffix,node('span',{class:'muted',text:'.geojson'})]),
          'Letters, numbers and _ . + - only.'),node('div',{class:'row'},[example]),
        field('Saved in',folder,'The Output folder at the top of the step (its working files are in the Sub-directory). Change it there.'));
      const download=node('button',{class:'btn small',text:'Workflow record',onclick:async()=>{
        download.disabled=true;status.textContent='Updating…';
        try{const value=await post(route('/annotation_workflow'),{});
          const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
          const link=node('a',{href:url,download:'annotation_workflow.json'});link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
          status.textContent='Saved with the project.';
        }catch(error){status.textContent=error.message;}finally{download.disabled=false;}}});
      naming.body.append(node('div',{class:'row'},[download,help('A JSON record of the settings, models and annotation history, for later reuse.')]));
      pane.append(imports.box,rules.box,naming.box);

      // Bottom: export the open image, or every ticked image, by the same rules.
      const one=node('button',{class:'btn',text:'This image',onclick:async()=>{
        one.disabled=true;status.textContent='Exporting…';
        try{await flushShapes?.();const image=imageLayer();
          const value=await post(route('/annotation_export'),{image_token:image?.token,merge_same_names:merge.checked});
          if(!pane.isConnected)return;downloadFile(value.geojson,value.filename);
          status.textContent=value.count+' annotations exported ('+value.names.join(', ')+')'+(value.inside?', inside '+value.inside:'')+
            (value.subtracted?.length?', without '+value.subtracted.join(', '):'')+' as '+value.filename+(value.saved_as?' in '+value.saved_as.replace(/\/[^/]*$/,''):'')+'.'+
            (value.small_pieces_dropped?' '+value.small_pieces_dropped+' small piece'+(value.small_pieces_dropped===1?'':'s')+' of '+value.inside+' dropped.':'')+
            (value.left_out?.length?' Nothing was left of '+value.left_out.join(', ')+(value.inside?' inside '+value.inside:'')+': choose another Keep inside if that is not meant.':'');
          current.refreshAnnotationPlan?.();  // the E letter; the button need not wait for it
        }catch(e){status.textContent=e.message;}finally{one.disabled=false;}
      }});
      const some=node('button',{class:'btn primary',text:'Ticked images',onclick:async()=>{
        const images=Array.from(current.annotationPicked||[]);
        if(!images.length){status.textContent='Tick the images first (tick the header box for all).';return;}
        some.disabled=true;
        try{await flushShapes?.();await post(route('/annotation_export'),{action:'batch',images,merge_same_names:merge.checked});
          status.textContent='Exporting '+images.length+' image'+(images.length===1?'':'s')+' in the background by these rules (an image left out is listed in Jobs).';
          refreshJobs();showTab('jobs');}
        catch(e){status.textContent=e.message;}finally{some.disabled=false;}
      }});
      current.runBar([node('div',{class:'row wrap'},[node('span',{class:'lbl',text:'Export:'}),node('label',{},[merge,node('span',{text:' Merge same names'})]),one,some,
        help('Writes one GeoJSON per image by the rules above: the annotations ticked Export, cut to Keep inside, without those ticked Subtract. ' +
          'This image also downloads it. Ticked images: every ticked image in the background. Merge same names: all polygons of one name become one annotation, as QuPath shows it.')]),status]);
      await load();
    };
    // The bottom row of a stage: its label, then the open image or the ticked ones (settings saved first when changed).
    const stageRow = (current, label, command, {save, isDirty, status, tip, selected = true, ask, started, argumentsFor, onImage}) => {
      const run = (button, which) => async () => {
        const images = Array.from(current.annotationPicked || []);
        if (which === 'picked' && !images.length) { status.textContent = 'Tick the images first.'; return; }
        if (ask && !ask(which, images)) return;
        button.disabled = true;
        try {
          await flushShapes?.();
          await current.annotationStepSave;
          if (isDirty?.()) { await save(); status.textContent = 'Settings saved. '; } else status.textContent = '';
          if (which === 'open' && onImage) { await onImage(); return; }
          const answer = await runCommand(command, {...(which === 'picked' ? {images} : {scope:'image'}), ...argumentsFor?.()}, {confirmed:true});
          showResult(answer); status.textContent += answer.message;
          if (answer.data?.job) { started?.(); await refreshJobs(); showTab('jobs'); }
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const one = node('button', {class:'btn', text:'This image', title:'The image open in the viewer'});
      one.addEventListener('click', run(one, 'open'));
      const some = selected ? node('button', {class:'btn primary', text:'Selected', title:'Each image ticked in the list'}) : null;
      if (some) some.addEventListener('click', run(some, 'picked'));
      return node('div', {class:'row'}, [node('span', {class:'lbl', text:label}), one, some, tip ? help(tip) : null]);
    };
    // A small Save in a section's heading, beside its Saved / Not saved chip.
    const headerSave = (box, dirty, write, status, current, holder, isDirty, mark) => {
      let timer = null, saving = null;
      const save = async () => {
        clearTimeout(timer);
        while (saving) await saving;
        if (!isDirty()) return;
        saving = (async () => {
          try {
            while (isDirty()) {
              dirty.textContent = 'Saving…'; dirty.className = 'chip';
              await write();
            }
            mark(); status.textContent = 'Settings saved.';
          } catch (error) {
            dirty.textContent = 'Not saved'; dirty.className = 'chip warn';
            status.textContent = 'Could not save settings: ' + error.message;
            throw error;
          }
        })();
        try { await saving; } finally { saving = null; }
      };
      const schedule = () => {
        clearTimeout(timer);
        timer = setTimeout(() => save().catch(() => {}), 500);
      };
      // Capture also handles controls whose change event does not bubble.
      const scope = holder.classList.contains('tissue-pieces-holder') ? box : holder;
      scope.addEventListener('input', schedule, true);
      scope.addEventListener('change', schedule, true);
      scope.addEventListener('click', schedule);
      const cleanup = () => {
        clearTimeout(timer);
        scope.removeEventListener('input', schedule, true);
        scope.removeEventListener('change', schedule, true);
        scope.removeEventListener('click', schedule);
      };
      const savers = current.annotationSavers || (current.annotationSavers = new Set());
      savers.add({save, cleanup, pending:() => !!saving || !!isDirty()});
      current.clearAnnotationAutosave = () => { savers.forEach(item => item.cleanup()); savers.clear(); };
      current.saveAnnotationSettings = async () => { for (const item of savers) await item.save(); };
      current.annotationSettingsPending = () => Array.from(savers).some(item => item.pending());
      const button = node('button', {class:'btn small ghost', text:'Save', title:'Save these settings (actions save them too)', onclick:async () => {
        try { await save(); status.textContent = 'Settings saved.'; } catch (error) { status.textContent = error.message; }
      }});
      box.querySelector('.studio-sec-h').insertBefore(button, box.querySelector('.studio-sec-h .grow'));
      box.querySelector('.studio-sec-h').insertBefore(dirty, button);
      return save;
    };
    const MODE_NAMES = {he_only:'H&E only', he_with_codex_edge_rescue:'H&E, markers at edges', he_codex_union:'H&E or markers',
      codex_only:'Markers only'};
    const TISSUE_KEYS = ['tissue_mask_mode', 'he_od_threshold_factor', 'tissue_morph_radius', 'min_tissue_area_um2', 'detect_blood_clot', 'blur_downsample', 'fill_holes_um2'];
    const tissuePanel = (current, holder) => {
      const {box, body} = section('Tissue settings', {tip:'Settings for finding the tissue boundary.'});
      const dirty = chip('Saved', 'ok');
      const polygons = section('Tissue polygons', {tip:'Each connected piece is separate. Click one, then use Move, Rectangle, Polygon or Freehand with Shift-drag to paint or Alt-drag to erase. ' +
        'Hold the same key and turn the mouse wheel to resize the brush. Ctrl+Z undoes a stroke. History keeps earlier versions.'});
      const tissueList = node('div', {id:'studio-tissue-polygons'}); polygons.body.append(tissueList);
      polygons.body.prepend(node('button',{class:'btn small',text:'Import tissue…',onclick:()=>hooks.importTissue?.(),
        title:'Choose an annotation class to use as editable Tissue.'}));
      holder.append(box, polygons.box);
      workWithin(polygons.body, 'tissue', () => { current.drawShapes?.(); current.updatePolygonSave?.(); });
      current.drawShapes = () => { if (tissueList.isConnected && !renderShapeLayer(tissueList, activeTissueId?.() || 'tissue-regions')) tissueList.replaceChildren(node('p', {class:'muted', text:'Open an image from the list.'})); };
      current.drawShapes();
      let saved = null, values = {};
      const changes = () => Object.fromEntries(TISSUE_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); };
      const number = (key, label, step) => {
        const input = node('input', {type:'number', step, min:'0', value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= 0) { values[key] = value; mark(); } else input.value = String(values[key]); });
        return input;
      };
      const draw = () => {
        const mode = choice(saved.modes.map(name => [name, MODE_NAMES[name] || name]), values.tissue_mask_mode, 'Tissue mode');
        mode.addEventListener('change', () => { values.tissue_mask_mode = mode.value; mark(); });
        const plan = current.annotationPlan || {label:'tissue', detail:[1, 2, 4, 8, 16], microns_per_pixel:0.5083};
        const label = node('input', {type:'text', value:plan.label, maxlength:'40', 'aria-label':'Tissue label'});
        label.addEventListener('change', () => { if (label.value.trim()) current.annotationStepSave=current.saveAnnotationStep?.({label:label.value.trim()}); });
        const detail = choice(plan.detail.map(step => [String(step), scaleLabel(step, imageSize?.()?.microns_per_pixel || plan.microns_per_pixel)]),
          String(values.blur_downsample || 2), 'Detection scale');
        detail.addEventListener('change', () => { values.blur_downsample = Number(detail.value); mark(); });
        body.replaceChildren(
          field('Label', label, 'The name of the tissue in Studio and in the export. Each connected piece becomes <label>_1, <label>_2… (a tissue region).'),
          field('Scale', detail, 'The image\u2019s resolution the tissue and artifacts are found at (a pyramid level): Full is the most detailed and ' +
            'slowest; 1/2 is the default; 1/4 or 1/8 are quicker for large images. Each artifact has its own saved scale.'),
          field('Mode', mode, 'What finds tissue: the H&E layers (default), with markers at the edges, either, or markers only.'),
          field('Stain level', number('he_od_threshold_factor', 'Stain threshold factor', '0.05'), 'Multiplies the automatic H&E threshold: lower keeps fainter tissue (default 0.85).'),
          field('Smoothing', number('tissue_morph_radius', 'Smoothing radius', '1'), 'Radius in pixels used to close small gaps (default 11).'),
          field('Smallest', node('div', {class:'inline'}, [number('min_tissue_area_um2', 'Smallest piece', '100'), node('span', {class:'muted', text:'µm²'})]),
            'Discard tissue pieces smaller than this area, after making the final outlines. The area stays the same when Scale changes. 0 keeps every piece.'),
          field('Fill holes', node('div', {class:'inline'}, [number('fill_holes_um2', 'Fill holes smaller than (µm²)', '10'), node('span', {class:'muted', text:'µm²'})]),
            'Fill enclosed holes smaller than this area. 0 keeps every hole. Larger holes and gaps between separate pieces stay. Uses the image’s pixel size, or the project’s value if it is missing. Detect tissue saves the result automatically.'));
        mark();
      };
      const load = async () => {
        try { saved = await get(route('/tissue_settings')); values = JSON.parse(JSON.stringify(saved.values)); if (holder.isConnected) draw(); }
        catch (error) { body.replaceChildren(node('p', {class:'hint', text:error.message})); }
      };
      let save = async () => {
        const pending = changes(); if (!Object.keys(pending).length) return;
        saved = await post(route('/tissue_settings'), {changes:pending, revision:saved.revision});
        mark();
      };
      const status = node('p', {class:'muted', role:'status'});
      save = headerSave(box, dirty, save, status, current, holder, () => !!saved && Object.keys(changes()).length > 0, mark);
      const polygonStatus = chip('Open an image'); polygonStatus.setAttribute('role', 'status');
      const savePolygons = node('button', {class:'btn primary', text:'Save polygons', 'data-save-tissue':'',
        title:'Save the Tissue polygons currently shown. Finishes waiting edits or saves the reviewed preview; does not run detection again.'});
      let savingPolygons = false;
      current.updatePolygonSave = () => {
        if (!holder.isConnected) return;
        const info = annotationState?.()[activeTissueId?.() || 'tissue-regions'];
        const busy = savingPolygons || info?.busy;
        const failed = !!(info?.error || info?.loadError);
        savePolygons.disabled = !info || !info.editable || !!(busy || info.loading || info.loadError) || (!info.saved && !info.preview && !info.count);
        const text = !info ? 'Open an image' : info.loading ? 'Loading…' : busy ? 'Saving…' : failed ? 'Save not confirmed' :
          info.pending ? 'Not saved' : info.saved ? 'Saved' : 'No polygons';
        polygonStatus.textContent = text;
        polygonStatus.className = 'chip ' + (text === 'Saved' ? 'ok' : info?.pending || failed ? 'warn' : '');
        polygonStatus.title = info?.loadError || info?.error || (text === 'Saved' ? info.count + ' polygons saved in the project.' :
          'Green T means saved tissue. Amber T means a preview or edits still need saving.');
      };
      savePolygons.onclick = async () => {
        if (savingPolygons) return;
        if (status.textContent.startsWith('Could not save polygons:')) status.textContent = '';
        savingPolygons = true; current.updatePolygonSave?.();
        try { await saveTissuePolygons?.(); }
        catch (error) { status.textContent = 'Could not save polygons: ' + error.message; }
        finally { savingPolygons = false; current.updatePolygonSave?.(); current.updateAnnotationStatus?.(); }
      };
      // Remove small pieces: the open image's tissue at once (Ctrl+Z undoes), or each ticked image's as a job.
      const smallest = node('input', {type:'number', min:'0', step:'100', placeholder:'µm²', 'aria-label':'Remove pieces smaller than (µm²)', class:'studio-area-input'});
      const smallValue = () => { const n = Number(smallest.value); if (!(n > 0)) { status.textContent = 'Enter the smallest piece to keep, in µm².'; return null; } return n; };
      const smallOne = node('button', {class:'btn', text:'This image', title:'Remove the open image’s tissue pieces smaller than this', onclick:() => {
        const n = smallValue(); if (n === null) return;
        const removed = hooks.removeSmaller?.(activeTissueId?.() || 'tissue-regions', n);
        status.textContent = removed === null ? 'Open an image with saved tissue first.' : removed ? removed + ' piece' + (removed === 1 ? '' : 's') + ' under ' + n + ' µm² removed. Ctrl+Z brings them back.' : 'No piece is smaller than ' + n + ' µm².';
      }});
      const smallSome = node('button', {class:'btn primary', text:'Selected', title:'Remove the small tissue pieces of each ticked image, in the background', onclick:async () => {
        const n = smallValue(); if (n === null) return;
        const images = Array.from(current.annotationPicked || []);
        if (!images.length) { status.textContent = 'Tick the images first.'; return; }
        if (!window.confirm('Remove the tissue pieces smaller than ' + n + ' µm² in ' + images.length + ' ticked image' + (images.length === 1 ? '' : 's') + '? Earlier versions stay in History.')) return;
        smallSome.disabled = true;
        try { await flushShapes?.(); await post(route('/remove_small_pieces'), {image_token:(currentSpec?.layers || []).find(l => l.type === 'image')?.token, images, min_um2:n});
          status.textContent = 'Removing small pieces from ' + images.length + ' image' + (images.length === 1 ? '' : 's') + ' in the background.'; await refreshJobs(); showTab('jobs'); }
        catch (error) { status.textContent = error.message; }
        finally { smallSome.disabled = false; }
      }});
      const piecesRow = node('div', {class:'row wrap'}, [node('span', {class:'lbl', text:'Remove small pieces:'}), smallest, node('span', {class:'muted', text:'µm²'}), smallOne, smallSome,
        help('Deletes the tissue pieces (polygons) smaller than this area: of the open image at once (Ctrl+Z brings them back), or of each ticked image in the background (each image’s own pixel size; earlier versions stay in History). The same as Remove smaller in the Tissue list.')]);
      const polygonSaveRow = node('div', {class:'row'}, [savePolygons, polygonStatus,
        help('Saves the current Tissue polygons. Existing tissue edits also save in the background. A preview stays unsaved until you save it. Green T means saved tissue; amber means changes are waiting. Earlier versions remain in History.')]);
      current.runBar([stageRow(current, 'Detect tissue', 'detect_tissue', {save, isDirty:() => !!saved && Object.keys(changes()).length > 0, status,
        started:() => { clearTissuePreview?.(); showSegPreview(null); },
        onImage:async () => {
          status.textContent='Detecting and saving tissue…';
          const token=(currentSpec?.layers||[]).find(l=>l.type==='image')?.token;
          const target=await post(route('/tissue_versions'),{image_token:token,canonical:true});
          if(!target.selection)throw Error('Studio’s server needs restarting to load the Tissue update. Start the updated launcher, open its printed URL, and reload this page.');
          const found=await hooks.updateWorkingTissue(target.layer, revision=>post(route('/tissue_preview'),{
            image_token:token,save:true,revision,settings:Object.fromEntries(TISSUE_KEYS.map(key=>[key,values[key]]))}));
          if(!found || !holder.isConnected)return;
          await chooseTissue?.(found.layer);
          status.textContent='Saved '+found.features.length+' tissue polygons · '+found.seconds+' s. Ctrl+Z undoes.';
          // The list and Work within catch up in the background: Detect is ready again as soon as the tissue is saved.
          (async () => {
            await current.refreshAnnotationPlan?.();
            await current.refreshTissueScope?.();
          })().catch(() => {});
        },
        tip:'Finds tissue as editable pieces. Earlier tissue is kept in History. Review artifacts in the next stage before subtracting them.',
        ask:(which, images) => !(which === 'picked' ? current.annotationPlan?.rows.some(row => images.includes(row.image) && row.done.tissue) : annotationState?.()['tissue-regions']?.saved) ||
          window.confirm('Detect tissue again and replace the current tissue, including hand edits? The earlier version stays in History.')}), piecesRow, status, polygonSaveRow]);
      current.updatePolygonSave();
      current.imageReady=()=>{if(holder.isConnected)current.refreshTissueScope?.();};
      load();
      const piecesHolder = node('div', {class:'tissue-pieces-holder'});
      polygons.box.before(piecesHolder);
      piecesPanel(current, piecesHolder);
    };

    // Artifact parts: each slider is one of Tab 4's settings; moving it redraws that
    // artifact on the image at once. An example drawn over an artifact sets the slider.
    const workWithin = async (holder, stage, changed) => {
      const box=node('div', {class:'annotation-scope'});holder.prepend(box);
      let image=(currentSpec?.layers||[]).find(l=>l.type==='image');if(!image)return;
      let data, selection, busy=false, sequence=0;
      const status=node('span',{class:'muted',role:'status'});
      const pending=choice([['tissue','Tissue']], 'tissue',stage==='tissue'?'Tissue annotation':'Work within');pending.disabled=true;
      box.append(field(stage==='tissue'?'Edit':'Work within',pending,'Loading available annotations…'),status);
      const active=()=>box.isConnected&&(currentSpec?.layers||[]).some(l=>l.type==='image'&&l.token===image.token);
      const save=async next=>{
        busy=true;status.textContent='Saving…';
        box.querySelectorAll('select').forEach(input=>input.disabled=true);
        try{await flushShapes?.();const result=await post(route('/annotation_sources'),{image_token:image.token,stage,selection:next});selection=result.selection;status.textContent='';if(active()){await load();changed?.();}}
        catch(error){status.textContent=error.message;}
        finally{busy=false;if(active())box.querySelectorAll('select').forEach(input=>input.disabled=!!data?.read_only);}
      };
      const draw=async()=>{
        if(!active())return;
        const tissue=data.tissue || [];
        // Older saved choices may reference the same tissue file by its path.
        // Keep their scope, but display it only once under its named version.
        const alias=selection.mode==='file'&&tissue.find(t=>t.path===selection.path);
        const value=selection.mode==='auto'?data.automatic:alias?alias.mode:
          selection.mode==='file'?'file:'+selection.path:selection.mode;
        const select=choice(stage==='tissue'?[]:[['image','Whole image']],null,stage==='tissue'?'Tissue annotation':'Work within');
        const group=(label,items)=>{
          if(!items.length)return;
          select.append(node('optgroup',{label},items.map(item=>node('option',{
            value:item.value,text:item.name,disabled:!!item.disabled,title:item.title||''}))));
        };
        // As Layers shows them: the image (its tissue, then every name), each imported file (all, then its names), other files.
        const imageName=String(image.name||'This image').replace(/(\.ome)?\.(tiff?|qptiff|svs)$/i,'');
        const names=(data.names||[]), named=(item,source)=>({value:'name:'+JSON.stringify({path:item.files[0],name:item.name}),
          name:item.name+(item.count>1?' ('+item.count+' polygons)':''),title:item.files.length>1?'In '+item.files.length+' layers; the first is used':item.files[0]});
        const tissuePath=tissue.find(t=>t.available)?.path, tissueName=tissue[0]?.name;
        const own=names.filter(n=>!n.source&&!(n.name===tissueName&&n.files.length===1&&n.files[0]===tissuePath&&!names.some(o=>!o.source&&o.name!==tissueName&&o.files.includes(tissuePath))));
        group(imageName,[...tissue.filter(t=>t.available||t.mode===value).map(t=>({
          value:t.mode,name:t.name+(t.available?(own.some(n=>n.files.includes(t.path))?' (all pieces)':''):' (not saved)'),disabled:!t.available,
          title:t.available?t.path:'Save these polygons in the Tissue tab first.'})),...own.map(item=>named(item,''))]);
        data.files.filter(f=>f.kind==='imported').forEach(f=>{
          const label=f.name, mine=names.filter(n=>n.source===label);
          group(label,[{value:'file:'+f.path,name:'All polygons',title:f.path},...mine.map(item=>named(item,label))]);
        });
        const others=data.files.filter(f=>f.kind!=='imported');
        group('Other annotation files',others.map(f=>({value:'file:'+f.path,name:f.name,title:f.path})));
        // A name chosen earlier: its option, whichever file it is in.
        if(selection.mode==='file'&&selection.class_name){
          const option=Array.from(select.options).find(o=>o.value.startsWith('name:')&&(()=>{const v=JSON.parse(o.value.slice(5));return v.path===selection.path&&v.name===selection.class_name;})());
          if(option)option.selected=true;
        }
        if(!Array.from(select.options).some(o=>o.value===value)&&!(select.selectedOptions[0]?.value||'').startsWith('name:')){
          select.append(node('option',{value,text:'Saved annotation unavailable',disabled:true,title:selection.path||''}));
        }
        const byName=(select.selectedOptions[0]?.value||'').startsWith('name:');
        if(!byName)select.value=value;
        select.disabled=!!data.read_only||busy;
        select.onchange=()=>{if(busy)return;
          if(select.value.startsWith('name:')){const v=JSON.parse(select.value.slice(5));save({mode:'file',path:v.path,class_name:v.name});return;}
          save(select.value.startsWith('file:')?{mode:'file',path:select.value.slice(5),class_name:''}:{mode:select.value,class_name:''});};
        box.replaceChildren(field(stage==='tissue'?'Edit':'Work within',select,'Choose the annotation to use. Subtraction updates and saves this layer; Ctrl+Z or History restores earlier versions. The choice follows this image into the next step. For an external file, Studio creates an editable project copy. Hiding a layer does not change this area.'),status);
        if(selection.mode==='file' && !byName && (!alias || selection.class_name)){
          try{const found=await get(route('/annotation_sources?stage='+stage+'&path='+encodeURIComponent(selection.path)));
            if(!active())return;
            if(alias && found.classes?.length===1 && found.classes[0]===selection.class_name)return;
            const classes=choice([['','All polygons'],...(found.classes||[]).map(n=>[n,n])],selection.class_name||'','Annotation name');
            if(selection.class_name && !(found.classes||[]).includes(selection.class_name)){
              classes.append(node('option',{value:selection.class_name,text:selection.class_name+' (unavailable)',disabled:true}));
              classes.value=selection.class_name;
            }
            classes.disabled=!!data.read_only||busy;
            classes.onchange=()=>{if(!busy)save({...selection,class_name:classes.value});};box.append(field('Class',classes,'Choose one named group inside this annotation file, or use all its polygons.'));
          }catch(error){status.textContent=error.message;}
        }
      };
      const load=async()=>{
        image=(currentSpec?.layers||[]).find(l=>l.type==='image');if(!image)return;
        const request=++sequence, token=image.token;
        const current=()=>request===sequence&&active()&&image.token===token;
        try {
          const next=await get(route('/annotation_sources?stage='+stage));if(!current())return;data=next;selection=data.selection;
          await draw();if(!current())return;
          if(selection.mode!=='image'){
            const target=await post(route('/tissue_versions'),{image_token:image.token,stage});
            if(!target.selection)throw Error('Restart Studio with the updated launcher to load the Tissue update.');
            if(!current())return;
            if(stage==='tissue')await chooseTissue?.(target.layer,false);
            else await ensureArtifactLayer?.(target.layer);
            data=await get(route('/annotation_sources?stage='+stage));selection=data.selection;
          }
          await draw();
        } catch(error){if(current()){status.textContent=error.message;if(!box.contains(status))box.append(status);}}
      };
      if(stage==='tissue'&&frame)frame.refreshTissueScope=load;
      await load();
    };
    const artifactReview = window.createArtifactReview({node,field,choice,help,get,post,route,workWithin,scaleLabel,pixelSize:()=>imageSize?.()?.microns_per_pixel || frame?.annotationPlan?.microns_per_pixel,
      image:()=>(currentSpec?.layers || []).find(l=>l.type==='image'),
      ensureLayer:ensureArtifactLayer, showPreview:showTissuePreview, savePolygons:saveArtifactPolygons,
      reviewed:reviewedArtifact, needsReview:artifactNeedsReview,
      drawExamples:drawArtifactExamples, exampleState:artifactExampleState, finishExamples:finishArtifactExamples,
      reloadExamples:hooks.reloadArtifactExamples, manualArtifacts:hooks.manualArtifacts,
      setTarget:setDrawTarget, drawingState:hooks.drawingState, renderLayer:renderShapeLayer, annotationState, flush:flushShapes,
      command:runCommand, result:answer=>showResult(answer), rebuild:()=>hooks.rebuildLayers?.(),
      updateWorkingTissue:hooks.updateWorkingTissue,
      picked:()=>Array.from(frame?.annotationPicked||[]),
      jobs:()=>{refreshJobs();showTab('jobs');}, refreshClean:()=>{}});
    const artifactPanel = (current, holder) => artifactReview.panel(current,holder);

    const classesPanel = window.createClassification({node,field,choice,help,section,get,post,route,workWithin,
      image:()=>(currentSpec?.layers||[]).find(l=>l.type==='image'),
      selectedMarkers:()=>hooks.selectedImageMarkers?.()||[],
      scaleLabel:(n,mpp)=>scaleLabel(n,mpp||imageSize?.()?.microns_per_pixel||frame?.annotationPlan?.microns_per_pixel),
      showRaster,heatmap:heatmapFigure,capture:captureView,flush:flushShapes,
      scopeName:()=>document.querySelector('[aria-label="Work within"]')?.selectedOptions[0]?.textContent,
      showPolygons:async found=>{await hooks.showClassificationPolygons?.(found);await frame?.refreshAnnotationPlan?.();},
      picked:()=>Array.from(frame?.annotationPicked||[]),jobs:()=>{refreshJobs();showTab('jobs');},refreshPlan:()=>frame?.refreshAnnotationPlan?.(),
      aiMessage:(text,user=false)=>{showTab('assistant');addMessage(user?'user':'assistant',text);}});

    // One request at a time; the newest settings always win. Returns a function that asks again.
    const latest = ask => {
      let busy = false, again = false;
      const go = async () => {
        if (busy) { again = true; return; }
        busy = true;
        try { await ask(); } finally { busy = false; if (again) { again = false; go(); } }
      };
      return go;
    };
    // Marker regions: where every chosen marker is above its level, smoothed, inside the tissue (or the whole image);
    // the regions are saved as the image's regions layer, to review, name, split or delete, then export.
    const regionsPanel = async (current, holder) => {
      const scopeHolder=node("div");
      const {box, body:settingsBody} = section('Marker settings', {tip:'Each row is a marker and the level it must be above; a pixel is kept ' +
        'where every row holds (for example Pan-Cytokeratin above 100 and E-Cadherin above 200). Smoothing blurs the markers first ' +
        '(sigma, in pixels at the chosen scale) so regions are not ragged. Levels are the image’s own values, as in the image card.'});
      const body=node('div');settingsBody.append(scopeHolder,body);
      const scopeReady=workWithin(scopeHolder,'regions',()=>showPreview('marker-regions-preview',null));
      const shown = section('Region polygons', {tip:'The regions found, as polygons: click one on the image to see its card ' +
        '(size, markers high in it); Ctrl-click adds more; Select (above the image) selects every region inside an outline. ' +
        'Use Apply label in the list or polygon card. Split and Merge use the viewer toolbar. Sort by place or size, zoom with Go, or remove small pieces. ' +
        'Changes are saved as you make them; Ctrl+Z undoes.'});
      const list = node('div', {id:'studio-marker-regions'});
      const editHolder=node('div');shown.body.append(editHolder,list);
      let editId='marker-regions', difference=null, differenceTimer=null, lastDifferenceRevision='';
      holder.append(box, shown.box);
      const status = node('p', {class:'muted', role:'status'});
      const line = status;
      current.refreshDifference = () => {
        if(!list.isConnected)return;
        const revision=JSON.stringify(Object.entries(annotationState?.()||{}).map(([id,value])=>[id,value.revision]));
        if(revision!==lastDifferenceRevision){lastDifferenceRevision=revision;clearTimeout(differenceTimer);
          differenceTimer=setTimeout(()=>difference?.refresh(),400);}
      };
      current.drawShapes = () => {
        if (!list.isConnected) return;
        if (!renderShapeLayer(list, editId)) list.replaceChildren(node('p', {class:'muted', text:'Open an image from the list to review its regions.'}));
        current.refreshDifference?.();
      };
      current.drawShapes();
      let found;
      try { await scopeReady; found = await get(route('/marker_regions')); }
      catch (error) { body.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      const values = JSON.parse(JSON.stringify(found.values));
      let saved = JSON.stringify(found.values);
      const dirty = chip('Saved', 'ok');
      const isDirty = () => JSON.stringify(values) !== saved;
      const mark = () => { const d = isDirty(); dirty.textContent = d ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (d ? 'warn' : 'ok'); };
      let save = async () => { const sent = JSON.stringify(values); await post(route('/marker_regions'), {values:JSON.parse(sent)}); saved = sent; mark(); };
      const changed = () => mark();
      save = headerSave(box, dirty, save, status, current, holder, isDirty, mark);
      const mpp = found.microns_per_pixel || current.annotationPlan?.microns_per_pixel;
      const draw = () => {
        const markers = found.markers || [];
        const rules = values.rules.map((rule, index) => {
          const marker = choice((markers.includes(rule.marker) ? markers : [rule.marker, ...markers]).map(name => [name, name]), rule.marker, 'Marker ' + (index + 1));
          marker.addEventListener('change', () => { rule.marker = marker.value; changed(); });
          const level = node('input', {type:'number', step:'any', value:String(rule.above), 'aria-label':'Level of marker ' + (index + 1)});
          level.addEventListener('change', () => { rule.above = Number(level.value); changed(); });
          const drop = node('button', {class:'btn small ghost', text:'Remove', onclick:() => { values.rules.splice(index, 1); draw(); changed(); }});
          const [before, after] = {raw:['above', ''], auto:['above', '× its level'], saved:['above', '× its level'], z:['z above', ''],
            groups:['at least', 'sure to be signal (0.9 = 90%)']}[values.cutoff] || ['above', ''];
          return node('div', {class:'row studio-rule'}, [marker, node('span', {class:'muted', text:before}), level,
            after ? node('span', {class:'muted', text:after}) : null, drop].filter(Boolean));
        });
        const add = node('button', {class:'btn small', text:'Add a marker', disabled:markers.length ? null : '', onclick:() => {
          values.rules.push({marker:markers.find(name => !values.rules.some(rule => rule.marker === name)) || markers[0],
            above:(found.rule_default || {})[values.cutoff] ?? 100}); draw(); changed();
        }});
        // How each rule's level is read; computed for each image from its own area, so one setting suits all images.
        const CUTOFF_NAMES = {raw:'Raw intensity', auto:'Auto level', saved:'Saved level', z:'Arcsinh z-score', groups:'Two groups'};
        const pool = choice([['slide', 'Slide (all its ROIs together)'], ['image', 'Image']], values.pool || 'slide', 'Fit two groups per');
        pool.addEventListener('change', () => { values.pool = pool.value; changed(); });
        const cutoff = choice((found.cutoffs || ['raw']).map(key => [key, CUTOFF_NAMES[key] || key]), values.cutoff || 'raw', 'Cutoff');
        cutoff.addEventListener('change', () => {
          values.cutoff = cutoff.value;
          values.rules.forEach(rule => { rule.above = (found.rule_default || {})[values.cutoff] ?? rule.above; });  // the new kind's usual value
          draw(); changed();
        });
        const method = choice((found.methods || ['Triangle']).map(name => [name, name]), values.method, 'Auto method');
        method.addEventListener('change', () => { values.method = method.value; changed(); draw(); });
        const cofactor = node('input', {type:'number', min:'0.01', step:'any', value:String(values.cofactor), 'aria-label':'Arcsinh cofactor'});
        cofactor.addEventListener('change', () => { const n = Number(cofactor.value); if (n > 0) { values.cofactor = n; changed(); } else cofactor.value = String(values.cofactor); });
        const number = (key, label, attrs) => {
          const input = node('input', {type:'number', value:String(values[key]), 'aria-label':label, ...attrs});
          input.addEventListener('change', () => { values[key] = Number(input.value); changed(); });
          return input;
        };
        const scale = choice(found.scales.map(step => [String(step), scaleLabel(step, imageSize?.()?.microns_per_pixel || mpp)]), String(values.scale), 'Region scale');
        scale.addEventListener('change', () => { values.scale = Number(scale.value); changed(); });
        const name = node('input', {type:'text', value:values.name, maxlength:'60', placeholder:'e.g. Epithelia', 'aria-label':'Name the regions'});
        name.addEventListener('change', () => { values.name = name.value.trim(); changed(); });
        const preview = node('button', {class:'btn small', text:'Preview', title:'Find them on the open image and draw them, without saving', onclick:async () => {
          preview.disabled = true; line.textContent = 'Looking…';
          try {
            const image = currentSpec?.id;
            const answer = await post(route('/marker_regions'), {values, preview:true,image_token:(currentSpec?.layers||[]).find(l=>l.type==='image')?.token});
            if (!holder.isConnected || currentSpec?.id !== image) return;
            showPreview('marker-regions-preview', {outlines:answer.outlines, colour:'#ffd600', kind:'example'}); clear.hidden = false;
            const levels = (answer.cutoffs || []).map(item => item.marker + ' above ' + Number(item.raw.toPrecision(4)) +
              ('sure' in item ? ' (' + Math.round(item.sure * 100) + '% sure, ' + item.fitted + ' ROI' + (item.fitted === 1 ? '' : 's') +
                ', separation ' + item.separation + (item.separation < 2 ? ': weak signal' : '') + ')' :
               'z' in item ? ' (z ' + item.z + ')' : 'level' in item ? ' (' + (item.method || 'saved') + ' level ' + Number(item.level.toPrecision(4)) + ')' : '')).join('; ');
            line.textContent = answer.regions + ' region' + (answer.regions === 1 ? '' : 's') + ' · ' + Math.round(answer.area_um2).toLocaleString() + ' µm² · read at ' +
              answer.level_microns + ' µm/px · ' + answer.ms + ' ms' + (levels ? ' · ' + levels : '') + ' (a preview: Detect regions saves them)';
          } catch (error) { line.textContent = error.message; }
          finally { preview.disabled = false; }
        }});
        const clear = node('button', {class:'btn small ghost', text:'Clear', hidden:true, onclick:() => { showPreview('marker-regions-preview', null); clear.hidden = true; line.textContent = ''; }});
        fill(body,
          field('Scale',scale,'Resolution used for threshold detection and saved polygons.'),
          field('Cutoff', cutoff, 'How each marker\u2019s level is set, for each image from its own Work within area (the clean tissue once ' +
            'artifacts are subtracted): Raw intensity, a number you type. Auto level: the image\u2019s background level by the chosen method ' +
            '(the right panel\u2019s Auto methods), times a factor (1 = the level). Saved level: the min shown in the right panel (Auto or by hand) ' +
            'or the Background step\u2019s level, times a factor. Arcsinh z-score: intensities are turned into arcsinh(x / cofactor), then into ' +
            'how many robust standard deviations (median and MAD of the area) above the area\u2019s middle they are; 1.5 is a usual start. ' +
            'Two groups: the tissue\u2019s pixels are split into background and signal; a pixel counts when it is at least this sure to be signal. ' +
            'Separation below 2 means the marker has no clear signal. ' +
            'The raw level each image got is shown after Preview and Detect.'),
          values.cutoff === 'auto' ? field('Method', method, 'The Auto method, as in the right panel (Triangle, Otsu, Geometric elbow…).') : null,
          values.cutoff === 'auto' && values.method === 'Percentile' ? field('Percentile',number('percentile','Percentile',{min:0,max:100,step:'any'})) : null,
          values.cutoff === 'auto' && values.method === 'Manual' ? field('Manual cutoff',number('manual','Manual cutoff',{step:'any'})) : null,
          values.cutoff === 'z' ? field('Cofactor', cofactor, 'arcsinh(intensity / cofactor) before the z-score. 5 is usual; larger values make low intensities count less.') : null,
          values.cutoff === 'groups' ? field('Fit per', pool, 'Slide: all ROIs of a slide share one cutoff (the Samples table says which slide). Image: each ROI gets its own.') : null,
          markers.length ? null : node('p', {class:'hint', text:'Open an image to choose among its markers.'}),
          ...rules, node('div', {class:'row'}, [add]),

          field('Smoothing', number('sigma', 'Smoothing sigma', {min:'0', max:'50', step:'0.5'}), 'Sigma of the blur applied to each marker first, in pixels at the chosen scale. 0 is none; 1 to 3 smooths the edges.'),
          field('Smallest', node('div', {class:'inline'}, [number('min_area_um2', 'Smallest region in square micrometres', {min:'0', step:'50'}), node('span', {class:'muted', text:'µm²'})]),
            'Regions smaller than this are left out.'),
          field('Result label', name, 'Label for the next detection and its new layer. Leave empty for unnamed regions. A different label creates another layer; rerunning the same label and Work within replaces that result with History. To label existing polygons, select them below and use Apply label.'),
          node('div', {class:'row'}, [preview, clear]));
      };
      const loadLayers=async preferred=>{
        if(!holder.isConnected)return;
        found=await get(route('/marker_regions'));if(!holder.isConnected)return;
        for(const layer of found.layers||[])await ensureArtifactLayer?.(layer);
        const layers=found.layers||[];
        editId=preferred||layers.find(l=>l.id===editId)?.id||layers.find(l=>l.saved_file===found.target)?.id||layers[0]?.id||'marker-regions';
        editHolder.replaceChildren();
        if(layers.length){const edit=choice(layers.map(l=>[l.id,l.name]),editId,'Edit marker region layer');
          edit.onchange=()=>{editId=edit.value;setDrawTarget?.(editId);current.drawShapes();};
          editHolder.append(field('Edit',edit,'Choose a saved result to split, name or delete. Changes save automatically and can be undone.'));
        }
        setDrawTarget?.(editId);current.drawShapes();
      };
      draw();await loadLayers();
      difference=window.createAnnotationDifference({node,help,get,post,route,
        image:()=>(currentSpec?.layers||[]).find(l=>l.type==='image'),flush:flushShapes,showPreview,
        readOnly:()=>!!found.read_only,picked:()=>Array.from(current.annotationPicked||[]),jobs:()=>{refreshJobs();showTab('jobs');},
        showResult:async answer=>{await hooks.showClassificationPolygons?.(answer);await loadLayers(answer.layer.id);await current.refreshAnnotationPlan?.();}});
      shown.box.before(difference.box);
      const rows = () => current.annotationPlan?.rows || [];
      current.runBar([stageRow(current, 'Detect regions', 'marker_regions', {save, isDirty, status,
        onImage:async()=>{status.textContent='Detecting…';const answer=await runCommand('marker_regions',{scope:'image'},{confirmed:true});showResult(answer);status.textContent=answer.message;showPreview('marker-regions-preview',null);await loadLayers(answer.data?.layer?.id);await current.refreshAnnotationPlan?.();},
        tip:'Finds the regions with these settings and saves them as the image’s regions (replacing earlier ones), for the open image or each ticked one. ' +
          'Then use the drawing toolbar to split, name and review them. Saved layers can be selected in Classification or Export.',
        ask:(which, images) => {
          const named = which === 'picked' ? rows().filter(row => images.includes(row.image) && row.done.regions).map(row => row.image) :
            (list.querySelector('.region-list .shape-row') ? ['the open image'] : []);
          return !named.length || window.confirm('Detect regions again with this label and area? If that result already exists, its earlier polygons and names stay in History.');
        }}), status]);
      current.planChanged=()=>loadLayers().catch(e=>{status.textContent=e.message;});
      current.refreshMarkers = () => { get(route('/marker_regions')).then(answer => { found.markers = answer.markers; if (body.isConnected) draw(); }).catch(() => {}); };
    };

    // Neck review lives inside Tissue. Finding candidates never changes tissue.
    const piecesPanel = async (current, holder) => {
      const state = current.pieces || (current.pieces = {});
      try { state.saved = await get(route('/pieces_settings')); }
      catch (error) { holder.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.values) state.values = {...state.saved.values};
      const keys = ['min_area_mm2','filter_by_area','detect_necks','neck_min_width_um','cut_at_necks',
        'neck_min_lobe_area_mm2','neck_min_lobe_fraction','filter_by_inscribed_circle','min_inscribed_diameter_um'];
      const changes = () => Object.fromEntries(keys.filter(key => state.values[key] !== state.saved.values[key]).map(key => [key,state.values[key]]));
      const {box,body} = section('Necks and pieces', {tip:'Turn on when needed, then Preview. Red marks suggested cuts: right-click one to approve or reject. Go zooms to it. Each side must meet both size limits. Ctrl+Z undoes a cut.'});
      const enabled = node('input',{type:'checkbox','aria-label':'Use neck tools'});
      enabled.checked = recall('annotation-neck-tools','off') === 'on';
      box.querySelector('.studio-sec-h').prepend(enabled);
      body.hidden = !enabled.checked;
      const dirty = chip('Saved','ok'), line = node('p',{class:'muted',role:'status'}), status = node('p',{class:'muted',role:'status'});
      const mark = () => { const d=Object.keys(changes()).length; dirty.textContent=d?'Not saved':'Saved'; dirty.className='chip '+(d?'warn':'ok'); };
      const list = node('div',{class:'neck-review-list'});
      const reviewTitle=node('summary',{text:'Necks'}),reviewBox=node('details',{class:'neck-review',hidden:true},[reviewTitle,list]);
      const restoreRejected=node('button',{class:'btn small ghost',hidden:true});reviewBox.append(restoreRejected);
      let found=null, original=null, image=null, busy=false, request=0, hidden=false, selected=null;
      const checkpoints=[];
      const rejected = new Set(), approved = new Set();
      const clearOverlays = () => { showPreview('pieces-kept',null); showPreview('pieces-dropped',null); showNeckReview?.(null); };
      const invalidate = () => { request++; found=null; original=null; checkpoints.length=0; list.replaceChildren();reviewBox.hidden=true; clearOverlays(); line.textContent='Preview out of date.'; };
      const valid = () => enabled.checked && holder.isConnected && image===currentSpec?.id && original?.features===tissueForPieces?.()?.features;
      const checkpoint = () => {checkpoints.push({tissue:original,approved:new Set(approved)});if(checkpoints.length>30)checkpoints.shift();};
      const sameGeometry = (a,b) => a?.token===b?.token && a?.image_token===b?.image_token && a?.features?.length===b?.features?.length &&
        ((!a.preview&&!b.preview&&a.revision&&a.revision===b.revision) ||
          a.features.every((f,i)=>f===b.features[i] || f.properties.id===b.features[i].properties.id && JSON.stringify(f.geometry)===JSON.stringify(b.features[i].geometry)));
      current.syncNecks = () => {
        if(!found||busy||!holder.isConnected||!enabled.checked)return;
        const tissue=tissueForPieces?.();
        if(tissue?.busy)return;
        if(image!==currentSpec?.id||!tissue){invalidate();return;}
        if(tissue.features===original?.features)return;
        // Reload/save can replace the array without changing its geometry.
        // Undo/redo can revisit one of this review's earlier cut states.
        const known=[...checkpoints].reverse().find(c=>sameGeometry(c.tissue,tissue));
        if(!known){invalidate();return;}
        original=tissue;approved.clear();known.approved.forEach(id=>approved.add(id));draw();
      };
      enabled.onchange = () => {
        body.hidden=!enabled.checked; remember('annotation-neck-tools',enabled.checked?'on':'off');
        state.values.detect_necks=enabled.checked;mark();if(!enabled.checked)invalidate();
      };
      const recordNeck=async(operation,neck)=>post(route('/annotation_workflow'),{
        image_token:currentSpec?.layers?.find(l=>l.type==='image')?.token,
        event:{operation,neck,source:found?.layer?.saved_file||found?.source,revision:original?.revision,preview:!!original?.preview,settings:{...state.values}}});
      const reject = neck => {
        if(busy)return;
        current.syncNecks?.();
        if(!valid()) { invalidate(); return; }
        const seq=request;
        rejected.add(neck.id);draw();
        recordNeck('reject_neck',neck).catch(error=>{if(seq===request&&valid()){rejected.delete(neck.id);draw();line.textContent=error.message;}});
      };
      const approve = async neck => {
        current.syncNecks?.();
        if(busy || !valid()) { if(!valid()) invalidate(); return; }
        if(approved.has(neck.id)||rejected.has(neck.id))return;
        const seq=request;
        let errorMessage='';
        busy=true;draw();line.textContent='Applying cut…';
        try {
          if(await approveNeck?.(neck)) {
            if(seq!==request||!enabled.checked||!holder.isConnected||image!==currentSpec?.id)return;
            approved.add(neck.id); original=tissueForPieces?.();checkpoint();
            // Green/grey outlines belonged to the earlier tissue; keep only remaining suggestions.
            showPreview('pieces-kept',null); showPreview('pieces-dropped',null);
          } else errorMessage='Cut not applied. See the message above the image.';
        } catch(error) {errorMessage=error.message;}
        finally { busy=false;if(seq===request&&valid()){draw();if(errorMessage)line.textContent=errorMessage;} }
      };
      const draw = () => {
        if(!found) return;
        const pending=found.neck_regions.filter(n=>!rejected.has(n.id)&&!approved.has(n.id));
        reviewBox.hidden=false;reviewTitle.textContent='Necks · '+pending.length+' to review';
        showNeckReview?.(hidden?null:{regions:pending,approve,reject,busy});
        const scroll=list.scrollTop;
        list.replaceChildren(...pending.map(neck=>node('div',{class:'row neck-review-row'+(selected===neck.id?' selected':''),'data-neck-id':neck.id},[
          node('span',{class:'swatch',style:'background:#ff4848'}),
          node('button',{class:'neck-name',text:'Neck '+(Number(neck.id)+1),onclick:()=>go(neck),disabled:busy,
            title:'Areas on the two sides: '+neck.side_areas_um2.map(a=>Math.round(a).toLocaleString()+' µm²').join(' and ')}),
          node('span',{class:'muted neck-width',text:neck.width_um.toFixed(1)+' µm'}),
          node('button',{class:'btn small',text:'Go',onclick:()=>go(neck),disabled:busy}),
          node('button',{class:'btn small',text:'Approve',title:'Cut only this neck. Ctrl+Z undoes it.',onclick:()=>approve(neck),disabled:busy}),
          node('button',{class:'btn small',text:'Reject',title:'Dismiss this suggestion without changing tissue.',onclick:()=>reject(neck),disabled:busy})
        ])));
        list.scrollTop=scroll;
        restoreRejected.hidden=!rejected.size;restoreRejected.disabled=busy;restoreRejected.textContent='Show rejected ('+rejected.size+')';
        find.disabled=busy;hide.disabled=busy;hide.textContent=hidden?'Show':'Hide';
        line.textContent=pending.length+' necks to review'+(approved.size?' · '+approved.size+' applied':'');
        line.title=found.counts.kept+' pieces · '+found.counts.rejected+' filtered · '+found.ms+' ms. Ctrl+Z undoes a cut.';
      };
      const go=neck=>{
        current.syncNecks?.();if(busy||!valid())return;
        selected=neck.id;hidden=false;draw();
        const span=Math.max(100,neck.width_um/(found.microns_per_pixel||0.5083)*4);
        zoomAnnotationBox?.([neck.x-span/2,neck.y-span/2,span,span]);
      };
      restoreRejected.onclick=()=>{
        current.syncNecks?.();if(busy||!valid())return;
        const seq=request,previous=Array.from(rejected);rejected.clear();draw();
        recordNeck('restore_necks',previous).catch(error=>{if(seq===request&&valid()){previous.forEach(id=>rejected.add(id));draw();line.textContent=error.message;}});
      };
      const find = node('button',{class:'btn primary',text:'Preview',title:'Show suggested neck cuts and optional fragment filters without changing tissue.'});
      const hide=node('button',{class:'btn small',text:'Hide',title:'Hide or show the suggestions; keep the review list.',onclick:()=>{hidden=!hidden;showPreview('pieces-kept',null);showPreview('pieces-dropped',null);draw();}});
      find.onclick = async () => {
        if(busy || !enabled.checked) return;
        busy=true; find.disabled=true; const seq=++request; image=currentSpec?.id;
        if(found)draw();
        line.textContent='Looking for necks…';
        try {
          await flushShapes?.(); original=tissueForPieces?.();
          if(original?.busy) throw Error('Finish the current edit first.');
          if(original && !original.features) throw Error('Wait for Tissue to load, then preview necks again.');
          state.values.detect_necks=true;mark();
          const answer=await post(route('/pieces_preview'),{changes:Object.fromEntries(keys.map(k=>[k,state.values[k]])),
            ...(original?{image_token:original.image_token}:{}),
            ...(original?.preview?{features:original.features}:original?{tissue_token:original.token,revision:original.revision}:{})});
          if(seq!==request || !holder.isConnected || currentSpec?.id!==image || !enabled.checked) return;
          if(!original) { await ensureTissueLayer?.(answer.layer); original=tissueForPieces?.(); }
          if(seq!==request || !valid()) return;
          found=answer; rejected.clear(); approved.clear();checkpoints.length=0;checkpoint();hidden=false;selected=null;reviewBox.open=true;
          showPreview('pieces-kept',{outlines:found.kept,colour:found.colours.kept,kind:'artifact'});
          showPreview('pieces-dropped',found.rejected.length?{outlines:found.rejected,colour:found.colours.rejected,kind:'example'}:null);
          draw();
        } catch(error) { if(seq===request && holder.isConnected) {clearOverlays();list.replaceChildren();reviewBox.hidden=true;found=null;line.textContent=error.message;} }
        finally { busy=false;find.disabled=false;if(seq===request&&found)draw(); }
      };
      const number = (key,label,unit,factor,min,max,step,tip,shortLabel=label) => {
        const input=node('input',{type:'number',value:String(state.values[key]*factor),min,max,step,'aria-label':label});
        input.onchange=()=>{if(input.value!==''&&input.checkValidity()){state.values[key]=Number(input.value)/factor;mark();invalidate();}else input.value=String(state.values[key]*factor);};
        return field(shortLabel,node('div',{class:'inline grow'},[input,node('span',{class:'muted',text:unit})]),tip);
      };
      const toggle = (key,label,tip) => {
        const input=node('input',{type:'checkbox','aria-label':label});input.checked=!!state.values[key];
        input.onchange=()=>{state.values[key]=input.checked;mark();invalidate();};
        return field(label,input,tip);
      };
      const neckWidth=number('neck_min_width_um','Widest neck','µm',1,1,2000,1,'Connections wider than this remain ordinary tissue. Move the slider or type a width, then Preview.');
      const widthInput=neckWidth.querySelector('input');widthInput.style.width='68px';widthInput.style.flex='0 0 68px';
      const widthSlider=node('input',{type:'range',min:'1',max:String(Math.max(500,state.values.neck_min_width_um)),step:'1',value:state.values.neck_min_width_um,'aria-label':'Neck width slider',class:'grow'});
      widthInput.before(widthSlider);
      widthSlider.oninput=()=>{widthInput.value=widthSlider.value;};
      widthSlider.onchange=()=>widthInput.dispatchEvent(new Event('change',{bubbles:true}));
      widthInput.addEventListener('change',()=>{widthSlider.max=String(Math.max(500,state.values.neck_min_width_um));widthSlider.value=String(state.values.neck_min_width_um);});
      body.append(neckWidth,
        number('neck_min_lobe_area_mm2','Minimum on each side','µm²',1e6,0,1e7,1000,'Both sides must be at least this large. Raise this to ignore small bumps.','Each side ≥'),
        number('neck_min_lobe_fraction','Minimum share on each side','%',100,0,50,1,'Both sides must also be this share of the original piece. 2 means 2%.','Share ≥'),
        node('div',{class:'row'},[find,hide]),line,reviewBox);
      const filters=node('details',{class:'piece-filters'});
      filters.append(node('summary',{text:'After cutting '},[help('Optional filters for fragments left after cutting. Smallest in Tissue settings applies to initial detection. These filters apply only when you use Apply all or Use preview.')]),
        toggle('cut_at_necks','Cut all detected necks','For Apply all below. Individual approval always cuts only the neck you approve.'),
        toggle('filter_by_area','Remove small fragments','Off keeps every fragment.'),
        number('min_area_mm2','Smallest fragment','µm²',1e6,0,1e8,1000,'Remove fragments below this area after cuts. Earlier tissue remains in History.'),
        toggle('filter_by_inscribed_circle','Remove thin fragments','Optional: a circle of this size must fit inside each kept piece.'),
        number('min_inscribed_diameter_um','Fitting circle','µm',1,1,5000,10,'Keeps pieces broad enough for the analysis.'));
      body.append(filters);holder.append(box);
      const save=headerSave(box,dirty,async()=>{
        state.saved=await post(route('/pieces_settings'),{changes:changes(),revision:state.saved.revision});mark();
      },status,current,holder,()=>Object.keys(changes()).length>0,mark);
      const use=node('button',{class:'btn',text:'Use preview',title:'Apply these cuts and filters to the temporary tissue preview. Save polygons keeps the result.',onclick:()=>{
        if(!found || !valid() || approved.size || rejected.size) {line.textContent='Find necks again before applying all pieces. Individual reviews are kept separate.';return;}
        if(!original.preview) {line.textContent='Use Apply all below for saved tissue.';return;}
        if(applyPiecePreview?.(found.features,original.features)){invalidate();line.textContent='Preview updated. Use Save polygons to keep it. Ctrl+Z undoes.';}
      }});
      filters.append(use,stageRow(current,'Apply all','apply_pieces',{argumentsFor:()=>({}),save,isDirty:()=>Object.keys(changes()).length>0,status,
        tip:'Applies the chosen cuts and optional filters to saved tissue. Earlier tissue stays in History.',
        ask:()=>{
          if(tissueForPieces?.()?.preview){status.textContent='Save polygons first, or use these pieces in preview.';return false;}
          if(approved.size||rejected.size){status.textContent='Apply all does not include individual review decisions. Find necks again first.';return false;}
          return window.confirm('Apply all chosen cuts and filters to tissue? The earlier version stays in History.');
        }}));
      body.append(status);mark();line.textContent='';
    };

    // Zones (Tab 4 step 4): pools join domains; rings of a set width go inside and outside chosen classes.
    const zonesPanel = async (current, holder) => {
      const state = current.zones || (current.zones = {});
      try { state.saved = await get(route('/zone_rules')); }
      catch (error) { holder.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.rules) state.rules = JSON.parse(JSON.stringify(state.saved.values));
      const mpp = state.saved.microns_per_pixel || 0.5083;
      const isDirty = () => JSON.stringify(state.rules) !== JSON.stringify(state.saved.values);
      const pools = section('Pools', {tip:'Joins domains into one larger compartment, for example Epithelia and Stroma into tissue. ' +
        'A pool can have its own zones below.'});
      const rings = section('Zones', {tip:'Rings of the chosen width inside and outside a class: outer_1, outer_2… grow outwards, ' +
        'inner_1… go inwards. Rings of the same level are joined across classes.'});
      const dirty = chip('Saved', 'ok');
      const mark = () => { const d = isDirty(); dirty.textContent = d ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (d ? 'warn' : 'ok'); };
      const line = node('p', {class:'muted', role:'status'});
      const legend = node('div', {class:'studio-domains'});
      holder.append(pools.box, rings.box);
      let classes = null, drawn = [];
      const targets = () => Array.from(new Set([...(classes || []), ...state.rules.pools.map(rule => rule.name).filter(Boolean)]));
      let timer = null;
      const changed = () => { mark(); clearTimeout(timer); timer = setTimeout(refresh, 250); };
      const drawPools = () => {
        pools.body.replaceChildren(...state.rules.pools.map((rule, index) => {
          const name = node('input', {type:'text', value:rule.name, maxlength:'60', placeholder:'Pool name', 'aria-label':'Pool ' + (index + 1) + ' name'});
          name.addEventListener('change', () => { rule.name = name.value.trim(); drawRings(); changed(); });
          const ticks = (classes || []).map(item => {
            const tick = node('input', {type:'checkbox', 'aria-label':'Pool ' + (index + 1) + ' has ' + item}); tick.checked = rule.sources.includes(item);
            tick.addEventListener('change', () => { rule.sources = tick.checked ? [...rule.sources, item] : rule.sources.filter(v => v !== item); changed(); });
            return node('label', {class:'studio-pool-class'}, [tick, node('span', {text:item})]);
          });
          const drop = node('button', {class:'btn small ghost', text:'Remove', onclick:() => { state.rules.pools.splice(index, 1); drawPools(); drawRings(); changed(); }});
          return node('div', {class:'studio-pool'}, [node('div', {class:'row'}, [name, drop]), node('div', {class:'studio-pool-classes'}, ticks)]);
        }), node('div', {class:'row'}, [node('button', {class:'btn small', text:'Add a pool', onclick:() => {
          state.rules.pools.push({name:'', sources:[], subtract:[]}); drawPools();
        }}), node('span', {class:'grow'}), (() => {
          const keep = node('input', {type:'checkbox', 'aria-label':'Keep the domains'}); keep.checked = state.rules.keep_sources;
          keep.addEventListener('change', () => { state.rules.keep_sources = keep.checked; changed(); });
          return node('label', {class:'studio-remove', title:'Also write every domain into the zones file'}, [keep, node('span', {text:'Keep domains'})]);
        })()]));
      };
      const drawRings = () => {
        const byLabel = Object.fromEntries(state.rules.zones.map(rule => [rule.label, rule]));
        const rows = targets().map(label => {
          const rule = byLabel[label] || {label, width_px:100, inner:0, outer:0};
          const set = () => {
            const kept = state.rules.zones.filter(item => item.label !== label);
            state.rules.zones = rule.inner || rule.outer || byLabel[label] ? [...kept, rule] : kept;
            changed();
          };
          const width = node('input', {type:'number', min:'1', step:'5', value:String(Math.round(rule.width_px * mpp * 10) / 10), 'aria-label':'Ring width of ' + label});
          width.addEventListener('change', () => { const um = Number(width.value); if (um > 0) { rule.width_px = Math.round(um / mpp * 10) / 10; set(); } });
          const count = (key, text) => {
            const input = node('input', {type:'number', min:'0', max:'20', step:'1', value:String(rule[key]), 'aria-label':text + ' rings of ' + label});
            input.addEventListener('change', () => { const n = Math.round(Number(input.value)); if (n >= 0 && n <= 20) { rule[key] = n; set(); } else input.value = String(rule[key]); });
            return input;
          };
          return node('div', {class:'studio-zone-row'}, [node('span', {class:'grow', text:label, title:label}), width, count('inner', 'Inside'), count('outer', 'Outside')]);
        });
        const clip = choice([['', 'Nothing'], ...targets().map(name => [name, name])], state.rules.clip || '', 'Clip outer rings to');
        clip.addEventListener('change', () => { state.rules.clip = clip.value || null; changed(); });
        rings.body.replaceChildren(node('div', {class:'studio-zone-row head'}, [node('span', {class:'grow', text:'Class'}), node('span', {text:'Width µm'}),
          node('span', {text:'Inside'}), node('span', {text:'Outside'})]), ...rows,
          field('Clip to', clip, 'Outer rings are cut to this class so they never leave the tissue; pick the pooled tissue when you have one.'), line, legend);
      };
      const refresh = latest(async () => {
        try {
          const image = currentSpec?.id;
          const found = await post(route('/zones_preview'), {rules:state.rules});
          if (!holder.isConnected || currentSpec?.id !== image) return;
          if (!classes) { classes = found.classes; drawPools(); drawRings(); }
          const draw = () => {
            drawn.forEach(name => showPreview(name, null));
            drawn = found.layers.filter(layer => !state.only || layer.name === state.only).map((layer, index) => {
              const name = 'zones-' + index;
              showPreview(name, {outlines:layer.rings, colour:layer.colour, kind:layer.kind === 'pool' ? 'example' : 'artifact'}); return name;
            });
            legend.querySelectorAll('.studio-class').forEach(row => row.classList.toggle('dimmed', !!state.only && row.dataset.layer !== state.only));
          };
          if (state.only && !found.layers.some(layer => layer.name === state.only)) state.only = null;
          legend.replaceChildren(...found.layers.map(layer => node('div', {class:'studio-class studio-pickable', role:'button', tabindex:'0', 'data-layer':layer.name,
            title:'Show only ' + layer.name + ' on the image (click again for all)', onclick:() => { state.only = state.only === layer.name ? null : layer.name; draw(); }},
            [node('span', {class:'swatch', style:'background:' + layer.colour}),
            node('span', {class:'grow', text:layer.name}), node('span', {class:'muted', text:layer.kind === 'pool' ? 'pool' : 'ring'})])));
          draw();
          line.textContent = found.layers.length ? found.layers.length + ' layer' + (found.layers.length === 1 ? '' : 's') + ' · ' + found.ms + ' ms · from ' + found.source :
            'No pools or rings yet: tick classes into a pool, or give a class inside or outside rings.';
        } catch (error) { line.textContent = error.message; if (!classes) pools.body.replaceChildren(node('p', {class:'hint', text:error.message})); }
      });
      const status = node('p', {class:'muted', role:'status'});
      let save = async () => {
        state.saved = await post(route('/zone_rules'), {rules:state.rules, revision:state.saved.revision});
        mark();
      };
      save = headerSave(rings.box, dirty, save, status, current, holder, isDirty, mark);
      current.runBar([stageRow(current, 'Make zones', 'make_zones', {save, isDirty, status,
        tip:'Writes the zones file (Tab 4 step 4\u2019s folder), which cell annotation reads, for the open image or each ticked one.'}), status]);
      mark(); line.textContent = 'Reading the domains…'; pools.body.replaceChildren(node('p', {class:'muted', text:'Reading the domains…'})); refresh();
    };

    // ------------------------------------------------------------ Segmentation
    // The model is chosen from the step's list (by id, saved at once), not typed here.
    const SEG_KEYS = ['tab4_diameter', 'tab4_flow', 'tab4_gpu', 'tab4_cellpose_batch_size', 'tab4_cellpose_batch_auto', 'tab4_gpu_device', 'tab4_last_nuclear_names', 'tab4_last_membrane_names'];
    TOOLS.tab5 = current => {
      // Tab 5 in the Studio design: folders (images, results, models), the images with where each is segmented and whether
      // it is done, the step's area, the markers' roles and the model, a test on the view, then Segment for the open image,
      // the ticked ones, or only inside the polygons picked on the image.
      const place = section('Folders and images', {tip:'Where the images to segment are, where the results go (<image>_cell_features.csv, ' +
        'and the cell outlines in geojson/), and the folder holding your segmentation models (a lab folder works too: its models appear in ' +
        'the Model list). Click an image to open it; its cells are shown once it is segmented.'});
      const {box, body} = section('Settings', {tip:'Cellpose settings save automatically. ' +
        'Test them on the view first.'});
      const form = node('div', {id:'studio-seg', class:'studio-form'});
      const dirty = chip('Saved', 'ok');
      body.append(form); box.querySelector('.studio-sec-h').insertBefore(dirty, box.querySelector('.studio-sec-h .grow'));
      const test = section('Test on the image', {tip:'Segments the visible area (its centre, up to 1,024 pixels square) with the ' +
        'settings above, without saving anything. The first test loads the model and takes longer.'});
      const result = node('span', {class:'muted'});
      const run = node('button', {class:'btn primary', text:'Test on this view'});
      const clear = node('button', {class:'btn small', text:'Clear', hidden:true, onclick:() => { showSegPreview(null); clear.hidden = true; result.textContent = ''; }});
      test.body.append(node('div', {class:'row'}, [run, clear]), result);
      current.body.append(place.box, box, test.box);
      const note = node('p', {class:'muted', role:'status'});
      const picked = new Set();
      let plan = null, openImage = null, saved = null, values = {}, tagging = 0;
      // ---- folders and images
      const loadPlan = async () => {
        try { plan = await get(route('/segmentation_images')); if (place.box.isConnected) { tagging = plan.tagging_rules; syncTag(); drawImages(); draw(); } }
        catch (error) { fill(place.body, node('p', {class:'hint', text:error.message})); }
        get(route('/step/tab5')).then(value => showOutputs(current, value)).catch(() => {});  // the saved files, in Outputs
      };
      const saveStep = async (changes, done) => {
        const folder = changes.folders ? Object.keys(changes.folders)[0] : null;
        note.textContent = folder ? 'Reading the new ' + folder + ' folder…' : 'Saving…';
        place.box.classList.add('studio-working');
        try {
          plan = await post(route('/segmentation_images'), {changes}); drawImages(); draw();
          if (changes.model || changes.folders?.models) current.refreshTrainingModels?.();
          note.textContent = done || (folder ? 'Folder changed. ' + plan.rows.length + ' images.' : 'Saved with the project.');
          return true;
        } catch (error) { note.textContent = error.message; return false; }
        finally { place.box.classList.remove('studio-working'); }
      };
      const folderBox = (kind, label, tip) => {
        const input = node('input', {type:'text', value:plan.folders[kind], maxlength:'1024', class:'grow', 'aria-label':label + ' folder'});
        input.addEventListener('change', () => { if (input.value.trim() && input.value.trim() !== plan.folders[kind]) saveStep({folders:{[kind]:input.value.trim()}}); });
        return field(label, input, tip);
      };
      const openRow = async row => {
        openImage = row.image;
        place.body.querySelectorAll('tr.selected').forEach(item => item.classList.remove('selected'));
        place.body.querySelector('tr[data-image="' + CSS.escape(row.image) + '"]')?.classList.add('selected');
        note.textContent = 'Opening ' + row.image + '…';
        try { const answer = await runCommand('open_segmentation_image', {image:row.image}); showResult(answer); note.textContent = answer.message; }
        catch (error) { note.textContent = error.message; }
      };
      // A select with optgroups: [[value, text]...] first, then [[group, [[value, text]...]]...].
      // The cell outlines' folder: <output>/geojson unless changed (folded, as most keep it).
      const outlinesFolder = () => {
        const input = node('input', {type:'text', value:plan.geojson_custom ? plan.folders.geojson : '', placeholder:plan.folders.geojson, maxlength:'1024', class:'grow', 'aria-label':'Outlines folder'});
        input.addEventListener('change', () => saveStep({folders:{geojson:input.value.trim()}}, input.value.trim() ? 'Outlines go to ' + input.value.trim() + '.' : 'Outlines go to geojson/ in the output folder.'));
        return node('details', {class:'studio-options'}, [node('summary', {text:'Outlines folder: ' + (plan.geojson_custom ? plan.folders.geojson : 'geojson/ in Output')}),
          field('Outlines', input, 'Where each image’s cell outlines (<image>_cell_boundaries.geojson) are written. Empty: geojson/ in the Output folder.')]);
      };
      const drawImages = () => {
        const head = pickAll(plan.rows.map(row => row.image), picked, drawImages);
        const rows = plan.rows.map(row => {
          const tick = node('input', {type:'checkbox', 'aria-label':'Pick ' + row.image}); tick.checked = picked.has(row.image);
          tick.addEventListener('change', () => { tick.checked ? picked.add(row.image) : picked.delete(row.image); head.sync(); });
          // One choice per image: the whole image or one of its annotations. A row left as it is follows Where to segment.
          const step = plan.layer || '__whole__';
          const now = row.area || step;
          // Grouped as Annotation's Export: Local names, then Imported ones (whichever file was imported for the image).
          const area = groupedChoice([['__whole__', 'Whole image']], row.saved_groups || [['This image', row.saved.map(name => [name, name])]],
            now, 'Where to segment ' + row.image);
          if (!Array.from(area.options).some(option => option.value === now)) { area.append(node('option', {value:now, text:now + ' (not saved for this image)'})); area.value = now; }
          area.title = 'Where this image is segmented: the whole image, or inside one of its annotations (all its polygons together)';
          area.addEventListener('change', () => saveStep({areas:{[row.image]:area.value === step ? '' : area.value}}, row.image + ': segmented ' +
            (area.value === '__whole__' ? 'in the whole image.' : area.value ? 'inside its ' + area.value + '.' : 'as Where to segment says.')));
          const state = node('span', {}, [workingChip(current, row.image),
            row.segmented ? chip(row.cells !== null ? row.cells.toLocaleString() + ' cells' : 'Segmented', 'ok', 'Segmented' + (row.outlines ? ', with cell outlines' : '')) : chip('Not run'),
            row.missing ? chip('No ' + row.effective, 'warn', 'This image has no saved ' + row.effective + ' annotation: save it in the Annotation step, or choose another area') : null]);
          const name = node('button', {type:'button', class:'studio-item-name studio-file', text:row.image, title:'Open ' + row.image + (row.segmented ? ' with its cells' : ''), onclick:() => openRow(row)});
          const line = node('tr', {'data-image':row.image, class:row.image === openImage ? 'selected' : ''});
          line.addEventListener('click', event => { if (!event.target.closest('button, select, input')) openRow(row); });
          fill(line, node('td', {}, [tick]), node('td', {}, [name]), node('td', {}, [area]), node('td', {}, [state]));
          return line;
        });
        // The area for several images at once: the ticked ones, or all when none is ticked.
        const everyGroup = new Map();
        plan.rows.forEach(row => (row.saved_groups || []).forEach(([group, items]) => {
          if (group === 'Earlier choice') return;  // one image's own file: not a choice for all
          const label = group;
          const known = everyGroup.get(label) || new Map(); items.forEach(([value, text]) => known.set(value, text)); everyGroup.set(label, known); }));
        const setAll = groupedChoice([['', 'Area (set all)'], ['__whole__', 'Whole image']],
          Array.from(everyGroup, ([group, items]) => [group, Array.from(items).sort((a, b) => a[1].localeCompare(b[1]))]), '', 'Area of the ticked images');
        setAll.title = 'Set where the ticked images are segmented (all images when none is ticked)';
        setAll.addEventListener('change', () => {
          if (!setAll.value) return;
          const targets = picked.size ? Array.from(picked) : plan.rows.map(row => row.image);
          saveStep({areas:Object.fromEntries(targets.map(image => [image, setAll.value]))},
            targets.length + ' image' + (targets.length === 1 ? '' : 's') + ': ' + (setAll.value === '__whole__' ? 'the whole image.' : 'inside ' + setAll.value + '.'));
        });
        const done = plan.rows.filter(row => row.segmented).length, missing = plan.rows.filter(row => row.missing).length;
        fill(place.body,
          folderBox('images', 'Images', 'The images to segment (ROIs, usually). Type the folder’s path and press Enter.'),
          folderBox('output', 'Output', 'Where each image’s cell table (<image>_cell_features.csv) is written; the outlines go to the Outlines folder below (geojson/ in it, unless changed).'),
          outlinesFolder(),
          // Annotations to segment in, from GeoJSON files: imported here for every image at once, then chosen by name below.
          node('details', {class:'studio-options'}, [node('summary', {text:'Annotations from GeoJSON files'}), importsBox(() => loadPlan()),
            node('p', {class:'muted', text:'Then choose “Area (set all)”: an Imported name (tissue, say) is each image’s own imported tissue.'})]),
          folderBox('models', 'Models', 'A folder with Cellpose models (yours, a colleague’s, or the model library): they are listed in Model below, ' +
            'with Cellpose’s own and the ones trained in this project. Nothing is copied.'),
          node('div', {class:'row'}, [chip(plan.rows.length + ' images'), chip(done + ' segmented', done ? 'ok' : ''),
            missing ? chip(missing + ' without their area', 'warn') : null]),
          plan.rows.length ? node('div', {class:'studio-pairs-scroll'}, [node('table', {class:'studio-counts studio-pairs studio-seg-list'},
            [node('tr', {}, [node('th', {}, [head]), node('th', {text:'Image'}), node('th', {}, [setAll]), node('th', {text:'State'})]), ...rows])])
            : node('p', {class:'muted', text:'No image in this folder: type the images folder above.'}),
          note);
      };
      // ---- settings: markers' roles, the model, size, strictness and the GPU; saved as they change
      let saving = null;
      const changes = () => saved ? Object.fromEntries(SEG_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]])) : {};
      const mark = text => { const n = Object.keys(changes()).length; dirty.textContent = text || (n ? 'Not saved' : 'Saved'); dirty.className = 'chip ' + (n && !text ? 'warn' : 'ok'); };
      const save = async () => {
        const pending = changes();
        if (!Object.keys(pending).length) return true;
        mark('Saving…');
        const next = await post(route('/segmentation_settings'), {changes:pending, revision:saved.revision});
        saved = next; values = {...JSON.parse(JSON.stringify(next.values)), ...Object.fromEntries(Object.entries(changes()).filter(([key]) => !(key in pending)))};
        mark(); return true;
      };
      const set = (key, value) => {
        values[key] = value; mark();
        clearTimeout(saving); saving = setTimeout(() => save().catch(error => { mark(); note.textContent = error.message; }), 500);
      };
      const reloadSettings = async () => { saved = await get(route('/segmentation_settings')); values = {...JSON.parse(JSON.stringify(saved.values)), ...changes()}; draw(); };
      const draw = () => {
        if (!saved || !plan) return;
        const markers = imageMarkers();
        const nuclear = values.tab4_last_nuclear_names.find(name => markers.includes(name)) || markers.find(name => /dapi/i.test(name)) || markers[0] || '';
        const borders = new Set(values.tab4_last_membrane_names);
        const roles = markers.length ? node('div', {class:'studio-pairs-scroll studio-seg-markers'}, [node('table', {class:'studio-counts studio-pairs'}, [
          node('tr', {}, ['Marker', 'Nucleus', 'Border'].map(text => node('th', {text}))),
          ...markers.map(name => {
            const isNucleus = node('input', {type:'radio', name:'studio-seg-nucleus', 'aria-label':'Nucleus marker ' + name}); isNucleus.checked = name === nuclear;
            isNucleus.addEventListener('change', () => { set('tab4_last_nuclear_names', [name]); draw(); });
            const isBorder = node('input', {type:'checkbox', 'aria-label':'Cell border marker ' + name}); isBorder.checked = borders.has(name);
            isBorder.addEventListener('change', () => { isBorder.checked ? borders.add(name) : borders.delete(name); set('tab4_last_membrane_names', Array.from(borders)); });
            return node('tr', {}, [node('td', {class:'studio-file', text:name, title:name}), node('td', {}, [isNucleus]), node('td', {}, [isBorder])]);
          })])]) : node('p', {class:'muted', text:'Open an image to choose its markers’ roles.'});
        // One model list, as the artifact models: Cellpose's own, this project's trained versions, the Models folder's.
        const model = node('select', {'aria-label':'Cellpose model'});
        if (!plan.model) model.append(node('option', {value:'', text:'Choose a model'}));
        ['Cellpose', 'Trained in this project', 'Models folder', 'Other'].forEach(group => {
          const items = plan.models.filter(item => item.group === group);
          if (items.length) model.append(node('optgroup', {label:group}, items.map(item => node('option', {value:item.id, text:item.name, title:item.title || ''}))));
        });
        model.value = plan.model;
        model.addEventListener('change', async () => {
          const chosen = plan.models.find(item => item.id === model.value);
          if (await saveStep({model:model.value}, (chosen?.name || model.value) + ' is the model now (saved).')) await reloadSettings();
        });
        const number = (key, label, step) => {
          const input = node('input', {type:'number', step, min:'0', value:String(values[key]), 'aria-label':label});
          input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= 0) set(key, value); else input.value = String(values[key]); });
          return input;
        };
        const gpu = node('input', {type:'checkbox', 'aria-label':'Use the GPU'}); gpu.checked = !!values.tab4_gpu;
        gpu.addEventListener('change', () => set('tab4_gpu', gpu.checked));
        const batchMode = node('select', {'aria-label':'GPU batch mode'}, ['Auto', 'Manual'].map(text => node('option', {value:text, text})));
        batchMode.value = values.tab4_cellpose_batch_auto ? 'Auto' : 'Manual';
        const batch = node('input', {type:'number', min:'1', max:'256', step:'1', value:String(values.tab4_cellpose_batch_size), 'aria-label':'GPU batch'});
        batch.hidden = !!values.tab4_cellpose_batch_auto;
        batch.addEventListener('change', () => { const n = Number(batch.value); if (Number.isInteger(n) && n >= 1 && n <= 256) set('tab4_cellpose_batch_size', n); else batch.value = String(values.tab4_cellpose_batch_size); });
        batchMode.addEventListener('change', () => { const auto = batchMode.value === 'Auto'; set('tab4_cellpose_batch_auto', auto); batch.hidden = auto; });
        const device = node('input', {type:'number', min:'0', max:'255', step:'1', value:String(values.tab4_gpu_device || 0), 'aria-label':'GPU index'});
        device.addEventListener('change', () => { const n = Number(device.value); if (Number.isInteger(n) && n >= 0 && n <= 255) set('tab4_gpu_device', n); else device.value = String(values.tab4_gpu_device); });
        const compute = node('details', {}, [node('summary', {text:'GPU settings'}),
          field('Batch', node('div', {class:'row grow'}, [batchMode, batch]), 'Auto tests block batches on the selected GPU and reuses a fast size. Manual sets a limit. Memory errors reduce the batch automatically.'),
          field('GPU index', device, 'The visible GPU to use, starting at 0. One GPU is used per job.')]);
        fill(form,
          node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Markers'}), node('span', {class:'grow'}),
            help('Nucleus: the marker Cellpose finds nuclei in (DAPI, usually). Border: markers combined to find where each cell ends; with none, ' +
              'the nucleus marker is used. N and M on a marker in Layers (right) set the same.')]),
          roles,
          field('Model', model, 'Choose a built-in model, a model trained in this project, or one from the Models folder.'),
          field('Diameter', number('tab4_diameter', 'Cell diameter', '1'), 'Expected cell diameter in pixels; 0 lets Cellpose estimate it.'),
          field('Flow', number('tab4_flow', 'Flow threshold', '0.1'), 'Reject cells whose predicted flows disagree with their shape. Lower values reject more cells.'),
          node('div', {class:'row field'}, [node('label', {class:'lbl', text:'GPU'}), gpu, node('span', {class:'grow'}),
            help('Use an available compatible GPU to speed up segmentation. Otherwise, use the CPU.')]), compute);
        mark();
        if (setMarkerRoles) setMarkerRoles({nuclear, membrane:Array.from(borders), onChange:(role, name) => {
          if (role === 'nuclear') set('tab4_last_nuclear_names', [name]);
          else set('tab4_last_membrane_names', borders.has(name) ? Array.from(borders).filter(item => item !== name) : [...borders, name]);
          draw();
        }});
      };
      const load = async () => {
        try {
          saved = await get(route('/segmentation_settings'));
          if (!form.isConnected) return;
          values = JSON.parse(JSON.stringify(saved.values)); draw();
        } catch (error) { form.replaceChildren(node('p', {class:'hint', text:error.message})); }
      };
      current.refreshMarkers = () => draw();
      run.addEventListener('click', async () => {
        const area = viewBounds();
        if (!area) { result.textContent = 'Open an image first.'; return; }
        const image = currentSpec?.id;
        run.disabled = true; result.textContent = 'Segmenting the visible area…';
        try {
          const found = await post(route('/segment_preview'), {bounds:area, settings:changes()});
          if (!form.isConnected || currentSpec?.id !== image) return;
          showSegPreview(found); clear.hidden = false;
          result.textContent = found.count.toLocaleString() + ' cells · ' + found.seconds + ' s · ' + found.device +
            (found.model_loaded ? ' (model loaded)' : '');
        } catch (error) { result.textContent = error.message; }
        finally { run.disabled = false; }
      });
      // ---- bottom: Segment [Segment again] [Tag cells] This image | Selected | Selected polygons
      const status = node('p', {class:'muted', role:'status'});
      const again = node('input', {type:'checkbox', 'aria-label':'Segment again'});
      const tag = node('input', {type:'checkbox', 'aria-label':'Tag cells after'}); tag.checked = true;
      const tagLabel = node('label', {class:'inline-check'}, [tag, node('span', {text:'Then tag cells'})]);
      const syncTag = () => { tag.disabled = !tagging; if (!tagging) tag.checked = false;
        tagLabel.title = tagging ? 'After segmenting, the new cell tables are tagged with the Cell tagging step’s settings (' + tagging +
          ' choice' + (tagging === 1 ? '' : 's') + ': its columns, names, units, tissue pieces and beds), in the same job. Change them in Cell tagging.' :
        'Nothing to tag with yet: choose the columns in the Cell tagging step first.'; };
      const segment = (button, scope, polygons) => async () => {
        button.disabled = true;
        try {
          const area = polygons ? hooks.pickedShapes?.() : null;
          if (polygons && !area) throw Error('Select polygons on the image first: click one, Ctrl-click to add more.');
          const names = Array.from(picked).filter(name => plan.rows.some(row => row.image === name));
          if (scope === 'images' && !names.length) throw Error('Tick the images first.');
          const every = scope === 'images' && names.length === plan.rows.length;
          if (scope === 'images' && !every && names.length > 128) throw Error('Tick at most 128 images, or all of them.');
          clearTimeout(saving); await save();
          status.textContent = '';
          const answer = await runCommand('segment_images', {scope:every ? 'all' : scope, ...(scope === 'images' && !every ? {images:names} : {}),
            again:again.checked, tag:tag.checked && !tag.disabled, ...(area ? {shape:area.shape, ids:area.ids} : {})}, {confirmed:true});
          showResult(answer); status.textContent = answer.message;
          if (answer.data?.job) { await refreshJobs(); showTab('jobs'); await loadPlan(); }
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
      };
      const one = node('button', {class:'btn', text:'This image'});
      const many = node('button', {class:'btn primary', text:'Selected'});
      const inside = node('button', {class:'btn', text:'Selected polygons', title:'Segment the open image only inside the polygons picked ' +
        'on it (click one, Ctrl-click to add more), from any drawn layer: tissue pieces, regions, classes'});
      one.addEventListener('click', segment(one, 'image')); many.addEventListener('click', segment(many, 'images'));
      inside.addEventListener('click', segment(inside, 'image', true));
      current.runBar([node('div', {class:'row wrap'}, [node('span', {class:'lbl', text:'Segment:'}),
          node('label', {class:'inline-check', title:'Segment images that are done already again; their earlier results move to previous/ first'}, [again, node('span', {text:'Again'})]),
          tagLabel, one, many, inside,
          help('Segments with the settings above, as a job (on the GPU when there is one). This image: the open one. Selected: the ticked ' +
            'images (tick the header box for all). Selected polygons: the open image, only inside the polygons picked on it. Each image is ' +
            'segmented in its row’s Area. Images already segmented are kept unless Again is ticked (the earlier results move to previous/). Then tag cells runs the Cell tagging step on the new tables, with its settings.')]),
        status]);
      // The list follows running segmentations: rows turn green as images finish; the open image shows its new cells.
      followJobs(current, ['segment_images', 'import_annotations'], async () => {
        // The image on screen, however it was opened (here, in Annotation, from the list on the left).
        const shown = () => openImage || currentSpec?.layers?.find(l => l.type === 'image')?.name;
        const before = plan?.rows.find(row => row.image === shown())?.stamp;
        await loadPlan();
        const now = plan?.rows.find(row => row.image === shown());
        if (now && now.segmented && now.stamp !== before) openRow(now);  // new outlines (first time or again): show them
      });
      cellposeTraining(current, {changes,
        useModel:async path => {
          const id = 'project:' + path.split('/').pop();
          if (!plan.models.some(item => item.id === id)) await loadPlan();
          if (await saveStep({model:id}, path.split('/').pop() + ' is the model now (saved).')) await reloadSettings();
        }});
      loadPlan(); load();
    };

    // Cellpose training, as the user designed it: tiles of a chosen size are placed on the tissue, for training or for
    // validation; the model in training predicts a tile, its cells are corrected on the image and the tile is kept;
    // Train makes the next version from the current one on every training tile, and that version predicts the next
    // tile; Validate compares the predictions with the corrections. Use makes a version Segmentation's model.
    const TRAINING_LAYER = 'Cells to correct';
    const cellposeTraining = (current, {changes, useModel}) => {
      const state = current.training || (current.training = {size:'512', role:'train'});
      const {box, body} = section('Train Cellpose', {tip:'Make Cellpose better on your cells, round by round: place a tile, ' +
        'correct the cells the model predicts (with the tools above the image: Freehand draws a missing cell, right-click deletes a wrong one, ' +
        'Merge joins two pieces of one cell, Split cuts two cells drawn as one, Ctrl+Z undoes), keep the tile, and Train. The new version predicts the next tile. Validation tiles, once corrected, ' +
        'show how far the model is from your cells (Validate); they can then join training.'});
      const line = node('p', {class:'muted', role:'status'});
      const sessionBox = node('div'), tilesBox = node('div', {class:'studio-examples'}), checked = node('div'), versions = node('div', {class:'studio-examples'});
      const scoresBox = node('div', {class:'studio-scores'});
      const scoreButton = node('button', {class:'btn small', text:'Score all versions', title:'Run the base model and every version on the same validation tiles'});
      const size = choice([['256', '256 px'], ['512', '512 px'], ['768', '768 px'], ['1024', '1,024 px']], state.size, 'Tile size');
      size.addEventListener('change', () => { state.size = size.value; });
      const role = choice([['train', 'Training'], ['validate', 'Validation']], state.role, 'Tile role');
      role.addEventListener('change', () => { state.role = role.value; });
      const place = node('button', {class:'btn', text:'Place a tile'});
      const keep = node('button', {class:'btn primary', text:'Keep tile', disabled:true});
      const discard = node('button', {class:'btn small ghost', text:'Discard', hidden:true});
      const epochs = node('input', {type:'number', min:'1', max:'2000', step:'10', value:'100', 'aria-label':'Epochs'});
      const trainButton = node('button', {class:'btn primary', text:'Train'});
      const validateButton = node('button', {class:'btn', text:'Validate'});
      body.append(sessionBox,
        node('div', {class:'row wrap'}, [size, role, place]), node('div', {class:'row wrap'}, [keep, discard]), line, tilesBox,
        field('Epochs', epochs, 'Training passes over the tiles each round (Cellpose’s trainer uses 100).'),
        node('div', {class:'row'}, [trainButton, validateButton]), checked, versions, scoresBox);
      current.body.append(box);
      let found = null, loadRequest = 0;
      const model = () => found?.session?.current || '';
      const modelLabel = path => !path ? '' : path.includes('/') ? path.split('/').pop() : path;
      // The model in training as Segmentation's settings, for predicting a tile.
      const modelSettings = () => { const m = model(); return m.includes('/') ? {tab4_model_type:'custom', tab4_custom_model:m} : m ? {tab4_model_type:m} : {}; };
      // The kept tiles on the open image, and (when asked) the corrected cells inside them; both can be hidden.
      if (state.showTiles === undefined) state.showTiles = true;
      let cellsRequest = 0;
      const drawTiles = () => {
        const mine = state.showTiles ? (found?.examples || []).filter(item => item.image === found.open_image) : [];
        const box_ = item => { const [x, y, w, h] = item.bounds; return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]; };
        const of = kind => mine.filter(item => (item.role || 'train') === kind).map(box_);
        showPreview('training-tiles-train', of('train').length ? {outlines:of('train'), colour:'#2ecc71', kind:'example'} : null);
        showPreview('training-tiles-validate', of('validate').length ? {outlines:of('validate'), colour:'#f0c35a', kind:'example'} : null);
        drawTileCells();
      };
      const drawTileCells = async () => {
        const request = ++cellsRequest;
        if (!state.showTileCells || !found?.open_image) { showPreview('training-tile-cells', null); return; }
        try {
          const value = await get(route('/training?outlines=1'));
          if (request !== cellsRequest || !box.isConnected) return;
          showPreview('training-tile-cells', value.outlines?.length ? {outlines:value.outlines, colour:'#00e5ff'} : null);
        } catch (error) { line.textContent = error.message; }
      };
      const drawSession = () => {
        const session = found.session;
        if (session && !state.restart) {
          const last = session.versions[session.versions.length - 1];
          sessionBox.replaceChildren(node('div', {class:'row'}, [chip(last ? last.name : 'not trained yet', last ? 'ok' : ''),
            node('span', {class:'muted grow', text:last ? 'from ' + modelLabel(session.base) + ' · ' + session.versions.length + ' round' +
              (session.versions.length === 1 ? '' : 's') : 'starts from ' + modelLabel(session.base)}),
            node('button', {class:'btn small ghost', text:'Start over', onclick:() => { state.restart = true; load(); }})]));
          return;
        }
        const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
        const name = node('input', {type:'text', value:state.draftName || 'studio_' + stamp, maxlength:'50', 'aria-label':'Model name'});
        name.oninput = () => { state.draftName = name.value; };
        const base = node('select', {'aria-label':'Start from'});
        const options = found.starting_models || [];
        for (const group of new Set(options.map(item => item.group))) {
          const items = options.filter(item => item.group === group);
          base.append(node('optgroup', {label:group}, items.map((item, index) => node('option', {
            value:item.base, disabled:!item.available,
            text:item.name + (index === 0 && item.path && item.available ? ' · newest' : ''),
            title:[item.title, item.path].filter(Boolean).join(' · ')}))));
        }
        base.value = options.some(item => item.base === state.draftBase && item.available) ? state.draftBase : found.selected_base;
        base.onchange = () => { state.draftBase = base.value; };
        const refresh = node('button', {class:'btn small', text:'↻', 'aria-label':'Refresh starting models',
          title:'Refresh saved models and the Models folder', onclick:load});
        const go = node('button', {class:'btn primary', text:'Start', onclick:async () => {
          try { await post(route('/training/start'), {name:name.value.trim(), base:base.value}); state.restart = false; delete state.draftBase; delete state.draftName; line.textContent = 'Session started. Place a tile.'; load(); }
          catch (error) { line.textContent = error.message; }
        }});
        sessionBox.replaceChildren(field('Model name', name, 'Versions are saved as this name with _v1, _v2 and so on.'),
          field('Start from', node('div', {class:'inline grow'}, [base, refresh]), 'Saved models are listed newest first in each group. Starts with Segmentation’s selected model. Train continues from these weights and saves a new version; the original is kept.'),
          node('div', {class:'row'}, [go, session ? node('button', {class:'btn small ghost', text:'Cancel', onclick:() => { state.restart = false; drawSession(); }}) : null]));
      };
      const load = async () => {
        const request = ++loadRequest;
        let next;
        try { next = await get(route('/training')); } catch (error) { if (request === loadRequest) tilesBox.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
        if (!box.isConnected || request !== loadRequest) return;
        found = next;
        drawSession(); drawTiles();
        const counts = found.counts;
        // The tiles: shown on the image or not, their cells or not; the list folded, scrolling when long.
        const showTiles = node('input', {type:'checkbox', 'aria-label':'Show the tiles on the image'}); showTiles.checked = state.showTiles;
        showTiles.addEventListener('change', () => { state.showTiles = showTiles.checked; drawTiles(); });
        const showCells = node('input', {type:'checkbox', 'aria-label':'Show the corrected cells in the tiles'}); showCells.checked = !!state.showTileCells;
        showCells.addEventListener('change', () => { state.showTileCells = showCells.checked; drawTileCells(); });
        const list = node('details', {class:'studio-options studio-tile-list'});
        list.open = !!state.tilesOpen;
        list.addEventListener('toggle', () => { state.tilesOpen = list.open; });
        list.append(node('summary', {text:'Tiles · ' + counts.train + ' training · ' + counts.validate + ' validation · ' + found.cells.toLocaleString() + ' cells'}),
          node('div', {class:'studio-tile-scroll'}, found.examples.map(item => node('div', {class:'studio-folder'}, [
            chip(item.role === 'validate' ? 'Validation' : 'Training', item.role === 'validate' ? 'warn' : 'ok'),
            node('span', {class:'grow', text:item.image + ' · ' + item.cells + ' cells', title:item.name}),
            item.image === found.open_image ? node('button', {class:'btn small', text:'Go', title:'Zoom to this tile', onclick:() => zoomAnnotationBox?.(item.bounds)}) : null,
            node('button', {class:'btn small ghost', text:'Delete', onclick:async () => {
              try { await post(route('/training/delete'), {name:item.name}); load(); } catch (error) { line.textContent = error.message; }
            }})]))));
        tilesBox.replaceChildren(node('div', {class:'row wrap'}, [node('label', {class:'inline-check'}, [showTiles, node('span', {text:'Show tiles'})]),
            node('label', {class:'inline-check', title:'The corrected cells kept with each tile on this image'}, [showCells, node('span', {text:'Show their cells'})])]),
          list);
        const session = found.session;
        // The versions in one list, newest first (as the artifact models), with Use to segment with the chosen one.
        const made = (session?.versions || []).slice().reverse();
        const pick = choice(made.map(item => [item.path, item.name + ' · ' + (item.ap_after !== null && item.ap_after !== undefined ?
          'validation AP ' + (item.ap_before ?? 0).toFixed(2) + ' → ' + item.ap_after.toFixed(2) : item.training_tiles + ' tiles')]), made[0]?.path || '', 'Trained version');
        versions.replaceChildren(...(made.length ? [field('Versions', node('div', {class:'inline grow'}, [pick, node('button', {class:'btn small', text:'Use',
          title:'Segment with this version (it becomes the Model above)', onclick:async () => {
            const item = made.find(entry => entry.path === pick.value);
            try { await useModel(item.path); line.textContent = item.name + ' is now Segmentation’s model (saved).'; } catch (error) { line.textContent = error.message; }
          }})]), 'Each round of training makes a new version; the newest is first. Use makes the chosen one Segmentation\u2019s model.')] : []));
        state.rounds = (session?.versions || []).length;
        drawScores();
      };
      // One row per version, all on the same validation tiles: is each round still helping?
      const drawScores = () => {
        const scores = found?.scores, session = found?.session;
        if (!session || !session.versions.length) { scoresBox.replaceChildren(); return; }
        const percent = value => Math.round(100 * value) + '%';
        const parts = [node('div', {class:'row'}, [node('span', {class:'lbl grow', text:'Scores of each version'}),
          help('All on the same validation tiles. Found correctly: the share of your cells the model found. Wrong: cells it drew that ' +
            'you did not. Missed: your cells it did not find. The best version has the fewest wrong and missed together (F1).'), scoreButton])];
        if (scores) {
          const best = scores.rows.length > 1 ? scores.rows.reduce((top, row) => row.f1 > top.f1 ? row : top, scores.rows[0]) : null;
          const table = node('table', {class:'studio-counts'});
          table.append(node('tr', {}, ['Version', 'Found correctly', 'Wrong', 'Missed'].map(text => node('th', {text}))));
          scores.rows.forEach(row => table.append(node('tr', {class:row === best ? 'best' : '', title:'Precision ' + percent(row.precision) +
              ' · F1 ' + row.f1.toFixed(2) + ' · AP ' + row.ap.toFixed(2)}, [node('td', {text:row.name + (row === best ? ' (best)' : '')}),
            node('td', {text:percent(row.recall)}), node('td', {text:String(row.fp)}), node('td', {text:String(row.fn)})])));
          const tiles = scores.tiles.length + ' validation tile' + (scores.tiles.length === 1 ? '' : 's') + ' (' + scores.cells.toLocaleString() + ' of your cells)';
          parts.push(table, node('p', {class:scores.current ? 'muted' : 'hint', text:'On ' + tiles +
            (scores.current ? '.' : '. Tiles or versions changed since: score again.')}));
        } else parts.push(node('p', {class:'muted', text:'Compare the base model and every version on the same validation tiles.'}));
        scoresBox.replaceChildren(...parts);
      };
      scoreButton.addEventListener('click', async () => {
        scoreButton.disabled = true;
        const before = found?.scores?.scored;
        try {
          const {job} = await post(route('/training/scores'), {});
          line.textContent = 'Scoring every version on the validation tiles…'; await refreshJobs();
          for (;;) {  // until the job ends; the table comes with the tiles
            await new Promise(resolve => setTimeout(resolve, 2000));
            if (!box.isConnected) return;
            const now = ((await get(route('/jobs'))).jobs || []).find(item => item.id === job.id);
            if (!now || /failed|cancel/.test(now.status)) { line.textContent = now?.message || 'Scoring stopped.'; break; }
            if (now.status === 'done') { await load(); line.textContent = found?.scores?.scored !== before ? 'Scored every version.' : line.textContent; break; }
          }
        } catch (error) { line.textContent = error.message; }
        finally { scoreButton.disabled = false; }
      });
      const reset = () => {
        dropCells(TRAINING_LAYER); showPreview('training-tile', null);
        state.tile = null; state.predicted = null; keep.disabled = true; discard.hidden = true;
      };
      place.addEventListener('click', async () => {
        if (!found?.session || state.restart) { line.textContent = 'Start a session first: name the model and choose where it starts from.'; return; }
        const point = await pickPoint();
        const image = imageSize();
        if (!point || !image) { line.textContent = 'No tile placed.'; return; }
        const side = Math.min(Number(state.size), image.width, image.height);
        const x = Math.round(Math.min(Math.max(0, point[0] - side / 2), image.width - side));
        const y = Math.round(Math.min(Math.max(0, point[1] - side / 2), image.height - side));
        reset(); state.tile = [x, y, side, side];
        showPreview('training-tile', {outlines:[[[x, y], [x + side, y], [x + side, y + side], [x, y + side]]], kind:'example'});
        discard.hidden = false; line.textContent = 'Predicting the tile with ' + modelLabel(model()) + '…';
        try {
          const prediction = await post(route('/segment_preview'), {bounds:state.tile, settings:{...changes(), ...modelSettings()}});
          state.predicted = prediction.outlines; state.model = model();
          const count = editCells(TRAINING_LAYER, prediction.outlines);
          keep.disabled = false;
          line.textContent = modelLabel(model()) + ' found ' + count + ' cells (' + prediction.seconds + ' s). Correct them with the tools ' +
            'above the image: Freehand draws a missing cell, right-click deletes one, Merge joins pieces, Split cuts; Ctrl+Z undoes. Then Keep tile.';
        } catch (error) { line.textContent = error.message; }
      });
      keep.addEventListener('click', async () => {
        const polygons = readCells(TRAINING_LAYER) || [];
        keep.disabled = true; line.textContent = 'Keeping ' + polygons.length + ' cells…';
        try {
          const saved = await post(route('/training/examples'), {bounds:state.tile, polygons, predicted:state.predicted, role:state.role,
            model:state.model, settings:changes()});
          reset(); line.textContent = 'Kept ' + saved.saved_cells + ' cells as a ' + (state.role === 'validate' ? 'validation' : 'training') + ' tile.'; load();
        } catch (error) { line.textContent = error.message; keep.disabled = false; }
      });
      discard.addEventListener('click', () => { reset(); line.textContent = 'Tile discarded.'; });
      trainButton.addEventListener('click', async () => {
        trainButton.disabled = true;
        try {
          const answer = await runCommand('train_cellpose', {epochs:Math.round(Number(epochs.value))}, {confirmed:true});
          showResult(answer); line.textContent = answer.message;
          if (answer.status === 'done') { await refreshJobs(); follow(); }
        } catch (error) { line.textContent = error.message; }
        finally { trainButton.disabled = false; }
      });
      validateButton.addEventListener('click', async () => {
        try {
          const result = await post(route('/training/validate'), {});
          const all = result.overall, percent = value => Math.round(100 * value) + '%';
          const add = node('button', {class:'btn small', text:'Add these tiles to training', onclick:async () => {
            try { await post(route('/training/role'), {names:result.tiles.map(item => item.name), role:'train'}); checked.replaceChildren(); load(); line.textContent = 'The validation tiles now train the model too.'; }
            catch (error) { line.textContent = error.message; }
          }});
          checked.replaceChildren(node('div', {class:'row wrap'}, [chip('AP ' + all.ap.toFixed(2), all.ap >= 0.8 ? 'ok' : 'warn'),
            chip('Precision ' + percent(all.precision)), chip('Recall ' + percent(all.recall)), chip('F1 ' + all.f1.toFixed(2))]),
            node('p', {class:'muted', text:all.tp + ' cells found as you drew them, ' + all.fp + ' extra, ' + all.fn + ' missed, over ' + result.tiles.length +
              ' validation tile' + (result.tiles.length === 1 ? '' : 's') + '.'}),
            ...result.tiles.map(item => node('div', {class:'studio-folder'}, [node('span', {class:'grow', text:item.image + ' · ' + modelLabel(item.model), title:item.name}),
              node('span', {class:'muted', text:item.predicted + ' predicted / ' + item.corrected + ' yours · F1 ' + item.f1.toFixed(2)})])),
            node('div', {class:'row'}, [add]));
        } catch (error) { line.textContent = error.message; }
      });
      // Another image: its own tiles are drawn, and a tile being corrected is left behind.
      current.refreshTrainingModels = load;
      const markersBefore = current.refreshMarkers;
      current.refreshMarkers = () => { markersBefore?.(); reset(); load(); };
      // After Train, look again until the new version is listed.
      let timer = null;
      const follow = async () => {
        clearTimeout(timer);
        const before = state.rounds || 0; await load();
        if (box.isConnected && (state.rounds || 0) === before) timer = setTimeout(follow, 5000);
        else if (box.isConnected) line.textContent = 'Trained ' + modelLabel(model()) + '. Place the next tile: this version predicts it.';
      };
      load();
    };

    // ------------------------------------------------------------ Phenotyping
    const phenotypingInputs = (current, body) => {
      const controls = node('div'), counts = node('div', {class:'row wrap'}), status = node('p', {class:'muted', role:'status'});
      const rows = node('div', {class:'studio-pairs-scroll'});
      let found;
      body.append(controls, counts, rows, status);
      const drawRows = () => {
        if (!found) return;
        rows.replaceChildren(node('table', {class:'studio-counts studio-pairs studio-phenotyping-pairs'}, [
          node('tr', {}, ['ROI image', 'Slide', 'Cells', 'Methods'].map(text => node('th', {text}))),
          ...found.rows.map(row => {
            const openImage = async () => {
              if (!row.image_path) return;
              const added = await post('/session/' + encodeURIComponent(session) + '/add-image', {path:row.image_path});
              await chooseImage(added.images.find(item => item.id === added.added));
            };
            const open = node('button', {class:'studio-item-name studio-file', text:row.image || row.roi, title:row.image_path || 'No matching ROI image', disabled:!row.image_path, onclick:async () => {
              try {
                await openImage();
              } catch (error) { status.textContent = error.message; }
            }});
            const badges = node('span', {class:'studio-stages-done'});
            (found.methods || []).forEach(item => {
              const column = item.columns.find(c => row.results[c] > 0) || item.columns[0];
              const count = row.results[column] || 0;
              const title = item.name + ': ' + (count ? count.toLocaleString() + ' / ' + row.cells.toLocaleString() + ' cells in ' + column : 'No saved result') + '. Click to open this method.';
              const badge = node('b', {'data-phenotyping-method':item.id, role:'button', tabindex:'0',text:item.letter,
                class:count ? (count >= row.cells ? 'done' : 'classified') : '',title,'aria-label':title});
              const go = async () => {
                try { await openImage(); current.selectMethod?.(item.id); if (column) { await hooks.refreshPhenotypingLinks?.(); await colourCells(column); await loadColours(current); current.colour.value=column; } }
                catch(error){status.textContent=error.message;}
              };
              badge.addEventListener('click',go);
              badge.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();go();}});
              badges.append(badge);
            });
            return node('tr', {'data-roi':row.roi}, [node('td', {}, [open]), node('td', {text:row.slide || '–'}), node('td', {class:'mono', text:row.cells ? row.cells.toLocaleString() : 'No cells'}), node('td', {}, [badges])]);
          })]));
      };
      const draw = value => {
        found = value;
        status.textContent = (found.warnings || []).join(' ');
        current.tablePath = value.path;
        const mutate = async (payload, tableChanged=false) => {
          status.textContent = 'Saving…';
          try {
            draw(await post(route('/phenotyping_inputs'), payload)); status.textContent = 'Saved.';
            if (payload.roles) window.dispatchEvent(new CustomEvent('studio-column-roles-changed'));
            if (payload.roles) await hooks.refreshPhenotypingLinks?.();
            if (tableChanged) {
              const name = (currentSpec?.layers || []).find(layer => layer.type === 'image')?.name;
              const image = (await loadImages()).find(item => item.name === name);
              if (image) await chooseImage(image);
              await chooseStep('tab7');
            }
          } catch (error) { status.textContent = error.message; }
        };
        const directory = (key, label, tip) => {
          const input = node('input', {type:'text', class:'grow', value:found.folders[key], 'aria-label':'Phenotyping ' + label.toLowerCase() + ' folder'});
          input.addEventListener('change', () => mutate({folders:{[key]:input.value}}));
          return field(label, input, tip);
        };
        const tables = Array.from(new Set([found.path, ...found.tables].filter(Boolean)));
        const table = choice([['', 'Choose a combined table…'], ...tables.map(path => [path, path.split('/').pop()])], found.path, 'Phenotyping combined table');
        table.title = found.path;
        table.addEventListener('change', () => { if (table.value) mutate({table:table.value}, true); });
        const roles = node('details', {class:'studio-options'}, [node('summary', {text:'Column roles'})]);
        const roleDraft = {...found.roles, label:found.segmentation_label || ''};
        [['roi', 'ROI', 'The column identifying the image or ROI for each cell.'],
         ['slide', 'Slide', 'The column identifying the physical slide containing each ROI, usually slide. Patient/timepoint sample IDs may identify a different grouping.'],
         ['cell_id', 'Cell ID', 'The unique ID of each cell across the combined table. Batch correction creates cell_id. Local segmentation labels repeat between images and cannot be used here.'],
         ['label', 'Segmentation label', 'The original numbered cell in the mask or boundaries, usually label. Together with ROI it connects each measurement to its polygon. This differs from the unique merged Cell ID.']].forEach(([key, label, tip]) => {
          const pick = choice([['', key === 'slide' ? 'No slide column' : 'Choose a column…'], ...found.columns.map(name => [name, name])], roleDraft[key], 'Phenotyping ' + label + ' column');
          pick.addEventListener('change', () => {
            roleDraft[key] = pick.value;
            if (!roleDraft.roi || !roleDraft.cell_id) { status.textContent = 'Choose the ROI and Cell ID columns to save.'; return; }
            mutate({roles:{...roleDraft}});
          });
          roles.append(field(label, pick, tip));
        });
        fill(controls,
          directory('images', 'Images', 'ROI images to open in the viewer. They are matched to the ROI names in the combined table.'),
          directory('tables', 'Tables', 'Folder containing combined CSV or Parquet measurements. Choose the Batch correction output below.'),
          field('Cell table', table, 'The combined input for whole-table phenotyping. Batch correction’s Use this table selects it here and remembers it across restarts. Image layer choices do not change the input for whole-table clustering.'),
          directory('output', 'Output', 'Phenotyping output folder shared with classic. Result columns are saved beside the selected combined table; jobs and their logs remain in the project’s Jobs list.'), roles);
        controls.append(node('button', {class:'btn small',text:'View data',onclick:() => hooks.showCombinedData?.()}));
        fill(counts, chip(found.rows.length + ' ROIs'), chip(found.cells.toLocaleString() + ' cells'),
          found.cell_id_column ? chip('Cell IDs available', 'ok', 'Cell IDs are checked for missing or repeated values before a run.') : chip('Choose Cell ID', 'warn', 'Choose the unique Cell ID column under Column roles, or select the merged output from Batch correction.'),
          chip(found.result_columns.length + ' result columns'));
        drawRows();
      };
      const load = async () => {
        try { const value = await get(route('/phenotyping_inputs')); if (body.isConnected) draw(value); }
        catch (error) { if (body.isConnected) status.textContent = error.message; }
      };
      const refresh = async () => { await load(); await hooks.refreshPhenotypingLinks?.(); loadColours(current); };
      current.refreshItems = refresh;
      followJobs(current, ['leiden', 'som', 'vortex', 'mr_ml', 'name_clusters', 'consensus_ml', 'subphenotype', 'resolve_all'], refresh);
      load();
    };
    TOOLS.tab7 = current => {
      const picker = choice(METHODS.map(item => [item.id, item.name + (item.studio ? '' : ' · classic')]), method, 'Phenotyping method');
      const tip = help('');
      const colour = node('select', {'aria-label':'Colour cells by'});
      const {box, body} = section('Method');
      body.append(node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Method'}), picker, tip]),
        field('Colour by', colour, 'Colour the cells on the image by any phenotype column of the table.'));
      const holder = node('div', {class:'studio-method'});
      const cells = section('Folders and tables', {tip:'Choose the combined measurements from Batch correction. Click an ROI image to open it. L Leiden · S SOM · V Vortex · T Threshold/RESOLVE · C Consensus ML · M MR-ML · R Manual reference. Green means saved results exist; a dashed border means only some cells have values. Click a letter to open that method and its result.'});
      current.body.append(cells.box, freshnessNote(['combined']), box, holder);
      phenotypingInputs(current, cells.body);
      current.colour = colour;
      colour.addEventListener('change', () => { if (colour.value !== '__live__') colourCells(colour.value); });
      loadColours(current);
      const draw = () => {
        const chosen = METHODS.find(item => item.id === method) || METHODS.find(item => item.id === 'resolve');
        method = chosen.id; picker.value = method; setTip(tip, chosen.tip);
        holder.replaceChildren(); selectionTools = null; current.runBar([]); current.sync = null; current.publicationRun = null; current.publicationUpdate?.(); clearFigures();
        (PANELS[chosen.id] || classicMethod)(current, holder, chosen);
      };
      picker.addEventListener('change', () => { method = picker.value; remember('method', method); draw(); });
      current.selectMethod = selected => { if (METHODS.some(item=>item.id===selected)) { method=selected;remember('method',method);draw(); } };
      draw();
      const figure = publicationFigure(current), heading = figure.querySelector('.studio-sec-h');
      heading.style.cssText = 'display:inline-flex; width:calc(100% - 18px); margin-bottom:0; vertical-align:middle';
      figure.replaceChildren(node('details', {class:'studio-options'}, [node('summary', {}, [heading]), ...figure.children]));
      current.body.append(figure);
    };
    // Shared by every phenotyping method: a heatmap or dot plot of any phenotype or cluster column, as classic makes them.
    const publicationFigure = current => {
      const {box, body} = section('Publication figure', {tip:'A clustered heatmap or a dot plot (dot size: share of the group’s cells above the ' +
        'threshold; colour: mean expression; with cell counts) of any phenotype or cluster column, for the markers chosen. The same figures as ' +
        'classic Tab 7’s, as PNG, SVG or PDF. Current clustering uses the full saved run and its transformations; clusters with the same name are averaged as in classic. Full combined table includes every labelled cell in all images for heatmaps. Dot plots sample up to 20,000 cells per group.'});
      const column = node('select', {'aria-label':'Figure column'});
      const kind = choice([['heatmap', 'Heatmap'], ['dotplot', 'Dot plot with cell counts']], 'heatmap', 'Figure kind');
      const source=choice([['run','Current clustering'],['table','Full combined table']], 'table', 'Figure data source');
      let sourceChosen=false;
      const values = {markers:[]};
      const markers = markerPicker({method:'figure', label:'Figure', values, changed:() => {}});
      const threshold = node('input', {type:'number', step:'0.1', value:'0.5', 'aria-label':'Positivity threshold'});
      let options = figureOptions({summary:true});
      const optionsHolder = node('div', {}, [options.box]);
      const thresholdRow = field('Positive above', threshold, 'For the dot plot: a cell counts as positive for a marker above this z-score (markers arcsinh-scaled, then z-scored).');
      thresholdRow.hidden = true;
      kind.addEventListener('change', () => { options = figureOptions({dot:kind.value === 'dotplot',summary:kind.value === 'heatmap'}); optionsHolder.replaceChildren(options.box); thresholdRow.hidden = kind.value !== 'dotplot'; current.publicationUpdate(); });
      const note = node('p', {class:'muted', role:'status'});
      const make = node('button', {class:'btn primary', text:'Make figure', onclick:async () => {
        const run=current.publicationRun?.(), fromRun=source.value==='run' && kind.value==='heatmap';
        if (fromRun && !run) { note.textContent='Choose a finished clustering, or Full combined table.'; return; }
        if (!fromRun && !column.value) { note.textContent = 'Choose a column.'; return; }
        if (!fromRun && !values.markers.length) { note.textContent = 'Choose the markers.'; return; }
        make.disabled = true; note.textContent = 'Drawing (a few seconds on a large table)…';
        try {
          const title=fromRun ? run.method+' phenotype heatmap' : column.value;
          const found = await post(route('/figure'), {source:fromRun?'run':'table', job_id:fromRun?run.job_id:undefined, names:fromRun?run.names:undefined,
            column:column.value, markers:values.markers, kind:kind.value, title,
            options:{...options.read(), threshold:Number(threshold.value)}});
          const url = downloadFile(found);
          note.textContent = 'Saved ' + found.name + '.';
          if (found.type !== 'application/pdf') {
            const {box:shown, body:inside} = section('Figure: ' + title, {});
            inside.append(node('img', {src:url, alt:'The figure', class:'studio-figure-preview'}));
            showFigure('publication', shown);
          }
        } catch (error) { note.textContent = error.message; }
        finally { make.disabled = false; }
      }});
      const columnRow=field('Column', column, 'Any phenotype or cluster column of the full combined table. Image and Data-panel filters do not limit this figure.');
      const markersRow=node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Markers'})]), markers]);
      const scope=node('span',{class:'muted',role:'status'});
      current.publicationUpdate=()=>{
        const run=current.publicationRun?.(), allowed=!!run && kind.value==='heatmap';
        source.querySelector('option[value=run]').disabled=!allowed;
        if(!sourceChosen)source.value=allowed?'run':'table';
        if(!allowed)source.value='table';
        const fromRun=source.value==='run';columnRow.hidden=markersRow.hidden=fromRun;
        scope.textContent=fromRun?'Full saved run · '+Number(run.cells||0).toLocaleString()+' cells':'All images in the combined table';
      };
      source.onchange=()=>{sourceChosen=true;current.publicationUpdate();};
      body.append(field('Data',source,'Current clustering uses all analysed cells and the saved transformations, regardless of the image shown. Full combined table calculates a profile for any labelled column across every image.'),scope,
        field('Figure', kind),columnRow,markersRow,thresholdRow,optionsHolder,node('div', {class:'row'}, [make]),note);
      current.publicationUpdate();
      // Read again whenever the list is opened: a run or new names add columns.
      const loadColumns = async () => {
        let found = {phenotypes:[]};
        try { found = await get(route('/marker_choices') + '?method=figure'); } catch (error) { note.textContent = error.message; return; }
        const before = column.value, columns = found.phenotypes || [];
        column.replaceChildren(...columns.map(name => node('option', {value:name, text:name})));
        if (columns.includes(before)) column.value = before;
      };
      column.addEventListener('focus', loadColumns); column.addEventListener('pointerdown', loadColumns);
      loadColumns();
      return box;
    };

    const loadColours = async current => {
      const colour = current.colour;
      if (!colour) return;
      try {
        const available = await get(route(current.id === 'tab7' ? '/phenotyping_inputs' : '/selection_options'));
        current.columns = available.columns;
      } catch (_) { current.columns = []; }
      syncColour(current);
    };
    const syncColour = current => {
      const colour = current?.colour;
      if (!colour || !colour.isConnected || document.activeElement === colour) return;
      const cells = cellsState();
      const options = [['', 'Single colour'], ...(current.columns || []).map(name => [name, name])];
      if (cells?.live) options.unshift(['__live__', 'Live RESOLVE']);
      if (cells?.color_by && !options.some(([key]) => key === cells.color_by)) options.push([cells.color_by, cells.color_by]);
      colour.replaceChildren(...options.map(([key, text]) => node('option', {value:key, text})));
      colour.value = cells ? (cells.live ? '__live__' : cells.color_by) : '';
      colour.disabled = !cells;
    };

    const PANELS = {};
    PANELS.resolve = async (current, holder) => {
      holder.append(node('p', {class:'muted', text:'Reading RESOLVE settings…'}));
      let saved;
      try { saved = await get(route('/resolve_setup')); }
      catch (error) { holder.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected || method !== 'resolve') return;
      holder.replaceChildren();
      const values = {...saved.values}, note = node('p', {class:'muted', role:'status'});
      let thresholds = saved.thresholds, busy = false;
      const controls = {}, dirty = () => { note.textContent = 'Not saved'; };
      const request = (action, extra={}) => post(route('/resolve_setup'), {action, values:read(), rules:rules.value, thresholds, revision:saved.revision, ...extra});
      const input = (key, label, type='text') => {
        const item = node('input', {type, 'aria-label':label, ...(type === 'number' ? {step:'any'} : {})});
        if (type === 'checkbox') item.checked = !!values[key]; else item.value = values[key] ?? '';
        item.addEventListener('input', dirty); controls[key] = item; return item;
      };
      const select = (key, label, choices) => {
        const item = choice(choices, values[key], label); controls[key] = item;
        item.addEventListener('change', dirty); return item;
      };
      const positivity = section('Marker positivity', {tip:'A marker is positive above its threshold. Background gates come from image pixels; local automatic gates come from cell measurements and can differ. Review local gates before using them. Each ROI is processed separately, as in classic RESOLVE.'});
      const mode = select('threshold_mode', 'RESOLVE positivity method', saved.modes.map(m => [m, m === 'Predefined (CSV)' ? 'Saved / editable thresholds' : m === 'None (No Thresholds)' ? 'Above zero (no thresholds)' : m]));
      const space = select('input_space', 'RESOLVE measurement scale', [['raw','Measured values (raw)'],['arcsinh','Already arcsinh (corrected)']]);
      const transform = select('tie_score_transform', 'RESOLVE score transformation', [['identity_residual','No extra transformation'],['asinh_difference','Arcsinh'],['log1p_difference','Log1p']]);
      const percentileInput=input('threshold_percentile','Percentile','number');percentileInput.min=0;percentileInput.max=100;
      const percentileRow=field('Percentile',percentileInput);
      const manualRow=field('Manual cutoff',input('threshold_manual','Manual cutoff','number'));
      const cutoff = input('z1_cutoff', 'RESOLVE Z1 cutoff', 'number');
      const cutoffRow = field('Z1 cutoff', cutoff, 'Positive above this Z-score within its ROI. This is a relative signal rule; it is not an absolute positive/negative measurement.');
      const adaptiveRow=field('Adaptive gates',input('adaptive_hierarchy','RESOLVE adaptive parent child gates','checkbox'),
        'Classic’s parent/child refinement, used only in Z1 or Above-zero mode. Otsu separates distinguishing markers (such as FoxP3 for Treg versus CD4) across parent candidates in the selected dataset. It can replace their initial positive calls. Turn off to keep only the chosen positivity rule.');
      let previousMarkerSource=null;
      const markerList = markerPicker({method:'resolve', label:'RESOLVE', values, changed:() => {
        dirty();
        const chosenSpace = holder.querySelector('[aria-label="RESOLVE marker values"]')?.value;
        if (chosenSpace && previousMarkerSource && chosenSpace!==previousMarkerSource) { space.value = chosenSpace === 'corrected' ? 'arcsinh' : 'raw'; if (space.value === 'arcsinh') transform.value = 'identity_residual'; }
        if(chosenSpace)previousMarkerSource=chosenSpace;
      }});
      const markerFold = node('details', {class:'studio-options'}, [node('summary', {text:'Markers'}), markerList]);
      positivity.body.append(markerFold, field('Positive signal', mode),percentileRow,manualRow, field('Value scale', space,
        'Background gates and cell measurements must come from the same image processing and value scale. Corrected values already in arcsinh space must not be transformed with arcsinh again. Calculate or enter corrected gates on that same scale.'), cutoffRow,adaptiveRow);
      const gates = node('details', {class:'studio-options', open:true}, [node('summary', {text:'Thresholds'})]);
      const directory = node('input', {type:'text',value:saved.threshold_folder,'aria-label':'RESOLVE Background folder'});
      const source = select('threshold_source', 'RESOLVE threshold source', ['Auto','Polygon values','Final scalar'].map(x => [x,x]));
      const statistic = select('threshold_statistic', 'RESOLVE background statistic', [['median','Median'],['mean','Mean'],['min','Minimum'],['percentile_05','5th percentile'],['percentile_25','25th percentile'],['percentile_75','75th percentile']]);
      const importing = node('details', {class:'studio-options'}, [node('summary',{text:'Background thresholds'}), field('Folder',directory),
        field('Read',source,'Auto uses manual overrides, then background polygons, then the final scalar. Polygon values ignores scalar overrides; Final scalar uses the saved final threshold.'),
        field('Polygons',statistic,'Median is less affected by an unusual background polygon. Mean averages polygons; minimum and lower percentiles make a less strict gate. These percentiles describe background polygons, not cells.')]);
      const thresholdsView = node('div'), roiPick = node('select', {'aria-label':'RESOLVE threshold ROI'});
      const drawGates = () => {
        const before = roiPick.value;
        roiPick.replaceChildren(...thresholds.map(row => node('option', {value:row.batch_id,text:row.batch_id})));
        if (thresholds.some(row => row.batch_id === before)) roiPick.value = before;
        const row = thresholds.find(row => row.batch_id === roiPick.value);
        thresholdsView.replaceChildren(...(row ? values.markers.map(marker => {
          const edit = node('input',{type:'number',step:'any',value:row[marker] ?? '', 'aria-label':marker + ' threshold'});
          edit.addEventListener('input',()=>{row[marker] = edit.value === '' ? null : Number(edit.value); mode.value='Predefined (CSV)'; dirty();});
          return field(marker,edit);
        }) : [node('p',{class:'muted',text:'No thresholds loaded yet.'})]));
      };
      roiPick.addEventListener('change',drawGates);
      const act = async (action, extra={}) => {
        if(busy)return; busy=true; note.textContent=action==='draft'?'Reading reference examples…':'Reading marker values…';
        holder.inert=true; output.disabled=overwrite.input.disabled=save.disabled=run.disabled=true;
        try {
          const answer=await request(action,extra);
          if(answer.thresholds){ thresholds=answer.thresholds; mode.value='Predefined (CSV)'; updateMode(); drawGates(); note.textContent=thresholds.length+' ROI thresholds ready to review. Save to use them.'; }
          if(answer.panel) note.textContent=answer.panel.table.length+' rules checked. Save to use them.';
          if(action==='draft'){
            if(answer.rules) rules.value=answer.rules;
            draftReport.replaceChildren(node('p',{class:'muted',text:answer.note}),
              ...answer.evidence.map(row=>node('p',{class:'muted',text:row.phenotype+': '+row.matched+'/'+row.labelled+' reference cells match; '+Math.round(row.purity*100)+'% of matched references have this label.'})),
              ...answer.skipped.map(text=>node('p',{class:'muted',text})));
            note.textContent=answer.rules?'Draft rules ready. Review, then Save settings.':'No reliable draft rule found; existing text kept.';
          }
        }catch(error){note.textContent=error.message;}
        finally{busy=false; holder.inert=false;output.disabled=overwrite.input.disabled=save.disabled=run.disabled=false;}
      };
      const loadBackground=node('button',{class:'btn small',text:'Load Background',title:'Open Background detection',onclick:async()=>{
        await chooseStep('tab2');
        const detection=document.getElementById('studio-background-detection');
        detection?.classList.add('studio-attention');detection?.setAttribute('tabindex','-1');detection?.focus();detection?.scrollIntoView({block:'center'});
        setTimeout(()=>detection?.classList.remove('studio-attention'),4000);
      }});
      importing.append(loadBackground,node('button',{class:'btn small',text:'Import saved thresholds',onclick:()=>act('background',{folder:directory.value})}));
      const calculate=node('button',{class:'btn small',text:'Calculate thresholds',onclick:()=>act('calculate')});
      const modeNote=node('p',{class:'muted'});
      gates.append(importing,calculate,modeNote,field('ROI',roiPick),node('div',{class:'studio-resolve-gates'},[thresholdsView]));
      positivity.body.append(gates);
      const ruleSection=section('Rules',{tip:'One phenotype per line. Markers inside [brackets] must all be positive. OR adds another accepted pattern. SUPPORT helps break ties. NOT follows classic RESOLVE’s score-based reassignment; it is not a strict negative gate.'});
      const rules=node('textarea',{rows:'7',class:'studio-rule-text','aria-label':'RESOLVE rules',placeholder:'CD8 = [CD3e, CD8]\nTreg = [CD3e, CD4, FoxP3]'});
      rules.value=saved.rules; rules.addEventListener('input',dirty);
      const builder=node('details',{class:'studio-options'},[node('summary',{text:'Build a rule'})]);
      const name=node('input',{'aria-label':'Rule phenotype',placeholder:'e.g. CD8'});
      const required=node('input',{'aria-label':'Required rule markers',placeholder:'CD3e, CD8'});
      const alternative=node('input',{'aria-label':'Alternative rule markers',placeholder:'Optional: another complete marker pattern'});
      const support=node('input',{'aria-label':'Supportive rule markers',placeholder:'Optional, separated by commas'});
      const excluded=node('input',{'aria-label':'Exclusion rule markers',placeholder:'Optional, separated by commas'});
      builder.append(field('Phenotype',name),field('Required',required,'Separate marker names with commas. Every marker in this pattern must be positive.'),field('Or',alternative,'An alternative pattern. Every marker here must be positive; either pattern can match.'),field('Support',support),field('Exclude',excluded,'Classic RESOLVE reassigns a call if an exclusion marker’s score exceeds the strongest required marker. This is not a hard negative gate.'),
        node('button',{class:'btn small',text:'Add rule',onclick:()=>{
          if(!name.value.trim()||!required.value.trim()){note.textContent='Enter a phenotype and its required markers.';return;}
          let line=name.value.trim()+' = ['+required.value.trim()+']';
          if(alternative.value.trim())line+=' OR ['+alternative.value.trim()+']';
          if(support.value.trim())line+='; SUPPORT=['+support.value.trim()+']';
          if(excluded.value.trim())line+='; NOT=['+excluded.value.trim()+']';
          rules.value+=(rules.value.trim()?'\n':'')+line; dirty();
        }}));
      const reference=choice([['','Choose a reference…'],...saved.references.map(x=>[x,x])],'','RESOLVE draft reference');
      const draftReport=node('div');
      const draft=node('details',{class:'studio-options'},[node('summary',{text:'Draft from Manual reference'}),field('Reference',reference,
        'Uses your labelled cells and the chosen positivity settings to propose up to three required markers per phenotype. At least five examples of two phenotypes are needed. Drafts replace the editor text only; review them before saving.'),
        node('button',{class:'btn small',text:'Draft rules',onclick:()=>act('draft',{reference:reference.value})}),draftReport]);
      ruleSection.body.append(rules,node('div',{class:'row'},[node('button',{class:'btn small',text:'Check rules',onclick:()=>act('validate')}),help('Example: CD8 = [CD3e, CD8]; SUPPORT=[CD45]. Use exact names from the chosen measurements, including _corr for corrected columns.')]),builder,draft);
      const advanced=node('details',{class:'studio-options studio-sec'},[node('summary',{text:'Tie scores and unmatched cells'})]);
      const cofactor=input('tie_score_cofactor','RESOLVE arcsinh cofactor','number');
      const normalizations=[['','None'],['Z1','Z-score each marker'],['Z2','Z-score each cell'],['CDF','Normal CDF'],['NLog','−log(1 − CDF)']];
      const slots=Array.from({length:4},(_,i)=>choice(normalizations,values.norm_order[i]||'','RESOLVE normalization step '+(i+1)));
      slots.forEach(item=>item.addEventListener('change',dirty));
      advanced.append(field('Scores',transform,'Only used to decide between matching phenotypes. Positivity stays above the saved threshold. Already-arcsinh values need no extra transformation.'),field('Cofactor',cofactor),
        ...slots.map((item,i)=>field('Step '+(i+1),item)),field('Unmatched',input('unclassified_label','RESOLVE unmatched label'), 'Cells that match no rule receive this name.'));
      const filter=node('details',{class:'studio-options studio-sec'},[node('summary',{text:'Cell filter (shared with classic)'}),
        field('Use filter',input('enable_filter','RESOLVE filter enabled','checkbox'),'Only matching cells are used to calculate thresholds, draft rules and assign phenotypes. Other cells have no result.'),
        field('Column',select('filter_col','RESOLVE filter column',[['','Choose a column…'],...saved.columns.map(c=>[c,c])])),
        field('Equals',input('filter_val','RESOLVE filter value'))]);
      const output=input('output_col','RESOLVE output column');
      const overwrite=overwriteOption([output]);
      const read=()=>{
        Object.entries(controls).forEach(([key,item])=>{values[key]=item.type==='number'?Number(item.value):item.type==='checkbox'?item.checked:item.value;});
        values.norm_order=slots.map(item=>item.value).filter(Boolean);
        ['Z1','Z2','CDF','NLog'].forEach((step,i)=>{values[['norm_z1','norm_z2','norm_cdf','norm_nlog'][i]]=values.norm_order.includes(step);});
        return {...values};
      };
      const persist=async()=>{saved=await request('save'); note.textContent='Saved for Studio and classic.';};
      const save=node('button',{class:'btn',text:'Save settings',onclick:async()=>{
        if(busy)return;busy=true;holder.inert=true;output.disabled=overwrite.input.disabled=save.disabled=run.disabled=true;
        try{await persist();}catch(error){note.textContent=error.message;}finally{busy=false;holder.inert=false;output.disabled=overwrite.input.disabled=save.disabled=run.disabled=false;}
      }});
      const run=node('button',{class:'btn primary',text:'Run RESOLVE',onclick:async()=>{
        if(busy)return;busy=true;holder.inert=true;output.disabled=overwrite.input.disabled=save.disabled=run.disabled=true;
        try{await persist(); const answer=await runCommand('run_resolve_all',{column:output.value,overwrite:overwrite.input.checked,settings_revision:saved.revision},{confirmed:true});
          showResult(answer);note.textContent=answer.message;if(answer.status==='done')overwrite.input.checked=false;await refreshJobs();showTab('jobs');
        }catch(error){note.textContent=error.message;}finally{busy=false;holder.inert=false;output.disabled=overwrite.input.disabled=save.disabled=run.disabled=false;}
      }});
      const updateMode=()=>{
        percentileRow.hidden=mode.value!=='Percentile';manualRow.hidden=mode.value!=='Manual';
        const z1=mode.value==='Z1 score (no threshold files)', none=mode.value==='None (No Thresholds)';
        cutoffRow.hidden=!z1; gates.hidden=z1||none;
        adaptiveRow.hidden=!(z1||none);
        calculate.disabled=mode.value==='Predefined (CSV)'||z1||none;
        modeNote.textContent=mode.value==='Predefined (CSV)'?'Edit gates below, then Save settings.':'Run calculates new '+mode.value+' gates. Calculate first to review and edit them.';
        transform.disabled=z1; slots.forEach(item=>{item.disabled=z1;});
      };
      mode.addEventListener('change',updateMode);
      space.addEventListener('change',()=>{if(space.value==='arcsinh')transform.value='identity_residual';});
      holder.append(positivity.box,ruleSection.box,advanced,filter,note);
      current.runBar([field('Save as',output),overwrite.row,node('div',{class:'row'},[save,run])], 'Runs on the whole combined table, using each ROI’s thresholds.');
      drawGates();updateMode();note.textContent='';
    };
    PANELS.reference = (current, holder) => {
      const labels = node('section', {id:'studio-labels', class:'studio-sec'});
      const rules = node('section', {class:'studio-sec'});
      holder.append(labels, rules);
      renderLabels(labels);
      selectionTools = window.createStudioSelection({holder:rules, node, get, session, runCommand, showResult, setSelectionTool});
      selectionTools.update(currentSpec);
    };
    let referencePredictionIntent = null;
    window.addEventListener('studio-reference-method', async event => {
      if (!['reference','mr_ml'].includes(event.detail?.method)) return;
      referencePredictionIntent = event.detail.method === 'mr_ml' ? event.detail : null;
      method=event.detail.method;remember('method',method);
      await chooseStep('tab7');
    });
    document.addEventListener('spacemap-reference-updated', () => {
      const current=frame;
      if(current?.id==='tab7') { loadColours(current); clearTimeout(current.referenceCountsTimer); current.referenceCountsTimer=setTimeout(()=>current.refreshItems?.(),600); }
    });
    PANELS.mr_ml = (current, holder) => {
      current.mrmlIntent = referencePredictionIntent; referencePredictionIntent = null;
      const training = node('section', {id:'studio-training', class:'studio-sec'});
      holder.append(training);
      let loaded=null, pending=false, retryAt=0, retryTimer=null;
      const identity=()=>JSON.stringify((currentSpec?.layers||[]).filter(l=>['image','cells'].includes(l.type)).map(l=>[l.type,l.token,l.table]));
      // This panel may open before the restored viewer session has its image.
      // Retry once it arrives; never leave the first empty-session response in place.
      const load=async()=>{
        if(!training.isConnected||pending)return;
        const key=identity();if(key===loaded||Date.now()<retryAt)return;
        if(!(currentSpec?.layers||[]).some(l=>l.type==='image')){
          training.replaceChildren(node('p',{class:'muted',text:'Waiting for an image with cell measurements…'}));return;
        }
        pending=true;clearTimeout(retryTimer);
        const content=node('div');training.replaceChildren(node('p',{class:'muted',text:'Reading cell measurements…'}),content);
        try{
          await trainingOptions(content, current);
          if(training.isConnected&&key===identity()){training.replaceChildren(content);loaded=key;retryAt=0;}
        }catch(error){
          if(training.isConnected&&key===identity()){
            const waiting=/Open an image first|reopen.*image|reconnect/i.test(error.message);
            training.replaceChildren(node('p',{class:'hint',text:waiting?'Connecting to the open image…':error.message}),
              node('button',{class:'btn small',text:'Retry',onclick:()=>{loaded=null;retryAt=0;load();}}));
            if(waiting){retryAt=Date.now()+1500;retryTimer=setTimeout(load,1550);}
            else loaded=key; // explicit Retry for a persistent data error
          }
        }finally{pending=false;if(training.isConnected&&key!==identity()){retryAt=0;load();}}
      };
      current.sync=load;
      load();
    };
    // The groups of the last finished job of *kind* (Leiden clusters, neighbourhoods): size, what marks them,
    // Show on the image, a name for each and Save names (a new column). Returns a function that looks again
    // and keeps looking while a run of that kind is going.
    // The markers a clustering method uses: every marker of the combined table by name, as measured or batch corrected
    // (the settings keep the real columns, CD45_corr say), with the saved sets of the viewer and Tab 7 one pick away.
    const markerPicker = ({method, label, values, changed}) => {
      const holder = node('div', {class:'studio-marker-picker'}, [node('p', {class:'muted', text:'Reading the markers…'})]);
      (async () => {
        let found;
        try { found = await get(route('/marker_choices') + '?method=' + encodeURIComponent(method)); }
        catch (error) { holder.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
        const suffix = found.suffix, chosen = new Set(found.chosen);
        let source = found.source;
        const anyCorrected = found.markers.some(item => item.corrected);
        const values_ = choice([['raw', 'As measured (raw)'], ['corrected', 'Batch corrected (' + suffix + ')']], source, label + ' marker values');
        values_.querySelector('option[value=corrected]').disabled = !anyCorrected;
        const sets = node('select', {'aria-label':label + ' marker set'}, [node('option', {value:'', text:found.sets.length ? 'Use a saved set…' : 'No saved sets yet'}),
          ...found.sets.map((item, index) => node('option', {value:String(index), text:item.name + ' (' + item.markers.length + ', ' + item.from + ')'}))]);
        sets.disabled = !found.sets.length;
        const count = chip(''), note = node('p', {class:'muted'});
        const grid = node('div', {class:'studio-markers studio-marker-grid'});
        const usable = item => source === 'corrected' ? item.corrected : item.raw;
        const save = () => {
          values.markers = found.markers.filter(item => chosen.has(item.name) && usable(item)).map(item => source === 'corrected' ? item.name + suffix : item.name);
          count.textContent = values.markers.length + ' chosen'; count.className = 'chip' + (values.markers.length ? ' ok' : ' warn');
          changed();
        };
        const draw = () => {
          grid.replaceChildren(...found.markers.map(item => {
            const input = node('input', {type:'checkbox', 'aria-label':label + ' marker ' + item.name});
            input.checked = chosen.has(item.name) && usable(item); input.disabled = !usable(item);
            input.addEventListener('change', () => { if (input.checked) chosen.add(item.name); else chosen.delete(item.name); save(); });
            return node('label', {title:usable(item) ? item.name + (source === 'corrected' ? suffix : '') : 'No ' + (source === 'corrected' ? 'corrected' : 'raw') + ' values for ' + item.name},
              [input, node('span', {text:item.name})]);
          }));
          save();
        };
        values_.addEventListener('change', () => { source = values_.value; draw(); });
        sets.addEventListener('change', () => {
          const picked = found.sets[Number(sets.value)]; sets.value = '';
          if (!picked) return;
          const known = new Set(found.markers.filter(usable).map(item => item.name));
          chosen.clear(); picked.markers.filter(name => known.has(name)).forEach(name => chosen.add(name));
          const missing = picked.markers.filter(name => !known.has(name));
          note.textContent = 'Set “' + picked.name + '”: ' + chosen.size + ' markers' + (missing.length ? '; not in the table: ' + missing.slice(0, 8).join(', ') + (missing.length > 8 ? '…' : '') : '') + '.';
          draw();
        });
        const all = node('button', {class:'btn small ghost', text:'All', onclick:() => { found.markers.filter(usable).forEach(item => chosen.add(item.name)); draw(); }});
        const none = node('button', {class:'btn small ghost', text:'None', onclick:() => { chosen.clear(); draw(); }});
        holder.replaceChildren(node('div', {class:'row wrap'}, [values_, sets]), node('div', {class:'row'}, [count, node('span', {class:'grow'}), all, none]), grid, note);
        draw();
      })();
      return holder;
    };

    // Tab 7's transformation and normalisation before clustering, in its order, shared with classic Tab 7.
    const NORMALISATION = [
      ['norm_sinh', 'Corrected values back to the measured scale', 'Batch-corrected columns (_corr) hold arcsinh values; sinh × cofactor turns them back first, as classic does.'],
      ['norm_z1', 'Z-score each marker', 'Each marker centred and scaled, within each ROI or slide when chosen below: brightness differences between markers stop dominating.'],
      ['norm_z2', 'Z-score each cell', 'Each cell centred and scaled across its markers: cells are compared by their pattern of markers, not their overall brightness.'],
      ['norm_cdf', 'Normal CDF (0 to 1)', 'Squeezes the z-scores into 0 to 1, taming extreme values.'],
      ['norm_nlog', '−log(1 − CDF)', 'Spreads the high end of the CDF; calculated as −log(1 − CDF). Requires CDF.'],
    ];
    const normalisationBlock = ({method, values, changed}) => {
      const box = node('details', {class:'studio-options'});
      const summary = node('summary');
      const within = node('select', {'aria-label':'Normalise within'});
      const describe = () => {
        const steps = NORMALISATION.filter(([key]) => values[key]).map(([key, label]) => key === 'norm_z1' && values.batch_id_col ?
          label + ' within each ' + (values.batch_id_col === 'batch_id' ? 'ROI' : values.batch_id_col) : label);
        summary.textContent = 'Transformation: ' + (steps.length ? steps.join(' → ') : 'none');
      };
      const ticks = NORMALISATION.map(([key, label, tip]) => {
        const input = node('input', {type:'checkbox', autocomplete:'off', 'data-setting':key, 'aria-label':label}); input.checked = !!values[key];
        input.addEventListener('change', () => {
          values[key] = input.checked;
          if (key === 'norm_nlog' && input.checked && !values.norm_cdf) { values.norm_cdf = true; ticks.find(item => item.key === 'norm_cdf').input.checked = true; }
          if (key === 'norm_cdf' && !input.checked && values.norm_nlog) { values.norm_nlog = false; ticks.find(item => item.key === 'norm_nlog').input.checked = false; }
          describe(); changed();
        });
        return {key, input, row:node('label', {class:'row'}, [input, node('span', {class:'grow', text:label}), help(tip)])};
      });
      within.addEventListener('change', () => { values.batch_id_col = within.value; describe(); changed(); });
      within.dataset.setting = 'batch_id_col';
      // Read the visible controls too: browser form restoration need not dispatch change.
      box.readSettings = () => {
        ticks.forEach(({key,input}) => { values[key] = input.checked; });
        if (within.options.length) values.batch_id_col = within.value;
        describe(); changed();
      };
      box.addEventListener('input', box.readSettings);
      box.addEventListener('toggle', box.readSettings);
      box.append(summary, ...ticks.map(item => item.row), field('Normalise within', within, 'Where marker z-scores are calculated: across the whole table, or separately for each value of the chosen column. This does not run separate clustering. After batch correction, use The whole table unless you deliberately want another normalization within slides or ROIs. Set the ROI and Slide columns under Column roles above.'));
      describe();
      const loadGroups = async () => {
        let groups = [], roles = {};
        try { const found = await get(route('/marker_choices') + '?method=' + encodeURIComponent(method)); groups = found.groups || []; roles = found.roles || {}; } catch (_) { /* whole table only */ }
        const options = [['', 'The whole table'], ...groups.map(name => [name,
          name === roles.roi ? 'Each ROI (' + name + ')' : name === roles.slide ? 'Each slide (' + name + ')' : 'Each value of ' + name])];
        if (values.batch_id_col && !options.some(([key]) => key === values.batch_id_col)) options.push([values.batch_id_col, values.batch_id_col]);
        within.replaceChildren(...options.map(([key, text]) => node('option', {value:key, text})));
        within.value = values.batch_id_col || '';
      };
      const rolesChanged = () => { if (box.isConnected) loadGroups(); else window.removeEventListener('studio-column-roles-changed', rolesChanged); };
      window.addEventListener('studio-column-roles-changed', rolesChanged);
      loadGroups();
      return box;
    };

    const overwriteOption = (outputs, tip) => {
      const input = node('input', {type:'checkbox', autocomplete:'off', 'aria-label':'Overwrite existing results'});
      outputs.forEach(output => ['input','change'].forEach(event => output.addEventListener(event, () => { input.checked = false; })));
      return {input, row:node('label', {class:'row'}, [input, node('span', {class:'grow',text:'Overwrite existing results'}),
        help(tip || 'Replace these output columns when the run succeeds. Leave unticked to keep earlier results and choose a different Save as name.')])};
    };

    const readPhenotypingSettings = (settings, values) => {
      settings.querySelectorAll('[data-setting]').forEach(input => {
        if (input.tagName === 'SELECT' && !input.options.length) return;
        if (!input.checkValidity()) throw Error('Check ' + (input.getAttribute('aria-label') || input.dataset.setting) + '.');
        values[input.dataset.setting] = input.type === 'checkbox' ? input.checked :
          ['number','range'].includes(input.type) ? Number(input.value) : input.value.trim();
      });
      settings.querySelectorAll('details').forEach(box => box.readSettings?.());
    };
    const lockPhenotypingSettings = (settings, buttons) => {
      const controls = [...settings.querySelectorAll('input,select,button'), ...buttons];
      const disabled = controls.map(input => input.disabled);
      controls.forEach(input => { input.disabled = true; });
      return () => controls.forEach((input,i) => { input.disabled = disabled[i]; });
    };

    // The cluster explorer, shared by every clustering method: a row per group with its top markers, a name and a
    // colour; clicking a group (in the list or the heatmap) shows its cells on the image, can turn its top markers
    // on, and lists the ROIs it is in (a click opens one, the group still shown). Names and colours are saved
    // together: the names in a new column, the colours where Tab 9 keeps them.
    const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080',
      '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffe119'];
    const colourKey = name => String(name).replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'item';
    const groupList = ({results, holder, state, kind, noun, running, none}) => {
      state.names = state.names || {}; state.colours = state.colours || {};
      let savedColours = null, presetNames = [];
      const look = async () => {
        let jobs;
        try { jobs = (await get(route('/jobs'))).jobs || []; } catch (_) { return false; }
        try { const palette = await get(route('/colours')); savedColours = palette.colours; presetNames = palette.phenotypes || []; } catch (_) { savedColours ||= {}; }
        const going = jobs.find(job => job.kind === kind && ['starting', 'running'].includes(job.status));
        const table = frame?.tablePath || (await post(route('/phenotyping_link'), {})).table?.path;
        const last = jobs.find(job => job.kind === kind && job.status === 'done' && job.result?.clusters && (!table || job.result.table === table));
        if (!holder.isConnected) return true;
        if (!last) { results.body.replaceChildren(node('p', {class:'muted', text:going ? running + ': ' + (going.message || '') : none})); return !going; }
        const result = last.result, inputs = {}, swatches = {};
        if (!Object.prototype.hasOwnProperty.call(state.names, last.id)) {
          try { state.names[last.id] = (await get(route('/cluster_names') + '?job_id=' + encodeURIComponent(last.id))).names || {}; }
          catch (_) { state.names[last.id] = {}; }
        }
        const names = state.names[last.id] = state.names[last.id] || {};
        state.phenotypeColourEdits ||= {};
        const colours = state.phenotypeColourEdits[last.id] ||= {};
        const sharedColour = name => {
          const key = colourKey(name).toLowerCase();
          return colours[key] || savedColours[Object.keys(savedColours).find(item => item.toLowerCase() === key)];
        };
        const refreshPalette = async () => {
          // Unnamed CN/cluster IDs use the same durable palette as cell layers and maps.
          const values = Array.from(new Set([...result.clusters.map(item => item.name), ...Object.values(names).filter(Boolean)]));
          const palette = values.length ? await post(route('/colours'), {names:values}) : await get(route('/colours'));
          savedColours = palette.colours; presetNames = palette.phenotypes || [];
        };
        await refreshPalette();
        const note = node('p', {class:'muted', role:'status'}), detail = node('div', {class:'studio-group-detail'});
        const known = () => Array.from(new Set([...Object.values(names).filter(Boolean), ...presetNames]));
        const colourOf = group => sharedColour(names[group] || group) || '#888888';
        let picked = state.picked?.[last.id] || [];
        // Suggestions from the reference cells in each cluster and the reference model's marker vote (not for
        // neighbourhoods, which are made of phenotypes).
        const suggests = kind !== 'neighbourhoods';
        let hints = null, heat = null;
        const hintBoxes = {};
        const summaryBoxes = {};
        const drawSummaries = mode => {
          if (kind !== 'neighbourhoods' || !result.profile?.raw) return;
          const profile = result.profile, share = mode === 'share';
          result.clusters.forEach(item => {
            const row = profile.rows.indexOf(item.name), box = summaryBoxes[item.name];
            if (row < 0 || !box) return;
            const matrix = share ? profile.raw : profile.values;
            const top = profile.columns.map((name,i) => ({name,value:matrix[row][i]}))
              .filter(item => Number.isFinite(item.value) && item.value > 0)
              .sort((a,b) => b.value-a.value || a.name.localeCompare(b.name)).slice(0,3);
            box.textContent = top.map(item => item.name + ' ' + (share ? (100*item.value).toFixed(1)+'%' : '+'+item.value.toFixed(2))).join(', ') || 'No above-average share';
            box.title = share ? 'Largest mean phenotype shares in the local windows. These are not the proportions of centre cells assigned to this CN.' :
              'Highest positive z-scores across neighbourhood types. A high score does not necessarily mean a large absolute share.';
            box.dataset.summary = share ? 'share' : 'relative';
          });
        };
        const clusterHints = group => hints?.clusters?.[group] || null;
        const askClusters = async groups => {
          if (chatState.busy) { note.textContent = 'Wait for the Assistant’s current reply.'; return; }
          await onPick(groups, true);
          await hooks.imageSettled?.(5000);
          await send('Suggest phenotype names for ' + (groups.length ? groups.join(', ') : 'all clusters') +
            ' from the ' + kind + ' run ' + last.id + '. Compare their full marker profiles with the other clusters, reference labels and ROI distribution. ' +
            'Use the attached cluster heatmap and current view as supporting context, explain uncertainty, and fill the cluster name boxes for review. The numerical profiles are supplied in RESULTS; use cluster_profiles or sample_cells if more evidence is needed. Return a name_clusters action. Do not infer marker positivity from display colour alone.', null, 'suggest',
            {job_id:last.id,clusters:groups,image:heat?.capture?.()});
        };
        const rows = result.clusters.map((item, index) => {
          const suggestion = (item.top_markers || []).slice(0, 2).map(marker => marker + '+').join(' ');
          const list = node('datalist', {id:'names-' + kind + '-' + index});
          const name = node('input', {type:'text', maxlength:'60', placeholder:item.name, value:names[item.name] || '', list:list.id, 'aria-label':noun + ' of ' + item.name});
          name.addEventListener('focus', () => list.replaceChildren(...Array.from(new Set([...(clusterHints(item.name)?.suggestions || []).map(s => s.label),
            suggestion, ...known(), ...(hints?.labels || []).map(entry => entry.label)])).filter(Boolean).map(value => node('option', {value}))));
          const swatch = node('input', {type:'color', value:colourOf(item.name, index), 'aria-label':'Colour of ' + item.name});
          name.addEventListener('change', async () => {
            names[item.name] = name.value.trim();
            drawHeatmap();
            try { await refreshPalette(); drawHeatmap(); }
            catch (error) { note.textContent = error.message; }
          });
          name.addEventListener('keydown', event => {
            if (event.key !== 'Enter') return;
            event.preventDefault(); name.dispatchEvent(new Event('change'));
            const next = Object.values(inputs)[index + 1]; if (next) next.focus(); else save.focus();
          });
          swatch.title = 'One colour per name, shared with the viewer, maps and figures when you save names.';
          swatch.addEventListener('input', () => {
            const value = names[item.name] || item.name;
            colours[colourKey(value).toLowerCase()] = swatch.value; drawHeatmap();
          });
          inputs[item.name] = name; swatches[item.name] = swatch;
          const summary = node('span', {class:'grow', text:(item.top_markers || []).join(', ') || '—', title:(item.top_markers || []).join(', ')});
          summaryBoxes[item.name] = summary;
          const line = node('div', {class:'studio-class studio-pickable', role:'button', tabindex:'0', 'data-group':item.name, title:'Show ' + item.name + ' on the image'}, [
            node('span', {class:'mono', text:item.name}),
            summary,
            // Sub-clusters carry classic's check: "review" when too small, confined to few batches, or not separated.
            item.check && item.check !== 'keep' ? chip(item.check, 'warn', item.reason || '') : null,
            node('span', {class:'muted', text:item.cells.toLocaleString()})].filter(Boolean));
          line.addEventListener('click', event => {
            const several = event.shiftKey || event.ctrlKey || event.metaKey;
            onPick(several ? (picked.includes(item.name) ? picked.filter(group => group !== item.name) : [...picked, item.name])
              : (picked.length === 1 && picked[0] === item.name ? [] : [item.name]), true);
          });
          line.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); line.click(); } });
          const hint = node('div', {class:'studio-hints'}); hintBoxes[item.name] = hint;
          return node('div', {class:'studio-class-row'}, [line, node('div', {class:'studio-class-name'}, [name, list, swatch,
            node('button', {class:'btn small',text:'Ask AI','aria-label':'Ask AI about ' + item.name,title:'Review this cluster using all marker profiles, reference labels and the current view',onclick:() => askClusters([item.name])})]), hint]);
        });
        const show = node('button', {class:'btn small', text:'Show on the image', onclick:async () => {
          try { showResult(await runCommand('show_job_result', {job_id:last.id})); } catch (error) { showResult({message:error.message}); }
        }});
        const save = node('button', {class:'btn primary', text:'Save names', onclick:async () => {
          Object.entries(inputs).forEach(([group, input]) => { names[group] = input.value.trim(); });
          const chosen = Object.fromEntries(Object.entries(names).filter(([, value]) => value));
          save.disabled = true;
          try {
            // Only intentional edits replace the shared palette; cluster order never chooses a phenotype colour.
            const byName = {};
            result.clusters.map(item => chosen[item.name] || item.name).forEach(value => {
              const edited = colours[colourKey(value).toLowerCase()];
              if (edited) byName[value] = edited;
            });
            if (Object.keys(byName).length) {
              savedColours = (await post(route('/colours'), {colours:byName})).colours;
              Object.keys(colours).forEach(key => delete colours[key]);
              document.dispatchEvent(new CustomEvent('spacemap-phenotype-presets'));
            }
            await refreshPalette();
            const answer = await runCommand('name_clusters', {job_id:last.id, ...(Object.keys(chosen).length ? {names:chosen} : {}), replace:true}, {confirmed:true}); showResult(answer);
            if (answer.status !== 'done' || !answer.data?.job) throw Error(answer.message || 'The names did not save.');
            note.textContent = answer.message; await refreshJobs();
            const job = await waitForJob(answer.data.job.id, message => { note.textContent = message; }, results.body);
            if (job.status !== 'done') throw Error(job.message || 'The names did not save.');
            window.dispatchEvent(new CustomEvent('studio-cluster-names', {detail:{job_id:last.id,names:job.result.names || chosen,replace:true,phase:'saved',column:job.result.column}}));
            await refreshJobs(); await hooks.refreshPhenotypingLinks?.(); await colourCells(job.result.column);
            if (frame?.colour) { await loadColours(frame); frame.colour.value = job.result.column; }
          }
          catch (error) { note.textContent = error.message; }
          finally { save.disabled = false; }
        }});
        const referencePick = node('select', {'aria-label':'Reference column for suggestions'});
        const hintNote = node('span', {class:'muted grow'});
        results.body.replaceChildren(...[node('p', {class:'muted', text:result.clusters.length + ' groups in ' + result.column + ' · ' + result.cells.toLocaleString() +
            ' cells · click a group to see it (Shift or Ctrl for several)'}),
          result.profile_notice ? node('p', {class:'muted',text:result.profile_notice}) : null,
          result.summary ? node('p', {class:'muted', text:result.summary + '.'}) : null,
          node('div', {class:'row'}, [show, node('button',{class:'btn small',text:'Ask AI',title:'Review selected clusters, or all clusters when none are selected',onclick:() => askClusters(picked)})]),
          suggests ? node('div', {class:'row wrap'}, [node('label', {class:'lbl', text:'Suggest from'}), referencePick, help('Your reference cells (a column you labelled in ' +
            'Manual reference) name the clusters two ways: which clusters hold them (the top three per cluster, as a share of each label’s ' +
            'reference cells), and the reference model’s vote on each cluster’s markers, as when labelling cells. The more reference cells a ' +
            'cluster holds, the more they count.'), hintNote]) : null,
          node('div', {class:'studio-classes'}, rows), detail, node('div', {class:'row'}, [save]), note].filter(Boolean));
        // The picked group: its cells on the image, its top markers, and where it is.
        const openRoi = async (roi, shown) => {
          let list = [];
          try { list = await loadImages(); } catch (_) { /* said below */ }
          const image = list.find(item => item.sample === roi) || list.find(item => item.name === roi || item.name.startsWith(roi + '.'));
          if (!image) { note.textContent = 'No image of ' + roi + ' in this project.'; return; }
          note.textContent = 'Opening ' + roi + '…';
          await chooseImage(image);
          for (let i = 0; i < 150; i++) {
            if (await showCellGroups(result.column, shown, result.profile?.rows || result.clusters.map(item => item.name))) { note.textContent = 'Showing ' + shown.join(', ') + ' in ' + roi + '.'; return; }
            await new Promise(resolve => setTimeout(resolve, 200));
          }
          note.textContent = roi + ' has no cells in this table.';
        };
        // A group's top markers as the open image names them: CD45_corr (a corrected column) is the image's CD45.
        const onImage = markers => {
          const channels = imageMarkers(), lower = new Map(channels.map(name => [name.toLowerCase(), name]));
          return Array.from(new Set(markers.map(name => {
            const plain = String(name).replace(/_norm$/, '');
            return lower.get(plain.toLowerCase()) || lower.get(plain.replace(/_[A-Za-z0-9]+$/, '').toLowerCase());
          }).filter(Boolean)));
        };
        const showTop = async group => {
          const item = result.clusters.find(entry => entry.name === group);
          const markers = onImage(item?.top_markers || []);
          if (!markers.length) return false;
          try { await runCommand('show_markers', {markers}); return markers; } catch (_) { return false; }
        };
        const drawDetail = () => {
          if (picked.length !== 1) { detail.replaceChildren(); return; }
          const group = picked[0], item = result.clusters.find(entry => entry.name === group) || {top_markers:[]};
          const places = (result.where || {})[group] || [];
          const top = kind === 'neighbourhoods' ? null : node('button', {class:'btn small', text:'Top markers on the image', title:'Show ' + item.top_markers.join(', ') + ' on the image', onclick:async () => {
            const shown = await showTop(group);
            note.textContent = shown ? 'Showing ' + shown.join(', ') + ' on the image.' : 'None of ' + item.top_markers.join(', ') + ' is a marker of the open image.';
          }});
          detail.replaceChildren(node('div', {class:'studio-import'}, [
            node('div', {class:'row'}, [node('b', {text:(names[group] ? names[group] + ' (' + group + ')' : group)}), node('span', {class:'muted grow', text:item.top_markers.join(', ')}), top]),
            places.length ? node('p', {class:'lbl', text:'Where: ' + places.length + ' ROI' + (places.length === 1 ? '' : 's') + ' (most cells first; click to open)'}) : null,
            places.length ? node('div', {class:'studio-where'}, places.slice(0, 24).map(place => node('button', {type:'button', class:'studio-measure', onclick:() => openRoi(place.roi, [group])}, [
              node('span', {class:'grow', text:place.roi}), node('span', {class:'mono', text:place.cells.toLocaleString()})]))) : null]));
        };
        // The naming bar above the heatmap: the picked clusters named together, from the suggestions or typed.
        const namer = node('div', {class:'studio-namer'});
        const nameGroups = async (groups, value) => {
          value = String(value || '').trim();
          if (!value) return;
          groups.forEach(group => {
            names[group] = value;
            if (inputs[group]) inputs[group].value = value;
          });
          try { await refreshPalette(); }
          catch (error) { note.textContent = error.message; return; }
          note.textContent = 'Named ' + groups.length + ' cluster' + (groups.length === 1 ? '' : 's') + ' “' + value + '”. Save names to keep them.';
          drawHeatmap(); drawNamer();
        };
        const combined = groups => {
          // The picked clusters' suggestions together: reference cells added up, marker votes weighted by size.
          const sizes = Object.fromEntries(result.clusters.map(item => [item.name, item.cells]));
          const byLabel = {};
          groups.forEach(group => (clusterHints(group)?.suggestions || []).forEach(entry => {
            const one = byLabel[entry.label] || (byLabel[entry.label] = {label:entry.label, score:0, weight:0, reference:0, of:0});
            one.score += entry.score * sizes[group]; one.weight += sizes[group]; one.reference += entry.reference_cells; one.of += entry.of_label;
          }));
          return Object.values(byLabel).map(entry => ({...entry, score:entry.score / Math.max(entry.weight, 1)})).sort((a, b) => b.score - a.score).slice(0, 4);
        };
        const drawNamer = () => {
          if (!picked.length) {
            namer.replaceChildren(node('p', {class:'muted', text:'Pick clusters here or in the list (Shift or Ctrl for several), then name them together.'}),
              node('button',{class:'btn small',text:'Ask AI',onclick:() => askClusters([])}));
            return;
          }
          const input = node('input', {type:'text', maxlength:'60', placeholder:'Name for ' + picked.length + ' cluster' + (picked.length === 1 ? '' : 's'), 'aria-label':'Name the picked clusters'});
          const options = node('datalist', {id:'namer-' + kind});
          options.append(...Array.from(new Set([...combined(picked).map(entry => entry.label), ...known(), ...(hints?.labels || []).map(entry => entry.label)])).map(value => node('option', {value})));
          input.setAttribute('list', options.id);
          const apply = node('button', {class:'btn small primary', text:'Name ' + picked.length, onclick:() => nameGroups(picked, input.value)});
          input.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); apply.click(); } });
          const chips = combined(picked).map(entry => node('button', {type:'button', class:'btn small ghost studio-suggestion',
            title:'Score ' + entry.score.toFixed(2) + ' · ' + entry.reference + ' reference cells in ' + (picked.length === 1 ? 'this cluster' : 'these clusters') +
              ' (' + Math.round(100 * entry.of) + '% of all ' + entry.label + ' references)', text:entry.label + ' · ' + Math.round(100 * entry.score) + '%',
            onclick:() => nameGroups(picked, entry.label)}));
          namer.replaceChildren(node('div', {class:'row wrap'}, [node('b', {text:picked.length + ' picked'}), input, options, apply]),
            chips.length ? node('div', {class:'row wrap'}, [node('span', {class:'muted', text:'Suggested:'}), ...chips]) : null);
          namer.append(node('button',{class:'btn small',text:'Ask AI',onclick:() => askClusters(picked)}));
        };
        const onPick = async (chosen, fromList) => {
          picked = chosen; state.picked = state.picked || {}; state.picked[last.id] = chosen;
          results.body.querySelectorAll('[data-group]').forEach(line => line.classList.toggle('picked', chosen.includes(line.dataset.group)));
          drawDetail(); drawNamer();
          if (fromList) heat?.update({picked:chosen});
          if (!(await showCellGroups(result.column, chosen, result.profile?.rows || result.clusters.map(item => item.name)))) {
            note.textContent = 'Open an image whose cells are in this table to see them.'; return;
          }
          note.textContent = chosen.length ? 'Showing ' + chosen.map(row => names[row] || row).join(', ') + ' on the image.' : 'Showing every group.';
          // One group picked: its top markers are turned on too (neighbourhoods are made of phenotypes, not markers).
          if (chosen.length === 1 && kind !== 'neighbourhoods') {
            const shown = await showTop(chosen[0]);
            if (shown) note.textContent += ' Markers: ' + shown.join(', ') + '.';
          }
        };
        const looks = () => ({
          labels:Object.fromEntries((result.profile?.rows || []).map(row => [row, names[row] ? names[row] + ' (' + row.replace(/^(Cluster|CN) /, '') + ')' : row])),
          colours:Object.fromEntries((result.profile?.rows || []).map((row, index) => {
            const at = result.clusters.findIndex(item => item.name === row);
            return [row, swatches[row]?.value || colourOf(row, at >= 0 ? at : index)];
          })),
          // Right of the colours: the reference label most of whose reference cells are in the cluster.
          notes:Object.fromEntries((result.profile?.rows || []).map(row => {
            const top = clusterHints(row)?.top?.[0];
            return [row, top ? top.label + ' ' + Math.round(100 * top.of_label) + '%' : ''];
          }).filter(([, text]) => text)),
        });
        const drawHeatmap = () => {
          result.clusters.forEach((item, index) => { if (swatches[item.name]) swatches[item.name].value = colourOf(item.name, index); });
          if (!result.profile) { figures.querySelector('[data-figure="'+kind+'"]')?.remove(); return; }
          if (heat && heat.box.isConnected) { heat.update(looks()); return; }
          heat = heatmapFigure({key:kind, title:{leiden:'Leiden clusters', neighbourhoods:'Neighbourhoods', som:'SOM clusters', vortex:'VORTEX clusters', subphenotype:'Sub-phenotypes'}[kind] || 'Clusters',
            profile:result.profile, onPick:chosen => onPick(chosen, false), picked, tools:namer, ...looks(),
            onDisplay:kind === 'neighbourhoods' ? drawSummaries : null,
            onAggregation:result.profile.summaries ? async aggregation => {
              await post(route('/heatmap_summary'), {method:kind,aggregation});
              await look();
            } : null,
            tip:'Each row is a group, each column a ' + (kind === 'neighbourhoods' ? 'phenotype. Share (%) shows composition; Relative (z-score) compares each phenotype across neighbourhood types. The three phenotypes on the left follow this choice. ' : 'marker; warm means higher than in the other groups, blue lower. ') +
              'Click a row to show its cells on the image (Shift or Ctrl for several), then name the picked rows together above. ' +
              (suggests ? 'Right of the colours: the reference label with the largest share of its reference cells in that cluster. ' : '') +
              'Large view shows it over the image; Show all brings every group back.'});
        };
        // Suggestions: loaded for the chosen reference column; each cluster's row gets its three best names.
        const loadHints = async () => {
          if (!suggests) return;
          hintNote.textContent = 'Reading the reference cells…';
          let found;
          try { found = await get(route('/cluster_suggestions') + '?job_id=' + last.id + '&reference=' + encodeURIComponent(referencePick.value || '')); }
          catch (error) { hints = null; hintNote.textContent = error.message; Object.values(hintBoxes).forEach(box => box.replaceChildren()); drawHeatmap(); drawNamer(); return; }
          if (!results.body.isConnected) return;
          hints = found;
          hintNote.textContent = found.labelled.toLocaleString() + ' reference cells, ' + found.labels.length + ' labels' + (found.by_markers ? '' : ' (marker vote off: ' + found.note + ')');
          result.clusters.forEach(item => {
            const mine = clusterHints(item.name), box = hintBoxes[item.name];
            if (!box) return;
            box.replaceChildren(...[
              ...(mine?.suggestions || []).filter(entry => entry.score > 0.05).map(entry => node('button', {type:'button', class:'btn small ghost studio-suggestion',
                title:(entry.by_markers !== null ? 'Markers vote ' + Math.round(100 * entry.by_markers) + '% · ' : '') + entry.reference_cells + ' reference cells here (' +
                  Math.round(100 * entry.of_label) + '% of all ' + entry.label + ' references)', text:entry.label + ' · ' + Math.round(100 * entry.score) + '%',
                onclick:() => nameGroups([item.name], entry.label)})),
              mine?.top?.length ? node('span', {class:'muted', text:'Reference cells: ' + mine.top.map(entry => entry.label + ' ' + entry.cells + ' (' + Math.round(100 * entry.of_label) + '%)').join(', ')}) : null
            ].filter(Boolean));
          });
          drawHeatmap(); drawNamer();
        };
        if (suggests) {
          referencePick.addEventListener('change', () => { remember('reference-column', referencePick.value); loadHints(); });
          (async () => {
            let columns = [];
            try { columns = (await get(route('/cluster_suggestions'))).reference_columns || []; } catch (_) { /* said by loadHints */ }
            referencePick.replaceChildren(...(columns.length ? columns : ['']).map(column => node('option', {value:column, text:column || 'No reference column yet'})));
            referencePick.disabled = !columns.length;
            const saved = recall('reference-column', '');
            if (columns.includes(saved)) referencePick.value = saved;
            if (columns.length) loadHints(); else hintNote.textContent = 'Label some cells in Manual reference to get suggestions.';
          })();
        }
        drawHeatmap(); drawDetail(); drawNamer();
        // For the assistant: this run, the picks and the names typed (saved or not).
        if(frame?.id==='tab7'){
          frame.publicationRun=()=>results.body.isConnected ? {job_id:last.id,method:kind,names:{...names},cells:result.cells} : null;
          frame.publicationUpdate?.();
        }
        explorerNow = () => results.body.isConnected ? {job_id:last.id, method:kind, clusters_column:result.column, picked,
          reference_column:referencePick.value, table:result.table,
          typed:Object.fromEntries(Object.entries(inputs).map(([group, input]) => [group, input.value.trim()]).filter(([, value]) => value))} : null;
        if (state.cellPickListener) document.removeEventListener('spacemap-cells-picked', state.cellPickListener);
        let selectionSequence = 0;
        state.cellPickListener = async event => {
          if (!results.body.isConnected) { document.removeEventListener('spacemap-cells-picked',state.cellPickListener); return; }
          const {layer,labels} = event.detail || {};
          if (!layer?.combinedTable) return;
          const sequence = ++selectionSequence;
          try {
            const found = await post(route('/data/groups'), {source:{cells:layer.token,table:layer.combinedTable},rows:{this_image:true},column:result.column,labels:labels || []});
            if (sequence !== selectionSequence || !results.body.isConnected) return;
            picked = found.groups.filter(group => result.clusters.some(item => item.name === group));
            state.picked = state.picked || {}; state.picked[last.id] = picked;
            results.body.querySelectorAll('[data-group]').forEach(line => line.classList.toggle('picked',picked.includes(line.dataset.group)));
            heat?.update({picked}); drawDetail(); drawNamer();
          } catch (error) { note.textContent = error.message; }
        };
        document.addEventListener('spacemap-cells-picked', state.cellPickListener);
        // Assistant suggestions fill the fields immediately; job receipts mark them saved.
        if (state.namesListener) window.removeEventListener('studio-cluster-names', state.namesListener);
        state.namesListener = async event => {
          if (event.detail.job_id !== last.id || !results.body.isConnected) return;
          const value = event.detail;
          if (value.replace) Object.keys(names).forEach(group => delete names[group]);
          (value.clear || []).forEach(group => delete names[group]);
          Object.entries(value.names || {}).forEach(([group, name]) => { if (inputs[group]) names[group] = name; });
          Object.entries(inputs).forEach(([group, input]) => { input.value = names[group] || ''; });
          note.textContent = value.phase === 'saved' ? 'Saved in ' + value.column + '.' : value.phase === 'failed' ? 'Not saved: ' + value.message :
            value.phase === 'saving' ? 'Names filled. Saving…' : 'Suggested names filled. Edit them, then Save names or ask the assistant to apply them.';
          try { await refreshPalette(); } catch (error) { note.textContent = error.message; }
          drawHeatmap(); drawDetail(); drawNamer();
        };
        window.addEventListener('studio-cluster-names', state.namesListener);
        if (state.paletteListener) document.removeEventListener('spacemap-phenotype-presets', state.paletteListener);
        state.paletteListener = async () => {
          if (!results.body.isConnected) return;
          Object.keys(colours).forEach(key => delete colours[key]);
          try { await refreshPalette(); drawHeatmap(); }
          catch (error) { note.textContent = error.message; }
        };
        document.addEventListener('spacemap-phenotype-presets', state.paletteListener);
        return !going && !result.profile_pending;
      };
      let timer = null;
      const follow = async () => { clearTimeout(timer); if (!(await look()) && holder.isConnected) timer = setTimeout(follow, 4000); };
      return follow;
    };

    // Leiden (Tab 7): its saved settings, a job on the combined table, then a phenotype for every cluster.
    const LEIDEN_KEYS = ['markers', 'resolution', 'n_neighbors', 'subsample_size', 'use_gpu', 'enable_low_expr_filter', 'low_expr_threshold', 'output_col',
      'norm_sinh', 'norm_z1', 'norm_z2', 'norm_cdf', 'norm_nlog', 'batch_id_col'];
    PANELS.leiden = async (current, holder) => {
      const state = current.leiden || (current.leiden = {names:{}});
      const settings = section('Leiden settings', {tip:'Tab 7’s Leiden settings, shared with classic Tab 7. Only the main ones ' +
        'are here; the others stay as Tab 7 has them.'});
      const results = section('Clusters', {tip:'The clusters of the last Leiden run, with their size and most raised markers. Show them ' +
        'on the image, type the phenotype of each, and Save names to record each cell’s phenotype. Saving names again updates this run’s phenotype column.'});
      const dirty = chip('Saved', 'ok');
      settings.box.querySelector('.studio-sec-h').insertBefore(dirty, settings.box.querySelector('.studio-sec-h .grow'));
      holder.append(settings.box, results.box);
      let saved;
      try { saved = await get(route('/leiden_settings')); }
      catch (error) { settings.body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.values) state.values = JSON.parse(JSON.stringify(saved.values));
      const values = state.values;
      const changes = () => Object.fromEntries(LEIDEN_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); };
      const markers = markerPicker({method:'leiden', label:'Leiden', values, changed:() => mark()});
      const resolution = node('input', {type:'range', min:'0.1', max:'5', step:'0.1', 'data-setting':'resolution', value:String(values.resolution), 'aria-label':'Resolution'});
      const shown = node('span', {class:'mono slider-value', text:Number(values.resolution).toFixed(1)});
      resolution.addEventListener('input', () => { values.resolution = Number(resolution.value); shown.textContent = values.resolution.toFixed(1); mark(); });
      const number = (key, label, step, min, max) => {
        const input = node('input', {type:'number', step, min, max, 'data-setting':key, value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= Number(min) && value <= Number(max)) { values[key] = value; mark(); } else input.value = String(values[key]); });
        return input;
      };
      const tick = (key, label) => {
        const input = node('input', {type:'checkbox', autocomplete:'off', 'data-setting':key, 'aria-label':label}); input.checked = !!values[key];
        input.addEventListener('change', () => { values[key] = input.checked; mark(); });
        return input;
      };
      const output = node('input', {type:'text', 'data-setting':'output_col', value:values.output_col, 'aria-label':'Leiden result column'});
      const overwrite = node('input', {type:'checkbox', 'aria-label':'Overwrite existing results'});
      output.addEventListener('change', () => { values.output_col = output.value.trim(); overwrite.checked = false; mark(); });
      settings.body.append(node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Markers'}),
          help('Choose measured or corrected markers. A saved marker set selects several at once.')]), markers]),
        field('Resolution', node('div', {class:'inline'}, [resolution, shown]), 'Higher makes more, smaller clusters.'),
        field('Neighbours', number('n_neighbors', 'Neighbours', '1', '2', '200'), 'Cells each cell is linked to in the graph (default 15).'),
        field('Cells used', number('subsample_size', 'Cells used', '10000', '1000', '10000000'), 'Cells used to build the graph; the rest are assigned to the nearest cluster (default 200,000).'),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'GPU'}), tick('use_gpu', 'Use the GPU'), node('span', {class:'grow'}), help('Uses the GPU when this machine has one.')]),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Dim cells'}), tick('enable_low_expr_filter', 'Keep dim cells apart'),
          number('low_expr_threshold', 'Dim cell cut', '0.05', '-10', '100000'), help('Cells whose mean marker z-score is at or below the cut are kept out of the clustering and called Low expression (default -0.75).')]),
        normalisationBlock({method:'leiden', values, changed:() => mark()}),
        field('Save as', output, 'The phenotype column for the names; the clusters go in the same name with _clusters.'),
        node('label', {class:'row'}, [overwrite, node('span', {class:'grow', text:'Overwrite existing results'}),
          help('Replace the Save as column and its _clusters column when this run succeeds. Old names reset to cluster numbers; name the new clusters afterward. Leave unticked to keep earlier results.') ]));
      mark();
      const status = node('p', {class:'muted', role:'status'});
      const save = async () => {
        readPhenotypingSettings(settings.body, values);
        saved = await post(route('/leiden_settings'), {changes:changes(), revision:saved.revision});
        Object.assign(values, saved.values); state.values = values; mark();
      };
      const saveButton = node('button', {class:'btn', text:'Save settings', onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try { await save(); status.textContent = 'Saved into Tab 7’s Leiden settings.'; } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      const runButton = node('button', {class:'btn primary', text:'Run Leiden', onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try {
          readPhenotypingSettings(settings.body, values);
          if (Object.keys(changes()).length) { await save(); status.textContent = 'Settings saved. '; } else status.textContent = '';
          const answer = await runCommand('run_leiden', {overwrite:overwrite.checked, settings_revision:saved.revision}, {confirmed:true});
          showResult(answer); status.textContent += answer.message;
          if (answer.status === 'done') { overwrite.checked = false; await refreshJobs(); showTab('jobs'); follow(); }
        } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      current.runBar([node('div', {class:'row'}, [saveButton, runButton]), status], 'Runs on the whole combined table in the background.');
      const follow = groupList({results, holder, state, kind:'leiden', noun:'Phenotype', running:'Leiden is running', none:'No Leiden run yet.'});
      follow();
    };
    // A Tab 7 clustering method with classic's settings file, run as a job, explored and named with the cluster explorer.
    const clusteringPanel = async (current, holder, {method, name, command, fields}) => {
      const state = current[method] || (current[method] = {names:{}});
      const settings = section(name + ' settings', {tip:'Tab 7’s ' + name + ' settings, shared with classic Tab 7. Only the main ones are here; ' +
        'the others stay as Tab 7 has them.'});
      const results = section('Clusters', {tip:'The groups of the last ' + name + ' run, with their size and most raised markers. Click one to ' +
        'see it on the image and where it is; type the phenotype of each, and Save names updates this run’s phenotype column.'});
      const dirty = chip('Saved', 'ok');
      settings.box.querySelector('.studio-sec-h').insertBefore(dirty, settings.box.querySelector('.studio-sec-h .grow'));
      holder.append(settings.box, results.box);
      let saved;
      try { saved = await get(route('/clustering/' + method)); }
      catch (error) { settings.body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.values) state.values = JSON.parse(JSON.stringify(saved.values));
      const overwrite = node('input', {type:'checkbox', autocomplete:'off', 'aria-label':'Overwrite existing results'});
      const values = state.values, keys = fields.flatMap(item => item.kind === 'normalisation' ? NORMALISATION.map(([key]) => key).concat('batch_id_col') : [item.key]);
      const changes = () => Object.fromEntries(keys.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); };
      fields.forEach(item => {
        let control;
        if (item.kind === 'markers') {
          settings.body.append(node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:item.label}), help(item.tip)]),
            markerPicker({method, label:name, values, changed:() => mark()})]));
          return;
        } else if (item.kind === 'normalisation') {
          settings.body.append(normalisationBlock({method, values, changed:() => mark()}));
          return;
        } else if (item.kind === 'choice') {
          control = choice(item.options, values[item.key], item.label); control.dataset.setting = item.key;
          control.addEventListener('change', () => { values[item.key] = control.value; mark(); });
        } else if (item.kind === 'bool') {
          control = node('input', {type:'checkbox', autocomplete:'off', 'data-setting':item.key, 'aria-label':item.label}); control.checked = !!values[item.key];
          control.addEventListener('change', () => { values[item.key] = control.checked; mark(); });
        } else if (item.kind === 'text') {
          control = node('input', {type:'text', 'data-setting':item.key, value:values[item.key], 'aria-label':item.label});
          control.addEventListener('change', () => { values[item.key] = control.value.trim(); if (item.key === 'output_col') overwrite.checked = false; mark(); });
        } else {
          control = node('input', {type:'number', 'data-setting':item.key, step:item.step, min:String(item.min), max:String(item.max), value:String(values[item.key]), 'aria-label':item.label});
          control.addEventListener('change', () => {
            const value = Number(control.value);
            if (Number.isFinite(value) && value >= item.min && value <= item.max) { values[item.key] = item.kind === 'int' ? Math.round(value) : value; mark(); }
            else control.value = String(values[item.key]);
          });
        }
        settings.body.append(field(item.label, control, item.tip));
      });
      settings.body.append(node('label', {class:'row'}, [overwrite, node('span', {class:'grow',text:'Overwrite existing results'}),
        help('Replace the Save as column and its _clusters column when this run succeeds. Old names reset to cluster numbers. Leave unticked to keep earlier runs.') ]));
      mark();
      const status = node('p', {class:'muted', role:'status'});
      const save = async () => {
        readPhenotypingSettings(settings.body, values);
        saved = await post(route('/clustering/' + method), {changes:changes(), revision:saved.revision});
        Object.assign(values, saved.values); state.values = values; mark();
      };
      const saveButton = node('button', {class:'btn', text:'Save settings', onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try { await save(); status.textContent = 'Saved into Tab 7’s ' + name + ' settings.'; } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      const runButton = node('button', {class:'btn primary', text:'Run ' + name, onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try {
          readPhenotypingSettings(settings.body, values);
          if (Object.keys(changes()).length) { await save(); status.textContent = 'Settings saved. '; } else status.textContent = '';
          const answer = await runCommand(command, {overwrite:overwrite.checked, settings_revision:saved.revision}, {confirmed:true});
          showResult(answer); status.textContent += answer.message;
          if (answer.status === 'done') { overwrite.checked = false; await refreshJobs(); showTab('jobs'); follow(); }
        } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      current.runBar([node('div', {class:'row'}, [saveButton, runButton]), status], 'Runs on the whole combined table in the background.');
      const follow = groupList({results, holder, state, kind:method, noun:'Phenotype', running:name + ' is running', none:'No ' + name + ' run yet.'});
      follow();
    };
    PANELS.som = (current, holder) => clusteringPanel(current, holder, {method:'som', name:'SOM', command:'run_som', fields:[
      {key:'markers', kind:'markers', label:'Markers', tip:'Choose measured or corrected markers. A saved marker set selects several at once.'},
      {key:'grid_dim', kind:'int', label:'Grid', step:'1', min:2, max:200, tip:'The SOM is N × N nodes of similar cells.'},
      {key:'n_meta_target', kind:'int', label:'Meta-clusters', step:'1', min:2, max:500, tip:'The nodes are merged into this many groups to name (default 45). Ask for more than the cell types you expect.'},
      {key:'mode', kind:'choice', label:'Quality', options:[['fast', 'Fast'], ['balanced', 'Balanced'], ['high_quality', 'High quality']], tip:'Longer training finds finer groups.'},
      {key:'subsample_size', kind:'int', label:'Cells used', step:'10000', min:1000, max:10000000, tip:'Cells the SOM is trained on (default 150,000); every cell is then placed on its nearest node.'},
      {key:'enable_low_expr_filter', kind:'bool', label:'Dim cells apart', tip:'Cells dim in every marker are kept out of the SOM and called Low expression.'},
      {key:'low_expr_threshold', kind:'float', label:'Dim cell cut', step:'0.05', min:-10, max:100000, tip:'The mean marker z-score at or below which a cell is dim (default -0.75).'},
      {key:'normalisation', kind:'normalisation'},
      {key:'output_col', kind:'text', label:'Save as', tip:'The phenotype column for the names; the groups go in the same name with _clusters.'}]});
    PANELS.vortex = (current, holder) => clusteringPanel(current, holder, {method:'vortex', name:'VORTEX', command:'run_vortex', fields:[
      {key:'markers', kind:'markers', label:'Markers', tip:'The markers the cells are clustered by: as measured, or batch corrected. A saved set ticks its markers in one step.'},
      {key:'auto_select_k', kind:'bool', label:'Choose K', tip:'Scan several K (neighbours per cell) on a sample and take the one giving about the target number of clusters.'},
      {key:'n_clusters_target', kind:'int', label:'Target clusters', step:'1', min:2, max:500, tip:'The number of clusters the K scan aims for (default 45).'},
      {key:'vortex_k', kind:'int', label:'K', step:'1', min:2, max:500, tip:'Neighbours per cell when Choose K is off (default 35): more neighbours, fewer and larger clusters.'},
      {key:'subsample_size', kind:'int', label:'Cells used', step:'10000', min:1000, max:10000000, tip:'Cells clustered directly (default 150,000); the rest join their nearest clustered neighbours.'},
      {key:'use_gpu', kind:'bool', label:'GPU', tip:'Uses the GPU when this machine has one.'},
      {key:'enable_low_expr_filter', kind:'bool', label:'Dim cells apart', tip:'Cells dim in every marker are kept out and called Low expression.'},
      {key:'low_expr_threshold', kind:'float', label:'Dim cell cut', step:'0.05', min:-10, max:100000, tip:'The mean marker z-score at or below which a cell is dim (default -0.75).'},
      {key:'normalisation', kind:'normalisation'},
      {key:'output_col', kind:'text', label:'Save as', tip:'The phenotype column for the names; the groups go in the same name with _clusters.'}]});
    // Consensus ML (Tab 7): the cells enough chosen methods agree on train a classifier that labels every cell.
    // Settings are classic's file; the four result columns wait beside the table (Jobs > Saved results).
    const CONSENSUS_KEYS = ['markers', 'phenotyping_methods', 'min_consensus_methods', 'exclude_phenotypes', 'ml_method', 'n_estimators',
      'cpu_workers', 'execution_backend', 'gpu_bins',
      'use_z_normalization', 'enable_filter', 'validation_ground_truth_col', 'output_phenotype_col', 'refined_confidence_threshold'];
    PANELS.consensus = async (current, holder) => {
      const state = current.consensus || (current.consensus = {});
      const settings = section('Consensus ML settings', {tip:'Cells where enough of the ' +
        'chosen methods agree train a classifier on their markers, which then labels every cell.'});
      const results = section('Result', {tip:'The last run: the share of cells the methods agreed on, the check against your reference cells ' +
        '(when chosen), and each phenotype’s markers. Click a phenotype to see its cells on the image.'});
      const dirty = chip('Saved', 'ok');
      settings.box.querySelector('.studio-sec-h').insertBefore(dirty, settings.box.querySelector('.studio-sec-h .grow'));
      holder.append(settings.box, results.box);
      let saved;
      try { saved = await get(route('/consensus_settings')); }
      catch (error) { settings.body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.values) state.values = JSON.parse(JSON.stringify(saved.values));
      const values = state.values;
      const changes = () => Object.fromEntries(CONSENSUS_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); };
      // The methods to combine: phenotype columns of automated methods (not reference labels, not earlier consensus results).
      const methodsBox = node('div', {class:'studio-markers'}, [node('p', {class:'muted', text:'Reading the phenotype columns…'})]);
      const agreement = node('input', {type:'number', min:'2', step:'1', value:String(values.min_consensus_methods), 'aria-label':'Agreement needed'});
      const agreementNote = node('span', {class:'muted'});
      const fitAgreement = () => {
        const n = values.phenotyping_methods.length;
        agreement.max = String(Math.max(2, n));
        if (n >= 2 && values.min_consensus_methods > n) { values.min_consensus_methods = n; agreement.value = String(n); }
        agreementNote.textContent = n ? 'of ' + n + ' methods' : '';
      };
      agreement.addEventListener('change', () => {
        const value = Math.round(Number(agreement.value));
        if (value >= 2 && value <= Math.max(2, values.phenotyping_methods.length)) values.min_consensus_methods = value; else agreement.value = String(values.min_consensus_methods);
        mark();
      });
      const check = node('select', {'aria-label':'Check against'});
      (async () => {
        let found = {phenotypes:[]};
        try { found = await get(route('/marker_choices') + '?method=consensus'); } catch (error) { methodsBox.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
        const reference = name => /(^|_)(ref\d*|reference|manual)(_|$)/i.test(name);
        const output = values.output_phenotype_col;
        const methods = (found.phenotypes || []).filter(name => !reference(name) && !/(_score|_consensus|_refined|_confidence|_clusters)$/.test(name) &&
          !(name === output || name.startsWith(output + '_')));
        methodsBox.replaceChildren(...(methods.length ? methods.map(name => {
          const input = node('input', {type:'checkbox', 'aria-label':'Combine ' + name}); input.checked = values.phenotyping_methods.includes(name);
          input.addEventListener('change', () => {
            values.phenotyping_methods = input.checked ? [...values.phenotyping_methods, name] : values.phenotyping_methods.filter(item => item !== name);
            fitAgreement(); mark();
          });
          return node('label', {}, [input, node('span', {text:name, title:name})]);
        }) : [node('p', {class:'muted', text:'Run at least two phenotyping methods first.'})]));
        const references = (found.phenotypes || []).filter(reference);
        check.replaceChildren(node('option', {value:'', text:'No reference: split the agreed cells'}), ...references.map(name => node('option', {value:name, text:name})));
        check.value = references.includes(values.validation_ground_truth_col) ? values.validation_ground_truth_col : '';
        fitAgreement();
      })();
      check.addEventListener('change', () => { values.validation_ground_truth_col = check.value; mark(); });
      const excluded = node('input', {type:'text', value:(values.exclude_phenotypes || []).join(', '), 'aria-label':'Cannot agree on'});
      excluded.addEventListener('change', () => { values.exclude_phenotypes = excluded.value.split(',').map(item => item.trim()).filter(Boolean); mark(); });
      const classifier = choice([['random_forest', 'Random forest'], ['gradient_boosting', 'Gradient boosting'], ['svm', 'Support vector machine']], values.ml_method, 'Classifier');
      classifier.addEventListener('change', () => { values.ml_method = classifier.value; mark(); });
      const number = (key, label, step, min, max) => {
        const input = node('input', {type:'number', 'data-setting':key, step, min:String(min), max:String(max), value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= min && value <= max) { values[key] = step === '1' ? Math.round(value) : value; mark(); } else input.value = String(values[key]); });
        return input;
      };
      const tick = (key, label) => { const input = node('input', {type:'checkbox', autocomplete:'off', 'data-setting':key, 'aria-label':label}); input.checked = !!values[key];
        input.addEventListener('change', () => { values[key] = input.checked; mark(); }); return input; };
      const output = node('input', {type:'text', 'data-setting':'output_phenotype_col', value:values.output_phenotype_col, 'aria-label':'Consensus result column'});
      const overwrite = overwriteOption([output]);
      output.addEventListener('change', () => { values.output_phenotype_col = output.value.trim(); mark(); });
      const compute = node('details', {class:'studio-options'});
      const computeSummary = node('summary');
      const backend = choice([['cpu','CPU · Classic compatible'], ['auto','GPU if available'], ['gpu','GPU only']], values.execution_backend || 'cpu', 'Consensus execution');
      backend.dataset.setting = 'execution_backend';
      const cores = number('cpu_workers', 'CPU cores', '1', 0, 4096);
      const gpuBins = field('GPU split bins', number('gpu_bins', 'GPU split bins', '1', 16, 4096),
        'Quantile bins used for GPU tree splits. More bins can improve split resolution but need more time and memory. CPU uses exact splits.');
      const resourceNote = node('p', {class:'muted', role:'status'});
      const gpuNote = node('p', {class:'muted', text:'GPU keeps all trees and training cells, but uses different splits. Compare validation scores before using its labels.'});
      const drawCompute = () => {
        const cpu = saved.hardware?.cpu, gpu = saved.hardware?.gpu;
        const requested = Number(cores.value) || 0;
        const count = requested ? Math.min(requested, cpu?.available || 4096) : (cpu?.auto_workers ?? cpu?.workers);
        const cpuLabel = 'CPU · ' + (count ? count + ' cores' : 'Auto');
        const fallsBack = values.execution_backend === 'auto' && gpu?.available === false;
        computeSummary.textContent = 'Compute · ' + (values.execution_backend === 'cpu' ? cpuLabel : fallsBack ? cpuLabel + ' · GPU unavailable' : values.execution_backend === 'gpu' ? (gpu?.available === false ? 'GPU unavailable' : 'GPU') : 'GPU if available');
        resourceNote.textContent = (cpu ? cpu.available + ' logical CPUs available · Auto: ' + (cpu.auto_workers ?? cpu.workers) + ' cores. ' : 'Auto checks allocated CPUs and available memory. ') +
          (gpu?.available ? 'GPU: ' + gpu.name + ' · cuML ' + gpu.version + '.' : (gpu?.reason || 'GPU availability is checked at launch.'));
        gpuBins.hidden = gpuNote.hidden = values.execution_backend === 'cpu';
        cores.disabled = values.ml_method !== 'random_forest';
      };
      backend.addEventListener('change', () => { values.execution_backend = backend.value; drawCompute(); mark(); });
      cores.addEventListener('input', drawCompute);
      classifier.addEventListener('change', () => {
        if (values.ml_method !== 'random_forest') { values.execution_backend = 'cpu'; backend.value = 'cpu'; }
        backend.disabled = values.ml_method !== 'random_forest';
        drawCompute(); mark();
      });
      backend.disabled = values.ml_method !== 'random_forest';
      compute.append(computeSummary, field('Run on', backend, 'CPU preserves Classic’s Random Forest specification. GPU requires compatible NVIDIA cuML. GPU if available explicitly falls back to CPU when unavailable; GPU only fails instead.'),
        field('CPU cores', cores, '0 = Auto: up to 32 workers, bounded by allocated CPUs and available memory. Enter a count to override Auto within your CPU allocation. More cores need more working memory. Applies to the next Random Forest run.'),
        resourceNote, gpuBins, gpuNote);
      drawCompute();
      settings.body.append(
        node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Methods to combine'}),
          help('The automated phenotyping results that vote. Reference columns you labelled by hand never vote; choose one below to check the result.')]), methodsBox]),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Agreement'}), agreement, agreementNote,
          help('How many of the chosen methods must give a cell the same phenotype for it to teach the model.')]),
        node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Markers'}),
          help('The markers the model learns from: as measured, or batch corrected. A saved set ticks its markers in one step.')]),
          markerPicker({method:'consensus', label:'Consensus', values, changed:() => mark()})]),
        field('Cannot agree on', excluded, 'Phenotypes that never count as agreement (Others, Unclassified…), comma-separated.'),
        field('Classifier', classifier, 'The model that labels the cells the methods did not agree on.'),
        field('Trees', number('n_estimators', 'Trees', '1', 10, 5000), 'Trees in the forest or boosting.'),
        compute,
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Z-score'}), tick('use_z_normalization', 'Z-score markers within each ROI'),
          node('span', {class:'grow'}), help('Scale each marker within each ROI (batch) before learning.')]),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Tissue only'}), tick('enable_filter', 'Only cells in tissue'),
          node('span', {class:'grow'}), help('Only cells marked as tissue (tissue = 1) are labelled, as classic does by default.')]),
        field('Check against', check, 'Optional: your reference cells (Manual reference). They are held out of training and the result is scored against them.'),
        field('Low score', number('refined_confidence_threshold', 'Low score', '0.05', 0, 1), 'Below this score a cell is called Others in the _refined column (default 0.75).'),
        field('Save as', output, 'The result column; the score, agreement and refined columns take its name with _score, _consensus and _refined.'), overwrite.row);
      mark();
      const status = node('p', {class:'muted', role:'status'});
      const save = async () => {
        readPhenotypingSettings(settings.body, values);
        saved = await post(route('/consensus_settings'), {changes:changes(), revision:saved.revision});
        Object.assign(values, saved.values); state.values = values; mark();
      };
      const saveButton = node('button', {class:'btn', text:'Save settings', onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try { await save(); status.textContent = 'Saved into Tab 7’s Consensus ML settings.'; } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      const runButton = node('button', {class:'btn primary', text:'Run Consensus ML', onclick:async () => {
        const unlock = lockPhenotypingSettings(settings.body, [saveButton, runButton]);
        try {
          readPhenotypingSettings(settings.body, values);
          if (Object.keys(changes()).length) { await save(); status.textContent = 'Settings saved. '; } else status.textContent = '';
          const answer = await runCommand('run_consensus_ml', {overwrite:overwrite.input.checked, settings_revision:saved.revision}, {confirmed:true});
          showResult(answer); status.textContent += answer.message;
          if (answer.status === 'done') { overwrite.input.checked = false; await refreshJobs(); follow(); }
        } catch (error) { status.textContent = error.message; }
        finally { unlock(); }
      }});
      current.runBar([node('div', {class:'row'}, [saveButton, runButton]), status], 'Runs on the whole combined table in the background; the results wait in Jobs > Saved results.');
      // The last finished run: its numbers, its phenotypes, and a clickable heatmap of their markers.
      let timer = null, pickedPhenotypes = [], pickedJob = null;
      const follow = async () => {
        clearTimeout(timer);
        let jobs = [];
        try { jobs = (await get(route('/jobs'))).jobs || []; } catch (_) { return; }
        if (!holder.isConnected) return;
        const going = jobs.find(job => job.kind === 'consensus_ml' && ['starting', 'running'].includes(job.status));
        const last = jobs.find(job => job.kind === 'consensus_ml' && job.status === 'done' && job.result?.phenotypes);
        if (going) { results.body.replaceChildren(node('p', {class:'muted', text:'Consensus ML is running: ' + (going.message || '')})); timer = setTimeout(follow, 4000); return; }
        if (!last) { results.body.replaceChildren(node('p', {class:'muted', text:'No Consensus ML run yet.'})); return; }
        if (pickedJob !== last.id) { pickedPhenotypes = []; pickedJob = last.id; }
        const result = last.result, numbers = result.report || {}, percent = value => Math.round(100 * value) + '%';
        current.publicationRun=()=>holder.isConnected ? {job_id:last.id,method:'consensus_ml',names:{},cells:result.cells} : null;
        current.publicationUpdate?.();
        const note = node('p', {class:'muted', role:'status'});
        results.body.replaceChildren(...[
          node('div', {class:'row wrap'}, [chip(result.column), result.consensus_share !== null ? chip('Agreed on ' + percent(result.consensus_share), 'ok', 'Cells the methods agreed on; the model labelled the rest') : null,
            numbers.validation_accuracy !== undefined ? chip((String(numbers.validation_mode || '').includes('manual') ? 'Vs your reference: ' : 'Held-out check: ') + percent(numbers.validation_accuracy),
              numbers.validation_accuracy >= 0.8 ? 'ok' : 'warn', 'Accuracy' + (numbers.macro_f1 !== undefined ? ' · macro F1 ' + Number(numbers.macro_f1).toFixed(2) : '')) : null].filter(Boolean)),
          node('p', {class:'muted', text:'Methods: ' + (result.methods || []).join(', ')}),
          result.profile_notice ? node('p', {class:'muted',text:result.profile_notice}) : null,
          node('div', {class:'studio-classes'}, result.phenotypes.map(item => node('div', {class:'studio-class'}, [node('span', {class:'grow', text:item.name}),
            node('span', {class:'muted', text:(item.top_markers || []).slice(0, 3).join(', ')}), node('span', {class:'muted', text:item.cells.toLocaleString()})]))),
          note].filter(Boolean));
        if (result.profile) heatmapFigure({key:'consensus', title:'Consensus phenotypes', profile:result.profile, picked:pickedPhenotypes,
          onAggregation:result.profile.summaries ? async aggregation => {await post(route('/heatmap_summary'),{method:'consensus_ml',aggregation});await follow();} : null,
          tip:'Each row is a phenotype of the result, each column a marker. Click a row to show only its cells on the image (Shift or Ctrl for several).',
          onPick:async picked => {
            pickedPhenotypes = picked;
            if (!(await showCellGroups(result.column, picked, result.profile.rows))) { note.textContent = 'Open an image whose cells are in this table to see them.'; return; }
            note.textContent = picked.length ? 'Showing ' + picked.join(', ') + ' on the image.' : 'Showing every phenotype.';
          }});
      };
      follow();
    };
    // Sub-phenotyping (Tab 7): chosen phenotypes split on the markers that vary inside each (found from the data,
    // editable), clustered and checked; the sub-clusters go to the shared cluster explorer. Settings: classic's file.
    const SUBPHENO_KEYS = ['parent_col', 'parent_values', 'markers', 'chosen_markers', 'method', 'resolution', 'n_neighbors', 'grid',
      'k_neighbors', 'n_meta_clusters', 'use_gpu', 'check_stability', 'studio_output_col'];
    PANELS.subphenotyping = async (current, holder) => {
      const state = current.subphenotyping || (current.subphenotyping = {names:{}});
      const settings = section('Sub-phenotyping settings', {tip:'Each chosen ' +
        'phenotype is split on its own, on the markers that really vary inside it.'});
      const markersBox = section('Markers inside each phenotype', {tip:'Find markers scores every panel marker inside each phenotype: markers flat ' +
        'there, driven by batch, or likely spillover from neighbouring cells are dropped (classic’s rules). Tick or untick before Run.'});
      const results = section('Sub-clusters', {tip:'The sub-clusters of the last run, with their size, most raised markers and check. Click one to ' +
        'see it on the image and where it is; name them, and Save names updates this run’s sub-phenotype column.'});
      const dirty = chip('Saved', 'ok');
      settings.box.querySelector('.studio-sec-h').insertBefore(dirty, settings.box.querySelector('.studio-sec-h .grow'));
      holder.append(settings.box, markersBox.box, results.box);
      let saved;
      try { saved = await get(route('/subpheno_settings')); }
      catch (error) { settings.body.append(node('p', {class:'hint', text:error.message})); return; }
      if (!holder.isConnected) return;
      if (!state.values) state.values = JSON.parse(JSON.stringify(saved.values));
      const values = state.values;
      values.chosen_markers = values.chosen_markers || {};
      const changes = () => Object.fromEntries(SUBPHENO_KEYS.filter(key => JSON.stringify(values[key]) !== JSON.stringify(saved.values[key])).map(key => [key, values[key]]));
      const mark = () => { const n = Object.keys(changes()).length; dirty.textContent = n ? 'Not saved' : 'Saved'; dirty.className = 'chip ' + (n ? 'warn' : 'ok'); };
      let panel = null, discovery = null;
      const column = node('select', {'aria-label':'Phenotype column to split'});
      const phenotypesBox = node('div', {class:'studio-markers'}, [node('p', {class:'muted', text:'Choose the phenotype column.'})]);
      const loadPhenotypes = async () => {
        if (!column.value) return;
        let found;
        try { found = await get(route('/phenotype_values') + '?column=' + encodeURIComponent(column.value)); }
        catch (error) { phenotypesBox.replaceChildren(node('p', {class:'hint', text:error.message})); return; }
        phenotypesBox.replaceChildren(...found.values.map(item => {
          const input = node('input', {type:'checkbox', 'aria-label':'Split ' + item.name}); input.checked = values.parent_values.includes(item.name);
          input.addEventListener('change', () => {
            values.parent_values = input.checked ? [...values.parent_values, item.name] : values.parent_values.filter(name => name !== item.name);
            mark();
          });
          return node('label', {title:item.cells.toLocaleString() + ' cells'}, [input, node('span', {text:item.name + ' (' + item.cells.toLocaleString() + ')'})]);
        }));
      };
      (async () => {
        let found = {phenotypes:[]};
        try { found = await get(route('/marker_choices') + '?method=subphenotyping'); } catch (_) { /* said below */ }
        const columns = (found.phenotypes || []).filter(name => !/(_score|_consensus|_clusters|_confidence)$/.test(name));
        column.replaceChildren(...columns.map(name => node('option', {value:name, text:name})));
        if (columns.includes(values.parent_col)) column.value = values.parent_col; else if (columns.length) { values.parent_col = column.value; mark(); }
        loadPhenotypes();
      })();
      column.addEventListener('change', () => { values.parent_col = column.value; values.parent_values = []; mark(); loadPhenotypes(); });
      const method = choice([['Leiden', 'Leiden'], ['SOM', 'SOM'], ['Vortex', 'VORTEX']], values.method, 'Sub-clustering method');
      const number = (key, label, step, min, max) => {
        const input = node('input', {type:'number', 'data-setting':key, step, min:String(min), max:String(max), value:String(values[key]), 'aria-label':label});
        input.addEventListener('change', () => { const value = Number(input.value); if (Number.isFinite(value) && value >= min && value <= max) { values[key] = step === '1' ? Math.round(value) : value; mark(); } else input.value = String(values[key]); });
        return input;
      };
      const detailHolder = node('div');
      const drawDetail = () => detailHolder.replaceChildren(values.method === 'Leiden' ?
        field('Resolution', number('resolution', 'Sub-clustering resolution', '0.1', 0.1, 3), 'Higher makes more, smaller sub-clusters.') :
        values.method === 'SOM' ? field('Grid', number('grid', 'SOM grid', '1', 4, 30), 'A grid × grid map before merging (default 10).') :
        field('K', number('k_neighbors', 'VORTEX K', '1', 5, 200), 'Neighbours per cell (default 35).'));
      method.addEventListener('change', () => { values.method = method.value; drawDetail(); mark(); });
      drawDetail();
      const tick = (key, label) => { const input = node('input', {type:'checkbox', autocomplete:'off', 'data-setting':key, 'aria-label':label}); input.checked = !!values[key];
        input.addEventListener('change', () => { values[key] = input.checked; mark(); }); return input; };
      const output = node('input', {type:'text', 'data-setting':'studio_output_col', value:values.studio_output_col, 'aria-label':'Sub-phenotype result column'});
      const overwrite = overwriteOption([output]);
      output.addEventListener('change', () => { values.studio_output_col = output.value.trim(); mark(); });
      settings.body.append(field('Phenotype column', column, 'The phenotypes to split come from this column (Consensus ML’s result, say).'),
        node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Phenotypes to split'}),
          help('Each is split on its own: clustering several together would let the differences between them dominate.')]), phenotypesBox]),
        node('div', {class:'field-block'}, [node('div', {class:'row'}, [node('label', {class:'lbl grow', text:'Marker panel'}),
          help('The markers surveyed inside each phenotype; Find markers keeps those that carry structure there.')]),
          markerPicker({method:'subphenotyping', label:'Sub-phenotyping', values, changed:() => {
            // The picker's first draw only reads the panel; a new panel makes the markers found inside each phenotype stale.
            const now = JSON.stringify(values.markers);
            if (panel !== null && now !== panel) { values.chosen_markers = {}; discovery = null; drawMarkers(); }
            panel = now; mark();
          }})]),
        field('Method', method, 'Leiden: the general choice. SOM: an even granularity. VORTEX: tight, well-separated states.'), detailHolder,
        field('Merge to', number('n_meta_clusters', 'Merge to', '1', 0, 60), 'Over-cluster, then merge look-alike sub-clusters down to this many per phenotype (0 keeps all; default 10).'),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'Stability'}), tick('check_stability', 'Check stability'), node('span', {class:'grow'}),
          help('Re-runs the split on overlapping samples and reports how well it reproduces (slower).')]),
        node('div', {class:'row field'}, [node('label', {class:'lbl', text:'GPU'}), tick('use_gpu', 'Use the GPU'), node('span', {class:'grow'}), help('Uses the GPU when this machine has one.')]),
        field('Save as', output, 'The sub-phenotype column for the names; the sub-clusters go in the same name with _clusters.'), overwrite.row);
      mark();
      // Find markers: a job; its table per phenotype, ticks editable (saved with the settings).
      const drawMarkers = () => {
        if (!discovery) {
          const kept = Object.entries(values.chosen_markers).filter(([, list]) => list.length);
          markersBox.body.replaceChildren(node('p', {class:'muted', text:kept.length ? 'Run uses the markers kept earlier for ' + kept.map(([name, list]) => name + ' (' + list.length + ')').join(', ') +
            '; Find markers to review them.' : 'Find markers to see which markers carry structure inside each phenotype; without it, Run uses classic’s suggestion.'}));
          return;
        }
        markersBox.body.replaceChildren(...Object.entries(discovery).map(([phenotype, info]) => {
          const box = node('details', {class:'studio-options', open:true});
          if (!info.markers.length) { box.append(node('summary', {text:phenotype}), node('p', {class:'hint', text:info.note || 'No markers scored.'})); return box; }
          const chosenNow = values.chosen_markers[phenotype] || info.markers.filter(row => row.recommended).map(row => row.marker);
          if (!values.chosen_markers[phenotype]) values.chosen_markers[phenotype] = chosenNow.slice();
          const summary = node('summary');
          const describe = () => { summary.textContent = phenotype + ': ' + values.chosen_markers[phenotype].length + ' of ' + info.markers.length + ' markers kept'; };
          const rows = info.markers.map(row => {
            const input = node('input', {type:'checkbox', 'aria-label':'Use ' + row.marker + ' for ' + phenotype}); input.checked = values.chosen_markers[phenotype].includes(row.marker);
            input.addEventListener('change', () => {
              values.chosen_markers[phenotype] = input.checked ? [...values.chosen_markers[phenotype], row.marker] : values.chosen_markers[phenotype].filter(m => m !== row.marker);
              describe(); mark();
            });
            return node('label', {class:'row', title:'Spread vs all cells ' + (row.spread ?? '—') + ' · batch share ' + (row.batch_share ?? '—') + ' · usable spread ' + row.score}, [input,
              node('span', {class:'grow', text:row.marker}), row.flag ? node('span', {class:'muted', text:row.flag}) : chip(row.verdict, row.verdict === 'enriched' ? 'ok' : '')]);
          });
          describe(); box.append(summary, ...rows);
          return box;
        }));
      };
      drawMarkers();
      const status = node('p', {class:'muted', role:'status'});
      const save = async () => {
        readPhenotypingSettings(settings.body, values);
        saved = await post(route('/subpheno_settings'), {changes:changes(), revision:saved.revision});
        Object.assign(values, saved.values); state.values = values; mark();
      };
      const start = async mode => {
        const unlock = lockPhenotypingSettings(settings.body, [findButton,runButton]);
        try {
          readPhenotypingSettings(settings.body, values);
          if (Object.keys(changes()).length) await save();
          const answer = await runCommand('run_subphenotyping', {mode, overwrite:mode==='run' && overwrite.input.checked, settings_revision:saved.revision}, {confirmed:true});
          showResult(answer); status.textContent = answer.message;
          if (answer.status !== 'done') return null;
          if (mode === 'run') overwrite.input.checked = false;
          await refreshJobs();
          return answer.data?.job?.id || null;
        } finally { unlock(); }
      };
      const findButton = node('button', {class:'btn', text:'Find markers', onclick:async () => {
        findButton.disabled = true;
        try {
          const id = await start('discover');
          for (let i = 0; id && i < 1200; i++) {
            await new Promise(resolve => setTimeout(resolve, 1500));
            if (!holder.isConnected) return;
            const job = ((await get(route('/jobs'))).jobs || []).find(item => item.id === id);
            if (!job || ['starting', 'running'].includes(job.status)) { status.textContent = job?.message || status.textContent; continue; }
            if (job.status !== 'done') { status.textContent = job.message || 'Finding markers stopped.'; break; }
            discovery = job.result.discovery; values.chosen_markers = {}; drawMarkers(); mark();
            status.textContent = 'Markers found; tick or untick, then Run.';
            break;
          }
        } catch (error) { status.textContent = error.message; }
        finally { findButton.disabled = false; }
      }});
      const runButton = node('button', {class:'btn primary', text:'Run sub-phenotyping', onclick:async () => {
        runButton.disabled = true;
        try { if (await start('run')) follow(); } catch (error) { status.textContent = error.message; }
        finally { runButton.disabled = false; }
      }});
      current.runBar([node('div', {class:'row'}, [findButton, runButton]), status], 'Runs on the whole combined table in the background.');
      const follow = groupList({results, holder, state, kind:'subphenotype', noun:'Phenotype', running:'Sub-phenotyping is running', none:'No sub-phenotyping run yet.'});
      follow();
    };
    const classicMethod = (current, holder, chosen) => {
      holder.append(node('section', {class:'studio-sec'}, [node('div', {class:'row'}, [chip('Classic Tab 7', 'warn'),
        help('Open this method in the classic Phenotyping tab to run it on the same cell table; ' +
          'its result column then appears in Colour by.'),
        node('a', {class:'btn small', text:'Open ' + chosen.name, href:current.step.classic, target:'spacemap-classic', rel:'noopener'})])]));
    };

    const trainingOptions = async (holder, current) => {
      const [available, initial] = await Promise.all([get(route('/selection_options')), get(route('/mrml_settings'))]);
      let saved = initial, models = initial.models || [];
      const values = JSON.parse(JSON.stringify(saved.values)), columns = initial.columns;
      const form = node('div', {class:'studio-form'}), status = node('p', {class:'muted', role:'status'});
      const dirty = chip('Saved', 'ok');
      const changes = () => Object.fromEntries(Object.keys(values).filter(k => JSON.stringify(values[k]) !== JSON.stringify(saved.values[k])).map(k => [k, values[k]]));
      const mark = () => { const changed=Object.keys(changes()).length;dirty.textContent=changed?'Not saved':'Saved';dirty.className='chip '+(changed?'warn':'ok'); };
      const input = (key, label, attrs={}) => {
        const box=node('input', {value:values[key], 'data-setting':key, 'aria-label':label, ...attrs});
        box.addEventListener('input',()=>{values[key]=box.type==='number'?Number(box.value):box.value;mark();});return box;
      };
      const select = (key, label, options) => {
        const box=choice(options, values[key], label);box.dataset.setting=key;
        if(!options.some(([value])=>value===values[key])&&values[key])box.append(node('option',{value:values[key],text:values[key]+' (unavailable)'}));
        box.value=values[key];box.onchange=()=>{values[key]=box.value;mark();};return box;
      };
      const tick = (key, label, tip) => {
        const box=node('input',{type:'checkbox','data-setting':key,'aria-label':label});box.checked=!!values[key];
        box.onchange=()=>{values[key]=box.checked;mark();};return node('label',{class:'row'},[box,node('span',{class:'grow',text:label}),tip?help(tip):null]);
      };
      const phenotypeColumns=columns.filter(c=>c.startsWith('phenotype_')&&!/(_score|_confidence|_consensus|_clusters)$/.test(c));
      const referenceColumns=Array.from(new Set([...(initial.references||[]),...phenotypeColumns]));
      const source=select('ground_truth_col','Training reference column',referenceColumns.map(c=>[c,c]));
      const workflow=select('workflow','MR-ML workflow',[['saved_model','Use a saved reference model'],['classic','Classic MR-ML']]);
      const description=node('p',{class:'hint'}), classic=node('div'), stored=node('div');
      form.append(node('div',{class:'studio-sec-h'},[node('span',{text:'MR-ML settings'}),dirty,
        help('Use an existing Manual Reference model, or run Classic MR-ML to train on reference cells and optionally add F1-weighted votes from other methods. Both operate on the combined table across all ROIs.')]),field('Workflow',workflow),description,field('Reference',source),stored,classic);
      const model=select('model_id','Saved reference model',[]), modelInfo=node('p',{class:'hint'});
      const drawModels=()=>{
        model.replaceChildren(node('option',{value:'',text:models.length?'Choose a saved model…':'No compatible model yet'}),...models.map(m=>node('option',{
          value:m.model_id,text:(m.auto_train?'Auto · ':'')+m.method+' · '+m.column+' · '+new Date(m.trained).toLocaleString(),
        })));
        if(!models.some(m=>m.model_id===values.model_id)){
          const replacement=models.find(m=>(m.replaces||[]).includes(values.model_id));
          if(replacement)values.model_id=replacement.model_id;
        }
        model.value=values.model_id;
      };
      const describeModel=()=>{
        const m=models.find(m=>m.model_id===model.value);
        modelInfo.textContent=m ? Number(m.examples||0).toLocaleString()+' training cells · '+(m.markers.length-(m.missing_markers||[]).length)+'/'+m.markers.length+' features available'+
          (m.held_out_balanced_accuracy!=null?' · held-out balanced accuracy '+(100*m.held_out_balanced_accuracy).toFixed(1)+'%':'')+
          ((m.missing_markers||[]).length?'. Missing: '+m.missing_markers.join(', ')+'. A model using available features will be prepared.':'.') : 'Train a model in Manual reference, then refresh this list.';
      };
      model.addEventListener('change',describeModel);
      model.addEventListener('change',()=>{const chosen=models.find(m=>m.model_id===model.value);if(chosen&&columns.includes(chosen.column)){source.value=chosen.column;values.ground_truth_col=chosen.column;mark();}});
      stored.append(field('Model',model),modelInfo,node('div',{class:'row'},[
        node('button',{class:'btn small',text:'Refresh models',onclick:async()=>{try{models=(await get(route('/mrml_settings'))).models||[];drawModels();describeModel();mark();}catch(e){status.textContent=e.message;}}}),
        node('button',{class:'btn small',text:'Manual reference…',onclick:()=>window.dispatchEvent(new CustomEvent('studio-reference-method',{detail:{method:'reference'}}))})]),
        tick('preserve_reference_labels','Keep existing reference labels','Keeps valid labels from the reference column below, with blank scores. Untick to predict every eligible cell. If that column is absent, all eligible cells are predicted.'),
        tick('match_scale','Match marker scales between projects','Optional quantile matching to the training data. Use only when measurement scales differ. Morphology and H&E retain their measured scale.'));
      drawModels();describeModel();
      const mode=choice([['reference','Reference cells only'],['ensemble','Reference + F1-weighted methods']],values.phenotyping_methods.length?'ensemble':'reference','MR-ML mode');
      const methods=node('div',{class:'studio-method-list'}), picks=[];
      const drawMethods=()=>{
        methods.replaceChildren();picks.length=0;
        phenotypeColumns.filter(c=>c!==source.value&&!(initial.references||[]).includes(c)&&!/(^|_)ref(erence)?[0-9_]*($|_)/i.test(c)).forEach(name=>{
          const box=node('input',{type:'checkbox','aria-label':'Combine '+name});box.checked=values.phenotyping_methods.includes(name);
          box.onchange=()=>{values.phenotyping_methods=picks.filter(([,b])=>b.checked).map(([n])=>n);mark();};picks.push([name,box]);
          methods.append(node('label',{},[box,node('span',{text:name})]));
        });
      };
      const methodBox=node('details',{class:'studio-options',open:true},[node('summary',{text:'Methods to combine'}),
        node('p',{class:'hint',text:'Each method’s vote is weighted by its per-phenotype F1 against training reference cells.'}),methods]);
      mode.onchange=()=>{methodBox.hidden=mode.value!=='ensemble';values.phenotyping_methods=mode.value==='ensemble'?picks.filter(([,b])=>b.checked).map(([n])=>n):[];mark();};
      source.addEventListener('change',()=>{values.phenotyping_methods=values.phenotyping_methods.filter(c=>c!==source.value);drawMethods();mark();});
      drawMethods();methodBox.hidden=mode.value!=='ensemble';
      classic.append(field('Mode',mode),methodBox,field('Markers',markerPicker({method:'mr_ml',label:'MR-ML',values,changed:mark})),
        field('Classifier',select('ml_method','MR-ML classifier',[['random_forest','Random forest'],['gradient_boosting','Gradient boosting'],['svm','SVM']])),
        field('Trees',input('n_estimators','MR-ML trees',{type:'number',min:10,max:5000,step:1})));
      const featureBox=node('details',{class:'studio-options'},[node('summary',{text:'Additional features'})]);
      const extraColumns=(available.measurements||[]).filter(c=>!c.startsWith('phenotype_')&&!['cell_id','label','batch_id'].includes(c));
      for(const [key,title] of [['morphology_cols','Morphology'],['compartment_cols','Compartments']]){
        const list=node('div',{class:'studio-method-list'}), controls=[];
        const search=node('input',{placeholder:'Find a measurement…','aria-label':'Find '+title.toLowerCase()+' feature'});
        extraColumns.forEach(name=>{
          const box=node('input',{type:'checkbox','aria-label':'Use '+title.toLowerCase()+' '+name});box.checked=values[key].includes(name);
          const row=node('label',{},[box,node('span',{text:name})]);controls.push([name,box,row]);list.append(row);
          box.onchange=()=>{values[key]=controls.filter(([,b])=>b.checked).map(([n])=>n);mark();};
        });
        search.oninput=()=>controls.forEach(([name,,row])=>{row.hidden=!name.toLowerCase().includes(search.value.toLowerCase());});
        list.style.maxHeight='160px';list.style.overflow='auto';
        const details=node('details',{},[node('summary',{text:title}),search,list]);
        details.readSettings=()=>{values[key]=controls.filter(([,b])=>b.checked).map(([n])=>n);};featureBox.append(details);
      }
      for(const key of ['markers','morphology','compartments']){
        const box=node('input',{type:'number',min:0,max:100,step:.1,value:values.feature_weights[key],'aria-label':key+' feature weight'});
        box.onchange=()=>{values.feature_weights[key]=Number(box.value);mark();};
        box.dataset.featureWeight=key;
        featureBox.append(field(key[0].toUpperCase()+key.slice(1)+' weight',box,'Zero leaves this feature group out of training.'));
      }
      classic.append(featureBox);
      const advanced=node('details',{class:'studio-options'},[node('summary',{text:'Training and validation'})]);
      advanced.append(field('CPU cores',input('cpu_workers','MR-ML CPU cores',{type:'number',min:0,max:4096,step:1}),'0 uses the automatic CPU budget.'),
        field('Hold-out share',input('test_size','MR-ML hold-out share',{type:'number',min:.05,max:.5,step:.05})),
        field('Random seed',input('random_state','MR-ML random seed',{type:'number',min:0,max:2147483647,step:1})),
        field('Check against',select('validation_ground_truth_col','MR-ML validation reference',[['','Split training reference'],...referenceColumns.map(c=>[c,c])]),'External validation uses cells absent from training and F1 weighting. Cells labelled in both references are omitted from validation when the option below is on.'),
        tick('drop_overlapping_validation','Exclude overlapping validation cells'),
        tick('use_z_normalization','Z-score markers within each batch'),
        field('Batch column',select('batch_normalization_col','MR-ML batch column',columns.map(c=>[c,c]))),
        field('Exclude labels',input('exclude_phenotypes_text','MR-ML labels to predict',{value:values.exclude_phenotypes.join(', ')}),'Comma-separated labels excluded from training and method votes.'));
      // This text control represents an array; it is not a separate persisted setting.
      const excluded=advanced.querySelector('[data-setting="exclude_phenotypes_text"]');excluded.removeAttribute('data-setting');
      excluded.oninput=()=>{delete values.exclude_phenotypes_text;values.exclude_phenotypes=excluded.value.split(',').map(x=>x.trim()).filter(Boolean);mark();};
      classic.append(advanced);
      // A reference is optional only when applying an existing model.
      const filterFields=node('div',{},[field('Where',select('filter_col','MR-ML filter column',columns.map(c=>[c,c]))),field('Equals',input('filter_val','MR-ML filter value'))]);
      const limitRows=tick('enable_filter','Limit rows','The job always includes all ROIs in the combined table. Untick to include every row; tick to apply the selected column/value filter.');
      filterFields.hidden=!values.enable_filter;limitRows.querySelector('input').addEventListener('change',()=>filterFields.hidden=!values.enable_filter);
      form.append(limitRows,filterFields);
      const output=input('output_phenotype_col','MR-ML result column'),score=input('confidence_col','MR-ML score column'),refined=input('refined_output_phenotype_col','MR-ML filtered column');
      let previousName=output.value;
      output.addEventListener('input',()=>{if(score.value===previousName+'_confidence'||score.value==='ml_mr_confidence'){score.value=output.value+'_confidence';values.confidence_col=score.value;}
        if(refined.value===previousName+'_refined'){refined.value=output.value+'_refined';values.refined_output_phenotype_col=refined.value;}previousName=output.value;mark();});
      const overwrite=overwriteOption([output,score,refined]);
      const outputs=node('details',{class:'studio-options'},[node('summary',{text:'Scores and output columns'}),
        field('Score column',score),field('Filtered labels',refined),
        field('Minimum score',input('refined_confidence_threshold','MR-ML minimum score',{type:'number',min:0,max:1,step:.05}),
          'Saved model: model support. Classic ensemble: normalized model-plus-method support. These are not calibrated probabilities.'),
        field('Below cutoff',input('refined_low_confidence_label','MR-ML low-score label'))]);
      form.append(field('Save as',output),outputs,overwrite.row);holder.append(form);
      const drawWorkflow=()=>{const useSaved=workflow.value==='saved_model';stored.hidden=!useSaved;classic.hidden=useSaved;
        description.textContent=useSaved?'Apply an existing Manual Reference model across the cohort. No new model is trained unless missing features require adaptation.':
          'Classic MR-ML: train from reference cells, optionally adding F1-weighted votes from other phenotyping methods.';mark();};
      workflow.addEventListener('change',drawWorkflow);drawWorkflow();
      const read=()=>{readPhenotypingSettings(form,values);form.querySelectorAll('[data-feature-weight]').forEach(box=>values.feature_weights[box.dataset.featureWeight]=Number(box.value));values.exclude_phenotypes=excluded.value.split(',').map(x=>x.trim()).filter(Boolean);delete values.exclude_phenotypes_text;
        if(mode.value==='reference')values.phenotyping_methods=[];mark();};
      const save=async()=>{read();saved=await post(route('/mrml_settings'),{changes:changes(),revision:saved.revision});Object.assign(values,saved.values);mark();};
      const saveButton=node('button',{class:'btn',text:'Save settings',onclick:async()=>{try{await save();status.textContent='Saved for Studio and Classic MR-ML.';}catch(e){status.textContent=e.message;}}});
      const runButton=node('button',{class:'btn primary',text:'Run MR-ML',onclick:async()=>{
        const unlock=lockPhenotypingSettings(form,[saveButton,runButton]);
        try{read();if(values.workflow==='classic'&&mode.value==='ensemble'&&!values.phenotyping_methods.length)throw Error('Choose methods to combine, or use Reference cells only.');
          await save();const answer=await runCommand('run_mr_ml',{settings_revision:saved.revision,overwrite:overwrite.input.checked},{confirmed:true});
          showResult(answer);status.textContent=answer.message;if(answer.status==='done'){overwrite.input.checked=false;await refreshJobs();showTab('jobs');}
        }catch(e){status.textContent=e.message;}finally{unlock();}
      }});
      current.runBar([node('div',{class:'row'},[saveButton,runButton]),status],'Runs on the whole combined table in the background; results appear in Jobs.');
      if(current.mrmlIntent){
        const intent=current.mrmlIntent;values.workflow='saved_model';workflow.value='saved_model';values.enable_filter=false;form.querySelector('[data-setting="enable_filter"]').checked=false;filterFields.hidden=true;
        if(columns.includes(intent.column)){values.ground_truth_col=intent.column;source.value=intent.column;}
        const preferred=models.find(m=>m.model_id===intent.model_id)||models.find(m=>m.column===intent.column);
        if(preferred){values.model_id=preferred.model_id;model.value=preferred.model_id;describeModel();}drawWorkflow();
      }
    };

    // ------------------------------------------------------------ jobs
    // Saved results: Studio never rewrites the table on its own. Its columns wait beside the combined table (read with it
    // at once) until Write into the table; tables Studio makes are Parquet, with a CSV copy for classic on request.
    const savedBox = section('Saved results', {tip:'What Studio saved: result columns wait beside the combined table (Parquet files) and are ' +
      'read with it at once; classic sees them too. Write into the table puts them in for good (the CSV, or Studio’s Parquet table), in ' +
      'the background. Tables Studio makes are Parquet; Save a CSV copy makes one for classic SPACEMAP.'});
    const savedStatus = node('p', {class:'muted', role:'status'});
    const drawSaved = value => {
      const rows = [];
      if (!value.table) rows.push(node('p', {class:'muted', text:'No combined table yet.'}));
      else {
        const fresh = value.pending.filter(item => !item.stale), stale = value.pending.filter(item => item.stale);
        rows.push(node('p', {class:'muted', text:value.table + ': ' + (fresh.length || value.labels ? fresh.length + ' column' + (fresh.length === 1 ? '' : 's') +
          (value.labels ? ' and ' + value.labels + ' label' + (value.labels === 1 ? '' : 's') : '') + ' waiting to be written in' : 'nothing waiting')}));
        fresh.forEach(item => rows.push(node('div', {class:'studio-folder'}, [node('span', {class:'grow mono', text:item.column, title:(item.source || '') + ' · ' + (item.saved_at || '')}),
          node('button', {class:'btn small ghost', text:'Discard', onclick:async () => {
            if (!confirm('Discard ' + item.column + '? It has not been written into the table, so it cannot be brought back.')) return;
            try { drawSaved(await post(route('/saved/discard'), {columns:[item.column]})); savedStatus.textContent = item.column + ' discarded.'; } catch (error) { savedStatus.textContent = error.message; }
          }})])));
        if (stale.length) rows.push(node('div', {class:'studio-import'}, [node('p', {class:'hint', text:stale.length + ' saved column' + (stale.length === 1 ? '' : 's') +
            ' no longer match the table (it was rewritten). Re-apply them by cell ID, or discard them.'}),
          node('div', {class:'row'}, [node('button', {class:'btn small', text:'Re-apply by cell ID', onclick:async () => {
            try { drawSaved(await post(route('/saved/reapply'), {})); } catch (error) { savedStatus.textContent = error.message; }
          }})])]));
        const writing = value.writing?.state === 'writing';
        if (fresh.length || value.labels || writing) rows.push(node('div', {class:'row'}, [node('button', {class:'btn primary small', text:writing ? 'Writing…' : 'Write into the ' + (value.parquet ? 'table' : 'CSV'),
          disabled:writing || undefined, onclick:async () => {
            if (!confirm('Write these results into ' + value.table + '? This rewrites the ' + (value.parquet ? 'table' : 'CSV') + ' (in the background; minutes for a large table).')) return;
            try { drawSaved(await post(route('/saved/write'), {})); savedStatus.textContent = 'Writing in the background…'; pollSaved(); } catch (error) { savedStatus.textContent = error.message; }
          }}), node('span', {class:'muted', text:value.writing?.message || ''})]));
      }
      if (value.tables?.length) {
        rows.push(node('p', {class:'lbl', text:'Parquet tables'}));
        value.tables.forEach(item => rows.push(node('div', {class:'studio-folder'}, [node('span', {class:'grow mono', text:item.name, title:item.megabytes + ' MB'}),
          item.in_use ? chip('In use', 'ok') : null,
          node('button', {class:'btn small ghost', text:item.csv ? 'Update CSV copy' : 'Save a CSV copy', title:'Includes saved predictions, scores and label edits; minutes for a large table', onclick:async () => {
            if (item.csv && !confirm('Replace ' + item.csv + ' with a new copy including the current saved results?')) return;
            try { await post(route('/saved/export'), {name:item.name, table:item.id}); savedStatus.textContent = 'Saving a CSV copy of ' + item.name + ' (see Background jobs)…'; await refreshJobs(); }
            catch (error) { savedStatus.textContent = error.message; }
          }})].filter(Boolean))));
      }
      savedBox.body.replaceChildren(...rows, savedStatus);
    };
    let savedTimer = null;
    const loadSaved = async () => { try { drawSaved(await get(route('/saved'))); } catch (error) { savedStatus.textContent = error.message; } };
    const pollSaved = async () => {
      clearTimeout(savedTimer);
      let value; try { value = await get(route('/saved')); } catch (_) { return; }
      drawSaved(value);
      if (value.writing?.state === 'writing') savedTimer = setTimeout(pollSaved, 3000);
      else if (value.writing?.state === 'done') savedStatus.textContent = 'Written into ' + value.table + '.';
    };
    // This user's SPACEMAP servers, wherever they run: an old one left running can be stopped (from its own machine).
    const serversBox = section('SPACEMAP servers', {tip:'Every SPACEMAP you have running, on any node. An old one left running can keep ' +
      'saving files you are editing here. Stop stops one on this machine; for another machine it says where it runs.'});
    const loadServers = async () => {
      try {
        const found = await get(route('/servers'));
        const rows = (found.servers || []).map(item => {
          const started = new Date(item.started * 1000).toLocaleString();
          const stop = item.this ? chip('This one', 'ok') : node('button', {class:'btn small', text:'Stop', onclick:async () => {
            if (!window.confirm('Stop the SPACEMAP running on ' + item.host + ':' + item.port + ' (started ' + started + ')?')) return;
            try { const done = await post(route('/servers'), {host:item.host, port:item.port}); showResult({message:done.done}); }
            catch (error) { showResult({message:error.message}); }
            loadServers();
          }});
          return node('div', {class:'row studio-server'}, [node('span', {class:'grow', text:item.host + ':' + item.port,
            title:'Code: ' + item.source + (item.slurm_job ? ' · Slurm job ' + item.slurm_job : '') + ' · process ' + item.pid}),
            node('span', {class:'muted', text:'since ' + started}), stop]);
        });
        fill(serversBox.body, ...(rows.length ? rows : [node('p', {class:'muted', text:'Only this one.'})]),
          rows.length > 1 ? node('p', {class:'hint', text:rows.length + ' SPACEMAP servers are running: stop the ones you no longer use.'}) : null);
      } catch (error) { serversBox.body.replaceChildren(node('p', {class:'hint', text:error.message})); }
    };
    onJobsShown = () => { loadSaved(); loadServers(); };
    if (!jobsPane.hidden) { loadSaved(); loadServers(); }
    const jobRows = node('div');
    const jobCards = new Map(), jobsError = node('p', {class:'hint studio-pad', hidden:true});
    const noJobs = node('p', {class:'muted studio-pad', text:'No Studio jobs yet.'});
    jobsPane.append(savedBox.box, section('Background jobs', {tip:'Jobs keep running when you change step or close Studio (or restart it). Each records ' +
      'time, CPU, memory and GPU in the run log. Stop one here if it would overwrite what you are editing.'}).box, jobsError, jobRows, serversBox.box);
    let jobStamp = '', jobsBusy = false, running = 0;
    let trainedJobs=null;
    const refreshJobs = async () => {
      if (jobsBusy || !context?.has_project) return; jobsBusy = true;
      try {
        const result = await get(route('/jobs'));
        jobsError.hidden = true;
        // Open logs refresh even during a long stage with no new status message.
        jobCards.forEach(card => { if (ACTIVE_JOBS.includes(card.job.status)) card.jobLog.refresh(); });
        const stamp = JSON.stringify(result); if (stamp === jobStamp) return; jobStamp = stamp;
        lastJobs = result.jobs; frame?.jobsChanged?.(result.jobs);  // the open step's list follows its running jobs
        if(result.auto_reference)document.dispatchEvent(new CustomEvent('spacemap-auto-reference',{detail:result.auto_reference}));
        const finished=result.jobs.filter(job=>job.kind==='train_reference_model'&&job.status==='done'&&!job.result?.auto_train);
        if(trainedJobs)finished.filter(job=>!trainedJobs.has(job.id)).forEach(job=>document.dispatchEvent(new CustomEvent('spacemap-reference-trained',{detail:job})));
        trainedJobs=new Set(finished.map(job=>job.id));
        const before = running;
        running = result.jobs.filter(job => ['starting', 'running', 'stopping'].includes(job.status)).length;
        [jobsBadge, tabBadge].forEach(badge => { badge.textContent = String(running); badge.classList.toggle('idle', !running); });
        jobsRunning = running; updateBusy();
        if (before && !running) { loadOverview(); frame?.refreshItems?.(); if (!jobsPane.hidden) loadSaved(); }
        const scroll = jobsPane.scrollTop, present = new Set(result.jobs.map(job => job.id));
        jobCards.forEach((card, id) => { if (!present.has(id)) { card.remove(); jobCards.delete(id); } });
        noJobs.remove();
        result.jobs.forEach((job, index) => {
          let card = jobCards.get(job.id);
          if (!card || card.jobStamp !== JSON.stringify(job)) {
            const previous = card, logScroll = previous?.jobLog.text.scrollTop || 0;
            card = jobCard(job, previous);
            if (previous) previous.replaceWith(card);
            jobCards.set(job.id, card);
            if (previous) card.jobLog.text.scrollTop = logScroll;
            card.jobLog.refresh();
          }
          if (jobRows.children[index] !== card) jobRows.insertBefore(card, jobRows.children[index] || null);
        });
        if (!result.jobs.length) jobRows.append(noJobs);
        jobsPane.scrollTop = scroll;
      } catch (error) { jobsError.textContent = error.message; jobsError.hidden = false; }
      finally { jobsBusy = false; }
    };
    const updateJobTimer = (timer, job) => {
      const active = ACTIVE_JOBS.includes(job.status), started = Date.parse(job.started), finished = Date.parse(job.finished);
      let duration = Number.isFinite(started) && (active || Number.isFinite(finished))
        ? ((active ? Date.now() : finished) - started) / 1000 : job.usage?.duration_s;
      const available = typeof duration === 'number' && Number.isFinite(duration);
      duration = available ? Math.max(0, Math.floor(duration)) : 0;
      const hours = Math.floor(duration / 3600), minutes = Math.floor(duration / 60) % 60, seconds = duration % 60;
      timer.textContent = (active ? 'Elapsed ' : 'Duration ') + (available ?
        (hours ? hours + ':' : '') + String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0') : '—');
      timer.title = Number.isFinite(started) ? 'Started ' + new Date(started).toLocaleString() +
        (Number.isFinite(finished) ? ' · Finished ' + new Date(finished).toLocaleString() : '') : 'Timing unavailable for this older job';
    };
    const makeJobLog = id => {
      const box = node('details', {class:'studio-job-log'}), text = node('pre');
      box.append(node('summary', {text:'Log'}), text);
      let busy = false;
      const refresh = async () => {
        if (!box.open || busy) return;
        busy = true;
        try {
          // Background log polling must not flash the global Loading indicator.
          const value = (await rawGet(route('/jobs/' + id))).log || '';
          box.title = '';
          if (text.textContent !== value) {
            const scroll = text.scrollTop, follow = !text.textContent || text.scrollHeight - scroll - text.clientHeight < 3;
            text.textContent = value;
            text.scrollTop = follow ? text.scrollHeight : scroll;
          }
        } catch (error) {
          box.title = error.message;
          if (!text.textContent) text.textContent = 'Could not load the log: ' + error.message;
        }
        finally { busy = false; }
      };
      box.addEventListener('toggle', () => { if (box.open) refresh(); });
      return {box, text, refresh};
    };
    const jobCard = (job, previous) => {
      const active = ['starting', 'running', 'stopping'].includes(job.status);
      const kind = active ? 'run' : job.status === 'done' ? 'ok' : job.status === 'failed' ? 'error' : 'warn';
      const timer = node('span', {class:'muted studio-job-timer', 'aria-label':'Job elapsed time'});
      updateJobTimer(timer, job);
      const card = node('div', {class:'studio-job', 'data-job-id':job.id}, [
        node('div', {class:'row'}, [node('strong', {text:JOB_NAMES[job.kind] || job.kind}), node('span', {class:'grow'}),
          timer,
          chip(active ? Math.round(100 * (job.progress || 0)) + '%' : job.status, kind)]),
        node('div', {class:'studio-bar'}, [node('div', {style:'width:' + Math.round(100 * (job.progress || (job.status === 'done' ? 1 : 0))) + '%'})]),
        node('p', {class:'muted', text:job.message})]);
      const actions = node('div', {class:'row'});
      if (active) {
        const stop = node('button', {class:'btn small', text:job.status === 'stopping' ? 'Stopping…' : 'Stop',
          disabled:job.status === 'stopping', title:'Stop this job. A result save already in progress is allowed to finish.', onclick:async () => {
          stop.disabled = true;
          stop.textContent = 'Stopping…';
          try {
            const stopped = await post(route('/jobs/' + job.id), {});
            card.querySelector('p.muted').textContent = stopped.message || 'Stopping computation…';
            jobStamp = ''; await refreshJobs();
          }
          catch (error) { stop.disabled = false; stop.textContent = 'Stop'; showResult({message:error.message}); }
        }}); actions.append(stop);
      }
      if (job.kind === 'segment_images' && job.result) card.append(node('p', {class:'muted', text:Number(job.result.cells).toLocaleString() +
        ' cells in ' + job.result.count + ' image' + (job.result.count === 1 ? '' : 's') + '.'}));
      else if (job.kind === 'clean_tissue' && job.result) card.append(node('p', {class:'muted', text:(job.result.images || []).map(item =>
        item.sample + ': ' + Number(item.artifact_percent || 0).toFixed(1) + '% removed').join('; ')}));
      else if (job.kind === 'export_classes' && job.result) card.append(node('p', {class:'muted', text:job.result.sample + ': ' +
        Object.keys(job.result.classes || {}).length + ' classes, ' + job.result.domains + ' domain' + (job.result.domains === 1 ? '' : 's') +
        ' at ' + Number(job.result.resolution_um).toFixed(2) + ' \u00b5m per pixel.'}));
      else if (job.kind === 'leiden' && job.result) card.append(node('p', {class:'muted', text:(job.result.clusters || []).length +
        ' clusters in ' + job.result.column + '. Name them in Phenotyping \u203a Leiden.'}));
      else if (job.kind === 'train_cellpose' && job.result) card.append(node('p', {class:'muted', text:job.result.name + ' from ' +
        job.result.training_tiles + ' training tiles' + (job.result.ap_after !== null ? ' · validation AP ' + Number(job.result.ap_before).toFixed(2) +
        ' \u2192 ' + Number(job.result.ap_after).toFixed(2) : '') + '.'}));
      else if (job.kind === 'tls_candidates' && job.result) card.append(node('p', {class:'muted', text:job.result.total + ' TLS candidates in ' + job.result.column + '. Review in Spatial analysis → TLS candidates.'}));
      else if (job.kind === 'neighbourhoods' && job.result) card.append(node('p', {class:'muted', text:(job.result.clusters || []).length +
        ' neighbourhoods in ' + job.result.column + '. Name them in Spatial analysis.'}));
      else if (job.kind === 'name_clusters' && job.result) card.append(node('p', {class:'muted', text:Object.keys(job.result.phenotypes || {}).length +
        ' phenotypes in ' + job.result.column + '.'}));
      else if (job.kind === 'apply_pieces' && job.result) card.append(node('p', {class:'muted', text:(job.result.images || []).map(item =>
        item.sample + ': ' + item.kept + ' kept, ' + item.rejected + ' dropped').concat(job.result.skipped || []).join('; ')}));
      else if (job.kind === 'make_zones' && job.result) card.append(node('p', {class:'muted', text:(job.result.images || []).map(item =>
        item.sample + ': ' + item.zones + ' ring' + (item.zones === 1 ? '' : 's') + (item.pooled.length ? ', pools ' + item.pooled.join(', ') : ''))
        .concat(job.result.skipped || []).join('; ')}));
      else if (job.kind === 'detect_tissue' && job.result) card.append(node('p', {class:'muted', text:'Tissue found for ' +
        job.result.count + ' image' + (job.result.count === 1 ? '' : 's') + '.'}));
      else if (job.kind === 'classify_shared' && job.result) card.append(node('p', {class:'muted', text:
        (job.result.fitted ? job.result.n_clusters + ' shared classes found (' + job.result.name + '); ' : 'With ' + job.result.name + ': ') +
        job.result.count + ' image' + (job.result.count === 1 ? '' : 's') + ' classified.' +
        ((job.result.images || []).filter(item => item.far_percent >= 2 || item.differ.length).length ? ' Check: ' + job.result.images.filter(item => item.far_percent >= 2 || item.differ.length)
          .map(item => item.image + (item.far_percent >= 2 ? ' (' + item.far_percent + '% fit no class)' : '') + (item.differ.length ? ' (classes ' + item.differ.join(', ') + ' differ)' : '')).join(', ') + '.' : '') +
        ((job.result.skipped || []).length ? ' Left out of the fit: ' + job.result.skipped.join('; ') + '.' : '') +
        ((job.result.failed || []).length ? ' Not classified: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') + '.' : '')}));
      else if (job.kind === 'save_class_polygons' && job.result) card.append(node('p', {class:'muted', text:
        (job.result.images || []).map(item => item.image + ' ' + item.polygons).join(', ') + ' polygons saved.' +
        ((job.result.failed || []).length ? ' Not saved: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') + '.' : '')}));
      else if (job.kind === 'remove_small_pieces' && job.result) card.append(node('p', {class:'muted', text:
        (job.result.images || []).map(item => item.image + ' −' + item.removed).join(', ') + ' pieces under ' + job.result.min_um2 + ' µm² removed.' +
        ((job.result.failed || []).length ? ' Not done: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') : '')}));
      else if (job.kind === 'subtract_annotations' && job.result) card.append(node('p', {class:'muted', text:
        (job.result.images || []).map(item => item.image + ' ' + item.count + (item.missing.length ? ' (no ' + item.missing.join(', ') + ')' : '')).join(', ') + ': ' + job.result.label + ' made.' +
        ((job.result.failed || []).length ? ' Not done: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') : '')}));
      else if (job.kind === 'artifact_batch' && job.result) card.append(node('p', {class:'muted', text:[
        (job.result.images || []).filter(item => 'found' in item).map(item => item.image + ' ' + item.found).join(', '),
        (job.result.images || []).filter(item => item.subtracted).length ? (job.result.images || []).filter(item => item.subtracted).length + ' subtracted from Work within' : '',
        (job.result.kept || []).length ? 'Kept their saved polygons: ' + job.result.kept.join(', ') : '',
        (job.result.failed || []).length ? 'Not done: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') : ''].filter(Boolean).join('. ') + '.'}));
      else if (['detect_artifacts', 'subtract_artifacts', 'marker_regions'].includes(job.kind) && job.result) card.append(node('p', {class:'muted', text:
        (job.result.images || []).map(item => item.sample || item.image).join(', ') + ': ' +
        (job.kind === 'detect_artifacts' ? 'artifacts ready to review; tissue unchanged.' : job.kind === 'subtract_artifacts' ? 'reviewed artifacts subtracted; earlier tissue kept in History.' : 'regions ready to review.') +
        ((job.result.skipped || []).length ? ' Skipped: ' + job.result.skipped.join('; ') : '')}));
      else if (job.kind === 'register_pairs' && job.result) card.append(node('p', {class:'muted', text:job.result.count + ' pair' +
        (job.result.count === 1 ? '' : 's') + ' registered.'}));
      else if (job.kind === 'import_annotations' && job.result) card.append(node('p', {class:'muted', text:job.result.count + ' file' +
        (job.result.count === 1 ? '' : 's') + ' imported from ' + job.result.folder + '.' +
        ((job.result.unmatched || []).length ? ' Matching no image: ' + job.result.unmatched.join(', ') + ' (import them in Annotation > Export to choose their image).' : '') +
        ((job.result.skipped || []).length ? ' Skipped: ' + job.result.skipped.join('; ') + '.' : '')}));
      else if (job.kind === 'export_annotations' && job.result) card.append(node('p', {class:'muted', text:job.result.count + ' image' +
        (job.result.count === 1 ? '' : 's') + ' exported to ' + job.result.folder + '.' +
        ((job.result.failed || []).length ? ' Not exported: ' + job.result.failed.map(item => item.image + ' (' + item.error + ')').join('; ') + '.' : '') +
        ((job.result.images || []).some(item => (item.left_out || []).length) ? ' Some had nothing left of a ticked annotation: see Log.' : '')}));
      else if (job.result?.images) card.append(node('p', {class:'muted', text:job.result.count + (job.kind === 'extract_rois' ?
        ' ROI image' + (job.result.count === 1 ? '' : 's') + ' written.' :
        ' background-removed image' + (job.result.count === 1 ? '' : 's') + ' written; the originals stay.')}));
      if (job.status === 'done' && job.result?.column) actions.append(node('button', {class:'btn small', text:'Show result', onclick:async () => {
        try { showResult(await runCommand('show_job_result', {job_id:job.id})); } catch (error) { showResult({message:error.message}); }
      }}));
      if (job.result?.agreement) Object.entries(job.result.agreement).forEach(([name, value]) => card.append(node('p', {class:'muted',
        text:name + ': ' + (100 * value.agreement).toFixed(1) + '% agreement across ' + value.cells.toLocaleString() + ' cells.'})));
      if (Object.keys(job.result?.agreement || {}).length) {
        const details = node('details'); details.append(node('summary', {text:'Agreement by phenotype'}));
        Object.entries(job.result.agreement).forEach(([name, value]) => {
          const table = node('table', {class:'studio-counts'}); table.append(node('caption', {text:name}));
          Object.entries(value.per_phenotype || {}).forEach(([phenotype, stats]) => table.append(node('tr', {}, [node('td', {text:phenotype}),
            node('td', {text:(100 * stats.agreement).toFixed(1) + '%'}), node('td', {text:stats.cells.toLocaleString()})])));
          details.append(table);
        });
        card.append(details);
      }
      if (job.result?.models) job.result.models.forEach(model => card.append(node('p', {class:'muted', text:model.method + ': ' +
        (100 * model.held_out_balanced_accuracy).toFixed(1) + '% agreement with held-out reference labels. Choose “Saved reference model” in Manual reference.'})));
      if (job.result?.reference_f1) {
        const details=node('details'), table=node('table',{class:'studio-counts'});
        details.append(node('summary',{text:'Training-reference F1 weights'}));
        const methods=Object.keys(Object.values(job.result.reference_f1)[0]||{});
        table.append(node('tr',{},[node('th',{text:'Phenotype'}),...methods.map(name=>node('th',{text:name}))]));
        Object.entries(job.result.reference_f1).forEach(([name, weights])=>table.append(node('tr',{},[node('td',{text:name}),
          ...methods.map(method=>node('td',{text:Number(weights[method]).toFixed(3)}))])));
        details.append(table);card.append(details);
      }
      // Retain the actual disclosure and log nodes through stage/status updates.
      card.jobLog = previous?.jobLog || makeJobLog(job.id);
      card.job = job; card.jobStamp = JSON.stringify(job); card.jobTimer = timer;
      actions.append(card.jobLog.box); card.append(actions);
      return card;
    };
    setInterval(refreshJobs, 3000);
    setInterval(() => jobCards.forEach(card => {
      if (ACTIVE_JOBS.includes(card.job.status)) updateJobTimer(card.jobTimer, card.job);
    }), 1000);

    // ------------------------------------------------------------ images, results, the finder
    let openingImage = false;
    const chooseImage = async image => {
      if (openingImage) return;
      openingImage = true; projectMenu.open = false;
      menu.hidden = true; search.setAttribute('aria-expanded', 'false'); search.value = '';
      showResult({message:'Opening ' + image.name + '…'});
      try {
        const result = await openImage(image);
        if (result.status !== 'done') throw new Error(result.message);
        showResult(result);
      }
      catch (error) { showResult({message:'Could not open this image: ' + error.message}); }
      finally { openingImage = false; }
    };
    const heatmapRenderer = () => selectionTools || (heatmapTools ||= window.createStudioSelection({holder:node('div'), node, get,
      session, runCommand, showResult, setSelectionTool}));
    const showResult = result => {
      notice.replaceChildren(node('p', {text:result.message || ''}));
      if (result.corrections?.length) notice.append(node('p', {class:'muted', text:result.corrections.map(item => item.requested + ' → ' + item.matched).join('; ')}));
      if (result.data?.counts) {
        const table = node('table', {class:'studio-counts'});
        Object.entries(result.data.counts).forEach(([name, count]) => table.append(node('tr', {}, [node('td', {text:name}), node('td', {text:Number(count).toLocaleString()})])));
        notice.append(table);
      }
      if (result.data?.heatmap) notice.append(heatmapRenderer().heatmap(result.data.heatmap));
      if (result.data?.reference_summary) {
        const summary = result.data.reference_summary, details = node('details');
        details.append(node('summary', {text:'Reference cells by phenotype and ROI'}));
        if (summary.note) details.append(node('p', {class:'muted', text:summary.note}));
        const table = node('table', {class:'studio-counts'});
        table.append(node('tr', {}, ['ROI', 'Phenotype', 'Cells'].map(text => node('th', {text}))));
        summary.rows.forEach(row => table.append(node('tr', {}, [node('td', {text:row.roi}), node('td', {text:row.phenotype}), node('td', {text:row.cells.toLocaleString()})])));
        details.append(table); notice.append(details);
      }
      if (result.undo) notice.append(node('button', {class:'btn small', text:'Undo', onclick:async event => {
        event.target.disabled = true;
        try { showResult(await runCommand('undo', {command_id:result.command_id}, {confirmed:true})); }
        catch (error) { showResult({message:error.message}); }
      }}));
      if (outputsPane.hidden) tabButtons.outputs.classList.add('fresh');
    };
    // Run a checked command; its follow-ups (save changes, choose a name) go to *place*: Results, or a chat card.
    const perform = async (command, options = {}, place = null) => {
      const result = await runCommand(command.name, command.args, options);
      const notice_ = place || notice;
      if (place) place.replaceChildren(node('p', {class:'muted', text:result.message || ''}));
      showResult(result);  // Results always shows the last action, wherever it was asked
      const fail = error => (place ? place.replaceChildren(node('p', {class:'hint', text:error.message})) : showResult({message:error.message}));
      if (result.status === 'needs_confirmation') {
        notice_.append(node('p', {text:'This will save changes to your project.'}), node('button', {class:'btn primary', text:'Save changes',
          onclick:() => perform(command, {confirmed:true}, place).catch(fail)}));
      }
      if (result.status === 'needs_choice') {
        const key = {'marker':'markers', 'marker set':'set', 'image':'image', 'phenotype':'phenotype', 'column':'column', 'rule':'rule_name'}[result.argument];
        result.choices.forEach(value => notice_.append(node('button', {class:'btn small', text:value, onclick:() => {
          const args = {...command.args};
          if (result.argument === 'marker' && command.name === 'set_threshold') args.marker = value;
          else if (result.argument === 'marker' && command.name === 'select_cells') args.rule = {...args.rule, conditions:args.rule.conditions.map(condition => condition.marker === result.requested ? {...condition, marker:value} : condition)};
          else if (result.argument === 'phenotype' && command.name === 'filter_cells') args.phenotypes = args.phenotypes.map(item => item === result.requested ? value : item);
          else if (Array.isArray(args[key])) args[key] = args[key].map(item => item === result.requested ? value : item);
          else args[key] = value;
          perform({...command, args}, {}, place).catch(fail);
        }})));
      }
      if (result.data?.job) { await refreshJobs(); if (!options.stay) showTab('jobs'); }
      return result;
    };
    const ask = async () => {
      const text = search.value.trim(); if (!text || askButton.disabled) return;
      askButton.disabled = true; menu.hidden = true; search.setAttribute('aria-expanded', 'false');
      showTab('assistant');
      // The Ask box talks to the assistant: the conversation opens beside the image.
      search.value = '';
      try { await send(text); }
      finally { askButton.disabled = false; }
    };
    const loadImages = async () => {
      if (images) return images;
      if (!imagesRequest) {
        imagesRequest = get('/session/' + encodeURIComponent(session) + '/images')
          .then(value => { images = value.images; return images; })
          .catch(error => { imagesRequest = null; throw error; });
      }
      return imagesRequest;
    };
    const drawImages = entries => {
      list.replaceChildren();
      const query = imageFilter.value.trim().toLowerCase();
      const shown = entries.filter(image => (image.name + ' ' + image.group).toLowerCase().includes(query));
      shown.slice(0, 300).forEach(image => list.append(node('button', {type:'button', class:'studio-image', onclick:() => chooseImage(image)},
        [node('span', {text:image.name}), node('small', {text:image.group})])));
      if (!entries.length) list.append(node('p', {class:'muted', text:context?.has_project ? 'No images found yet.' : 'Load a project in classic SPACEMAP first.'}));
      else if (!shown.length) list.append(node('p', {class:'muted', text:'No image matches.'}));
      if (shown.length > 300) list.append(node('p', {class:'muted', text:'Type more to narrow the list.'}));
    };
    projectMenu.addEventListener('toggle', async () => {
      if (!projectMenu.open) return;
      list.replaceChildren(node('p', {class:'muted', text:'Reading the image list…'}));
      try { drawImages(await loadImages()); } catch (error) { list.textContent = error.message; }
    });
    imageFilter.addEventListener('input', () => { if (images) drawImages(images); });
    document.addEventListener('click', event => { if (projectMenu.open && !projectMenu.contains(event.target)) projectMenu.open = false; });
    const searchChoices = () => {
      const query = search.value.trim().toLocaleLowerCase();
      matches = [
        ...(context?.steps || []).map(step => ({name:step.name, detail:'Step', choose:() => chooseStep(step.id)})),
        ...(images || []).map(image => ({name:image.name, detail:image.group, choose:() => chooseImage(image)}))
      ].filter(item => (item.name + ' ' + item.detail).toLocaleLowerCase().includes(query));
      menu.replaceChildren();
      matches.slice(0, 40).forEach(item => menu.append(node('button', {type:'button', onclick:item.choose},
        [node('span', {text:item.name}), node('small', {text:item.detail})])));
      if (!matches.length) menu.append(node('p', {class:'muted', text:'No matching step or image. Press Enter to ask.'}));
      if (matches.length > 40) menu.append(node('p', {class:'muted', text:'Type more to narrow the list.'}));
      menu.hidden = false; search.setAttribute('aria-expanded', 'true');
    };
    search.addEventListener('focus', async () => {
      searchChoices();
      try { await loadImages(); if (document.activeElement === search) searchChoices(); }
      catch (error) { showResult({message:error.message}); }
    });
    search.addEventListener('input', searchChoices);
    finder.addEventListener('focusout', event => { if (!finder.contains(event.relatedTarget)) { menu.hidden = true; search.setAttribute('aria-expanded', 'false'); } });
    finder.addEventListener('keydown', event => {
      const buttons = Array.from(menu.querySelectorAll('button'));
      if (event.key === 'Escape') { menu.hidden = true; search.setAttribute('aria-expanded', 'false'); event.stopPropagation(); }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const index = buttons.indexOf(document.activeElement), direction = event.key === 'ArrowDown' ? 1 : -1;
        buttons[(index + direction + buttons.length) % buttons.length]?.focus();
      }
      if (event.key === 'Enter' && event.target === search) {
        event.preventDefault();
        if (matches.length === 1) matches[0].choose();
        else if (matches.length) buttons[0]?.focus();
        else ask();
      }
    });
    document.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault(); search.focus(); search.select();
      }
    });

    // ------------------------------------------------------------ project context and step states
    const loadOverview = async () => {
      try {
        const value = await get(route('/overview'));
        steps.querySelectorAll('button').forEach(button => {
          const state = value.states[button.dataset.step] || 'unknown';
          button.querySelector('.studio-dot').dataset.state = state;
          button.title = {done:'Done', partial:'Partly done', none:'Not run'}[state] || '';
        });
      } catch (_) { /* the dots stay grey */ }
    };
    const refresh = async () => {
      try {
        const value = await get(route('/context'));
        const changed = !context || context.project !== value.project || context.project_id !== value.project_id;
        if (changed) { images = null; imagesRequest = null; jobStamp = ''; }
        context = value; projectName.textContent = value.project;
        classic.href = value.steps[0].classic; assistant.href = value.assistant; assistant.textContent = value.assistant_label;
        if (changed) {
          steps.replaceChildren();
          value.steps.forEach((step, index) => steps.append(node('button', {type:'button', 'data-step':step.id, onclick:() => chooseStep(step.id)},
            [node('span', {class:'studio-step-number', text:String(index)}), node('span', {class:'studio-step-name', text:step.name}),
              node('i', {class:'studio-dot', 'data-state':'unknown'})])));
          chooseStep(selected || value.step); refreshJobs();
          if (value.has_project) loadOverview();
        }
      } catch (error) { projectName.textContent = 'Could not connect'; showResult({message:error.message}); }
    };
    refresh();
    // Classic SPACEMAP may have opened another project meanwhile: look again when this window is back in front.
    window.addEventListener('focus', () => refresh());
    let shownImage = null;
    const refreshLive = () => {
      const body = tools.querySelector('.studio-step-body'), scroll = body?.scrollTop;
      const resolve = document.getElementById('studio-resolve'); if (resolve) renderResolve(resolve);
      frame?.thresholdBenchmark?.sync?.();
      const levels = document.getElementById('studio-background'); if (levels) renderBackground(levels, frame?.saved);
      if (document.getElementById('studio-rois')) frame?.drawRois?.();
      frame?.drawShapes?.(); frame?.syncNecks?.(); artifactReview.sync();
      frame?.updatePolygonSave?.(); frame?.updateAnnotationStatus?.();
      if (document.getElementById('studio-compare')) frame?.drawCompare?.();
      const labels = document.getElementById('studio-labels'); if (labels) renderLabels(labels);
      if (body) body.scrollTop = scroll;
      syncColour(frame); frame?.sync?.();
    };
    return {showResult, refreshLive, askCell:async evidence=>{
      if(chatState.busy)throw Error('Wait for the Assistant’s current reply, then ask again.');
      return send(evidence.text,evidence);
    }, imageReady:spec=>{if(currentSpec===spec)frame?.imageReady?.();},
      useImportedTissue:async found=>{
        const valid=()=>currentSpec?.layers?.some(l=>l.type==='image'&&l.token===found.layer.image_token);
        if(!valid())return false;
        await chooseTissue(found.layer);if(!valid())return false;
        remember('annotation-part','tissue');await chooseStep('tab4');if(!valid())return false;
        const shown=await showTissuePreview(found);refreshLive();return shown;
      },renderArtifactControls:artifactReview.renderLayer, shapeStatusChanged:() => {frame?.updatePolygonSave?.(); frame?.updateAnnotationStatus?.();frame?.syncNecks?.();frame?.refreshDifference?.();frame?.tlsReview?.changed?.();},
      phenotyping:() => selected === 'tab7', viewChanged:spec => {
      currentSpec = spec; selectionTools?.update(spec);
      empty.hidden = (spec.layers || []).some(layer => layer.type === 'image');
      document.title = 'SPACEMAP Studio' + (spec.title ? ' · ' + spec.title : '');
      if (spec.id !== shownImage) {
        shownImage = spec.id; if (frame?.colour) loadColours(frame); frame?.readSaved?.(); frame?.refreshMarkers?.();
        showPreview(null); showRaster('classes', null);
        frame?.imageChanged?.();
      }
      refresh(); refreshLive();
    }};
  };
})();
