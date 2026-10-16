"""Read-only browser timing check for an existing project and ROI.

Usage: PYTHONPATH=src python tests/viewer_project_browser.py PROJECT ROI
Requires an existing cell-label pyramid and current Parquet copy. Saves no
project data; the test server rejects every write route except in-memory
Live RESOLVE open/threshold changes. Browser state lives in a temporary folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time


DRIVER = r"""
<script>
const started = performance.now();
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check) {
  for (let i = 0; i < 6000; i++) { if (check()) return; await delay(25); }
  throw new Error('Timed out waiting for viewer: ' + String(check).slice(0, 160));
}
const assert = (value, text) => { if (!value) throw new Error(text); };
const button = text => Array.from(document.querySelectorAll('button')).find(b => b.textContent === text && b.getClientRects().length);
const change = (input, value) => { input.value = value; input.dispatchEvent(new Event('change')); };
window.addEventListener('error', e => fetch('/result?error=' + encodeURIComponent(e.message)));
setTimeout(async () => {
  try {
    const state = window.__state;
    await until(() => state.layers.some(l => l.type === 'image' && l.item && l.inputs && l.item.getFullyLoaded()));
    const image = state.layers.find(l => l.type === 'image');
    const cells = state.layers.find(l => l.type === 'cells');
    const result = {open_image_ms: performance.now() - started};
    await until(() => cells.item && cells.item.getFullyLoaded());
    result.open_cells_ms = performance.now() - started;
    result.open_annotation_requests = performance.getEntriesByType('resource').filter(r => r.name.includes('/annotations/')).length;
    assert(!state.layers.some(l => l.type === 'annotations' && (l.visible || l.open)) ? result.open_annotation_requests === 0 : true,
      'Unselected annotations must stay unloaded');
    // Live RESOLVE needs a saved RESOLVE panel; projects without one skip its timings.
    if (window.hasResolvePanel) {
    await until(() => !!button('Start live RESOLVE'));
    const startLive = performance.now();
    button('Start live RESOLVE').click();
    await until(() => !!cells.live && !!document.querySelector('.resolve-link'));
    result.open_resolve_ms = performance.now() - startLive;
    await delay(250);
    await until(() => cells.item && cells.item.getFullyLoaded() && !(cells.retiring || []).length);
    const channel = image.channels.find(c => cells.live.markers.some(m => m.marker === c.name));
    const marker = () => cells.live.markers.find(m => m.marker === channel.name);
    const original = marker().threshold;
    result.threshold_ms = [];
    result.repaint_ms = [];
    for (let i = 0; i < 5; i++) {
      const item = cells.item;
      const generation = cells.generation || 0;
      const target = original + (i + 1) * 0.125;
      const start = performance.now();
      change(image.inputs[channel.index].low, target);
      await until(() => !cells.live.busy && !cells.live.pending && marker().threshold === target);
      result.threshold_ms.push(performance.now() - start);
      await until(() => cells.item && cells.item !== item && cells.generation > generation &&
        cells.item.getFullyLoaded() && !(cells.retiring || []).length);
      result.repaint_ms.push(performance.now() - start);
    }
    change(image.inputs[channel.index].low, original);
    await until(() => !cells.live.busy && !cells.live.pending && marker().threshold === original);
    image.inputs[channel.index].row.querySelector('.cname').click();
    await until(() => document.querySelectorAll('.hist input[type=number]')[1]?.value);
    const hist = await (await fetch('/spacemap-viewer/api/resolve/' + cells.resolve + '/histogram?marker=' +
      encodeURIComponent(channel.name) + '&transforms=Arcsinh')).json();
    const canvas = document.querySelector('.hist canvas');
    const rect = canvas.getBoundingClientRect();
    // Synthetic pointer events exercise the real handlers; only native pointer
    // capture is replaced because Firefox will not capture an untrusted pointer.
    const capture = canvas.setPointerCapture;
    canvas.setPointerCapture = () => {};
    const dragStart = performance.now();
    canvas.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 1, clientX: rect.left + rect.width * .25}));
    canvas.dispatchEvent(new PointerEvent('pointermove', {pointerId: 1, clientX: rect.left + rect.width * .65}));
    canvas.dispatchEvent(new PointerEvent('pointerup', {pointerId: 1}));
    canvas.setPointerCapture = capture;
    const raw = Math.sinh(hist.axis[0] + .65 * (hist.axis[1] - hist.axis[0])) * 5;
    await until(() => !cells.live.busy && !cells.live.pending && Math.abs(marker().threshold - raw) < 1e-6);
    result.drag_confirmation_ms = performance.now() - dragStart;
    const confirmed = await (await fetch('/spacemap-viewer/api/resolve/' + cells.resolve + '/open', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'
    })).json();
    assert(Math.abs(confirmed.markers.find(m => m.marker === channel.name).threshold - raw) < 1e-6,
      'Dragged threshold must retain the correct raw value');
    change(image.inputs[channel.index].low, original);
    await until(() => !cells.live.busy && !cells.live.pending && marker().threshold === original);
    await delay(250);
    await until(() => cells.item.getFullyLoaded() && !(cells.retiring || []).length);
    } else result.resolve_skipped = 'No saved RESOLVE panel in this project';
    const panStart = performance.now();
    state.viewer.viewport.zoomBy(2);
    state.viewer.viewport.panBy(new OpenSeadragon.Point(0.04, 0.02));
    await delay(250);
    await until(() => image.item.getFullyLoaded() && cells.item.getFullyLoaded());
    result.pan_zoom_ms = performance.now() - panStart;
    assert(document.querySelectorAll('.channel').length > 0, 'Marker controls present after navigation');
    if (window.referenceReview) {
      const reviewStarted = performance.now();
      const response = await fetch('/spacemap-viewer/api/cells/' + cells.token + '/review?t=' +
        encodeURIComponent(cells.table) + '&column=phenotype_ref1&mode=all');
      const reviewed = await response.json();
      assert(response.ok && Array.isArray(reviewed.items), 'Read-only reference review loads');
      result.reference_review_ms = performance.now() - reviewStarted;
      result.reference_review_total = reviewed.total;
      if (reviewed.items.length) {
        const response = await fetch('/spacemap-viewer/api/cells/' + cells.token + '/at?t=' +
          encodeURIComponent(cells.table) + '&label=' + reviewed.items[0].label + '&details_only=1');
        const inspected = await response.json();
        assert(inspected.label === reviewed.items[0].label, 'Review opens the chosen cell');
        change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_ref1');
        await until(() => !!document.querySelector('[aria-label="Save labels in column"]'));
        const choicesResponse = await fetch('/spacemap-viewer/api/cells/' + cells.token + '/reference-options?t=' + encodeURIComponent(cells.table));
        const choices = await choicesResponse.json();
        assert(choicesResponse.ok && choices.columns.length, 'Live reference choices load');
        document.querySelector('[aria-label="Save labels in column"]').click();
        const columnMenu = document.querySelector('[aria-label="column choices"]');
        await until(() => choices.columns.every(c => Array.from(columnMenu.querySelectorAll('button')).some(b => b.textContent === c)));
        assert(!columnMenu.hidden, 'All phenotype columns can be seen despite an existing column name');
        result.reference_column_choices = choices.columns.length;
        document.querySelector('[aria-label="Save labels in column"]').dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}));
        document.querySelector('[aria-label="Phenotype name to assign"]').click();
        const nameMenu = document.querySelector('[aria-label="phenotype choices"]');
        await until(() => choices.phenotypes.every(c => Array.from(nameMenu.querySelectorAll('button')).some(b => b.textContent === c)));
        assert(!nameMenu.hidden, 'Consensus presets can be seen');
        result.consensus_name_choices = choices.phenotypes.length;
        document.querySelector('[aria-label="Phenotype name to assign"]').dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}));
        if (!window.profileProbe) {
          const x = inspected.details?.x_centroid, y = inspected.details?.y_centroid;
          assert(Number.isFinite(x) && Number.isFinite(y), 'The checked cell has a position');
          state.viewer.raiseEvent('canvas-click', {quick: true,
            position: state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x, y))});
          await until(() => state.inspectedCell?.result.profile?.status === 'ready');
          const card = document.querySelector('#cellcard');
          const calls = card.querySelector('h4').nextElementSibling;
          assert(calls.classList.contains('cell-calls') && calls.nextElementSibling.classList.contains('cell-suggestions'),
            'Suggestions sit directly under the cell\'s phenotype calls');
          assert(!card.querySelector('.cell-markers').open, 'Marker bars are folded away');
          assert(card.querySelectorAll('.cell-profile .bar').length > 0, 'Marker bars remain available');
          result.summary_controls_checked = true;
        }
      }
    }
    if (window.profileProbe) {
      const probe = window.profileProbe;
      const clickCell = () => state.viewer.raiseEvent('canvas-click', {quick: true,
        position: state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(probe.x, probe.y))});
      let measuredAt = performance.now();
      clickCell();
      await until(() => state.inspectedCell?.result.profile?.status === 'ready');
      result.saved_profile_ms = performance.now() - measuredAt;
      assert(state.inspectedCell.result.label === probe.label, 'Profile belongs to the clicked cell');
      measuredAt = performance.now();
      change(document.querySelector('#cellcard .profile-source'), 'image');
      const panDuringPreparation = performance.now();
      state.viewer.viewport.panBy(new OpenSeadragon.Point(0.01, 0));
      await delay(250);
      await until(() => image.item.getFullyLoaded());
      result.pan_while_measuring_ms = performance.now() - panDuringPreparation;
      await until(() => state.inspectedCell.result.profile.status !== 'preparing');
      const profile = state.inspectedCell.result.profile;
      assert(profile.status === 'ready', profile.message || 'Image profile failed');
      result.first_image_profile_ms = performance.now() - measuredAt;
      result.profile_cell_pixels = profile.pixels;
      result.profile_markers = profile.markers.length;
      assert(profile.pixels > 0 && profile.markers.length > 0, 'Cell image markers measured');
      measuredAt = performance.now();
      clickCell();
      await until(() => state.inspectedCell?.result.profile?.status === 'ready');
      result.repeat_image_profile_ms = performance.now() - measuredAt;
    }
    if (window.checkSuggestions) {
      change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_ref1');
      change(document.querySelector('#cellcard .profile-source'), 'table');
      const enabled = document.querySelector('[aria-label="Suggest phenotypes"]');
      if (!enabled.checked) enabled.click();
      const start = performance.now();
      await until(() => state.inspectedCell?.result.suggestions && state.inspectedCell.result.suggestions.status !== 'preparing');
      assert(state.inspectedCell.result.suggestions.status === 'ready', state.inspectedCell.result.suggestions.message);
      result.first_table_suggestion_ms = performance.now()-start;
      result.training_examples = state.inspectedCell.result.suggestions.provenance.examples;
      assert(state.inspectedCell.result.suggestions.provenance.method === 'Random Forest', 'Uses the saved Random Forest');
      result.warm_click_suggestions_ms = [];
      for (let i = 0; i < 5; i++) {
        const probe = window.profileProbe;
        const before = state.inspectedCell.result;
        const clickedAt = performance.now();
        state.viewer.raiseEvent('canvas-click', {quick:true,
          position:state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(probe.x, probe.y))});
        await until(() => state.inspectedCell?.result !== before && state.inspectedCell?.result.suggestions?.status === 'ready');
        result.warm_click_suggestions_ms.push(performance.now()-clickedAt);
      }
      result.phenotypes_needing_examples = state.inspectedCell.result.suggestions.provenance.needs_examples.length;
      await until(() => state.inspectedCell?.result.profile?.status === 'ready');
      assert(document.querySelectorAll('#cellcard .cell-profile .bar').length > 0, 'Top marker bars are visible');
      const nextStart = performance.now();
      button('Next least-sure cell').click();
      await until(() => !!cells.suggestionSearch);
      await until(() => !cells.suggestionSearch);
      assert(cells.unsureMessage.includes('Showing the least-sure'), cells.unsureMessage);
      result.first_least_sure_ms = performance.now()-nextStart;
      await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
      const nextAgain = performance.now();
      button('Next least-sure cell').click();
      await until(() => !!cells.suggestionSearch);
      await until(() => !cells.suggestionSearch);
      result.next_least_sure_ms = performance.now()-nextAgain;
      await until(() => state.inspectedCell?.result.profile?.status === 'ready');
      const imageStart = performance.now();
      result.suggestion_backgrounds_defaulted_to_zero = 0;
      image.channels.forEach(c => {
        if (!Number.isFinite(c.low)) { c.low = 0; result.suggestion_backgrounds_defaulted_to_zero++; }
      });
      change(document.querySelector('#cellcard .profile-source'), 'image');
      await until(() => state.inspectedCell?.result.suggestions && state.inspectedCell.result.suggestions.status !== 'preparing');
      assert(state.inspectedCell.result.suggestions.status === 'ready', state.inspectedCell.result.suggestions.message);
      result.suggestions_with_image_bars_ms = performance.now()-imageStart;
      assert(state.inspectedCell.result.suggestions.provenance.source === 'table', 'Image bars do not change saved-model inputs');
    }
    result.image_size = [image.info.width, image.info.height];
    result.slowest_requests = performance.getEntriesByType('resource').filter(r => r.name.includes('/api/'))
      .sort((a, b) => b.duration - a.duration).slice(0, 8)
      .map(r => ({path: new URL(r.name).pathname.replace(/\/(cells|image|resolve)\/[^/]+/g, '/$1/token'),
        ms: r.duration, waiting_ms: r.requestStart - r.startTime, response_ms: r.responseEnd - r.requestStart,
        render_ms: r.serverTiming?.find(t => t.name === 'render')?.duration}));
    const failures = performance.getEntriesByType('resource').filter(r => r.responseStatus >= 400);
    assert(!failures.length, 'Failed browser requests: ' + failures.map(r => r.name).join(', '));
    if (window.panelAt) {
      const card = Array.from(document.querySelectorAll('#panel .layer')).find(c => c.querySelector('.name')?.textContent.includes(window.panelAt));
      assert(card, 'No panel card named ' + window.panelAt);
      card.scrollIntoView({block: 'start'});
    }
    fetch('/result?ok=' + encodeURIComponent(JSON.stringify(result)));
  } catch (e) { fetch('/result?error=' + encodeURIComponent(e.message + '\n' + e.stack)); }
}, 50);
</script>
"""


STUDIO_DRIVER = r"""
<script>
const started=performance.now(),delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<6000;i++){if(check())return;await delay(25);}throw Error('Timed out: '+check);};
const assert=(okay,message)=>{if(!okay)throw Error(message);};
window.addEventListener('error',event=>fetch('/result?error='+encodeURIComponent(event.message)));
setTimeout(async()=>{
 try{
  const state=window.__state;await until(()=>state.imagesReady);const image_ms=performance.now()-started;
  await until(()=>document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');picker.value='reference';picker.dispatchEvent(new Event('change'));
  await until(()=>state.cellsReady && document.querySelector('#studio-select'));
  const cells_ms=performance.now()-started,viewer=state.viewer,image=state.layers.find(item=>item.type==='image');
  const button=(holder,text)=>Array.from(holder.querySelectorAll('button')).find(item=>item.textContent===text);
  assert([...Array.from({length:10},(_,i)=>'tab'+i),'tagging'].every(step=>document.querySelector('#studio-steps [data-step="'+step+'"]')),'All workflow steps and Cell tagging are present');
  assert(document.querySelector('#studio-labels'),'Manual reference is a Phenotyping method');
  // Ask now opens the persistent assistant chat, which this read-only check blocks.
  // Exercise the same checked counting command without creating a conversation.
  state.studio.showResult(await runCommand('count_cells',{}));
  await until(()=>document.querySelector('#studio-action-status').textContent.includes('cells counted'));
  assert(document.querySelector('.studio-counts'),'Counts appear beside the view');
  assert(state.viewer===viewer,'Counting preserves the image');
  const selection=document.querySelector('#studio-select');selection.open=true;
  await until(()=>selection.querySelector('[aria-label="Rule value"]'));
  selection.querySelector('[aria-label="Rule value"]').value='0';
  const selecting=performance.now();button(selection,'Apply selection').click();
  await until(()=>state.spec.selection && !state.commandBusy);const selection_ms=performance.now()-selecting;
  const first=state.spec.selection.count;
  assert(first>=0 && first<=state.spec.selection.total,'Selection count is within the cell total');
  assert(state.viewer===viewer && state.layers.includes(image),'Selection preserves the image');
  const again=performance.now();button(selection,'Apply selection').click();
  await delay(10);await until(()=>!state.commandBusy && state.spec.selection.id!==undefined);
  // Wait for this command to finish before auditing the readonly project.
  await state.commandQueue;
  assert(first===state.spec.selection.count,'Repeated selection returns the same cells');
  const repeat_ms=performance.now()-again;
  assert(!state.layers.some(item=>item.loadFailed),'Image and cell tiles loaded successfully');
  if(window.studioMethod && window.studioStep!=='tab4'){
   picker.value=window.studioMethod;picker.dispatchEvent(new Event('change'));await delay(1500);
   // For a screenshot of the RESOLVE method: start it (in memory only) and open one histogram.
   const start=Array.from(document.querySelectorAll('#studio-resolve button')).find(item=>item.textContent==='Start live RESOLVE');
   if(start){const cells=state.layers.find(item=>item.type==='cells');start.click();await until(()=>cells.live && !cells.live.busy);
    await delay(1500);document.querySelector('#studio-resolve .marker-name')?.click();await delay(2500);}
  }
  if(window.studioStep && window.studioStep!=='tab7'){document.querySelector('[data-step='+window.studioStep+']').click();await delay(3000);}
  if(window.studioStep==='tab5'){
   // Show the image's markers with their N and M toggles.
   await until(()=>document.querySelector('#panel .role-toggle'));
   document.querySelector('#studio-right [data-tab=layers]').click();await delay(800);
  }
  if(window.studioStep==='tab9'){
   // Make a figure (the download is caught) and show it full-page for the screenshot.
   const shown=[];HTMLAnchorElement.prototype.click=function(){shown.push(this.href);};
   const cells=state.layers.find(layer=>layer.type==='cells');
   if(cells){cells.color_by='phenotype_ml_consensus';cells.visible=true;cells.mode='both';}
   const title=document.querySelector('#studio-tools [aria-label="Figure title"]');title.value='P11_roi_1 · phenotype_ml_consensus';title.dispatchEvent(new Event('input'));
   Array.from(document.querySelectorAll('.studio-runbar button')).find(b=>b.textContent==='Save figure').click();
   await until(()=>shown.length);
   const picture=document.createElement('img');picture.src=shown[0];
   picture.style.cssText='position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#222;z-index:99999';
   document.body.appendChild(picture);await delay(1500);
  }
  if(window.studioStep==='tab6'){
   await until(()=>/EMD reduction \d|rror|No combined/.test(document.querySelector('#studio-tools').textContent));
   if(!window.studioScreenshot){
    // Clicking a batch in the maps beside the image opens that ROI.
    // (Only this ROI is registered in the read-only check, so its own row is clicked.)
    const batchRow=document.querySelector('#studio-figures [data-figure="batches-after"] .row[data-row="P11_roi_1"]');
    if(batchRow){batchRow.dispatchEvent(new MouseEvent('click',{bubbles:true}));
     const notice=()=>document.querySelector('#studio-outputs .studio-sec-body')?.textContent||'';
     for(let i=0;i<600 && !/Opening|Could not|No image/.test(notice());i++)await delay(100);
     for(let i=0;i<300 && /Opening/.test(notice());i++)await delay(100);
     window.batch_open=notice().slice(0,200);}
    // Time colouring the cells by a corrected marker.
    performance.setResourceTimingBufferSize(100000);performance.clearResourceTimings();  // tiles fill the default 250 entries
    const first=document.querySelector('#studio-tools .studio-measure:not(.head)'), started=performance.now();
    first.click();
    const coloured=()=>performance.getEntriesByType('resource').some(r=>r.name.includes('/tile') && r.name.includes('_corr') && r.responseEnd>0);
    for(let i=0;i<200 && !coloured();i++)await delay(100);
    window.colour_ms=coloured()?Math.round(performance.now()-started):'not coloured: '+Array.from(document.querySelectorAll('#studio-tools p')).map(p=>p.textContent).slice(-2).join(' | ')+' | cells '+JSON.stringify(state.layers.filter(l=>l.type==='cells').map(l=>({color_by:l.color_by,visible:l.visible,tables:(l.tables||[]).length})));

   }
  }
  const partPicker=window.studioStep==='tab4' && window.studioMethod && document.querySelector('[aria-label="Annotation part"]');
  if(partPicker){
   partPicker.value=window.studioMethod;partPicker.dispatchEvent(new Event('change'));
   await until(()=>/% of the tissue|classes over|kept \(|Add a pool|rror/.test(document.querySelector('#studio-tools').textContent));
   const cut=document.querySelector('#studio-tools [aria-label="Blur cut"]');
   if(cut){cut.value='0.5';cut.dispatchEvent(new Event('input'));await delay(1500);}
   if(window.studioMethod==='zones'){
    // Rules typed in the page only (never saved): a tissue pool and two rings outside the tumour.
    const tools=document.querySelector('#studio-tools');
    await until(()=>Array.from(tools.querySelectorAll('button')).some(b=>b.textContent==='Add a pool'));
    Array.from(tools.querySelectorAll('button')).find(b=>b.textContent==='Add a pool').click();
    const name=tools.querySelector('[aria-label="Pool 1 name"]');name.value='tissue';name.dispatchEvent(new Event('change'));
    tools.querySelectorAll('[aria-label^="Pool 1 has "]').forEach(tick=>tick.click());
    const outside=tools.querySelector('[aria-label="Outside rings of Tumor"]');outside.value='2';outside.dispatchEvent(new Event('change'));
    const inside=tools.querySelector('[aria-label="Inside rings of Tumor"]');inside.value='1';inside.dispatchEvent(new Event('change'));
    const clip=tools.querySelector('[aria-label="Clip outer rings to"]');clip.value='tissue';clip.dispatchEvent(new Event('change'));
    await until(()=>/outer_2/.test(tools.querySelector('.studio-domains')?.textContent||''));
    await delay(1500);
   }
   if(window.studioMethod==='classes'){
    // Names typed in the page only (never saved): the Domains view of 8 classes.
    const tools=document.querySelector('#studio-tools'), count=tools.querySelector('[aria-label="Classes"]');
    count.value='8';count.dispatchEvent(new Event('change'));
    await until(()=>/8 classes over/.test(tools.textContent));
    [[0,'Epithelia'],[1,'Epithelia'],[2,'Stroma'],[3,'Immune']].forEach(([id,name])=>{const input=tools.querySelector('[aria-label="Domain of class '+id+'"]');input.value=name;input.dispatchEvent(new Event('change'));});
    const view=tools.querySelector('[aria-label="Colour by"]');view.value='domains';view.dispatchEvent(new Event('change'));
    await until(()=>tools.querySelectorAll('.studio-domains .studio-class').length===3);
   }
   await delay(1500);
  }
  const tissueTest=!partPicker && window.studioStep==='tab4' && Array.from(document.querySelectorAll('#studio-tools button')).find(b=>b.textContent==='Test on this image');
  if(tissueTest){
   for(const [label,value] of Object.entries(window.tissueSettings || {})){
    const control=document.querySelector('#studio-tools [aria-label="'+label+'"]');
    assert(control,'No tissue setting named '+label);control.value=String(value);control.dispatchEvent(new Event('change'));
   }
   const startPreview=performance.now();
   tissueTest.click();
   await until(()=>!tissueTest.disabled && /% tissue|rror/.test(tissueTest.parentElement.parentElement.textContent));
   assert(/% tissue/.test(tissueTest.parentElement.parentElement.textContent),tissueTest.parentElement.parentElement.textContent);
   const layer=state.layers.find(l=>l.id==='tissue-regions');
   assert(layer?.preview && layer.features.length,'Preview polygons appear in the Tissue layer');
   assert(document.querySelectorAll('#shapes [data-layer="tissue-regions"] path').length===layer.features.length,'Every preview polygon is drawn');
   const countPoints=coords=>typeof coords[0]==='number'?1:coords.reduce((sum,c)=>sum+countPoints(c),0);
   window.tissueResult={preview_ms:performance.now()-startPreview,polygons:layer.features.length,vertices:layer.features.reduce((sum,f)=>sum+countPoints(f.geometry.coordinates),0)};
   const selectAll=document.querySelector('#studio-tools [aria-label="Select all polygons in Tissue"]');
   const selectedAt=performance.now();selectAll.click();
   window.tissueResult.select_all_ms=performance.now()-selectedAt;
   assert(state.picks.ids.size===layer.features.length,'All preview polygons can be selected');
   await delay(1500);
  }
  const swipe=window.studioStep==='tab1' && document.querySelector('#studio-compare [aria-label=Swipe]');
  if(swipe){
   swipe.value='50';swipe.dispatchEvent(new Event('input'));
   const he=state.layers.find(item=>item.id==='he');
   await until(()=>he.item && he.item.getClip());
   assert(Math.abs(he.item.getClip().width/he.item.getContentSize().x-0.5)<0.01,'The swipe shows the H&E on the left half');
   await delay(3000);
  }
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks:9+(window.tissueResult?3:0),open_image_ms:image_ms,cells_ready_ms:cells_ms,selection_ms,repeat_ms,selected:first,total:state.spec.selection.total,colour_ms:window.colour_ms,batch_open:window.batch_open,tissue:window.tissueResult})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-action-status')?.textContent));}
},100);
</script>
"""


# The viewer's parts share their names on the page; this hands the checks what they need.
EXPOSE = "<script>window.__state=state;</script>"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("roi")
    parser.add_argument("--profile", action="store_true", help="Print time spent preparing the label legend.")
    parser.add_argument(
        "--full-view", action="store_true", help="Use the normal image picker, saved markers and all project layers."
    )
    parser.add_argument(
        "--studio", action="store_true", help="Check Studio controls and selection without writing project data."
    )
    parser.add_argument(
        "--cell-profiles",
        action="store_true",
        help="Click a cell and time saved and image measurements; derived indexes stay in scratch storage.",
    )
    parser.add_argument(
        "--reference-review",
        action="store_true",
        help="Read the assigned-cell list without changing labels or review flags.",
    )
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Read-only suggestions and least-sure navigation; use with --cell-profiles.",
    )
    parser.add_argument("--screenshot", type=Path, help="Save a screenshot outside the project after the checks.")
    parser.add_argument("--panel-at", help="Scroll the side panel to the card with this name before the screenshot.")
    parser.add_argument("--size", default="1440,1100", help="Browser window size for the screenshot, as WIDTH,HEIGHT.")
    parser.add_argument("--step", default="tab7", help="Studio step to show at the end (with --studio).")
    parser.add_argument("--tissue-settings", type=json.loads, help="JSON mapping of tissue control labels to temporary values for the read-only preview.")
    parser.add_argument("--method", help="Phenotyping method to show at the end (with --studio), e.g. resolve.")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="A folder outside the project with Tab 4 files (valid tissue, domains) for Studio to read instead.",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    if args.screenshot and args.screenshot.resolve().is_relative_to(project):
        parser.error("Save the screenshot outside the read-only project.")
    if args.annotations and args.annotations.resolve().is_relative_to(project):
        parser.error("Keep the annotation folder outside the read-only project.")
    config = json.loads((project / f"{project.name}.json").read_text())
    with tempfile.TemporaryDirectory(prefix="spacemap-project-browser-") as folder:
        root = Path(folder)
        os.environ.update(
            SPACEMAP_HOME=str(root / "state"), MPLCONFIGDIR=str(root / "matplotlib"), SPACEMAP_FAST_COPY="on"
        )
        from starlette.applications import Starlette
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import HTMLResponse, Response
        from starlette.routing import Mount, Route
        import uvicorn

        from spacemap import dataio, labels as label_store
        from spacemap.viewer import reference
        from spacemap_gui.viewer import api, registry
        from spacemap_gui.viewer.component import FRONTEND_DIR

        if args.profile:
            import cProfile
            import pstats

            original_lut = api._cell_lut

            def profiled_lut(*values, **options):
                profiler = cProfile.Profile()
                with profiler:
                    result = original_lut(*values, **options)
                pstats.Stats(profiler).sort_stats("cumulative").print_stats(18)
                return result

            api._cell_lut = profiled_lut

        if args.annotations:
            from spacemap_gui.studio import annotation as studio_annotation

            tab4_folders = studio_annotation.folders
            studio_annotation.folders = lambda root, mappings: (
                tab4_folders(root, mappings)[0],
                args.annotations.resolve(),
            )
        image = project / "outputs/rois" / f"{args.roi}.ome.tiff"
        pyramid = project / "outputs/viewer_cache/cell_labels" / f"{args.roi}.ome_cell_boundaries.cell_labels.tif"
        # Newer projects leave unified_csv_filename empty and name the table in Tab 7.
        table_name = config.get("unified_csv_filename") or (
            config.get("tab_configurations", {}).get("tab7_phenotyping", {}).get("global_pheno_file")
        )
        table = project / "outputs/batch_corrected" / table_name
        cache = project / "outputs/phenotyping/cache"
        assert image.is_file() and pyramid.is_file(), "Prepare the image and cell outlines in SPACEMAP first."
        assert dataio.fast_copy_info(table)["current"], "Prepare a current fast table copy in SPACEMAP first."

        # A read must never start writing a fast copy behind the test's back.
        def no_build(*args, **kwargs):
            raise AssertionError("Read-only check cannot build a table copy.")

        dataio.start_fast_copy = no_build

        # Block the storage functions too, so a future read-route regression
        # cannot write research labels even if it bypasses the HTTP guard.
        def no_labels(*args, **kwargs):
            raise AssertionError("Read-only check cannot change saved labels.")

        label_store.append = no_labels
        label_store.clear = no_labels
        label_store.commit = no_labels
        reference.save = no_labels
        reference.set_flag = no_labels
        watched = [
            table,
            dataio.fast_copy_path(table),
            cache / "resolve_pheno_panel.json",
            cache / "resolve_thresholds_table.csv",
            Path(str(table) + ".labels.parquet"),
            reference.path_for(table),
        ]
        stamps = {p: (p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in watched}
        image_token = registry.register(image)
        cell_token = registry.register_object(
            "cells", image=str(image), pyramid=str(pyramid), roi=args.roi, tables={"t0": str(table)}, colors={}
        )
        resolve = registry.register_object(
            "resolve",
            cells=cell_token,
            table=str(table),
            roi=args.roi,
            panel=str(cache / "resolve_pheno_panel.json"),
            thresholds=str(cache / "resolve_thresholds_table.csv"),
            settings=str(cache / "resolve_params.json"),
        )
        layers = [
            {
                "id": "image",
                "type": "image",
                "token": image_token,
                "name": args.roi,
                "visible": True,
                "opacity": 1,
                "open": True,
            },
            {
                "id": "cells",
                "type": "cells",
                "token": cell_token,
                "image_token": image_token,
                "resolve": resolve,
                "name": "Cells",
                "visible": True,
                "opacity": 1,
                "open": True,
                "tables": [{"id": "t0", "name": "Cells"}],
                "table": "t0",
            },
        ]
        temporary_display_tokens = set()
        if args.full_view:
            from spacemap.viewer import cells
            from spacemap_gui.viewer import ui

            original_prepare = cells.prepare_label_pyramid

            def existing_pyramid(path, width, height, cache_dir):
                target = cells.label_pyramid_path(cache_dir, path)
                wanted = {"source": str(path), "fingerprint": cells.fingerprint(path), "width": width, "height": height}
                assert cells._up_to_date(target, wanted), "Prepare current cell outlines before this read-only check."
                return target, True

            cells.prepare_label_pyramid = existing_pyramid
            start_layers = time.perf_counter()
            setup_profiler = cProfile.Profile() if args.profile else None
            try:
                if setup_profiler:
                    setup_profiler.enable()
                with ui.using_context(
                    {
                        "root": str(project),
                        "mappings": config.get("directory_mappings", {}),
                        "unified": str(table),
                        "colors": {},
                    }
                ):
                    layers, _ = ui.layers_for_image(
                        {"group": "ROIs", "name": image.name, "path": str(image), "sample": args.roi}, context="tab0"
                    )
            finally:
                if setup_profiler:
                    setup_profiler.disable()
                    pstats.Stats(setup_profiler).sort_stats("cumulative").print_stats(15)
                cells.prepare_label_pyramid = original_prepare
            print(
                json.dumps(
                    {
                        "prepare_layers_ms": (time.perf_counter() - start_layers) * 1000,
                        "annotation_layers": sum(l["type"] == "annotations" for l in layers),
                    }
                ),
                flush=True,
            )
            for layer in layers:
                if layer["type"] in ("image", "cells"):
                    layer["open"] = True
                for token_key in ("token", "display_token"):
                    kind = "display" if token_key == "display_token" else layer["type"]
                    obj = registry.resolve_object(layer.get(token_key, ""), kind)
                    if obj and obj.get("path"):
                        path = Path(obj["path"])
                        if path.exists():
                            stamps[path] = (path.stat().st_size, path.stat().st_mtime_ns)
                        if kind == "display":
                            copy = root / (layer["id"] + "-display.json")
                            if path.exists():
                                copy.write_bytes(path.read_bytes())
                            layer["display_token"] = registry.register_object(
                                "display", path=str(copy), image=str(image)
                            )
                            temporary_display_tokens.add(layer["display_token"])
            # The real view offers Live RESOLVE only when the project has a RESOLVE panel.
            resolve = next((l.get("resolve") for l in layers if l["type"] == "cells"), None)
        registry.publish(
            "project-check",
            {"id": args.roi, "title": "Read-only viewer check", "layers": layers},
            project={
                "root": str(project),
                "mappings": config.get("directory_mappings", {}),
                "unified": str(table),
                "tab": "tab7",
                "read_only": True,
            }
            if args.studio
            else None,
        )
        probe = None
        if args.cell_profiles:
            from spacemap.viewer import cells

            layer = registry.resolve_object(next(l["token"] for l in layers if l["type"] == "cells"), "cells")
            _, labels = cells.roi_index(str(table), args.roi)
            for label in labels[:20]:
                row = cells.cell_details(str(table), args.roi, int(label))
                x, y = row["x_centroid"], row["y_centroid"]
                if api._label_at(layer, x, y) == int(label):
                    probe = {"label": int(label), "x": x, "y": y}
                    break
            assert probe, "No cell centre found for the read-only click check."
        done = threading.Event()
        results = []

        async def page(request):
            shim = "<script>window.requestAnimationFrame=cb=>setTimeout(()=>cb(performance.now()),16);window.cancelAnimationFrame=clearTimeout;</script>"
            html = (FRONTEND_DIR / "index.html").read_text().replace("<head>", "<head>" + shim)
            probe_script = (
                "<script>window.profileProbe="
                + json.dumps(probe)
                + ";window.referenceReview="
                + json.dumps(args.reference_review)
                + ";window.checkSuggestions="
                + json.dumps(args.suggestions)
                + ";window.panelAt="
                + json.dumps(args.panel_at)
                + ";window.studioMethod="
                + json.dumps(args.method)
                + ";window.studioStep="
                + json.dumps(args.step)
                + ";window.tissueSettings="
                + json.dumps(args.tissue_settings)
                + ";window.studioScreenshot="
                + json.dumps(bool(args.screenshot))
                + ";window.hasResolvePanel="
                + json.dumps((cache / "resolve_pheno_panel.json").is_file())
                + ";</script>"
            )
            return HTMLResponse(
                html.replace(
                    "</body>",
                    EXPOSE + probe_script + (STUDIO_DRIVER if args.studio else DRIVER) + '<img src="/slow" hidden></body>',
                )
            )

        async def result(request):
            results.append(dict(request.query_params))
            done.set()
            return Response()

        async def slow(request):
            if args.screenshot:
                await asyncio.to_thread(done.wait, 300)
            else:
                await asyncio.sleep(300)
            return Response(b"", media_type="image/gif")

        app = Starlette(
            routes=[
                Route(api.MOUNT_PATH + "/window/index.html", page),
                Route("/result", result),
                Route("/slow", slow),
                Mount(api.MOUNT_PATH, api.app),
            ]
        )

        async def read_only(request, call_next):
            # Changes to marker display preferences land in the temporary copy.
            display_save = request.url.path in {
                f"{api.MOUNT_PATH}/display/{token}/save" for token in temporary_display_tokens
            }
            measure_cell = (
                (args.cell_profiles or args.reference_review)
                and request.url.path
                == f"{api.MOUNT_PATH}/cells/{next(l['token'] for l in layers if l['type'] == 'cells')}/profile"
            )
            suggest = (
                request.url.path
                == f"{api.MOUNT_PATH}/cells/{next(l['token'] for l in layers if l['type'] == 'cells')}/suggestions"
            )
            # Studio previews read the image and save nothing.
            preview = args.studio and request.url.path in {
                f"{api.MOUNT_PATH}/studio/project-check/tissue_preview",
                f"{api.MOUNT_PATH}/studio/project-check/segment_preview",
                f"{api.MOUNT_PATH}/studio/project-check/artifact_preview",
                f"{api.MOUNT_PATH}/studio/project-check/artifact_example",
                f"{api.MOUNT_PATH}/studio/project-check/pixel_classes",
                f"{api.MOUNT_PATH}/studio/project-check/pieces_preview",
                f"{api.MOUNT_PATH}/studio/project-check/zones_preview",
            }
            studio_command = args.studio and request.url.path in {
                f"{api.MOUNT_PATH}/studio/project-check/command",
                f"{api.MOUNT_PATH}/studio/project-check/ask",
            }
            if (
                request.method != "GET"
                and request.url.path
                not in (f"{api.MOUNT_PATH}/resolve/{resolve}/open", f"{api.MOUNT_PATH}/resolve/{resolve}/thresholds")
                and not display_save
                and not measure_cell
                and not suggest
                and not studio_command
                and not preview
            ):
                return Response("Read-only browser test", status_code=403)
            started = time.perf_counter()
            response = await call_next(request)
            if args.profile and time.perf_counter() - started > 1:
                print("Slow request:", request.url.path, round(time.perf_counter() - started, 3), flush=True)
            return response

        app.add_middleware(BaseHTTPMiddleware, dispatch=read_only)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
        profile = root / "firefox"
        profile.mkdir()
        (profile / "user.js").write_text(
            'user_pref("network.proxy.type", 0);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n'
        )
        if args.studio and args.step == "tab6":
            # The batch check reads a sample of the whole table once (about 15 s); do it before the page asks.
            from spacemap_gui.studio import batches

            batches.check(str(table), config)
        with (root / "firefox.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "firefox",
                    "--headless",
                    "--no-remote",
                    "--profile",
                    str(profile),
                    "--window-size",
                    args.size,
                    "--screenshot",
                    str(root / "viewer.png"),
                    f"http://127.0.0.1:{port}{api.MOUNT_PATH}/window/index.html?session=project-check"
                    + ("&studio=1&step=tab7" if args.studio else ""),
                ],
                stdout=log,
                stderr=log,
            )
            try:
                started = time.monotonic()
                if not done.wait(300):
                    raise AssertionError("Browser timed out: " + (root / "firefox.log").read_text())
                assert results and "ok" in results[0], results
                if args.screenshot:
                    import shutil

                    process.wait(timeout=30)
                    shutil.copyfile(root / "viewer.png", args.screenshot)
                print(json.dumps({"browser_checks": json.loads(results[0]["ok"])}), flush=True)
                for path, stamp in stamps.items():
                    current = (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None
                    assert current == stamp, f"Changed during check: {path}"
                print(
                    json.dumps(
                        {
                            "project": project.name,
                            "roi": args.roi,
                            "seconds": time.monotonic() - started,
                            "result": json.loads(results[0]["ok"]),
                            "project_files_unchanged": True,
                        }
                    ),
                    flush=True,
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                server.should_exit = True


if __name__ == "__main__":
    main()
