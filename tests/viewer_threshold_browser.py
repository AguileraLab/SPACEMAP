"""Run real viewer controls in headless Firefox against a disposable project.

Usage: PYTHONPATH=src python tests/viewer_threshold_browser.py
Requires Firefox; no browser-driver package or research data is used.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading


DRIVER = r"""
<script>
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check) {
  for (let i = 0; i < 300; i++) { if (check()) return; await delay(100); }
  throw new Error('Timed out: ' + String(check) + '; ' + (document.querySelector('#cellcard')?.textContent || ''));
}
let checks = 0;
let lastCheck = 'start';
const assert = (value, text) => { if (!value) throw new Error(text); checks++; lastCheck = text; };
const button = text => Array.from(document.querySelectorAll('button')).find(b => b.textContent === text && b.getClientRects().length);
const change = (input, value) => { input.value = value; input.dispatchEvent(new Event('change')); };
// Headless Firefox cannot answer dialogs; record each question and answer yes.
const confirms = [];
window.confirm = text => { confirms.push(text); return true; };
window.addEventListener('error', e => fetch('/result?error=' + encodeURIComponent(String(e.message) + ' (after: ' + lastCheck + ') | ' + (e.error?.stack || ''))));
setTimeout(async () => {
  try {
    const state = window.__state;
    await until(() => state.layers.some(l => l.type === 'image' && l.inputs));
    const image = state.layers.find(l => l.type === 'image');
    const cells = state.layers.find(l => l.type === 'cells');
    const channel = image.channels[0];
    const original = channel.low;
    const selects = () => document.querySelectorAll('.auto-controls select');
    assert(selects()[0].value === 'Triangle', 'Triangle is default');
    assert(selects()[0].options.length === 10, 'All ten shared methods available');
    await until(() => image.item && image.item.getFullyLoaded());
    assert(!performance.getEntriesByType('resource').some(r => r.name.includes('/annotations/')),
      'Hidden annotations do not delay image opening');
    await until(() => cells.item && cells.item.getFullyLoaded());
    const opened = performance.getEntriesByType('resource');
    const imageTiles = opened.filter(r => /\/image\/.*\/tile\//.test(r.name));
    const cellTiles = opened.filter(r => /\/cells\/.*\/tile\//.test(r.name));
    assert(imageTiles.length && cellTiles.length && Math.min(...cellTiles.map(r => r.startTime)) >=
      Math.max(...imageTiles.map(r => r.responseEnd)), 'Image tiles finish before cells start');
    change(selects()[1], Array.from(selects()[1].options).find(o => o.textContent.endsWith(': all areas')).value);
    await until(() => Array.from(selects()[1].options).some(o => o.textContent.endsWith(': tissue')));
    const annotation = state.layers.find(l => l.type === 'annotations');
    const card = Array.from(document.querySelectorAll('.layer')).find(c => c.querySelector('.name').textContent === annotation.name);
    card.querySelector('.layer-head').click();
    await until(() => card.querySelector('.legend .item'));
    assert(performance.getEntriesByType('resource').filter(r => /\/annotations\/.*\/legend/.test(r.name)).length === 1,
      'Annotation settings and Auto share one class request');
    change(selects()[0], 'Otsu');
    change(selects()[1], Array.from(selects()[1].options).find(o => o.textContent.endsWith(': tissue')).value);
    image.inputs[0].row.querySelector('button').click();
    await until(() => !image.autoRun);
    assert(channel.low !== original, 'Auto changed minimum');
    assert(image.autoMessage.includes('pixels'), 'Measurement provenance visible');
    button('Undo Auto').click();
    assert(channel.low === original, 'Undo restored original minimum');
    button('Auto all markers').click();
    await until(() => !image.autoRun);
    assert(image.autoUndo.length === 2, 'Auto all included hidden marker');
    button('Undo Auto').click();
    image.channels[1].low = undefined; image.channels[1].high = undefined;
    document.querySelector('[data-layer-group=cells]:not([open]) > summary')?.click();  // Cells and maps start folded
    document.querySelector('[data-layer-id=cells].collapsed .layer-head .name')?.click();  // and its card starts compact
    button('Start live RESOLVE').click();
    await until(() => !!cells.live && !!document.querySelector('.resolve-link'));
    assert(channel.low === 50, 'Starting RESOLVE loads saved raw threshold');
    assert(image.channels[1].high === undefined, 'Hidden marker brightness stays unset until needed');
    const beforeUpper=cells.live.version, beforeUpperGates=cells.live.markers.map(m=>m.threshold), beforeUpperLows=image.channels.map(c=>c.low);
    button('Auto upper').click();
    await until(()=>!image.autoRun);await delay(200);
    assert(cells.live.version===beforeUpper&&cells.live.markers.every((m,i)=>m.threshold===beforeUpperGates[i])&&image.channels.every((c,i)=>c.low===beforeUpperLows[i]),'Auto upper does not update linked RESOLVE thresholds or version');
    button('Undo Auto').click();await delay(200);
    assert(cells.live.version===beforeUpper&&image.channels[1].high===undefined,'Undo upper does not write cell gates and restores an unset hidden maximum');
    button('Auto all markers').click();
    await until(() => !image.autoRun && !cells.live.busy && !cells.live.pending);
    button('Undo Auto').click();
    await until(() => !cells.live.busy && !cells.live.pending);
    assert(image.channels[1].low === 50 && cells.live.markers.find(m => m.marker === 'M1').threshold === 50,
      'Undo restores hidden marker minimum and linked threshold');
    assert(image.channels[1].high === undefined, 'Undo keeps an unmeasured hidden maximum unset');
    image.inputs[1].row.querySelector('input[type=checkbox]').click();
    await until(() => Number.isFinite(image.channels[1].high));
    assert(image.channels[1].low === 50, 'Showing a marker preserves its linked threshold');
    assert(image.channels[1].high > 51, 'Showing a marker measures its brightness rather than using threshold plus one');
    image.inputs[1].row.querySelector('input[type=checkbox]').click();
    const start = performance.now();
    change(image.inputs[0].low, 123.375);
    await until(() => !cells.live.busy && !cells.live.pending);
    const elapsed = performance.now() - start;
    const marker = () => cells.live.markers.find(m => m.marker === channel.name);
    assert(marker().threshold === 123.375, 'Marker min linked precisely');
    const confirmed = await (await fetch('/spacemap-viewer/api/resolve/' + cells.resolve + '/open', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'
    })).json();
    assert(confirmed.markers.find(m => m.marker === channel.name).threshold === 123.375, 'Server threshold matches');
    const version = cells.live.version;
    change(image.inputs[0].high, 700);
    await delay(200);
    assert(cells.live.version === version, 'Max does not change RESOLVE');
    image.inputs[0].row.querySelector('.resolve-link input').click();
    change(image.inputs[0].low, 20);
    assert(marker().threshold === 123.375, 'Unlink isolates contrast');
    image.inputs[0].row.querySelector('.cname').click();
    await until(() => document.querySelector('.hist input[type=number]'));
    await delay(400);
    const inputs = document.querySelectorAll('.hist input[type=number]');
    change(inputs[1], Math.asinh(97.25 / 5));
    await until(() => Math.abs(marker().threshold - 97.25) < 1e-8);
    assert(channel.low === 20, 'Histogram edit respects unlink');
    image.inputs[0].row.querySelector('.resolve-link input').click();
    assert(Math.abs(channel.low - 97.25) < 1e-8, 'Relink uses the confirmed raw threshold');
    button('Auto all markers').click();
    button('Cancel').click();
    await until(() => !image.autoRun);
    assert(image.autoMessage.startsWith('Cancelled'), 'Cancel visible');
    await delay(800);
    const saved = await (await fetch('/spacemap-viewer/api/display/' + image.display_token)).json();
    assert(saved.auto.method === 'Otsu' && saved.auto.scope.includes('tissue'), 'Preferences saved');
    document.querySelector('.phenotype-editor summary').click();
    await until(() => document.querySelector('[aria-label="Phenotype name"]'));
    const phenotypeName = document.querySelector('[aria-label="Phenotype name"]');
    phenotypeName.value = 'Renamed A'; phenotypeName.dispatchEvent(new Event('input'));
    button('Save panel').click();
    await until(() => cells.live.phenotypes.includes('Renamed A'));
    assert(Math.abs(marker().threshold - 97.25) < 1e-8, 'Panel save preserves edited thresholds');
    const clickCell = () => state.viewer.raiseEvent('canvas-click', {quick: true,
      position: state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(15, 15))});
    clickCell();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready' && state.inspectedCell?.result.why);
    assert(document.querySelector('#cellcard').textContent.includes('Live RESOLVE:'),
      'The RESOLVE explanation remains alongside the cell measurement');
    change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_ref1');
    await until(() => document.querySelector('.reference-counts button'));
    assert(cells.color_by === 'phenotype_ref1' && !cells.live, 'Reference selection displays saved labels');
    document.querySelector('.reference-counts button').click();
    assert(!!state.labelName, 'Saved phenotype can be selected for assignment');
    clickCell();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(state.inspectedCell.result.label === 1 && state.inspectedCell.result.profile.markers[0].mean === 1,
      'Clicked cell shows its own saved measurement');
    assert(document.querySelectorAll('#cellcard .cell-profile .bar').length === 2,
      'Clicked cells show the blue marker bars');
    const markerStep = text => Array.from(document.querySelectorAll('.marker-transforms button')).find(b => b.textContent === text);
    assert(markerStep('Raw').getAttribute('aria-pressed') === 'true', 'Marker bars start in Raw');
    markerStep('Z2').click(); markerStep('Z1').click(); markerStep('Arcsinh').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(state.inspectedCell.result.profile.steps.join(',') === 'Arcsinh,Z1,Z2',
      'Clicked transformations combine in the fixed order, regardless of click order');
    assert(document.querySelector('.marker-transform-order').textContent === '1. Arcsinh → 2. Z1 → 3. Z2',
      'The active transformation order is shown');
    markerStep('CDF').click(); markerStep('NLog').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(state.inspectedCell.result.profile.steps.length === 5 && document.querySelectorAll('#cellcard .cell-profile .bar').length === 2,
      'The complete chain keeps the marker bars');
    assert(state.inspectedCell.result.profile.markers[0].mean === 1 && Number.isFinite(state.inspectedCell.result.profile.markers[0].view),
      'Transformed views preserve the original measurement');
    markerStep('Raw').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(!state.inspectedCell.result.profile.steps?.length && cells.markerTransforms.length === 0 &&
      document.querySelectorAll('#cellcard .marker-transforms [aria-pressed=true]').length === 1, 'Raw clears every other transformation');
    change(document.querySelector('[aria-label="Cell measurements from"]'), 'image');
    markerStep('Arcsinh').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(state.inspectedCell.result.profile.source === 'image' && state.inspectedCell.result.profile.steps[0] === 'Arcsinh',
      'Arcsinh can display the clicked cell image signal');
    markerStep('Z1').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    assert(state.inspectedCell.result.profile.source === 'table' && document.querySelector('#cellcard .profile-source').value === 'table',
      'Z1 visibly selects saved measurements for its ROI comparison');
    markerStep('Raw').click();
    await until(() => state.inspectedCell?.result.profile?.status === 'ready');
    await until(() => Array.from(document.querySelectorAll('#label-names option')).some(o => o.value === 'Consensus T'));
    assert(Array.from(document.querySelectorAll('#label-columns option')).some(o => o.value === 'phenotype_training'),
      'Save column offers phenotype-prefixed columns');
    assert(Array.from(document.querySelectorAll('#label-names option')).some(o => o.value === 'Consensus B'),
      'Preset names include consensus phenotypes from another loaded table');
    document.querySelector('[aria-label="Save labels in column"]').click();
    const columnMenu = () => document.querySelector('[aria-label="column choices"]');
    assert(!columnMenu().hidden && Array.from(columnMenu().querySelectorAll('button')).some(b => b.textContent === 'phenotype_training'),
      'Clicking Column shows other columns even when the input already contains a full column name');
    Array.from(columnMenu().querySelectorAll('button')).find(b => b.textContent === 'phenotype_training').click();
    await until(() => cells.color_by === 'phenotype_training' && document.querySelector('.reference-counts button'));
    assert(state.labelColumn === 'phenotype_training', 'Choosing a column loads its existing cell labels');
    assert(confirms.length === 1 && confirms[0].includes('"phenotype_training" holds another method'),
      'Labelling into a method column that is not a reference column asks first');
    change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_ref1');
    await until(() => document.querySelector('.reference-counts button'));
    const typedName = document.querySelector('[aria-label="Phenotype name to assign"]');
    typedName.value = 'New name with no match'; typedName.dispatchEvent(new Event('input'));
    document.querySelector('[aria-label="Phenotype name to assign"]').click();
    const phenotypeMenu = document.querySelector('[aria-label="phenotype choices"]');
    assert(!phenotypeMenu.hidden && Array.from(phenotypeMenu.querySelectorAll('button')).some(b => b.textContent === 'Consensus T'),
      'Clicking Phenotype shows all presets regardless of the typed text');
    Array.from(phenotypeMenu.querySelectorAll('button')).find(b => b.textContent === 'Consensus T').click();
    assert(state.labelName === 'Consensus T' && typedName.value === 'Consensus T', 'A preset can be selected for assignment');
    assert(document.querySelector('#cellcard > .cell-suggestions') && document.querySelector('#cellcard > .cell-calls') &&
      !document.querySelector('#cellcard .cell-markers').open,
      'Phenotype calls and suggestions have their own visible sections while marker bars start folded away');
    assert(!button('Choose') && Array.from(document.querySelectorAll('.lbl')).some(n => n.textContent === 'Phenotype'),
      'Column and Phenotype open directly with no Choose button');
    typedName.focus(); typedName.value = 'Consensus'; typedName.dispatchEvent(new Event('input'));
    assert(phenotypeMenu.querySelectorAll('[role=option]').length === 2, 'Typing filters the phenotype list');
    typedName.dispatchEvent(new KeyboardEvent('keydown', {key:'ArrowDown', bubbles:true}));
    typedName.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    assert(state.labelName === 'Consensus B' && phenotypeMenu.hidden, 'Keyboard selection accepts a phenotype and closes the list');
    typedName.click(); typedName.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}));
    assert(phenotypeMenu.hidden && typedName.getAttribute('aria-expanded') === 'false', 'Escape closes the dropdown');
    typedName.value = 'New typed phenotype'; typedName.dispatchEvent(new Event('input'));
    typedName.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    assert(state.labelName === 'New typed phenotype', 'New phenotype names remain allowed');
    const realModelFetch = window.fetch;
    let reviewReads = 0, pendingProfiles = 0;
    window.fetch = async (url, options) => {
      if (String(url).includes('/review?') && String(url).includes('&label=')) reviewReads++;
      if (String(url).includes('/profile')) {
        pendingProfiles++;
        await delay(2500);
        pendingProfiles--;
      }
      return realModelFetch(url, options);
    };
    change(document.querySelector('[aria-label="Cell measurements from"]'), 'image');
    button('Show suggestions').click();
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(state.inspectedCell.result.suggestions.provenance.method === 'Random Forest' &&
      state.inspectedCell.result.suggestions.provenance.column === 'phenotype_ml_consensus',
      'The default model uses saved consensus labels');
    assert(pendingProfiles > 0 && state.inspectedCell.result.profile.status === 'preparing',
      'Saved-model suggestions arrive without waiting for image measurements');
    const stableReview = button('Mark needs review');
    stableReview.focus();
    await until(() => state.inspectedCell?.result.profile.status === 'ready');
    assert(button('Mark needs review') === stableReview && document.activeElement === stableReview && reviewReads === 0,
      'Review button and focus survive repeated profile/suggestion refreshes without fetching the flag again');
    window.fetch = realModelFetch;
    change(document.querySelector('[aria-label="Cell measurements from"]'), 'table');
    const suggestToggle = document.querySelector('[aria-label="Suggest phenotypes"]');
    suggestToggle.checked = false; suggestToggle.dispatchEvent(new Event('change'));
    const realChoiceFetch = window.fetch;
    window.fetch = (url, options) => String(url).includes('/reference-options?') ?
      Promise.resolve(new Response('Not Found', {status:404})) : realChoiceFetch(url, options);
    cells.referenceOptionsRequest = null;
    change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_ref1');
    await until(() => !!button('Retry choices'));
    assert(document.querySelector('#panel').textContent.includes('extra choices could not be loaded') &&
      !document.querySelector('#panel').textContent.includes('HTTP 404'), 'A missing choices route explains recovery in plain words');
    document.querySelector('[aria-label="Save labels in column"]').click();
    assert(Array.from(columnMenu().querySelectorAll('button')).some(b => b.textContent === 'phenotype_training'),
      'Column choices remain available when the extra-name request fails');
    document.querySelector('[aria-label="Save labels in column"]').click();
    window.fetch = realChoiceFetch;
    button('Retry choices').click();
    await until(() => Array.from(document.querySelectorAll('#label-names option')).some(o => o.value === 'Consensus T'));
    assert(!button('Retry choices'), 'Retry recovers the extra names without rebuilding the viewer');
    change(document.querySelector('#cellcard .profile-source'), 'image');
    await until(() => state.inspectedCell?.result.profile.status === 'ready');
    const measured = state.inspectedCell.result.profile;
    assert(measured.source === 'image' && measured.pixels === 400, 'Image measurement follows the full cell outline');
    assert(measured.markers[0].signal >= 0 && measured.markers[0].background === image.channels[0].low,
      'Image measurement uses the current marker background');
    change(document.querySelector('#cellcard .profile-source'), 'corrected');
    await until(() => state.inspectedCell?.result.profile.status === 'ready');
    const corrected = state.inspectedCell.result.profile;
    assert(corrected.source === 'corrected' && corrected.markers.every(m => m.signal === m.mean && m.background === 0),
      'Background-subtracted image is not subtracted twice');
    change(document.querySelector('#cellcard .profile-source'), 'table');
    await until(() => state.inspectedCell?.result.profile.status === 'ready');
    assert(state.inspectedCell.result.profile.source === 'table', 'Measurement source switches back to the table');
    change(document.querySelector('#cellcard .profile-source'), 'image');
    change(document.querySelector('#cellcard .profile-source'), 'table');
    await until(() => state.inspectedCell?.result.profile.status === 'ready');
    await delay(200);
    assert(state.inspectedCell.result.profile.source === 'table', 'Older measurements cannot replace the newest choice');
    const nameInput = document.querySelector('[aria-label="Phenotype name to assign"]');
    nameInput.value = 'New T'; nameInput.dispatchEvent(new Event('input'));
    button('Click cells').click();
    const assignmentName = document.querySelector('[aria-label="Phenotype name to assign"]');
    const clickAt = (x,y) => state.viewer.raiseEvent('canvas-click', {quick:true,
      position: state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y))});
    const clicksAt = performance.now();
    clickAt(15,15); clickAt(55,15);
    const feedbackMs = performance.now() - clicksAt;
    assert(window.__labelSaves.queue.length === 1 && window.__labelSaves.queue[0].where.points.length === 2,
      'Rapid clicks are grouped for background saving');
    assert(document.querySelectorAll('.label-preview circle').length === 2, 'Queued clicks show immediately on the image');
    assert(Object.keys(localStorage).some(k => k.startsWith('spacemap-label-draft-')), 'Waiting labels have a recovery draft');
    button('Save now').click();
    await until(() => !window.__labelSaves.busy && !window.__labelSaves.queue.length);
    const reviewURL = '/spacemap-viewer/api/cells/' + cells.token + '/review?t=t0&column=phenotype_ref1';
    let recent = await (await fetch(reviewURL)).json();
    assert(recent.items.length === 2 && recent.items.every(i => i.value === 'New T'), 'Both assigned cells are saved and listed');
    await delay(1000);
    assert(document.querySelector('[aria-label="Phenotype name to assign"]') === assignmentName,
      'Saving does not rebuild the name input');
    button('Undo').click();
    await until(() => !window.__labelSaves.busy && !window.__labelSaves.queue.length);
    recent = await (await fetch(reviewURL)).json();
    assert(recent.items.every(i => i.value === 'A'), 'Undo restores the whole saved group');
    clickAt(95,15); button('Undo').click();
    assert(window.__labelSaves.queue.length === 0, 'An unsaved click can be removed before saving');
    const normalFetch = window.fetch;
    window.fetch = (url, options) => String(url).endsWith('/health') ?
      Promise.resolve(new Response(JSON.stringify({ok:true}), {status:200})) : normalFetch(url,options);
    clickAt(95,15); button('Save now').click();
    await until(() => !!window.__labelSaves.error && !window.__labelSaves.busy);
    assert(window.__labelSaves.error.includes('Restart SPACEMAP') && !window.__labelSaves.queue[0].frozen,
      'An older running service cannot silently use the old save method');
    button('Discard waiting').click();
    assert(!window.__labelSaves.queue.length && !window.__labelSaves.error, 'Waiting changes can be discarded without saving');
    window.fetch = (url, options) => String(url).endsWith('/assign') ?
      Promise.resolve(new Response(JSON.stringify({error:'Test interrupted connection'}), {status:503})) : normalFetch(url,options);
    clickAt(95,15); button('Save now').click();
    await until(() => !!window.__labelSaves.error && !window.__labelSaves.busy);
    assert(document.querySelector('.label-save-status').textContent.startsWith('Not saved'), 'A failed save is clearly marked');
    const currentId = state.spec.id;
    await window.__init({...state.spec, id: 'must-not-open-yet'});
    assert(state.spec.id === currentId && window.__labelSaves.queue.length === 1, 'Image changes cannot discard waiting labels');
    window.fetch = normalFetch;
    button('Save now').click();
    await until(() => !window.__labelSaves.busy && !window.__labelSaves.queue.length);
    recent = await (await fetch(reviewURL)).json();
    assert(recent.items.find(i => i.label === 3).value === 'New T', 'Retry saves the waiting assignment');
    button('Inspect').click();
    document.querySelector('.reference-review summary').click();
    const reviewCell = () => Array.from(document.querySelectorAll('.review-cell')).find(b => b.textContent.includes('Cell 3 ·'));
    await until(() => reviewCell());
    assert(!reviewCell().disabled, 'Cells in the review list can be opened');
    reviewCell().click();
    await until(() => state.inspectedCell?.result.label === 3 && button('Mark needs review') && !button('Mark needs review').disabled);
    button('Mark needs review').click();
    await until(() => !!button('Needs review · mark checked'));
    let flagged = await (await fetch(reviewURL + '&mode=flagged')).json();
    assert(flagged.total === 1 && flagged.items[0].value === 'New T', 'Review flag keeps the phenotype unchanged');
    button('Needs review · mark checked').click();
    await until(() => !!button('Mark needs review'));
    flagged = await (await fetch(reviewURL + '&mode=flagged')).json();
    assert(flagged.total === 0, 'A checked cell leaves the review list');
    assert(!Object.keys(localStorage).some(k => k.startsWith('spacemap-label-draft-')), 'Confirmed saves clear their recovery drafts');
    button('Click cells').click();
    window.__labelSaves.error = 'Hold for draft recovery check';
    const recoveryName = document.querySelector('[aria-label="Phenotype name to assign"]');
    recoveryName.value = 'First draft'; recoveryName.dispatchEvent(new Event('input')); clickAt(135,15);
    recoveryName.value = 'Last draft'; recoveryName.dispatchEvent(new Event('input')); clickAt(135,15);
    const drafts = Object.keys(localStorage).filter(k => k.startsWith('spacemap-label-draft-')).map(k => [k, localStorage.getItem(k)]);
    // Reinsert in reverse order: browser storage order must not decide which
    // phenotype wins after an interrupted viewer is restored.
    drafts.forEach(([k]) => localStorage.removeItem(k));
    drafts.reverse().forEach(([k,v]) => localStorage.setItem(k,v));
    window.__labelSaves.queue = []; window.__labelSaves.error = null;
    window.__restoreLabelDrafts();
    await until(() => !window.__labelSaves.busy && !window.__labelSaves.queue.length);
    recent = await (await fetch(reviewURL)).json();
    assert(recent.items.find(i => i.label === 4).value === 'Last draft', 'Restored drafts keep the original click order');
    button('Inspect').click();
    change(document.querySelector('[aria-label="Save labels in column"]'), 'phenotype_review');
    change(document.querySelector('[aria-label="Suggestion method"]'), 'examples');
    change(document.querySelector('[aria-label="Learn phenotypes from"]'), 'phenotype_training');
    document.querySelector('[aria-label="Suggest phenotypes"]').click();
    clickCell();
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(document.querySelectorAll('.suggestion-choice').length === 2 &&
      document.querySelector('.cell-suggestions').textContent.includes('phenotype_training'),
      'Suggestions disclose the chosen training column and show probabilities');
    const summary = document.querySelector('#cellcard');
    assert(summary.querySelector('h4').nextElementSibling.classList.contains('cell-calls') &&
      summary.querySelector('.cell-calls').nextElementSibling.classList.contains('cell-suggestions') &&
      summary.querySelector('.suggestion-choice').getBoundingClientRect().bottom <= summary.getBoundingClientRect().bottom,
      'The suggested names are visible below the cell\'s phenotype calls without scrolling');
    const setMinimum = image.channels[0].low;
    image.channels[0].low = undefined;
    change(document.querySelector('#cellcard .profile-source'), 'image');
    await until(() => state.inspectedCell?.result.suggestions?.status === 'error');
    assert(document.querySelector('.cell-suggestions').textContent.includes('Set backgrounds for 1 marker'),
      'Missing backgrounds explain why image suggestions are unavailable at the top of the summary');
    button('Use saved measurements').click();
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(state.inspectedCell.result.suggestions.provenance.source === 'table', 'The summary can switch explicitly to saved measurements');
    image.channels[0].low = setMinimum;
    clickCell();
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    const beforeAccept = state.inspectedCell.result.label;
    const input = document.querySelector('[aria-label="Phenotype name to assign"]');
    input.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    assert(!state.inspectedCell.result.acceptedSuggestion, 'Typing Enter in a text box does not assign a suggestion');
    document.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    assert(state.inspectedCell.result.acceptedSuggestion && window.__labelSaves.queue[0].source === 'accepted',
      'Enter accepts the first suggestion and queues the correct cell');
    button('Save now').click();
    await until(() => !window.__labelSaves.busy && !window.__labelSaves.queue.length);
    const accepted = await (await fetch('/spacemap-viewer/api/cells/' + cells.token + '/review?t=t0&column=phenotype_review')).json();
    assert(accepted.items[0].label === beforeAccept && accepted.items[0].source === 'accepted',
      'Accepted suggestions retain their source in the saved review record');
    button('Next least-sure cell').click();
    await until(() => !cells.suggestionSearch && state.inspectedCell?.result.label !== beforeAccept);
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(state.inspectedCell.result.suggestions.provenance.source === 'table', 'Least-sure navigation opens an unlabelled cell');
    change(document.querySelector('#cellcard .profile-source'), 'corrected');
    await until(() => state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(state.inspectedCell.result.suggestions.provenance.source === 'corrected', 'Suggestions follow the selected image measurement source');
    const oldCell = state.inspectedCell.result.label;
    button('Next least-sure cell').click();
    await until(() => !cells.suggestionSearch && state.inspectedCell?.result.suggestions?.status === 'ready');
    assert(state.inspectedCell.result.suggestions.provenance.source === 'corrected' &&
      document.querySelectorAll('#cellcard .cell-profile .bar').length === 2, 'Image-based navigation keeps the marker bars');
    const initial = JSON.parse(JSON.stringify(state.spec));
    await Promise.all([window.__init({...initial, id: 'earlier-open'}), window.__init({...initial, id: 'latest-open'})]);
    await until(() => state.imagesReady);
    assert(state.spec.id === 'latest-open', 'The newest image choice wins');
    assert(state.layers.length === initial.layers.length && new Set(state.layers.map(l => l.id)).size === state.layers.length,
      'Overlapping openings do not duplicate layers');
    fetch('/result?ok=' + encodeURIComponent(JSON.stringify({threshold_round_trip_ms: elapsed, assignment_feedback_ms: feedbackMs, checks})));
  } catch (e) { fetch('/result?error=' + encodeURIComponent(e.message + ' (after: ' + lastCheck + ')\n' + e.stack)); }
}, 500);
</script>
"""


# The viewer's parts share their names on the page; this hands the checks what they need.
EXPOSE = "<script>performance.setResourceTimingBufferSize(100000);window.__state=state;window.__labelSaves=labelSaves;window.__restoreLabelDrafts=restoreLabelDrafts;window.__init=init;</script>"

def main(driver=DRIVER, studio=False, timeout=75, setup=None, screenshot=True, pixel_ratio=None, compress=False):
    if not shutil.which("firefox"):
        raise SystemExit("Firefox is required for this browser check.")
    with tempfile.TemporaryDirectory(prefix="spacemap-browser-") as folder:
        root = Path(folder)
        os.environ["SPACEMAP_HOME"] = str(root / "state")
        os.environ["SPACEMAP_FAST_COPY"] = "on"
        os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
        import numpy as np
        import pandas as pd
        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse, Response
        from starlette.routing import Mount, Route
        import uvicorn

        from spacemap import dataio
        from spacemap.viewer import cells
        from spacemap_gui.viewer import api, registry
        from spacemap_gui.viewer.component import FRONTEND_DIR
        from test_viewer import _square, _write_channel_pages

        image = root / "roi.ome.tiff"
        _write_channel_pages(image, channels=2, height=400, width=400)
        image_token = registry.register(image)
        corrected_image = root / "roi_bg_subtracted.ome.tiff"
        shutil.copyfile(image, corrected_image)
        seg = root / "cells.geojson"
        seg.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [_square(i + 1, 40 * (i % 8) + 10, 40 * (i // 8) + 10) for i in range(40)],
                }
            )
        )
        pyramid = cells.ensure_label_pyramid(seg, 400, 400, root / "cache")
        table = root / "cells.csv"
        pd.DataFrame(
            {
                "batch_id": ["roi"] * 40,
                "label": np.arange(1, 41),
                "cell_id": np.arange(40),
                "M0": np.arange(40) * 10 + 1,
                "M1": np.arange(40) * 12 + 1,
                "x_centroid": [40 * (i % 8) + 20 for i in range(40)],
                "y_centroid": [40 * (i // 8) + 20 for i in range(40)],
                "phenotype_training": ["A"] * 20 + ["B"] * 20,
                "phenotype_ml_consensus": ["A"] * 20 + ["B"] * 20,
                "phenotype_ref1": ["A"] * 20 + ["B"] * 20,
            }
        ).to_csv(table, index=False)
        dataio.ensure_fast_copy(table, force=True)
        consensus = root / "other_loaded_table.csv"
        pd.DataFrame({"phenotype_ml_consensus": ["Consensus T", "Consensus B"]}).to_csv(consensus, index=False)
        cache = root / "outputs/phenotyping/cache" if studio else root
        cache.mkdir(parents=True, exist_ok=True)
        panel = cache / "panel.json"
        panel.write_text(
            json.dumps(
                {
                    "criteria_cols": ["Criteria 1"],
                    "table": [{"Phenotype": "A", "Criteria 1": ["M0"]}, {"Phenotype": "B", "Criteria 1": ["M1"]}],
                }
            )
        )
        thresholds = cache / "thresholds.csv"
        pd.DataFrame([{"batch_id": "roi", "M0": 50, "M1": 50}]).to_csv(thresholds, index=False)
        cell_token = registry.register_object(
            "cells",
            image=str(image),
            pyramid=str(pyramid),
            roi="roi",
            segmentation=str(seg),
            background_image=str(corrected_image),
            tables={"t0": str(table), "t1": str(consensus)},
            colors={},
        )
        resolve = registry.register_object(
            "resolve", cells=cell_token, table=str(table), roi="roi", panel=str(panel), thresholds=str(thresholds)
        )
        display = registry.register_object("display", path=str(root / "display.json"), image=str(image))
        annotation = root / "tissue.geojson"
        feature = _square(0, 10, 10, 80)
        feature["properties"] = {"classification": {"name": "tissue"}}
        annotation.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
        ann = registry.register_object("annotations", image=str(image), path=str(annotation))
        if setup:
            setup(root)  # project files a check needs (ROI files, cell tables...)
        registry.publish(
            "browser-check",
            {
                "id": "synthetic",
                "title": "Threshold browser check",
                "layers": [
                    {
                        "id": "image",
                        "type": "image",
                        "token": image_token,
                        "display_token": display,
                        "name": "Synthetic image",
                        "visible": True,
                        "opacity": 1,
                        "open": True,
                        "channels": [
                            {"index": 0, "low": 30, "high": 4000, "visible": True},
                            {"index": 1, "low": 30, "high": 5000, "visible": False},
                        ],
                    },
                    {
                        "id": "cells",
                        "type": "cells",
                        "token": cell_token,
                        "image_token": image_token,
                        "roi": "roi",
                        "resolve": resolve,
                        "name": "Cells",
                        "visible": True,
                        "opacity": 1,
                        "open": True,
                        "has_background_image": True,
                        "tables": [{"id": "t0", "name": "Cells"}],
                        "table": "t0",
                    },
                    {
                        "id": "annotation",
                        "type": "annotations",
                        "token": ann,
                        "image_token": image_token,
                        "name": "Tissue",
                        "visible": False,
                        "opacity": 0.2,
                    },
                ],
            },
            project={
                "root": str(root),
                "tab": "tab7",
                "unified": str(table),
                "mappings": {"tab3_output_dir": str(root)},
            }
            if studio
            else None,
        )
        done = threading.Event()
        results = []

        async def page(request):
            shim = "<script>window.requestAnimationFrame=cb=>setTimeout(()=>cb(performance.now()),16);window.cancelAnimationFrame=clearTimeout;</script>"
            html = (FRONTEND_DIR / "index.html").read_text().replace("<head>", "<head>" + shim)
            return HTMLResponse(html.replace("</body>", EXPOSE + driver + '<img src="/slow" hidden></body>'))

        async def result(request):
            if "progress" in request.query_params:
                print("Browser: " + request.query_params["progress"], flush=True)
                return Response()
            results.append(dict(request.query_params))
            done.set()
            return Response()

        async def slow(request):
            await asyncio.sleep(max(55, timeout - 10))
            return Response(b"", media_type="image/gif")

        app = Starlette(
            routes=[
                Route(api.MOUNT_PATH + "/window/index.html", page),
                Route("/result", result),
                Route("/slow", slow),
                Mount(api.MOUNT_PATH, api.app),
            ]
        )
        if compress:
            # Production st.App wraps custom viewer routes in this middleware.
            from streamlit.web.server.starlette.starlette_gzip_middleware import SelectiveGZipMiddleware
            app = SelectiveGZipMiddleware(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        profile = root / "firefox"
        profile.mkdir()
        (profile / "user.js").write_text(
            'user_pref("network.proxy.type", 0);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n'
            + (f'user_pref("layout.css.devPixelsPerPx", "{float(pixel_ratio)}");\n' if pixel_ratio is not None else '')
        )
        with (root / "firefox.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "firefox",
                    "--headless",
                    "--no-remote",
                    "--profile",
                    str(root / "firefox"),
                    *(["--screenshot", str(root / "viewer.png")] if screenshot else []),
                    "--window-size",
                    "1600,1100",
                    f"http://127.0.0.1:{port}{api.MOUNT_PATH}/window/index.html?session=browser-check"
                    + ("&studio=1&step=tab7" if studio else ""),
                ],
                stdout=log,
                stderr=log,
            )
            try:
                if not done.wait(timeout):
                    raise AssertionError("Browser timed out: " + (root / "firefox.log").read_text())
                assert results and "ok" in results[0], results
                print(json.dumps(results), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                server.should_exit = True
                # Let requests still running finish: exiting in the middle of one aborts the process (pyarrow).
                thread.join(timeout=30)


if __name__ == "__main__":
    main()
