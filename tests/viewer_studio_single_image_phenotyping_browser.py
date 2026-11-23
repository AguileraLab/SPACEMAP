"""Direct tagging input, manual labels and compact cross-image controls in Firefox."""

import json
import shutil

import pandas as pd

from viewer_studio_leiden_browser import setup as prepare
from viewer_threshold_browser import main


def setup(root):
    from spacemap import dataio
    from spacemap_gui import persistence
    persistence.load_projects_index = lambda: {root.name: str(root)}
    prepare(root)
    table = pd.read_csv(root / "cells.csv").drop(columns=["batch_id", "cell_id"])
    table["tissue_roi"] = [1] * 20 + [2] * 20
    table['phenotype_ref2'] = table['phenotype_ref1']
    table['bed'] = ['tumor'] * 20 + ['stroma'] * 20
    table.to_csv(root / "cells.csv", index=False)
    dataio.ensure_fast_copy(root / "cells.csv", force=True)
    table.to_csv(root / "second_cell_features_tagged.csv", index=False)
    shutil.copyfile(root / "roi.ome.tiff", root / "phenotyping_rois/second.ome.tiff")
    path = root / (root.name + ".json")
    config = json.loads(path.read_text())
    config["directory_mappings"]["global_pheno_dir"] = str(root)
    config["tab_configurations"] = {"tab7_phenotyping": {"studio_table_columns": {
        "table": str(root / "cells.csv"), "roi": "", "slide": "", "cell_id": "label", "image": "roi"}}}
    path.write_text(json.dumps(config))
    from spacemap_gui.studio import reference_models, tasks
    saved = root / 'outputs' / 'training_examples.parquet'
    saved.parent.mkdir(exist_ok=True)
    table.to_parquet(saved, index=False)
    settings = {'table': str(saved), 'markers': ['M0', 'M1'], 'training_column': 'phenotype_ref1',
                'roi_column': '', 'label_column': 'label', 'project_root': str(root)}
    settings['revision'] = tasks.source_revision(settings)
    fitted = reference_models.train(settings, root, lambda *args: None)
    # Exercise the real older-format failure through the UI: examples are
    # present, but classes and a permanent holdout were not recorded.
    import joblib
    import numpy as np
    source = reference_models.model_path(fitted['models'][0]['model_id'])
    artifact = joblib.load(source)
    sample, held = artifact['training_sample'], artifact['validation_sample']
    full = {k: np.concatenate([sample[k], held[k]]) for k in ('values', 'targets', 'rows')}
    order = np.argsort(full['rows'])
    full = {k: v[order] for k, v in full.items()}
    artifact['model'].fit(full['values'], full['targets'])
    artifact.update(rows=full['rows'], oob=artifact['model'].oob_decision_function_,
                    training_sample={**full, 'markers': settings['markers']})
    artifact.pop('validation_sample'); artifact.pop('example_archive'); artifact.pop('validation_archive')
    meta = json.loads(source.with_suffix('.json').read_text())
    for key in ('classes', 'validation_held_out', 'held_out_per_class'):
        meta.pop(key)
    joblib.dump(artifact, source)
    source.with_suffix('.json').write_text(json.dumps(meta))
    from spacemap_gui.studio import artifact_learning, training, auto_reference, reference_examples
    auto_reference._path(root).parent.mkdir(parents=True, exist_ok=True)
    auto_reference._path(root).write_text(json.dumps({reference_examples.source_key(root/'cells.csv'):
        {'enabled':False, 'state':'off', 'settings':{'table':str(root/'cells.csv'),'training_column':'phenotype_ref2'}}}))
    (root/'empty-models').mkdir()
    _, models = training.folders(root)
    (models / 'models').mkdir(parents=True, exist_ok=True)
    (models / 'models' / 'cellpose_demo').write_bytes(b'test fixture weights')
    artifacts = artifact_learning.folder(root, 'clot')
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': 'test fixture'}, artifacts / 'clot_demo.joblib')
    (artifacts / 'versions.json').write_text(json.dumps([{'id': 'clot_demo', 'name': 'Clot demo'}]))


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1800;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check+' UI: '+document.querySelector('#studio-tools')?.textContent?.slice(0,2500));};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
setTimeout(async()=>{
 try{
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools()?.querySelector('[aria-label="Phenotyping Image column"]'));
  const rows=()=>[...tools().querySelectorAll('tr[data-roi]')];
  assert(rows().length===2 && rows().every(row=>['roi','second'].includes(row.dataset.roi)), 'Only actual images are listed');
  await until(()=>rows().every(row=>row.querySelector('[data-cell-count]').textContent==='40'));
  assert(rows().every(row=>row.querySelector('select')?.value), 'Every image has a paired table and its own cell count');
  assert(!tools().textContent.includes('provenance') && !tools().textContent.includes('No cells'), 'No misleading correction or segmentation warning');
  assert(tools().querySelector('[aria-label="Phenotyping Image column"]').selectedOptions[0].textContent.includes('No image column'), 'Image grouping is optional');
  const pick=tools().querySelector('[aria-label="Phenotyping method"]');pick.value='reference';pick.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('.reference-model-training'));
  tools().querySelector('.reference-model-training').open=true;
  await until(()=>tools().querySelector('[data-reference-train-selected]')?.dataset.imageCount==='2');
  assert(tools().querySelectorAll('[aria-label^="Process image "]').length===2,'Both image tables are selectable in the main table without correction');
  assert(!tools().querySelector('[aria-label="Across images prediction column"]'),'Prediction settings stay in MR-ML');
  assert(!tools().querySelector('.reference-collection'),'Training scopes share one action row');
  await until(()=>tools().querySelector('[aria-label="Phenotyping model to load"]')?.options.length===3);
  const sharing=tools().querySelector('[aria-label="Phenotyping model to load"]');
  assert(sharing.parentElement.querySelector('button').textContent==='Use','One Use action activates the model');
  const actions=tools().querySelector('.reference-training-actions');
  assert(actions.querySelector('[data-reference-train]') && actions.querySelector('[aria-label="Auto-train suggestions"]'),'Manual and automatic training stay together');
  assert(actions.querySelector('[data-reference-train]').textContent==='Current image'&&actions.querySelector('[data-reference-train-selected]').textContent==='Selected images','Two clearly named actions choose the training scope');
  assert(!tools().querySelector('.reference-training-details').open,'File paths and alternate training labels are hidden by default');
  assert([...sharing.options].slice(1).every(option=>/Random Forest|Extra Trees/.test(option.textContent)), 'One dropdown describes both trained model types');
  assert(sharing.options[0].textContent.includes('Random Forest + Extra Trees'), 'The training algorithms are explicit');
  assert(!tools().querySelector('[aria-label="Phenotyping model to share"]') && !actions.querySelector('select'),'No duplicate model selectors for sharing or across images');
  const labels=tools().querySelector('[aria-label="Learn model from column"]');
  assert(labels.options[0].textContent==='Manual labels · phenotype_ref1','Training names the current reference column');
  const bed=labels.querySelector('option[value="bed"]');
  assert(bed.parentElement.label.startsWith('Other table labels') && bed.textContent.includes('annotation region'),'Region labels remain available but separate from phenotype labels');
  const library=await (await fetch('/spacemap-viewer/api/studio/browser-check/model_library')).json();
  assert(library.library.length===2,'Manual training automatically exports both model versions');
  assert(tools().textContent.includes('Saved in Models folder'),'The selected model shows it is already exported');
  const modelFolder=tools().querySelector('[aria-label="Phenotyping models folder"]');
  assert(modelFolder && modelFolder.getBoundingClientRect().top<tools().querySelector('.manual-reference-card').getBoundingClientRect().top,'Model directory is above the image list and training panel');
  const split=tools().querySelector('[aria-label="Training percentage"]');
  assert(split.value==='70' && split.parentElement.textContent.includes('30% validation'),'Default split is clearly 70/30');
  split.value='80';split.dispatchEvent(new Event('change'));
  await until(()=>split.parentElement.textContent.includes('20% validation'));
  const loadChoice=tools().querySelector('[aria-label="Phenotyping model to load"]');
  await until(()=>loadChoice.options.length>=3);
  loadChoice.value=[...loadChoice.options].find(o=>o.value&&o.textContent.includes('Random Forest')).value;loadChoice.dispatchEvent(new Event('change'));
  loadChoice.parentElement.querySelector('button').click();
  await until(()=>tools().querySelector('.reference-setup').dataset.modelId);
  const setup=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();
  assert(setup.training_percent===80 && setup.base_model_id,'Loaded model and chosen split are saved');
  const legacy=setup.models.find(m=>m.id===setup.base_model_id);
  assert(legacy.classes.length===2 && legacy.reusable_examples>0,'Older model shows its actual phenotypes and reusable examples');
  assert(!legacy.validation_held_out,'Older final-fit examples are not presented as independent validation');
  const cells=window.__state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.combinedTable);
  assert(cells.combinedTable,'A single-image table links to segmentation without a batch_id column');
  window.__state.layers.filter(l=>l.type==='annotations').forEach(l=>l.visible=false);
  setTool('off');
  const image=window.__state.layers.find(l=>l.type==='image');
  window.__state.viewer.raiseEvent('canvas-click',{quick:true,position:window.__state.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(55,15)),originalEvent:{}});
  await until(()=>window.__state.inspectedCell?.result.suggestions?.status==='ready');
  assert(window.__state.inspectedCell.result.suggestions.provenance.panel_check.needs_checking.length===2,'Older examples without a permanent holdout are clearly marked for checking');
  assert(document.querySelector('#cellcard').textContent.includes(' · check'),'Cell suggestions show when their available-marker validation is missing');
  const picked={...cells};window.__state.layers[window.__state.layers.indexOf(cells)]=picked;
  document.dispatchEvent(new CustomEvent('spacemap-cells-picked',{detail:{layer:picked,labels:[1,2,3,4,5,6,7]}}));
  await until(()=>tools().querySelector('[aria-label="Cells to label"]').selectedOptions[0]?.textContent.includes('7'));
  const name=tools().querySelector('[aria-label="Phenotype name to assign"]');name.value='Custom phenotype';name.dispatchEvent(new Event('input'));
  const assignButton=[...tools().querySelectorAll('button')].find(button=>button.textContent==='Assign 7 cells');
  assert(assignButton&&!assignButton.disabled,'A rebuilt cell layer keeps the seven-cell selection available for custom labelling');
  assignButton.click();
  await until(()=>tools().textContent.includes('Custom phenotype · 7'));
  assert(true,'Custom phenotype is saved for all seven selected cells');
  setTool('click');
  await until(()=>tools().querySelector('.manual-reference-card')?.referenceInfo?.counts.groups.some(([label])=>label==='Custom phenotype'));
  const stableRows=rows(), stableCard=tools().querySelector('.manual-reference-card');
  const request=window.fetch;let tableReloads=0;
  window.fetch=(url,...args)=>{if(String(url).includes('/phenotyping_inputs'))tableReloads++;return request(url,...args);};
  const input=tools().querySelector('[aria-label="Phenotype name to assign"]');input.focus();
  await assign(picked,{labels:[8,9]},'Silent phenotype');await flushLabels();
  await until(()=>tools().textContent.includes('Silent phenotype · 2'));
  await delay(1700);
  assert(tableReloads===0 && stableRows.every((row,index)=>row===rows()[index]),'Click labelling saves without reloading image pairings or cell counts');
  assert(tools().querySelector('.manual-reference-card')===stableCard,'Labelling keeps the manual label panel in place');
  assert(document.activeElement===input,'Labelling keeps typing focus in place');
  await undo();
  await until(()=>!tools().querySelector('.reference-counts').textContent.includes('Silent phenotype'));
  assert(true,'Quiet count updates also handle Undo');
  window.fetch=request;setTool('off');
  assert(tools().textContent.includes('Use checked labels only'),'The training filter has a concise descriptive label');
  tools().querySelector('.reference-training-details').open=true;
  const modelSelect=tools().querySelector('[aria-label="Phenotyping model to load"]');
  const trainingLabels=tools().querySelector('[aria-label="Learn model from column"]');
  const savedFile=tools().querySelector('[aria-label="Saved phenotyping model file"]');
  assert(Math.abs(modelSelect.getBoundingClientRect().left-trainingLabels.getBoundingClientRect().left)<2 && Math.abs(savedFile.getBoundingClientRect().left-trainingLabels.getBoundingClientRect().left)<2,'Model, training labels and model file fields align');
  const beforeUpdate=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();
  tools().querySelector('[data-reference-train]').click();
  let afterUpdate;
  await until(async()=>{afterUpdate=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();return afterUpdate.base_model_id!==beforeUpdate.base_model_id;});
  const updated=afterUpdate.models.find(m=>m.id===afterUpdate.base_model_id);
  assert(updated.column==='phenotype_ref1','A disabled old Auto-train setting cannot change the visible reference column');
  assert(updated.parent===beforeUpdate.base_model_id && updated.classes.includes('Custom phenotype'),'Train model extends the older loaded model with the new phenotype');
  await until(()=>tools().querySelector('[aria-label="Phenotyping model to load"]')?.selectedOptions[0]?.value===afterUpdate.models.find(m=>m.id===updated.id).key);
  assert(true,'Successful training selects the new version in the dropdown');
  assert(updated.shared_path && afterUpdate.models.some(m=>m.id===beforeUpdate.base_model_id),'The new version is automatically saved and the original remains available');
  const methodPick=tools().querySelector('[aria-label="Phenotyping method"]');methodPick.value='mr_ml';methodPick.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('[aria-label="Saved reference model"]'));
  assert([...tools().querySelectorAll('[aria-label^="Process image "]')].every(p=>p.checked),'MR-ML uses the shared image selection');
  methodPick.value='reference';methodPick.dispatchEvent(new Event('change'));
  await until(()=>tools().querySelector('.manual-reference-card'));
  rows().find(row=>row.dataset.roi==='second').querySelector('button').click();
  await until(()=>tools().querySelector('[aria-label="Phenotyping cell table"]')?.value.endsWith('second_cell_features_tagged.csv'));
  assert(true,'Opening another image selects its corresponding independent table');
  const otherSetup=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();
  assert(otherSetup.base_model_id===updated.id,'The same learned model stays active when opening another image');
  const folderInput=tools().querySelector('[aria-label="Phenotyping models folder"]');
  // The fixture's root is returned by the independent table path.
  const tablePath=tools().querySelector('[aria-label="Phenotyping cell table"]').value;
  folderInput.value=tablePath.slice(0,tablePath.lastIndexOf('/'))+'/empty-models';
  folderInput.parentElement.querySelector('button').click();
  await until(()=>tools().querySelector('[aria-label="Phenotyping model to load"]').options.length===1);
  const empty=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();
  assert(empty.models.length===0&&!empty.base_model_id,'An empty Models folder lists no cached models');
  assert(window.__state.layers.filter(l=>l.type==='cells').every(l=>!l.suggestionModelId&&!l.suggestionsOn),'Changing folder clears old suggestion models');
  assert(formatModelTime('2026-10-19T19:42:37+00:00').includes('2:42')&&formatModelTime('2026-10-19T19:42:37+00:00').includes('CDT'),'Model dates use the browser time zone');
  document.querySelector('[data-step="tab0"]').click();
  await until(()=>tools().querySelector('[aria-label="Share Phenotyping model"]'));
  assert(tools().querySelector('[aria-label="Share Phenotyping model"]').options.length===3,'Project library includes the new phenotype version in its dropdown');
  assert(tools().querySelector('[aria-label="Share Segmentation (Cellpose) model"]') && tools().querySelector('[aria-label="Share Artifacts model"]'),'Segmentation and artifact models have separate dropdowns');
  const libraryPick=tools().querySelector('[aria-label="Share Phenotyping model"]');
  libraryPick.selectedIndex=1;libraryPick.dispatchEvent(new Event('change'));
  assert(libraryPick.parentElement.querySelector('button')||libraryPick.parentElement.textContent.includes('In the library'),'Changing a model updates its matching Share action');
  await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){await fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(error.stack||'')));}
},0);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=130, setup=setup, screenshot=False)
