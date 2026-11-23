"""Inspect, validate, group and merge models, with figures in the shared dock."""
from viewer_mr_ml_reference_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+' '+document.querySelector('[data-model-review-status]')?.textContent);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const method=document.querySelector('[aria-label="Phenotyping method"]');method.value='reference';method.dispatchEvent(new Event('change'));
 await until(()=>document.querySelector('.reference-model-review'));
 const panel=document.querySelector('.reference-model-review');panel.open=true;
 const button=text=>[...panel.querySelectorAll('button')].find(b=>b.textContent===text);
 await until(()=>panel.querySelectorAll('.model-review-choices input').length===2);
 assert(panel.previousElementSibling.textContent.includes('Compare predictions'),'Model review is separate from training');
 assert(getComputedStyle(panel).borderTopWidth==='1px','Model review has a visible separator');
 assert(Math.abs(panel.getBoundingClientRect().left-panel.closest('.studio-sec').getBoundingClientRect().left)<2,'The separator spans the section width');
 panel.querySelectorAll('.model-review-choices input').forEach(b=>b.click());button('Inspect models').click();
 await until(()=>panel.querySelectorAll('.model-review-table tbody tr').length===2);
 assert(panel.textContent.includes('2 shared features'),'Shared model inputs are shown');
 assert(panel.querySelectorAll('.model-review-features input:checked').length===2,'Shared features are selectable for retraining');
 await until(()=>document.querySelectorAll('#studio-figures [data-figure^="model-"] img').length===4);
 assert([...document.querySelectorAll('#studio-figures img')].some(i=>i.alt.includes('reference feature profiles')),'Input versus phenotype heatmaps appear in Figures');
 assert([...document.querySelectorAll('#studio-figures img')].some(i=>i.alt.includes('feature importance')),'Model feature importance appears beside the heatmaps');
 button('Validate models').click();
 await until(()=>panel.querySelector('[data-model-review-status]').textContent.startsWith('Compared'));
 await until(()=>document.querySelectorAll('#studio-figures [data-figure$="-validation"] img').length===2);
 assert([...panel.querySelectorAll('.model-review-table tbody tr')].every(r=>r.lastChild.textContent.includes('cells')),'Validation results show their cell counts');
 assert(document.querySelector('#studio-figures').textContent.includes('Rows are reference labels; columns are predictions'),'Confusion matrices identify their axes');
 panel.querySelector('[aria-label="Keep model feature M1"]').click();
 panel.querySelectorAll('[data-phenotype]').forEach(i=>{i.value='Epithelia';});
 panel.querySelector('[aria-label="New model name"]').value='Merged types';
 button('Merge selected models').click();
 await until(()=>panel.querySelector('[data-model-review-status]').textContent.startsWith('Saved in Models'));
 await until(()=>panel.querySelectorAll('.model-review-choices input').length===3);
 const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/reference_setup')).json();
 const merged=found.models.find(m=>m.name==='Merged types');
 assert(merged&&merged.features===1&&merged.classes.length===2,'The new model excludes the unchecked feature and keeps detailed phenotypes');
 assert(found.models.length===3&&merged.shared_path.includes('CDT'),'Merging keeps originals and creates a dated folder in local time');
 panel.querySelectorAll('.model-review-choices input').forEach(b=>{if(b.checked)b.click();});
 const chosen=[...panel.querySelectorAll('.model-review-choices input')].find(b=>b.value===merged.key);chosen.click();
 button('Inspect models').click();
 await until(()=>panel.querySelectorAll('.model-review-table tbody tr').length===1);
 assert([...panel.querySelectorAll('[data-phenotype]')].every(i=>i.value==='Epithelia'),'Saved broad phenotype groups can be reviewed and edited');
 panel.querySelector('[aria-label="New model name"]').value='Custom types';button('Save custom copy').click();
 await until(()=>panel.querySelectorAll('.model-review-choices input').length===4);
 assert(true,'Saving a custom copy leaves its source model available');
 await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){await fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(error.stack||'')));}},0);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup, screenshot=False)
