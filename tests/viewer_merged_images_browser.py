"""All 44 merged-table images can be selected for shared reference training."""
import json
import shutil
import numpy as np
import pandas as pd
from viewer_threshold_browser import main
from viewer_studio_leiden_browser import setup as prepare


def setup(root):
    from spacemap import dataio
    prepare(root)
    frame = pd.read_csv(root / 'cells.csv')
    pieces = []
    for i in range(44):
        name = 'roi' if i == 0 else f'roi_{i:02d}'
        pieces.append(frame.assign(batch_id=name, cell_id=np.arange(len(frame)) + i * len(frame)))
        if i:
            shutil.copy2(root / 'roi.ome.tiff', root / 'phenotyping_rois' / (name + '.ome.tiff'))
    pd.concat(pieces).to_csv(root / 'cells.csv', index=False)
    dataio.ensure_fast_copy(root / 'cells.csv', force=True)
    config = root / (root.name + '.json')
    values = json.loads(config.read_text())
    values['directory_mappings']['global_pheno_dir'] = str(root)
    config.write_text(json.dumps(values))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;};
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const method=document.querySelector('[aria-label="Phenotyping method"]');method.value='reference';method.dispatchEvent(new Event('change'));
 const picks=()=>[...document.querySelectorAll('[aria-label^="Process image "]')];
 await until(()=>picks().length===44&&document.querySelector('.reference-model-training'));
 assert(picks().every(b=>b.checked),'All 44 merged-table images can be selected');
 document.querySelector('.reference-model-training').open=true;
 const selected=()=>document.querySelector('[data-reference-train-selected]');
 const trainingStatus=()=>document.querySelector('[data-reference-training-status]');
 await until(()=>selected()?.dataset.imageCount==='44');
 assert(!selected().disabled,'All merged images are available from the Selected images action');
 assert(!trainingStatus().textContent,'No idle training explanation or stale progress is displayed');
 document.querySelector('[aria-label="Select all phenotyping images"]').click();
 await until(()=>selected().dataset.imageCount==='0');
 assert(selected().disabled,'Training is disabled when no images are selected');
 picks()[0].click();picks()[1].click();
 await until(()=>selected().dataset.imageCount==='2');
 selected().click();
 await until(()=>trainingStatus().textContent==='Training selected images…');
 assert(selected().disabled&&document.querySelector('[data-reference-train]').disabled,'Both training actions wait for the running job');
 let job;
 await until(async()=>{const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json();job=found.jobs.find(j=>j.kind==='reference_collection');return job&&['done','failed'].includes(job.status);});
 assert(job.status==='done','Selected merged images train successfully: '+job.message);
 assert(job.result.examples===80,'Only the two selected images contribute labels');
 await until(()=>trainingStatus().textContent==='Training complete. Saved in Models.');
 assert(!selected().disabled,'Completion replaces progress and enables training again');
 await until(()=>document.querySelector('[aria-label="Phenotyping model to load"]').options.length===3);
 assert(true,'Completed pooled training refreshes the model dropdown automatically');
 const chosen=await (await fetch('/spacemap-viewer/api/studio/browser-check/phenotyping_inputs')).json();
 assert(chosen.selected_images.length===2,'Image selection is persisted');
 const fetchBeforeFailure=window.fetch;
 window.fetch=(url,options)=>String(url).includes('/reference_collection')&&options?.method==='POST'?
   Promise.resolve(new Response(JSON.stringify({error:'Choose a phenotype column.'}),{status:400,headers:{'Content-Type':'application/json'}})):fetchBeforeFailure(url,options);
 selected().click();
 await until(()=>trainingStatus().textContent.includes('Choose a phenotype column.'));
 assert(!selected().disabled,'Training errors appear beside the actions and allow retry');
 window.fetch=fetchBeforeFailure;
 method.value='mr_ml';method.dispatchEvent(new Event('change'));
 await until(()=>document.querySelector('[aria-label="Saved reference model"]'));
 assert(picks().filter(b=>b.checked).length===2,'MR-ML keeps the same image selection');
 await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){await fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(error.stack||'')));}},0);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=150, setup=setup, screenshot=False)
