"""Choose previously unassigned identity columns without needing a classic-tab detour."""

import json

from viewer_studio_leiden_browser import setup as prepare
from viewer_threshold_browser import main


def setup(root):
    prepare(root)
    path = root / (root.name + ".json")
    config = json.loads(path.read_text())
    config["tab_configurations"] = {"tab7_phenotyping": {"studio_table_columns": {
        "table": str(root / "cells.csv"), "roi": "", "slide": "", "cell_id": ""}}}
    path.write_text(json.dumps(config))


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<900;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
setTimeout(async()=>{
 try{
  const role=label=>document.querySelector('[aria-label="Phenotyping '+label+' column"]');
  await until(()=>role('Cell ID'));
  assert(!role('ROI').value && !role('Cell ID').value, 'Both roles start unassigned');
  const choose=(label,value)=>{const input=role(label);input.value=value;input.dispatchEvent(new Event('change'));};
  choose('ROI','batch_id');
  assert(document.querySelector('#studio-tools').textContent.includes('Choose the ROI and Cell ID columns to save.'), 'The first selection is kept until the required pair is complete');
  choose('Cell ID','cell_id');
  await until(()=>document.querySelector('tr[data-roi="roi"] td:nth-child(3)')?.textContent==='40');
  assert(role('ROI').value==='batch_id' && role('Cell ID').value==='cell_id', 'Both selections save together');
  choose('Slide','slide');
  await until(()=>document.querySelector('tr[data-roi="roi"]').textContent.includes('physical_slide_1'));
  const response=await fetch('/spacemap-viewer/api/studio/browser-check/phenotyping_inputs');
  const plan=await response.json();
  assert(plan.roles.slide==='slide' && plan.cells===40, 'The chosen physical slide and cell counts are saved');
  assert(role('Segmentation label').value==='label', 'The local mask label is separate from the global cell ID');
  const cells=window.__state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.combinedTable);
  cells.table='t1';
  const colour=document.querySelector('[aria-label="Colour cells by"]');
  await until(()=>Array.from(colour.options).some(option=>option.value==='phenotype_ml_consensus'));
  colour.value='phenotype_ml_consensus';colour.dispatchEvent(new Event('change'));
  await until(()=>cells.table===cells.combinedTable && cells.color_by==='phenotype_ml_consensus');
  assert(cells.table!=='t1', 'Colour by selects the combined input even when an older table has the same column');
  Array.from(document.querySelectorAll('#studio-tools button')).find(button=>button.textContent==='View data').click();
  await until(()=>dataView.source?.combined && dataView.result?.total===40);
  assert(dataView.info.label_column==='label' && dataView.info.cell_id_column==='cell_id', 'View data uses both saved cell identities');
  await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){await fetch('/result?error='+encodeURIComponent(error.stack||String(error)));}
},0);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=100, setup=setup, screenshot=False)
