"""Merged tagging controls and automatic data refresh, on a disposable project."""
from viewer_threshold_browser import main
from viewer_studio_tagging_browser import setup as tagging_setup

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(const end=Date.now()+90000;Date.now()<end;){if(await f())return;await delay(75);}throw Error('Timed out: '+f);};
const tools=()=>document.querySelector('#studio-tools');
const change=(el,value)=>{el.value=value;el.dispatchEvent(new Event('change',{bubbles:true}));};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const route='/spacemap-viewer/api/studio/browser-check';
setTimeout(async()=>{try{
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tagging]'));
 await state.phenotypingLink;
 document.querySelector('[data-step=tagging]').click();
 await until(()=>tools().querySelector('[aria-label="One merged table"]'));
 check(!tools().querySelector('[aria-label="Update existing table"]'),'Update is only offered for merged tables');
 tools().querySelector('[aria-label="One merged table"]').click();
 await until(()=>tools().querySelector('[aria-label="Merged table"]')?.value==='cells.parquet');
 const settings=async()=>(await (await fetch(route+'/tagging_settings')).json()).values;
 tools().querySelector('[aria-label="Update existing table"]').click();
 await until(async()=>(await settings()).seg_tag_update_merged);
 await until(()=>!tools().querySelector('[aria-label="Output folder"]'));
 check(!tools().querySelector('[aria-label="Tagged name ending"]'),'In-place updates hide separate output and suffix controls');
 check(document.querySelector('.studio-runbar').textContent.includes('Update:'),'The run bar identifies an update');
 const update=tools().querySelector('[aria-label="Update existing table"]');
 check(update.getBoundingClientRect().height>0&&update.closest('.field').querySelector('[aria-label="Merged table"]'),
   'The update option is rendered beside the table choice');
 document.querySelector('[aria-label="Clear tagging columns"]').click();
 await until(async()=>!(await settings()).seg_tag_rules.some(r=>r['Binary (0/1)']||r['ROI ID']||r['ROI Size']||r.Distance));
 await until(()=>!tools().querySelector('[aria-label="Tumor Inside"]').checked&&document.querySelector('.studio-runbar').textContent.includes('0 columns'));
 tools().querySelector('[aria-label="Tumor Inside"]').click();
 await until(()=>tools().querySelector('[aria-label="Column name Tumor inside"]'));
 change(tools().querySelector('[aria-label="Column name Tumor inside"]'),'new_region');
 await until(async()=>(await settings()).seg_tag_column_names['Tumor|inside']==='new_region');
 check(document.querySelector('.studio-runbar').textContent.includes('1 column'),'Only the explicitly selected new field will be written');
 tools().querySelector('[aria-label="Pick roi"]').click();
 showDataTab(); await until(()=>dataView.result?.rows.length);
 const cells=state.layers.find(l=>l.type==='cells');
 setCellColour(cells,'M0');
 const image=state.layers.find(l=>l.type==='image'), item=image.item, generation=state.initGeneration;
 const centre=state.viewer.viewport.getCenter(),zoom=state.viewer.viewport.getZoom();
 Array.from(document.querySelectorAll('.studio-runbar button')).find(b=>b.textContent==='Selected').click();
 let job;
 await until(async()=>{const jobs=(await(await fetch(route+'/jobs')).json()).jobs;
   job=jobs.find(j=>j.kind==='cell_tagging');return job&&['done','failed'].includes(job.status);});
 check(job.status==='done'&&job.result.updated&&job.result.count===2,'Update publishes both formats: '+job.message);
 check(job.result.cells===40&&job.result.rows===80&&job.result.columns.join(',')==='new_region',
   'One selected image is updated while the complete cohort is retained');
 await until(()=>dataView.info?.columns.some(c=>(c.name||c)==='new_region'));
 check(cells.color_by==='M0','The selected marker colour is preserved when updated data arrives');
 check(state.initGeneration===generation&&image.item===item&&Math.abs(state.viewer.viewport.getZoom()-zoom)<1e-8&&
   state.viewer.viewport.getCenter().distanceTo(centre)<1e-8,'The loaded image, tiles and viewport remain unchanged');
 check(dataView.info.columns.some(c=>(c.name||c)==='phenotype_ref1'),'Existing manual reference data remains available');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+tools()?.textContent.slice(-1200)));}},100);
</script>'''


def setup(root):
    import pandas as pd
    from spacemap_gui.studio import tagging
    from spacemap_gui.studio.settings import read_config, revision
    from spacemap_gui.tabs import cell_tagging_ui as tab5
    tagging_setup(root)
    frame = pd.read_csv(root / 'cells.csv')
    other = frame.copy(); other['batch_id'] = 'other'; other['cell_id'] += len(frame)
    combined = pd.concat([frame, other], ignore_index=True)
    combined.to_csv(root / 'cells.csv', index=False)
    combined.to_parquet(root / 'cells.parquet', index=False)
    state = tagging.values(root, read_config(root))
    tagging.save(root, {tab5.MERGED_DIR_KEY: str(root), tab5.MERGED_FILE_KEY: 'cells.parquet',
                        tagging.UNITS_KEY: 'px', tab5.CLEAN_PATH_KEY: False}, revision(state))


if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=240)
