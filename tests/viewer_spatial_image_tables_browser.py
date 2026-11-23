"""Actual Studio controls and linked results over unmerged CSV and Parquet inputs."""
import json
import shutil

import pandas as pd

from viewer_threshold_browser import main
from viewer_studio_leiden_browser import setup as prepare


def setup(root):
    from spacemap import dataio
    from spacemap_gui.studio import tls
    prepare(root)
    frame=pd.read_csv(root/'cells.csv').drop(columns=['batch_id','cell_id'])
    frame.to_csv(root/'cells.csv',index=False)
    dataio.ensure_fast_copy(root/'cells.csv',force=True)
    frame.to_parquet(root/'second_cell_features.parquet',index=False)
    frame.to_parquet(root/'manual_input.parquet',index=False)
    alternate=root/'alternative_tables';alternate.mkdir()
    frame.to_parquet(alternate/'second_cell_features.parquet',index=False)
    shutil.copyfile(root/'roi.ome.tiff',root/'phenotyping_rois/second.ome.tiff')
    shutil.copyfile(root/'roi.ome.tiff',root/'phenotyping_rois/unpaired.ome.tiff')
    seg=root/'outputs/segmentation';seg.mkdir(parents=True,exist_ok=True)
    for name in ('roi','second'):
        shutil.copyfile(root/'cells.geojson',seg/(name+'_cell_boundaries.geojson'))
    raw=root/'data/codex';raw.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'roi.ome.tiff',raw/'second_CODEX.qptiff')
    path=root/(root.name+'.json');config=json.loads(path.read_text())
    config['directory_mappings']['global_pheno_dir']=str(root)
    config['tab_configurations']={'tab7_phenotyping':{'studio_table_columns':{
        'table':str(root/'cells.csv'),'roi':'','slide':'','cell_id':'label','image':'roi'}},
        'tab8_analysis':{'studio_tls':{**tls.DEFAULTS,'phenotype_col':'phenotype_ml_consensus',
        'b_labels':['A'],'t_labels':['B'],'roi_col':'','min_b':1,'min_t':1,'min_cells':2,
        'max_gap':65.,'pixel_um':1.,'min_fraction':0.}}}
    path.write_text(json.dumps(config))
    # The base fixture was built as a combined-table viewer. Give its initial
    # t0 the same no-image-column roles as an actual per-image layer.
    from spacemap_gui.viewer import registry
    publish = registry.publish
    def publish_roles(channel,spec,**kwargs):
        registry.publish = publish
        for layer in spec['layers']:
            if layer['type']=='cells':
                keys={'roi':'roi','roi_column':'','label_column':'label','cell_id_column':'label'}
                registry.connect_cell_table(layer['token'],'t0',str(root/'cells.csv'),keys)
                layer.update(keys,table_roles={'t0':keys})
        return publish(channel,spec,**kwargs)
    registry.publish = publish_roles


DRIVER=r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const tools=()=>document.querySelector('#studio-tools'),bar=()=>document.querySelector('.studio-runbar');
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+' '+bar()?.textContent+' '+tools()?.textContent);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const by=label=>tools().querySelector('[aria-label="'+label+'"]');
const set=(label,value)=>{const e=by(label);e.value=value;e.dispatchEvent(new Event('change'));};
const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 const originalViewer=state.viewer,originalImage=state.layers.find(l=>l.type==='image').item;
 document.querySelector('[data-step=tab8]').click();
 await until(()=>by('Phenotype column')&&tools().querySelectorAll('tr[data-roi]').length===3);
 const rows=()=>[...tools().querySelectorAll('tr[data-roi]')];
 await until(()=>rows().filter(r=>r.dataset.roi!=='unpaired').every(r=>r.querySelector('[data-cell-count]').textContent==='40'));
 assert(rows().map(r=>r.dataset.roi).join()==='roi,second,unpaired','Separate CSV and Parquet images appear with correct counts');
 assert(by('Pick unpaired').disabled&&tools().querySelector('tr[data-roi="unpaired"]').textContent.includes('No paired table'),'Unpaired images stay visible and cannot silently join a run');
 assert(['Spatial images folder','Spatial tables folder','Spatial output folder'].every(label=>by(label)),'Spatial has the same source folders as Phenotyping and its own Output');
 assert(state.viewer===originalViewer&&state.layers.find(l=>l.type==='image').item===originalImage,'Moving from Phenotyping to Spatial keeps the image and viewer instance');
 assert(by('Cell table for second').value.endsWith('second_cell_features.parquet'),'Automatic matching offers an editable table per image');
 assert(tools().textContent.includes('Per-image cell tables')&&!tools().textContent.includes('make it in Batch correction'),'No merged table or correction required');
 const second=()=>tools().querySelector('tr[data-roi=second]');
 button(second(),'second.ome.tiff').click();
 await until(()=>state.spec?.title?.startsWith('second')&&state.imagesReady);
 assert(state.spec.layers.find(l=>l.type==='image').name==='second.ome.tiff','Opening a row uses its processed TIFF even when a raw QPTIFF has the same sample');
 button(tools().querySelector('tr[data-roi=roi]'),'roi.ome.tiff').click();
 await until(()=>state.spec?.title?.startsWith('roi')&&state.imagesReady&&state.cellsReady);
 const out=by('Spatial output folder').value+'/chosen';set('Spatial output folder',out);
 await until(()=>by('Spatial output folder')?.value===out&&!tools().textContent.includes('Saving…'));
 await delay(400);
 set('Phenotype column','phenotype_ml_consensus');set('Neighbours',5);set('Neighbourhoods',2);set('Smallest region',5);set('Region column','');
 by('Select all').click();button(bar(),'Selected').click();
 await until(()=>by('Name of CN 0'));
 const run=(await jobs()).find(j=>j.kind==='neighbourhoods');
 assert(run?.status==='done'&&run.result.tables.length===2,'Neighbourhood results save to both source files');
 assert(JSON.stringify(run.result.images)==='["roi","second"]','Select all submits both explicit image names');
 await until(()=>rows().filter(r=>r.dataset.roi!=='unpaired').every(r=>r.textContent.includes('Neighbourhoods')));
 assert(true,'Both image rows show completed neighbourhood results');
 await until(()=>document.querySelector('#studio-figures .studio-heatmap .row'));
 document.querySelector('#studio-figures .studio-heatmap .row[data-row="CN 0"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>state.layers.find(l=>l.type==='cells').color_by==='neighborhood_cn');
 assert(state.layers.find(l=>l.type==='cells').hidden.has('CN 1'),'Clicking a neighbourhood filters the open image with its physical table');
 button(tools(),'TLS candidates').click();
 await until(()=>bar().textContent.includes('Find TLS'));
 button(bar(),'This image').click();
 await until(async()=>{const j=(await jobs()).find(j=>j.kind==='tls_candidates');if(j?.status==='failed')throw Error(j.message);return j?.status==='done';});
 const tls=(await jobs()).find(j=>j.kind==='tls_candidates');
 assert(tls.result.export_folder?.includes('/chosen/tls_candidates/'),'TLS exports use the method subdirectory');
 assert(tls.result.tables.length===1&&tls.result.counts.roi>=1,'TLS works on one per-image table and leaves the other untouched');
 await until(()=>tools().querySelector('tr[data-roi="roi"]').textContent.includes('TLS candidates'));
 const shapes=await fetch('/spacemap-viewer/api/studio/browser-check/spatial/tls_shapes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:tls.id,image:'roi'})}).then(r=>r.json());
 assert(shapes.features?.length>0,'Per-image TLS boundaries are available for viewer review');
 const manual=[...by('Cell table for unpaired').options].find(o=>o.value.endsWith('/manual_input.parquet')).value;
 set('Cell table for unpaired',manual);
 await until(()=>!by('Pick unpaired').disabled);
 const shared=await fetch('/spacemap-viewer/api/studio/browser-check/phenotyping_inputs?summaries=0').then(r=>r.json());
 assert(shared.rows.find(r=>r.roi==='unpaired').table_path===manual,'Manual table assignment in Spatial is shared with Phenotyping');
 by('Find spatial images').value='unpaired';by('Find spatial images').dispatchEvent(new Event('input'));
 by('Select all').click();by('Find spatial images').value='';by('Find spatial images').dispatchEvent(new Event('input'));
 assert(by('Pick roi').checked&&by('Pick second').checked&&by('Pick unpaired').checked,'Search select-all preserves selections outside the search');
 button(tools(),'Feature screening').click();await until(()=>by('Screening image column'));
 assert([...by('Screening image column').options].some(o=>o.value==='batch_id'),'Screening offers the image identity without changing source files');
 const priorImage=state.layers.find(l=>l.type==='image').item;
 const priorBounds=state.viewer.viewport.getBounds().toString();
 const root=by('Spatial tables folder').value;
 set('Spatial images folder',root+'/data/codex');
 await until(()=>rows().length===1&&button(rows()[0],'second_CODEX.qptiff'));
 set('Spatial tables folder',root+'/alternative_tables');
 await until(()=>by('Cell table for second')?.value.endsWith('/alternative_tables/second_cell_features.parquet'));
 assert(state.layers.find(l=>l.type==='image').item===priorImage&&state.viewer.viewport.getBounds().toString()===priorBounds,'Changing image and table folders preserves the current image and view');
 const sharedFolders=await fetch('/spacemap-viewer/api/studio/browser-check/phenotyping_inputs?summaries=0').then(r=>r.json());
 assert(sharedFolders.folders.images===root+'/data/codex'&&sharedFolders.folders.tables===root+'/alternative_tables','User-chosen QPTIFF and table directories stay synchronized with Phenotyping');
 button(rows()[0],'second_CODEX.qptiff').click();
 await until(()=>state.imagesReady&&state.layers.some(l=>l.type==='image'&&l.name==='second_CODEX.qptiff'&&l.item?.getFullyLoaded()));
 assert(true,'An explicit image selection opens the user-chosen unregistered QPTIFF');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__=='__main__':
    main(DRIVER,studio=True,setup=setup,timeout=240,capture='/tmp/spatial-inputs-final.png')
