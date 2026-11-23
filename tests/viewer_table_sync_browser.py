"""Saved-result discard, Data deletion, and reload keep image tables in sync."""
from viewer_threshold_browser import main
from viewer_studio_leiden_browser import setup as prepare


def setup(root):
    import pandas as pd
    from spacemap import labels, pending_columns

    prepare(root)
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    other = frame.iloc[:2].copy()
    other['batch_id'] = 'other_image'
    other['cell_id'] = ['other_1', 'other_2']
    frame = pd.concat([frame, other], ignore_index=True)
    frame.to_csv(table, index=False)
    pending_columns.save(table, {
        'phenotype_discard': ['Delete'] * len(frame),
        'phenotype_keep': ['Keep'] * len(frame),
    }, anchor_column='cell_id', anchor_values=frame.cell_id)
    labels.append(table, [{'row':0, 'column':'phenotype_ref2', 'value':'Reviewed'}])


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+' '+dataView.status?.textContent);};
let checks=Number(sessionStorage.getItem('sync-checks')||0);
const assert=(value,text)=>{if(!value)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
const ready=()=>dataView.queryReady&&!dataView.loading&&!dataView.busy;
const colour=()=>document.querySelector('[aria-label="Colour cells by"]');
const hasOption=name=>[...colour().options].some(o=>o.value===name);
const fetchBefore=window.fetch;let deferredLink=false;
window.fetch=async(url,...args)=>{
 const response=await fetchBefore(url,...args);
 if(!deferredLink&&String(url).includes('/phenotyping_link')){
   const result=await response.clone().json();
   const tokens=Object.keys(result.layers||{});
   if(tokens.length){
     deferredLink=true;
     for(const token of tokens)result.layers[token]={linked:false,retryable:true,reason:'Making a fast copy of the cell table'};
     return new Response(JSON.stringify(result),{status:200,headers:{'Content-Type':'application/json'}});
   }
 }
 return response;
};
window.confirm=()=>true;
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&window.__state.cellsReady&&window.__state.studio&&colour());
 showDataTab();
 await until(()=>ready()&&dataView.source?.combined&&dataView.result?.rows.length===40);
 const cells=()=>window.__state.layers.find(l=>l.type==='cells');
 await until(()=>deferredLink&&cells()?.combinedTable);
 assert(cells().combinedTable&&!document.querySelector('[aria-label="Retry cell-table link"]'),'A table that was still preparing automatically reconnects without reloading');
 if(!sessionStorage.getItem('sync-reloaded')){
   cells().combinedTable=null;cells().combinedLinkPending=false;cells().combinedLinkNote='Temporary read failure';buildPanel();
   document.querySelector('[aria-label="Retry cell-table link"]').click();
   await until(()=>cells().combinedTable&&!document.querySelector('[aria-label="Retry cell-table link"]'));
   assert(true,'Retry link reconnects the existing cell layer');
   assert(dataView.result.columns.includes('phenotype_keep')&&dataView.result.columns.includes('phenotype_ref2'),'Data includes both pending predictions and manual-label sidecars');
   document.querySelector('[aria-label="Show entire dataset"]').click();
   await until(()=>ready()&&dataView.result.rows.length===42);
   assert(!dataView.rows.this_image&&!dataView.follow,'Whole dataset shows both images and keeps the chosen table scope');
   document.querySelector('[aria-label="Show current image data"]').click();
   await until(()=>ready()&&dataView.result.rows.length===40);
   assert(dataView.rows.this_image,'Current image restores the open image rows');
   await dataChoose('combined');await until(ready);
   cells().color_by='phenotype_discard';cells().filters=[{col:'phenotype_discard',values:['Delete']}];bump(cells());
   dataView.filters=[{col:'phenotype_discard',values:['Delete']}];await dataLoad();
   document.querySelector('[aria-controls="studio-jobs"]').click();
   const discard=()=>[...document.querySelectorAll('#studio-jobs .studio-folder')].find(row=>row.textContent.includes('phenotype_discard'))?.querySelector('button');
   await until(discard);discard().click();
   await until(()=>ready()&&!dataView.result.columns.includes('phenotype_discard')&&!hasOption('phenotype_discard'));
   assert(dataView.result.rows.length===40&&!cells().color_by&&!cells().filters.length,'Discard removes obsolete colours and filters without hiding cells');
   assert(dataView.result.columns.includes('phenotype_keep')&&!discard(),'Kept results stay readable while discarded results leave Jobs and Data');
   assert(!(await columnsFor(cells())).includes('phenotype_discard'),'The cell-layer column picker also drops the discarded column');
   sessionStorage.setItem('sync-checks',String(checks));sessionStorage.setItem('sync-reloaded','yes');
   await delay(500);location.reload();return;
 }
 assert(dataView.result.columns.includes('phenotype_keep')&&!dataView.result.columns.includes('phenotype_discard'),'After a real page reload the table and retained pending results reconnect automatically');
 await until(()=>hasOption('phenotype_ref2'));
 assert(await dataMutate({kind:'delete_columns',columns:['phenotype_ref2']}),'Delete the saved reference column in Data');
 await until(()=>ready()&&!hasOption('phenotype_ref2'));
 assert(!(await columnsFor(cells())).includes('phenotype_ref2'),'A draft deletion leaves both colour menus immediately');
 await dataHistory('undo');await until(()=>ready()&&hasOption('phenotype_ref2'));
 assert((await columnsFor(cells())).includes('phenotype_ref2'),'Undo restores the column to both Data and cell colouring');
 await dataMutate({kind:'delete_columns',columns:['phenotype_ref2']});
 assert(await dataSave('overwrite'),'Save makes the column deletion permanent');
 await until(()=>ready()&&!hasOption('phenotype_ref2'));
 const found=await (await fetch('/spacemap-viewer/api/studio/browser-check/phenotyping_inputs')).json();
 const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/saved')).json();
 assert(!found.columns.includes('phenotype_ref2')&&found.columns.includes('phenotype_keep'),'The image overview uses the saved schema and retains other results');
 assert(!saved.tables.some(t=>t.name.endsWith('.labels.parquet')),'Label sidecars are never offered as data tables');
 assert((await columnsFor(cells())).includes('phenotype_keep')&&!dataView.result.columns.includes('phenotype_ref2'),'Cell colouring and Data agree after save');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=240, setup=setup, screenshot=False)
