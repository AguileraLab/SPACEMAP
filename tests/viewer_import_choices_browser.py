"""Real CSV drop/upload: register first, explicitly select/link later; keep current edits and geometry."""
from viewer_threshold_browser import main

DRIVER = r'''<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<800;i++){if(await f())return;await wait(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const control=n=>document.querySelector('[aria-label="'+n+'"]');
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const button=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const drop=file=>{const data=new DataTransfer();data.items.add(file);document.querySelector('#osd').dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.cellsReady&&s.studio);await s.phenotypingLink;
 const cells=activeCellLayer(),table=cells.table,image=s.layers.find(l=>l.type==='image');
 s.layerGroups.cells=true;buildPanel();await openCellData(cells);await until(()=>dataView.queryReady);
 await showViewerCellGroups('phenotype_training',['A'],['A','B']);await until(()=>dataView.queryReady&&cells.color_by==='phenotype_training');
 const key=dataView.sourceKey,query=JSON.stringify(dataBody({})),selection=cellColourQuery(cells),bindings=JSON.stringify(cells.tables);
 const contents='label,x_centroid,y_centroid,old_type,score\n1,20,20,Old A,99\n2,60,20,Old B,88';
 // Dropping and even dismissing the linking dialog must preserve the current source.
 drop(new File([contents],'old_data.csv',{type:'text/csv'}));
 await until(()=>document.querySelector('.cell-data-dialog'));
 check(dataView.sourceKey===key&&JSON.stringify(dataBody({}))===query&&cellColourQuery(cells)===selection,'Dropping a CSV preserves Data rows, filters and viewer colours');
 check(JSON.stringify(cells.tables)===bindings,'Unlinked imports do not modify any cell table binding');
 button(document.querySelector('.cell-data-dialog'),'Keep in list').click();
 const imported=dataView.opened.find(o=>o.name==='old_data.csv');
 check(imported&&[...control('Import cells').options].some(o=>o.value==='file:'+imported.file),'Imported CSV is available in the bottom source list');
 // The Data toolbar import uses the same registration-only semantics, even with unsaved edits.
 dataDirtySources.add(key);
 await dataOpenUpload(new File([contents.replace('99','999')],'older_data.csv',{type:'text/csv'}));
 check(dataView.sourceKey===key&&dataDirtySources.has(key)&&cells.table===table,'Data import preserves the active table and unsaved edits');
 dataDirtySources.delete(key);
 change(control('Import cells'),'file:'+imported.file);
 await until(()=>dataView.queryReady&&dataView.source?.file===imported.file);
 check(control('Data source').textContent==='Imported: old_data.csv'&&dataView.result.columns.includes('old_type'),'Explicit choice displays the imported file with its own name and columns');
 check(cells.table===table&&cellColourQuery(cells)===selection&&activeCellLayer()===cells,'Reviewing an old table leaves the viewer source and colours unchanged');
 check(control('Image follows the filters').disabled,'An unlinked table cannot filter unrelated cells');
 // Linking adds a stable table ID and does not choose it.
 button(dataView.head,'Link to Cells…').click();await until(()=>document.querySelector('.cell-data-dialog'));
 change(control('Show as'),'cells:label');button(document.querySelector('.cell-data-dialog'),'Add to Cells').click();
 await until(()=>!document.querySelector('.cell-data-dialog')&&cells.tables.some(t=>t.name==='Data: old_data.csv'));
 const linked=cells.tables.find(t=>t.name==='Data: old_data.csv');
 check(linked.id.startsWith('import_')&&cells.table===table&&dataView.source.file===imported.file,'Linking keeps both active sources unchanged');
 const original=await (await fetch(api('/cells/'+cells.token+'/legend?t='+encodeURIComponent(table)+'&col=phenotype_training'))).json();
 check(original.items.some(i=>i.value==='A')&&original.items.some(i=>i.value==='B'),'Original measurements remain intact after linking');
 change(control('Import cells'),'table:'+cells.id+'|'+linked.id);
 await until(()=>dataView.queryReady&&cells.table===linked.id&&dataView.source?.table===linked.id);
 check(dataView.result.columns.includes('old_type')&&control('Cell data').value===cells.id+'|'+linked.id,'Explicit linked choice synchronizes Data and Cells');
 await until(()=>control('Colour the cells by')?.querySelector('option[value=old_type]'));change(control('Colour the cells by'),'old_type');
 await until(()=>control('Show group Old A'));
 check(true,'Imported measurements can colour their matched cells');
 check(image===s.layers.find(l=>l.type==='image')&&image.item,'Imports preserve the open image');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}})();
</script>'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=150)
