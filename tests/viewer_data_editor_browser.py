"""Real-browser editing workflows using a disposable multi-ROI cell table."""
from viewer_threshold_browser import main

def setup(root):
    import pandas as pd
    frame=pd.DataFrame({'batch_id':['A']*240+['B']*240,'label':list(range(1,241))*2,
                        'phenotype_som':['Epithelia']*120+['B']*120+['Epithelia']*120+['CD8']*120,
                        'marker':[float(i) for i in range(480)],'unwanted':['remove']*480})
    frame.to_parquet(root/'edit_me.parquet',index=False,row_group_size=73)

DRIVER=r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const button=(box,text)=>Array.from(box.querySelectorAll('button')).find(b=>b.textContent===text);
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
const header=name=>document.querySelector('.data-grid th[data-column="'+name+'"]');
const context=(element)=>element.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:700,clientY:650}));
const dialog=()=>document.querySelector('dialog.data-edit-dialog');
const ready=()=>!dataView.busy&&!dataView.loading&&dataView.queryReady;
const valueInput=value=>document.querySelector('input[aria-label="Rename value '+value+'"]');
const menuReady=()=>ready()&&document.querySelector('#ctxmenu .data-value-name:not(:disabled)');
const allGroups=async()=>(await postJSON(dataRoute('summary'),dataBody({column:'phenotype_som',rows:{this_image:false},filters:[]}))).groups;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady&&window.__state.studio);
 showDataTab();await until(()=>dataView.result&&ready());
 const source=await (await fetch('/editor-source')).json();
 dataOpenFile(source);await until(()=>ready()&&dataView.info?.name==='edit_me.parquet');
 assert(dataView.result.total===240&&dataView.result.rows.length===200,'The image subset remains paged');
 button(dataView.head,'All data').click();await until(()=>ready()&&dataView.result.matched===480);
 assert(dataView.result.rows.length===200&&!dataView.follow&&!dataView.onlySelected,'All data spans images with only one page in the browser');
 change(document.querySelector('[aria-label="Data page"]'),'3');await until(()=>ready()&&dataView.result.offset===400);
 assert(dataView.result.rows.length===80,'Page navigation reaches the final dataset rows');
 button(dataView.head,'All data').click();await until(()=>ready()&&dataView.result.offset===0);
 // Keep one image visible: the default rename must still change both images.
 dataView.rows={this_image:true,column:'batch_id',value:'A'};await dataLoad();
 document.querySelector('[aria-label="Filter phenotype_som"]').click();await until(()=>menuReady());
 let name=valueInput('Epithelia');
 const rect=name.getBoundingClientRect(),check=name.previousElementSibling.getBoundingClientRect(),countRect=name.nextElementSibling.getBoundingClientRect();
 assert(name.value==='Epithelia'&&rect.width>120&&check.width>=12&&rect.left>=check.right&&countRect.left>=rect.right,'Category names, filter checkboxes and counts are visible without overlap');
 const all=button(document.querySelector('#ctxmenu'),'All'),none=button(document.querySelector('#ctxmenu'),'None');
 assert(all.offsetWidth<70&&none.offsetWidth<70,'All and None remain compact instead of squeezing the names');
 assert(!document.querySelector('[aria-label="Rename selected cells only"]').checked,'Category renaming defaults to the whole dataset');
 name.focus();name.value='Epithelial';name.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>menuReady()&&valueInput('Epithelial'));
 let groups=await allGroups();
 assert(groups.some(g=>g[0]==='Epithelial'&&g[1]===240)&&!groups.some(g=>g[0]==='Epithelia')&&dataView.result.matched===240,'Default inline rename changes both images while only one image is displayed');
 assert(dataView.result.edited===1&&!dialog(),'Enter commits one undoable edit without a second dialog');
 closeContextMenu();button(dataView.head,'All data').click();await until(()=>ready()&&dataView.result.matched===480);
 button(dataView.head,'Undo').click();await until(()=>ready()&&dataView.result.edited===0);
 assert(dataView.result.rows[0][3]==='Epithelia','Undo restores the original category');
 change(dataView.head.querySelector('[aria-label="More data tools"]'),'redo');await until(()=>ready()&&dataView.result.edited===1);
 dataView.selected=new Set([0,1]);dataDraw();
 document.querySelector('[aria-label="Filter phenotype_som"]').click();await until(()=>menuReady());
 name=valueInput('Epithelial');name.focus();name.value='Cancelled';name.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));await delay(30);
 assert(dataView.result.edited===1&&name.value==='Epithelial','Escape cancels an inline category edit');
 // The scope click happens after blur: it must be honoured before committing.
 name.focus();name.value='Selected phenotype';name.dispatchEvent(new Event('input',{bubbles:true}));
 name.dispatchEvent(new FocusEvent('blur'));document.querySelector('[aria-label="Rename selected cells only"]').click();
 await delay(1000);
 assert(!dataView.lastError,'Selected rename request succeeds: '+dataView.lastError);
 assert(!dataView.busy,'Selected rename finishes');
 assert(valueInput('Selected phenotype'),'The selected rename refreshes the value list');
 await until(()=>menuReady()&&valueInput('Selected phenotype'));
 groups=await allGroups();
 assert(groups.some(g=>g[0]==='Selected phenotype'&&g[1]===2)&&groups.some(g=>g[0]==='Epithelial'&&g[1]===238),'Selected cells only changes the two selected rows, including when checked on blur');
 closeContextMenu();button(dataView.head,'Undo').click();await until(()=>ready()&&dataView.result.edited===1);
 document.querySelector('[aria-label="Filter phenotype_som"]').click();await until(()=>menuReady());
 assert(!document.querySelector('[aria-label="Rename selected cells only"]').checked,'Reopening the menu defaults back to the entire dataset');
 name=valueInput('Epithelial');name.focus();name.value='B';name.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>menuReady()&&!valueInput('Epithelial'));
 assert(valueInput('B')&&document.querySelectorAll('input[aria-label="Rename value B"]').length===1&&(await allGroups()).some(g=>g[0]==='B'&&g[1]===360),'Renaming to an existing category merges it and refreshes its count');
 closeContextMenu();button(dataView.head,'Undo').click();await until(()=>ready()&&dataView.result.edited===1);
 document.querySelector('[aria-label="Filter phenotype_som"]').click();await until(()=>menuReady());
 name=valueInput('Epithelial');name.focus();name.value='Reviewed';name.dispatchEvent(new Event('input',{bubbles:true}));
 document.body.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));
 await until(()=>ready()&&dataView.result.edited===2);
 assert(!document.querySelector('#ctxmenu')&&(await allGroups()).some(g=>g[0]==='Reviewed'&&g[1]===240),'Clicking away commits the name before the menu is removed');
 button(dataView.head,'Undo').click();await until(()=>ready()&&dataView.result.edited===1);
 context(header('phenotype_som'));await until(()=>document.querySelector('[aria-label="Column actions"]'));
 assert(header('phenotype_som').classList.contains('column-selected'),'Right-click highlights the selected column');
 button(document.querySelector('#ctxmenu'),'Rename column…').click();await until(()=>dialog());
 change(dialog().querySelector('[aria-label="New column name"]'),'phenotype_corrected');button(dialog(),'Apply').click();
 await until(()=>ready()&&header('phenotype_corrected')&&!dialog());
 assert(!header('phenotype_som'),'Column header is renamed without changing values');
 context(header('unwanted'));button(document.querySelector('#ctxmenu'),'Delete columns').click();await until(()=>ready()&&!header('unwanted'));
 assert(!dataView.result.columns.includes('unwanted'),'Column deletion removes the complete column');
 button(dataView.head,'Undo').click();await until(()=>ready()&&header('unwanted'));
 assert(dataView.result.columns.includes('unwanted'),'Undo restores the deleted column');
 const row=document.querySelector('.data-grid tbody tr'),id=Number(row.dataset.row);
 row.click();assert(document.querySelector('.data-grid tbody tr')===row,'Selecting a row preserves its DOM for native double-click editing');
 const cell=row.querySelector('[data-col="marker"]');cell.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
 const input=document.querySelector('.data-edit');input.value='123.5';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>ready()&&document.querySelector('.data-grid tbody tr td[data-col="marker"]')?.textContent==='123.5');
 assert(dataView.result.rows[0][4]===123.5,'Double-click editing preserves numeric values');
 context(document.querySelector('.data-grid tbody tr td[data-col="marker"]'));
 button(document.querySelector('#ctxmenu'),'Delete selected rows').click();await until(()=>ready()&&dataView.result.matched===479);
 assert(!dataView.result.rows.some(r=>r[0]===id),'Row deletion uses stable source row IDs');
 button(dataView.head,'Undo').click();await until(()=>ready()&&dataView.result.matched===480);
 assert(dataView.result.rows[0][0]===id,'Undo restores row position and edited measurements');
 const disk=await (await fetch('/editor-disk')).json();assert(disk.columns.includes('phenotype_som')&&disk.first_marker===0,'Staged edits do not change the original file');
 button(dataView.head,'Save').click();await until(()=>ready()&&dataView.result.edited===0);
 const saved=await (await fetch('/editor-disk')).json();
 assert(saved.columns.includes('phenotype_corrected')&&saved.first_marker===123.5&&saved.backups===1,'Save writes the edited table and preserves a backup');
 button(dataView.head,'Save as…').click();await until(()=>dialog());change(dialog().querySelector('[aria-label="Save file name"]'),'reviewed.csv');button(dialog(),'Save').click();
 await until(()=>ready()&&dataView.info?.name==='reviewed.csv'&&!dialog());
 assert(dataView.result.matched===240&&dataView.result.columns.includes('phenotype_corrected'),'Save as creates and opens the named copy');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+JSON.stringify({error:dataView.lastError,selected:[...dataView.selected],busy:dataView.busy,edited:dataView.result?.edited,menu:document.querySelector('#ctxmenu')?.textContent,inputs:[...document.querySelectorAll('#ctxmenu input')].map(i=>({label:i.getAttribute('aria-label'),value:i.value,checked:i.checked,disabled:i.disabled}))})));}},100);
</script>
'''

if __name__=='__main__':
    # Reuse the established harness, adding read-only observations of its own
    # disposable fixture. The production editor routes perform every mutation.
    import inspect
    import viewer_threshold_browser
    source=inspect.getsource(viewer_threshold_browser)
    source=source.replace('        async def slow(request):', '''        async def editor_source(request):
            from starlette.responses import JSONResponse
            path=root/'edit_me.parquet'
            token=registry.register_object('datafile',path=str(path))
            return JSONResponse({'file_token':token,'name':path.name})

        async def editor_disk(request):
            from starlette.responses import JSONResponse
            frame=pd.read_parquet(root/'edit_me.parquet')
            return JSONResponse({'columns':frame.columns.tolist(),'first_marker':float(frame.marker.iloc[0]),
                                 'backups':len(list((root/'previous').rglob('edit_me.parquet')))})

        async def slow(request):''')
    source=source.replace('                Route("/result", result),','                Route("/result", result), Route("/editor-source",editor_source), Route("/editor-disk",editor_disk),')
    namespace={'__name__':'data_editor_harness','__file__':viewer_threshold_browser.__file__}
    exec(compile(source,viewer_threshold_browser.__file__,'exec'),namespace)
    namespace['main'](DRIVER,studio=True,timeout=240,setup=setup)
