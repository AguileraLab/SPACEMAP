"""Real browser: new job columns, draft conflicts and explicit recovery."""
from viewer_studio_leiden_browser import setup

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+' '+dataView.status?.textContent);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const ready=()=>dataView.queryReady&&!dataView.loading&&!dataView.busy;
const layer=()=>state.layers.find(l=>l.type==='cells');
const reload=()=>[...dataView.status.querySelectorAll('button')].find(b=>b.textContent==='Reload saved data');
setTimeout(async()=>{try{
 await until(()=>state.imagesReady&&state.cellsReady&&state.studio&&layer()?.combinedTable);
 showDataTab();await dataChoose('combined');await until(ready);
 const id=dataView.result.rows[0][0];
 assert(await dataMutate({kind:'cells',changes:[{row:id,col:'M0',value:77}]}),'Create an unsaved Data draft');
 dataView.columns=['label','M0'];dataView.filters=[{col:'label',lo:1,hi:2}];await dataLoad();
 const scope=JSON.stringify(dataView.rows),filters=JSON.stringify(dataView.filters);
 await fetch('/publish-result');await dataRefreshCombined();await until(ready);
 assert(dataView.result.columns.includes('phenotype_new')&&dataView.result.edited===1,'A completed prediction appears beside an unsaved draft');
 assert(JSON.stringify(dataView.rows)===scope&&JSON.stringify(dataView.filters)===filters&&dataView.columns.join(',')==='label,M0','Result refresh preserves scope, filters and selected columns');
 assert(dataView.result.rows[0][2]===77&&layer().combinedTable,'Draft values and image linking survive new predictions');
 assert((await columnsFor(layer())).includes('phenotype_new'),'Layers offers the new prediction column');
 dataView.columns=['label','M0','phenotype_new'];await dataLoad(true);
 assert(document.querySelector('.data-grid th[data-column="phenotype_new"]')&&
   document.querySelector('.data-grid td[data-col="phenotype_new"]')?.textContent==='Predicted',
   'The rendered grid displays the new saved prediction values');
 await dataHistory('undo');await until(ready);
 assert(dataView.result.columns.includes('phenotype_new')&&dataView.result.rows[0][2]!==77,'Undo changes the draft without losing the prediction');
 await dataHistory('redo');await until(ready);
 await fetch('/publish-result?conflict=1');await dataRefreshCombined();
 await until(reload);
 assert(layer().combinedTable&&!document.querySelector('[aria-label="Retry cell-table link"]'),'An actual Data conflict never blocks the image link');
 assert(dataView.result.rows[0][2]===77&&dataView.result.edited===1,'The conflicting draft is still retained');
 window.confirm=()=>false;reload().click();await delay(150);
 assert(reload()&&dataView.result.edited===1,'Cancelling recovery preserves the draft');
 window.confirm=()=>true;reload().click();await until(ready);
 assert(!reload()&&dataView.result.edited===0&&dataView.result.columns.includes('phenotype_new'),'Explicit reload opens the latest saved data');
 assert(dataView.result.rows[0][dataView.result.shown.indexOf('M0')+1]===9,'Reload displays saved values rather than the discarded draft');
 // An earlier refresh must not replace a later explicit Data-source choice.
 const other=await (await fetch('/publish-result?file=1')).json();
 (dataView.opened||=[]).push({file:other.token,name:'other_loaded_table.csv'});
 const nativeFetch=window.fetch;let release,held=false;
 window.fetch=async(url,...args)=>{const response=await nativeFetch(url,...args);if(!held&&String(url).includes('/data/info')){held=true;await new Promise(r=>release=r);}return response;};
 const refreshing=dataRefreshCombined();await until(()=>held);
 const local=dataSources().find(s=>s.source.file===other.token);
 await dataChoose(local.key);await until(ready);release();await refreshing;
 assert(dataView.sourceKey===local.key&&dataView.info.name==='other_loaded_table.csv','A delayed refresh respects a newer explicit source choice');
 window.fetch=nativeFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''

if __name__ == '__main__':
    import inspect
    import viewer_threshold_browser
    source=inspect.getsource(viewer_threshold_browser)
    source=source.replace('        async def slow(request):', '''        async def publish_result(request):
            from spacemap import pending_columns
            from starlette.responses import JSONResponse
            if 'file' in request.query_params:
                return JSONResponse({'token':registry.register_object('datafile',path=str(root/'other_loaded_table.csv'))})
            path=root/'cells.csv'
            frame=pd.read_csv(path)
            column='M0' if 'conflict' in request.query_params else 'phenotype_new'
            pending_columns.save(path,{column:[9 if column=='M0' else 'Predicted']*len(frame)},
                                 anchor_column='cell_id',anchor_values=frame.cell_id)
            return JSONResponse({'ok':True})

        async def slow(request):''')
    source=source.replace('                Route("/result", result),', '                Route("/result", result), Route("/publish-result",publish_result),')
    namespace={'__name__':'refresh_harness','__file__':viewer_threshold_browser.__file__}
    exec(compile(source,viewer_threshold_browser.__file__,'exec'),namespace)
    namespace['main'](DRIVER,studio=True,timeout=180,setup=setup,screenshot=False)
