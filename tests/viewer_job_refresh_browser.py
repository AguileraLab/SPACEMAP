"""Background job progress preserves the rendered viewer; saved results refresh once."""
from viewer_threshold_browser import main

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(const end=Date.now()+60000;Date.now()<end;){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 await state.phenotypingLink;
 const method=document.querySelector('[aria-label="Phenotyping method"]');
 method.value='subphenotyping';method.dispatchEvent(new Event('change'));
 await until(()=>document.querySelector('[aria-label="Sub-phenotyping marker values"]'));
 const cells=state.layers.find(l=>l.type==='cells'),image=state.layers.find(l=>l.type==='image');
 showDataTab();await dataChoose('combined');
 await state.studio.refreshLive();
 setCellColour(cells,'phenotype_ref1');
 await until(()=>cells.item?.getFullyLoaded()&&!cells.retiring?.length&&dataView.queryReady);
 await delay(1000);
 const nativeFetch=window.fetch;let polls=0,links=0;
 const started=new Date().toISOString();
 const running=['subphenotype','leiden','som','vortex','mr_ml','consensus_ml','resolve_all','name_clusters','reference_collection'];
 const jobs=running.map((kind,i)=>({id:'progress-'+i,kind,status:'running',started,progress:0,message:''}));
 window.fetch=async(url,...args)=>{
   if(String(url).endsWith('/studio/browser-check/jobs')){
     polls++;return new Response(JSON.stringify({jobs:jobs.map(job=>({...job,
       progress:Math.min(.9,polls/100),message:'Computing stage '+polls}))}),{headers:{'Content-Type':'application/json'}});
   }
   if(String(url).endsWith('/phenotyping_link'))links++;
   return nativeFetch(url,...args);
 };
 const before={version:state.versions[cells.id],item:cells.item,image:image.item,channels:JSON.stringify(image.channels),
   generation:state.initGeneration,zoom:state.viewer.viewport.getZoom(),centre:state.viewer.viewport.getCenter()};
 const canvas=state.viewer.drawer.canvas;
 const pixels=canvas.toDataURL();
 await until(()=>polls>=3);await delay(1500);
 check(state.versions[cells.id]===before.version&&cells.item===before.item,
   'Sub-phenotyping and other phenotyping progress leave the current cell tiles unchanged');
 check(links===0,'Progress never reloads the cell-data links');
 check(canvas.toDataURL()===pixels,'Rendered viewer pixels remain identical through progress polls');
 check(image.item===before.image&&JSON.stringify(image.channels)===before.channels&&state.initGeneration===before.generation&&
   state.viewer.viewport.getZoom()===before.zoom&&state.viewer.viewport.getCenter().distanceTo(before.centre)<1e-8,
   'Progress preserves the image, marker settings, zoom and pan');
 await dataRefreshCombined();await dataRefreshCombined();await delay(400);
 check(state.versions[cells.id]===before.version&&cells.item===before.item,
   'Explicit refresh of unchanged saved data also leaves cell tiles intact');
 const beforeFilterItem=cells.item;
 cells.filters=[{col:'M0',lo:1,hi:300}];cells.hidden=new Set(['B']);bump(cells);
 dataView.filters=[{col:'M0',lo:1,hi:300}];dataView.sort={col:'M0',desc:true};
 dataView.columns=['label','M0'];await dataLoad();
 await until(()=>cells.item!==beforeFilterItem&&cells.item.getFullyLoaded()&&!cells.retiring?.length);await delay(500);
 const cellFilters=JSON.stringify(cells.filters),dataFilters=JSON.stringify(dataView.filters),source=cells.table;
 const oldItem=cells.item,oldPixels=canvas.toDataURL(),version=state.versions[cells.id],linksBefore=links;
 const nativeAdd=state.viewer.addTiledImage;let delayed=false,oldVisible=false;
 state.viewer.addTiledImage=function(options){
   if(!delayed){delayed=true;oldVisible=oldItem.getOpacity()>0;
     setTimeout(()=>nativeAdd.call(this,options),1200);return;}
   return nativeAdd.call(this,options);
 };
 await nativeFetch('/publish-job-result');
 jobs[0].status='done';jobs[0].finished=new Date().toISOString();
 await until(()=>delayed);await delay(200);
 check(oldVisible&&oldItem.getOpacity()>0&&canvas.toDataURL()===oldPixels,
   'Saved results keep the rendered overlay visible while replacement tiles are delayed');
 await until(()=>cells.item!==oldItem&&cells.item.getFullyLoaded()&&!cells.retiring?.length&&dataView.result?.columns.includes('phenotype_job'));
 state.viewer.addTiledImage=nativeAdd;
 check(links>=linksBefore+1&&links<=linksBefore+2&&state.versions[cells.id]===version+1,
   'One finished result refreshes once even while other jobs are still running');
 check(cells.table===source&&cells.color_by==='phenotype_ref1'&&cells.hidden.has('B')&&JSON.stringify(cells.filters)===cellFilters&&
   JSON.stringify(dataView.filters)===dataFilters&&dataView.sort.desc&&dataView.columns.join()==='label,M0',
   'Saved results retain cell colouring, source, hidden groups and Data filters/sorting');
 dataView.columns=['label','phenotype_job'];await dataLoad(true);
 check(document.querySelector('.data-grid td[data-col="phenotype_job"]')?.textContent==='Recovered',
   'New saved values are rendered in Data');
 const once=state.versions[cells.id];await dataRefreshCombined();await delay(400);
 check(state.versions[cells.id]===once,'Repeated refresh of the completed result does not redraw it');
 // A result overwritten in the same column must still invalidate its data.
 document.querySelector('[data-step=tab1]').click();await delay(700);
 await nativeFetch('/publish-job-result?value=Updated');
 const linksAtOtherStep=links;
 jobs[1].status='done';jobs[1].finished=new Date().toISOString();
 await until(()=>dataView.result?.rows[0]?.[2]==='Updated');
 check(links>=linksAtOtherStep+1&&links<=linksAtOtherStep+2&&state.versions[cells.id]===once+1,
   'An overwritten result updates once while another Studio step is open');
 // A late saved-data request cannot return the viewer to its earlier source.
 let release,held=false;
 window.fetch=async(url,...args)=>{
   const response=await nativeFetch(url,...args);
   if(String(url).endsWith('/phenotyping_link')){held=true;await new Promise(r=>release=r);}
   return response;
 };
 await nativeFetch('/publish-job-result?value=Latest');
 const refreshing=dataRefreshCombined();await until(()=>held);
 const alternative={id:'local-choice',name:'Local data'};cells.tables.push(alternative);
 claimCellView(cells,'Cell data changed');cells.table=alternative.id;cells.color_by='M0';
 const laterVersion=state.versions[cells.id];
 release();await refreshing;
 check(cells.table===alternative.id&&cells.color_by==='M0'&&state.versions[cells.id]===laterVersion,
   'A delayed result refresh respects the later explicit cell-data choice');
 window.fetch=nativeFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,polls})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>'''

if __name__ == '__main__':
    import inspect
    import viewer_threshold_browser
    source = inspect.getsource(viewer_threshold_browser)
    source = source.replace('        async def slow(request):', '''        async def publish_job_result(request):
            from spacemap import pending_columns
            from starlette.responses import JSONResponse
            frame=pd.read_csv(table)
            value=request.query_params.get('value', 'Recovered')
            pending_columns.save(table, {'phenotype_job':[value]*len(frame)},
                                 anchor_column='cell_id',anchor_values=frame.cell_id)
            return JSONResponse({'ok':True})

        async def slow(request):''')
    source = source.replace('                Route("/result", result),',
                            '                Route("/result", result), Route("/publish-job-result", publish_job_result),')
    namespace = {'__name__':'job_refresh_harness','__file__':viewer_threshold_browser.__file__}
    exec(compile(source, viewer_threshold_browser.__file__, 'exec'), namespace)
    namespace['main'](DRIVER, studio=True, timeout=180, screenshot=False)
