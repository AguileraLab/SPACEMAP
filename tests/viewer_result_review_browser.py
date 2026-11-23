"""Real browser: multi-group review actions in every shared cluster explorer."""
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui.studio import cluster_profiles, review_images, cell_layers
    from test_studio_chat import _job
    table=root/'cells.csv'; frame=pd.read_csv(table)
    methods=['leiden','som','vortex','subphenotype','neighbourhoods']
    for method in methods: frame['phenotype_'+method+'_clusters']=['Cluster 0']*16+['Cluster 1']*16+['Cluster 2']*8
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    # Disposable geometry is the viewer's actual 40-cell overlay.
    cell_layers.own=lambda *args:str(root/'cells.geojson')
    cell_layers.table_of=lambda *args:str(table)
    review_images.plan=lambda channel:{'rows':[{'roi':'roi','image':'roi.ome.tiff','image_path':str(root/'roi.ome.tiff'),'table_path':str(table)}]}
    for method in methods:
        col='phenotype_'+method+'_clusters'
        clusters,profile=cluster_profiles.build(frame,['M0','M1'],col,markers=['M0','M1'],name=str)
        _job(root,method,{'table':str(table),'column':col,'phenotype_column':'phenotype_'+method,
            'markers':['M0','M1'],'clusters':clusters,'cells':40,'images':['roi'],'profile':profile,'where':{}})


DRIVER=r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
const buttons=(box,text)=>[...box.querySelectorAll('button')].filter(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
if(!new URLSearchParams(location.search).has('review'))setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 const api='/spacemap-viewer/api/studio/browser-check/review';
 const queue=async()=>await(await fetch(api)).json();
 await until(()=>s.imagesReady&&tools().querySelector('[aria-label="Phenotyping method"]'));
 for(const method of ['leiden','som','vortex','subphenotype']){
  change(tools().querySelector('[aria-label="Phenotyping method"]'),method==='subphenotype'?'subphenotyping':method);
  await until(()=>tools().querySelector('[data-group="Cluster 0"]'));
  const row=name=>tools().querySelector('[data-group="'+name+'"]');
  row('Cluster 0').click();row('Cluster 1').dispatchEvent(new MouseEvent('click',{bubbles:true,shiftKey:true}));
  await until(()=>buttons(tools(),'Flag for review').some(b=>!b.disabled));
  assert(buttons(document.querySelector('#studio-figures'),'Flag for review').some(b=>!b.disabled),method+' exposes review beside the selected heatmap groups');
  buttons(tools(),'Flag for review').find(b=>!b.disabled).click();
  await until(async()=>{const q=await queue();return q.items.filter(i=>i.kind===(method==='subphenotype'?'subcluster':method)).length===2;});
  assert(true,method+' flags both selected groups');
 }
 const q=await queue(),item=q.items.find(i=>i.kind==='subcluster');
 const context=await(await fetch(api+'?action=result_context&id='+item.id)).json();
 assert(context.profile.rows.length===3&&context.groups.map(g=>g.cells).join()==='16,16,8','Review retains the complete heatmap and exact image counts');
 const response=await fetch(api+'?action=view&id='+item.id);
 assert(response.ok,'Review builds an image-linked cell layer');
 const view=await response.json();
 assert(view.spec.layers.some(l=>l.type==='cells'&&l.color_by==='Review group'),'The reviewed group is available for cell colouring');
 const shared=await(await fetch(api,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'share',name:'Fixture reviewer',items:[item.id]})})).json();
 assert(!!shared.url,'Selected result can be shared');
 const base=shared.url.replace(location.origin,'').replace(/\/$/,'');
 const publicContext=await(await fetch(base+'/result_context?id='+item.id)).json();
 assert(publicContext.profile.rows.length===3&&!JSON.stringify(publicContext).includes('/tmp/'),'Shared context includes the full heatmap without source paths');
 const outside=q.items.find(i=>i.kind==='leiden');
 assert((await fetch(base+'/result_context?id='+outside.id)).status===403,'Unshared result context is inaccessible');
 const child=window.open(shared.url,'result-review-fixture','width=1500,height=950');
 await until(()=>child?.__state?.imagesReady&&child.document.querySelector('.review-list tr[data-id]'));
 child.document.querySelector('.review-list tr[data-id]').click();
 await until(()=>child.document.querySelectorAll('.review-result-heatmap [data-review-group]').length===3&&child.__state.layers.some(l=>l.type==='cells'&&l.filters?.[0]?.values.includes('Cluster 0')));
 assert(true,'Reviewer sees the full rendered heatmap and the selected cells');
 const other=child.document.querySelector('[data-review-group="Cluster 2"]');
 buttons(other,'Suggest review')[0].click();
 await until(async()=>(await queue()).items.some(i=>i.kind==='subcluster'&&i.state==='requested'));
 assert(true,'Reviewer can nominate an unflagged group for owner approval');
 const label=child.document.querySelector('[aria-label="Review label"]');label.value='Reviewed group';
 buttons(child.document,'Use label')[0].click();
 await until(async()=>(await queue()).items.find(i=>i.id===item.id)?.decision?.label==='Reviewed group');
 assert((await queue()).items.find(i=>i.id===item.id).state==='answered','Reviewer decision awaits owner approval');
 child.close();
 document.querySelector('[data-step=tab8]').click();
 await until(()=>buttons(tools(),'Neighbourhoods').length);buttons(tools(),'Neighbourhoods')[0].click();
 await until(()=>tools().querySelector('[data-group="Cluster 0"]'));
 tools().querySelector('[data-group="Cluster 0"]').click();
 await until(()=>buttons(tools(),'Flag for review').some(b=>!b.disabled));
 buttons(tools(),'Flag for review').find(b=>!b.disabled).click();
 await until(async()=>(await queue()).items.some(i=>i.kind==='neighbourhood'));
 assert(true,'Neighbourhood groups use the same review action');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''
if __name__=='__main__':main(DRIVER,studio=True,timeout=180,setup=setup,allow_popups=True,screenshot=False)
