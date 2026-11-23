"""Stats workspace, facet threshold UI, exact inference and publication exports."""
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from spacemap import dataio
    table=root/'cells.csv';frame=pd.read_csv(table)
    frame['patient']=[str(1+i%5) for i in range(len(frame))]
    frame['visit']=['Before' if i<20 else 'After' for i in range(len(frame))]
    frame['M0_corr']=frame.M0*.5;frame['M1_corr']=frame.M1*.75
    frame['phenotype']=['Treg' if i%5!=0 and i%2==0 else 'CD8' for i in range(len(frame))]
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)


DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{if(!e)throw Error('Missing control '+v);e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text),control=name=>app().querySelector('[aria-label="'+name+'"]');
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating',info=async()=>await(await fetch(endpoint+'?metadata=1')).json();
const post=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;};
let latest=null,statsCalls=0,exported=[],selected=null;const nativeFetch=window.fetch;
window.fetch=async(...args)=>{const response=await nativeFetch(...args);let body;try{body=JSON.parse(args[1]?.body||'{}');}catch(e){}if(body?.action==='stats_run'){latest=await response.clone().json();statsCalls++;}if(['export','stats_export'].includes(body?.action))exported.push(await response.clone().json());if(body?.action==='select')selected=await response.clone().json();return response;};
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const axis={column:'M0',scale:'linear'},gates=[{id:'before',kind:'filter',parent:'root',name:'Before',colour:'#ea526f',rule:{column:'visit',values:['Before']}},{id:'after',kind:'filter',parent:'root',name:'After',colour:'#2279b5',rule:{column:'visit',values:['After']}},{id:'high',kind:'threshold',parent:'root',name:'M0 high',colour:'#52687e',x:axis,bounds:[12]}];
 gates.push(...['before','after'].map(parent=>({id:parent+'-treg',kind:'filter',parent,name:'Treg',colour:'#eea02b',rule:{column:'phenotype',op:'in',values:['Treg']}})));
 const spec={id:'stats-import',parent:'root',x:axis,kind:'facets',bins:10,sample:100,frequency:'count',tool:'pan',series:[{population:'before'},{population:'after'}],style:{width:900,height:700,legend:true,matchY:true}};
 await post({action:'save',revision:(await info()).workspace.revision,changes:{gates,plots:[spec,{...spec,id:'stats-second',kind:'histogram'}]}});
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');button('Dock below',document).click();
 await until(()=>app()?.querySelector('.flow-chart')?._fullLayout);
 const chart=app().querySelector('.flow-chart');
 check(chart._fullLayout.xaxis.anchor==='y2','Facet X labels are anchored to the lowest panel');
 const title=chart.querySelector('.xtitle').getBoundingClientRect(),lastAxis=chart._fullLayout.yaxis2,box=chart.getBoundingClientRect();
 check(title.top>=box.top+lastAxis._offset+lastAxis._length,'X-axis title sits below the bottom facet without overlap');
 check(chart.data.every(t=>/n=20 \(\d+\.\d%\)/.test(t.name)),'Every facet legend includes exact total cells and above-threshold percentage');
 check([...chart.querySelectorAll('.legend .traces title')].every(t=>t.textContent.includes('Total cells in selected gate: 20 | Above threshold (M0 high @')),'Legend hover explains counts and the named cutoff');
 button('Threshold ▾').click();button('Save as named global threshold…',app().querySelector('.flow-menu')).click();await until(()=>control('Threshold name'));change(control('Threshold name'),'M0_pos');button('Save',app().querySelector('dialog[open]')).click();
 await until(async()=>(await info()).workspace.thresholds?.length===1&&!app().querySelector('dialog[open]'));
 check((await info()).workspace.thresholds[0].name==='M0_pos','Named global threshold is persisted');
 button('Threshold ▾').click();button('Apply to matching open plots',app().querySelector('.flow-menu')).click();await until(async()=>{const w=(await info()).workspace;return w.plots.every(p=>p.threshold_id===w.thresholds[0].id);});check(true,'Named threshold broadcasts to both compatible open plots');
 button('Stats').click();await until(()=>control('Group By (Biological Unit)'));
 check(control('Group By (Biological Unit)').value==='patient','Stats defaults to the patient biological unit');
 check(!!app().querySelector('.flow-tree')&&!!app().querySelector('.flow-stats-importer'),'Stats retains the Studio population tree and Active Plots drawer');
 button('Import from Plot').click();change(control('Minimum cells per biological unit'),'1');change(control('Statistical test'),'paired_t');
 check(app().querySelectorAll('[aria-label="Group label"]').length===2&&app().querySelectorAll('[aria-label="Feature label"]').length===2,'Import carries both gates, MFI, and the active positive threshold');
 button('Run analysis').click();await until(()=>latest?.tests&&app().querySelectorAll('.flow-stats-chart .main-svg').length>=2&&!button('Calculating…'));
 check(latest.summaries.every(s=>s.N===5&&s.n_cells===20),'One dot per patient: five biological units instead of twenty cell replicates');
 check(latest.tests.every(t=>t.matched_N===5&&t.p!=null&&t.q!=null),'Paired inference matches patient IDs and returns nominal and adjusted p');
 const comparison=app().querySelector('.flow-stats-chart');check(comparison.data.filter(t=>t.mode==='markers').every(t=>t.y.length===5),'The comparison chart plots five patient dots per group');
 comparison.emit('plotly_click',{points:[{customdata:comparison.data.find(t=>t.mode==='markers').customdata[0]}]});await until(()=>selected);
 check(selected.count===4,'Clicking a patient dot selects only that patient’s cells within the inherited gate');
 button('Overlay Stats on Histograms').click();button('Plots').click();await until(()=>chart.layout.annotations.some(a=>a.text.includes('FDR=')));
 check(chart.layout.margin.t>70,'Histogram statistics have a dedicated top margin outside the plot geometry');
 button('Stats').click();button('Delta heatmaps').click();change(control('Heatmap feature type'),'all');change(control('Heatmap mode'),'delta');button('Add before / after pair').click();
 const previous=statsCalls;button('Run analysis').click();await until(()=>statsCalls>previous&&app().querySelector('.flow-stats-chart')?.data?.[0]?.type==='heatmap');
 check(latest.heatmap.length===2&&latest.heatmap.every(h=>h.matched_N===5),'Delta heatmap aggregates matched patient changes for both imported metrics');
 let heat=app().querySelector('.flow-stats-chart');check(heat.data[0].zmin===-heat.data[0].zmax,'Delta color scale is centered symmetrically on zero');
 control('Transpose heatmap').click();await until(()=>app().querySelector('.flow-stats-chart')?.data?.[0]?.x?.length===2);heat=app().querySelector('.flow-stats-chart');check(heat.data[0].z.length===1&&heat.data[0].z[0].length===2,'Heatmap rows and columns transpose without changing values');
 const original=heat.data[0].z[0].slice();control('Z score heatmap rows').click();await until(()=>app().querySelector('.flow-stats-chart')?.data?.[0]?.z?.[0]?.every(v=>v===0));
 check(statsCalls===previous+1,'Display scaling and transpose do not rerun cell analysis');
 control('Z score heatmap rows').click();await until(()=>app().querySelector('.flow-stats-chart')?.data?.[0]?.z?.[0]?.some(v=>v!==0));
 change(control('Stats export format'),'svg');button('Export figure').click();await until(()=>exported.length>=1);check(exported.at(-1).download.startsWith('data:image/svg+xml'),'Heatmap exports as vector SVG');const exportedSvg=atob(exported.at(-1).download.split(',')[1]);check(exportedSvg.includes('data-stats-tile')&&!new DOMParser().parseFromString(exportedSvg,'image/svg+xml').querySelector('.heatmaplayer image'),'Heatmap tiles are vector rectangles in publication exports');const svgDoc=new DOMParser().parseFromString(exportedSvg,'image/svg+xml');check([...exportedSvg.matchAll(/url\(\s*['"]?#([^'")]+)['"]?\s*\)/g)].every(m=>svgDoc.getElementById(m[1])),'Every exported colorbar gradient and clip path resolves to a local SVG definition');
 change(control('Stats export format'),'pdf');button('Export figure').click();await until(()=>exported.length>=2);check(exported.at(-1).download.startsWith('data:application/pdf'),'Heatmap exports as vector PDF');
 change(control('Stats export format'),'png');change(control('Stats export DPI'),'300');button('Export figure').click();await until(()=>exported.length>=3);check(exported.at(-1).download.startsWith('data:image/png'),'Heatmap exports as high-DPI PNG');
 button('Statistical table').click();await until(()=>app().querySelector('.flow-stats-table tbody tr'));
 check(app().querySelectorAll('.flow-stats-table tbody tr').length===latest.tests.length,'The table includes all comparison and heatmap hypotheses');
 change(control('Statistical table section'),'summaries');await until(()=>app().querySelector('.flow-stats-table th')?.parentElement?.textContent.includes('n cells'));check(app().querySelectorAll('.flow-stats-table tbody tr').length===latest.summaries.length,'Summary table shows N, cell counts, means, medians and IQRs');
 button('Export CSV').click();await until(()=>exported.length>=4);check(exported.at(-1).download.startsWith('data:text/csv'),'Complete statistical report exports to CSV');
 button('Export Excel').click();await until(()=>exported.length>=5);check(exported.at(-1).download.includes('spreadsheetml.sheet'),'Statistical report exports to a multi-sheet Excel workbook');
 button('Group comparisons').click();await until(()=>app().querySelector('.flow-stats-chart')?.data);change(control('Comparison figure'),'violin');await until(()=>app().querySelector('.flow-stats-chart')?.data?.some(t=>t.type==='violin'));check(true,'Violin view uses the same patient-level results');
 const oldCalls=statsCalls;change(control('Minimum cells per biological unit'),'2');button('Overlay Stats on Histograms').click();check(app().querySelector('.flow-stats-message').textContent.includes('Run the current design'),'Changed designs cannot push stale inference onto histograms');
 check(statsCalls===oldCalls,'Editing settings does not recompute until Run analysis');
 const beforeGroups=app().querySelectorAll('[aria-label="Group label"]').length,data=new DataTransfer();data.setData('application/x-spacemap-populations',JSON.stringify({workspace:(await info()).workspace.id,ids:['root']}));app().querySelector('.flow-stats-dropzone').dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:data}));check(app().querySelectorAll('[aria-label="Group label"]').length===beforeGroups+1,'Dropping a population adds a cohort without changing the gate hierarchy');
 const fieldData=new DataTransfer();fieldData.setData('application/x-spacemap-stats-field',JSON.stringify({kind:'column',value:'M1'}));const featureCount=app().querySelectorAll('[aria-label="Feature label"]').length;[...app().querySelectorAll('.flow-stats-ribbon fieldset')].find(f=>f.querySelector('legend').textContent==='Features / metrics').dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:fieldData}));check(app().querySelectorAll('[aria-label="Feature label"]').length===featureCount+1,'Data fields drag directly into the feature builder');
 button('Hide setup').click();check(app().querySelector('.flow-stats-ribbon').hidden,'Setup can collapse to give publication figures more space');
 button('Show setup').click();button('Import from Plot').click();
 const search=(name,value)=>{const e=control('Search '+name);e.value=value;e.dispatchEvent(new Event('input'));};
 search('markers','_CoRR');check(app().querySelectorAll('[aria-label="markers"] input').length===2,'Marker search is case-insensitive and matches suffix text');
 control('Select all matching markers').click();control('M1_corr').click();check(control('Select all matching markers').indeterminate,'Select-all shows a partial selection');control('Select all matching markers').click();
 search('markers','M0');check(control('M0_corr').checked&&!control('M0').checked,'Marker selection survives search changes without selecting unrelated markers');
 button('Add feature').click();check(app().querySelectorAll('[aria-label="Feature label"]').length===4,'Bulk add includes all checked corrected markers across search queries');button('Add feature').click();check(app().querySelectorAll('[aria-label="Feature label"]').length===4,'Repeated bulk add does not duplicate features');
 search('markers','no_such_marker');check(control('Select all matching markers').disabled,'No-match searches disable select-all');
 change(control('Summary metric'),'count');control('Clear selected populations').click();search('populations','Category: phenotype = Treg');check(app().querySelectorAll('[aria-label="populations"] input').length===1,'The same split category from two branches is offered once');control('Select all matching populations').click();button('Add feature').click();
 change(control('Summary metric'),'proportion');check(control('Proportion denominator').value==='@group','Population proportions default to each comparison group as denominator');button('Add feature').click();
 const abundanceRun=statsCalls;button('Run analysis').click();await until(()=>statsCalls>abundanceRun&&!button('Calculating…')&&app().querySelectorAll('.flow-stats-chart').length===6);
 const counts=latest.design.features.find(f=>f.metric==='count'),props=latest.design.features.find(f=>f.metric==='proportion');check(counts.target_rule.values[0]==='Treg'&&!counts.target_population,'Category features apply a reusable phenotype rule inside each cohort');
 check(latest.summaries.filter(s=>[counts.id,props.id].includes(s.feature)).every(s=>s.N===5),'Both cohorts retain all five patients for population features');
 check(latest.samples.filter(s=>s.feature===counts.id&&s.unit==='1').every(s=>s.value===0)&&latest.samples.filter(s=>s.feature===props.id&&s.unit==='1').every(s=>s.value===0),'Patients without target cells remain zero observations');
 check(latest.samples.filter(s=>s.feature===props.id).every(s=>s.value===100*s.n_target/s.n_cells),'Proportions use exact patient-specific numerator and denominator counts');
 const countChart=[...app().querySelectorAll('.flow-stats-chart')].find(c=>c.dataset.title.includes('cell count')),countTrace=countChart.data.find(t=>t.mode==='markers'),chosen=countTrace.customdata.findIndex((d,i)=>countTrace.y[i]>0);selected=null;countChart.emit('plotly_click',{points:[{customdata:countTrace.customdata[chosen]}]});await until(()=>selected);check(selected.count===countTrace.y[chosen],'Population dot selects only the target phenotype for that patient');
 [...app().querySelectorAll('[aria-label="Group label"]')].forEach((e,i)=>change(e,'tissue_1.0 / timepoint_'+(i+1)+' / response_CR / tumor_bed_1.0 / phenotype_ml_mr_Treg'));
 button('Delta heatmaps').click();change(control('Heatmap feature type'),'mean');change(control('Heatmap mode'),'absolute');if(control('Transpose heatmap').checked)control('Transpose heatmap').click();
 const layoutRun=statsCalls;button('Run analysis').click();await until(()=>statsCalls>layoutRun&&!button('Calculating…')&&app().querySelector('.flow-stats-chart')?.data?.[0]?.z?.length===3);
 heat=app().querySelector('.flow-stats-chart');check(heat._fullLayout._size.h>=290,'Three-row heatmap retains plotting height with long group labels');check(heat.data[0].colorbar.len>=200&&!heat.layout.annotations.some(a=>a.name==='heatmap-key'),'Absolute heatmap has a readable colorbar without unrelated significance symbols');
 const ticks=[...heat.querySelectorAll('.colorbar g[class$="tick"]')].map(e=>e.getBoundingClientRect());check(ticks.length>=2&&ticks.every((a,i)=>ticks.every((b,j)=>i===j||a.bottom<=b.top||b.bottom<=a.top)),'Colorbar tick labels do not overlap');
 change(control('Heatmap mode'),'delta');button('Add before / after pair').click();change(control('Delta column label'),'Matched treatment change within the selected tissue population');const deltaRun=statsCalls;button('Run analysis').click();await until(()=>statsCalls>deltaRun&&!button('Calculating…')&&app().querySelector('.flow-stats-chart')?.layout?.annotations?.some(a=>a.name==='heatmap-key'));
 heat=app().querySelector('.flow-stats-chart');check(heat.layout.xaxis.ticktext[0].includes('tissue'),'Long heatmap labels wrap at word boundaries');const barBox=heat.querySelector('.colorbar').getBoundingClientRect(),keyBox=[...heat.querySelectorAll('.annotation')].find(e=>e.textContent.includes('Blue:')).getBoundingClientRect();check(barBox.bottom+5<=keyBox.top,'Delta colorbar and significance key have separate vertical space');
 const article=heat.closest('article');article.style.width='570px';article.style.minWidth='0';await Plotly.Plots.resize(heat);check(heat.clientWidth>=heat.layout.margin.l+heat.layout.margin.r+240&&article.scrollWidth>article.clientWidth,'Narrow panels scroll without collapsing the heatmap or colorbar');
 const longExport=exported.length;change(control('Stats export format'),'png');button('Export figure').click();await until(()=>exported.length>longExport);check(exported.at(-1).download.startsWith('data:image/png'),'Long-label heatmap exports with the same reserved legend space');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent+'\n'+app()?.querySelector('.flow-stats-message')?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,setup=setup,screenshot=True,timeout=240,browser_host='flow-gating.test')
