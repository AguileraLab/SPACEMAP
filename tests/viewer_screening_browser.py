"""Real Firefox checks with a disposable unified table; never alters a project."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+'\n'+document.querySelector('.studio-runbar')?.textContent);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const tools=()=>document.querySelector('#studio-tools'),bar=()=>document.querySelector('.studio-runbar');
const by=label=>tools().querySelector('[aria-label="'+label+'"]');
const choose=(label,value)=>{const el=by(label);el.value=value;el.dispatchEvent(new Event('change'));};
const click=label=>by(label).click();
const button=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 document.querySelector('[data-step=tab8]').click();
 await until(()=>tools().querySelector('.studio-stage-tabs'));
 button(tools(),'Feature screening').click();
 await until(()=>by('Screening sample column'));
 assert(['Screen Density','Screen Proportion','Screen Ratios','Screen Diversity','Screen Neighbour distance','Screen Pairs (dyads)','Screen Triplets (triads)'].every(n=>by(n)),'All seven independently selectable feature families');
 assert(['1. Cells and regions','2. What to measure','3. What to compare'].every(t=>tools().textContent.includes(t))&&!tools().textContent.includes('4. Compare'),'Three workflow sections');
 assert(by('Screening sample column').closest('.field').querySelector('.help').getAttribute('data-tip').includes('ONE person'),'Sample ID has a concrete, keyboard-accessible explanation');
 choose('Screening sample column','sample');
 choose('Screening cell type column','phenotype_ml_consensus');
 await until(()=>by('All matching Screening cell types'));
 by('All matching Screening cell types').closest('details').open=true;
 const search=by('Find Screening cell types');search.value='A';search.dispatchEvent(new Event('input'));
 click('All matching Screening cell types');
 assert(by('Screening cell types: A').checked&&!by('Screening cell types: B').checked,'Search select-all affects matching types only');
 search.value='';search.dispatchEvent(new Event('input'));click('All matching Screening cell types');
 by('All matching Screening cell types').closest('details').open=false;
 click('All seven features');
 assert(by('Screen Density').checked&&by('Screen Triplets (triads)').checked,'One option selects all seven without removing separate settings');
 assert([...tools().querySelectorAll('.screening-feature > details')].every(d=>!d.open),'Feature settings start folded');
 by('All four diversity measures').closest('details').open=true;
 click('All four diversity measures');
 assert(['Shannon','Simpson','Number of types','Evenness'].every(n=>by('Screen diversity '+n).checked),'All four diversity measures can be selected together');
 click('Screen diversity Simpson');
 assert(by('All four diversity measures').indeterminate&&by('Screen diversity Shannon').checked,'A diversity measure can be toggled without clearing the others');
 click('Screen diversity Simpson');
 assert(by('All four diversity measures').closest('details').open,'Choosing measures preserves the open settings');
 choose('Screening correction scope','classic');
 choose('Screening resampling','subsample');
 assert(!by('Screening subset fraction').closest('.field').hidden,'Classic sample-subset stability exposes its fraction');
 choose('Screening resampling','bootstrap');
 assert(by('Screening subset fraction').closest('.field').hidden,'With-replacement bootstrap does not imply subset sampling');
 choose('Screening resampling','off');
 choose('Screening density area source','estimate');
 assert(by('Screening zone area CSV').closest('.field').hidden&&!by('Screening area percentile').closest('.field').hidden,'Estimated area requires no CSV');
 choose('Screening density area source','csv');
 assert(!by('Screening zone area CSV').closest('.field').hidden,'Area CSV appears only when explicitly selected');
 choose('Screening distance column','tumor_distance');choose('Screening distance units','px');
 choose('Screening pixel size','1');choose('Screening zone widths','100, 200');
 choose('Screening minimum cells','1');choose('Screening comparison','describe');
 choose('Screening neighbour count','1');choose('Screening dyad maximum','65');choose('Screening triad maximum','65');
 choose('Screening zone area CSV',window.__screeningArea);
 button(tools(),'Save').click();await until(()=>[...tools().querySelectorAll('.chip.ok')].some(e=>e.textContent==='Saved'));
 button(tools(),'Density').click();await until(()=>by('Density group column'));
 button(tools(),'Feature screening').click();await until(()=>by('Screening sample column'));
 assert(by('Screening sample column').value==='sample'&&by('Screen Triplets (triads)').checked,'Selections persist across spatial tabs');
 assert(by('All four diversity measures').checked,'All four diversity choices persist across tabs');
 assert(by('Screening correction scope').value==='classic','Declared correction scope persists across tabs');
 const image=state.layers.find(l=>l.type==='image'), original=image.item;
 const started=performance.now();button(bar(),'This image').click();
 await until(()=>by('Screening result feature type'));
 assert(!bar().textContent.includes('Done')&&[...tools().querySelectorAll('.chip.warn')].some(e=>e.textContent.startsWith('Preview')),'Live preview is visible before the background job finishes');
 await until(()=>bar().textContent.includes('Done'));
 assert(state.layers.find(l=>l.type==='image').item===original,'Background scan preserves the loaded image');
 assert(bar().textContent.includes('Done'),'Completion status follows the background job');
 assert([...by('Screening result feature type').options].length===8,'Every selected family has a result figure');
 assert(tools().querySelector('.screening-run-design').textContent.includes('Sample ID: sample'),'Saved sample identity is visible');
 choose('Screening significance filter','fdr');
 assert(tools().querySelectorAll('tr.spatial-hit').length===0,'Untested descriptive results never appear as FDR significant');
 assert(document.querySelector('[data-figure="screening-density"]').textContent.includes('No findings match this filter'),'FDR filtering does not leave a nominal finding in the linked figure');
 choose('Screening significance filter','all');
 const significant=[...tools().querySelectorAll('a')].find(a=>a.textContent==='FDR significant');
 const sigCSV=await fetch(significant.href).then(r=>r.text());
 assert(sigCSV.includes('q_family')&&sigCSV.trim().split('\n').length===1,'FDR export keeps headers and reports no invented significant rows');
 choose('Screening sample column','batch_id');
 assert(!tools().querySelector('.screening-run-design .chip').hidden,'Changed controls are distinguished from saved results');
 choose('Screening sample column','sample');
 choose('Screening result feature type','ratio');
 assert(by('Screening result feature type').value==='ratio'&&document.querySelector('[data-figure="screening-ratio"] .screening-statistics').textContent.includes('q='),'Feature-type selection synchronizes table and statistical figure');
 choose('Screening result feature type','diversity');
 const diversity=document.querySelector('[data-figure="screening-diversity"]');
 const finding=diversity.querySelector('[aria-label="Screening interactive finding"]');
 assert(['Shannon','Simpson','Number of types','Evenness'].every(name=>[...finding.options].some(o=>o.textContent.startsWith(name+' ·'))),'Each diversity measure has its own selectable findings');
 finding.value=[...finding.options].find(o=>o.textContent.startsWith('Simpson ·')).value;finding.dispatchEvent(new Event('change'));
 assert(diversity.querySelector('svg').textContent.includes('Simpson')&&diversity.querySelector('img').src.includes('diversity_simpson.png'),'Selected diversity measure controls the live axis and publication figure');
 const divLink=diversity.querySelector('a[href$="diversity_simpson.svg"]');
 assert(divLink&&await fetch(divLink.href).then(r=>r.ok),'Each diversity measure has a separate vector export');
 choose('Screening result feature type','triad');
 await until(()=>document.querySelector('[data-figure="screening-triad"] details'));
 document.querySelector('[data-figure="screening-triad"] details').open=true;
 const picture=document.querySelector('[data-figure="screening-triad"] img');const picReply=await fetch(picture.src);assert(picReply.ok,'Publication image HTTP '+picReply.status+' '+(picReply.ok?picReply.headers.get('content-type'):await picReply.text()));
 await until(()=>document.querySelector('[data-figure="screening-triad"] img')?.naturalWidth>0);
 assert(true,'Triplet figure rendered in Studio');
 assert(document.querySelector('[data-figure="screening-triad"] svg circle[role=button]'),'Live sample plot has clickable representative dots');
 const row=tools().querySelector('tr.spatial-hit');row.click();
 await until(()=>button(tools(),'Show context cells'));
 document.querySelector('[data-figure="screening-triad"] svg circle[role=button]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>!!state.layers.find(l=>l.type==='cells').flowSelection);
 assert(dataView.flowSelection===state.layers.find(l=>l.type==='cells').flowSelection&&dataView.source?.combined,'Sample dot links the identical membership to Data and the image');
 const allTests=[...tools().querySelectorAll('a')].find(a=>a.textContent==='All tests');
 const table=await fetch(allTests.href).then(r=>r.text());
 assert(table.includes('dyad')&&table.includes('triad')&&table.includes('n_samples'),'Full results CSV downloads with sample counts');
 const svg=document.querySelector('[data-figure="screening-triad"] a[href$=".svg"]');
 assert(svg&&await fetch(svg.href).then(r=>r.ok),'Vector figure export downloads');
 const runs=by('Screening saved run');assert(runs.options.length===2,'New run appears in saved runs');
 assert([...tools().querySelectorAll('.screening-grid')].every(el=>el.scrollWidth<=el.clientWidth+2),'Related controls fit the narrow dock');
 const overflow=[...tools().querySelectorAll('.screening-section')].filter(el=>el.scrollWidth>el.clientWidth+2);
 assert(!overflow.length,'Whole screening sections fit the narrow dock'+(overflow.length?' '+JSON.stringify(overflow.map(el=>({text:el.textContent.slice(0,80),width:el.clientWidth,scroll:el.scrollWidth,wide:[...el.querySelectorAll('*')].filter(x=>x.getBoundingClientRect().right>el.getBoundingClientRect().right+2).map(x=>({tag:x.tagName,css:x.className,text:x.textContent.slice(0,80),width:x.getBoundingClientRect().width})).slice(0,12)}))):''));
 assert(by('Screening saved run').value===by('Screening saved run').options[1].value,'Saved-run picker identifies the displayed result');
 // Leave the compact controls and selected live figure visible in the capture.
 [...tools().querySelectorAll('.screening-feature > details')].forEach(d=>d.open=false);
 by('Screening correction scope').closest('details').open=true;
 by('Screening resampling').closest('details').open=true;
 choose('Screening resampling','bootstrap');
 by('Screening correction scope').scrollIntoView({block:'center'});
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,elapsed_ms:performance.now()-started})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+bar()?.textContent));}},100);
</script>
"""


def setup(root):
    import pandas as pd
    from spacemap import dataio
    global DRIVER
    table=root/'cells.csv'
    frame=pd.read_csv(table)
    frame['sample']='sample-1'
    frame['tumor_distance']=frame.x_centroid-160
    frame.loc[frame.label % 3 == 0,'phenotype_ml_consensus']='C'
    frame.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    area=root/'areas.csv'
    pd.DataFrame([dict(sample='sample-1',group='All',image='roi',region='All',zone=z,area_mm2=.02*(z+1)) for z in range(2)]).to_csv(area,index=False)


if __name__=='__main__':
    # The fixture path is discovered from the table endpoint, then used solely
    # for a disposable input area table.
    def fixture(root):
        setup(root)
        from spacemap_gui import persistence
        from spacemap_gui.studio import screening
        c=screening.defaults({'area_table':str(root/'areas.csv')})
        persistence.save_project_config(str(root),{'tab_configurations':{'tab8_analysis':{'studio_screening':c}}})
    DRIVER=DRIVER.replace("choose('Screening zone area CSV',window.__screeningArea);","")
    import os
    main(DRIVER,studio=True,setup=fixture,timeout=300,capture=os.environ.get('SPACEMAP_TEST_CAPTURE'))
