"""Real Studio comparison builder, background results, exports and linked visits."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+'\n'+document.querySelector('.studio-runbar')?.textContent);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const tools=()=>document.querySelector('#studio-tools'),bar=()=>document.querySelector('.studio-runbar');
const by=label=>tools().querySelector('[aria-label="'+label+'"]');
const choose=(label,value)=>{const el=by(label);el.value=value;el.dispatchEvent(new Event('change'));};
const button=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.imagesReady&&state.cellsReady&&document.querySelector('[data-step=tab8]'));
 document.querySelector('[data-step=tab8]').click();
 await until(()=>tools().querySelector('.studio-stage-tabs'));
 button(tools(),'Feature screening').click();
 await until(()=>by('Screening sample column'));
 assert(by('Screening sample column').value==='person','Independent subject identity loaded');
 const more=[...tools().querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='More comparisons');
 assert(more&&!more.open,'Additional comparison controls start folded');more.open=true;
 by('Screening splitting columns').open=true;by('Screening splitting columns: visit').click();
 by('Screening splitting columns').open=false;
 button(tools(),'Add comparison').click();choose('Additional comparison column 1','visit');
 by('Match within samples in comparison 1').click();
 by('Also compare changes').click();choose('Screening repeated observation column','visit');
 await until(()=>by('Screening before observation'));
 choose('Screening before observation','Before');choose('Screening after observation','After');
 if(!by('Pick roi').checked)by('Pick roi').click();
 button(tools(),'Preview comparisons').click();
 await until(()=>more.querySelectorAll('tbody tr').length===5);
 assert(more.textContent.includes('matched: 10')&&more.textContent.includes('Control: 5'),'Preview counts ten paired subjects and five per independent group');
 assert(!more.textContent.includes('[object Object]'),'Nested complete-visit counts remain readable');
 button(tools(),'Save').click();await until(()=>[...tools().querySelectorAll('.chip.ok')].some(e=>e.textContent==='Saved'));
 const image=state.layers.find(l=>l.type==='image'),original=image.item;
 button(bar(),'Selected').click();
 await until(()=>by('Screening result comparison')&&[...by('Screening result comparison').options].some(o=>o.textContent.includes('visit=Before')&&!o.disabled));
 const before=[...by('Screening result comparison').options].find(o=>o.textContent.includes('visit=Before')).value;
 choose('Screening result comparison',before);
 await until(()=>by('Screening result comparison')?.value===before&&tools().querySelector('.screening-run-design')?.textContent.includes('Saved run'));
 await until(()=>bar().textContent.includes('Done'));
 assert(by('Screening result comparison').value===before,'Explicit comparison choice survives background completion');
 assert(state.layers.find(l=>l.type==='image').item===original,'Background comparisons preserve the image');
 await until(()=>[...by('Screening result comparison').options].every(o=>o.textContent.endsWith('complete')));
 assert([...by('Screening result comparison').options].length===5,'All five completed questions remain selectable');
 const all=[...tools().querySelectorAll('a')].find(a=>a.textContent==='All comparisons');
 const csv=await fetch(all.href).then(r=>r.text());
 assert(csv.includes('q_global')&&csv.includes('comparison_label')&&csv.trim().split('\n').length===11,'Full-scan CSV has every question and both feature tests');
 const sig=[...tools().querySelectorAll('a')].find(a=>a.textContent==='All FDR findings');
 assert(await fetch(sig.href).then(r=>r.ok),'All-comparison FDR export is available');
 const delta=[...by('Screening result comparison').options].find(o=>o.textContent.includes('change After')).value;
 choose('Screening result comparison',delta);
 await until(()=>document.querySelector('[data-figure="screening-proportion"] svg')?.textContent.includes('Change in'));
 const figure=document.querySelector('[data-figure="screening-proportion"]');
 assert(figure.querySelector('.screening-statistics').textContent.includes('change (After − Before)'),'Change effects identify their subtraction direction');
 figure.querySelector('details').open=true;
 await until(()=>document.querySelector('[data-figure="screening-proportion"] img')?.naturalWidth>0);
 assert(true,'Change publication figure renders');
 figure.querySelector('svg circle[role=button]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await until(()=>!!state.layers.find(l=>l.type==='cells').flowSelection);
 const visits=figure.querySelector('[aria-label="Review repeated observations"]');
 assert(visits.querySelectorAll('button').length===2&&visits.textContent.includes('Before')&&visits.textContent.includes('After'),'Each change dot offers both observations for image review');
 assert(dataView.flowSelection===state.layers.find(l=>l.type==='cells').flowSelection,'Change finding shares exact context membership with Data');
 const selectionBefore=state.layers.find(l=>l.type==='cells').flowSelection;
 visits.querySelectorAll('button')[1].click();
 await until(()=>state.layers.find(l=>l.type==='cells').flowSelection!==selectionBefore);
 assert(dataView.flowSelection===state.layers.find(l=>l.type==='cells').flowSelection,'After observation updates the shared viewer and Data selection');
 const pic=await fetch(figure.querySelector('img').src);assert(pic.ok,'Selected comparison image endpoint returns the correct export');
 assert([...tools().querySelectorAll('.screening-grid')].every(el=>el.scrollWidth<=el.clientWidth+2),'Comparison controls fit the narrow dock');
 more.open=false;by('Screening result comparison').scrollIntoView({block:'center'});
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+bar()?.textContent));}},100);
</script>
"""


def fixture(root):
    import numpy as np
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui import persistence
    from spacemap_gui.studio import screening
    table=root/'cells.csv'
    frame=pd.read_csv(table)
    ids=np.arange(len(frame))
    frame['person']=['Person '+str(i//4) for i in ids]
    frame['visit']=np.where(ids%4 < 2,'Before','After')
    frame['outcome']=np.where(ids//4 < 5,'Control','Case')
    frame['phenotype_ml_consensus']=np.where(ids%2 == 0,'A','B')
    frame.loc[(frame.outcome=='Case')&(frame.visit=='After'),'phenotype_ml_consensus']='A'
    frame.to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    c=screening.defaults({'sample_col':'person','image_col':'batch_id','phenotype_col':'phenotype_ml_consensus',
        'phenotypes':['A','B'],'families':['proportion'],'zone_source':'whole','pixel_um':1,
        'min_cells':1,'min_samples':3,'widths':[100], 'correction_scope':'family','permutations':999,
        'endpoint':{'type':'binary','group_col':'outcome','group_a':'Control','group_b':'Case'}})
    persistence.save_project_config(str(root),{'tab_configurations':{'tab8_analysis':{'studio_screening':c}}})


if __name__=='__main__':
    import os
    main(DRIVER,studio=True,setup=fixture,timeout=300,capture=os.environ.get('SPACEMAP_TEST_CAPTURE'))
