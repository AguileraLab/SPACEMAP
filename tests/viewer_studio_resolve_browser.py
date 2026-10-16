"""Exercise Studio RESOLVE's editable gates, reference rules and method letters."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const tools=()=>document.querySelector('#studio-tools');
const input=label=>tools().querySelector('[aria-label="'+label+'"]');
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
setTimeout(async()=>{try{
 await until(()=>window.__state.imagesReady && input('Phenotyping method'));
 input('Phenotyping method').value='resolve';input('Phenotyping method').dispatchEvent(new Event('change'));
 await until(()=>input('RESOLVE marker M3'));
 assert(input('RESOLVE marker M2').checked && input('RESOLVE marker M3').checked,'Saved RESOLVE markers load');
 assert(input('RESOLVE rules').value.includes('A = [M2]'),'Classic phenotype rules load in the editor');
 assert(input('RESOLVE positivity method').options.length===13,'All ten shared methods, saved gates, Z1 and zero modes are available');
 assert(!!input('Overwrite existing results'),'RESOLVE has the shared overwrite choice');
 input('RESOLVE positivity method').value='Z1 score (no threshold files)';input('RESOLVE positivity method').dispatchEvent(new Event('change'));
 assert(!input('RESOLVE adaptive parent child gates').closest('.field').hidden && input('RESOLVE adaptive parent child gates').checked,'Classic adaptive parent/child gates are explicit in Z1 mode');
 input('RESOLVE adaptive parent child gates').click();
 input('RESOLVE positivity method').value='None (No Thresholds)';input('RESOLVE positivity method').dispatchEvent(new Event('change'));
 assert(input('RESOLVE Z1 cutoff').closest('.field').hidden && !input('RESOLVE adaptive parent child gates').checked,'Above-zero mode preserves the choice to disable adaptive gates');
 await until(()=>tools().querySelectorAll('[data-phenotyping-method]').length>=7);
 assert(['L','S','V','T','C','M','R'].every(letter=>Array.from(tools().querySelectorAll('[data-phenotyping-method]')).some(b=>b.textContent===letter)),'ROI list shows all seven method letters');
 input('RESOLVE positivity method').value='Otsu';input('RESOLVE positivity method').dispatchEvent(new Event('change'));
 button(tools(),'Calculate thresholds').click();
 await until(()=>tools().textContent.includes('ROI thresholds ready'));
 assert(input('RESOLVE positivity method').value==='Predefined (CSV)' && Number(input('M2 threshold').value)>10,'Computed gates become editable saved-threshold drafts');
 input('M2 threshold').value='200';input('M2 threshold').dispatchEvent(new Event('input'));
 input('M3 threshold').value='200';input('M3 threshold').dispatchEvent(new Event('input'));
 input('RESOLVE draft reference').value='phenotype_ref1';
 button(tools(),'Draft rules').click();
 await until(()=>tools().textContent.includes('Draft rules ready'));
 assert(input('RESOLVE rules').value.includes('T = [M2]') && input('RESOLVE rules').value.includes('B = [M3]'),'Manual references draft positive-marker rules');
 assert(tools().textContent.includes('not independent accuracy'),'Draft review separates reference agreement from independent accuracy');
 input('RESOLVE rules').value='Wrong=[NotMeasured]';
 button(tools(),'Check rules').click();
 await until(()=>tools().textContent.includes('unknown marker'));
 assert(true,'Unknown rule markers fail with a useful message');
 input('RESOLVE rules').value='T = [M2]\nB = [M3]';
 input('RESOLVE output column').value='phenotype_browser_resolve';
 button(document.querySelector('.studio-runbar'),'Run RESOLVE').click();
 const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
 await until(async()=>{const job=(await jobs()).find(j=>j.kind==='resolve_all');return job && ['done','failed'].includes(job.status);});
 const job=(await jobs()).find(j=>j.kind==='resolve_all');
 assert(job.status==='done' && job.result.column==='phenotype_browser_resolve','RESOLVE saves the exact chosen column: '+job.message);
 assert(job.result.counts.T===20 && job.result.counts.B===20,'Classic rules assign both populations correctly');
 await until(()=>tools().querySelector('[data-phenotyping-method="resolve"].done'));
 assert(true,'The T indicator turns green when every cell has a result');
 const state=await (await fetch('/spacemap-viewer/api/studio/browser-check/resolve_setup')).json();
 assert(state.thresholds[0].M2===200 && state.values.output_col==='phenotype_browser_resolve','Edited gates and output persist in the shared workspace');
 input('Phenotyping method').value='leiden';input('Phenotyping method').dispatchEvent(new Event('change'));
 await until(()=>input('Resolution'));
 tools().querySelector('[data-phenotyping-method="resolve"]').click();
 await until(()=>input('RESOLVE rules'));
 assert(input('Phenotyping method').value==='resolve','Clicking T opens the RESOLVE panel');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+tools()?.textContent.slice(-2000)));}},100);
</script>
"""


def setup(root):
    import json
    import pandas as pd
    table = pd.read_csv(root/'cells.csv')
    table['M2'] = [400]*20 + [20]*20
    table['M3'] = [20]*20 + [400]*20
    table['phenotype_ref1'] = ['T']*10 + [None]*10 + ['B']*10 + [None]*10
    table.to_csv(root/'cells.csv',index=False)
    cache=root/'outputs/phenotyping/cache'; cache.mkdir(parents=True,exist_ok=True)
    (cache/'resolve_params.json').write_text(json.dumps({'markers':['M2','M3'],'norm_z1':True,'norm_z2':False,'norm_cdf':False,'norm_nlog':False,'norm_order':['Z1']}))
    (cache/'resolve_pheno_panel.json').write_text(json.dumps({'criteria_cols':['Criteria 1'],'table':[{'Phenotype':'A','Criteria 1':['M2']},{'Phenotype':'B','Criteria 1':['M3']}]}))
    pd.DataFrame([{'batch_id':table.batch_id.iloc[0],'M2':100,'M3':100}]).to_csv(cache/'resolve_thresholds_table.csv',index=False)


if __name__=='__main__':
    main(DRIVER,studio=True,timeout=240,setup=setup)
