"""Saved cell subsets, multiple methods and Classic figures in Studio Firefox."""
from viewer_threshold_browser import main
from viewer_studio_single_image_phenotyping_browser import setup as prepare


def setup(root):
    prepare(root)
    from spacemap_gui.studio import reference_models, tasks
    table = root / 'cells.csv'
    settings = {'table': str(table), 'markers': ['M0','M1'], 'training_column': 'phenotype_ref1',
                'label_column': 'label', 'roi_column': '', 'project_root': str(root)}
    settings['revision'] = tasks.source_revision(settings)
    reference_models.train(settings, root/'fit', lambda *a: None)


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const tools=()=>document.querySelector('#studio-tools'),fig=()=>document.querySelector('#studio-figures [data-figure="viz-agreement"]');
 const button=(p,t)=>[...p.querySelectorAll('button')].find(b=>b.textContent===t);
 await until(()=>document.querySelector('[data-step=tab9]')&&window.__state.cellsReady);
 document.querySelector('[data-step=tab9]').click();
 await until(()=>button(tools(),'Method agreement'));button(tools(),'Method agreement').click();
 await until(()=>tools().querySelector('[aria-label="Assessment cells"]'));
 const subset=tools().querySelector('[aria-label="Assessment cells"]');
 assert(subset.value==='validation','Saved validation is the default subset');
 const methods=[...tools().querySelectorAll('[data-assessment-method]')];
 const selected=methods.filter(b=>['phenotype_training','phenotype_ml_consensus'].includes(b.dataset.assessmentMethod));
 assert(selected.length===2,'Independent phenotype columns can be selected together');
 for(const input of methods){input.checked=selected.includes(input);input.dispatchEvent(new Event('change'));}
 let found;const old=window.fetch;
 window.fetch=async(url,args)=>{const result=await old(url,args);if(String(url).includes('/phenotype_assessment')&&args?.body){const data=JSON.parse(args.body);if(!data.action)found=await result.clone().json();}return result;};
 const compare=async()=>{found=null;button(document.querySelector('.studio-runbar'),'Compare').click();await until(()=>found?.id&&fig()?.querySelector('img')?.naturalWidth>0);};
 await compare();
 assert(found.cells===12&&found.summary.length===2&&found.summary.every(m=>m.n_test===12),'Both methods use the same twelve held-out cells');
 assert(fig().textContent.includes('Macro F1')&&fig().textContent.includes('Validation cells'),'Scores and subset appear beside the Classic figure');
 const plot=fig().querySelector('[aria-label="Assessment figure"]');plot.value='confusion';plot.dispatchEvent(new Event('change'));
 await until(()=>!fig().querySelector('[aria-label="Confusion method"]').hidden&&fig().querySelector('img').naturalWidth>0);
 assert(true,'Confusion matrix supports choosing the assessed method');
 for(const [value,count] of [['training',28],['both',40],['all',80]]){
   subset.value=value;subset.dispatchEvent(new Event('change'));await compare();
   assert(found.cells===count,value+' scores the requested cell subset');
 }
 assert(fig().querySelector('[aria-label="Assessment figure format"]').options.length===3&&button(fig(),'Save figure')&&button(fig(),'Download scores'),'Figures export PNG SVG PDF and scores');
 window.fetch=old;
 await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){await fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('.studio-runbar')?.textContent));}},0);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=130, setup=setup, screenshot=False)
