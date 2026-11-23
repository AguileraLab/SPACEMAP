"""Completed gland jobs must not freeze Jobs cards at their last saving stage."""
from viewer_threshold_browser import main


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<300;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const check=(value,text)=>{if(!value)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
const originalFetch=window.fetch;
const jobs=[
 {id:'gland-fit',kind:'gland_types',status:'running',progress:.98,message:'Saving the result…'},
 {id:'gland-review',kind:'gland_types',status:'running',progress:.8,message:'Saving profiles and sample-level comparisons…'},
 {id:'training',kind:'train_reference_model',status:'done',progress:1,message:'Finished.',result:{models:[{method:'Random Forest',held_out_balanced_accuracy:.8}]}}
].map(job=>({...job,started:new Date(Date.now()-60000).toISOString()}));
window.fetch=async(resource,options)=>{
 const path=typeof resource==='string'?resource:resource.url;
 if(path.endsWith('/jobs'))return new Response(JSON.stringify({jobs}),{headers:{'Content-Type':'application/json'}});
 return originalFetch(resource,options);
};
const card=id=>document.querySelector('[data-job-id="'+id+'"]');
const buttons=el=>[...el.querySelectorAll('button')].map(b=>b.textContent);
setTimeout(async()=>{try{
 await until(()=>card('gland-fit')&&card('gland-review'));
 document.querySelector('[data-tab="jobs"]').click();
 check(card('gland-fit').textContent.includes('98%')&&card('gland-review').textContent.includes('80%'),'Both active saving stages are visible');
 check(card('training').textContent.includes('80.0% validation balanced accuracy'),'Training model arrays still render validation');
 const viewer=window.__state.viewer;
 const finish=(job,diagnose)=>Object.assign(job,{status:'done',progress:1,finished:new Date().toISOString(),message:'Finished. The result is ready.',
  result:{diagnose,images:['roi'],glands:20,beds:18,models:{interior:{k:6,silhouette:.42},exterior:{k:8,silhouette:.21}}}});
 finish(jobs[0],false);finish(jobs[1],true);
 await until(()=>card('gland-fit').job.status==='done'&&card('gland-review').job.status==='done');
 for(const id of ['gland-fit','gland-review']){
  const c=card(id);
  check(c.querySelector('.chip').textContent==='done'&&c.querySelector('.studio-bar > div').style.width==='100%','Finished gland card shows done at 100%: '+id);
  check(!buttons(c).some(name=>/Stop/.test(name))&&c.querySelector('.studio-job-timer').textContent.startsWith('Duration'),'Finished gland card stops its clock and removes Stop: '+id);
  check(!c.textContent.includes('undefined')&&!c.textContent.includes('background-removed'),'Gland summary describes gland results: '+id);
 }
 check(card('gland-review').textContent.includes('K review')&&card('gland-fit').textContent.includes('20 glands'),'K review is distinguished from fitted gland results');
 const duration=card('gland-fit').querySelector('.studio-job-timer').textContent;await delay(1300);
 check(duration===card('gland-fit').querySelector('.studio-job-timer').textContent,'Completed duration no longer increases');
 check(window.__state.viewer===viewer,'Displaying completed model details does not reopen the viewer');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=70, screenshot=False)
