"""Jobs Stop feedback, pending-save message, completion and request failure."""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const originalFetch=window.fetch.bind(window), id='e'.repeat(32);
let jobState='running', stopPosts=0, failStop=false, pendingMessage='Stopping computation…';
window.fetch=async(input,options={})=>{
 const url=typeof input==='string'?input:input.url;
 if(url.endsWith('/api/studio/browser-check/jobs/'+id)&&options.method==='POST'){
  stopPosts++;
  if(failStop)return new Response(JSON.stringify({error:'Stop could not be requested'}),{status:400,headers:{'Content-Type':'application/json'}});
  jobState='stopping';
  return new Response(JSON.stringify({id,status:jobState,message:pendingMessage}),{headers:{'Content-Type':'application/json'}});
 }
 if(url.endsWith('/api/studio/browser-check/jobs')){
  return new Response(JSON.stringify({jobs:[{id,kind:'consensus_ml',status:jobState,progress:.4,
   message:jobState==='stopping'?pendingMessage:jobState==='cancelled'?'Stopped by the user.':'Training classifier'}]}),{headers:{'Content-Type':'application/json'}});
 }
 return originalFetch(input,options);
};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<600;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const card=()=>Array.from(document.querySelectorAll('.studio-job')).find(c=>c.textContent.includes('Consensus ML'));
const stop=()=>Array.from(card()?.querySelectorAll('button')||[]).find(b=>/^Stop/.test(b.textContent));
setTimeout(async()=>{try{
 await until(()=>stop());
 assert(stop().textContent==='Stop'&&!stop().disabled,'Running job has an enabled Stop');
 assert(stop().title.includes('save'),'Stop explains that an existing save is protected');
 stop().click();
 await until(()=>stop()?.textContent==='Stopping…');
 assert(stop().disabled,'Stop is disabled while termination is pending');
 assert(stopPosts===1,'One click requests cancellation exactly once');
 await until(()=>card().textContent.includes('Stopping computation'));
 pendingMessage='Finishing the result save safely…';
 await until(()=>card().textContent.includes(pendingMessage));
 assert(stop().disabled,'Saving remains protected while the UI waits');
 jobState='cancelled';
 await until(()=>card().textContent.includes('cancelled'));
 assert(!stop(),'Cancelled job has no Stop button');
 jobState='running';failStop=true;
 await until(()=>stop()&&!stop().disabled);
 stop().click();
 await until(()=>stopPosts===2&&!stop().disabled);
 assert(stop().textContent==='Stop','Failed stop request restores the button for retry');
 originalFetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){originalFetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=100, screenshot=False)
