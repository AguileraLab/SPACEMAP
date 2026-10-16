"""Jobs keep independent timers and readable logs through polling and status changes."""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const originalFetch=window.fetch.bind(window), id='a'.repeat(32), other='b'.repeat(32), old='c'.repeat(32);
const started=Date.now()-65000;
let status='running', phase='Validation fit', revision=0, failedPoll=false, polls=0, logReads=0;
let lines=Array.from({length:100},(_,i)=>'Stage detail '+i).join('\n');
const response=data=>new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
window.fetch=async(input,options={})=>{
 const url=typeof input==='string'?input:input.url;
 if(url.endsWith('/api/studio/browser-check/jobs')){
  polls++;
  if(failedPoll)throw Error('Temporary jobs connection failure');
  return response({jobs:[{id,kind:'consensus_ml',status,progress:.4,message:phase,
   started:new Date(started).toISOString(),finished:status==='done'?new Date(started+125000).toISOString():undefined},
   {id:other,kind:'som',status:'done',progress:1,message:'Finished '+revision,
    started:new Date(started-90000).toISOString(),finished:new Date(started-25000).toISOString()},
   {id:old,kind:'vortex',status:'failed',message:'Older job'}]});
 }
 if(url.endsWith('/api/studio/browser-check/jobs/'+id)){
  logReads++;
  return response({log:lines});
 }
 return originalFetch(input,options);
};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<600;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const card=job=>document.querySelector('[data-job-id="'+job+'"]');
setTimeout(async()=>{try{
 await until(()=>card(id));document.querySelector('[data-tab="jobs"]').click();
 const timer=card(id).querySelector('.studio-job-timer'), completed=card(other).querySelector('.studio-job-timer');
 assert(timer.textContent.startsWith('Elapsed '),'Active job has a separate elapsed counter');
 assert(completed.textContent==='Duration 01:05','Finished job shows its recorded duration');
 assert(card(old).querySelector('.studio-job-timer').textContent==='Duration —','Legacy jobs do not invent a duration');
 const initialTime=timer.textContent, initialCard=card(id);
 await until(()=>timer.textContent!==initialTime);
 assert(card(id)===initialCard,'Timer updates leave the job card intact');
 assert(logReads===0,'The elapsed counter does not request or print logs');
 assert(completed.textContent==='Duration 01:05','Completed durations remain frozen while other jobs run');
 const log=card(id).querySelector('.studio-job-log');log.open=true;
 const text=log.querySelector('pre');await until(()=>text.textContent.includes('Stage detail 99'));
 assert(text.scrollHeight>text.clientHeight,'Fixture log is long enough to exercise scrolling');
 text.scrollTop=70;const scroll=text.scrollTop;
 phase='Final fit';lines+='\nFinal fit started';
 await until(()=>card(id).querySelector('p.muted').textContent==='Final fit'&&text.textContent.includes('Final fit started'));
 assert(card(id).querySelector('.studio-job-log')===log&&log.open,'A new stage retains the same expanded log');
 assert(Math.abs(text.scrollTop-scroll)<2,'A stage change preserves the reading position');
 lines+='\nNew detail without a status change';
 await until(()=>text.textContent.includes('New detail without'));
 assert(log.open&&Math.abs(text.scrollTop-scroll)<2,'Logs refresh without status changes and without scrolling away');
 const beforeOther=card(id);revision++;
 await until(()=>card(other).textContent.includes('Finished 1'));
 assert(card(id)===beforeOther&&log.open,'Another job updating does not rebuild this card or collapse its log');
 failedPoll=true;await until(()=>document.querySelector('#studio-jobs').textContent.includes('Temporary jobs connection failure'));
 assert(card(id)===beforeOther&&log.open&&text.textContent.includes('Stage detail 99'),'A polling error preserves the readable log');
 failedPoll=false;
 const warningVisible=()=>Array.from(document.querySelectorAll('#studio-jobs p')).some(p=>!p.hidden&&p.textContent==='Temporary jobs connection failure');
 await until(()=>!warningVisible());
 assert(!warningVisible(),
   'An unchanged successful response clears the connection warning');
 text.scrollTop=text.scrollHeight;lines+='\nFollowing newest output';await until(()=>text.textContent.includes('Following newest output'));
 assert(text.scrollHeight-text.scrollTop-text.clientHeight<3,'Readers at the bottom follow newly appended output');
 log.open=false;await delay(200);const readsWhenClosed=logReads;phase='Writing result';
 await until(()=>card(id).textContent.includes('Writing result'));await delay(3200);
 assert(!log.open&&logReads===readsWhenClosed,'Closed logs stay closed and stop requesting log contents');
 log.open=true;status='done';phase='Finished';lines+='\nFinished successfully';
 await until(()=>card(id).querySelector('.studio-job-timer').textContent==='Duration 02:05');
 await until(()=>text.textContent.includes('Finished successfully'));
 assert(log.open&&card(id).querySelector('.studio-job-log')===log,'Completion preserves the expanded log');
 const finalTime=card(id).querySelector('.studio-job-timer').textContent;await delay(2100);
 assert(card(id).querySelector('.studio-job-timer').textContent===finalTime,'The elapsed counter stops on completion');
 originalFetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,polls,logReads})));
}catch(error){originalFetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=130, screenshot=False)
