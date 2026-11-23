"""Stopping a model remains a pending state in Studio until the next status poll."""
from viewer_threshold_browser import main

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<1000;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(c,m)=>{if(!c)throw Error(m);checks++;};
const button=(root,label)=>[...root.querySelectorAll('button')].find(b=>b.textContent===label);
setTimeout(async()=>{try{
 await until(()=>document.querySelector('#studio-assistant')&&window.__state.imagesReady);
 document.querySelector('[data-tab=assistant]').click();
 const root=()=>document.querySelector('#studio-assistant');
 const original=window.fetch;let phase='ready',stops=0;
 window.fetch=async(url,options)=>{
   if(/\/ai(?:\/stop)?$/.test(String(url))){
     if(String(url).endsWith('/stop')){phase='stopping';stops++;}
     const payload={enabled:true,pinned:false,hardware:'Test',missing:[],install:'',
       models:[{id:'test',name:'Test model',installed:true,active:true,downloading:false,size_gb:1,fits:true}],
       runtime:{running:phase!=='stopped',ready:phase==='ready',stopping:phase==='stopping',message:'Test model '+phase,port:8091,model:'test'},
       precision:'auto',precisions:[['auto','Automatic']],stop_with_spacemap:true,leftovers:[],
       online:{chosen:false,model:'test',models:[['test','Test online']],details:{},checks:{},key_saved:false}};
     return new Response(JSON.stringify(payload),{status:200,headers:{'Content-Type':'application/json'}});
   }return original(url,options);
 };
 await until(()=>button(root(),'AI setup'));button(root(),'AI setup').click();
 const setup=()=>root().querySelector('.studio-ai-setup');
 await until(()=>setup()&&button(setup(),'Stop'));
 button(setup(),'Stop').click();
 await until(()=>button(setup(),'Stopping…'));
 assert(button(setup(),'Stopping…').disabled,'A pending stop cannot be clicked repeatedly');
 assert(setup().textContent.includes('In use · stopping'),'State says stopping, not loading');
 assert(!setup().textContent.includes('Stopped.'),'Stop is not reported complete early');
 assert(!button(setup(),'Start'),'Cannot start a replacement before shutdown completes');
 phase='stopped';await until(()=>button(setup(),'Start'));
 assert(stops===1,'Polling updates completion without submitting another Stop');
 assert(setup().textContent.includes('In use · stopped'),'Completion offers Start again');
 window.fetch=original;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(e+'\n'+e.stack));}},100);
</script>
'''

if __name__=='__main__':
    main(DRIVER,studio=True,timeout=90)
