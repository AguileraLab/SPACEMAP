"""Settings autosave: ambiguous aborts, retries, newest edits and real failures."""

from viewer_threshold_browser import main
from viewer_saved_view_browser import setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1500;i++){if(await check())return;await delay(20);}throw Error('Timed out: '+check);};
let checks=0;const assert=(yes,note)=>{if(!yes)throw Error(note);checks++;};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady);
 const {specs}=await getJSON('/saved-view-fixtures');await init(specs[0]);await until(()=>s.imagesReady);
 await flushViewerSettings();
 const layer=s.layers.find(l=>l.id==='image'),path='/workspace/'+s.spec.workspace+'/view';
 const originalFetch=window.fetch, originalMessage=message;
 const notices=[];message=(text,error)=>{notices.push({text,error});originalMessage(text,error);};
 let mode='lost-reply', attempts=0, block, release;
 window.fetch=async(url,options)=>{
  if(String(url)===api(path)&&options?.method==='POST'){
   attempts++;
   if(mode==='lost-reply'){
    mode='ok';await originalFetch(url,options);
    throw new DOMException('signal is aborted without reason','AbortError');
   }
   if(mode==='hold') {mode='ok';block=new Promise(r=>release=r);await block;}
   if(mode==='offline')throw new TypeError('Network request failed');
   if(mode==='proxy')return new Response('Temporarily unavailable',{status:502});
   if(mode==='rejected')return new Response(JSON.stringify({error:'This view is invalid.'}),{status:400});
  }
  return originalFetch(url,options);
 };
 layer.opacity=.42;saveView();await flushViewerSettings();
 const job=viewerSettingWrites.get(api(path));
 assert(job.pending && job.failures===1,'An ambiguous abort retains the exact pending save');
 assert(!notices.some(n=>n.error),'A single aborted response does not show a red failure');
 assert((await getJSON(path)).view.layers.image.opacity===.42,'The server may already have written a save whose reply was lost');
 // Automatic retry (no manual flush), then a real read of the saved settings.
 await until(()=>!job.pending&&!job.running);
 assert(attempts>=2&&!job.failure,'Automatic retry confirms the save and clears the failure');
 assert(document.getElementById('message').hidden,'Recovered settings dismiss their own transient notice');
 mode='hold';layer.opacity=.51;saveView();const flushing=flushViewerSettings();await until(()=>release);
 layer.opacity=.73;saveView();release();await flushing;
 assert(!job.pending&&(await getJSON(path)).view.layers.image.opacity===.73,'A newer edit during a running save wins');
 mode='offline';layer.opacity=.81;saveView();
 await flushViewerSettings();await flushViewerSettings();await flushViewerSettings();
 assert(notices.some(n=>n.error&&n.text.includes('waiting for confirmation')),'Repeated network failures remain visible');
 assert((await loadProjectView(s.spec)).layers.image.opacity===.81,'Returning to an image uses the pending view instead of old server settings');
 const displayPath='/display/'+layer.display_token+'/save';
 const channels=channelSettings(layer);channels[0].low=19.125;
 queueViewerSetting(displayPath,{channels,auto:{method:'Otsu',scope:''}});
 const displayJob=viewerSettingWrites.get(api(displayPath));clearTimeout(displayJob.timer);
 const reloaded={...layer,channels:[]};await prepareChannels(reloaded,{channels:[]});
 assert(reloaded.channels[0].low===19.125&&reloaded.autoMethod==='Otsu','Marker controls restore their pending changes too');
 mode='ok';message('Unrelated label failure',true);await flushViewerSettings();
 assert(document.getElementById('message').textContent==='Unrelated label failure'&&!document.getElementById('message').hidden,
   'Recovering an autosave does not erase another error');
 mode='rejected';layer.opacity=.91;saveView();await flushViewerSettings();
 assert(job.pending&&notices.some(n=>n.error&&n.text.includes('This view is invalid.')),'A real server rejection remains a red actionable warning');
 mode='ok';await flushViewerSettings();
 assert(!job.pending&&(await getJSON(path)).view.layers.image.opacity===.91,'Retry preserves the latest value after a real failure');
 mode='proxy';layer.opacity=.92;saveView();const beforeProxy=notices.length;await flushViewerSettings();
 assert(job.pending&&job.failure.transient&&!notices.slice(beforeProxy).some(n=>n.error),
   'A proxy HTML error is treated as a temporary failure');
 mode='ok';await flushViewerSettings();
 // A successful marker-settings save must not hide a failed workspace save.
 mode='rejected';layer.opacity=.93;saveView();await flushViewerSettings();
 channels[0].low=20.25;queueViewerSetting(displayPath,{channels,auto:{method:'Otsu',scope:''}});
 await writeViewerSetting(displayJob);
 assert(!displayJob.pending&&document.getElementById('message').textContent.includes('This view is invalid.'),
   'An unrelated successful save does not clear another settings failure');
 mode='ok';await flushViewerSettings();
 assert(!notices.some(n=>n.text.includes('signal is aborted')),'Raw browser abort jargon never reaches the viewer');
 window.fetch=originalFetch;message=originalMessage;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER, setup=setup, timeout=100)
