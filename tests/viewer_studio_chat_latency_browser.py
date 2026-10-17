"""Progress must arrive through Streamlit's compression before completion; Stop cancels edits."""

import json
import time

from viewer_studio_active_labels_browser import JOB, setup as prepare
from viewer_threshold_browser import main


def setup(root):
    from spacemap_assistant.model_client import LocalModelClient
    from spacemap_gui.studio import chat

    prepare(root)
    class Model(LocalModelClient):
        def stream_complete(self, messages, on_progress=None, **kwargs):
            text = messages[-1]["content"]
            if isinstance(text, list):
                text = text[0]["text"]
            if "cancel" in text:
                for _ in range(100):
                    on_progress({"status":"Slow test model is connected…"})
                    time.sleep(.1)
                yield json.dumps({"say":"Proposed labels", "action":{"name":"name_clusters", "args":{"job_id":JOB, "names":{"Cluster 0":"Must not apply"}}}})
            else:
                on_progress({"status":"Test model is reasoning…"})
                time.sleep(1.5)
                yield "Streaming finished successfully."
    chat._client = lambda: Model()
    from spacemap_gui.studio import ai_setup
    ai_setup._online_checks = {}
    def check(model):
        ai_setup._online_checks[model] = {"ok":True,"message":"Connection check passed","seconds":.7,"checked_at":"2026-10-10"}
    ai_setup.check_online = check
    ai_setup.state = lambda: {
        "enabled":True,"pinned":False,"hardware":"Browser test","missing":[],"models":[],"runtime":{},
        "precisions":[["auto","Automatic"]],"precision":"auto","stop_with_spacemap":True,"leftovers":[],
        "online":{"chosen":True,"model":"test-nvidia","models":[["test-nvidia","Test NVIDIA"]],"key_saved":True,
                  "details":{"test-nvidia":{"images":False,"description":"Test model"}},"checks":dict(ai_setup._online_checks)},
    }


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(25);}throw Error('Timed out: '+check);};
let checks=0;const assert=(value,message)=>{if(!value)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const ask=text=>{const input=document.querySelector('[aria-label="Message the assistant"]');input.value=text;input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));};
const panel=()=>document.querySelector('#studio-assistant');
setTimeout(async()=>{try{
  await until(()=>window.__state.cellsReady && document.querySelector('[aria-label="Message the assistant"]'));
  const started=performance.now();ask('Explain this view');
  await until(()=>panel().textContent.includes('Test model is reasoning'));
  assert(!panel().textContent.includes('Streaming finished successfully'), 'Progress reaches the browser before completion through Streamlit gzip');
  await until(()=>panel().textContent.includes('Streaming finished successfully'));
  assert(performance.now()-started<15000, 'A streamed answer completes normally');
  await until(()=>!panel().querySelector('.studio-chat-input button').disabled);
  ask('Label these clusters; cancel this test');
  await until(()=>panel().textContent.includes('Slow test model is connected'));
  const stopped=performance.now();document.querySelector('[aria-label="Stop assistant reply"]').click();
  await until(()=>panel().textContent.includes('Stopped. No new labels were applied.') && !panel().querySelector('.studio-chat-input button').disabled);
  assert(performance.now()-stopped<1000, 'Stop returns control immediately');
  ask('Explain this view again');
  await until(()=>Array.from(panel().querySelectorAll('.assistant p')).filter(p=>p.textContent==='Streaming finished successfully.').length===2);
  const jobs=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  assert(!jobs.some(job=>job.kind==='name_clusters'), 'A stopped review cannot apply a late naming result');
  assert(document.querySelector('[aria-label="Stop assistant reply"]').hidden, 'Stop disappears when the reply is finished');
  Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='AI setup').click();
  await until(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Check connection'));
  assert(document.querySelector('[aria-label="AI setup"]').textContent.includes('Text only'), 'AI setup identifies text-only models');
  Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Check connection').click();
  await until(()=>document.querySelector('[aria-label="AI setup"]').textContent.includes('Connection check passed'));
  assert(document.querySelector('[aria-label="AI setup"]').textContent.includes('0.7 s'), 'Connection check displays the measured response time');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+panel()?.textContent.slice(-1400)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup, screenshot=False, compress=True)
