"""The reported native tool reply, through real chat streaming and marker crops."""

import json

from viewer_studio_active_labels_browser import JOB as OLD_JOB, setup as prepare
from viewer_threshold_browser import main

JOB = "81fe69ebf58a4f15a9a9c255cc9287d0"
NATIVE = '<|tool_call>call:spacemap_ai:sample_cells{job_id: "81fe69ebf58a4f15a9a9c255cc928d0",clusters:[4],count:6}<tool_call|><|tool_call>call:spacemap_ai:sample_cells{job_id: "81fe69ebf58a4f15a9a9c255cc928d0",clusters:[25],count:6}<tool_call|>'


def setup(root):
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui.studio import chat

    prepare(root)
    table = root / "cells.csv"
    frame = pd.read_csv(table)
    frame["phenotype_leiden_clusters"] = ["Cluster 4"]*20 + ["Cluster 25"]*20
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    folder = root / "outputs/studio/jobs" / OLD_JOB
    folder.rename(folder.with_name(JOB))
    status = folder.with_name(JOB) / "status.json"
    saved = status.read_text().replace(OLD_JOB, JOB).replace("Cluster 0", "Cluster 4").replace("Cluster 1", "Cluster 25")
    status.write_text(saved)

    class Model:
        def stream_complete(self, messages, **kwargs):
            last = messages[-1]["content"]
            text = last[0]["text"] if isinstance(last, list) else last
            if text.startswith("TOOLS RESULTS"):
                checks = json.loads(text.split("\n", 1)[1])
                assert [r["args"]["clusters"] for r in checks] == [["Cluster 4"], ["Cluster 25"]]
                assert all(r["args"]["job_id"] == JOB and r["data"]["sample_size"] == 6 for r in checks)
                assert all(r["data"]["matched_cells"] == 20 for r in checks)
                assert all(c["images_available_to_model"] and "M0" in c["measurements"] and c["crop"] for r in checks for c in r["data"]["cells"])
                assert isinstance(last, list) and len(last) == 4  # Three sheets, all twelve cells.
                answer = "Reviewed 6 cells from Cluster 4 and 6 from Cluster 25 with their marker tiles and measurements. Their identity remains tentative."
            else:
                answer = NATIVE
            for char in answer:
                yield char  # Protocol prefixes may be split at any byte boundary.

    chat._client = lambda: Model()
    chat.sees_images = lambda: True
    chat.model_state = lambda: {"enabled":True, "provider":"online", "model":"Protocol test model", "images":True}


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1800;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(value,message)=>{if(!value)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const ask=text=>{const input=document.querySelector('[aria-label="Message the assistant"]');input.value=text;input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));};
const thread=()=>document.querySelector('#studio-assistant');
const chatHistory=async()=> (await (await fetch('/spacemap-viewer/api/studio/browser-check/chat?conversation='+localStorage.getItem('spacemap-studio-chat'))).json()).messages;
setTimeout(async()=>{try{
  await until(()=>window.__state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const method=document.querySelector('[aria-label="Phenotyping method"]');method.value='leiden';method.dispatchEvent(new Event('change'));
  await until(()=>document.querySelector('[aria-label="Phenotype of Cluster 4"]'));
  let leaked=false;
  const observer=new MutationObserver(()=>{if(/tool_call|call:spacemap_ai/.test(thread().textContent))leaked=true;});
  observer.observe(thread(),{childList:true,characterData:true,subtree:true});
  ask('Compare clusters 4 and 25 visually');
  await until(()=>thread().textContent.includes('Reviewed 6 cells from Cluster 4'));
  await until(()=>!thread().querySelector('.studio-chat-input button').disabled);
  const reply=(await chatHistory()).at(-1);
  assert(reply.timings.model_calls===2 && reply.investigation.length===2, 'Native tool text executes both sample requests and resumes the model');
  assert(reply.investigation.every(r=>r.args.job_id==='81fe69ebf58a4f15a9a9c255cc9287d0'), 'A dropped digit binds only to the selected saved run');
  const evidence=thread().querySelector('.studio-investigation');
  assert(evidence.querySelectorAll('img').length===3 && evidence.querySelectorAll('button').length===12, 'All twelve cell samples have marker tiles and Show cell links');
  assert(reply.investigation.every(r=>r.data.cells.every(c=>c.images_available_to_model)), 'The viewer screenshot does not displace the last group’s tiles');
  assert(!leaked, 'Tool syntax never appears in streamed or final chat bubbles');
  ask('Ok, you still need to test marker tiles for smple of cells from these clusters and decide on what to call them, can you do that?');
  await until(()=>thread().querySelectorAll('.studio-investigation').length===2);
  await until(()=>!thread().querySelector('.studio-chat-input button').disabled);
  const next=(await chatHistory()).at(-1);
  assert(next.timings.model_calls===1 && next.investigation.length===2, 'The exact follow-up collects both groups before the first model call');
  assert(next.investigation.map(r=>r.args.clusters[0]).join(',')==='Cluster 4,Cluster 25', 'These clusters refers to the previous user request');
  const jobs=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  assert(!jobs.some(job=>job.kind==='name_clusters') && !next.action, 'Reviewing sampled cells does not silently save phenotype labels');
  observer.disconnect();
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+thread()?.textContent.slice(-1800)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup, screenshot=False, compress=True)
