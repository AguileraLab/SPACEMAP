"""The real chat/command path with a local test model and disposable saved clusters."""

import json

from viewer_studio_leiden_browser import setup as prepare
from viewer_threshold_browser import main

JOB = "b" * 32


def setup(root):
    import pandas as pd
    from spacemap import ai_settings, dataio
    from spacemap_gui.studio import chat

    prepare(root)
    table = root / "cells.csv"
    frame = pd.read_csv(table)
    frame["phenotype_leiden_clusters"] = ["Cluster 0"] * 20 + ["Cluster 1"] * 20
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    folder = root / "outputs/studio/jobs" / JOB
    folder.mkdir(parents=True)
    result = {"table":str(table), "column":"phenotype_leiden_clusters", "phenotype_column":"phenotype_leiden",
        "markers":["M0","M1"], "cells":40, "where":{"Cluster 0":{"roi":20}, "Cluster 1":{"roi":20}},
        "clusters":[{"name":"Cluster 0", "id":0, "cells":20, "top_markers":["M0"]},
                    {"name":"Cluster 1", "id":1, "cells":20, "top_markers":["M1"]}],
        "profile":{"columns":["M0","M1"], "rows":["Cluster 0","Cluster 1"], "values":[[1,0],[0,1]], "note":"Test profiles"}}
    (folder / "request.json").write_text(json.dumps({"project":str(root), "kind":"leiden", "settings":{}}))
    (folder / "status.json").write_text(json.dumps({"id":JOB,"kind":"leiden","status":"done","progress":1,"result":result}))
    class Model:
        def complete(self, messages, **kwargs):
            action = {"say":"Proposed labels", "plan":[{"name":"name_clusters", "args":{"job_id":JOB,"names":{"Cluster 0":"CD4","Cluster 1":"Others"}}}]}
            # The exact malformed plan-container closer reported by the user.
            return json.dumps(action)[:-2] + "}}"
    chat._client = lambda: Model()
    ai_settings.is_enabled = lambda: True
    chat.model_state = lambda: {"enabled":True,"provider":"local","model":"Local test model","running":True,"images":False}


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const jobs=async()=> (await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
const tools=()=>document.querySelector('#studio-tools');
const label=n=>tools().querySelector('[aria-label="Phenotype of Cluster '+n+'"]');
const ask=text=>{const input=document.querySelector('[aria-label="Message the assistant"]');input.value=text;input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));};
setTimeout(async()=>{try{
  await until(()=>window.__state.cellsReady && tools().querySelector('[aria-label="Phenotyping method"]'));
  const method=tools().querySelector('[aria-label="Phenotyping method"]');method.value='leiden';method.dispatchEvent(new Event('change'));
  await until(()=>label(0));
  Array.from(tools().querySelectorAll('button')).find(button=>button.textContent==='Ask AI').click();
  await until(()=>label(0).value==='CD4' && label(1).value==='Others');
  assert(document.querySelector('#studio-figures').textContent.includes('CD4'), 'Ask AI fills the cluster fields and heatmap');
  assert(!(await jobs()).some(job=>job.kind==='name_clusters'), 'A suggestion stays editable until requested to save');
  const saved=async count=>{await until(async()=> (await jobs()).filter(job=>job.kind==='name_clusters'&&job.status==='done').length===count);await until(()=>!document.querySelector('#studio-assistant .studio-chat-input button').disabled);return (await jobs()).find(job=>job.kind==='name_clusters');};
  ask('ok, do that');
  const first=await saved(1);
  const cells=window.__state.layers.find(layer=>layer.type==='cells');
  await until(()=>cells.color_by===first.result.column);
  assert(first.result.names['Cluster 0']==='CD4' && first.result.names['Cluster 1']==='Others', 'Conversational approval saves the proposal and paints its phenotype column');
  ask('ok, do that');
  await until(()=>document.querySelector('#studio-assistant').textContent.includes('Already saved'));
  assert((await jobs()).filter(job=>job.kind==='name_clusters').length===1, 'Repeated approval does not create another result');
  ask('Rename Cluster 0 to CD8');
  const second=await saved(2);
  assert(label(0).value==='CD8' && second.result.names['Cluster 1']==='Others', 'A direct edit changes one label and keeps the others');
  ask('Remove the label for Cluster 0');
  const third=await saved(3);
  assert(label(0).value==='' && label(1).value==='Others' && !third.result.names['Cluster 0'], 'Clearing a name updates the boxes and preserves other names');
  ask('Label these clusters');
  const fourth=await saved(4);
  assert(label(0).value==='CD4' && cells.color_by===fourth.result.column, 'An explicit request runs without a separate Do it click');
  const remembered=await (await fetch('/spacemap-viewer/api/studio/browser-check/cluster_names?job_id=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')).json();
  assert(remembered.names['Cluster 0']==='CD4' && remembered.column===fourth.result.column, 'The saved names can be restored when the panel reopens');
  const conversation=localStorage.getItem('spacemap-studio-chat');
  const history=await (await fetch('/spacemap-viewer/api/studio/browser-check/chat?conversation='+conversation)).json();
  assert(history.messages.some(message=>message.execution?.status==='done'&&message.execution.column===fourth.result.column), 'Conversation history reports saved execution receipts');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-assistant')?.textContent.slice(-1800)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=400, setup=setup, screenshot=False)
