"""Replay both reported model formats through the real 34-cluster review/save UI."""

import json

from viewer_studio_active_labels_browser import JOB, setup as prepare
from viewer_threshold_browser import main

NAMES = dict(enumerate([
    'Epithelia', 'Others', 'Epithelia', 'Macrophage', 'Fibroblast', 'Macrophage', 'Epithelia',
    'Macrophage', 'Epithelia', 'Epithelia', 'Macrophage', 'Fibroblast', 'Fibroblast', 'CD8',
    'CD4', 'Epithelia', 'Macrophage', 'Macrophage', 'Fibroblast', 'Epithelia', 'Fibroblast',
    'Macrophage', 'Macrophage', 'Fibroblast', 'Epithelia', 'Others', 'Fibroblast', 'Macrophage',
    'B', 'Others', 'Others', 'CD8', 'Macrophage', 'Epithelia',
]))


def setup(root):
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui.studio import chat

    prepare(root)
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    frame['phenotype_leiden_clusters'] = [f'Cluster {i % 34}' for i in range(len(frame))]
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    path = root / 'outputs/studio/jobs' / JOB / 'status.json'
    status = json.loads(path.read_text())
    groups = [{'name':f'Cluster {i}', 'id':i, 'cells':int((frame.phenotype_leiden_clusters == f'Cluster {i}').sum()),
               'top_markers':['M0']} for i in range(34)]
    status['result'].update(clusters=groups, where={c['name']:{'roi':c['cells']} for c in groups},
        profile={'columns':['M0','M1'], 'rows':[c['name'] for c in groups],
                 'values':[[i/34, 1-i/34] for i in range(34)], 'note':'Synthetic UI regression profiles'})
    path.write_text(json.dumps(status))

    class Model:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            assert self.calls <= 2, 'No formatting repair call or inference on approval should occur'
            args = {'job_id':JOB[:-1], 'names':{f'Cluster {i}':name for i,name in NAMES.items()}}
            if self.calls == 1:
                value = {'name_clusters':args}
            else:
                args['names']['Cluster 0'] = 'Others'
                value = {'action':'name_clusters', 'args':args}
            return 'Names proposed from supplied profiles.\n```json\n' + json.dumps(value) + '\n```'
    model = Model()
    chat._client = lambda: model


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async check=>{for(let i=0;i<2000;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(ok,msg)=>{if(!ok)throw Error(msg);checks++;fetch('/result?progress='+encodeURIComponent(msg));};
const tools=()=>document.querySelector('#studio-tools');
const label=n=>tools().querySelector('[aria-label="Phenotype of Cluster '+n+'"]');
const jobs=async()=>(await(await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
const chatHistory=async()=>(await(await fetch('/spacemap-viewer/api/studio/browser-check/chat?conversation='+localStorage.getItem('spacemap-studio-chat'))).json()).messages;
const idle=()=>!document.querySelector('#studio-assistant .studio-chat-input button').disabled;
const ask=text=>{const input=document.querySelector('[aria-label="Message the assistant"]');input.value=text;input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));};
window.addEventListener('error', e=>fetch('/result?error='+encodeURIComponent(String(e.message)+'\n'+(e.error?.stack||''))));
window.addEventListener('unhandledrejection', e=>fetch('/result?error='+encodeURIComponent(String(e.reason)+'\n'+(e.reason?.stack||''))));
setTimeout(async()=>{try{
  await until(()=>window.__state.cellsReady && tools().querySelector('[aria-label="Phenotyping method"]'));
  const method=tools().querySelector('[aria-label="Phenotyping method"]');method.value='leiden';method.dispatchEvent(new Event('change'));
  await until(()=>label(33));
  Array.from(tools().querySelectorAll('button')).find(b=>b.textContent==='Ask AI').click();
  await until(()=>label(0).value==='Epithelia' && label(33).value==='Epithelia' && idle());
  assert(tools().querySelectorAll('[aria-label^="Phenotype of Cluster "]').length===34 && Array.from(tools().querySelectorAll('[aria-label^="Phenotype of Cluster "]')).every(e=>e.value), 'Keyed JSON with a dropped run-ID digit fills all 34 name boxes');
  assert(document.querySelector('#studio-figures').textContent.includes('Epithelia'), 'Recovered names update the heatmap');
  const draft=(await chatHistory()).at(-1);
  assert(draft.timings.model_calls===1 && !(await jobs()).some(j=>j.kind==='name_clusters'), 'Draft review uses one model call and does not save');
  ask('Name the cluster then');
  await until(async()=>idle() && (await jobs()).some(j=>j.kind==='name_clusters' && j.status==='done'));
  let saved=(await jobs()).filter(j=>j.kind==='name_clusters');
  assert(saved.length===1 && Object.keys(saved[0].result.names).length===34, 'Name the cluster then saves the existing 34 names');
  const approval=(await chatHistory()).filter(m=>m.role==='assistant').at(-1);
  assert(approval.timings.model_calls===0, 'Approval does not regenerate or change the proposed labels');
  const cells=window.__state.layers.find(l=>l.type==='cells');
  await until(()=>cells.color_by===saved[0].result.column);
  assert(true, 'Saved phenotype column colours the viewer cell layer');
  ask('Label all clusters using these profiles');
  await until(async()=>idle() && (await jobs()).filter(j=>j.kind==='name_clusters'&&j.status==='done').length===2);
  assert(label(0).value==='Others' && label(33).value==='Epithelia', 'Flat action JSON is accepted and saved through the explicit edit request');
  const last=(await chatHistory()).filter(m=>m.role==='assistant').at(-1);
  assert(last.timings.model_calls===1 && last.execution.status==='done', 'The saved action has a completion receipt and no formatting retry');
  assert(!document.querySelector('#studio-assistant').textContent.includes('did not return a valid Studio action'), 'Neither reported format produces the false invalid-action error');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+tools()?.textContent.slice(-1800)+'\n'+document.querySelector('#studio-assistant')?.textContent.slice(-1800)));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=300, setup=setup, screenshot=False, compress=True)
