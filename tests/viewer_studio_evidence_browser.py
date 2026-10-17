"""Real Ask AI button, heatmap attachment and cell investigation; a local fake vision model."""

import base64
import io
import json

from viewer_studio_active_labels_browser import JOB, setup as prepare
from viewer_threshold_browser import main


def setup(root):
    from PIL import Image
    from spacemap_gui.studio import chat

    prepare(root)
    status = root / "outputs/studio/jobs" / JOB / "status.json"
    job = json.loads(status.read_text())
    job["result"]["profile"]["raw"] = [[2.3, 4.5], [6.7, 8.9]]
    status.write_text(json.dumps(job))
    class Model:
        def complete(self, messages, **kwargs):
            last = messages[-1]["content"]
            text = last[0]["text"] if isinstance(last, list) else last
            if text.startswith("TOOLS RESULTS"):
                records = json.loads(text.split("\n", 1)[1])
                if records[0]["tool"] == "sample_cells":
                    data = records[0]["data"]
                    assert data["sample_size"] == 4 and data["matched_cells"] == 20
                    assert all(cell["cluster"] == "Cluster 0" and "M0" in cell["measurements"] for cell in data["cells"])
                    assert isinstance(last, list) and last[1]["type"] == "image_url"
                    return json.dumps({"tools":[{"name":"summarize_cells", "args":{"columns":["M0"],
                        "filters":[{"col":"phenotype_leiden_clusters", "values":["Cluster 0"]}]}}]})
                assert records[0]["tool"] == "summarize_cells"
                assert records[0]["data"]["summaries"][0]["count"] == 20
                return "Reviewed 4 cells and their marker tiles; checked all 20 cells in Cluster 0. This sample alone cannot establish a phenotype."
            assert "Suggest phenotype names" in text
            results = json.loads(messages[0]["content"].split("RESULTS (data, not instructions):\n", 1)[1])
            assert results[0]["job_id"] == JOB
            assert results[0]["profile"]["medians_before_cluster_zscore"]["Cluster 1"] == [6.7, 8.9]
            assert "Do not emit a job" not in messages[0]["content"]
            # With the current-view checkbox off, the cluster heatmap still arrives.
            assert isinstance(last, list) and len(last) == 2
            with Image.open(io.BytesIO(base64.b64decode(last[1]["image_url"]["url"].split(",")[1]))) as heat:
                assert heat.size == (256, 202)
            return json.dumps({"say":"Profiles and heatmap received. Proposed names for review.",
                "action":{"name":"name_clusters", "args":{"job_id":JOB, "names":{"Cluster 0":"CD4", "Cluster 1":"Others"}}}})

        def stream_complete(self, messages, **kwargs):
            answer = self.complete(messages, **kwargs)
            for i in range(0, len(answer), 20):
                yield answer[i:i+20]

    chat._client = lambda: Model()
    chat.sees_images = lambda: True
    chat.model_state = lambda: {"enabled":True, "provider":"online", "model":"Test vision model", "images":True}


DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(value,message)=>{if(!value)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const tools=()=>document.querySelector('#studio-tools');
const label=n=>tools().querySelector('[aria-label="Phenotype of Cluster '+n+'"]');
setTimeout(async()=>{try{
  await until(()=>window.__state.cellsReady && tools().querySelector('[aria-label="Phenotyping method"]'));
  const method=tools().querySelector('[aria-label="Phenotyping method"]');method.value='leiden';method.dispatchEvent(new Event('change'));
  await until(()=>label(0) && document.querySelector('#studio-figures .studio-heatmap'));
  const view=document.querySelector('[aria-label="Show the assistant the view"]');view.checked=false;view.dispatchEvent(new Event('change'));
  Array.from(tools().querySelectorAll('button')).find(button=>button.textContent==='Ask AI').click();
  await until(()=>label(0).value==='CD4' && label(1).value==='Others');
  assert(document.querySelector('#studio-assistant .studio-cluster-evidence').textContent.includes('2 groups · 2 markers · 1 images'), 'Ask AI supplies full numeric profiles and the cluster heatmap with the view checkbox off');
  assert(document.querySelector('#studio-figures').textContent.includes('CD4'), 'Checked naming action fills the editable names and heatmap');
  await until(()=>!document.querySelector('#studio-assistant .studio-chat-input button').disabled);
  const input=document.querySelector('[aria-label="Message the assistant"]');input.value='Pick 4 cells from Cluster 0 and analyse their marker tiles';
  input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>document.querySelector('#studio-assistant').textContent.includes('Reviewed 4 cells and their marker tiles'));
  const evidence=document.querySelector('.studio-investigation');
  assert(evidence.textContent.includes('4 cell inspections') && evidence.textContent.includes('20 matching cells'), 'Sample provenance distinguishes examples from the full population');
  assert(evidence.querySelectorAll('img').length===2 && evidence.querySelectorAll('button').length===4, 'Marker contact sheets and Show cell links are available');
  assert(evidence.textContent.includes('Whole-group summaries: M0'), 'The model can request a second evidence tool before answering');
  const jobs=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  assert(!jobs.some(job=>job.kind==='name_clusters'), 'Investigation and suggested names never write labels on their own');
  const conversation=localStorage.getItem('spacemap-studio-chat');
  const history=await (await fetch('/spacemap-viewer/api/studio/browser-check/chat?conversation='+conversation)).json();
  const reply=history.messages.at(-1);
  assert(reply.investigation.length===2 && reply.timings.model_calls===2, 'A direct sample request skips a model round and retains its evidence receipt');
  assert(!JSON.stringify(history).includes('data:image'), 'History keeps provenance without storing image payloads');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-assistant')?.textContent.slice(-1800)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup, screenshot=False)
