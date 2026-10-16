"""Browser-check the assistant's streamed answers and multi-step plans on a disposable project in headless Firefox.

A fake model (set up in the server's process) streams a plain answer, or a plan: change Leiden's resolution, then
run Leiden. The plan runs after one Do it, waiting for the job, and the open Leiden panel shows the new setting.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_plan_browser.py
"""

from viewer_threshold_browser import main

ANSWER = "Open Phenotyping, choose Leiden, pick the markers and press Run Leiden; then click a cluster to see it on the image."
DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
const ANSWER=%s;
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[aria-label="Phenotyping method"]'));
  const picker=document.querySelector('[aria-label="Phenotyping method"]');picker.value='leiden';picker.dispatchEvent(new Event('change'));
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Resolution"]'));
  document.querySelector('[data-tab=assistant]').click();
  const pane=()=>document.querySelector('#studio-assistant');
  await until(()=>/Local|Online/.test(pane().textContent) && pane().querySelector('.studio-bubble.assistant'));
  const input=pane().querySelector('[aria-label="Message the assistant"]');
  input.value='how do I cluster cells?';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  // Streaming: part of the answer shows before the whole of it.
  let partial=false;
  await until(()=>{const said=Array.from(pane().querySelectorAll('.studio-bubble.assistant p')).map(p=>p.textContent);const last=said[said.length-1]||'';
    if(last && last!==ANSWER && ANSWER.startsWith(last))partial=true;return last===ANSWER && !pane().querySelector('.studio-bubble.note');});
  assert(partial,'The answer shows while it is written');
  input.value='run leiden at resolution 1.5 (plan)';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>pane().querySelector('.studio-plan-card'));
  const card=pane().querySelector('.studio-plan-card');
  const steps=card.querySelectorAll('.studio-plan-step');
  assert(steps.length===2 && /change settings: method leiden/.test(steps[0].textContent) && /run leiden/.test(steps[1].textContent),'The plan lists its steps in order');
  assert(Array.from(steps).every(step=>/changes project/.test(step.textContent) && step.querySelector('input[type=checkbox]').checked),'Each step says it changes the project and is ticked');
  assert(!/Preparing an action/.test(pane().textContent) && !/"plan"/.test(pane().textContent),'The plan JSON is never shown as text');
  button(card,'Do it').click();
  await until(()=>/Every ticked step ran|The plan stopped/.test(card.textContent));
  assert(/Every ticked step ran/.test(card.textContent) && Array.from(card.querySelectorAll('.studio-plan-state')).every(s=>/^done/.test(s.textContent)),
    'The steps ran in order, the run waited for its job: '+card.textContent.slice(0,400));
  const saved=await (await fetch('/spacemap-viewer/api/studio/browser-check/leiden_settings')).json();
  assert(saved.values.resolution===1.5,'The resolution is saved in Tab 7’s Leiden settings');
  const jobs=(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  assert(jobs.some(job=>job.kind==='leiden' && job.status==='done'),'Leiden ran as a job');
  await until(()=>tools().querySelector('[aria-label="Resolution"]')?.value==='1.5');
  assert(true,'The open Leiden panel shows the new setting');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-assistant')?.textContent.slice(-900)));}
},100);
</script>
"""


class FakeStreamer:
    """Streams a plain answer, or a plan when asked for one (the last message says plan)."""

    def complete(self, messages, **options):
        return "".join(self.stream_complete(messages, **options))

    def stream_complete(self, messages, **options):
        import json
        import time

        if "plan" in str(messages[-1]["content"]):
            answer = json.dumps(
                {
                    "say": "1. Set Leiden's resolution to 1.5. 2. Run Leiden.",
                    "plan": [
                        {"name": "change_settings", "args": {"method": "leiden", "changes": {"resolution": 1.5}}},
                        {"name": "run_leiden", "args": {}},
                    ],
                }
            )
        else:
            answer = ANSWER
        for start in range(0, len(answer), 8):
            time.sleep(0.06)
            yield answer[start : start + 8]


def setup(root):
    """Tab 7's Leiden markers, and the fake model in place of the real one (this process serves the page)."""
    import json

    from spacemap import ai_settings
    from spacemap_gui.studio import chat

    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "leiden_params.json").write_text(json.dumps({"markers": ["M0", "M1"], "n_neighbors": 5, "subsample_size": 1000}))
    ai_settings.is_enabled = lambda: True
    chat._client = FakeStreamer
    chat.sees_images = lambda: False


if __name__ == "__main__":
    import json

    main(DRIVER % json.dumps(ANSWER), studio=True, timeout=400, setup=setup)
