"""Browser-check Studio's Project step: the open project, the samples grid on one line per sample, switching projects.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_project_browser.py
"""

import json
import os

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<1200;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab0]'));
  document.querySelector('[data-step=tab0]').click();
  const tools=()=>document.querySelector('#studio-tools');
  const title=()=>tools().querySelector('.studio-project-title')?.textContent;
  await until(()=>title() && tools().querySelector('.studio-matrix-row.head'));
  assert(/Liver cohort/.test(tools().textContent), 'The project shows its description');
  const head=tools().querySelector('.studio-matrix-row.head');
  assert(head.querySelectorAll('button.studio-matrix-step').length===10 && head.getBoundingClientRect().height<30, 'All ten step numbers sit on the header line');
  assert(!tools().querySelector('[aria-label="Refresh list"]') && !/AI setup/.test(tools().textContent), 'No second project list, no assistant settings');
  const row=name=>tools().querySelector('tr[data-project="'+name+'"]');
  assert(row(title())?.classList.contains('selected') && /Not found/.test(row('Gone').textContent), 'The list marks the open project and a missing folder');
  row('Gone').querySelector('button').click();
  await until(()=>!row('Gone'));
  row('Other').querySelector('button.studio-item-name').click();
  await until(()=>title()==='Other' && /Other/.test(document.querySelector('.studio-project strong').textContent));
  assert(row('Other').classList.contains('selected'), 'Switching opens the other project in Studio');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+(document.querySelector('#studio-tools')?.textContent||'').slice(0,800)));}
},100);
</script>
"""


def setup(root):
    """This project's settings file, another project, and a listed project whose folder is gone, in a throwaway home."""
    from spacemap_gui.persistence import save_project_to_index

    os.environ["HOME"] = str(root / "home")
    (root / f"{root.name}.json").write_text(json.dumps({"project_name": root.name, "description": "Liver cohort"}))
    other = root / "elsewhere" / "Other"
    other.mkdir(parents=True)
    (other / "Other.json").write_text(json.dumps({"project_name": "Other", "created_at": "2026-10-06T09:00:00"}))
    save_project_to_index(root.name, str(root))
    save_project_to_index("Other", str(other))
    save_project_to_index("Gone", str(root / "moved_away"))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup)
