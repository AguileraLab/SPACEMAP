"""Browser-check Studio's Registration pairs on a disposable project in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_registration_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<3600;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
window.confirm=()=>true;
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-step=tab1]'));
  document.querySelector('[data-step=tab1]').click();
  const tools=()=>document.querySelector('#studio-tools');
  const pairs=async()=>await (await fetch('/spacemap-viewer/api/studio/browser-check/registration_pairs')).json();
  await until(()=>tools().querySelector('.studio-pairs') || /No CODEX image follows/.test(tools().textContent));
  await until(()=>tools().querySelector('tr[data-sample="CODEX_CRC02"]'));
  assert(/No H&E/.test(tools().querySelector('tr[data-sample="CODEX_CRC03"]').textContent),
    'CODEX images outside the naming rule are listed, to pair by hand or by changing the rule');
  // The naming rule: strip CODEX_ from CODEX names and -HE from H&E names.
  const set=(label,value)=>{const input=tools().querySelector('[aria-label="'+label+'"]');input.value=value;input.dispatchEvent(new Event('change'));};
  set('codex suffix','');
  await until(async()=>(await pairs()).rule.tab1_batch_codex_suffix==='');
  set('codex prefix','CODEX_');
  await until(async()=>(await pairs()).rule.tab1_batch_codex_prefix==='CODEX_');
  set('he suffix','-HE');
  await until(()=>tools().querySelector('tr[data-sample="CRC02"]') && /Auto: CRC02-HE\.tif/.test(tools().textContent));
  assert(tools().querySelectorAll('.studio-pairs tr[data-sample]').length===2,'The rule pairs CRC02 and CRC03');
  // A pair by hand: CRC03's CODEX with the oddly named H&E.
  const partner=tools().querySelector('[aria-label="H&E for CODEX_CRC03.qptiff"]');
  partner.value='S7 HE scan.tif';partner.dispatchEvent(new Event('change'));
  await until(async()=>(await pairs()).rows.find(row=>row.sample==='CRC03').he==='S7 HE scan.tif');
  const saved=await pairs();
  assert(saved.rows.find(row=>row.sample==='CRC03').choice==='S7 HE scan.tif' && saved.unpaired_he.includes('CRC03-HE.tif'),
    'A hand-picked H&E is saved as Tab 1’s choice; the H&E it replaced is listed as unpaired');
  // Register the ticked pair: a job starts for exactly that pair.
  await until(()=>tools().querySelector('[aria-label="Pick CRC03"]'));
  tools().querySelector('[aria-label="Pick CRC03"]').click();
  button(document.querySelector('.studio-runbar'),'Register picked').click();
  const jobs=async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/jobs')).json()).jobs;
  await until(async()=>(await jobs()).some(job=>job.kind==='register_pairs'));
  assert(true,'Register picked starts a registration job');
  // A registered pair shows its matched-points figure beside the image.
  await until(()=>tools().querySelector('tr[data-sample="CRC02"]') && button(tools().querySelector('tr[data-sample="CRC02"]'),'Matches'));
  button(tools().querySelector('tr[data-sample="CRC02"]'),'Matches').click();
  await until(()=>document.querySelector('#studio-figures [data-figure="registration-matches"] img'));
  assert(true,'The matched-points figure opens beside the image');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-900)));}
},100);
</script>
"""


def setup(root):
    """CODEX and H&E images whose names do not follow the default rule; CRC02 already registered, with its figure."""
    from PIL import Image

    for folder, names in (
        ("data/codex", ["CODEX_CRC02.qptiff", "CODEX_CRC03.qptiff"]),
        ("data/he", ["CRC02-HE.tif", "CRC03-HE.tif", "S7 HE scan.tif"]),
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / folder / name).touch()
    registered = root / "outputs/registered"
    (registered / "intermediate_images").mkdir(parents=True, exist_ok=True)
    (registered / "CRC02_merged.ome.tiff").touch()
    Image.new("RGB", (400, 200), "white").save(registered / "intermediate_images" / "CRC02-HE_he_registered_feature_matches.png")


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300, setup=setup)
