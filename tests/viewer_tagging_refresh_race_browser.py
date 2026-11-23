"""A delayed saved-tag refresh cannot override a later cell-data/colour choice."""
from viewer_threshold_browser import main
from viewer_studio_tagging_browser import setup as tagging_setup

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(const end=Date.now()+60000;Date.now()<end;){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 await until(()=>window.__state?.imagesReady&&window.__state?.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 await state.phenotypingLink;
 const cells=state.layers.find(l=>l.type==='cells'), original=cells.table, generation=state.initGeneration;
 const settings=await (await fetch('/spacemap-viewer/api/studio/browser-check/tagging_settings')).json();
 const outputs=[settings.paths.output+'/'+settings.tagged['roi_cell_features.csv']];
 const linked=await postJSON('/studio/browser-check/tagging_result',{token:cells.token,outputs});
 check(linked.linked,linked.linked?'The completed table matches the current cells':'The completed table is not linked: '+JSON.stringify(linked));
 const realFetch=window.fetch;let release,held=false;
 window.fetch=async(url,...args)=>{
   const response=await realFetch(url,...args);
   if(String(url).endsWith('/tagging_result')){held=true;await new Promise(resolve=>{release=resolve;});}
   return response;
 };
 const pending=refreshTaggedCells(outputs);
 await until(()=>held);
 setCellColour(cells,'M1'); const version=state.versions[cells.id];
 release();const applied=await pending;window.fetch=realFetch;
 check(!applied&&cells.table===original&&cells.color_by==='M1'&&state.versions[cells.id]===version,
   'A newer colour choice wins over a delayed tagging response');
 check(state.initGeneration===generation,'The rejected response does not reinitialize the image');
 check(!await refreshTaggedCells([settings.paths.output+'/another_image.csv'])&&cells.table===original,
   'Another image finishing does not change current cell data');
 showDataTab();await until(()=>dataView.result?.rows.length);
 const sourceKey=dataView.sourceKey;dataDirtySources.add(sourceKey);
 check(!await refreshTaggedCells(outputs)&&cells.table===original&&dataView.sourceKey===sourceKey,
   'Unsaved Data edits prevent an automatic source replacement');
 dataDirtySources.delete(sourceKey);
 check(await refreshTaggedCells(outputs)&&cells.table.startsWith('tagged_')&&cells.color_by==='M1',
   'Saved tagging can refresh the current cells without changing their chosen colour');
 const saved=state.versions[cells.id];
 check(!await refreshTaggedCells(outputs)&&state.versions[cells.id]===saved,
   'The same saved table is not applied repeatedly');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>'''


def setup(root):
    import pandas as pd
    from spacemap_gui.studio import tagging
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.tabs import cell_tagging_ui as tab5
    from pathlib import Path

    tagging_setup(root)
    state = tagging.values(root, read_config(root))
    folder = Path(state[tab5.OUTPUT_DIR_KEY])
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / 'cells.csv')
    frame['tissue'] = 1
    frame.to_csv(folder / 'roi_cell_features_tagged.csv', index=False)


if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=180)
