"""Image changes and an actual browser reload with the local cache cleared."""
from viewer_threshold_browser import main


def setup(root):
    import json
    import numpy as np
    import tifffile
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api, registry
    from spacemap_gui.viewer.ui import using_context, layers_for_image, _workspace_token

    specs = []
    for folder in ('a', 'b'):
        image = root / folder / 'same.ome.tiff'
        image.parent.mkdir()
        values = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(3, 64, 64)
        tifffile.imwrite(image, values, metadata={'axes': 'CYX', 'Channel': {
            'Name': ['CD3', 'DAPI', 'PanCK'], 'Color': [-16776961, 65535, 16711935]}})
        with using_context({'root': str(root), 'mappings': {}}):
            layers, _ = layers_for_image({'path': str(image), 'name': image.name, 'sample': folder, 'group': 'Images'}, context='tab4')
            workspace = _workspace_token(str(image))
        path = root / (folder + '.geojson')
        path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{'type': 'Feature',
            'properties': {'classification': {'name': 'tissue'}}, 'geometry': {'type': 'Polygon',
                'coordinates': [[[2, 2], [20, 2], [20, 20], [2, 20], [2, 2]]]}}]}))
        token = registry.register_object('annotations', path=str(path), image=str(image))
        layers.append({'id': 'ann0', 'type': 'annotations', 'name': 'Imported tissue', 'token': token,
                       'image_token': layers[0]['token'], 'visible': True, 'opacity': .7})
        layers.append({'id': 'he', 'type': 'image', 'name': 'H&E', 'token': layers[0]['token'], 'visible': True,
                       'channels': [{'index': 0, 'visible': True, 'low': 0, 'high': 12000}]})
        specs.append({'id': 'view-' + folder, 'layers': layers, 'workspace': workspace})

    async def fixture(request):
        return JSONResponse({'specs': specs})
    api.app.router.routes.append(Route('/saved-view-fixtures', fixture))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<700;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=Number(sessionStorage.getItem('saved-view-checks')||0);
const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state;fetch('/result?progress=Opening%20browser%20phase%20'+sessionStorage.getItem('saved-view-phase'));await until(()=>s.imagesReady);
 const {specs}=await getJSON('/saved-view-fixtures'), a=specs[0], b=specs[1];
 const open=async spec=>{await init(JSON.parse(JSON.stringify(spec)));await until(()=>s.imagesReady);};
 const image=()=>s.layers.find(l=>l.id==='image');
 const ann=()=>s.layers.find(l=>l.type==='annotations');
 const server=async spec=>(await getJSON('/workspace/'+spec.workspace+'/view')).view;
 const change=(input,v)=>{input.value=v;input.dispatchEvent(new Event('change'));};
 if(sessionStorage.getItem('saved-view-phase')==='reload'){
   await open({...a,id:'changed-after-app-restart',layers:a.layers.map(l=>l.id==='ann0'?{...l,id:'ann9'}:l)});
   assert(image().channels.every(c=>!c.visible),'An all-off marker selection survives a real reload with browser cache cleared');
   assert(image().channels[2].color==='#123456'&&image().channels[2].low===17.25&&image().channels[2].high===2345,'Custom colours and precise display ranges restore from project JSON');
   assert(ann().visible&&ann().opacity===.32&&ann().hidden.has('tissue'),'Imported layer visibility, opacity and hidden classes restore even if its numbered ID changed');
   assert(!s.layers.some(l=>l.id==='drawings'),'A removed layer stays removed');
   assert(s.layers.some(l=>l.id==='extra-review'&&l.visible===false),'An added annotation layer survives cache clearing and reload');
   assert(s.layers.find(l=>l.id==='he').visible&&s.layers.find(l=>l.id==='he').channels[0].high===12345,'H&E visibility and channel display ranges restore');
   assert(s.layerGroups.annotations===true&&s.layersShownOnly===false,'Panel folding and filter restore');
   const v=s.viewer.viewport;assert(Math.abs(v.getZoom()-3)<.1&&Math.abs(v.getCenter().x-.35)<.02,'Zoom and position restore');
   // A saved image with all channels off must not seed the next unseen image.
   await open(b);assert(image().channels[0].visible&&!image().channels[1].visible,'The second image keeps its own marker choices');
   fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));return;
 }
 await open(a);
 assert(image().channels.filter(c=>c.visible).map(c=>c.name).join(',')==='DAPI','An unseen image starts with DAPI, even when it is not channel zero');
 assert(image().channels.every(c=>c.color===image().info.colors[c.index]),'Unseen markers use their original image colours');
 assert(s.layers.filter(l=>l.id!=='image').every(l=>!l.visible),'All overlays, including H&E, start off');
 // Use real marker controls and switch immediately, before the debounce timeout.
 image().inputs[2].row.querySelector('input[type=checkbox]').click();
 await until(()=>Number.isFinite(image().channels[2].high));
 change(image().inputs[2].low,17.25);change(image().inputs[2].high,2345);
 const color=image().inputs[2].row.querySelector('input[type=color]');color.value='#123456';color.dispatchEvent(new Event('input'));
 ann().hidden.add('tissue');setOpacity(ann(),.32);setVisible(ann(),true);
 s.layerGroups.annotations=true;buildPanel();
 const he=s.layers.find(l=>l.id==='he');he.channels[0].high=12345;remember(he);setVisible(he,true);
 removeLayer(s.layers.find(l=>l.id==='drawings'));
 await addLayerToView({...a.layers.find(l=>l.id==='ann0'),id:'extra-review',visible:false});
 s.viewer.viewport.zoomTo(3,null,true);s.viewer.viewport.panTo(new OpenSeadragon.Point(.35,.4),true);saveView();
 await open(b);
 assert(image().channels.filter(c=>c.visible).map(c=>c.name).join(',')==='DAPI','Same-named images in different folders do not share checkboxes');
 assert(!image().channels[2].custom&&image().channels[2].color===image().info.colors[2]&&image().channels[2].low===undefined,'New images do not inherit custom colours or thresholds');
 image().channels.forEach(c=>c.visible=c.index===0);remember(image());
 await open(a);
 assert(image().channels[2].visible&&image().channels[2].color==='#123456'&&image().channels[2].low===17.25,'An immediate image switch flushes the last marker edits');
 assert(ann().visible&&ann().opacity===.32,'Each image restores its own annotation settings');
 const saved=await server(a);assert(saved.keyedLayers&&saved.layers.image.channels[2].color==='#123456','The complete view is stored in a server JSON record');
 // All-off is intentional; do not force the first marker back on.
 image().channels.forEach(c=>c.visible=false);remember(image());refreshLayer(image());
 await Promise.race([flushViewerSettings(),delay(15000).then(()=>{throw Error('Save queue timed out: '+JSON.stringify(Array.from(viewerSettingWrites.values(),j=>({key:j.key,pending:!!j.pending,running:!!j.running,body:j.pending?.slice(-160)}))))})]);
 assert((await getJSON('/display/'+image().display_token)).channels.every(c=>!c.visible),'All-off is also saved in the existing marker settings file');
 await until(()=>image().item?.getFullyLoaded());assert(true,'All-off marker tiles finish loading');
 s.viewer.viewport.zoomTo(3,null,true);s.viewer.viewport.panTo(new OpenSeadragon.Point(.35,.4),true);saveView();await Promise.race([flushViewerSettings(),delay(15000).then(()=>{throw Error('Save queue timed out: '+JSON.stringify(Array.from(viewerSettingWrites.values(),j=>({key:j.key,pending:!!j.pending,running:!!j.running,body:j.pending?.slice(-160)}))))})]);
 sessionStorage.setItem('saved-view-phase','reload');sessionStorage.setItem('saved-view-checks',String(checks));
 window.addEventListener('pagehide',()=>localStorage.clear());
 localStorage.clear();location.reload();
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""


if __name__ == '__main__':
    main(DRIVER, setup=setup, timeout=150, screenshot=False)
