"""Native zoom, screen detail, source-level selection, and durable interpolation controls."""
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import cv2
    import tifffile
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api
    from spacemap_gui.viewer.ui import using_context, image_layer, _workspace_token

    path = root/'detail.ome.tiff'
    y, x = np.indices((2048, 2048))
    pixels = np.stack([(x + y) % 256, (x % 2) * 255, (y % 2) * 255], axis=-1).astype('uint8')
    with tifffile.TiffWriter(path) as writer:
        writer.write(pixels, photometric='rgb', tile=(256, 256), subifds=3,
                     metadata={'axes': 'YXS', 'PhysicalSizeX': .5083, 'PhysicalSizeY': .5083})
        for factor in (2, 4, 8):
            writer.write(cv2.resize(pixels, (2048//factor, 2048//factor), interpolation=cv2.INTER_AREA),
                         photometric='rgb', tile=(256, 256), subfiletype=1)
    with using_context({'root': str(root), 'mappings': {}}):
        spec = {'id': 'pixel-detail', 'layers': [image_layer(str(path))], 'workspace': _workspace_token(str(path))}
    async def fixture(request):
        return JSONResponse(spec)
    api.app.router.routes.append(Route('/image-detail-fixture', fixture))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1400;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;
const assert=(ok,text)=>{if(!ok)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
  const s=window.__state;await until(()=>s.imagesReady);
  const spec=await getJSON('/image-detail-fixture');
  await init(spec);await until(()=>s.imagesReady);
  const layer=()=>s.layers.find(l=>l.type==='image');
  const zoom=p=>{s.viewer.viewport.zoomTo(layer().item.imageToViewportZoom(p/100),null,true);updateScaleBar();};
  const settled=async()=>{await until(()=>layer().item.getFullyLoaded());await delay(250);};
  const nativeDrawn=()=>layer().item._lastDrawn.length>0&&layer().item._lastDrawn.every(item=>item.level===layer().info.max_zoom);
  const menu=()=>{
    const canvas=s.viewer.canvas,b=canvas.getBoundingClientRect();
    canvas.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:b.left+b.width/2,clientY:b.top+b.height/2}));
    return document.querySelector('#ctxmenu');
  };
  const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(b=>b.textContent===text);
  zoom(75);await settled();
  await until(nativeDrawn);
  assert(nativeDrawn(),'75% zoom uses native detail instead of enlarging the half-resolution pyramid');
  assert(performance.getEntriesByType('resource').some(r=>/\/tile\/0\/.*\.png/.test(r.name)),'Native-resolution image requests use lossless PNG');
  assert(s.viewer.displaySmoothing===(OpenSeadragon.pixelDensityRatio*.75<1),'Reductions are filtered according to the actual display density');
  document.querySelector('#btn-native').click();
  await until(()=>Math.abs(imageZoom()-1)<.0001);await settled();
  assert(Math.abs(layer().item.viewportToImageZoom(s.viewer.viewport.getZoom(true))-1)<.0001,'100% shows one source pixel per viewer pixel independent of µm calibration');
  assert(s.viewer.displaySmoothing===false&&!s.viewer.drawer.context.imageSmoothingEnabled,'Native and enlarged detail do not receive bilinear blur by default');
  const ctx=s.viewer.drawer.canvas.getContext('2d');
  for(const [x,y] of [[1024,1024],[1025,1024],[1024,1025],[1035,1051]]){
    const p=layer().item.imageToViewerElementCoordinates(new OpenSeadragon.Point(x+.5,y+.5));
    const pixel=Array.from(ctx.getImageData(Math.floor(p.x*OpenSeadragon.pixelDensityRatio),Math.floor(p.y*OpenSeadragon.pixelDensityRatio),1,1).data).slice(0,3);
    assert(JSON.stringify(pixel)===JSON.stringify([(x+y)%256,(x%2)*255,(y%2)*255]),'Rendered native pixel '+x+','+y+' equals the source colour exactly');
  }
  let context=menu();
  assert(button(context,'100% (native)')&&button(context,'400%')&&!context.textContent.includes('magnification')&&!button(context,'100×'),'Only percentage zoom presets are shown');
  assert(!context.querySelector('[aria-label="Smooth enlarged pixels"]').checked,'The smoothing preference correctly starts unchecked');
  button(context,'200%').click();await until(()=>Math.abs(imageZoom()-2)<.001);await settled();
  assert(!s.viewer.displaySmoothing,'200% enlarges the source pixels without smoothing them');
  context=menu();context.querySelector('[aria-label="Smooth enlarged pixels"]').click();closeContextMenu();
  assert(s.viewer.displaySmoothing&&s.viewer.drawer.context.imageSmoothingEnabled,'Smooth enlarged pixels immediately restores optional interpolation');
  await flushViewerSettings();
  assert((await getJSON('/workspace/'+spec.workspace+'/view')).view.pixelSmoothing===true,'The interpolation preference is saved in project JSON');
  await init({...spec,id:'detail-reopened'});await until(()=>s.imagesReady);await settled();
  assert(s.pixelSmoothing===true&&s.viewer.displaySmoothing===true,'Reopening restores the chosen interpolation');
  context=menu();context.querySelector('[aria-label="Smooth enlarged pixels"]').click();closeContextMenu();
  await flushViewerSettings();
  assert((await getJSON('/workspace/'+spec.workspace+'/view')).view.pixelSmoothing===false,'Turning smoothing off is also persisted');
  // Run with real Firefox device scaling, rather than changing OSD's cached ratio.
  zoom(75);await settled();
  await until(nativeDrawn); // Cached overview tiles can still be fading out after getFullyLoaded().
  assert(nativeDrawn()&&s.viewer.displaySmoothing===(OpenSeadragon.pixelDensityRatio*.75<1),'Resolution and interpolation follow display density '+OpenSeadragon.pixelDensityRatio);
  document.querySelector('#btn-native').click();await until(()=>Math.abs(imageZoom()-1)<.001);await settled();
  assert(Math.abs(imageZoom()-1)<.001,'100% logical zoom is consistent on high-density displays');
  const requests=[];const originalFetch=window.fetch;
  window.fetch=(url,options)=>{requests.push(String(url));return originalFetch(url,options);};
  zoom(400);await settled();const exported=await fullCanvas(false);window.fetch=originalFetch;
  assert(exported.canvas.width>0&&requests.filter(u=>/\/image\/.*\/tile\//.test(u)).every(u=>u.includes('.png?')),'Full-resolution export requests lossless tiles too');
  assert(document.querySelectorAll('#drawtools [data-tool]').length===6&&document.querySelector('#btn-save-full'),'The existing drawing and image-export controls remain present');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""

if __name__ == '__main__':
    import sys
    main(DRIVER, timeout=240, setup=setup, pixel_ratio=2 if '--hidpi' in sys.argv else 1)
