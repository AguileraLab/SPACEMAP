"""Check actual canvas pixels across tile junctions, opacity and checkerboard holes."""
from viewer_threshold_browser import main


def setup(root):
    import numpy as np
    import tifffile
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api
    from spacemap_gui.viewer.ui import using_context, image_layer

    specs = []
    for name, color in [('background', (240, 230, 220)), ('checker', (20, 80, 150))]:
        path = root / (name + '.ome.tiff')
        with tifffile.TiffWriter(path) as writer:
            for i, size in enumerate((2048, 1024, 512, 256)):
                pixels = np.full((size, size, 3), color, dtype='uint8')
                writer.write(pixels, photometric='rgb', tile=(256, 256),
                             **({'subifds': 3, 'metadata': {'axes': 'YXS'}} if i == 0 else {'subfiletype': 1}))
        with using_context({'root': str(root), 'mappings': {}}):
            specs.append(image_layer(str(path)))

    async def fixture(request):
        return JSONResponse({'id': 'tile-seams', 'layers': [specs[0]], 'checker': specs[1]})

    api.app.router.routes.append(Route('/tile-seams-fixture', fixture))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1400;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;
const assert=(ok,text)=>{if(!ok)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
  const s=window.__state;await until(()=>s.imagesReady);
  const emulateLegacy=false;
  if(emulateLegacy){const original=tileSource;tileSource=layer=>{const source=original(layer);delete source.hasTransparency;return source;};}
  const fixture=await getJSON('/tile-seams-fixture');
  await init({id:fixture.id,layers:fixture.layers});await until(()=>s.imagesReady);
  const layer=()=>s.layers.find(l=>l.type==='image'&&l.item);
  const settled=async()=>{
    await until(()=>s.layers.filter(l=>l.type==='image'&&l.visible).every(l=>l.item?.getFullyLoaded()&&
      l.item._lastDrawn.length&&l.item._lastDrawn.every(t=>t.level===l.info.max_zoom&&t.tile.opacity===1)));
    await delay(200);
  };
  const position=(percent,dx,dy)=>{
    const item=layer().item;
    s.viewer.viewport.zoomTo(item.imageToViewportZoom(percent/100),null,true);
    s.viewer.viewport.panTo(item.imageToViewportCoordinates(new OpenSeadragon.Point(1024+dx,1024+dy)),true);
    updateScaleBar();
  };
  const point=(x,y)=>layer().item.imageToViewerElementCoordinates(new OpenSeadragon.Point(x,y)).times(OpenSeadragon.pixelDensityRatio);
  const checkJunction=(alpha=255)=>{
    const p=point(1024,1024),ctx=s.viewer.drawer.context;
    const pixels=ctx.getImageData(Math.floor(p.x)-30,Math.floor(p.y)-30,60,60).data;
    let bad=0,minAlpha=255,maxAlpha=0,maxError=0;
    for(let i=0;i<pixels.length;i+=4){
      const error=Math.max(Math.abs(pixels[i]-240),Math.abs(pixels[i+1]-230),Math.abs(pixels[i+2]-220));
      maxError=Math.max(maxError,error);minAlpha=Math.min(minAlpha,pixels[i+3]);maxAlpha=Math.max(maxAlpha,pixels[i+3]);
      if(error>1||Math.abs(pixels[i+3]-alpha)>1)bad++;
    }
    return {bad,minAlpha,maxAlpha,maxError};
  };
  for(const percent of [83,100,137])for(const [dx,dy] of [[.25,.375],[.65,.8]]){
    position(percent,dx,dy);await settled();
    const pixels=checkJunction();
    assert(pixels.bad===0,'No seam at '+percent+'%, pan '+dx+','+dy+', density '+OpenSeadragon.pixelDensityRatio+': '+JSON.stringify(pixels));
  }
  layer().item.setOpacity(.6);position(83,.25,.375);await settled();
  const opacity=checkJunction(153);
  assert(opacity.bad===0,'Layer opacity stays uniform across tile joins: '+JSON.stringify(opacity));
  const checker={...fixture.checker,id:'checker-top',checker:64,visible:true,opacity:1};
  await init({id:'checker-seams',layers:[fixture.layers[0],checker]});await until(()=>s.imagesReady);
  position(83,.25,.375);await settled();
  for(const [x,y,want] of [[1008,1008,[20,80,150]],[1040,1008,[240,230,220]]]){
    const p=point(x,y),got=Array.from(s.viewer.drawer.context.getImageData(Math.floor(p.x),Math.floor(p.y),1,1).data);
    assert(got.slice(0,3).every((v,i)=>Math.abs(v-want[i])<=1)&&got[3]===255,'Checkerboard '+x+','+y+' retains its intended colour and transparency: '+got);
  }
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
"""


if __name__ == '__main__':
    import sys
    driver = DRIVER.replace('const emulateLegacy=false;', 'const emulateLegacy=true;') if '--legacy' in sys.argv else DRIVER
    main(driver, timeout=240, setup=setup, pixel_ratio=2 if '--hidpi' in sys.argv else 1)
