"""Linked GeoJSON metadata remains usable while rendering stalls, with retryable failures."""
from viewer_annotation_sources_browser import setup
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async(f,ms=15000)=>{const end=performance.now()+ms;while(performance.now()<end){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.cellsReady&&s.studio);
 const specs=(await(await fetch('/spacemap-viewer/api/annotation-sources-fixture')).json()).layers;
 const previousGate=s.whenCellsReady;
 let release;s.cellsReady=false;s.whenCellsReady=new Promise(r=>{release=r;});
 const linked=await addLayerToView(specs[0]);
 s.layerGroups.annotations=true;s.annotationBlocksOpen[annotationFileKey(linked)]=true;buildPanel();
 const branch=layer=>[...document.querySelectorAll('[data-annotation-key]')].find(n=>n.dataset.annotationKey===annotationFileKey(layer));
 const classes=layer=>branch(layer)?.querySelectorAll('.annotation-linked-class').length;
 await until(()=>classes(linked)===3,3000);
 assert(!s.cellsReady&&!linked.item,'GeoJSON classes load before the stalled cell renderer completes');
 const list=branch(linked).querySelector('.annotation-classes');
 assert(!list.textContent.includes('Loading')&&list.getBoundingClientRect().height>0,'Class names are rendered in the open source panel');
 s.cellsReady=true;release();s.whenCellsReady=previousGate;
 await until(()=>linked.item?.getFullyLoaded());
 assert(linked.item.getOpacity()===linked.opacity,'Annotation polygons render after the cell gate releases');
 const oldHidden=[...linked.hidden];
 branch(linked).querySelector('[aria-label="Show Tumor"]').click();
 assert(linked.hidden.has('Tumor'),'Loaded class controls still change polygon visibility');
 linked.hidden=new Set(oldHidden);bump(linked);
 // Retry must recover without reopening the image or changing visibility.
 const realFetch=window.fetch.bind(window);let failures=1,requests=0;
 window.fetch=(url,options)=>{
  if(String(url).includes('/annotations/'+linked.token+'/legend')){
   requests++;if(failures-->0)return Promise.resolve(new Response(JSON.stringify({error:'Temporary test failure'}),{status:503,headers:{'Content-Type':'application/json'}}));
  }
  return realFetch(url,options);
 };
 linked.legendRequest=null;buildPanel();
 const retry=layer=>branch(layer)?.querySelector('.annotation-classes button');
 await until(()=>retry(linked));
 assert(branch(linked).textContent.includes('Temporary test failure'),'A failed legend shows its error rather than Loading');
 retry(linked).click();await until(()=>classes(linked)===3);
 assert(requests===2,'The source header and body share a request; Retry makes one new request');
 assert(linked.visible&&linked.hidden.size===oldHidden.length,'Retry preserves annotation visibility choices');
 window.fetch=realFetch;
 // A request which sends no response is bounded. Compress only this request's deadline.
 const second=await addLayerToView(specs[1]);await annotationLegend(second);
 s.annotationBlocksOpen[annotationFileKey(second)]=true;
 let aborted=false;
 window.fetch=(url,options)=>String(url).includes('/annotations/'+second.token+'/legend') ? new Promise((resolve,reject)=>{
  const abort=()=>{aborted=true;reject(new DOMException('Aborted','AbortError'));};
  if(options.signal.aborted)abort();else options.signal.addEventListener('abort',abort,{once:true});
 }) : realFetch(url,options);
 const realTimeout=window.setTimeout;window.setTimeout=(fn,ms,...args)=>realTimeout(fn,ms===60000?80:ms,...args);
 second.legendRequest=null;buildPanel();window.setTimeout=realTimeout;
 await until(()=>retry(second));
 assert(aborted&&branch(second).textContent.includes('took too long'),'A stalled request stops and exposes Retry');
 window.fetch=realFetch;retry(second).click();await until(()=>classes(second)===3);
 assert(true,'Retry after a timeout loads the real class list');
 // OpenSeadragon does not fetch tiles for a fully transparent initial layer.
 const cells=s.layers.find(l=>l.type==='cells'),item=cells.item,loaded=item.getFullyLoaded.bind(item),opacity=cells.opacity;
 cells.opacity=0;item.setOpacity(0);item.getFullyLoaded=()=>false;
 s.cellsReady=false;
 const transparent=openLayersInOrder(s.viewer);
 assert(await Promise.race([transparent.then(()=>true),delay(1500).then(()=>false)]),'A transparent cell layer cannot block later annotations');
 item.getFullyLoaded=loaded;cells.opacity=opacity;item.setOpacity(opacity);
 // A snapshot of a removed cell layer must not hold the old loading phase open.
 item.getFullyLoaded=()=>false;s.cellsReady=false;
 const removed=openLayersInOrder(s.viewer);await delay(80);removeLayer(cells);
 assert(await Promise.race([removed.then(()=>true),delay(1500).then(()=>false)]),'Removing a pending cell layer releases the annotation loading phase');
 assert(s.cellsReady,'The remaining viewer finishes opening');
 // Read the actual drawn canvas, not just a successful API response.
 await until(()=>linked.item?.getFullyLoaded()&&second.item?.getFullyLoaded());await delay(250);
 const canvas=document.querySelector('#osd .openseadragon-canvas canvas'),pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
 let visible=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i]+pixels[i+1]+pixels[i+2]>0)visible++;
 assert(visible>100,'The viewer contains rendered image/annotation pixels after recovery');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,visiblePixels:visible})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=100)
