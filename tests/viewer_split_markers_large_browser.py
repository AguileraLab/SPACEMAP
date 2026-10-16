"""Fifty markers stay scrollable and only visible crops use network/decoding work."""
from viewer_threshold_browser import main


def setup(root):
    from test_viewer import _write_channel_pages
    _write_channel_pages(root / 'roi.ome.tiff', channels=50, width=800, height=800)


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.cellsReady);
 const image=s.layers.find(l=>l.type==='image'),pan=()=>document.querySelector('#split-marker-panel');
 const originalFetch=window.fetch;let active=0,peak=0,count=0,limits=0;
 window.fetch=async(url,args)=>{
  if(String(url).includes('/limits?'))limits++;
  if(String(url).includes('/image/')&&String(url).includes('/tile/')){count++;active++;peak=Math.max(peak,active);try{return await originalFetch(url,args);}finally{active--;}}
  return originalFetch(url,args);
 };
 assert(image.channels.length===50,'Synthetic image has 50 actual channels');
 Object.assign(splitMarkerOptions(),{open:true,all:true,size:160,height:260,zoom:1});
 splitMarkers.point={x:511,y:511};syncSplitMarkers();
 await until(()=>!splitMarkers.rendering&&[...pan().querySelectorAll('[data-rendered]')].length>0);
 assert(splitMarkers.cards.length===50,'All 50 markers have named cards');
 assert(count<50&&peak<=4,'Only visible cards load ('+count+' requests; '+peak+' concurrent)');
 assert(splitMarkers.grid.scrollHeight>splitMarkers.grid.clientHeight,'Many markers scroll inside the drawer');
 const firstCount=count,firstLimits=limits;
 splitMarkers.point={x:508,y:507};scheduleSplitMarkers();await delay(500);
 assert(count===firstCount&&limits===firstLimits,'Cursor updates reuse both pixel tiles and default limits');
 const pos=(x,y)=>s.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y));
 const visible=()=>splitMarkers.cards.filter(({card})=>{const r=card.getBoundingClientRect(),g=splitMarkers.grid.getBoundingClientRect();return r.bottom>g.top&&r.top<g.bottom;});
 const lags=[];
 for(let i=0;i<20;i++){
  const started=performance.now();followSplitMarkerCursor(pos(505+i%8,505+i%5));
  const center=splitMarkers.point.x+','+splitMarkers.point.y;
  for(let tries=0;tries<200&&!visible().every(c=>c.card.dataset.center===center);tries++)await delay(5);
  assert(visible().every(c=>c.card.dataset.center===center),'Cached frame '+i+' reaches every visible marker');
  lags.push(performance.now()-started);
 }
 const p95=lags.slice().sort((a,b)=>a-b)[18];
 assert(p95<90,'Cached pointer-to-tile latency stays below 90 ms (95th percentile '+p95.toFixed(1)+' ms)');
 const centers=new Set();
 for(let i=0;i<60;i++){
  followSplitMarkerCursor(pos(505+i%8,505+i%5));await delay(8);
  centers.add(visible().at(-1).card.dataset.center);
 }
 assert(centers.size>10,'Even the last visible marker updates throughout continuous motion ('+centers.size+' positions)');
 assert(count===firstCount,'Continuous warm movement needs no additional image downloads');
 splitMarkers.grid.scrollTop=splitMarkers.grid.scrollHeight;splitMarkers.grid.dispatchEvent(new Event('scroll'));
 await until(()=>splitMarkers.cards[49].card.dataset.rendered);
 assert(count>firstCount,'Scrolling loads markers at the end of the panel');
 // Cold movement drops obsolete queued/in-flight preview downloads; it must
 // not wait for a whole old marker panel before showing the current location.
 let aborted=0;
 const measuredFetch=window.fetch;
 window.fetch=async(url,args)=>{
  if(String(url).includes('/tile/')&&args?.signal){
   await new Promise((resolve,reject)=>{const timer=setTimeout(resolve,120);
    const stop=()=>{clearTimeout(timer);aborted++;reject(new DOMException('Moved','AbortError'));};
    if(args.signal.aborted)stop();else args.signal.addEventListener('abort',stop,{once:true});
   });
  }
  return measuredFetch(url,args);
 };
 image.channels.forEach(c=>c.color='#91abcd');followSplitMarkerCursor(pos(80,80));syncSplitMarkers();
 await until(()=>splitMarkers.active>0);
 followSplitMarkerCursor(pos(710,710));
 await until(()=>!splitMarkers.rendering&&visible().every(c=>c.card.dataset.center===splitMarkers.point.x+','+splitMarkers.point.y));
 assert(aborted>0,'Moving to another cold region cancels obsolete image downloads');
 assert(splitMarkers.queue.length===0&&splitMarkers.pending.size===0,'No obsolete preview work remains queued');
 assert(peak<=4,'Cold movement keeps the four-download concurrency limit');
 window.fetch=measuredFetch;
 const sharedStyle={...image.channels[0],color:'#674321',low:0,high:3000};
 const sharedUrl=markerCropTiles(image,sharedStyle,{x:50,y:50,width:64,z:0,scale:1})[0].url;
 splitMarkers.wanted=new Set([sharedUrl]);
 const preview=splitMarkerBitmap(sharedUrl,{preview:true}),evidence=splitMarkerBitmap(sharedUrl);
 stopSplitMarkerPreview();
 const [previewBitmap,evidenceBitmap]=await Promise.all([preview,evidence]);
 assert(previewBitmap===evidenceBitmap&&evidenceBitmap.width>0,'Cancelling the preview preserves a shared AI evidence download');
 // Compare a tile-boundary crop with its known source pixels, away from overlays.
 const canvas=document.createElement('canvas'),style={...image.channels[0],color:'#ffffff',low:0,high:4000};
 const geo={x:504,y:504,width:64,z:0,scale:1};
 await paintMarkerCrop(canvas,image,style,geo,{pixels:64});
 const p=canvas.getContext('2d').getImageData(7,2,3,1).data;
 const expected=x=>Math.round((((506*800+x)%4000)/4000)*255);
 assert([0,1,2].every(i=>Math.abs(p[i*4]-expected(511+i))<=1),'Crops cross a source tile boundary without black seams');
 window.fetch=originalFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,count,peak,cached_p95_ms:p95,continuous_positions:centers.size,aborted})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup, screenshot=False)
