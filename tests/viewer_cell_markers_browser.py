"""Top markers follows the clicked cell's current profile and transformations."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<900;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.cellsReady);
 const img=()=>s.layers.find(l=>l.type==='image'),cells=()=>s.layers.find(l=>l.id==='cells');
 s.layerGroups.cells=true;buildPanel();
 const options=()=>document.querySelector('.cell-marker-controls');
 const on=()=>options().querySelector('input[type=checkbox]'), count=()=>options().querySelector('input[type=number]');
 const shown=()=>img().channels.filter(c=>c.visible).map(c=>c.name).join(',');
 const click=(x=15,y=15)=>s.viewer.raiseEvent('canvas-click',{quick:true,position:s.viewer.viewport.pixelFromPoint(img().item.imageToViewportCoordinates(x,y)),originalEvent:{}});
 const number=n=>{count().value=n;count().dispatchEvent(new Event('input'));};
 const top=()=>rankedCellMarkers(s.inspectedCell?.result.profile || {}).filter(row=>img().channels.some(c=>c.name===row.marker)).map(row=>row.marker);
 const ready=()=>s.inspectedCell?.result.profile?.status==='ready';
 assert(options().parentElement.dataset.layerGroup==='cells'&&options().previousElementSibling.tagName==='SUMMARY','Compact controls sit directly under Cells and maps');
 assert(!on().checked&&count().value==='3'&&count().disabled,'Cell top markers starts off, independently of annotations');
 const initial=shown();click(55,15);await until(ready);
 assert(shown()===initial,'Clicking while off leaves visibility unchanged');
 const style=JSON.stringify(img().channels.map(c=>[c.color,c.low,c.high]));
 let reads=0;const originalFetch=window.fetch;
 window.fetch=(url,args)=>{if(String(url).endsWith('/profile'))reads++;return originalFetch(url,args);};
 on().click();number(1);await until(()=>shown()===top().slice(0,1).join(','));
 assert(reads===0,'Enabling the option reuses the current profile');
 assert(shown()==='M1','Raw marker ranking selects the largest saved measurement');
 const transform=name=>Array.from(document.querySelectorAll('#cellcard .marker-transforms button')).find(b=>b.textContent===name);
 for(const step of ['Arcsinh','Z1','Z2','CDF','NLog','Raw']){
   transform(step).click();await until(()=>ready()&&shown()===top().slice(0,1).join(','));
   assert(true,'Displayed channels match the cell feature ranking after '+step);
 }
 assert(JSON.stringify(img().channels.map(c=>[c.color,c.low,c.high]))===style,'Colour and contrast limits are preserved');
 count().focus();number(2);await until(()=>img().channels.filter(c=>c.visible).length===2);
 assert(document.activeElement===count(),'Changing Number preserves focus');
 number(0);const unchanged=shown();click(95,15);await until(()=>ready()&&s.inspectedCell.result.label===3);
 assert(shown()===unchanged&&!count().checkValidity(),'Invalid counts do not blank or alter the image');
 number(1);await until(()=>shown()===top()[0]);
 on().click();const off=shown();transform('Z1').click();await until(ready);click(135,15);await until(()=>ready()&&s.inspectedCell.result.label===4);
 assert(shown()===off,'Disabled option leaves channels unchanged on clicks and transformations');
 on().click();await until(()=>shown()===top()[0]);
 let waiting=false,release;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);if(String(url).endsWith('/profile')&&JSON.parse(args.body).label===1){waiting=true;await new Promise(r=>release=r);}return response;};
 click();await until(()=>waiting);click(55,15);await until(()=>ready()&&s.inspectedCell.result.label===2&&shown()===top()[0]);
 const newest=shown();release();await delay(200);
 assert(s.inspectedCell.result.label===2&&shown()===newest,'A slower earlier cell cannot overwrite the latest selection');
 waiting=false;click();await until(()=>waiting);on().click();const disabled=shown();release();await until(ready);await delay(150);
 assert(shown()===disabled,'Switching off while measurements load prevents late channel changes');
 window.fetch=originalFetch;on().click();await until(()=>shown()===top()[0]);
 const noData=shown();s.inspectedCell.result.profile={status:'ready',markers:[{marker:'not in image',mean:50}]};
 await applyCellMarkers(s.inspectedCell);
 assert(shown()===noData,'Missing marker matches never blank the image');
 assert(JSON.parse(localStorage.getItem('spacemap-viewer-cell-markers')).count===1,'Cell Number is remembered');
 buildPanel();assert(on().checked&&count().value==='1','Rebuilding the panel preserves the options');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, timeout=150)
