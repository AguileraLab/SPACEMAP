"""Exercise marker crops, display synchronization and cell AI evidence on disposable data."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>[...p.querySelectorAll('button')].find(b=>b.textContent===t);
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.cellsReady);
 const img=s.layers.find(l=>l.type==='image'),cells=s.layers.find(l=>l.type==='cells');
 const pos=(x,y)=>s.viewer.viewport.pixelFromPoint(img.item.imageToViewportCoordinates(x,y));
 const click=(x,y,ctrl=false)=>s.viewer.raiseEvent('canvas-click',{quick:true,position:pos(x,y),originalEvent:{ctrlKey:ctrl}});
 const panel=()=>document.querySelector('#split-marker-panel'),cards=()=>[...panel().querySelectorAll('.split-marker-tile')];
 const ready=()=>s.inspectedCell?.result.profile?.status==='ready';
 const rendered=()=>cards().length&&cards().every(c=>c.dataset.rendered);
 let requests=0,active=0,peak=0,chatBody;
 const originalFetch=window.fetch;
 window.fetch=async(url,args)=>{
   const where=String(url);
   if(where.includes('/chat')&&args?.method==='POST'){
     chatBody=JSON.parse(args.body);
     return new Response(JSON.stringify({done:{conversation:'test',say:'Tentative A based on measured M1.',action:null}})+'\n',{headers:{'Content-Type':'application/x-ndjson'}});
   }
   if(where.includes('/chat')&&(!args?.method||args.method==='GET'))return new Response(JSON.stringify({model:{enabled:true,images:true,provider:'local',model:'Test only'},messages:[],memories:[]}));
   if(where.includes('/image/')&&where.includes('/tile/')&&where.includes('.png')){requests++;active++;peak=Math.max(peak,active);try{return await originalFetch(url,args);}finally{active--;}}
   return originalFetch(url,args);
 };
 assert(!splitMarkerOptions().open&&document.querySelector('#split-marker-toggle'),'Marker drawer starts collapsed at the bottom');
 document.querySelector('#split-marker-toggle').click();followSplitMarkerCursor(pos(200,200));
 await until(rendered);
 await delay(250);const visibleCard=cards()[0].card||cards()[0],box=visibleCard.getBoundingClientRect();
 assert(panel().contains(document.elementFromPoint(box.left+20,box.top+30)),'Docked marker tiles are visible and clickable above the image');
 assert(cards().length===1&&cards()[0].dataset.marker==='M0','Selected mode shows only checked image markers');
 const first=cards()[0].querySelector('canvas'),pixel=first.getContext('2d').getImageData(20,20,1,1).data;
 assert(pixel[0]+pixel[1]+pixel[2]>0,'The marker crop contains rendered image pixels');
 const savedStyle=JSON.stringify(img.channels);
 panel().querySelector('[aria-label="Marker tile scope"]').value='all';panel().querySelector('[aria-label="Marker tile scope"]').dispatchEvent(new Event('change'));
 await until(()=>cards().length===2&&rendered());
 assert(JSON.stringify(img.channels)===savedStyle,'All markers adds tiles without changing main image channels or ranges');
 const before=requests;followSplitMarkerCursor(pos(202,202));await delay(500);
 assert(requests===before,'Nearby cursor movement reuses decoded tiles');
 img.channels[1].color='#ff0000';refreshLayer(img);await until(rendered);
 const red=cards()[1].querySelector('canvas').getContext('2d').getImageData(20,20,1,1).data;
 assert(red[0]>0&&red[1]===0&&red[2]===0,'Tiles use the marker colour from Layers');
 panel().dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:400,clientY:300}));
 await until(()=>document.querySelector('#ctxmenu.split-marker-menu'));
 button(document.querySelector('#ctxmenu'),'100% (native)').click();
 assert(splitMarkerOptions().zoom===1&&panel().querySelector('[aria-label="Marker tile zoom"]').value==='1','Right-click changes tile zoom independently');
 const resize=panel().querySelector('[aria-label="Resize marker panel"]'),height=splitMarkerOptions().height;
 resize.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));
 assert(splitMarkerOptions().height===height+24,'Panel height is resizable');
 button(panel(),'Float').click();assert(panel().parentElement===document.body&&panel().classList.contains('floating'),'Panel floats above Studio sidebars');
 button(panel(),'Dock').click();assert(panel().parentElement.id==='stage','Dock restores the bottom panel');
 button(panel(),'Hide ▾').click();const hiddenRequests=requests;followSplitMarkerCursor(pos(300,300));await delay(350);
 assert(requests===hiddenRequests&&!splitMarkerOptions().floating&&!splitMarkerOptions().open,'Hidden panel stops image requests and returns to its bottom handle');
 document.querySelector('#split-marker-toggle').click();
 panel().querySelector('[aria-label="Marker tile scope"]').value='selected';panel().querySelector('[aria-label="Marker tile scope"]').dispatchEvent(new Event('change'));
 s.layerGroups.cells=true;buildPanel();
 const on=document.querySelector('[aria-label="Cell top markers"]');on.click();
 const count=document.querySelector('[aria-label="Number of cell top markers"]');count.value=1;count.dispatchEvent(new Event('input'));
 click(55,15);await until(()=>ready()&&s.inspectedCell.result.label===2&&cards()[0]?.dataset.marker==='M1');
 assert(splitMarkers.pinned&&splitMarkers.selection===s.inspectedCell.result&&cards().length===1,'Cell Top markers updates split tiles and pins the selected cell');
 const pinned=JSON.stringify(splitMarkers.point);followSplitMarkerCursor(pos(350,350));
 assert(JSON.stringify(splitMarkers.point)===pinned,'Moving into the tile panel cannot lose the clicked cell');
 s.markersOpen=true;showCellCard(cells,s.inspectedCell.result);
 const transforms=document.querySelector('#cellcard .marker-transforms');button(transforms,'Z1').click();
 await until(()=>ready()&&cards()[0]?.dataset.marker===rankedCellMarkers(s.inspectedCell.result.profile)[0].marker);
 assert(true,'Transformed Top markers propagate through to marker tiles');
 click(95,15,true);await until(()=>ready()&&s.inspectedCell.result.members?.length===2);
 assert(splitMarkers.selection.members.length===2&&splitMarkers.span>0,'Ctrl-selected cells share a field of view');
 click(55,15);await until(()=>ready()&&s.inspectedCell.result.label===2);
 button(panel(),'Follow cursor').click();followSplitMarkerCursor(pos(310,300));showCellCard(cells,s.inspectedCell.result);
 assert(!splitMarkers.pinned&&Math.abs(splitMarkers.point.x-310)<1,'A profile redraw does not override Follow cursor');
 await until(()=>!document.querySelector('#cellcard .cell-ask-ai').disabled);
 document.querySelector('#cellcard .cell-ask-ai').click();await until(()=>chatBody);
 const ev=chatBody.view.cell_inspection;
 assert(ev.cell_ids.join()==='2'&&ev.measurements.length===2&&ev.transforms.includes('Z1'),'Ask AI sends the clicked cell measurements and selected transformation');
 assert(Math.abs(ev.crop.center.x-55)<1&&Math.abs(ev.crop.center.y-15)<1,'AI crops stay centred on the clicked cell even while the drawer follows another location');
 assert(chatBody.images.length===1&&chatBody.images[0].startsWith('data:image/png;base64,'),'Ask AI attaches labelled marker contact sheets');
 assert(ev.dataset_phenotypes.includes('A')&&ev.dataset_phenotypes.includes('B'),'AI context includes the dataset phenotype vocabulary');
 assert(ev.crop.markers[0].colour===img.channels.find(c=>c.visible).color,'AI tiles and metadata preserve the selected colour');
 await until(()=>document.querySelector('#studio-assistant').textContent.includes('Tentative A'));
 assert(!document.querySelector('#studio-assistant .studio-action'),'Cell AI suggestions do not create automatic label actions');
 button(panel(),'Follow cursor').click();followSplitMarkerCursor(pos(310,300));assert(!splitMarkers.pinned&&Math.abs(splitMarkers.point.x-310)<1,'Follow cursor releases a pinned cell');
 const saved=JSON.parse(localStorage.getItem('spacemap-viewer-view:'+s.spec.id));
 assert(saved.splitMarkers?.open&&saved.splitMarkers.zoom===1,'Marker panel display settings join the saved image view');
 click(55,15);await until(()=>ready()&&splitMarkers.pinned);
 followSplitMarkerCursor(pos(250,240));button(document.querySelector('#cellcard'),'Close').click();
 assert(!splitMarkers.pinned&&!s.inspectedCell&&Math.abs(splitMarkers.point.x-250)<1,'Closing the cell card immediately follows the latest pointer, without another click');
 click(55,15);await until(()=>ready()&&splitMarkers.pinned);
 click(95,15,true);await until(()=>ready()&&s.inspectedCell.result.members?.length===2);
 click(95,15,true);await until(()=>ready()&&s.inspectedCell.result.label===2);
 assert(splitMarkers.pinned,'Removing one of several cells keeps the remaining cell locked');
 click(55,15,true);await until(()=>!s.inspectedCell);
 assert(!splitMarkers.pinned&&!splitMarkers.selection,'Ctrl-deselecting the last cell releases the tiles');
 click(55,15);await until(()=>ready()&&splitMarkers.pinned);
 click(300,300);await until(()=>!s.inspectedCell&&!splitMarkers.pinned);
 followSplitMarkerCursor(pos(280,270));
 assert(Math.abs(splitMarkers.point.x-280)<1,'Clicking empty image space restores cursor following');
 const lock=panel().querySelector('[aria-label="Lock marker tiles on cell selection"]');lock.click();
 click(55,15);await until(ready);followSplitMarkerCursor(pos(260,250));
 assert(!splitMarkers.pinned&&!splitMarkerOptions().lockOnCell&&Math.abs(splitMarkers.point.x-260)<1,'Lock on cell can be disabled while keeping cell inspection available');
 assert(JSON.parse(localStorage.getItem('spacemap-split-markers')).lockOnCell===false,'The lock preference is saved');
 lock.click();
 assert(splitMarkers.pinned&&splitMarkers.selection===s.inspectedCell.result,'Enabling Lock on cell locks the current selection');
 window.fetch=originalFetch;
 await init({...s.spec,id:'another-test-image'});await until(()=>s.imagesReady);
 assert(!splitMarkers.pinned&&!splitMarkers.selection,'Changing images clears the previous cell target');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,peak})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#split-marker-panel')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, screenshot=False)
