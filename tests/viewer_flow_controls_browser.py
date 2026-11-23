"""Marker menus, nested trees, gate resizing/deletion and spatial context actions."""
from viewer_threshold_browser import main

DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating';
const info=async()=>await(await fetch(endpoint)).json();
const post=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;};
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');
 button('Dock below',document).click();await until(()=>app()?.querySelector('.flow-chart .main-svg'));
 const card=app().querySelector('.flow-plot-card'),chart=card.querySelector('.flow-chart'),x=card.querySelector('[aria-label="X measurement"]');
 let state=await info();x.focus();
 const choices=()=>card.querySelector('[aria-label="X measurement choices"]');
 check(choices().querySelectorAll('[role="option"]').length===state.numeric.length,'Opening a populated axis input shows every numeric measurement');
 x.value='M1';x.dispatchEvent(new Event('input'));
 check([...choices().children].every(o=>o.textContent.includes('M1')),'Typing filters measurements');
 x.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}));x.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 check(x.value==='M1'&&choices().hidden,'Keyboard selection accepts a different measurement');
 x.click();check(choices().children.length===state.numeric.length,'Reopening restores the full list after searching');x.blur();
 change(x,'M0');change(card.querySelector('[aria-label="Y measurement"]'),'M1');
 await until(()=>chart.layout.xaxis.title.text.includes('M0')&&chart.layout.yaxis.title.text.includes('M1'));
 const axes={x:{column:'M0',scale:'linear'},y:{column:'M1',scale:'linear'}};
 const parent={id:'parent',name:'Parent',colour:'#e15759',kind:'rectangle',parent:'root',...axes,bounds:[20,250,20,300]};
 const child={id:'child',name:'Child',colour:'#59a14f',kind:'range',parent:'parent',x:axes.x,bounds:[20,200]};
 await post({action:'save',revision:(await info()).workspace.revision,changes:{gates:[parent,child]}});
 await until(()=>app().querySelector('[data-population="child"]'));
 const rootEntry=app().querySelector('[data-population="root"] .flow-tree-entry'),childEntry=app().querySelector('[data-population="child"] .flow-tree-entry');
 check(parseInt(childEntry.style.paddingLeft)>parseInt(rootEntry.style.paddingLeft),'Nested populations indent the whole row entry');
 const toggle=app().querySelector('[aria-label="Collapse Parent"]');check(toggle.textContent==='−','Expanded parents show a minus box');toggle.click();
 check(!app().querySelector('[data-population="child"]')&&app().querySelector('[aria-label="Expand Parent"]').textContent==='+','Plus box collapses children');
 app().querySelector('[aria-label="Expand Parent"]').click();check(!!app().querySelector('[data-population="child"]'),'Plus box restores descendants');
 // Select the rectangle in plot coordinates, away from its child range.
 const point=()=>{const r=chart.getBoundingClientRect(),fl=chart._fullLayout;return {clientX:r.left+fl.xaxis._offset+fl.xaxis.d2p(220),clientY:r.top+fl.yaxis._offset+fl.yaxis.d2p(260)};};
 await until(()=>chart.layout.shapes.some(s=>s.name==='parent'));
 chart.dispatchEvent(new MouseEvent('click',{bubbles:true,...point()}));
 check(chart.querySelectorAll('.flow-gate-handle').length===4,'Selecting a rectangle exposes four resize handles');
 chart.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,...point()}));
 const menu=app().querySelector('.flow-menu');check(!!button('Show cells on image',menu)&&!!button('Delete gate',menu),'Gate right-click offers spatial viewing and deletion');
 button('Show cells on image',menu).click();await until(()=>window.__state.layers.some(l=>l.flowPopulation?.name==='Parent'));
 const layer=window.__state.layers.find(l=>l.flowPopulation?.name==='Parent');
 check(layer.flowMode==='filter'&&window.__state.layers.filter(l=>l.type==='cells').length===1,'Gate context action filters the existing segmentation layer');
 showDataTab();await dataChoose('combined');await until(()=>dataView.queryReady&&dataView.flowSelection);
 const oldRevision=(await info()).workspace.revision,handle=chart.querySelector('.flow-gate-handle');
 const r=chart.getBoundingClientRect(),fl=chart._fullLayout;
 handle.dispatchEvent(new PointerEvent('pointerdown',{pointerId:99,bubbles:true,clientX:r.left+fl.xaxis._offset+fl.xaxis.d2p(20),clientY:r.top+fl.yaxis._offset+fl.yaxis.d2p(20)}));
 handle.dispatchEvent(new PointerEvent('pointermove',{pointerId:99,bubbles:true,clientX:r.left+fl.xaxis._offset+fl.xaxis.d2p(120),clientY:r.top+fl.yaxis._offset+fl.yaxis.d2p(130)}));
 handle.dispatchEvent(new PointerEvent('pointerup',{pointerId:99,bubbles:true}));
 await until(async()=>(await info()).workspace.revision>oldRevision);
 state=await info();check(Math.abs(state.workspace.gates.find(g=>g.id==='parent').bounds[0]-120)<1,'Dragging a handle saves revised gate geometry');
 check(state.stats.find(s=>s.id==='parent').n<23,'Resize recomputes exact population counts');
 const revisedCount=state.stats.find(s=>s.id==='parent').n;
 await until(()=>layer.flowPopulation?.count===revisedCount&&dataView.result?.matched===revisedCount);
 check(true,'Resizing updates the displayed cells and Data rows');
 button('Undo').click();await until(async()=>(await info()).workspace.gates.find(g=>g.id==='parent').bounds[0]===20);
 check(true,'Undo restores gate geometry');
 // Delete selected child by keyboard; parents and unrelated data are retained.
 change(card.querySelector('[aria-label="Plot type"]'),'histogram');
 await until(()=>chart.data?.[0]?.type==='scatter');
 const childPoint=()=>{const r=chart.getBoundingClientRect(),fl=chart._fullLayout;return {clientX:r.left+fl.xaxis._offset+fl.xaxis.d2p(100),clientY:r.top+fl.yaxis._offset+fl.yaxis._length/2};};
 // Row action deletion exercises dependency-safe delete, then Undo.
 app().querySelector('[data-population="child"]').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:120,clientY:220}));
 button('Delete gate',app().querySelector('.flow-menu')).click();
 await until(()=>app().querySelector('dialog[open]'));check((await info()).workspace.gates.some(g=>g.id==='child'),'Delete prompts before changing any gate');
 button('Cancel',app().querySelector('dialog')).click();check((await info()).workspace.gates.some(g=>g.id==='child'),'Cancelling deletion keeps the gate');
 app().querySelector('[data-population="child"]').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:120,clientY:220}));button('Delete gate',app().querySelector('.flow-menu')).click();
 await until(()=>app().querySelector('dialog[open]'));
 button('Delete',app().querySelector('dialog')).click();await until(async()=>!(await info()).workspace.gates.some(g=>g.id==='child'));
 check((await info()).workspace.gates.some(g=>g.id==='parent'),'Deleting a child preserves its parent');
 button('Undo').click();await until(()=>app().querySelector('[data-population="child"]'));
 // Keyboard deletion targets the selected root rectangle and asks for dependants.
 change(card.querySelector('[aria-label="Plot type"]'),'scatter');await until(()=>chart.data?.[0]?.type==='scattergl');
 chart.dispatchEvent(new MouseEvent('click',{bubbles:true,...point()}));document.activeElement?.blur();
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}));
 await until(()=>app().querySelector('dialog[open]'));
 check(app().querySelector('dialog').textContent.includes('dependent'),'Delete key identifies dependent populations');button('Delete',app().querySelector('dialog')).click();
 await until(async()=>(await info()).workspace.gates.length===0);check(true,'Keyboard deletion removes the gate and its descendants');
 await until(()=>!layer.flowSelection&&!dataView.flowSelection&&dataView.result?.matched===40);
 check(true,'Deleting the displayed gate clears image and Data filters');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,screenshot=False,timeout=180,browser_host='flow-gating.test')
