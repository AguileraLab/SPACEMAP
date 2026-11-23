"""Multi-selection, branch drag/drop, bulk deletion and forgiving gate handles."""
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from spacemap import dataio
    table=root/'cells.csv';frame=pd.read_csv(table);frame['response']=['CR','PR']*20
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)


DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating';
const info=async()=>await(await fetch(endpoint)).json(),post=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;};
const row=id=>app().querySelector('[data-population="'+id+'"]'),name=id=>row(id).querySelector('.flow-name');
const right=id=>row(id).dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:120,clientY:180}));
const mark=(id,keys={ctrlKey:true})=>name(id).dispatchEvent(new MouseEvent('click',{bubbles:true,...keys}));
const drag=async(id,target,keys={})=>{const revision=(await info()).workspace.revision,dataTransfer=new DataTransfer(),source=name(id);
 source.dispatchEvent(new DragEvent('dragstart',{bubbles:true,cancelable:true,dataTransfer,...keys}));
 row(target).dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer,...keys}));
 check(row(target).dataset.drop===(keys.shiftKey?'move':'copy'),'Drop target shows the chosen copy/move action');
 row(target).dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer,...keys}));source.dispatchEvent(new DragEvent('dragend',{bubbles:true,dataTransfer}));
 await until(async()=>(await info()).workspace.revision>revision);
};
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const filter=(id,parent,name,column,value,colour)=>({id,parent,name,colour,kind:'filter',rule:{column,op:'in',values:[value]}});
 const gates=[filter('a','root','A','phenotype_training','A','#ff8800'),filter('b','root','B','phenotype_training','B','#4433aa'),filter('cr','a','CR','response','CR','#123456'),filter('pr','a','PR','response','PR','#abcdef'),{id:'nested',parent:'cr',name:'High',colour:'#ab1234',kind:'range',x:{column:'M0',scale:'linear'},bounds:[60,400]}];
 // A saved plot can outlive a deleted population; restore must repair its view.
 await post({action:'save',revision:(await info()).workspace.revision,changes:{gates,plots:[{id:'old',parent:'removed',series:[{population:'removed'}],x:{column:'M0',scale:'linear'},y:{column:'M1',scale:'linear'},kind:'scatter',sample:100}]}});
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');await until(()=>button('Dock below',document));button('Dock below',document).click();
 await until(()=>app()?.querySelector('.flow-plot-foot')?.textContent.includes('Exact counts'));
 const card=app().querySelector('.flow-plot-card'),chart=card.querySelector('.flow-chart');
 check(card.querySelector('[aria-label="Plot parent"]').value==='root'&&chart.data.length===1,'Reopening a plot whose populations were deleted restores Analysis cells');
 mark('cr');mark('pr');check(row('cr').getAttribute('aria-selected')==='true'&&row('pr').getAttribute('aria-selected')==='true','Ctrl-click selects multiple population rows');
 right('cr');button('Duplicate 2',app().querySelector('.flow-menu')).click();await until(async()=>(await info()).workspace.gates.length===8);
 let state=await info();check(state.workspace.gates.some(g=>g.name==='CR copy')&&state.workspace.gates.some(g=>g.name==='PR copy'),'Duplicate copies selected branches, including descendants');
 button('Undo').click();await until(async()=>(await info()).workspace.gates.length===5);await until(()=>row('cr')&&!app().querySelector('[data-population="'+state.workspace.gates.at(-1).id+'"]'));
 mark('cr');mark('pr');await drag('cr','b',{ctrlKey:true});state=await info();
 const copied=state.workspace.gates.filter(g=>g.parent==='b'),cr=copied.find(g=>g.name==='CR'),pr=copied.find(g=>g.name==='PR'),nested=state.workspace.gates.find(g=>g.parent===cr.id);
 check(copied.length===2&&state.stats.find(s=>s.id===nested.id).n===10&&state.stats.find(s=>s.id==='nested').n===7,'Ctrl-drag reapplies copied gates to the destination cells and preserves originals');
 await until(()=>row(cr.id));name(cr.id).click();await until(()=>window.__state.layers.some(l=>l.flowPopulation?.name==='CR'));
 showDataTab();await dataChoose('combined');await until(()=>dataView.queryReady&&dataView.result?.matched===10);
 check(window.__state.layers.filter(l=>l.type==='cells').length===1,'Copied population selection filters image and Data through the existing cell layer');
 mark(pr.id);await drag(cr.id,'root',{shiftKey:true});state=await info();
 check([cr.id,pr.id].every(id=>state.workspace.gates.find(g=>g.id===id).parent==='root')&&state.workspace.gates.find(g=>g.id===nested.id).parent===cr.id,'Shift-drag moves selected branches with stable IDs and children');
 await until(()=>dataView.result?.matched===20);check(true,'Moving the displayed population refreshes its exact Data membership');
 // Delete all selected branches together, with confirmation and cancellation.
 right(cr.id);button('Delete selected (2)',app().querySelector('.flow-menu')).click();await until(()=>app().querySelector('dialog[open]'));
 check(app().querySelector('dialog').textContent.includes('1 dependent'),'Bulk delete identifies descendants before changing data');
 button('Cancel',app().querySelector('dialog')).click();check((await info()).workspace.gates.length===8,'Cancelling bulk deletion keeps every selected population');
 document.activeElement?.blur();document.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}));await until(()=>app().querySelector('dialog[open]'));
 button('Delete',app().querySelector('dialog')).click();await until(async()=>(await info()).workspace.gates.length===5);
 await until(()=>!dataView.flowSelection&&dataView.result?.matched===40);check(true,'Delete key removes the selected branches and clears the removed image/Data selection');
 button('Undo').click();await until(async()=>(await info()).workspace.gates.length===8);check(true,'One Undo restores the complete bulk deletion');
 // Shift-click selects a visible range. Moving a parent into its child is blocked.
 name('cr').click();mark('pr',{shiftKey:true});check(row('nested').getAttribute('aria-selected')==='true','Shift-click selects the visible row range');
 name('a').click();const dt=new DataTransfer(),rev=(await info()).workspace.revision;
 name('a').dispatchEvent(new DragEvent('dragstart',{bubbles:true,cancelable:true,dataTransfer:dt}));row('cr').dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:dt,shiftKey:true}));
 check(!row('cr').dataset.drop,'Moving a branch onto its own descendant is visibly blocked');name('a').dispatchEvent(new DragEvent('dragend',{bubbles:true,dataTransfer:dt}));
 // A selected rectangle offers large corners, whole-edge targets and a move handle.
 const rectangle={id:'rectangle',parent:'root',name:'Box',colour:'#e15759',kind:'rectangle',x:{column:'M0',scale:'linear'},y:{column:'M1',scale:'linear'},bounds:[20,250,20,300]};
 await post({action:'save',revision:(await info()).workspace.revision,changes:{gates:[...(await info()).workspace.gates,rectangle]}});
 await until(()=>row('rectangle'));name('root').click();await until(()=>chart.layout.shapes.some(s=>s.name==='rectangle'));
 button('Expand',card).click();await delay(200);
 const xy=(x,y)=>{const r=chart.getBoundingClientRect(),fl=chart._fullLayout;return {clientX:r.left+fl.xaxis._offset+fl.xaxis.d2p(x),clientY:r.top+fl.yaxis._offset+fl.yaxis.d2p(y)};};
 chart.dispatchEvent(new MouseEvent('click',{bubbles:true,...xy(220,260)}));
 const corner=chart.querySelector('.flow-gate-handle'),edge=chart.querySelector('.flow-gate-edge'),move=chart.querySelector('.flow-gate-move');
 check(corner.getBoundingClientRect().width===26&&edge.getBoundingClientRect().height>26&&!!move,'Large corner targets, full edges and a central move handle are available');
 const h=edge.getBoundingClientRect(),hit=document.elementFromPoint(h.left+4,h.top+h.height*.3);
 check(hit===edge,'The gate edge is draggable away from its midpoint');
 const revision=(await info()).workspace.revision;
 edge.dispatchEvent(new PointerEvent('pointerdown',{pointerId:9,bubbles:true,...xy(20,100)}));edge.dispatchEvent(new PointerEvent('pointermove',{pointerId:9,bubbles:true,...xy(120,100)}));
 check(chart.querySelector('.flow-gate-preview path').getAttribute('d').includes('M'),'Dragging shows an immediate outline preview');
 check((await info()).workspace.revision===revision,'Preview dragging does not repeatedly recalculate gate membership');
 edge.dispatchEvent(new PointerEvent('pointerup',{pointerId:9,bubbles:true}));await until(async()=>(await info()).workspace.revision>revision);
 check(Math.abs((await info()).workspace.gates.find(g=>g.id==='rectangle').bounds[0]-120)<1,'Releasing an edge saves the resized gate once');
 await until(()=>chart.querySelector('.flow-gate-move')!==move);
 const current=chart.querySelector('.flow-gate-move'),before=(await info()).workspace.revision;
 current.dispatchEvent(new PointerEvent('pointerdown',{pointerId:10,bubbles:true,...xy(185,160)}));current.dispatchEvent(new PointerEvent('pointermove',{pointerId:10,bubbles:true,...xy(205,180)}));current.dispatchEvent(new PointerEvent('pointerup',{pointerId:10,bubbles:true}));
 await until(async()=>(await info()).workspace.revision>before);let bounds=(await info()).workspace.gates.find(g=>g.id==='rectangle').bounds;
 check(bounds.every((v,i)=>Math.abs(v-[140,270,40,320][i])<1),'Dragging the move handle translates the complete gate without resizing it');
 const lastRevision=(await info()).workspace.revision;await until(()=>chart.querySelector('.flow-gate-move')!==current);const cancel=chart.querySelector('.flow-gate-move');
 cancel.dispatchEvent(new PointerEvent('pointerdown',{pointerId:11,bubbles:true,...xy(205,180)}));cancel.dispatchEvent(new PointerEvent('pointermove',{pointerId:11,bubbles:true,...xy(215,190)}));document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
 check((await info()).workspace.revision===lastRevision&&!chart.querySelector('.flow-gate-preview path').getAttribute('d'),'Escape cancels the geometry preview without saving');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}})();
</script>'''

if __name__=='__main__':
    main(driver=DRIVER,studio=True,setup=setup,screenshot=False,timeout=240,browser_host='flow-gating.test')
