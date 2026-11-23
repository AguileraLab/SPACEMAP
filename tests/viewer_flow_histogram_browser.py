"""Histogram paint order, fill opacity and matched facet scales in real Plotly."""
from viewer_threshold_browser import main

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
const app=()=>document.querySelector('.flow-app'),button=(text,root=app())=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating',info=async()=>await(await fetch(endpoint)).json();
const post=async body=>{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;};
const legendOrder=chart=>[...chart.querySelectorAll('.flow-legend-item')].map(r=>Number(r.dataset.series));
const paintOrder=chart=>chart.data.map(t=>t.meta.flowSeries);
const dragLegend=async(chart,from,to,before=true)=>{
 const rows=chart.querySelectorAll('.flow-legend-item'),row=[...rows].find(r=>+r.dataset.series===from),target=[...rows].find(r=>+r.dataset.series===to),a=row.getBoundingClientRect(),b=target.getBoundingClientRect();
 const x=a.left+a.width/2,y=a.top+a.height/2,targetX=b.left+b.width/2,targetY=before?b.top+1:b.bottom-1;
 check(!!document.elementFromPoint(x,y)?.closest('.flow-legend-item'),'Visible legend entry receives the drag');
 row.dispatchEvent(new PointerEvent('pointerdown',{pointerId:87,bubbles:true,button:0,clientX:x,clientY:y}));
 row.querySelector('.legendtoggle').dispatchEvent(new MouseEvent('mousedown',{bubbles:true,clientX:x,clientY:y}));
 document.dispatchEvent(new PointerEvent('pointermove',{pointerId:87,bubbles:true,clientX:targetX,clientY:targetY}));
 check(!!app().querySelector('.flow-legend-drop-line'),'Legend drag shows an insertion marker');
 document.dispatchEvent(new PointerEvent('pointerup',{pointerId:87,bubbles:true,clientX:targetX,clientY:targetY}));
 const released=document.elementFromPoint(targetX,targetY);released?.dispatchEvent(new MouseEvent('mouseup',{bubbles:true,clientX:targetX,clientY:targetY}));released?.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:targetX,clientY:targetY}));
 await delay(400);
};
const ranges=chart=>Object.keys(chart._fullLayout).filter(k=>/^yaxis\d*$/.test(k)).map(k=>chart._fullLayout[k].range);
const equalRanges=chart=>ranges(chart).every(r=>Math.abs(r[0]-ranges(chart)[0][0])<1e-9&&Math.abs(r[1]-ranges(chart)[0][1])<1e-9);
(async()=>{try{
 await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
 const spec={id:'histogram-style-test',parent:'root',x:{column:'M0',scale:'linear'},y:{column:'M1',scale:'linear'},kind:'histogram',bins:10,sample:100,frequency:'count',tool:'pan',
  series:[{population:'root',name:'All',colour:'#ff3300'},{population:'root',name:'Small',colour:'#33ff00',rule:{column:'M0',op:'<',value:12}},{population:'root',name:'Medium',colour:'#ff9900',rule:{column:'M0',op:'<',value:112}},{population:'root',name:'Empty',colour:'#33ddbb',rule:{column:'M0',op:'>',value:500}}],style:{font:13,width:900,height:700,fillOpacity:.35,legend:true}};
 await post({action:'save',revision:(await info()).workspace.revision,changes:{plots:[spec]}});
 window.open=()=>null;change(document.querySelector('[aria-label="Phenotyping method"]'),'flow');button('Dock below',document).click();
 await until(()=>app()?.querySelectorAll('.flow-legend-item').length===4);
 const card=app().querySelector('.flow-plot-card'),chart=card.querySelector('.flow-chart');
 let plotCalls=0;const rawFetch=window.fetch;window.fetch=(...args)=>{try{if(JSON.parse(args[1]?.body||'{}').action==='plot')plotCalls++;}catch(e){}return rawFetch(...args);};
 const divider=app().querySelector('[role="separator"][aria-label="Resize population panel"]'),sidebar=app().querySelector('.flow-sidebar'),initialWidth=sidebar.getBoundingClientRect().width,r=divider.getBoundingClientRect();
 check(document.elementFromPoint(r.left+r.width/2,r.top+r.height/2)===divider,'Population divider is reachable along the panel edge');
 divider.dispatchEvent(new PointerEvent('pointerdown',{pointerId:55,bubbles:true,button:0,clientX:r.left,clientY:r.top+80}));
 divider.dispatchEvent(new PointerEvent('pointermove',{pointerId:55,bubbles:true,clientX:r.left+75,clientY:r.top+80}));
 check(Math.abs(sidebar.getBoundingClientRect().width-initialWidth-75)<1,'Dragging the divider resizes the population panel immediately');
 divider.dispatchEvent(new PointerEvent('pointerup',{pointerId:55,bubbles:true}));
 check(Number(localStorage.getItem('spacemap-flow-sidebar-width'))===Math.round(sidebar.getBoundingClientRect().width),'Divider position is remembered without a server request');
 divider.dispatchEvent(new KeyboardEvent('keydown',{key:'End',bubbles:true}));check(app().querySelector('.flow-workspace').getBoundingClientRect().width>=380,'Divider limits leave room for the plots');
 divider.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));check(Math.round(sidebar.getBoundingClientRect().width)===290,'Double-click restores the default panel width');
 divider.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',shiftKey:true,bubbles:true}));const rememberedWidth=sidebar.getBoundingClientRect().width;
 check(rememberedWidth===330&&plotCalls===0,'Keyboard resizing changes layout without recalculating cells');
 app().querySelector('[aria-label="Show populations"]').click();check(divider.hidden&&sidebar.hidden,'Hiding populations also hides the divider');app().querySelector('[aria-label="Show populations"]').click();
 button('Expand',card).click();await delay(300);
 check(legendOrder(chart).join()=== '0,1,2,3'&&paintOrder(chart).join()==='3,2,1,0','Top legend entry draws in front of the other histograms');
 const originalValues=Object.fromEntries(chart.data.map(t=>[t.meta.flowSeries,JSON.stringify(t.y)]));
 await dragLegend(chart,3,0);await until(()=>legendOrder(chart).join()==='3,0,1,2');
 check(paintOrder(chart).join()==='2,1,0,3','Dragging the last legend to the top changes SVG paint order');
 check([...chart.querySelectorAll('.scatterlayer .trace')].map(el=>el.__data__[0].trace.meta.flowSeries).join()==='2,1,0,3','Rendered SVG layers follow the changed histogram order');
 check(chart._fullData.every(t=>t.visible!== 'legendonly'),'Dropping a dragged legend does not hide a curve');
 check(chart.data.every(t=>JSON.stringify(t.y)===originalValues[t.meta.flowSeries]),'Reordering retains every exact histogram bin');
 await until(async()=>(await info()).workspace.plots[0].style.seriesOrder?.join()==='3,0,1,2');
 check(plotCalls===0,'Legend reordering saves appearance without requesting cell analysis');
 await dragLegend(chart,3,2,false);await until(()=>legendOrder(chart).join()==='0,1,2,3');
 check(paintOrder(chart).at(-1)===0,'Dragging down restores the original front curve');
 const slider=card.querySelector('[aria-label="Histogram fill opacity"]');slider.value=0;slider.dispatchEvent(new Event('input'));slider.dispatchEvent(new Event('change'));
 await until(()=>chart.data.every(t=>t.fillcolor.endsWith(',0)')));
 check(chart.data.every(t=>t.opacity===1&&t.line.width>0),'Zero fill opacity keeps outlines visible');
 slider.value=100;slider.dispatchEvent(new Event('input'));slider.dispatchEvent(new Event('change'));await until(()=>chart.data.every(t=>t.fillcolor.endsWith(',1)')));
 check(chart.data.every(t=>t.fillcolor.endsWith(',1)')),'Full opacity removes colour blending');
 slider.value=60;slider.dispatchEvent(new Event('input'));slider.dispatchEvent(new Event('change'));await until(()=>chart.data.every(t=>t.fillcolor.endsWith(',0.6)')));
 await until(async()=>(await info()).workspace.plots[0].style.fillOpacity===.6);
 check(plotCalls===0,'Fill opacity updates locally and saves without recomputing cells');
 // A click still hides one curve, even when several series use the same population.
 const toggled=[...chart.querySelectorAll('.flow-legend-item')].find(r=>+r.dataset.series===1).querySelector('.legendtoggle');
 toggled.dispatchEvent(new MouseEvent('mousedown',{bubbles:true}));toggled.dispatchEvent(new MouseEvent('mouseup',{bubbles:true}));
 await until(()=>chart._fullData.find(t=>t.meta.flowSeries===1).visible==='legendonly');
 check(chart._fullData.filter(t=>t.visible==='legendonly').length===1,'Ordinary legend clicks hide only the chosen filtered series');
 const row=[...chart.querySelectorAll('.flow-legend-item')].find(r=>+r.dataset.series===2);row.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',altKey:true,bubbles:true}));
 await until(()=>legendOrder(chart).join()==='0,2,1,3');
 check(chart._fullData.find(t=>t.meta.flowSeries===1).visible==='legendonly','Reordering preserves hidden-curve state');
 await Plotly.restyle(chart,{visible:true});
 change(card.querySelector('[aria-label="Plot type"]'),'facets');await until(()=>ranges(chart).length===4);
 check(!equalRanges(chart),'Facets initially keep independent Y scales');
 const beforeMatch=plotCalls,match=card.querySelector('[aria-label="Match Y scales"]');match.click();await until(()=>equalRanges(chart));
 const peak=Math.max(...chart.data.flatMap(t=>t.y));check(ranges(chart).every(r=>r[0]===0&&r[1]>=peak&&r[1]<=peak*1.06),'Matched linear facets share the tallest curve range');
 check(chart.data.every((t,i)=>i===0||chart._fullLayout[t.yaxis==='y'?'yaxis':'yaxis'+t.yaxis.slice(1)].domain[1]<chart._fullLayout[i===1?'yaxis':'yaxis'+i].domain[1]),'Facet order follows the legend from top to bottom');
 await until(async()=>(await info()).workspace.plots[0].style.matchY===true);
 check(plotCalls===beforeMatch,'Matching facet scales does not request cell analysis');
 match.click();await until(()=>!equalRanges(chart));check(true,'Turning matched scales off restores independent ranges');match.click();await until(()=>equalRanges(chart));
 button('Settings',card).click();let d=app().querySelector('dialog');d.querySelector('[aria-label="Histogram Y"]').value='percent';button('Apply',d).click();
 await until(()=>!app().querySelector('dialog')&&chart.data.some(t=>Math.max(...t.y)>=50));
 check(equalRanges(chart)&&ranges(chart)[0][1]>=Math.max(...chart.data.flatMap(t=>t.y)),'Matched facets use percentage values when normalization changes');
 button('Settings',card).click();d=app().querySelector('dialog');const log=[...d.querySelectorAll('.flow-row')].find(r=>r.firstChild.textContent==='Log Y').querySelector('input');log.checked=true;button('Apply',d).click();
 await until(()=>!app().querySelector('dialog')&&chart._fullLayout.yaxis.type==='log');
 check(equalRanges(chart)&&ranges(chart).every(r=>r.every(Number.isFinite)),'Log facets share a finite range even with an empty population');
 // Frozen publication panels keep their own order, transparency and matched scale.
 card.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:500,clientY:200}));button('Freeze to layout',app().querySelector('.flow-menu')).click();
 await until(()=>app().querySelector('.flow-layout-panel .flow-legend-item'));const panel=app().querySelector('.flow-layout-panel'),snapshot=panel.querySelector('.flow-chart');
 check(equalRanges(snapshot)&&snapshot.data.every(t=>t.fillcolor.endsWith(',0.6)')),'Publication snapshot retains fill and matched scales');
 const snapshotRow=[...snapshot.querySelectorAll('.flow-legend-item')].find(r=>+r.dataset.series===3);snapshotRow.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',altKey:true,bubbles:true}));
 await until(async()=>(await info()).workspace.layouts[0].panels[0].snapshot.plot.style.seriesOrder?.join()==='0,2,3,1');
 check(legendOrder(chart).join()==='0,2,1,3','Editing publication legend order preserves the live plot order');
 button('Style',panel).click();d=app().querySelector('dialog');d.querySelector('[aria-label="Panel fill opacity"]').value=20;button('Apply',d).click();
 await until(()=>!app().querySelector('dialog')&&app().querySelector('.flow-layout-panel .flow-chart')?.data?.every(t=>t.fillcolor.endsWith(',0.2)')));
 const publication=app().querySelector('.flow-layout-panel .flow-chart');check(equalRanges(publication)&&chart.data.every(t=>t.fillcolor.endsWith(',0.6)')),'Panel opacity is independent and preserves matched scales');
 const svg=await Plotly.toImage(publication,{format:'svg',width:700,height:550});const svgText=svg.includes(';base64,')?atob(svg.split(',')[1]):decodeURIComponent(svg.split(',')[1]);
 check(svgText.includes('fill-opacity: 0.2')&&svgText.includes('All'),'SVG export retains fill opacity and legend');
 const exported=await post({action:'export',name:'histogram-style-check',format:'pdf',svg});check(exported.download.startsWith('data:application/pdf'),'Styled histogram facets export to PDF');
 button('Plots').click();button('Expand',card).click();
 const frame=document.createElement('iframe');frame.style='position:fixed;inset:0;width:1200px;height:900px;z-index:100000';frame.src='/spacemap-viewer/api/window/flow-gating.html?session=browser-check&api=/spacemap-viewer/api';document.body.append(frame);
 await until(()=>frame.contentDocument?.querySelectorAll('.flow-legend-item').length===4);const reopened=frame.contentDocument.querySelector('.flow-chart');
 check(legendOrder(reopened).join()==='0,2,1,3'&&equalRanges(reopened)&&reopened.data.every(t=>t.fillcolor.endsWith(',0.6)')),'Reopening Flow restores legend order, opacity and matched facet ranges');
 check(frame.contentDocument.querySelector('.flow-sidebar').getBoundingClientRect().width===rememberedWidth,'Reopening Flow restores the population divider width');frame.remove();
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+app()?.querySelector('.flow-status')?.textContent));}})();
</script>'''

if __name__ == '__main__':
    main(driver=DRIVER, studio=True, screenshot=False, timeout=240, browser_host='flow-gating.test')
