"""Real-browser comparison, raw-unit projection, apply/restore and navigation."""
from viewer_threshold_browser import main
from viewer_studio_resolve_browser import setup as resolve_setup

def setup(root):
    resolve_setup(root)
    import json
    from spacemap_gui.studio import background
    folder=root/"linked_annotations";folder.mkdir()
    geo=json.loads((root/"tissue.geojson").read_text())
    geo["features"][0]["properties"]["classification"]["name"]="epithelium"
    (folder/"roi.geojson").write_text(json.dumps(geo))
    background.save_step(root,{"folders":{"images":str(root),"annotations":str(folder)}})
DRIVER=r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const button=(box,text)=>Array.from(box.querySelectorAll('button')).find(b=>b.textContent===text);
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
setTimeout(async()=>{try{
 const tools=document.querySelector('#studio-tools');
 await until(()=>window.__state.imagesReady&&tools.querySelector('[aria-label="Phenotyping method"]'));
 change(tools.querySelector('[aria-label="Phenotyping method"]'),'resolve');
 await until(()=>button(tools,'Load Background'));
 const methodOptions=Array.from(tools.querySelector('[aria-label="RESOLVE positivity method"]').options).map(o=>o.value);
 assert(methodOptions.includes('Percentile')&&methodOptions.includes('Manual')&&methodOptions.includes('Geometric elbow'),'RESOLVE exposes the shared methods');
 button(tools,'Load Background').click();
 await until(()=>document.querySelector('#studio-background-detection.studio-attention'));
 assert(document.querySelector('[data-step=tab2]').getAttribute('aria-current')==='step','Load Background navigates and highlights detection controls');
 await until(()=>document.querySelector('.studio-threshold-benchmark [aria-label="Benchmark channels"]'));
 await until(()=>Array.from(document.querySelector('[aria-label="Benchmark annotation"]')?.options||[]).some(o=>o.textContent.includes(': epithelium')));
 const box=document.querySelector('.studio-threshold-benchmark'),image=window.__state.layers.find(l=>l.type==='image');
 const original=image.channels.map(c=>c.low);
 assert(box.querySelectorAll('.threshold-picker').length===2&&Array.from(box.querySelectorAll('.threshold-picker')).every(d=>!d.open),'Channel and method pickers start collapsed');
 assert(!box.querySelector('.threshold-options').open,'Plot settings start collapsed');
 assert(box.querySelector('[aria-label="Benchmark Manual cutoff"]').closest('.field').hidden,'Manual parameter is hidden until selected');
 const channelPicker=box.querySelector('[aria-label="Channels selector"]');channelPicker.open=true;
 const search=box.querySelector('[aria-label="Find benchmark channel"]');search.value='M1';search.dispatchEvent(new Event('input'));
 assert(Array.from(box.querySelectorAll('[aria-label="Benchmark channels"] label')).filter(l=>!l.hidden).length===1,'Channel search filters aligned marker rows');
 search.value='';search.dispatchEvent(new Event('input'));

 const methods=Array.from(document.querySelector('#studio-background [aria-label="Auto method"]').options).map(o=>o.value);
 assert(methods.every(m=>methodOptions.includes(m))&&methods.length===10,'Background and RESOLVE share the same ten-method catalog');
 const scope=box.querySelector('[aria-label="Benchmark annotation"]');
 await until(()=>Array.from(document.querySelector('[aria-label="Benchmark annotation"]').options).some(o=>o.textContent.includes(': epithelium')));
 assert(true,'Matched Annotation-directory classes load without opening an extra overlay');
 await until(()=>Array.from(scope.options).some(o=>o.textContent.includes(': tissue')));
 change(scope,Array.from(scope.options).find(o=>o.textContent.includes(': tissue')).value);
 for(const check of box.querySelectorAll('[aria-label="Benchmark methods"] input')){check.checked=['Compare Otsu','Compare Manual'].includes(check.getAttribute('aria-label'));check.dispatchEvent(new Event('change'));}
 change(box.querySelector('[aria-label="Benchmark Manual cutoff"]'),'100');
 assert(document.querySelector('#studio-background [aria-label="Manual cutoff"]').value==='100','Benchmark parameters stay synchronized with Background and Image controls');
 for(const check of box.querySelectorAll('[aria-label="Benchmark channels"] input')){check.checked=true;check.dispatchEvent(new Event('change'));}
 change(box.querySelector('[aria-label="Benchmark sampling resolution"]'),'native');
 button(box,'Compare thresholds').click();
 await until(()=>box.querySelectorAll('.threshold-chart').length===1&&!button(box,'Compare thresholds').disabled);
 assert(box.querySelector('[aria-label="View compared channel"]').options.length===2&&box.querySelectorAll('.threshold-cutoff').length===4,'Two channels share a plot selector and aligned whole-image/tissue cutoff table');
 assert(Array.from(box.querySelectorAll('.threshold-picker')).every(d=>!d.open),'Pickers close after comparison');
 assert(box.querySelector('svg').textContent.includes('Pixel count (log)'),'Default Y axis matches Image Layer log counts');
 change(box.querySelector('[aria-label="View compared channel"]'),String(image.channels[1].index));
 assert(box.querySelector('.threshold-chart strong').textContent===image.channels[1].name,'Compared channel selector switches the plot');
 change(box.querySelector('[aria-label="View compared channel"]'),String(image.channels[0].index));
 const gate=box.querySelector('.threshold-cutoff[data-method="Manual"]');gate.click();
 assert(image.channels[0].low===100&&image.thresholdPreview.threshold===100,'Clicking a legend entry applies the raw cutoff and preview');
 await until(()=>performance.getEntriesByType('resource').some(r=>r.name.includes('threshold_preview')));
 assert(true,'Viewport requests the threshold overlay tiles');
 change(box.querySelector('[aria-label="Threshold inspection display"]'),'mask');
 assert(image.thresholdPreview.mode==='mask','Binary mask inspection is available');
 change(box.querySelector('[aria-label="Benchmark display scale"]'),'Log10');
 await until(()=>box.querySelectorAll('.threshold-chart').length===1&&!button(box,'Compare thresholds').disabled&&box.querySelector('svg')?.textContent.includes('Log10'));
 assert(image.channels[0].low===100,'Switching display scale preserves raw applied levels');
 const hit=box.querySelector('svg [role="button"][aria-label*="Manual cutoff 100"]');
 assert(!!hit,'The manual cutoff remains clickable at its projected coordinate');hit.dispatchEvent(new MouseEvent('click',{bubbles:true}));
 button(box,'Restore previous levels').click();
 assert(image.channels.every((c,i)=>c.low===original[i])&&!image.thresholdPreview,'Restore removes the preview and restores previous levels');
 assert(box.scrollWidth<=box.clientWidth+1,'Benchmark controls and cutoff table fit the narrow Studio dock');
 image.inputs[0].row.querySelector('.cname').click();
 await until(()=>image.histogramHolder.querySelector('.hist canvas')&&image.histogramHolder.textContent.includes('full intensity range'));
 assert(image.histogramHolder.querySelector('[aria-label="Image histogram display scale"]').value==='Log10','Image Layer reuses the benchmark X axis');
 assert(image.histogramHolder.querySelector('[aria-label="Image histogram population"]').value===image.autoScope,'Image Layer starts with the same annotation population as Auto and the benchmark');
 change(image.histogramHolder.querySelector('[aria-label="Image histogram display scale"]'),'Linear');
 await until(()=>box.querySelector('svg')?.textContent.includes('Linear intensity (raw)')&&!button(box,'Compare thresholds').disabled);
 assert(box.querySelector('[aria-label="Benchmark display scale"]').value==='Linear','Changing Image Layer scale synchronizes the benchmark');
 const calls=performance.getEntriesByType('resource').filter(r=>r.name.includes('/threshold_benchmark')).length;
 change(box.querySelector('[aria-label="Benchmark vertical scale"]'),'fraction');await delay(300);
 assert(box.querySelector('svg').textContent.includes('Pixel fraction')&&performance.getEntriesByType('resource').filter(r=>r.name.includes('/threshold_benchmark')).length===calls,'Y axis changes redraw locally without recalculating thresholds');
 change(box.querySelector('[aria-label="Benchmark vertical scale"]'),'log');
 // Upper-only display adjustment must not write the lower thresholds.
 const panel=document.querySelector('.auto-controls:has([aria-label="Auto display adjustment"])');
 assert(!!panel,'Image Layer exposes display-range options separately from Background');
 const lows=image.channels.map(c=>c.low), highs=image.channels.map(c=>c.high);
 const cells=window.__state.layers.find(l=>l.live), gates=JSON.stringify(cells?.live?.markers.map(m=>m.threshold));
 button(panel,'Auto upper').click();await until(()=>!image.autoRun);
 assert(image.channels.every((c,i)=>c.low===lows[i])&&JSON.stringify(cells?.live?.markers.map(m=>m.threshold))===gates,'Auto upper preserves image lower levels and any linked cell gates');
 assert(image.channels.every((c,i)=>c.high!==highs[i]&&c.high>c.low),'Auto upper computes a valid upper display level for each marker');
 const displayPanel=()=>document.querySelector('.auto-controls:has([aria-label="Auto display adjustment"])');
 button(displayPanel(),'Undo Auto').click();
 assert(image.channels.every((c,i)=>c.low===lows[i]&&c.high===highs[i]),'Undo upper restores display levels');
 change(displayPanel().querySelector('[aria-label="Auto method"]'),'Manual');
 change(displayPanel().querySelector('[aria-label="Manual cutoff"]'),'100');
 button(displayPanel(),'Auto all markers').click();await until(()=>!image.autoRun);
 assert(image.channels.every(c=>c.low===100&&c.high>100),'Auto all markers sets the selected lower cutoff and automatic upper level');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''
if __name__=='__main__':main(DRIVER,studio=True,timeout=240,setup=setup)
