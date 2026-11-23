"""Image sections, marker controls and independent Whole-image histogram in Firefox."""
from viewer_threshold_browser import main


def setup(root):
    from test_viewer import _write_channel_pages
    _write_channel_pages(root / "roi.ome.tiff", channels=12, height=400, width=400)


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1000;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const change=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
const button=(box,text)=>[...box.querySelectorAll('button')].find(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
window.addEventListener('unhandledrejection',e=>fetch('/result?error='+encodeURIComponent(String(e.reason))));
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.inputs);
 const image=s.layers.find(l=>l.type==='image'),annotation=s.layers.find(l=>l.type==='annotations');
 image.autoScope=JSON.stringify({annotation:annotation.token});buildPanel();
 const panel=()=>document.querySelector('.image-layer'),section=name=>panel().querySelector('[data-image-section="'+name+'"]');
 assert([...panel().querySelectorAll('[data-image-section]')].map(n=>n.dataset.imageSection).join(',')==='display,sets,auto,markers,histogram','Five related image sections have a consistent order');
 assert(panel().querySelectorAll('.image-section-head .help').length===5,'Each section has compact contextual help');
 const opacity=panel().querySelector('[aria-label="Image opacity"]'),percent=panel().querySelector('[aria-label="Image opacity percent"]');
 change(percent,'65');assert(image.opacity===.65&&opacity.value==='65'&&image.item.getOpacity()===.65,'Numeric opacity updates the slider and rendered image');
 opacity.value='100';opacity.dispatchEvent(new Event('input'));
 assert(percent.value==='100'&&image.opacity===1,'Opacity slider updates the percentage');
 change(percent,'150');assert(image.opacity===1&&percent.value==='100','Invalid opacity does not alter the image');
 const search=panel().querySelector('[aria-label="Find a marker"]');
 assert(section('markers').querySelector('.channel-columns').textContent.includes('LowHigh'),'Marker table labels its display-level columns');
 const before=image.channels.map(c=>c.visible);search.value='M1';search.dispatchEvent(new Event('input'));
 const all=section('markers').querySelector('.all-top input');all.click();
 await until(()=>image.channels.filter(c=>c.name.includes('M1')).every(c=>c.visible));
 assert(image.channels.every((c,i)=>c.name.includes('M1')||c.visible===before[i]),'Select matching markers preserves choices outside the search');
 search.value='';search.dispatchEvent(new Event('input'));
 button(section('sets'),'Save selected').click();panel().querySelector('[aria-label="Marker set name"]').value='Lymphocytes';button(section('sets'),'Save').click();
 await until(()=>panel().querySelector('[aria-label="Image marker set"]').value==='Lymphocytes');
 assert(section('sets').querySelector('.marker-set-count').textContent==='4/4 here','Saved marker set shows its availability beside the selector');
 assert(panel().querySelector('[aria-label="Image marker set"]').selectedOptions[0].textContent==='Lymphocytes','Marker set names remain unchanged and concise');
 const low=image.channels[0].low,high=image.channels[0].high;
 image.inputs[0].row.querySelector('.cname').click();
 const hist=()=>section('histogram'),scope=()=>hist().querySelector('[aria-label="Image histogram population"]');
 const pixels=()=>Number(hist().querySelector('[data-hist-stat=pixels] strong')?.textContent.replaceAll(',',''));
 await until(()=>pixels()===160000);
 assert(scope().value===''&&image.autoScope!=='','Histogram defaults to Whole image independently of Auto levels');
 assert(!hist().querySelector('details').open,'Histogram display options start folded');
 const canvas=hist().querySelector('canvas'),px=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
 assert(px.some((v,i)=>i%4!==3&&v>100),'Histogram paints actual pixel counts');
 assert(hist().querySelectorAll('[data-hist-stat]').length===4,'Histogram shows range, median, pixels and resolution without paragraphs');
 scope().dispatchEvent(new Event('pointerdown'));await until(()=>[...scope().options].some(o=>o.textContent.includes(': tissue')));
 const tissue=[...scope().options].find(o=>o.textContent.includes(': tissue')).value;
 change(scope(),tissue);await until(()=>pixels()>0&&pixels()<160000);
 const tissuePixels=pixels();assert(image.channels[0].low===low&&image.channels[0].high===high,'Choosing histogram cells changes neither display levels nor measurements');
 image.inputs[1].row.querySelector('.cname').click();await until(()=>pixels()===160000);
 assert(scope().value==='','A newly inspected marker also starts on Whole image');
 image.inputs[0].row.querySelector('.cname').click();await until(()=>pixels()===tissuePixels);
 assert(scope().value===tissue,'An explicit histogram area survives marker switching');
 const originalFetch=window.fetch;let count=0,release;
 window.fetch=async(url,options)=>{if(String(url).includes('/histogram?'))count++;return originalFetch(url,options);};
 change(hist().querySelector('[aria-label="Image histogram vertical scale"]'),'count');await delay(150);
 assert(count===0&&pixels()===tissuePixels,'Changing the histogram Y axis redraws without reading pixels again');
 window.fetch=async(url,options)=>{const response=await originalFetch(url,options);if(String(url).includes('/histogram?')&&!String(url).includes('annotation=')){await new Promise(r=>release=r);}return response;};
 change(scope(),'');await until(()=>release);change(scope(),tissue);release();await delay(250);
 assert(scope().value===tissue&&pixels()===tissuePixels,'A late Whole-image response cannot overwrite a newer area choice');
 window.fetch=originalFetch;
 change(scope(),'');await until(()=>pixels()===160000);
 change(hist().querySelector('[aria-label="Marker display name"]'),'T-cell marker');
 assert(image.inputs[0].row.querySelector('.cname').textContent==='T-cell marker (M0)'&&hist().querySelector('.histogram-marker').textContent==='T-cell marker','Marker renaming synchronizes the table and histogram title');
 const dock=document.querySelector('#panel');
 for(const width of [760,350]){
  dock.style.flex='0 0 '+width+'px';dock.style.width=width+'px';dock.style.maxWidth='none';dock.style.minWidth='0';await delay(100);
  assert(panel().scrollWidth<=panel().clientWidth+1,'Image controls fit a '+width+'px dock');
  const list=section('markers').querySelector('.channels'),head=list.querySelector('.channel-columns'),row=list.querySelector('.channel');
  assert(Math.abs(head.children[4].getBoundingClientRect().x-row.children[4].getBoundingClientRect().x)<2,'Marker headings align with values at '+width+'px');
 }
 dock.style.flex='0 0 600px';dock.style.width='600px';document.querySelector('#panel').scrollTop=0;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, timeout=120, setup=setup)
