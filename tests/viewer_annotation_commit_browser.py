"""Relabel exact polygons, preserve display, and explicitly write in Jobs."""
from viewer_review_browser import setup as review_setup
from viewer_threshold_browser import main

def setup(root):
    import json
    from collections import Counter
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api
    review_setup(root)
    async def original_counts(request):
        data=json.loads((root/'tissue.geojson').read_text())
        return JSONResponse(dict(Counter(f['properties']['classification']['name'] for f in data['features'])))
    api.app.router.routes.append(Route('/original-annotation-counts',original_counts))

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(f){fetch('/result?progress='+encodeURIComponent(String(f)));for(let i=0;i<600;i++){if(await f())return;await delay(75);}throw Error('Timed out '+f+' '+document.querySelector('#message')?.textContent);}
let checks=0;const check=(ok,text)=>{if(!ok)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
const btn=(root,text)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.studio&&s.imagesReady&&s.layers.some(l=>l.type==='annotations'));
 const original=s.layers.find(l=>l.type==='annotations'),image=s.layers.find(l=>l.type==='image'),item=image.item;
 setVisible(original,true);s.layerGroups.annotations=true;buildPanel();
 const source=()=>document.querySelector('[data-annotation-token="'+original.token+'"]');
 await until(()=>source());if(!source().open)source().querySelector('summary').click();
 await until(()=>source().querySelector('[data-component-class=tumor]'));
 original.hidden=new Set(['absent']);setAnnotationOpacity(.42);
 selectAnnotations(original,['polygon-0000','polygon-0001'],{replace:true});
 const p=screenOf({x:9,y:9}),r=s.viewer.element.getBoundingClientRect();
 s.viewer.element.querySelector('canvas').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:r.left+p.x,clientY:r.top+p.y}));
 await until(()=>btn(document.querySelector('#ctxmenu'),'Use “normal”'));
 btn(document.querySelector('#ctxmenu'),'Use “normal”').click();
 await until(()=>!s.annotationRenameBusy&&s.layers.some(l=>l.type==='annotations'&&l.token!==original.token));
 const copy=s.layers.find(l=>l.type==='annotations');
 await until(()=>copy.annotationLegend?.items.some(c=>c.value==='normal'));
 check(copy.name==='tissue.geojson','Editable copy displays the original filename without a hash');
 check(copy.hidden.has('absent')&&s.annotationOpacity===.42,'Relabelling preserves visibility choices and annotation opacity');
 check(image.item===item&&annotationSelection(copy).size===2,'Relabelling keeps the image and both selected polygon IDs');
 const rows=(await getJSON(componentQuery(copy,{}))).rows;
 check(rows.filter(v=>['polygon-0000','polygon-0001'].includes(v.id)).every(v=>v.name==='normal'),'Both selected polygons receive the chosen existing annotation');
 const normal=copy.annotationLegend.items.find(c=>c.value==='normal');
 check(normal.color.toLowerCase()==='#00ff00','Renamed polygons use their destination family colour');
 check(displayFamily(copy)===displayFamily(original),'Editable copies retain the original annotation display family');
 const old=await getJSON('/original-annotation-counts');
 check(old.tumor===305,'Source file stays unchanged while edits are pending');
 document.querySelector('[title="Background jobs"]').click();
 await until(()=>btn(document.querySelector('#studio-jobs'),'Write selected GeoJSON files'));
 const jobs=document.querySelector('#studio-jobs'),write=btn(jobs,'Write selected GeoJSON files');
 check(write.disabled&&jobs.textContent.includes('tissue.geojson'),'Jobs lists pending edits with the readable original name');
 jobs.querySelector('[aria-label="Write tissue.geojson"]').click();
 await until(()=>!btn(jobs,'Write selected GeoJSON files').disabled);
 btn(jobs,'Write selected GeoJSON files').click();
 await delay(1200);
 await fetch('/result?progress='+encodeURIComponent('Jobs after write: '+jobs.textContent.slice(0,900)));
 await until(async()=>!(await getJSON('/studio/browser-check/saved')).annotations.length);
 const updated=await getJSON('/original-annotation-counts');
 check(updated.normal===3&&updated.tumor===303,'Explicit Jobs write updates only the selected polygons in the original GeoJSON');
 await until(()=>!btn(document.querySelector('#studio-jobs'),'Write selected GeoJSON files'));
 check(jobs.textContent.includes('Write annotation files'),'Completed annotation write appears as a background job');
 await fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(error+'\n'+error.stack));}},100);
</script>'''

if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=240, screenshot=False)
