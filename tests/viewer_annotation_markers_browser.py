"""Click-to-show marker inspection across annotations, including late-response guards."""
from viewer_threshold_browser import main


def setup(root):
    import json
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api, registry
    from test_studio_regions import _image

    image = _image(root / 'inspection.ome.tiff')
    token = registry.register(image)
    def square(name, x, y, size):
        return {'type':'Feature', 'properties':{'id':name, 'name':name}, 'geometry':{
            'type':'Polygon', 'coordinates':[[[x,y],[x+size,y],[x+size,y+size],[x,y+size],[x,y]]]}}
    features = [square('Upper',30,30,60), square('CD3 region',301,301,18)]
    layers = [{'id':'image','type':'image','token':token,'name':'Markers','visible':True,'channels':[
        {'index':0,'low':5,'high':1000,'visible':True},
        {'index':1,'low':7,'high':1100,'visible':False},
        {'index':2,'low':9,'high':1200,'visible':False}]}]
    for name, extra in [('artifact',{'artifact_part':'debris'}), ('good',{'training_part':'debris'}),
                        ('tissue',{'annotation_kind':'tissue'}), ('classes',{'annotation_group':'classes'}),
                        ('drawing',{'regions':False}), ('external',{'type':'annotations','regions':False})]:
        path = root / (name+'.geojson')
        path.write_text(json.dumps({'type':'FeatureCollection','features':features}))
        kind = extra.get('type','shapes')
        editable = name == 'drawing'
        registered = registry.register_object(kind, path=str(path), image=str(image), editable=editable)
        layers.append({'id':name,'type':kind,'token':registered,'image_token':token,'name':name,
                       'visible':False,'regions':True,'editable':editable,**extra})
    async def fixture(request):
        return JSONResponse({'id':'annotation-marker-inspection','layers':layers})
    api.app.router.routes.append(Route('/annotation-marker-fixture', fixture))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<600;i++){if(await f())return;await delay(40);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady);
 const spec=await (await fetch(api('/annotation-marker-fixture'))).json();
 await window.__init(spec);await until(()=>s.imagesReady);
 const img=()=>s.layers.find(l=>l.id==='image'),card=()=>document.querySelector('#cellcard');
 const options=()=>document.querySelector('.annotation-marker-controls');
 const on=()=>options().querySelector('[aria-label="Top markers"]');
 const count=()=>options().querySelector('[aria-label="Number of top markers"]');
 const shown=()=>img().channels.filter(c=>c.visible).map(c=>c.name).join(',');
 const number=n=>{count().value=n;count().dispatchEvent(new Event('input'));};
 const click=(x,y)=>s.viewer.raiseEvent('canvas-click',{quick:true,position:s.viewer.viewport.pixelFromPoint(img().item.imageToViewportCoordinates(x,y)),originalEvent:{}});
 const activate=id=>{s.layers.filter(l=>l.type!=='image').forEach(l=>setVisible(l,l.id===id));buildPanel();};
 s.layerGroups.annotations=true;buildPanel();
 assert(options().parentElement.dataset.layerGroup==='annotations'&&options().previousElementSibling.tagName==='SUMMARY','One compact Top markers / Number row sits directly under Annotations');
 assert(!on().checked&&count().value==='3'&&count().disabled,'Top markers defaults off, with Number 3');
 s.drawInto='drawing';activate('artifact');
 const toolbar=()=>document.querySelector('#drawtools');
 assert(!toolbar().hidden&&Array.from(toolbar().querySelectorAll('button')).every(b=>!b.hidden),'All seven drawing and selection tools remain visible when the assigned layer is hidden');
 toolbar().querySelector('[data-tool=rect]').click();
 const drawingLayer=s.layers.find(l=>l.id==='drawing');
 assert(drawingLayer.visible&&drawTarget()===drawingLayer&&s.tool==='rect','Choosing Rectangle reveals the assigned drawing layer without choosing another class');
 const at=(x,y)=>s.viewer.viewport.pixelFromPoint(img().item.imageToViewportCoordinates(x,y));
 s.viewer.raiseEvent('canvas-press',{position:at(220,220),originalEvent:{}});
 s.viewer.raiseEvent('canvas-drag',{position:at(240,240),originalEvent:{}});
 s.viewer.raiseEvent('canvas-release',{position:at(240,240),originalEvent:{}});
 await until(()=>drawingLayer.features.length===3&&!drawingLayer.savePending&&!drawingLayer.saveBusy);
 assert(true,'The restored toolbar creates a polygon in its assigned layer');
 undoShapes(drawingLayer);await until(()=>drawingLayer.features.length===2&&!drawingLayer.savePending&&!drawingLayer.saveBusy);
 setTool('off');
 const style=JSON.stringify(img().channels.map(c=>[c.color,c.low,c.high]));
 let reads=0;const originalFetch=window.fetch;
 window.fetch=(url,opts)=>{if(String(url).endsWith('/profile'))reads++;return originalFetch(url,opts);};
 activate('artifact');click(310,310);
 await until(()=>s.inspectedAnnotation?.rows&&card().textContent.includes('High here'));
 assert(shown()==='PanCK','Clicking an artifact while off leaves all channel checkboxes unchanged');
 const firstReads=reads;on().click();number(1);
 await until(()=>shown()==='CD3');
 assert(reads===firstReads,'Enabling Top markers reuses the clicked annotation ranking without another image read');
 assert(!toolbar().hidden,'Top markers preserves the drawing toolbar');
 const numberBox=count();numberBox.focus();number(2);await until(()=>img().channels.filter(c=>c.visible).length===2);
 assert(document.activeElement===numberBox,'Typing Number keeps focus and does not rebuild the layer panel');
 assert(JSON.stringify(img().channels.map(c=>[c.color,c.low,c.high]))===style,'Automatic visibility preserves existing channel colours and contrast limits');
 number(99);await until(()=>img().channels.every(c=>c.visible));
 assert(count().value==='99','Counts larger than the available panel show all available ranked markers');
 number(0);click(40,40);await until(()=>s.inspectedAnnotation?.feature.properties.name==='Upper'&&s.inspectedAnnotation.rows);
 assert(img().channels.every(c=>c.visible)&&!count().checkValidity(),'Invalid counts do not change channel visibility');
 number(1);await until(()=>img().channels.filter(c=>c.visible).length===1);
 for(const id of ['good','tissue','classes','drawing','external']){
  activate(id);click(310,310);
  await until(()=>s.inspectedAnnotation?.layer.id===id&&s.inspectedAnnotation.rows&&shown()==='CD3');
  assert(card().textContent.includes('High here'),'Top markers works for '+id+' annotations');
 }
 activate('artifact');s.annotationProfiles=new WeakMap();
 let release,waiting=false;
 window.fetch=async(url,opts)=>{const response=await originalFetch(url,opts);if(String(url).endsWith('/profile')&&JSON.parse(opts.body).geometry?.coordinates[0][0][0]===30){waiting=true;await new Promise(r=>release=r);}return response;};
 click(40,40);await until(()=>waiting);click(310,310);
 await until(()=>s.inspectedAnnotation?.rows&&shown()==='CD3');release();await delay(200);
 assert(shown()==='CD3'&&s.inspectedAnnotation.feature.properties.name==='CD3 region','A delayed earlier profile cannot replace the latest annotation or its channels');
 s.annotationProfiles=new WeakMap();waiting=false;click(40,40);await until(()=>waiting);on().click();release();await delay(200);
 assert(shown()==='CD3'&&!on().checked,'Turning Top markers off while a profile is pending prevents any channel switch');
 window.fetch=originalFetch;on().click();await until(()=>shown()!=='CD3');
 s.annotationProfiles=new WeakMap();img().channels.find(c=>c.name==='CD3').high=undefined;
 waiting=false;window.fetch=async(url,opts)=>{const response=await originalFetch(url,opts);if(String(url).includes('/limits?ch=2')){waiting=true;await new Promise(r=>release=r);}return response;};
 click(310,310);await until(()=>waiting);on().click();const before=shown();release();await delay(200);
 assert(shown()===before&&img().channels.find(c=>c.name==='CD3').high===undefined,'A late display-range response cannot change channels after the option is disabled');
 window.fetch=originalFetch;
 assert(JSON.parse(localStorage.getItem('spacemap-viewer-annotation-markers')).count===1,'The chosen count is remembered');
 on().click();await until(()=>shown()==='CD3');
 number(2);await until(()=>img().channels.filter(c=>c.visible).length===2);
 const styles=JSON.stringify(img().channels);on().click();activate('good');click(40,40);
 await until(()=>s.inspectedAnnotation?.layer.id==='good'&&s.inspectedAnnotation.rows);
 assert(JSON.stringify(img().channels)===styles,'Disabling the option keeps the current view and later annotation clicks do not touch channels');
 on().click();s.annotationProfiles=new WeakMap();waiting=false;
 window.fetch=async(url,opts)=>{const response=await originalFetch(url,opts);if(String(url).endsWith('/profile')){waiting=true;await new Promise(r=>release=r);}return response;};
 click(310,310);await until(()=>waiting);
 await window.__init({...spec,id:'another-view'});await until(()=>s.imagesReady);const nextImage=JSON.stringify(img().channels);
 release();await delay(200);
 assert(JSON.stringify(img().channels)===nextImage,'Changing images prevents a previous annotation response from changing the new image');
 assert(on().checked&&count().value==='2','Top markers and Number persist when opening another image');
 window.fetch=originalFetch;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#cellcard')?.textContent));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, timeout=150, setup=setup)
