"""Annotation source identity, compact headers and independent controls in Firefox."""
import json

from viewer_threshold_browser import main


def setup(root):
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api, registry

    layers = []
    for folder in ['original', 'reviewed']:
        path = root / folder / 'review.geojson'
        path.parent.mkdir()
        features = [dict(type='Feature', properties={'classification': {'name': name, 'color': [70, 170, 120]}},
                         geometry={'type': 'Polygon', 'coordinates': [[[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]]})
                    for name in ['Tissue', 'Tumor', 'Stroma']]
        path.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
        layers.append(api._annotation_layer(str(path), str(root / 'roi.ome.tiff'),
                                           registry.register(root / 'roi.ome.tiff'), folder + ' / review.geojson'))

    async def fixture(request):
        return JSONResponse({'layers': layers})

    api.app.router.routes.append(Route('/annotation-sources-fixture', fixture))


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<900;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.confirm=()=>true;
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.studio);
 const features=Array.from({length:10},(_,i)=>({type:'Feature',properties:{classification:{name:'Class '+i,color:[40,180,90]}},
   geometry:{type:'Polygon',coordinates:[[[10+i*20,10],[25+i*20,10],[25+i*20,30],[10+i*20,30],[10+i*20,10]]]}}));
 await importAnnotationCopy({name:'review.geojson',features});
 await importAnnotationCopy({name:'review.geojson',features:features.slice(0,3)});
 const copies=s.layers.filter(l=>l.annotation_group==='imported'&&l.source_name==='review');
 assert(copies.length===2,'Fixture has two editable copies with the same display name');
 const linked=(await(await fetch('/spacemap-viewer/api/annotation-sources-fixture')).json()).layers;
 for(const layer of linked)await addLayerToView(layer);
 s.layerGroups.annotations=true;buildPanel();
 const group=()=>document.querySelector('[data-layer-group=annotations]');
 const branch=layer=>Array.from(group().querySelectorAll('[data-annotation-key]')).find(n=>n.dataset.annotationKey===annotationFileKey(layer));
 const summary=layer=>branch(layer).querySelector('summary');
 const master=layer=>summary(layer).querySelector('input[type=checkbox]');
 const count=layer=>summary(layer).querySelector('.group-count').textContent;
 const copy=copies.find(l=>l.features.length===10),other=copies.find(l=>l!==copy);
 assert(branch(copy)!==branch(other)&&count(copy)==='10'&&count(other)==='3','Same-name files keep separate counts and headers');
 assert(summary(copy).querySelector('.name').textContent!==summary(other).querySelector('.name').textContent,'Duplicate copies have distinct stable labels');
 assert(summary(copy).querySelector('.help').dataset.tip.includes(copy.saved_file),'Source help includes the exact file path');
 const before=JSON.stringify(copies.map(l=>l.features));
 branch(copy).open=true;branch(other).open=true;await delay(40);
 master(copy).click();
 assert(!copy.visible&&other.visible,'Hiding one same-name copy leaves the other visible');
 assert(branch(copy).open,'Checkbox does not collapse its source');
 master(copy).click();
 const child=(layer,name)=>branch(layer).querySelector('[aria-label="Show '+name+'"]');
 child(copy,'Class 1').click();
 assert(master(copy).indeterminate&&child(other,'Class 1').checked,'Class visibility and mixed state belong to one source');
 const search=()=>branch(copy).querySelector('input[type=search]');
 search().value='Class 1';search().dispatchEvent(new Event('input'));
 assert(branch(copy).querySelectorAll('.annotation-name:not([hidden])').length===1,'Class search narrows a large list');
 branch(copy).querySelector('[aria-label="Show matching classes"]').click();
 assert(child(copy,'Class 1').checked&&child(copy,'Class 2').checked,'Select matching changes only matching classes');
 branch(copy).querySelector('[aria-label="Show matching classes"]').click();
 assert(!child(copy,'Class 1').checked&&child(copy,'Class 2').checked,'Hide matching preserves classes outside the search');
 assert(search().value==='Class 1','Search survives a panel rebuild');
 search().value='';search().dispatchEvent(new Event('input'));
 const labelBefore=summary(copy).querySelector('.name').textContent;
 s.layers.reverse();buildPanel();
 assert(summary(copy).querySelector('.name').textContent===labelBefore,'Copy identity stays stable when drawing order changes');
 for(const spec of linked){
   const layer=s.layers.find(l=>l.token===spec.token);
   branch(layer).open=true;
 }
 await until(()=>linked.every(spec=>branch(s.layers.find(l=>l.token===spec.token))?.querySelectorAll('.annotation-linked-class').length===3));
 const file=s.layers.find(l=>l.token===linked[0].token),second=s.layers.find(l=>l.token===linked[1].token);
 assert(!branch(file).querySelector('.layer-head')&&branch(file).querySelectorAll('summary').length===1,'Linked file has a single header with no nested duplicate');
 assert(summary(file).querySelector('.name').textContent!==summary(second).querySelector('.name').textContent,'Linked files with the same basename retain distinct locations');
 assert(summary(file).querySelector('input').nextElementSibling.className==='name','Visibility is next to the disclosure and source label');
 child(file,'Tumor').click();
 await until(()=>master(file).indeterminate);
 assert(second.visible&&!second.hidden.size,'A linked class toggle leaves the other file intact');
 master(file).click();
 await until(()=>master(file).checked);
 assert(!file.hidden.size,'Linked source master restores its classes');
 const opacity=branch(file).querySelector('[aria-label="Annotation opacity"]');
 opacity.value=35;opacity.dispatchEvent(new Event('input'));
 assert(file.opacity===0.35,'Opacity still controls the linked file');
 assert(['Edit a copy','Use as Tissue…'].every(text=>Array.from(branch(file).querySelectorAll('button')).some(b=>b.textContent===text)),'File actions remain available without repeating the header');
 const controls=()=>group().querySelector('.annotation-marker-controls');
 assert(controls().parentElement===group().firstElementChild,'Top markers shares the Annotations header');
 const top=controls().querySelector('[aria-label="Top markers"]');top.click();
 assert(group().open&&top.checked,'Top markers checkbox does not fold Annotations');
 const number=controls().querySelector('input[type=number]');number.value=5;number.dispatchEvent(new Event('input'));
 assert(annotationMarkerOptions().count===5&&group().open,'Top marker count remains functional in the header');
 group().firstElementChild.click();
 assert(!controls().getClientRects().length,'Top markers disappears when Annotations is folded');
 group().firstElementChild.click();
 assert(!!controls().getClientRects().length&&controls().querySelector('input[type=number]').value==='5','Expanding reveals the retained marker choice');
 assert(JSON.stringify(copies.map(l=>l.features))===before,'Visibility and layout never modify polygon data');
 const app=document.querySelector('#app'),dock=document.querySelector('#studio-right');
 const resize=width=>{app.style.setProperty('--studio-right',width+'px');window.dispatchEvent(new Event('resize'));};
 for(const width of [760,350]){
   resize(width);await delay(100);
   const row=branch(copy).querySelector('.annotation-name .layer-head');
   const swatch=row.querySelector('input[type=color]');
   assert(row.getBoundingClientRect().height<=28,'Class row stays compact at '+width+'px');
   assert(swatch.getBoundingClientRect().width>=26&&swatch.getBoundingClientRect().height>=18,'Colour swatch uses the checkbox height at '+width+'px');
   assert(branch(file).scrollWidth<=branch(file).clientWidth+2&&group().scrollWidth<=group().clientWidth+2,'Annotation controls fit the dock at '+width+'px');
   assert(Math.abs(dock.getBoundingClientRect().width-width)<2&&dock.getBoundingClientRect().right<=innerWidth+1,'Dock stays inside the viewport at '+width+'px');
 }
 const removed=other.saved_file;
 summary(other).querySelector('[aria-label^="Remove "]').click();
 await until(()=>!s.layers.includes(other));
 assert(s.layers.includes(copy)&&copy.features.length===10,'Removing one imported copy preserves the other same-name file');
 assert(!Array.from(group().querySelectorAll('[data-annotation-key]')).some(n=>n.dataset.annotationKey===removed),'Removed copy leaves only its own source block');
 // Keep the compact panel visible for a rendered screenshot.
 resize(600);
 s.layers.filter(l=>l.type==='image').forEach(l=>l.open=false);buildPanel();
 document.querySelector('#panel').scrollTop=0;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=180)
