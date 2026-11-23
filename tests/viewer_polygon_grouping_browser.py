"""Lossless display grouping, viewport culling and compact polygon controls."""
from viewer_threshold_browser import main

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.error?.stack)));
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.studio);
 const square=(x,y,w)=>[[x,y],[x+w,y],[x+w,y+w],[x,y+w],[x,y]];
 const feature=(name,rings)=>({type:'Feature',properties:{classification:{name,color:name==='Stroma'?[0,200,0]:name==='Endothelia'?[255,0,0]:[0,100,255]}},geometry:{type:'Polygon',coordinates:rings}});
 const features=[feature('Tissue',[square(5,5,95).reverse(),square(20,20,40).reverse()]),feature('Tissue',[square(40,40,40)])];
 for(const name of ['Stroma','Endothelia'])for(let i=0;i<1000;i++){
   const x=115+(i%40)*6,y=10+Math.floor(i/40)*14+(name==='Endothelia'?5:0);
   const ring=Array.from({length:25},(_,j)=>[x+2*Math.cos(j*Math.PI/12),y+2*Math.sin(j*Math.PI/12)]);
   features.push(feature(name,[ring]));
 }
 await importAnnotationCopy({name:'review.geojson',features});
 const layer=s.layers.find(l=>l.annotation_group==='imported'&&l.source_name==='review');
 const original=JSON.stringify(layer.features),revision=layer.revision;
 s.layers.filter(l=>l.type==='shapes'&&l!==layer).forEach(l=>l.visible=false);
 const image=s.layers.find(l=>l.type==='image');s.layerGroups.annotations=true;
 s.annotationBlocksOpen[annotationFileKey(layer)]=true;renderShapes();buildPanel();
 const branch=()=>document.querySelector('[data-annotation-key="'+CSS.escape(annotationFileKey(layer))+'"]');
 const row=name=>branch().querySelector('[data-annotation-name="'+name+'"]');
 const button=(box,text)=>Array.from(box.querySelectorAll('button')).find(n=>n.textContent===text);
 const paths=()=>s.svgRoot.querySelector('[data-layer="'+layer.id+'"]');
 assert(paths().querySelectorAll('.annotation-group').length===3,'Two thousand polygons draw as three same-name groups');
 assert(new Set(Array.from(paths().querySelectorAll('.annotation-group'),p=>p.dataset.annotation)).size===3,'Tissue, Stroma and Endothelia remain separate');
 assert(!row('Stroma').querySelector('.region-controls'),'Grouped classes keep individual polygon controls folded');
 // Rasterize only the polygon overlay: reversed winding, a hole, and a same-class overlap.
 const svg='<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><style>.shape{fill-opacity:.12;stroke-width:.2;fill-rule:evenodd;}</style>'+paths().outerHTML+'</svg>';
 const img=new Image();img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg);await img.decode();
 const canvas=document.createElement('canvas');canvas.width=canvas.height=400;const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);
 const alpha=(x,y)=>ctx.getImageData(x,y,1,1).data[3];
 assert(alpha(30,30)===0&&alpha(50,50)>0&&alpha(10,10)>0,'Grouped display preserves holes and fills overlaps without cancellation');
 ctx.clearRect(0,0,400,400);drawShapesOnCanvas(ctx,(x,y)=>[x,y],.1);
 assert(alpha(30,30)===0&&alpha(50,50)>0&&Math.abs(alpha(70,70)-alpha(10,10))<=1,'Image export preserves grouped holes and same-class overlap fill');
 assert(!contains(layer.features[0],30,30),'Hit testing respects a polygon hole');
 row('Stroma').querySelector('.annotation-fold').click();
 assert(row('Stroma').querySelector('.region-controls')&&!button(row('Stroma'),'Group'),'The arrow exposes individual polygons without grouping buttons');
 assert(paths().querySelectorAll('.annotation-group').length===3&&paths().querySelectorAll('path').length===3,'Expanded controls keep the same efficient rendering');
 const selected=layer.features.find(f=>annotationName(layer,f)==='Stroma');
 const node=paths().querySelector('[data-annotation=Stroma]');
 renderShapes();assert(paths().querySelector('[data-annotation=Stroma]')===node&&node.isConnected,'Unchanged outlines reuse their SVG nodes');
 s.selected={layer,id:selected.properties.id};renderShapes();
 assert(paths().querySelector('.selected').getAttribute('d')===groupedPathOf(selected.geometry),'Selection shows the exact individual boundary');
 assert(node.getAttribute('d').includes(groupedPathOf(selected.geometry)),'Compound display keeps every coordinate');
 const area=featureArea(selected),bounds=featureBounds(selected),oldGeometry=selected.geometry;
 selected.geometry={type:'Polygon',coordinates:[square(140,140,7)]};renderShapes();
 assert(featureArea(selected)===49&&featureBounds(selected).x0===140&&node.getAttribute('d').includes(groupedPathOf(selected.geometry)),'Replacing geometry invalidates bounds, area and drawing caches');
 selected.geometry=oldGeometry;renderShapes();
 assert(featureArea(selected)===area&&featureBounds(selected)===bounds,'Restoring a geometry restores its exact cached measurements');
 const whole=s.viewer.viewport.getBounds(true);
 s.viewer.viewport.fitBounds(image.item.imageToViewportRectangle(125,125,30,30),true);placeOverlay();
 const view=shapeViewport(),cache=shapeRenderCache.get(layer);
 assert(paths().querySelectorAll('path').length<100,'Zooming draws only nearby polygons');
 ctx.clearRect(0,0,400,400);drawShapesOnCanvas(ctx,(x,y)=>[x,y],.1);
 assert(alpha(10,10)>0,'Full image export includes polygons outside the current viewport');
 const visible=layer.features.filter(f=>annotationName(layer,f)==='Stroma').filter(f=>{const b=featureBounds(f);return b.x1>=view.x0&&b.x0<=view.x1&&b.y1>=view.y0&&b.y0<=view.y1;});
 assert(visible.length>0&&visible.every(f=>paths().querySelector('[data-annotation=Stroma]')?.getAttribute('d').includes(groupedPathOf(f.geometry))),'Every polygon intersecting the view is retained');
 s.viewer.viewport.fitBounds(image.item.imageToViewportRectangle(280,250,30,30),true);placeOverlay();
 const next=shapeViewport();
 assert(layer.features.filter(f=>annotationName(layer,f)==='Stroma').filter(f=>{const b=featureBounds(f);return b.x1>=next.x0&&b.x0<=next.x1&&b.y1>=next.y0&&b.y0<=next.y1;}).every(f=>paths().querySelector('[data-annotation=Stroma]')?.getAttribute('d').includes(groupedPathOf(f.geometry))),'Panning brings the next polygons into view');
 s.viewer.viewport.fitBounds(whole,true);placeOverlay();
 row('Stroma').querySelector('.annotation-fold').click();
 assert(paths().querySelectorAll('.annotation-group').length===3&&!row('Stroma').querySelector('.region-controls'),'Folding the tools keeps the class batches');
 assert(JSON.stringify(layer.features)===original&&layer.revision===revision,'Opening and closing lists never rewrite geometry, labels or the saved revision');
 s.picks={layer,ids:new Set(layer.features.filter(f=>annotationName(layer,f)==='Stroma').map(f=>f.properties.id))};renderShapes();
 assert(paths().querySelectorAll('path').length===4&&paths().querySelectorAll('.selected').length===1,'Grouping a thousand selected polygons retains one selection overlay');
 s.picks=null;s.selected=null;renderShapes();
 row('Tissue').querySelector('.annotation-fold').click();
 const app=document.querySelector('#app'),body=()=>row('Tissue').querySelector('.region-controls');
 assert(button(row('Tissue').querySelector('.layer-head'),'Use as Tissue…')&&!button(body(),'Use as Tissue…'),'Use as Tissue shares the expanded annotation header');
 app.style.setProperty('--studio-right','1000px');await delay(100);
 const boxes=Array.from(body().querySelector('.region-tools').children).map(n=>n.getBoundingClientRect());
 assert(boxes.every(b=>Math.abs(b.top-boxes[0].top)<2),'Label and minimum-size controls share one line in a wide panel');
 const footer=body().querySelector('.region-storage'),items=Array.from(footer.querySelectorAll('button,summary,[data-shape-status]')).filter(n=>n.getClientRects().length&&!n.closest('details[open]'));
 assert(items.every(n=>Math.abs(n.getBoundingClientRect().top-items[0].getBoundingClientRect().top)<8),'Saved status, reload, file, download and history share one line');
 assert(row('Tissue').getBoundingClientRect().left>branch().querySelector('summary').getBoundingClientRect().left,'Annotation names are indented beneath the source');
 assert(body().querySelector('.shape-row').getBoundingClientRect().left>row('Tissue').querySelector('.layer-head').getBoundingClientRect().left,'Polygons are indented beneath their annotation');
 assert(body().querySelector('.shape-row').getBoundingClientRect().height<=23,'Polygon rows have reduced vertical spacing');
 app.style.setProperty('--studio-right','350px');await delay(100);
 assert(branch().scrollWidth<=branch().clientWidth+2,'Expanded controls wrap without horizontal overflow in a narrow dock');
 row('Tissue').querySelector('.annotation-fold').click();
 assert(!button(row('Tissue'),'Use as Tissue…'),'Use as Tissue hides when its annotation is folded');
 // Header grouping is reversible even after toggling class visibility.
 const show=row('Stroma').querySelector('[aria-label="Show Stroma"]');show.click();
 assert(!Array.from(paths().querySelectorAll('.annotation-group')).some(p=>p.dataset.annotation==='Stroma'),'Hidden classes are excluded from grouped rendering');
 ctx.clearRect(0,0,400,400);drawShapesOnCanvas(ctx,(x,y)=>[x,y],.1);
 assert(alpha(115,10)===0&&alpha(115,15)>0,'Image export respects hidden classes while retaining other annotations');
 assert(!regionAt({x:115,y:10})||annotationName(regionAt({x:115,y:10}).layer,regionAt({x:115,y:10}).feature)!=='Stroma','Hidden polygons cannot win hit testing');
 row('Stroma').querySelector('[aria-label="Show Stroma"]').click();
 app.style.setProperty('--studio-right','950px');row('Tissue').querySelector('.annotation-fold').click();
 s.layers.filter(l=>l.type==='image').forEach(l=>l.open=false);buildPanel();document.querySelector('#panel').scrollTop=0;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
'''

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180)
