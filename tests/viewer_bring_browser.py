"""Browser-check files brought in by hand in Studio: a GeoJSON dropped on the image becomes an editable annotation
layer; a same-size PNG dropped on it can be laid over the image as its own layer.

Usage: PYTHONPATH=src:tests python tests/viewer_bring_browser.py
"""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<600;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&s.studio&&document.querySelector('[data-step=tab5]'));
 const drop=files=>{const data=new DataTransfer();files.forEach(f=>data.items.add(f));
   document.querySelector('#osd').dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));};
 // A GeoJSON dropped on the image: an editable annotation layer.
 const square={type:'Feature',properties:{classification:{name:'Dropped'}},geometry:{type:'Polygon',coordinates:[[[10,10],[90,10],[90,90],[10,90],[10,10]]]}};
 drop([new File([JSON.stringify({type:'FeatureCollection',features:[square]})],'dropped.geojson',{type:'application/geo+json'})]);
 await until(()=>s.layers.some(l=>l.annotation_group==='imported'&&l.features?.length===1));
 assert(true,'A dropped GeoJSON becomes an editable annotation layer');
 // A PNG of the image's size: laid over it, as its own layer.
 const canvas=document.createElement('canvas');canvas.width=s.base.info.width;canvas.height=s.base.info.height;
 const context=canvas.getContext('2d');context.fillStyle='#fff';context.fillRect(50,50,100,100);
 const blob=await new Promise(r=>canvas.toBlob(r,'image/png'));
 drop([new File([blob],'mask.png',{type:'image/png'})]);
 await until(()=>document.querySelector('#bring-dialog'));
 assert(/the same size as the open image/.test(document.querySelector('#bring-dialog').textContent),'A same-size image offers to lie over the open one');
 Array.from(document.querySelectorAll('#bring-dialog button')).find(b=>b.textContent==='Lay over this image').click();
 await until(()=>s.layers.some(l=>l.overlay&&l.type==='image'));
 const overlay=s.layers.find(l=>l.overlay);
 assert(overlay.added&&document.querySelector('#panel [data-layer-id="'+overlay.id+'"]'),'The overlay is a layer with its own controls (and ×)');
 // A cell table dropped on the image: the Cell data dialog guesses x and y; Voronoi cells, coloured by a column.
 const rows=['x_centroid,y_centroid,score,kind'];for(let i=0;i<60;i++)rows.push((20+(i%10)*35)+','+(20+Math.floor(i/10)*60)+','+i+','+(i%2?'A':'B'));
 drop([new File([rows.join('\n')],'cells.csv',{type:'text/csv'})]);
 await until(()=>document.querySelector('.cell-data-dialog'));
 const dialog=document.querySelector('.cell-data-dialog');
 assert(dialog.querySelector('[aria-label="X column"]').value==='x_centroid'&&dialog.querySelector('[aria-label="Y column"]').value==='y_centroid','The table is recognised as cell data, its position columns guessed');
 const show=dialog.querySelector('[aria-label="Show as"]');show.value='voronoi';show.dispatchEvent(new Event('change'));
 Array.from(dialog.querySelectorAll('button')).find(b=>b.textContent==='Show on the image').click();
 await until(()=>s.layers.some(l=>l.data_layer)&&!document.querySelector('.cell-data-dialog'));
 const data=s.layers.find(l=>l.data_layer);
 assert(data.type==='cells'&&/Voronoi/.test(data.name)&&data.tables.some(t=>t.name==='Data: cells.csv'),'Voronoi cells are a cell layer with the table to colour by');
 const block=()=>document.querySelector('[data-cell-layer="'+data.id+'"]');
 await until(()=>block()&&block().querySelector('.cell-data-line'));
 assert(/60 of 60 rows · Voronoi cells/.test(block().textContent)&&Array.from(block().querySelectorAll('button')).some(b=>b.textContent==='Export GeoJSON'),'Its block says what it shows and offers the GeoJSON');
 const exported=await fetch(api('/studio/browser-check/cell_data?image_token='+encodeURIComponent(data.image_token)+'&export='+data.data_layer));
 const geo=await exported.json();
 assert(exported.ok&&geo.features.length===60&&geo.features.every(f=>f.properties.label),'The Voronoi cells export as GeoJSON, each with its number');
 await until(()=>Array.from(block().querySelector('[aria-label="Colour the cells by"]').options).some(o=>o.value==='score'));
 const add=block().querySelector('[aria-label="Colour the cells by"]');add.value='score';add.dispatchEvent(new Event('change'));
 await until(()=>s.layers.find(l=>l.data_layer)?.color_by==='score');
 assert(true,'Its cells colour by any column of the table');
 const firstId=s.layers.find(l=>l.data_layer).id;
 const reach=block().querySelector('[aria-label="Voronoi reach µm"]');reach.value='5';reach.dispatchEvent(new Event('change'));
 await until(()=>s.layers.some(l=>l.data_layer&&l.id!==firstId));
 const again=s.layers.find(l=>l.data_layer);
 assert(again.data_reach===5&&again.color_by==='score'&&s.layers.filter(l=>l.data_layer).length===1,'Changing the reach draws the same table again, keeping its colour');
 const block2=()=>document.querySelector('[data-cell-layer="'+again.id+'"]');
 await until(()=>block2());
 block2().querySelector(':scope > summary button[aria-label^="Remove"]').click();
 await until(()=>!s.layers.some(l=>l.data_layer));
 assert(true,'Its × takes it off the image');
 // Cell outlines dropped with their cell table: one cells layer, the table as its data.
 const outlines=[];const table=['label,x_centroid,y_centroid,score'];
 for(let i=0;i<60;i++){const x=20+(i%10)*35,y=20+Math.floor(i/10)*40;
   outlines.push({type:'Feature',properties:{label:i+1},geometry:{type:'Polygon',coordinates:[[[x,y],[x+20,y],[x+20,y+20],[x,y+20],[x,y]]]}});table.push((i+1)+','+(x+10)+','+(y+10)+','+i);}
 const before=s.layers.filter(l=>l.imported_cells).length;
 drop([new File([JSON.stringify({type:'FeatureCollection',features:outlines})],'run2.ome_cell_boundaries.geojson',{type:'application/geo+json'}),
   new File([table.join('\n')],'run2.ome_cell_features.csv',{type:'text/csv'})]);
 await until(()=>s.layers.filter(l=>l.imported_cells).length===before+1);
 const paired=s.layers.filter(l=>l.imported_cells).pop();
 assert(!document.querySelector('.cell-data-dialog')&&(paired.tables||[]).some(t=>t.name==="Its cells' table"),'Outlines dropped with their table make one cells layer, the table as its data: '+JSON.stringify(paired.tables));
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=90)
