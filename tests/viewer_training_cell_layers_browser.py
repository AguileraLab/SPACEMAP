"""Training cell grouping, editing, visibility and late-response regressions in Firefox.

Uses real tile save/read routes on disposable images; only model inference is
replaced with two deterministic cells so this UI check needs no GPU or weights.
"""
from viewer_threshold_browser import main


def setup(root):
    from spacemap_gui.studio import segmentation, training

    def square(x, y):
        return [[x, y], [x + 20, y], [x + 20, y + 20], [x, y + 20]]

    training.start(root, "browser_cells", "nuclei")
    training.save_example(
        root, root / "roi.ome.tiff", ["M0", "M1"], [0, 0, 256, 256],
        [square(25, 25), square(70, 25)], segmentation.values(root, {}),
    )

    def predict(image, names, bounds, settings):
        x, y, _, _ = bounds
        return {"bounds": bounds, "outlines": [square(x + 30, y + 30), square(x + 70, y + 30)],
                "count": 2, "seconds": 0.01, "device": "fixture"}

    segmentation.preview = predict


DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<600;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(parent,text)=>[...parent.querySelectorAll('button')].find(b=>b.textContent===text);
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 const tools=()=>document.querySelector('#studio-tools');
 const left=name=>tools().querySelector('[aria-label="'+name+'"]');
 const field=name=>document.querySelector('#panel [aria-label="'+name+'"]');
 const card=()=>document.querySelector('[data-training-cells]');
 const current=()=>s.layers.find(l=>l.cell_training==='current');
 const saved=()=>s.layers.find(l=>l.cell_training==='saved');
 await until(()=>left('Show the corrected cells in the tiles')&&card());
 const original=s.layers.find(l=>l.type==='cells');
 const originalState=JSON.stringify([original.table,original.visible,original.opacity,original.filters,original.colourBy]);
 const annotations=s.layers.filter(l=>l.type==='shapes'||l.type==='annotations');
 const annotationState=JSON.stringify(annotations.map(l=>[l.id,l.features,l.visible,l.hidden]));
 s.layerGroups.cells=true;buildPanel();
 assert(card().closest('[data-layer-group=cells]')&&!document.querySelector('[data-layer-group=annotations] [data-training-cells]'),
   'Training cells have their own panel under Cells and maps');
 assert(!document.querySelector('[data-cell-inspection]').open,'Cell inspection starts folded below the cell layers');
 const cellGroup=document.querySelector('[data-layer-group=cells]');
 assert([...cellGroup.children].indexOf(card())>[...cellGroup.children].findIndex(n=>n.matches('[data-cell-layer]')),
   'Original segmentation and Training cells are adjacent');
 // A late saved-cell response must not override a newer Hide action.
 const originalFetch=window.fetch;let outlineRequests=0,release,holdSaved=true;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);
   if(String(url).includes('/training?outlines=1')){outlineRequests++;if(holdSaved){holdSaved=false;await new Promise(r=>release=r);}}
   return response;};
 left('Show the corrected cells in the tiles').click();
 await until(()=>release);
 assert(field('Show saved cells').checked&&/Loading/.test(card().textContent),'Loading and visibility agree in both panels');
 field('Show saved cells').click();release();await delay(180);
 assert(!saved()&&!left('Show the corrected cells in the tiles').checked&&!document.querySelector('.training-cell-overlay'),
   'Hiding saved cells wins over a delayed response');
 field('Show saved cells').click();await until(()=>saved()?.cellOutlines.length===2);
 assert(!saved().editable&&!saved().features.length&&document.querySelectorAll('.training-cell-overlay').length===1,
   'Saved cells render as one read-only SVG path');
 const requestCount=outlineRequests;
 field('Show saved cells').click();field('Show saved cells').click();await until(()=>saved());
 assert(outlineRequests===requestCount,'Showing unchanged saved cells reuses the cached outlines');
 field('Show training tile outlines').click();
 assert(!left('Show the tiles on the image').checked&&!s.previews['training-tiles-train']&&saved(),
   'Tile outlines and saved cells have independent synchronized switches');
 left('Show the tiles on the image').click();
 assert(field('Show training tile outlines').checked&&s.previews['training-tiles-train'],
   'The left tile switch immediately updates Layers');
 const image=s.layers.find(l=>l.type==='image'), viewer=s.viewer;
 const at=(x,y)=>viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(x,y));
 const place=async()=>{left('Tile size').value='256';left('Tile size').dispatchEvent(new Event('change'));
   button(tools(),'Place a tile').click();await delay(50);viewer.raiseEvent('canvas-click',{quick:true,position:at(200,200)});};
 await place();await until(()=>current()?.features.length===2);
 const cells=current();
 assert(cells.editable&&s.drawInto===cells.id&&card().textContent.includes('Current tile'),
   'Place a tile adds editable Current tile cells to the same panel');
 field('Show training tile outlines').click();
 assert(!s.previews['training-tile']&&cells.visible&&saved().visible,'Tile outlines also hides the current tile box without hiding its cells');
 field('Show training tile outlines').click();
 assert(!document.querySelector('[data-layer-group=annotations]').textContent.includes('Cell 1')&&!card().textContent.includes('Export'),
   'Training cells are excluded from annotation names and export controls');
 document.querySelector('[data-tool=polygon]').click();
 [[180,180],[220,180],[220,220],[180,220]].forEach(([x,y])=>viewer.raiseEvent('canvas-click',{quick:true,position:at(x,y)}));
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>cells.features.length===3);
 assert(card().textContent.includes('3 cells'),'Drawing updates the current tile count');
 document.querySelector('[data-tool=freehand]').click();
 const bounds=viewer.element.getBoundingClientRect(), spot=at(200,200);
 (viewer.canvas||viewer.element).dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:bounds.left+spot.x,clientY:bounds.top+spot.y}));
 await until(()=>cells.features.length===2);
 document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}));
 await until(()=>cells.features.length===3);
 assert(true,'Right-click deletion and Undo still edit the current tile');
 document.querySelector('[data-tool=split]').click();
 viewer.raiseEvent('canvas-press',{position:at(200,170)});
 viewer.raiseEvent('canvas-drag',{position:at(200,230)});
 viewer.raiseEvent('canvas-release',{position:at(200,230)});
 await until(()=>cells.features.length===4);
 document.querySelector('[data-tool=merge]').click();
 viewer.raiseEvent('canvas-click',{quick:true,position:at(190,200)});
 viewer.raiseEvent('canvas-click',{quick:true,position:at(210,200)});
 await until(()=>cells.features.length===3);
 assert(true,'Split and Merge preserve cell correction behavior');
 document.querySelector('[data-tool=off]').click();
 field('Show current tile cells').click();
 assert(!cells.visible&&saved().visible&&original.visible&&field('Show training cells').indeterminate,
   'Current tile can be hidden independently of saved cells and original segmentation');
 field('Show current tile cells').click();
 field('Training cell appearance').value='both';field('Training cell appearance').dispatchEvent(new Event('change'));
 field('Training cell colour').value='#ff8800';field('Training cell colour').dispatchEvent(new Event('input'));
 field('Training cell opacity').value='60';field('Training cell opacity').dispatchEvent(new Event('input'));
 const svg=document.querySelector('.training-cell-overlay');
 assert(svg.style.stroke==='#ff8800'||svg.style.stroke==='rgb(255, 136, 0)','Training colour updates the actual SVG');
 assert(Number(svg.style.fillOpacity)===0.22&&cells.opacity===0.6&&saved().opacity===0.6,
   'Shared appearance and opacity apply to both training cell sources');
 // Check the rendered controls in a narrow dock as well as the standard width.
 const panel=document.querySelector('#panel'),previousWidth=panel.style.width;
 panel.style.width='340px';await delay(80);
 const cardRect=card().getBoundingClientRect();
 assert([...card().querySelectorAll('input,select')].every(n=>{const r=n.getBoundingClientRect();return r.width>0&&r.left>=cardRect.left&&r.right<=cardRect.right+1;}),
   'Training controls stay inside a narrow dock');
 const imports=document.querySelector('.cell-import'),importRect=imports.getBoundingClientRect();
 assert([...imports.querySelectorAll('select,input[type=text],button')].every(n=>{const r=n.getBoundingClientRect();return r.width>=90&&r.right<=importRect.right+1;}),
   'Cell import controls wrap without crushing the path field in a narrow dock');
 panel.style.width=previousWidth;
 field('Show training cells').click();
 assert(!cells.visible&&!saved()&&original.visible&&!left('Show the corrected cells in the tiles').checked,
   'Training master hides both sources without hiding original cells');
 field('Show training cells').click();await until(()=>saved());
 assert(cells.visible&&left('Show the corrected cells in the tiles').checked,'Training master restores both sources');
 button(tools(),'Keep tile').click();
 await until(()=>!current()&&s.trainingCells?.savedCount===5&&saved()?.cellOutlines.length===5);
 assert(!s.drawInto&&card().textContent.includes('5 cells'),'Keep tile saves corrections, removes Current tile and refreshes Saved cells');
 assert(outlineRequests===requestCount+1,'Saving invalidates the saved-cell cache exactly once');
 assert(JSON.stringify([original.table,original.visible,original.opacity,original.filters,original.colourBy])===originalState,
   'Training actions preserve the original segmentation display and Cell data');
 assert(JSON.stringify(annotations.map(l=>[l.id,l.features,l.visible,l.hidden]))===annotationState,
   'Training correction leaves annotation geometry and visibility unchanged');
 // Discard while inference is still in flight: no late geometry may return.
 let releasePrediction;
 window.fetch=async(url,args)=>{const response=await originalFetch(url,args);
   if(String(url).includes('/segment_preview'))await new Promise(r=>releasePrediction=r);return response;};
 await place();await until(()=>releasePrediction);
 button(tools(),'Discard').click();releasePrediction();await delay(180);
 assert(!current()&&!s.previews['training-tile']&&button(tools(),'Keep tile').disabled&&saved(),
   'Discard wins over late predictions and leaves saved cells intact');
 window.fetch=originalFetch;
 document.querySelector('[data-step=tab4]').click();
 await until(()=>!s.layers.some(l=>l.cell_training)&&!card());
 assert(s.drawInto!==cells.id,'Leaving Segmentation clears the training edit target');
 document.querySelector('[data-step=tab5]').click();
 await until(()=>saved()?.cellOutlines.length===5&&card());
 assert(field('Show saved cells').checked&&left('Show the corrected cells in the tiles').checked,
   'Returning restores saved-cell visibility consistently');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,outlineRequests})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+'\n'+document.querySelector('#studio-tools')?.textContent.slice(-650)));}},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=180, setup=setup)
