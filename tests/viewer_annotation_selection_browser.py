"""Selection continuity and flagging mixed selections outside the Review tab."""
from viewer_review_browser import setup as review_setup
from viewer_threshold_browser import main


def setup(root):
    import time
    from spacemap_gui.studio import annotation_components
    review_setup(root)
    render = annotation_components.selection_tile
    def slow_tile(*args, **kwargs):
        time.sleep(.25)
        return render(*args, **kwargs)
    annotation_components.selection_tile = slow_tile


DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<400;i++){if(await f())return;await delay(50);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(scope,text)=>[...scope.querySelectorAll('button')].find(b=>b.textContent===text);
setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.studio&&state.imagesReady&&state.layers.some(l=>l.type==='annotations'));
 const source=state.layers.find(l=>l.type==='annotations'),image=state.layers.find(l=>l.type==='image'),route=reviewRoute();
 setVisible(source,true);state.layerGroups.annotations=true;buildPanel();
 await until(()=>source.item?.getFullyLoaded());
 const original=image.item;
 await pickAnnotation({x:9,y:9},false);
 const svg=source.annotationOutlineSvg,outline=svg.firstChild;
 check(svg.isConnected&&svg.children.length===2,'Clicked polygon has exact black/white outline immediately after hit testing');
 check(!source.annotationSelectionOverlay&&!source.annotationOutlinePending,'A single click needs no selection tiles');
 const first=outline.getAttribute('d');
 await pickAnnotation({x:24,y:9},true);
 check(svg===source.annotationOutlineSvg&&outline===svg.firstChild&&outline.getAttribute('d').includes(first),'Adding a polygon preserves existing rendered outline nodes');
 for(let i=0;i<15;i++){renderShapes();await delay(20);check(svg.isConnected&&outline.getAttribute('d'),'Selected outlines never disappear during redraws');}
 const known=await getJSON(componentQuery(source,{id:'polygon-0002'}));rememberAnnotationOutline(source,{...known.feature,componentRevision:known.revision});
 const start=performance.now();selectAnnotations(source,['polygon-0002'],{toggle:true});
 const ms=performance.now()-start;
 check(source.annotationOutlineSvg.firstChild.getAttribute('d').includes(pathOf(known.feature.geometry)),'Cached boundary selection is synchronous');
 await fetch('/result?progress='+encodeURIComponent('Cached exact selection + linked UI: '+ms.toFixed(1)+' ms'));
 const ids=Array.from({length:90},(_,i)=>'polygon-'+String(i).padStart(4,'0'));
 selectAnnotations(source,ids,{replace:true});
 await until(()=>source.annotationSelectionOverlay?.item?.getFullyLoaded());
 const old=source.annotationSelectionOverlay.item;
 selectAnnotations(source,['polygon-0100']);
 await until(()=>source.annotationOutlinePending?.item);
 check(source.annotationSelectionOverlay.item===old&&state.viewer.world.getIndexOfItem(old)>=0,'Old tiled outline stays rendered while delayed replacement tiles load');
 check(source.annotationOutlinePending.item.getOpacity()===0,'Incomplete replacement is preloaded invisibly');
 await until(()=>source.annotationSelectionOverlay.item!==old);
 check(source.annotationSelectionOverlay.item.getFullyLoaded()&&source.annotationSelectionOverlay.item.getOpacity()===1,'Replacement only becomes visible with complete pixels');
 selectAnnotations(source,['polygon-0101']);clearAnnotationSelection();await delay(800);
 check(!source.annotationSelectionOverlay&&!source.annotationOutlineSvg&&!source.annotationOutlinePending,'Clearing selection cancels pending borders without resurrecting old choices');
 await pickAnnotation({x:9,y:9},false);
 const flag=await flagSelectedAnnotations(),firstItem=flag.items[0];
 let answered=await postJSON(route,{action:'decide',id:firstItem.id,version:firstItem.version,decision:'keep'});
 await postJSON(route,{action:'apply',id:answered.id,version:answered.version});
 await pickAnnotation({x:24,y:9},false);await flagSelectedAnnotations();
 state.reviewPolygonLoad=null;ensureImageReviewPolygons();
 await until(()=>state.layers.find(l=>l.review_inline)?.features.some(f=>f.properties.review_state==='applied'));
 const preview=state.layers.find(l=>l.review_inline);preview.visible=true;
 selectAnnotations(source,['polygon-0000','polygon-0001','polygon-0002','polygon-0003'],{replace:true});
 const point=screenOf({x:9,y:9}),bounds=state.viewer.element.getBoundingClientRect();
 state.viewer.element.querySelector('canvas').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:bounds.left+point.x,clientY:bounds.top+point.y}));
 await until(()=>document.querySelector('#ctxmenu .annotation-context-actions'));
 await until(()=>button(document.querySelector('#ctxmenu'),'Flag 2 new polygons for review'));
 const menu=document.querySelector('#ctxmenu');
 check(!document.querySelector('.review-list'),'Flag menu works outside Review');
 check(!button(menu,'Flag 2 new polygons for review').disabled,'A completed review item does not block flagging new polygons');
 check(!button(menu,'Keep current label').disabled&&menu.textContent.includes('1 completed or locked item'),'Mixed review menu acts on editable items and identifies locked items');
 button(menu,'Flag 2 new polygons for review').click();
 await until(()=>state.lastFlaggedReview?.length===2);
 const queue=await getJSON(route);
 check(queue.total===4&&queue.items.filter(i=>i.state==='applied').length===1,'Only new exact source polygons are flagged; completed decision is preserved');
 await until(()=>state.layers.find(l=>l.review_inline)?.features.length===4);
 state.annotationBlocksOpen['section::Review polygons']=true;buildPanel();
 const row=document.querySelector('[data-review-polygon]');
 check(row.querySelector('.review-polygon-name')&&!button(row,'Go'),'Review polygon name replaces the Go button');
 clearAnnotationSelection();
 check([...document.querySelectorAll('[data-review-feature] input[type=checkbox]')].every(t=>!t.checked),'Clearing viewer selection updates Review polygon checkboxes outside the Review tab');
 await pickAnnotation({x:9,y:9},false);
 check([...document.querySelectorAll('[data-review-feature] input[type=checkbox]')].some(t=>t.checked),'Viewer selection checks Review polygon rows outside Review');
 row.querySelector('.review-polygon-name').click();
 check(annotationSelection(preview).has(row.dataset.reviewFeature)&&preview.visible,'Clicking the name selects and reveals the polygon');
 check(source.annotationOutlineSvg?.isConnected,'Clicking a review name immediately outlines its linked source');
 check(image.item===original,'Selection and flagging never reload the image');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,selection_ms:ms})));
}catch(e){fetch('/result?error='+encodeURIComponent(e.message+' '+e.stack));}},100);
</script>'''


if __name__ == '__main__':
    main(DRIVER, studio=True, setup=setup, timeout=150,
         capture='/tmp/spacemap-annotation-selection.png')
