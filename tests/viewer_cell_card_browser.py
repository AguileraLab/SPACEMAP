"""Cell-card fields and named marker lists stay compact across cell changes."""
from viewer_threshold_browser import main

DRIVER = r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(40);}throw Error('Timed out '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const control=n=>document.querySelector('[aria-label="'+n+'"]'),change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));};
(async()=>{try{
 const s=window.__state;await until(()=>s.cellsReady&&s.imagesReady);
 const layer=s.layers.find(l=>l.type==='cells'),image=s.layers.find(l=>l.type==='image'),result=()=>s.inspectedCell?.result;
 const click=label=>{const n=label-1;s.viewer.raiseEvent('canvas-click',{quick:true,position:s.viewer.viewport.pixelFromPoint(image.item.imageToViewportCoordinates(40*(n%8)+15,40*Math.floor(n/8)+15)),originalEvent:{}});};
 for(const [name,markers] of [['Lymphocytes',['M1']],['Phenotyping',['M0','M1']]])await postJSON('/display/'+image.display_token+'/sets',{name,markers:markers.map(name=>({name,color:'#44aa88',low:0,high:500}))});
 image.markerSets=(await getJSON('/display/'+image.display_token+'/sets')).sets;
 s.markersOpen=true;click(1);await until(()=>result()?.profile?.status==='ready');
 const allFields=()=>Array.from(document.querySelectorAll('#cellcard [data-cell-field]')),shown=()=>allFields().filter(r=>!r.hidden).map(r=>r.dataset.cellField);
 check(allFields().length>=2,'Real cell details are available in the Fields menu');
 control('Choose cell fields').click();const search=control('Find cell fields');search.value='training';search.dispatchEvent(new Event('input'));
 const matching=allFields().filter(r=>r.dataset.cellField.includes('training')).map(r=>r.dataset.cellField),initial=shown();
 control('Select all matching cell fields').click();
 check(matching.length>0&&matching.every(k=>!shown().includes(k))&&shown().length===initial.length-matching.length,'Search and uncheck all affects only matching fields');
 check(document.querySelector('#cellcard .call-main .chip').textContent.length>0,'The main phenotype remains visible');
 const chosen=shown().slice();click(2);await until(()=>result()?.profile?.status==='ready'&&result().label===2);
 check(JSON.stringify(shown())===JSON.stringify(chosen),'Field choices carry over to the next cell');
 delete s.cellCardFields;showCellCard(layer,result());check(JSON.stringify(shown())===JSON.stringify(chosen),'Field choices restore from saved browser preferences');
 const sets=control('Cell card marker set');check(['Lymphocytes','Phenotyping'].every(n=>Array.from(sets.options).some(o=>o.text===n)),'The card uses the exact marker-set names saved for the Image layers');
 const before=image.channels.map(c=>[c.visible,c.color,c.low,c.high]),profile=result().profile;
 change(sets,'Lymphocytes');
 const markers=()=>Array.from(document.querySelectorAll('#cellcard .cell-profile td.k')).map(c=>c.textContent);
 check(markers().join(',')==='M1','Choosing a marker set filters the card to its measured markers');
 change(control('Cell card marker set'),'Phenotyping');const count=control('Cell card top marker count');count.value='1';count.dispatchEvent(new Event('input'));
 check(markers().length===1&&markers()[0]===rankedCellMarkers(profile)[0].marker,'Show top ranks the current values and honours the chosen count');
 check(JSON.stringify(before)===JSON.stringify(image.channels.map(c=>[c.visible,c.color,c.low,c.high]))&&result().profile===profile,'Card filters preserve image visibility, contrast and calculated measurements');
 control('Show all cell card markers').click();check(markers().length===2&&control('Cell card top marker count').disabled,'All shows every measured marker in the selected set');
 control('Show all cell card markers').click();click(3);await until(()=>result()?.profile?.status==='ready'&&result().label===3);
 check(markers().length===1&&control('Cell card marker set').value==='Phenotyping','The marker set and top count carry over to the next cell');
 delete s.cellCardMarkers;showCellCard(layer,result());check(control('Cell card top marker count').value==='1'&&control('Cell card marker set').value==='Phenotyping','Marker choices restore from saved browser preferences');
 change(control('Cell card marker set'),'Lymphocytes');delete image.markerSets.Lymphocytes;document.dispatchEvent(new CustomEvent('spacemap-marker-sets',{detail:{layer:image}}));
 check(!markers().length&&control('Cell card marker set').selectedOptions[0].text.includes('unavailable'),'A removed marker set is explicit and does not silently display unrelated markers');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}})();
</script>'''

if __name__=='__main__':
    main(DRIVER,timeout=120,screenshot=False)
