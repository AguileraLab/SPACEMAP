"""GPU setting persistence and imported annotation group visibility in Firefox."""
from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1200;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab5]'));
 document.querySelector('[data-step=tab5]').click();
 const field=name=>document.querySelector('#studio-seg [aria-label="'+name+'"]');
 const settings=async()=>(await(await fetch('/spacemap-viewer/api/studio/browser-check/segmentation_settings')).json()).values;
 await until(()=>field('GPU batch mode'));
 assert(field('GPU batch mode').value==='Auto'&&field('GPU batch').hidden&&!field('GPU batch mode').closest('details').open,'GPU tuning defaults to Auto inside folded settings');
 field('GPU batch mode').closest('details').open=true;
 field('GPU batch mode').value='Manual';field('GPU batch mode').dispatchEvent(new Event('change'));
 field('GPU batch').value='128';field('GPU batch').dispatchEvent(new Event('change'));
 field('GPU index').value='2';field('GPU index').dispatchEvent(new Event('change'));
 await until(async()=>{const v=await settings();return !v.tab4_cellpose_batch_auto&&v.tab4_cellpose_batch_size===128&&v.tab4_gpu_device===2;});
 assert(!field('GPU batch').hidden,'Manual batch and GPU selection save together');
 document.querySelector('[data-step=tab4]').click();
 await delay(200);document.querySelector('[data-step=tab5]').click();
 await until(()=>field('GPU batch mode')?.value==='Manual');
 assert(field('GPU batch').value==='128'&&field('GPU index').value==='2','GPU settings survive navigation');
 const helpText=Array.from(document.querySelectorAll('#studio-seg [title]')).map(n=>n.title).join(' ');
 assert(!/NucleusC|A100|23,545|Chosen:/.test(helpText),'Segmentation help has no machine-specific benchmark or selected-model history');
 field('GPU batch mode').value='Auto';field('GPU batch mode').dispatchEvent(new Event('change'));
 await until(async()=>(await settings()).tab4_cellpose_batch_auto);
 assert(field('GPU batch').hidden,'Auto can be restored without deleting the manual limit');
 const features=['Tissue','Tumor','Stroma'].map((name,i)=>({type:'Feature',properties:{classification:{name,color:[0,255,0]}},
   geometry:{type:'Polygon',coordinates:[[[20+i*70,20],[60+i*70,20],[60+i*70,70],[20+i*70,70],[20+i*70,20]]]}}));
 await importAnnotationCopy({name:'review.geojson',features});
 const layer=()=>s.layers.find(l=>l.annotation_group==='imported'&&(l.source_name||l.name)==='review');
 await until(()=>layer());
 const branch=()=>document.querySelector('#panel [data-annotation-source="review"]');
 const master=()=>branch()?.querySelector('[aria-label="Show all annotations in review"]');
 const child=name=>branch().querySelector('[aria-label="Show '+name+'"]');
 await until(()=>master());
 assert(master().checked&&!master().indeterminate,'Imported file has one master checkbox for all annotation names');
 const before=JSON.stringify(layer().features);
 const open=branch().open;master().click();
 assert(!layer().visible&&!master().checked&&branch().open===open,'Master hides every sublayer without opening the panel');
 master().click();
 assert(layer().visible&&['Tissue','Tumor','Stroma'].every(n=>child(n).checked),'Master restores every annotation name');
 child('Tumor').click();
 assert(master().indeterminate&&layer().hiddenNames.has('Tumor'),'One hidden name gives the master a mixed state');
 saveView();
 const remembered=loadView(s.spec.id);
 assert(remembered.layers[viewLayerKey(layer())].hiddenNames.includes('Tumor'),'Individual name visibility is persisted');
 layer().hiddenNames=null;applySavedView(remembered);buildPanel();renderShapes();
 assert(!child('Tumor').checked&&master().indeterminate,'Restored view retains mixed visibility');
 master().click();
 assert(['Tissue','Tumor','Stroma'].every(n=>child(n).checked)&&!master().indeterminate,'Clicking mixed state shows all names');
 assert(JSON.stringify(layer().features)===before,'Visibility changes do not edit polygon geometry or names');
 assert(!branch().querySelector('.annotation-origin'),'Collapsed names keep source details out of the compact list');
 const legacy=s.layers.find(l=>l.type==='annotations');
 const legend=await annotationLegend(legacy);
 legacy.visible=true;legacy.hidden=new Set(legend.items.map(i=>i.value));buildPanel();
 const legacyMaster=()=>Array.from(document.querySelectorAll('#panel input')).find(n=>n.getAttribute('aria-label')==='Show all annotations in '+legacy.name);
 await until(()=>legacyMaster()&&!legacyMaster().checked&&!legacyMaster().indeterminate);
 assert(true,'Read-only GeoJSON with every class hidden shows an unchecked master');
 legacyMaster().click();
 await until(()=>legacyMaster()?.checked);
 assert(legacy.visible&&!legacy.hidden.size,'Read-only master clears hidden class filters when showing all');
 legacyMaster().click();
 assert(!legacy.visible&&layer().visible,'Hiding one GeoJSON group leaves other imported files visible');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}},100);
</script>
"""

if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180)
