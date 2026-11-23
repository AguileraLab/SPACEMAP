"""Lists navigation, compact rows, exact-source deletion and typed imports in Firefox."""
import json

from viewer_threshold_browser import main


def setup(root):
    import shutil
    from spacemap_gui.studio import annotation_inputs
    from spacemap_gui import persistence
    persistence.save_project_config(str(root), {'directory_mappings': {'annot_ta_input_dir': str(root), 'tab3_output_dir': str(root)}})
    features=[]
    for i,name in enumerate(['Endothelia','Tumor','Epithelia','Stroma']):
        x=20+i*40
        features.append(dict(type='Feature',properties={'classification':{'name':name,'color':[60,180,120]}},
                             geometry={'type':'Polygon','coordinates':[[[x,10],[x+20,10],[x+20,30],[x,30],[x,10]]]}))
    for i in range(3):
        annotation_inputs.import_polygons(root,root/'roi.ome.tiff','review.geojson',features[:4-i])
    folder=root/'outputs/segmentation';folder.mkdir(exist_ok=True,parents=True)
    (folder/'roi.ome_cell_features.csv').write_text('label,x_centroid,y_centroid,M0\n1,20,20,15\n2,60,20,35\n')
    shutil.copy2(root/'roi.ome.tiff',root/'z_other.ome.tiff')
    incoming=root/'incoming-annotations';(incoming/'nested').mkdir(parents=True)
    for path in [incoming/'roi.geojson',incoming/'z_other.geojson',incoming/'unknown.geojson']:
        path.write_text(json.dumps({'type':'FeatureCollection','features':features}))
    (incoming/'nested/ignored.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    (incoming/'broken.geojson').write_text('invalid JSON')
    (incoming/'named.json').write_text('{"settings":true}')
    (incoming/'empty.geojson').write_text('{"type":"FeatureCollection","features":[]}')
    history=incoming/'studio_history/old.geojson/states';history.mkdir(parents=True)
    (history/'snapshot.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))


DRIVER=r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1600;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f+' · '+document.querySelector('.lists-status')?.textContent+' · '+document.querySelector('#studio-action-status')?.textContent);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.confirm=()=>true;
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(b=>b.textContent===text);
const tab=()=>document.querySelector('[role=tab][data-tab=outputs]');
const step=id=>document.querySelector('[data-list-step="'+id+'"]');
const group=id=>document.querySelector('[data-list-group="'+id+'"]');
const openStep=async id=>{step(id).open=true;await until(()=>step(id).querySelector('.lists-group'));};
const openGroup=async id=>{group(id).open=true;await until(()=>group(id).querySelector('.lists-file,.lists-empty'));};
const request=async(path,body)=>{const r=await fetch('/spacemap-viewer/api/studio/browser-check/'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});if(!r.ok)throw Error(await r.text());return r.json();};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.studio&&tab());
 assert(Array.from(document.querySelectorAll('#studio-right > .studio-tabs [role=tab]')).map(b=>b.querySelector('span').textContent).join(',')==='Layers,Lists,Jobs,Assistant','Right dock order is Layers, Lists, Jobs, Assistant');
 assert(tab().title.includes('Inputs, outputs and imports'),'Lists tooltip describes its purpose');
 assert(!document.querySelector('.lists-step'),'Inventory is not scanned before Lists is opened');
 tab().click();await until(()=>step('tab4'));
 assert(document.querySelectorAll('.lists-step').length===11,'All workflow steps have matching sections');
 await openStep('tab4');
 assert(getComputedStyle(group('imports').querySelector('.lists-actions')).display==='none','Collapsed groups hide selection and actions');
 await openGroup('imports');
 const rows=()=>group('imports').querySelectorAll('.lists-file');
 assert(rows().length===3,'Every duplicate imported GeoJSON has its own row');
 assert(new Set(Array.from(rows(),r=>r.dataset.listItem)).size===3,'Duplicate rows retain distinct identities');
 assert(Array.from(rows()).every(r=>r.title.startsWith('/')&&r.querySelector('.lists-name').title===r.title),'Full paths appear on each file and filename');
 const largest=Array.from(rows()).find(r=>r.querySelectorAll('.lists-class').length===4);
 assert(!!largest,'Class initials show the file’s four annotation classes');
 assert(largest.querySelector('.lists-class').title.includes('Endothelia'),'Class hover gives its full name and count');
 assert(rows()[0].getBoundingClientRect().height<=26,'File rows have compact spacing without small text');
 const first=largest,id=first.dataset.listItem;
 first.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:500,clientY:200}));
 assert(document.querySelector('.lists-context [role=menuitem]').textContent==='Copy path','Right-click offers Copy path');
 document.querySelector('.lists-context').remove();
 first.querySelector('.lists-class').click();
 await until(()=>document.querySelector('[data-step=tab4]').getAttribute('aria-current')==='step'&&document.querySelector('[data-tab=layers]').getAttribute('aria-selected')==='true');
 await until(()=>s.layers.some(l=>l.saved_file===first.title&&l.visible));
 const loaded=s.layers.find(l=>l.saved_file===first.title);
 assert((loaded.type==='shapes'?loaded.hiddenNames:loaded.hidden)?.has('Stroma')&&!(loaded.type==='shapes'?loaded.hiddenNames:loaded.hidden)?.has('Endothelia'),'Class click opens exact source and chosen annotation: '+JSON.stringify({type:loaded.type,hidden:[...(loaded.hiddenNames||[])],classes:[...(loaded.hidden||[])],names:loaded.features?.map(f=>annotationName(loaded,f))}));
 tab().click();await until(()=>rows().length===3);
 Array.from(rows()).find(r=>r.dataset.listItem===id).querySelector('input').click();Array.from(rows()).find(r=>r.dataset.listItem!==id).querySelector('input').click();
 const search=group('imports').querySelector('input[type=search]');search.value='no-match';search.dispatchEvent(new Event('input'));
 assert(!group('imports').querySelector('.lists-file'),'Search filters file rows');
 search.value='';search.dispatchEvent(new Event('input'));
 assert(Array.from(rows()).filter(r=>r.querySelector('input').checked).length===2,'Filtering preserves selection');
 const removedPaths=Array.from(rows()).filter(r=>r.querySelector('input').checked).map(r=>r.title);
 button(group('imports').querySelector('summary'),'Remove').click();
 await until(()=>rows().length===1);
 assert(removedPaths.every(path=>!s.layers.some(l=>l.saved_file===path)),'Multi-remove clears loaded layers immediately');
 const plan=await request('segmentation_images');
 assert(removedPaths.every(p=>!JSON.stringify(plan).includes(p)),'Removed imports disappear from segmentation sources');
 await openStep('tab5');await openGroup('measurements');
 const measurement=group('measurements').querySelector('.lists-name');assert(measurement.textContent==='roi.ome_cell_features.csv','Measurements are listed under Segmentation');
 measurement.click();
 await until(()=>document.querySelector('[data-step=tab5]').getAttribute('aria-current')==='step'&&dataView.source?.file);
 await until(()=>dataView.result?.rows?.length===2);
 assert(document.querySelector('[data-tab=layers]').getAttribute('aria-selected')==='true','Opening a table switches the dock to Layers');
 assert(s.layers.find(l=>l.type==='image').name==='roi.ome.tiff','Measurement click opens its matching image');
 tab().click();await openStep('tab1');
 await openGroup('codex');assert(group('codex').querySelector('.lists-empty'),'Registration input groups are available even when empty');
 group('imports').open=true;step('tab4').open=true;
 const file=new File([JSON.stringify({type:'FeatureCollection',features:[{type:'Feature',properties:{classification:{name:'New class'}},geometry:{type:'Polygon',coordinates:[[[0,0],[10,0],[10,10],[0,10],[0,0]]]}}]})],'roi_new.geojson',{type:'application/geo+json'});
 const transfer=new DataTransfer();transfer.items.add(file);
 group('imports').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer}));
 await until(()=>document.querySelector('.lists-import-dialog'));
 button(document.querySelector('.lists-import-dialog'),'Import').click();
 await until(()=>!document.querySelector('.lists-import-dialog')&&rows().length===2);
 assert(Array.from(rows()).some(r=>r.textContent.includes('roi_new.geojson')),'Dropping GeoJSON into its group imports and refreshes the list');
 const all=group('imports').querySelector('[aria-label="Select matching Imported GeoJSON"]');all.click();
 assert(Array.from(rows()).every(r=>r.querySelector('input').checked),'Select-all acts on every matching row');
 const project=first.title.split('/outputs/studio/')[0];
 button(group('imports').querySelector('summary'),'Import').click();
 await until(()=>document.querySelector('.lists-import-dialog'));
 let dialog=document.querySelector('.lists-import-dialog');dialog.querySelector('textarea').value=project+'/incoming-annotations';
 dialog.querySelector('select').value='roi.ome.tiff';button(dialog,'Import').click();
 await until(()=>rows().length===5&&dialog.querySelector('.lists-error').textContent.includes('1 failed'));
 assert(!Array.from(rows()).some(r=>r.title.includes('/nested/')),'Folder imports ignore all subdirectories');
 assert(document.querySelector('.lists-status').textContent.includes('3 imported'),'Folder import continues after an invalid file and retains unmatched files');
 assert(document.querySelector('.lists-status').textContent.includes('1 empty files skipped'),'Directory imports skip empty geometry, metadata and internal history');
 const problems=dialog.querySelector('.lists-import-problems');
 assert(problems&&!problems.open&&!problems.hidden&&dialog.getBoundingClientRect().height<300,'Detailed import errors stay collapsed in a compact dialog');
 dialog.querySelector('[aria-label="Close import"]').click();
 const linked=Array.from(rows()).find(r=>r.title.endsWith('/z_other.geojson'));
 assert(linked&&linked.title===project+'/incoming-annotations/z_other.geojson','Folder imports link original paths without copying');
 linked.querySelector('.lists-class').click();
 await until(()=>s.layers.find(l=>l.type==='image')?.name==='z_other.ome.tiff'&&document.querySelector('[data-tab=layers]').getAttribute('aria-selected')==='true');
 assert(s.layers.some(l=>l.saved_file===linked.title&&l.type==='annotations'),'Each folder file opens its own image and a read-only original');
 tab().click();await until(()=>rows().length===5);
 const unmatched=Array.from(rows()).find(r=>r.title.endsWith('/unknown.geojson'));
 button(unmatched,'Match…').click();await until(()=>document.querySelector('[aria-label="Match imported annotation"]'));
 dialog=document.querySelector('.lists-import-dialog');const match=dialog.querySelector('select');match.value=Array.from(match.options).find(o=>o.textContent==='roi.ome.tiff').value;button(dialog,'Match').click();
 await until(()=>!document.querySelector('.lists-import-dialog')&&!button(Array.from(rows()).find(r=>r.title.endsWith('/unknown.geojson')),'Match…'));
 assert(true,'Unmatched linked imports can be assigned without another copy');
 document.querySelector('#app').style.setProperty('--studio-right','920px');await delay(150);
 const header=group('imports').querySelector('summary');
 assert(button(header,'Import').getBoundingClientRect().top<header.getBoundingClientRect().top+10,'Actions share the group header in a wide dock');
 document.querySelector('#app').style.setProperty('--studio-right','350px');await delay(150);
 assert(document.querySelector('#studio-outputs').scrollWidth<=document.querySelector('#studio-outputs').clientWidth+2,'Narrow Lists dock has no horizontal overflow');
 document.querySelector('#app').style.setProperty('--studio-right','780px');await delay(150);
 step('tab1').open=false;step('tab5').open=false;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(error.message+'\n'+error.stack));}},100);
</script>
"""

if __name__=='__main__':
    main(DRIVER,studio=True,setup=setup,timeout=180)
