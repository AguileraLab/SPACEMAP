"""Working/imported roots, artifact subgroups and exact-file display identity."""
import json
from viewer_threshold_browser import main


def setup(root):
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api, registry, ui
    from spacemap_gui.viewer.artifact_api import _layer, _examples_layer
    from spacemap_gui.studio import annotation, annotation_edits, artifact_learning, annotation_inputs
    image=str(root/'roi.ome.tiff');output=root/'outputs/processed_annotations/working_files'
    def write(path,name,count):
        features=[dict(type='Feature',properties={'classification':{'name':name,'color':[60,160,90]}},
            geometry={'type':'Polygon','coordinates':[[[20+i*40,20],[45+i*40,20],[45+i*40,45],[20+i*40,45],[20+i*40,20]]]}) for i in range(count)]
        annotation_edits.replace(path,features,annotation_edits.revision(path),'Fixture')
    tissue=annotation.valid_tissue(output,'roi');write(tissue,'tissue',3)
    write(artifact_learning.polygons_path(output,'roi','clot'),'Blood_Clot',2)
    write(artifact_learning.example_path(root,'clot',image),'Good tissue',1)
    layers=[ui.shapes_layer(tissue,image,name='tissue',layer_id='tissue-regions',regions=True,annotation_kind='tissue'),
        _layer(output,'roi','clot',image),_examples_layer(root,'clot',image)]
    for parent in ('external-a','external-b'):
        path=root/parent/'roi.geojson';write(path,'Stroma',1)
        layers.append(api._annotation_layer(str(path),image,registry.register(image),parent+' / roi.geojson'))
    duplicate=api._annotation_layer(str(tissue),image,registry.register(image),'Old tissue view')
    duplicate.update(annotation_inputs.display_metadata(root,image,tissue));layers.append(duplicate)
    async def fixture(request):return JSONResponse({'layers':layers})
    api.app.router.routes.append(Route('/hierarchy-fixture',fixture))


DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(const end=Date.now()+60000;Date.now()<end;){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.studio);
 const fixture=await(await fetch('/spacemap-viewer/api/hierarchy-fixture')).json();
 for(const layer of fixture.layers)await addLayerToView(layer);
 s.layerGroups.annotations=true;buildPanel();
 const group=()=>document.querySelector('[data-layer-group=annotations]');
 const root=key=>group().querySelector('[data-annotation-root='+key+']');
 check(group().querySelectorAll(':scope > [data-annotation-root]').length===2,'Only Working annotations and Imported GeoJSON are source roots');
 check(root('working').querySelector('summary .name').textContent==='Working annotations','Working annotations has an explicit title');
 check(root('working').querySelector('[data-annotation-section=Tissue] [data-annotation-name=tissue]'),'Tissue outputs are nested under Tissue');
 const artifact=()=>root('working').querySelector('[data-annotation-section=Artifacts] [data-annotation-name="Blood clots"]');
 check(artifact(),'Artifacts contains its Blood clots category');
 artifact().querySelector('.annotation-fold').click();
 const bad=()=>artifact().querySelector('[data-layer-id=artifact-clot]');
 const good=()=>artifact().querySelector('[data-layer-id=artifact-examples-clot]');
 check(bad().tagName==='DETAILS'&&good().tagName==='DETAILS','Good and bad tissue are separate collapsible subgroups');
 check(/Bad tissue/.test(bad().firstElementChild.textContent)&&/Good tissue/.test(good().firstElementChild.textContent),'Artifact roles have clear names');
 check(!bad().querySelector('.layer-body')&&!good().querySelector('.layer-body'),'Folded roles do not build polygon controls');
 bad().open=true;await until(()=>bad().querySelector('.region-list'));
 check(!good().open&&bad().querySelectorAll('.region-list input[type=checkbox]').length===2,'Expanding bad tissue shows its individual polygons only');
 check(!bad().querySelector('.artifact-layer-settings'),'Layers has no prediction parameters');
 const before=JSON.stringify(s.layers.filter(l=>l.type==='shapes').map(l=>l.features));
 bad().querySelector('summary input[type=checkbox]').click();
 check(!s.layers.find(l=>l.id==='artifact-clot').visible,'Artifact visibility changes its own layer');
 artifact().querySelector('.annotation-fold').click();
 check(!bad().getClientRects().length,'Collapsing an artifact hides all nested roles');
 check(JSON.stringify(s.layers.filter(l=>l.type==='shapes').map(l=>l.features))===before,'Hierarchy and visibility preserve polygon geometry');
 const tissue=s.layers.find(l=>l.id==='tissue-regions');
 check(s.layers.filter(l=>l.saved_file===tissue.saved_file).length===1,'A generated GeoJSON shown as working polygons has one display layer');
 const imports=[...root('imports').querySelectorAll('[data-annotation-key]')].filter(b=>/external-[ab]/.test(b.dataset.annotationKey));
 check(imports.length===2,'Separate imported files retain their identities');
 for(const branch of imports){check(branch.querySelector('summary .name').title===branch.dataset.annotationKey,'Imported name hover gives its full absolute path');branch.open=true;}
 await until(()=>imports.every(b=>b.querySelector('[data-component-class]')));
 artifact().querySelector('.annotation-fold').click();bad().open=true;good().open=true;
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(e.message+'\n'+e.stack));}},100);
</script>'''

if __name__=='__main__':main(DRIVER,studio=True,setup=setup,timeout=150)
