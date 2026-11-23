"""Display carryover and stable annotation toggles in a disposable Studio."""
import json
import shutil
from viewer_threshold_browser import main


def setup(root):
    import pandas as pd
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from spacemap_gui.viewer import api, registry
    from spacemap_gui.viewer.ui import using_context, layers_for_image, _workspace_token, _display_scope
    from spacemap_gui.studio.annotation_components import display_style
    from spacemap.viewer.cells import ensure_label_pyramid
    from test_viewer import _square
    specs=[]
    folder=root/'display_annotations';folder.mkdir()
    for name in ('a','b','c'):
        image=root/(name+'.ome.tiff');shutil.copyfile(root/'roi.ome.tiff',image)
        if name=='c':
            import numpy as np
            import tifffile
            tifffile.imwrite(image,np.full((3,400,400),1200,dtype=np.uint16),metadata={'axes':'CYX','Channel':{'Name':['M0','M2','M3']}})
        table=root/(name+'_cells.csv');frame=pd.read_csv(root/'cells.csv');frame['batch_id']=name;frame.to_csv(table,index=False)
        seg=root/(name+'_cells.geojson');shutil.copyfile(root/'cells.geojson',seg)
        pyramid=ensure_label_pyramid(seg,400,400,root/(name+'_pyramid'))
        with using_context({'root':str(root),'mappings':{}}):
            layers,_=layers_for_image({'path':str(image),'name':image.name,'sample':name,'group':'Images'},context='tab7')
            workspace=_workspace_token(str(image))
            display_scope=_display_scope()
        layers=[l for l in layers if l['type']=='image']
        it=layers[0]['token']
        classes=['tissue','tumor','stroma']+(['extra'] if name=='b' else [])
        if name=='c':classes=['tissue','stroma']
        features=[]
        for i,label in enumerate(classes):
            f=_square(i,20+85*i,20,60);f['properties']={'classification':{'name':label,'color':[255 if i==1 else 0,255 if i==0 else 0,255 if i==2 else 0]}};features.append(f)
        ann=folder/(name+'.geojson');ann.write_text(json.dumps({'type':'FeatureCollection','features':features}))
        layers.extend([
            {'id':'ann0','type':'annotations','name':ann.name,'saved_file':str(ann),**display_style(ann),'token':registry.register_object('annotations',path=str(ann),image=str(image)),'image_token':it,'visible':False,'open':True},
            {'id':'he','type':'image','name':'H&E','token':it,'visible':False,'channels':[{'index':0,'name':'M0','visible':True,'low':0,'high':4000}]},
            {'id':'cells','type':'cells','name':'Cells','source':'segmentation/cells.geojson','token':registry.register_object('cells',image=str(image),pyramid=str(pyramid),segmentation=str(seg),roi=name,tables={'t0':str(table)},colors={}),
             'image_token':it,'visible':False,'roi':name,'table':'t0','tables':[{'id':'t0','name':"This segmentation’s cells",'file':str(table)}]}
        ])
        specs.append({'id':'display-'+name,'path':str(image),'session':'browser-check','display_scope':display_scope,'layers':layers,'workspace':workspace})
    async def fixture(request):return JSONResponse({'specs':specs})
    api.app.router.routes.append(Route('/display-fixtures',fixture))


DRIVER=r'''<script>
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<500;i++){if(await f())return;await pause(60);}throw Error('Timed out '+f+' '+document.querySelector('#message')?.textContent);};
let checks=0;const assert=(ok,text)=>{if(!ok)throw Error(text);checks++;fetch('/result?progress='+encodeURIComponent(text));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message)));
setTimeout(async()=>{try{
 const s=window.__state;await until(()=>s.imagesReady&&s.studio);
 const specs=(await getJSON('/display-fixtures')).specs;
 const open=async i=>{
  await postJSON('/session/browser-check/add-image',{path:specs[i].path});await init(structuredClone(specs[i]));await until(()=>s.imagesReady);await s.displayTransferReady;
  s.layerGroups.annotations=true;buildPanel();
  const branch=panel().querySelector('[data-annotation-source="'+ann().name+'"]');if(!branch.open)branch.querySelector('summary').click();
  await until(()=>check('tumor'));
 };
 const ann=()=>s.layers.find(l=>l.type==='annotations'),img=()=>s.layers.find(l=>l.id==='image'),he=()=>s.layers.find(l=>l.id==='he'),cell=()=>s.layers.find(l=>l.id==='cells');
 const panel=()=>document.querySelector('#panel'),check=name=>panel().querySelector('[aria-label="Show '+name+'"]');
 await open(0);
 await until(()=>check('tumor'));
 const source=panel().querySelector('[aria-label="Show all annotations in a.geojson"]');
 if(!source.checked)source.click();source.click();
 const holder=panel().querySelector('.component-browser'),tumor=check('tumor');
 const requests=()=>performance.getEntriesByType('resource').filter(r=>r.name.includes('/components/')).length;
 const before=requests();tumor.click();
 assert(ann().visible&&ann().hidden.has('tissue')&&ann().hidden.has('stroma')&&!ann().hidden.has('tumor'),'After hiding the source, tumor shows alone');
 assert(tumor===check('tumor')&&holder.isConnected,'Toggling a class keeps the exact same menu and checkbox nodes');
 tumor.click();tumor.click();await pause(700);
 assert(requests()===before&&check('tumor')===tumor,'Repeated class toggles do not reload component metadata or rebuild the menu');
 assert(source.indeterminate,'The parent checkbox shows a partial selection');
 check('tumor').click();check('tissue').click();
 setAnnotationOpacity(.38);setVisible(he(),true);setOpacity(he(),.63);
 img().channels.forEach((c,i)=>{c.visible=i===1;c.color=i===1?'#55bb33':c.color;c.low=11;c.high=999;});remember(img());
 cell().visible=true;cell().color_by='phenotype_ref1';cell().filters=[{col:'M0',op:'>=',value:50}];cell().hidden=new Set(['B']);cell().groupColours={A:'#ab1234'};cell().mode='both';
 const density=()=>s.layers.find(l=>isCellDensityMap(l)&&!l.densityDormant);
 densityFromCellsButton(cell()).click();await until(()=>density()&&!cell().densityPending);
 density().radius=85;density().opacity=.42;const oldDensityToken=density().token;
 s.viewer.viewport.zoomTo(3,null,true);s.viewer.viewport.panTo(new OpenSeadragon.Point(.4,.45),true);saveView();
 await open(1);await until(()=>ann().hidden.has('extra'));
 assert(he().visible&&he().opacity===.63,'H&E visibility and opacity carry to the next image');
 assert(img().channels[1].visible&&!img().channels[0].visible&&img().channels[1].color==='#55bb33'&&img().channels[1].low===11,'Marker visibility, colour and range carry by marker name');
 assert(ann().visible&&!ann().hidden.has('tissue')&&['tumor','stroma','extra'].every(n=>ann().hidden.has(n))&&ann().opacity===.38,'Only tissue remains visible on the next image at the chosen opacity, including when it has extra classes');
 assert(cell().visible&&cell().color_by==='phenotype_ref1'&&cell().filters[0]?.col==='M0'&&cell().hidden.has('B'),'Compatible per-image cell data keeps colour and filters');
 assert(density()?.visible&&density().radius===85&&density().opacity===.42&&density().token===cell().token&&density().token!==oldDensityToken,'An enabled density map carries its settings using the new image’s cell source');
 densityFromCellsButton(cell()).click();assert(!density(),'Turning Density off removes the active map');
 assert(Math.abs(s.viewer.viewport.getZoom()-3)>.1,'A new image does not inherit the previous zoom');
 s.viewer.viewport.zoomTo(1.7,null,true);saveView();
 await until(()=>check('tumor'));check('tissue').click();check('tumor').click();setAnnotationOpacity(.71);setVisible(he(),false);saveView();
 await open(0);
 assert(ann().visible&&!ann().hidden.has('tumor')&&ann().hidden.has('tissue')&&ann().opacity===.71&&!he().visible,'Returning uses the latest annotation and H&E choices rather than an old image display');
 assert(!density(),'A previously saved density map stays off after the latest explicit off choice');
 assert(Math.abs(s.viewer.viewport.getZoom()-3)<.05,'Returning restores that image’s own zoom');
 await open(1);assert(Math.abs(s.viewer.viewport.getZoom()-1.7)<.05,'The other image retains its own independent zoom');
 s.layerGroups.annotations=true;buildPanel();await until(()=>check('tumor'));
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(e+'\n'+e.stack));}},100);
</script>'''

if __name__=='__main__':
    main(DRIVER,studio=True,setup=setup,timeout=180,capture='/tmp/spacemap-annotation-display.png')
