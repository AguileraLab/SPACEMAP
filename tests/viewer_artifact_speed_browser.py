"""Browser-check how quickly many artifact polygons respond: selecting one, and marking it as good tissue (and back).

Usage: PYTHONPATH=src:tests python tests/viewer_artifact_speed_browser.py
"""
from viewer_threshold_browser import main
from viewer_artifact_review_browser import setup as artifact_setup

DRIVER = r"""
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<2400;i++){if(await f())return;await delay(25);}throw Error('Timed out: '+f);};
let checks=0;const timings={};const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(p,t)=>Array.from(p.querySelectorAll('button')).find(b=>b.textContent===t);
window.confirm=()=>true;
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 const s=window.__state,tools=()=>document.querySelector('#studio-tools');
 await until(()=>s.imagesReady&&document.querySelector('[data-step=tab4]'));document.querySelector('[data-step=tab4]').click();
 await until(()=>tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name'));
 tools().querySelector('tr[data-image="markers.ome.tiff"] button.studio-item-name').click();
 await until(()=>s.imagesReady&&s.layers.find(l=>l.type==='image')?.name==='markers.ome.tiff');
 Array.from(document.querySelectorAll('.studio-stage-tabs button')).find(b=>b.textContent==='Artifacts').click();
 const red=()=>s.layers.find(l=>l.id==='artifact-clot');
 await until(()=>red()?.features?.length===400&&tools().querySelector('[data-example-role=positive]'));
 const list=()=>tools().querySelector('.artifact-polygons .region-list');
 await until(()=>list()?.querySelector('input[type=checkbox]'));
 let t0=performance.now();
 list().querySelector('input[type=checkbox]').click();
 await until(()=>s.picks?.ids?.size===1);
 timings.select=Math.round(performance.now()-t0);
 const good=Array.from(tools().querySelectorAll('button')).find(b=>/^Mark as good/.test(b.textContent));
 t0=performance.now();good.click();
 await until(()=>red().features.length===399&&!red().brushBusy);
 timings.mark_good=Math.round(performance.now()-t0);
 // Go on a polygon far down the list: the image goes there and the list stays where it was scrolled.
 list().scrollTop=list().scrollHeight;list().dispatchEvent(new Event('scroll'));
 const end=list().scrollTop,rows=list().querySelectorAll('.shape-row'),last=rows[rows.length-1];
 button(last,'Go').click();
 await until(()=>s.picks?.ids?.has(last.dataset.id)&&list()?.querySelector('.shape-row.on'));
 await delay(100);
 assert(end>0&&Math.abs(list().scrollTop-end)<5,'Go keeps the list where it was ('+Math.round(list().scrollTop)+' of '+Math.round(end)+')');
 // Colours: Blood clots (the artifacts) and Good tissue each keep their own.
 s.layerGroups=s.layerGroups||{};s.layerGroups.annotations=true;buildPanel();
 const row=()=>document.querySelector('#panel [data-annotation-name="Blood clots"]');
 await until(()=>row()?.querySelector('[aria-label="Colour of Blood clots Good tissue (training)"]'));
 const set=(input,value)=>{input.value=value;input.dispatchEvent(new Event('change'));};
 const greenLayer=()=>s.layers.find(l=>l.id==='artifact-examples-clot');
 set(row().querySelector('[aria-label="Colour of Blood clots Good tissue (training)"]'),'#00ff00');
 await until(()=>greenLayer().features.every(f=>f.properties.colour==='#00ff00'));
 assert(red().features.every(f=>f.properties.colour!=='#00ff00'),'Good tissue takes its own colour; the clots keep theirs');
 set(row().querySelector('[aria-label="Colour of Blood clots"]'),'#0000ff');
 await until(()=>red().features.every(f=>f.properties.colour==='#0000ff'));
 assert(greenLayer().features.every(f=>f.properties.colour==='#00ff00'),'The Blood clots colour changes the clots only');
 assert(timings.mark_good<15000,'Marking one of 400 polygons as good tissue: '+timings.mark_good+' ms; selecting: '+timings.select+' ms');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,timings})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack+' '+JSON.stringify(timings)));}},100);
</script>
"""


def setup(root):
    """400 Blood clots polygons, each with 40 one-pixel holes, as a raw prediction is saved."""
    import json

    from shapely.geometry import Polygon, mapping

    from spacemap_gui.studio import artifact_learning

    artifact_setup(root)
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    features = []
    for i in range(400):
        x, y = 10 + (i % 20) * 19, 10 + (i // 20) * 19
        holes = [[(x + 2 + a * 3, y + 2 + b * 3), (x + 3 + a * 3, y + 2 + b * 3), (x + 3 + a * 3, y + 3 + b * 3), (x + 2 + a * 3, y + 3 + b * 3)] for a in range(5) for b in range(5)]
        ring = [(x + 8 + 8 * __import__("math").cos(k / 30 * 6.283), y + 8 + 8 * __import__("math").sin(k / 30 * 6.283)) for k in range(30)]
        poly = Polygon([(x, y), (x + 16, y), (x + 16, y + 16), (x, y + 16)], holes)
        features.append({"type": "Feature", "geometry": mapping(poly), "properties": {"classification": {"name": artifact_learning.CLASSES["clot"]}, "source": "reviewed_model"}})
    path = artifact_learning.polygons_path(output, "markers", "clot")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=240, setup=setup)
