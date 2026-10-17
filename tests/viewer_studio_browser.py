"""Browser-check the Studio frame and real image switching with disposable data."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading

DRIVER = r"""
<script>
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const until = async check => { for(let i=0;i<400;i++) {if(check()) return; await delay(50);} throw Error('Timed out: '+check); };
let checks=0;
const assert = (value,text) => {if(!value) throw Error(text); checks++;};
setTimeout(async () => {
 try {
  const state=window.__state;fetch('/progress?step=start');
  await until(()=>document.querySelectorAll('#studio-steps nav button').length===11);
  assert(document.body.classList.contains('studio'), 'Studio mode opens');
  assert(!performance.getEntriesByType('resource').some(r=>r.name.endsWith('/images')), 'Image list stays lazy until requested');
  assert(!document.querySelector('#studio-empty').hidden,'Empty Studio explains how to begin');
  const header=document.querySelector('#studio-header'), nav=document.querySelector('#studio-steps'), tools=document.querySelector('#studio-tools'), stage=document.querySelector('#stage'), right=document.querySelector('#studio-right');
  const box=element=>element.getBoundingClientRect();
  assert(box(header).bottom<=box(stage).top,'Command bar is above the image');
  assert(box(nav).right<=box(tools).left && box(tools).right<=box(stage).left && box(stage).right<=box(right).left,'Steps, step tools, image and outputs form four columns');
  assert(right.querySelector('#panel') && right.querySelectorAll('[role=tab]').length===4,'Layers sit in the right-hand tabs with Outputs, Jobs and the Assistant');
  right.querySelector('[data-tab=outputs]').click();
  assert(document.querySelector('#panel').hidden && !document.querySelector('#studio-outputs').hidden,'The tabs switch the right-hand pane');
  right.querySelector('[data-tab=layers]').click();
  const paneBox=box(document.querySelector('#panel')), rightBox=box(right);
  assert(paneBox.height>0.8*rightBox.height,'The Layers tab fills the right-hand pane: '+paneBox.height+' of '+rightBox.height);
  assert(document.querySelector('#studio-dock').hidden,'The figures dock stays out of the way until there is a figure');
  const handle=right.querySelector('.studio-resize-x');
  handle.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,clientX:rightBox.left,pointerId:1}));
  handle.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,clientX:window.innerWidth-520,pointerId:1}));
  handle.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,clientX:window.innerWidth-520,pointerId:1}));
  await delay(100);
  assert(Math.abs(box(right).width-520)<4,'Dragging the left edge widens the right-hand pane: '+box(right).width);
  handle.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
  assert(Math.abs(box(right).width-480)<4,'The arrow keys resize it too');
  // The two left panels widen from their right edge the same way.
  for(const [pane,width] of [[tools,420],[nav,240]]){
    const grip=pane.querySelector(':scope > .studio-resize-x'), left=box(pane).left;
    grip.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,clientX:box(pane).right,pointerId:2}));
    grip.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,clientX:left+width,pointerId:2}));
    grip.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,clientX:left+width,pointerId:2}));
    await delay(100);
    assert(Math.abs(box(pane).width-width)<4,'Dragging its right edge resizes the '+pane.id+' panel: '+box(pane).width);
  }
  // A (?) shows its text above every panel, inside the window.
  const question=tools.querySelector('.help');question.dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));
  const tip=document.querySelector('#floating-tip'), tipBox=tip&&box(tip);
  assert(tip && !tip.hidden && tip.textContent===question.dataset.tip && tipBox.left>=0 && tipBox.right<=window.innerWidth && tipBox.bottom<=window.innerHeight &&
    tip.parentElement===document.body && getComputedStyle(tip).position==='fixed' && Number(getComputedStyle(tip).zIndex)>=10000,'The help text floats on top, inside the window');
  question.dispatchEvent(new MouseEvent('mouseout',{bubbles:true,relatedTarget:document.body}));
  assert(tip.hidden,'It goes when the pointer leaves');
  await until(()=>document.querySelector('[data-step=tab0] .studio-dot').dataset.state!=='unknown');
  assert(true,'Each step shows its state as a dot');
  for(const button of document.querySelectorAll('#studio-steps nav button')) {
   button.click();
   assert(button.getAttribute('aria-current')==='step','Each step opens');
   assert(document.querySelector('#studio-tools .studio-work-list') || ['tab0','tab1','tab2','tab3','tab4','tab5','tagging','tab6','tab8','tab9'].includes(button.dataset.step),'Each step lists its items (Project, Registration, Background, ROIs, Annotation and Segmentation, Cell tagging, Batch correction, Spatial analysis and Visualization in their own way)');
   assert(document.querySelector('.studio-step-head a').href.includes('/'+({tab0:'project',tab1:'registration',tab2:'bg_subtraction',tab3:'roi',tab4:'annotation',tab5:'segmentation',tagging:'segmentation',tab6:'batch_correction',tab7:'phenotyping',tab8:'analysis',tab9:'visualization'})[button.dataset.step]),'Each step links to its existing classic page');
  }
  document.querySelector('[data-step=tab0]').click();
  await until(()=>document.querySelectorAll('#studio-tools .studio-matrix-row').length>1 && document.querySelector('#studio-tools .studio-folder'));
  const projectPage=document.querySelector('#studio-tools').textContent;
  assert(/\d+ samples?/.test(projectPage) && document.querySelectorAll('#studio-tools .studio-matrix-row .studio-dot').length>0,'The Project step shows each sample\'s state per step: '+projectPage);
  assert(/Registered images \(out\)/.test(projectPage) && /Projects/.test(projectPage) && !/AI setup/.test(projectPage) && !/null|undefined/.test(projectPage),'It lists the step folders and the projects to switch to: '+projectPage);
  document.querySelector('.studio-image-list').open=true;
  await until(()=>document.querySelectorAll('.studio-image').length===2);
  document.querySelectorAll('.studio-image')[0].click();
  await until(()=>state.layers.some(l=>l.type==='image' && l.item?.getFullyLoaded()));
  const first=state.layers.find(l=>l.type==='image');
  assert(document.querySelector('#studio-empty').hidden,'An image replaces the empty state');
  document.querySelector('[data-step=tab7]').click();
  assert(state.layers.find(l=>l.type==='image').item===first.item,'Switching steps preserves the image and zoom');
  const method=document.querySelector('[aria-label="Phenotyping method"]');
  assert(Array.from(method.options).some(o=>o.value==='resolve') && Array.from(method.options).some(o=>o.value==='leiden'),'RESOLVE is one Phenotyping method among the others');
  assert(!Array.from(method.options).some(o=>o.value==='scimap'),'SCIMAP is not offered in Studio');
  method.value='tools';method.dispatchEvent(new Event('change'));
  assert(document.querySelector('.studio-method').textContent.includes('Classic Tab 7'),'Methods without Studio controls open classic Tab 7');
  method.value='resolve';method.dispatchEvent(new Event('change'));
  assert(document.querySelector('#studio-resolve'),'The RESOLVE method has its panel');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'k',ctrlKey:true,bubbles:true}));
  const search=document.querySelector('[aria-label="Find a step or image"]');
  assert(document.activeElement===search,'Ctrl+K focuses the command bar');
  search.value='second'; search.dispatchEvent(new Event('input'));
  assert(document.querySelectorAll('#studio-matches button').length===1,'The finder filters images');
  search.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>state.layers.some(l=>l.type==='image' && l.token!==first.token && l.item?.getFullyLoaded()));
  assert(state.spec.title.includes('second'),'The finder opens the chosen project image');
  search.focus(); search.value='Registration';search.dispatchEvent(new Event('input'));
  search.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  assert(document.querySelector('[data-step=tab1]').getAttribute('aria-current')==='step','The finder opens a step');
  const stepsButton=Array.from(header.querySelectorAll('button')).find(b=>b.textContent==='Steps');const wide=box(stage).width;stepsButton.click();
  assert(document.body.classList.contains('studio-hide-steps') && box(stage).width>wide,'Steps collapse to give the image more room');stepsButton.click();
  assert(!Array.from(document.querySelectorAll('a')).some(a=>a.href.includes('undefined')),'Classic and assistant links are valid');
  const active=state.layers.find(layer=>layer.type==='image'), viewer=state.viewer;
  const step=name=>fetch('/progress?step='+encodeURIComponent(name));step('ask');
  const ask=Array.from(header.querySelectorAll('button')).find(button=>button.textContent==='Ask');
  search.value='show markers '+active.channels[1].name;ask.click();
  await until(()=>!ask.disabled && active.channels[1].visible && !active.channels[0].visible);
  assert(state.viewer===viewer,'Marker commands keep the image viewer open');
  assert(!document.querySelector('#studio-assistant').hidden,'Answers open the Assistant tab beside the image');
  const undo=Array.from(document.querySelectorAll('#studio-action-status button')).find(button=>button.textContent==='Undo');
  step('undo');assert(undo,'The completed action offers Undo');undo.click();
  await until(()=>active.channels[0].visible && !active.channels[1].visible);
  assert(state.viewer===viewer,'Undo keeps the image viewer open');
  step('missing');search.value='show markers MissingMarker';ask.click();await until(()=>!ask.disabled);
  assert(document.querySelector('#studio-action-status').textContent.includes('No matching marker'),'Unknown names get a plain explanation');
  assert(active.channels[0].visible && !active.channels[1].visible,'An invalid command leaves the view unchanged');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks, image:state.spec.title})));
 } catch(error) {fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}
},100);
</script>
"""


# The viewer's parts share their names on the page; this hands the checks what they need.
EXPOSE = "<script>window.__state=state;</script>"

def main():
    with tempfile.TemporaryDirectory(prefix="spacemap-studio-browser-") as folder:
        root = Path(folder)
        os.environ.update(SPACEMAP_HOME=str(root / "state"), MPLCONFIGDIR=str(root / "mpl"), HOME=str(root / "home"))
        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse, Response
        from starlette.routing import Route, Mount
        import uvicorn
        from spacemap_gui.viewer import api, registry
        from spacemap_gui.viewer.component import FRONTEND_DIR
        from test_viewer import _write_channel_pages

        roi = root / "outputs/rois"
        roi.mkdir(parents=True)
        for name in ("first", "second"):
            _write_channel_pages(roi / (name + ".ome.tiff"), channels=2, height=256, width=256)
        registry.publish(
            "studio-browser",
            {"id": "empty", "session": "studio-browser", "layers": []},
            project={"root": str(root), "tab": "tab0", "mappings": {}, "unified": None, "colors": {}},
        )
        done = threading.Event()
        results = []

        async def page(request):
            shim = "<script>window.requestAnimationFrame=cb=>setTimeout(()=>cb(performance.now()),16);window.cancelAnimationFrame=clearTimeout;</script>"
            html = (FRONTEND_DIR / "index.html").read_text().replace("<head>", "<head>" + shim)
            return HTMLResponse(html.replace("</body>", EXPOSE + DRIVER + '<img src="/slow" hidden></body>'))

        async def result(request):
            results.append(dict(request.query_params))
            done.set()
            return Response()

        steps_reached = []

        async def progress(request):
            steps_reached.append(request.query_params.get("step", ""))
            return Response()

        async def slow(request):
            await asyncio.sleep(20)
            return Response(b"")

        app = Starlette(
            routes=[
                Route(api.MOUNT_PATH + "/window/index.html", page),
                Route("/result", result),
                Route("/progress", progress),
                Route("/slow", slow),
                Mount(api.MOUNT_PATH, api.app),
            ]
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
        profile = root / "firefox"
        profile.mkdir()
        (profile / "user.js").write_text('user_pref("network.proxy.type",0);\n')
        with (root / "firefox.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "firefox",
                    "--headless",
                    "--no-remote",
                    "--profile",
                    str(profile),
                    "--window-size",
                    "1600,1100",
                    "--screenshot",
                    "/tmp/spacemap-studio-preview.png",
                    f"http://127.0.0.1:{port}{api.MOUNT_PATH}/window/index.html?studio=1&session=studio-browser",
                ],
                stdout=log,
                stderr=log,
            )
            try:
                # Steps the page reported (step()) show how far a timed-out run got.
                assert done.wait(180), "Browser timed out after " + " > ".join(steps_reached)
                assert results and "ok" in results[0], results
                print(json.dumps(results), flush=True)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                process.terminate()
                process.wait(timeout=10)
                server.should_exit = True


if __name__ == "__main__":
    main()
