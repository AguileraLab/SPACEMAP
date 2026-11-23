"""Exercise a real non-secure HTTP origin, like Studio's network-IP address.

Firefox resolves the test hostname locally; the browser still treats it as a
non-secure origin. Unlike localhost, crypto.randomUUID is actually unavailable.
"""
from viewer_threshold_browser import main


DRIVER = r"""
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<400;i++){if(f())return;await wait(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(value,message)=>{if(!value)throw Error(message);checks++;fetch('/result?progress='+encodeURIComponent(message));};
const openFlow=async query=>{const frame=document.createElement('iframe');frame.style='position:fixed;inset:10px;width:1200px;height:850px;z-index:100000';
  const loaded=new Promise(resolve=>frame.onload=resolve);frame.src='/spacemap-viewer/api/window/flow-gating.html'+query;document.body.append(frame);await loaded;return frame;};
(async()=>{try{
  check(location.protocol==='http:'&&!isSecureContext,'Test uses non-secure HTTP, not localhost');
  check(typeof crypto.randomUUID==='undefined'&&typeof crypto.getRandomValues==='function','Browser exposes the same crypto API restriction as network HTTP');
  const frame=await openFlow('?session=browser-check&api=/spacemap-viewer/api');
  await wait(100);
  check(!!frame.contentDocument.querySelector('.flow-toolbar'),'Detached startup draws controls before analysis on non-secure HTTP');
  await until(()=>frame.contentDocument.querySelector('.flow-chart .main-svg'));
  check(true,'Detached HTTP workspace loads real plot data');
  frame.remove();
  const endpoint='/spacemap-viewer/api/studio/browser-check/flow_gating';
  const info=await(await fetch(endpoint)).json();await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'save',revision:info.workspace.revision,changes:{live:{enabled:true,source:'image'}}})});
  await until(()=>window.__state.cellsReady&&document.querySelector('[aria-label="Phenotyping method"]'));
  window.open=()=>null;const method=document.querySelector('[aria-label="Phenotyping method"]');method.value='flow';method.dispatchEvent(new Event('change'));
  await until(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Dock below'));
  [...document.querySelectorAll('button')].find(b=>b.textContent==='Dock below').click();
  await until(()=>document.querySelector('.flow-docked .flow-plot-foot')?.textContent.includes('in live scope'));
  check(true,'Docked startup restores Live context after the controller is ready');
  const invalid=await openFlow('');
  await until(()=>invalid.contentDocument.querySelector('[role="alert"]'));
  check(invalid.contentDocument.querySelector('[role="alert"]').textContent.includes('Studio'),'Missing session gives a visible recovery message');
  invalid.remove();
  const unavailable=await openFlow('?session=missing-session&api=/spacemap-viewer/api');
  await until(()=>unavailable.contentDocument.querySelector('.flow-error'));
  check(!!unavailable.contentDocument.querySelector('.flow-error').textContent,'Backend startup failure stays visible instead of a blank window');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}})();
</script>
"""

if __name__ == '__main__':
    main(driver=DRIVER,studio=True,timeout=150,screenshot=False,browser_host='flow-gating.test')
