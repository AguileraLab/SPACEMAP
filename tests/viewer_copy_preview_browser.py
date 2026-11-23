"""Actual PNG clipboard round trip plus HTTP/permission fallback in disposable Firefox."""
from viewer_threshold_browser import main

DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const assert=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
const bitmapCanvas=async blob=>{const bitmap=await createImageBitmap(blob),c=document.createElement('canvas');c.width=bitmap.width;c.height=bitmap.height;c.getContext('2d').drawImage(bitmap,0,0);bitmap.close();return c;};
const samePixels=(a,b)=>{if(a.width!==b.width||a.height!==b.height)return false;const x=a.getContext('2d').getImageData(0,0,a.width,a.height).data,y=b.getContext('2d').getImageData(0,0,b.width,b.height).data;return x.every((v,i)=>v===y[i]);};
setTimeout(async()=>{try{
 const state=window.__state;
 await until(()=>state.layers.filter(l=>l.type==='image'||l.type==='cells').every(l=>l.item?.getFullyLoaded())&&scaleBarModel());
 await delay(300);
 const source=state.viewer.drawer.canvas,preview=viewCanvas(true),withoutScale=viewCanvas(false);
 assert(preview.width===source.width&&preview.height===source.height,'Preview captures the viewer canvas at its displayed pixel resolution');
 assert(!samePixels(preview,withoutScale),'The scale bar is drawn into the copied image');
 const toolbar=document.getElementById('toolbar');toolbar.style.background='#ff00ff';
 assert(samePixels(preview,viewCanvas(true)),'Viewer toolbar styling and buttons do not enter the capture');
 document.getElementById('btn-copy-preview').click();
 await until(()=>document.getElementById('message').textContent.startsWith('Preview copied'));
 const items=await navigator.clipboard.read(),png=await items[0].getType('image/png');
 assert(png.type==='image/png'&&samePixels(preview,await bitmapCanvas(png)),'Toolbar action writes a real PNG to the clipboard with the exact Save image pixels');
 const bounds=state.viewer.element.getBoundingClientRect();setTool('off');
 state.viewer.element.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:bounds.right-5,clientY:bounds.top+100}));
 await until(()=>document.querySelector('#ctxmenu'));
 const menuCopy=[...document.querySelectorAll('#ctxmenu button')].find(b=>b.textContent==='Copy preview');
 assert(menuCopy,'Image right-click menu includes Copy preview');
 const before=await navigator.clipboard.read();menuCopy.click();
 await delay(150);const menuPng=await (await navigator.clipboard.read())[0].getType('image/png');
 assert(samePixels(preview,await bitmapCanvas(menuPng)),'Context-menu action uses the same clean PNG with scale bar');
 Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});
 document.getElementById('btn-copy-preview').click();
 await until(()=>document.querySelector('#copy-preview-dialog img')?.naturalWidth>0);
 let dialog=document.getElementById('copy-preview-dialog'),picture=dialog.querySelector('img');
 const fallback=await fetch(picture.src).then(r=>r.blob());
 assert(samePixels(preview,await bitmapCanvas(fallback)),'An HTTP browser gets the same image for native Copy Image');
 const event=new MouseEvent('contextmenu',{bubbles:true,cancelable:true});picture.dispatchEvent(event);
 assert(!event.defaultPrevented,'Fallback image allows the native browser copy-image menu');
 assert(!document.getElementById('message').textContent.startsWith('Preview copied'),'Fallback does not claim the clipboard write succeeded');
 dialog.close();await until(()=>!document.getElementById('copy-preview-dialog'));
 Object.defineProperty(navigator,'clipboard',{configurable:true,value:{write:()=>Promise.reject(new DOMException('Denied','NotAllowedError'))}});
 document.getElementById('btn-copy-preview').click();
 await until(()=>document.querySelector('#copy-preview-dialog img')?.naturalWidth>0);
 assert(true,'Clipboard permission denial also offers native image copying');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''

if __name__=='__main__':
    main(DRIVER, timeout=120, clipboard=True, headless=False, screenshot=False)
