"""Browser-check Studio's assistant tab on a disposable project in headless Firefox (no model: common requests).

Usage: PYTHONPATH=src:tests python tests/viewer_studio_assistant_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(await check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && document.querySelector('[data-tab=assistant]'));
  const pane=()=>document.querySelector('#studio-assistant');
  document.querySelector('[data-tab=assistant]').click();
  await until(()=>/Common requests|Local|Online/.test(pane().textContent) && pane().querySelector('.studio-bubble.assistant'));
  assert(/Explicit requests run here/.test(pane().textContent),'The assistant greets with what it can do');
  const input=pane().querySelector('[aria-label="Message the assistant"]');
  input.value='show markers M1';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>pane().querySelector('.studio-action'));
  assert(/I can do this: show markers/.test(pane().textContent),'A common request becomes a checked action');
  const image=state.layers.find(layer=>layer.type==='image');
  await until(()=>pane().querySelector('.studio-action-result p') && image.channels.filter(c=>c.visible).map(c=>c.name).join()==='M1');
  assert(true,'Looking at markers changes nothing in the project, so it runs at once');
  // The Ask box in the header talks to the same assistant.
  const search=document.querySelector('[aria-label="Find a step or image"]');search.value='what is the tissue made of?';
  Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Ask').click();
  await until(()=>/set up a model in AI setup/i.test(pane().textContent));
  assert(Array.from(pane().querySelectorAll('.studio-bubble.user')).some(b=>/tissue made of/.test(b.textContent)),'Ask sends the question to the chat');
  const memory=pane().querySelector('.studio-memories');memory.open=true;
  const add=memory.querySelector('[aria-label="New memory"]');add.value='Our tumour marker is PanCK.';
  button(memory,'Remember').click();
  await until(()=>/Memory \(1\)/.test(memory.textContent) && /PanCK/.test(memory.textContent));
  button(memory,'Forget').click();
  await until(()=>/Memory \(0\)/.test(memory.textContent));
  assert(true,'Memories can be added and forgotten');
  // AI setup opens here, not in classic: the models, the memory mode and stopping with SPACEMAP.
  assert(!Array.from(pane().querySelectorAll('a')).some(a=>/Set up AI/.test(a.textContent)),'No link sends the user to classic for AI setup');
  button(pane(),'AI setup').click();
  const setup=()=>pane().querySelector('.studio-ai-setup');
  await until(()=>setup() && !setup().hidden && setup().querySelector('[aria-label="Stop the model when SPACEMAP stops"]'));
  assert(!/null|undefined/.test(setup().textContent) && /Local model/.test(setup().textContent) && setup().querySelectorAll('.studio-folder').length>=1 && /Online model instead/.test(setup().textContent),'AI setup lists the local models and the optional online one');
  const stopWith=setup().querySelector('[aria-label="Stop the model when SPACEMAP stops"]');
  assert(stopWith.checked,'The model stops with SPACEMAP by default');
  stopWith.click();
  await until(async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/ai')).json()).stop_with_spacemap===false);
  const mode=setup().querySelector('[aria-label="GPU memory mode"]');mode.value='int8';mode.dispatchEvent(new Event('change'));
  await until(async()=>(await (await fetch('/spacemap-viewer/api/studio/browser-check/ai')).json()).precision==='int8');
  assert(true,'Settings changed in Studio are saved');
  // A picture attached (as a pasted screenshot is: Firefox empties the clipboard of a paste a script makes, so the
  // check attaches through the file picker, which takes the same path) goes with the next question.
  const canvas=document.createElement('canvas');canvas.width=60;canvas.height=40;canvas.getContext('2d').fillRect(0,0,30,20);
  const blob=await new Promise(done=>canvas.toBlob(done,'image/png'));
  const clip=new DataTransfer();clip.items.add(new File([blob],'shot.png',{type:'image/png'}));
  const picker=pane().querySelector('[aria-label="Attach images"]');picker.files=clip.files;picker.dispatchEvent(new Event('change'));
  await until(()=>pane().querySelectorAll('.studio-attachments img').length===1);
  assert(true,'An attached screenshot waits for the next question');
  input.value='what is in this picture?';input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  await until(()=>/cannot look at images/.test(pane().textContent));
  assert(pane().querySelectorAll('.studio-bubble.user .studio-attachments img').length===1 && !pane().querySelector('.studio-attachments:not(.studio-bubble *) img'),
    'The picture shows in the question and leaves the attachment strip');
  const conversation=localStorage.getItem('spacemap-studio-chat');
  const again=await (await fetch('/spacemap-viewer/api/studio/browser-check/chat?conversation='+conversation)).json();
  assert(again.messages.length>=4 && again.messages.some(m=>m.action && m.action.name==='show_markers'),'The conversation is kept, with its proposed action');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-assistant')?.textContent.slice(-600)));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300)
