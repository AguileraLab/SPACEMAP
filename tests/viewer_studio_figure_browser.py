"""Browser-check Studio's figure (Visualization) on a disposable project in headless Firefox.

Usage: PYTHONPATH=src:tests python tests/viewer_studio_figure_browser.py
"""

from viewer_threshold_browser import main

DRIVER = r"""
<script>
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const until=async check=>{for(let i=0;i<2400;i++){if(check())return;await delay(50);}throw Error('Timed out: '+check);};
let checks=0;const assert=(condition,message)=>{if(!condition)throw Error(message);checks++;};
const button=(container,text)=>Array.from(container.querySelectorAll('button')).find(item=>item.textContent===text);
// Catch the download: the figure's file name and its picture.
const saved=[];
HTMLAnchorElement.prototype.click=function(){saved.push({name:this.download,href:this.href});};
const size=href=>new Promise(resolve=>{const image=new Image();image.onload=()=>resolve([image.width,image.height]);image.src=href;});
setTimeout(async()=>{
 try{
  const state=window.__state;
  await until(()=>state.imagesReady && state.cellsReady && document.querySelector('[data-step=tab9]'));
  const cells=state.layers.find(layer=>layer.type==='cells');
  cells.color_by='phenotype_ml_consensus';cells.visible=true;
  document.querySelector('[data-step=tab9]').click();
  const tools=()=>document.querySelector('#studio-tools');
  await until(()=>tools().querySelector('[aria-label="Figure title"]'));
  assert(tools().textContent.includes('Classic Tab 9'),'The other figures point to classic Tab 9');
  const title=tools().querySelector('[aria-label="Figure title"]');title.value='Test figure';title.dispatchEvent(new Event('input'));
  button(document.querySelector('.studio-runbar'),'This image').click();
  await until(()=>/Saved a \d+ × \d+ px figure|rror|Open an image/.test(tools().textContent));
  await until(()=>saved.length===1);  // the browser writes the PNG a moment later
  assert(saved.length===1 && saved[0].name.endsWith('_figure.png'),'The figure is downloaded as a PNG: '+JSON.stringify(saved.map(s=>s.name)));
  const [width,height]=await size(saved[0].href);
  const view=state.viewer.drawer.canvas;
  assert(width>view.width && height>=view.height,'The figure adds the title band and the legend beside the view: '+width+'x'+height+' vs '+view.width+'x'+view.height);
  tools().querySelector('[aria-label="Add the legend"]').click();title.value='';title.dispatchEvent(new Event('input'));
  button(document.querySelector('.studio-runbar'),'This image').click();
  await until(()=>saved.length===2);
  const [plainWidth,plainHeight]=await size(saved[1].href);
  assert(plainWidth===view.width && plainHeight===view.height,'Without title and legend it is the view itself');
  const figureSize=tools().querySelector('[aria-label="Figure size"]');figureSize.value='full';figureSize.dispatchEvent(new Event('change'));
  button(document.querySelector('.studio-runbar'),'This image').click();
  await until(()=>saved.length===3);
  assert(saved[2].name.endsWith('_figure.png'),'Full resolution makes a figure too');
  // Selected: each ticked image is opened in turn and saved.
  await until(()=>tools().querySelector('.studio-pairs tr[data-image] input[type=checkbox]'));
  const first=tools().querySelector('.studio-pairs tr[data-image]');
  first.querySelector('input[type=checkbox]').click();
  button(document.querySelector('.studio-runbar'),'Selected').click();
  await until(()=>/Saved 1 figure\./.test(document.querySelector('.studio-runbar').textContent)&&saved.length===4);
  assert(state.imagesReady,'Selected opens each ticked image and saves its figure ('+first.dataset.image+')');
  fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks,figure:[width,height]})));
 }catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack+'\n'+document.querySelector('#studio-tools')?.textContent));}
},100);
</script>
"""


if __name__ == "__main__":
    main(DRIVER, studio=True, timeout=300)
