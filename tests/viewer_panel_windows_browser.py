"""Independent panel collapse and real browser windows, sharing live viewer state."""
from viewer_threshold_browser import main

DRIVER=r'''<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(const end=Date.now()+60000;Date.now()<end;){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=0;const check=(v,m)=>{if(!v)throw Error(m);checks++;fetch('/result?progress='+encodeURIComponent(m));};
const button=(doc,text)=>[...doc.querySelectorAll('button')].find(b=>b.textContent===text);
const change=(e,v)=>{e.value=v;e.dispatchEvent(new Event('change',{bubbles:true}));};
window.addEventListener('error',e=>fetch('/result?error='+encodeURIComponent(e.message+' '+e.filename+':'+e.lineno)));
setTimeout(async()=>{try{
 await until(()=>window.__state?.imagesReady&&window.__state?.cellsReady&&document.querySelector('[data-pane-toggle=tools]')&&document.querySelector('[aria-label="Phenotyping method"]'));
 const s=window.__state,app=document.querySelector('#app'),stage=document.querySelector('#stage');
 app.style.setProperty('--studio-tools','355px');app.style.setProperty('--studio-right','365px');
 await delay(100);const width=stage.clientWidth,tools=document.querySelector('#studio-tools'),content=tools.firstElementChild;
 for(const key of ['steps','tools','right']){
   const btn=document.querySelector('[data-pane-collapse='+key+']');btn.click();await delay(30);
   check(document.querySelector('#studio-'+key).hidden&&btn.getAttribute('aria-expanded')==='false','Collapse '+key+' independently');
   const restore=document.querySelector('[data-pane-toggle='+key+']'), rect=restore.getBoundingClientRect();
   check(!restore.hidden&&/^.[ ]Show /.test(restore.textContent)&&rect.width>70&&rect.height>0&&document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2)===restore,'A labelled restore button stays visible and clickable for '+key);
 }
 check(stage.clientWidth>width+800,'Collapsing panels gives their space to the viewer');
 check(['steps','tools','right'].every(k=>localStorage.getItem('spacemap-studio-pane-'+k)==='closed'),'Collapsed preferences are remembered');
 for(const key of ['steps','tools','right'])document.querySelector('[data-pane-toggle='+key+']').click();
 await delay(50);
 check(tools.firstElementChild===content&&tools.getBoundingClientRect().width===355&&document.querySelector('#studio-right').getBoundingClientRect().width===365,'Expanding preserves controls and resized widths');
 document.querySelector('[data-pane-collapse=right]').click();document.querySelector('#studio-header [title="Background jobs"]').click();
 check(!document.querySelector('#studio-right').hidden,'Opening Jobs reveals a collapsed right dock');
 showDataTab();await until(()=>dataView.result?.rows.length===40);
 dataView.sort={col:'M0',desc:true};await dataLoad();
 const holder=dataView.holder,selectedSource=dataView.source;
 const realOpen=window.open;window.open=()=>null;
 button(splitMarkers.panel,'Float').click();window.open=realOpen;
 check(!viewerPanelWindow('data')&&holder.ownerDocument===document&&splitMarkerOptions().tab==='data','A blocked popup keeps Data docked without losing state');
 button(splitMarkers.panel,'Float').click();const dw=viewerPanelWindow('data');
 await until(()=>dw?.document.querySelector('.data-grid'));
 check(holder.ownerDocument===dw.document&&!document.querySelector('.data-view'),'Data moves into a real separate window');
 check(!dw.document.querySelector('.split-marker-grid')&&dataView.sort.desc&&dataView.source===selectedSource,'Data window contains only Data and preserves sorting and source');
 check(dw.document.querySelector('.data-scroll').clientHeight>100,'Detached Data uses the window height');
 openSplitTab('markers');button(splitMarkers.panel,'Float').click();const mw=viewerPanelWindow('markers');
 await until(()=>mw?.document.querySelector('.split-marker-tile'));
 check(mw!==dw&&!mw.document.querySelector('.data-view')&&viewerPanelWindows.size===2,'Marker tiles and Data float independently');
 const image=s.layers.find(l=>l.type==='image'&&!l.info.rgb),p=image.item.imageToViewerElementCoordinates(new OpenSeadragon.Point(200,200));
 followSplitMarkerCursor(p);
 await until(()=>mw.document.querySelector('.split-marker-tile')?.dataset.center?.split(',').every(v=>Math.abs(Number(v)-200)<.01));
 check(mw.document.querySelector('.split-marker-tile canvas').width>0,'Detached marker tiles continue rendering the live cursor');
 change(mw.document.querySelector('[aria-label="Marker tile zoom"]'),'4');
 check(splitMarkerOptions().zoom===4,'Detached marker controls update the shared viewer settings');
 mw.document.querySelector('.split-marker-grid').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:150,clientY:150}));
 check(mw.document.querySelector('#ctxmenu')&&!document.querySelector('#ctxmenu'),'Tile menus open in the marker window');closeContextMenu();
 dw.document.querySelector('[aria-label="Filter M0"]').click();
 await until(()=>dw.document.querySelector('.data-filter-menu [aria-label=From]'));
 check(!document.querySelector('.data-filter-menu'),'Data filter menus open in the Data window');
 const from=dw.document.querySelector('[aria-label=From]');from.value='200';from.dispatchEvent(new Event('input',{bubbles:true}));
 await until(()=>dataView.result.matched===20);closeContextMenu();
 const cells=dataLinkedLayer();
 check(cells.filters.some(f=>f.col==='M0'&&f.lo===200),'Detached Data filters still update the main image');
 dw.document.querySelector('.data-grid tbody tr').click();
 await until(()=>s.cellOutline?.length);
 check(dw.document.querySelector('tr.selected'),'Detached row selection outlines the main image cell');
 const cell=dw.document.querySelector('td[data-col="phenotype_training"]');cell.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
 const edit=dw.document.querySelector('.data-edit');edit.value='Window edit';edit.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}));
 await until(()=>dataView.result.edited===1);
 dataEditDialog({kind:'rename_column',column:'M0'});
 check(dw.document.querySelector('dialog[open]'),'Data edit dialogs stay in the Data window');dw.document.querySelector('dialog').close();
 button(dw.document,'Dock back').click();
 check(!viewerPanelWindow('data')&&viewerPanelWindow('markers')===mw&&holder.ownerDocument===document,'Data docks back without closing Marker tiles');
 check(dataView.result.edited===1&&dataView.filters[0].lo===200&&dataView.holder===holder,'Docking preserves unsaved edits and filters');
 mw.close();await until(()=>!viewerPanelWindow('markers')&&splitMarkers.grid.ownerDocument===document);
 check(splitMarkerOptions().tab==='markers'&&splitMarkerOptions().zoom===4,'Closing a marker window docks it back with its settings');
 showDataTab();button(splitMarkers.panel,'Float').click();const second=viewerPanelWindow('data');second.close();
 await until(()=>dataView.holder.ownerDocument===document);
 check(dataView.result.edited===1,'Closing the Data window preserves pending edits');
 await dataHistory('undo');
 check(dataView.result.edited===0,'The restored Data view can undo its window edit');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
 }catch(e){fetch('/result?error='+encodeURIComponent(e.message+'\n'+e.stack));}},100);
</script>'''

if __name__=='__main__':
    main(DRIVER,studio=True,screenshot=False,allow_popups=True,timeout=180)
