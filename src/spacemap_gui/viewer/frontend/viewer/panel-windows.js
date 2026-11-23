/* Move live controls into same-origin browser windows. One state and one set of
 * requests remain owned by the viewer; docking never copies or resets edits. */
"use strict";
const viewerPanelWindows = new Map();
const panelWindowOwner = Math.random().toString(36).slice(2);
let closingViewerWindows = false;

function viewerPanelWindow(key) {
  const entry=viewerPanelWindows.get(key);
  return entry && !entry.window.closed ? entry.window : null;
}

function dockViewerPanel(key, select=true) {
  const entry=viewerPanelWindows.get(key);if(!entry)return;
  closeContextMenu();
  viewerPanelWindows.delete(key);clearInterval(entry.watch);
  for(const {node,anchor} of entry.moved)anchor.replaceWith(node);
  if(!entry.window.closed)entry.window.close();
  if(!closingViewerWindows)entry.onDock(select);
}

function floatViewerPanel(key, title, {tools=[], content=[], onDock, onResize}) {
  const existing=viewerPanelWindow(key);if(existing){existing.focus();return existing;}
  // Must happen synchronously in the click handler, before any async work.
  const child=window.open('', 'spacemap-'+key+'-'+panelWindowOwner, 'popup,width=1050,height=720');
  if(!child)return null;
  const doc=child.document;
  doc.open();doc.write('<!doctype html><html><head><meta charset="utf-8"></head><body></body></html>');doc.close();
  doc.title=title+' · SPACEMAP';doc.documentElement.lang='en';doc.body.className='viewer-panel-window';
  for(const original of document.querySelectorAll('link[rel=stylesheet]')){
    const link=doc.createElement('link');link.rel='stylesheet';link.href=original.href;link.onload=onResize;doc.head.append(link);
  }
  const entry={window:child,moved:[],onDock};viewerPanelWindows.set(key,entry);
  const move=(node,parent)=>{const anchor=document.createComment('detached '+key);node.before(anchor);entry.moved.push({node,anchor});parent.append(node);};
  const head=el('header',{class:'split-marker-head'},[el('strong',{text:title})]);
  const body=el('main',{class:'viewer-window-content'});
  doc.body.append(head,body);
  tools.forEach(node=>move(node,head));
  head.append(el('button',{class:'btn small',text:'Dock back',title:'Return '+title+' below the image',onclick:()=>dockViewerPanel(key)}));
  content.forEach(node=>move(node,body));
  installViewerHelp(doc);
  doc.addEventListener('pointerdown',event=>{if(!event.target.closest?.('#ctxmenu'))closeContextMenu();},true);
  doc.addEventListener('keydown',event=>{if(event.key==='Escape')closeContextMenu();});
  child.addEventListener('resize',onResize);
  child.addEventListener('pagehide',()=>dockViewerPanel(key));
  // Handles close as well as browsers that omit pagehide for a blank popup.
  entry.watch=setInterval(()=>{if(child.closed)dockViewerPanel(key);},400);
  child.focus();return child;
}

window.addEventListener('pagehide',()=>{
  closingViewerWindows=true;
  for(const key of [...viewerPanelWindows.keys()])dockViewerPanel(key,false);
});
