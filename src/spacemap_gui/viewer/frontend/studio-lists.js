/* Project files, independent of the active workflow step. No analysis settings. */
(() => {
  'use strict';
  window.createStudioLists = ({holder, node, help, get, post, api, route, open, changed, imageChoices}) => {
    const sections = new Map(), panels = new Map();
    let loaded = false, readOnly = false, navigation = Promise.resolve(), navigationId = 0;
    const status = node('span', {class:'lists-status', role:'status'});
    const refresh = node('button', {class:'btn small ghost', text:'↻', title:'Refresh Lists', 'aria-label':'Refresh Lists', onclick:() => invalidate()});
    holder.append(node('div', {class:'lists-toolbar'}, [node('strong', {text:'Project files'}),
      help('Inputs, outputs and imports for every Studio step. Expand a list to select, import, export or remove files. Hover a file for its full path; right-click to copy it. Remove only unlists items from this Studio project. Files and sidecars stay at their original paths.'), refresh, status]));
    const tree = node('div', {class:'lists-tree'}); holder.append(tree);
    const stop = element => { for (const name of ['click','keydown']) element.addEventListener(name, e => e.stopPropagation()); return element; };
    const say = error => { status.textContent = error?.message || String(error || ''); };
    const query = (step, group) => route('/lists') + '?' + new URLSearchParams({step, ...(group ? {group} : {})});
    const call = (action, panel, extra={}) => post(route('/lists'), {action, step:panel.step, group:panel.group.id, ...extra});
    const checked = panel => panel.items.filter(item => panel.picked.has(item.id));
    function contextMenu(event, path) {
      event.preventDefault(); event.stopPropagation();
      document.querySelector('.lists-context')?.remove();
      const menu = node('div', {class:'lists-context', role:'menu'});
      const button = node('button', {class:'btn small', role:'menuitem', text:'Copy path', onclick:async () => {
        try { await navigator.clipboard.writeText(path); say('Path copied.'); }
        catch (_) { const input=node('textarea'); input.value=path; document.body.append(input); input.select(); document.execCommand('copy'); input.remove(); say('Path copied.'); }
        menu.remove();
      }});
      menu.append(button); document.body.append(menu);
      menu.style.left = Math.min(event.clientX, window.innerWidth - menu.offsetWidth - 8) + 'px';
      menu.style.top = Math.min(event.clientY, window.innerHeight - menu.offsetHeight - 8) + 'px';
      button.focus();
      setTimeout(() => document.addEventListener('pointerdown', e => {if (!menu.contains(e.target)) menu.remove();}, {once:true}),0);
      menu.onkeydown=e=>{if(e.key==='Escape')menu.remove();};
    }
    function openItem(panel, item, annotation) {
      if(item.kind==='annotation'&&!item.image){matchDialog(panel,item);return;}
      const request = ++navigationId;
      navigation = navigation.catch(()=>{}).then(async () => {
        if(request !== navigationId)return;
        try {
          say('Opening…');
          const result = await call('open', panel, {ids:[item.id]});
          if(request !== navigationId)return;
          await open(panel.step, result.item, annotation);
          if(request === navigationId)say('');
        } catch(error) {say(error);}
      });
    }
    function draw(panel) {
      const filter=panel.search.value.trim().toLowerCase();
      const shown=panel.items.filter(item => !filter || [item.name,item.path,item.image,...(item.classes||[]).map(c=>c.name)].join(' ').toLowerCase().includes(filter));
      const selected=checked(panel);
      panel.all.checked=!!shown.length&&shown.every(item=>panel.picked.has(item.id));
      panel.all.indeterminate=shown.some(item=>panel.picked.has(item.id))&&!panel.all.checked;
      panel.all.disabled=!shown.length;
      panel.export.disabled=!selected.length;
      panel.remove.disabled=readOnly||!selected.length||selected.some(item=>!item.removable);
      panel.count.textContent=(selected.length ? selected.length+' / ' : '')+panel.items.length.toLocaleString();
      panel.count.title=selected.length+' selected · '+panel.items.length+' items';
      panel.rows.replaceChildren();
      // Render a bounded page while select-all acts on every search match.
      shown.slice(0,panel.limit).forEach(item=>{
        const box=node('input',{type:'checkbox','aria-label':'Select '+item.name}); box.checked=panel.picked.has(item.id);
        box.onchange=()=>{box.checked?panel.picked.add(item.id):panel.picked.delete(item.id);draw(panel);};
        const name=node('button',{class:'lists-name',text:item.name,title:item.path,onclick:()=>openItem(panel,item)});
        const badges=node('span',{class:'lists-badges'});
        (item.classes||[]).slice(0,12).forEach(cls=>{
          const label=cls.name+(cls.count==null?'':' · '+Number(cls.count).toLocaleString());
          badges.append(node('button',{class:'lists-class',text:cls.name.slice(0,1).toUpperCase(),title:label,
            style:'--class-colour:'+(/^#[0-9a-f]{6}$/i.test(cls.colour||'')?cls.colour:'#7eb3ef'),
            'aria-label':label,onclick:()=>openItem(panel,item,cls.name)}));
        });
        if((item.classes||[]).length>12)badges.append(node('span',{class:'muted',text:'+'+(item.classes.length-12),title:item.classes.slice(12).map(c=>c.name+(c.count==null?'':' · '+c.count)).join('\n')}));
        const row=node('div',{class:'lists-file', 'data-list-item':item.id, title:item.path, oncontextmenu:e=>contextMenu(e,item.path)},[
          box,name,item.copy?node('span',{class:'lists-copy',text:item.copy,title:(item.linked?'Linked original · ':'Imported copy · ')+item.path}):null,
          item.missing?node('span',{class:'lists-error',text:'Missing',title:'The linked file is no longer at its saved path'}):
            item.kind==='annotation'&&!item.image?node('button',{class:'btn small ghost',text:'Match…',title:'No matching image; choose one',disabled:readOnly,onclick:()=>matchDialog(panel,item)}):null,badges,
          item.images!=null?node('span',{class:'lists-number',text:item.images+' img',title:item.images+' processed images'}):null,
          item.count!=null?node('span',{class:'lists-number',text:Number(item.count).toLocaleString(),title:item.kind==='annotation'?'Polygons':'Assigned cells'}):null
        ]);
        panel.rows.append(row);
      });
      if(!shown.length)panel.rows.append(node('span',{class:'muted lists-empty',text:filter?'No matches':'No files'}));
      if(shown.length>panel.limit)panel.rows.append(node('button',{class:'btn small',text:'More ('+(shown.length-panel.limit)+')',onclick:()=>{panel.limit+=200;draw(panel);}}));
      if(panel.done){panel.done.textContent=panel.items.some(i=>i.count>0)?'✓':'—';panel.done.title=panel.items.some(i=>i.count>0)?'Saved assignments':'No saved assignments';}
    }
    async function loadPanel(panel, force=false) {
      if(panel.loading)return panel.loading;
      if(panel.loaded&&!force)return;
      const revision=panel.revision;
      panel.rows.setAttribute('aria-busy','true');
      panel.loading=(async()=>{try{
        const result=await get(query(panel.step,panel.group.id));
        if(revision!==panel.revision)return;
        panel.items=result.items; panel.loaded=true;
        const ids=new Set(panel.items.map(i=>i.id));panel.picked=new Set([...panel.picked].filter(id=>ids.has(id)));
        draw(panel);
      }catch(error){panel.rows.replaceChildren(node('span',{class:'lists-error',text:error.message}));}
      finally{panel.rows.removeAttribute('aria-busy');panel.loading=null;}})();
      await panel.loading;
      if(revision!==panel.revision&&panel.box.open)await loadPanel(panel);
    }
    async function perform(panel, action) {
      const items=checked(panel);if(!items.length)return;
      if(action==='remove'&&!window.confirm('Remove '+items.length+' selected item'+(items.length===1?'':'s')+' from the project? Files stay unchanged at their original paths.'))return;
      panel.actions.inert=true;
      try{
        const result=await call(action,panel,{ids:items.map(i=>i.id)});
        if(result.download){const link=node('a',{href:api(result.download),download:'',hidden:true});document.body.append(link);link.click();link.remove();}
        else{panel.picked.clear();await changed(result);await invalidate();say(result.removed+' removed.');}
      }catch(error){say(error);}finally{panel.actions.inert=false;}
    }
    async function matchDialog(panel,item) {
      if(readOnly)return;
      const dialog=node('dialog',{class:'lists-import-dialog'}),error=node('span',{class:'lists-error',role:'status'});
      const images=node('select',{'aria-label':'Match imported annotation'},[node('option',{value:'',text:'Choose image…'})]);
      for(const image of await imageChoices())images.append(node('option',{value:image.path||image.name,text:image.name}));
      const submit=node('button',{class:'btn small primary',text:'Match',onclick:async()=>{
        submit.disabled=true;
        try{await call('match',panel,{ids:[item.id],image:images.value});await changed({assigned:true});await invalidate();dialog.close();}
        catch(e){error.textContent=e.message;}finally{submit.disabled=false;}
      }});
      dialog.append(node('form',{method:'dialog'},[node('div',{class:'lists-import-head'},[node('strong',{text:item.name}),node('button',{class:'btn small',text:'×','aria-label':'Close match'})])]),
        node('div',{class:'lists-import-controls'},[images,submit]),error);
      dialog.addEventListener('close',()=>dialog.remove());document.body.append(dialog);dialog.showModal();
    }
    async function importDialog(panel, dropped=[]) {
      if(readOnly)return;
      const dialog=node('dialog',{class:'lists-import-dialog'});
      const picker=node('input',{type:'file',multiple:true,hidden:true,accept:panel.group.id==='imports'?'.geojson,.json':panel.group.role==='model'?'.zip,.joblib,.pt,.pth':(panel.group.suffixes||[]).join(',')});
      const paths=node('textarea',{rows:2,placeholder:'Full file paths, one per line (or a folder)',title:'Folders include direct files only. Subdirectories are ignored.','aria-label':'Import file paths'});
      const selected=node('span',{class:'muted'}),error=node('span',{role:'status',class:'lists-error'});
      const problems=node('details',{class:'lists-import-problems',hidden:true}),problemRows=node('div');
      problems.append(node('summary',{text:'File details'}),problemRows);
      let files=Array.from(dropped);
      const update=()=>{selected.textContent=files.length?files.length+' file'+(files.length===1?'':'s'):'';};update();
      const choose=node('button',{class:'btn small',text:'Choose files…',onclick:()=>picker.click()});
      picker.onchange=()=>{files=Array.from(picker.files);update();};
      const images=node('select',{'aria-label':'Import annotation image',title:'Match each file to its image. Folder imports always match each filename independently.'},[node('option',{value:'',text:'Match image names'})]);
      if(panel.group.id==='imports')for(const image of await imageChoices())images.append(node('option',{value:image.name,text:image.name}));
      const submit=node('button',{class:'btn small primary',text:'Import',onclick:async()=>{
        submit.disabled=true;error.textContent='';
        let imported=0,skipped=0;const failed=[],unmatched=[];
        const received=found=>{imported+=found.added.length;skipped+=(found.skipped||[]).length;failed.push(...(found.errors||[]).map(e=>e.name+': '+e.error));unmatched.push(...(found.unmatched||[]));};
        try{
          if(files.length){for(const file of files){
            try{
              const url=route('/lists/upload')+'?'+new URLSearchParams({step:panel.step,group:panel.group.id,name:file.name,image:images.value});
              const response=await fetch(api(url),{method:'POST',body:file});const found=await response.json();
              if(!response.ok)throw Error(found.error||'Import failed');received(found);
            }catch(e){failed.push(file.name+': '+e.message);}
          }}else{
            received(await call('import',panel,{paths:paths.value.split('\n').map(p=>p.trim()).filter(Boolean),image:images.value}));
          }
          const message=imported+' imported'+(skipped?' · '+skipped+' empty files skipped':'')+(unmatched.length?' · '+unmatched.length+' need an image match':'')+(failed.length?' · '+failed.length+' failed':'')+'.';
          say(message);status.title=[...unmatched.map(n=>n+': no matching image'),...failed].join('\n');
          if(failed.length){error.textContent=message;problems.hidden=false;problemRows.replaceChildren(...failed.map(text=>node('div',{text})));files=[];paths.value='';update();}
          else dialog.close();
        }catch(e){const text=e.message||String(e);error.textContent=(imported?imported+' imported. ':'')+(text.length>200?'Import could not finish. See file details.':text);if(text.length>200){problems.hidden=false;problemRows.replaceChildren(node('div',{text}));}}
        finally{submit.disabled=false;if(imported){await changed({added:true});await invalidate();}}
      }});
      dialog.append(node('form',{method:'dialog'},[node('div',{class:'lists-import-head'},[node('strong',{text:'Import · '+panel.group.name}),node('button',{class:'btn small',text:'×','aria-label':'Close import'})])]),
        node('div',{class:'lists-import-controls'},[choose,picker,selected,panel.group.id==='imports'?images:null,submit]),paths,error,problems);
      dialog.addEventListener('close',()=>dialog.remove());
      document.body.append(dialog);dialog.showModal();
    }
    function groupPanel(step, group, parent) {
      const key=step+':'+group.id;
      const panel={step,group,picked:new Set(),items:[],limit:200,revision:0,loaded:false};
      panel.box=node('details',{class:'lists-group','data-list-group':group.id});
      panel.count=node('span',{class:'lists-number',text:''});
      panel.search=node('input',{type:'search',placeholder:'Find…','aria-label':'Find in '+group.name});
      panel.search.oninput=()=>{panel.limit=200;draw(panel);};
      panel.all=node('input',{type:'checkbox','aria-label':'Select matching '+group.name});
      panel.all.onchange=()=>{const text=panel.search.value.toLowerCase();panel.items.filter(i=>[i.name,i.path,i.image,...(i.classes||[]).map(c=>c.name)].join(' ').toLowerCase().includes(text)).forEach(i=>panel.all.checked?panel.picked.add(i.id):panel.picked.delete(i.id));draw(panel);};
      panel.remove=node('button',{class:'btn small ghost',text:'Remove',disabled:true,onclick:()=>perform(panel,'remove')});
      panel.export=node('button',{class:'btn small ghost',text:'Export',disabled:true,onclick:()=>perform(panel,'export')});
      panel.actions=stop(node('span',{class:'lists-actions'},[panel.all,panel.search,
        group.importable?node('button',{class:'btn small ghost',text:'Import',disabled:readOnly,onclick:()=>importDialog(panel)}):null,panel.export,panel.remove]));
      panel.done=group.method&&group.role!=='model'?node('button',{class:'lists-done',text:'…',title:'Expand to check saved assignments · Open '+group.name,onclick:e=>{e.preventDefault();e.stopPropagation();open(step,{method:group.method});}}):null;
      panel.rows=node('div',{class:'lists-rows'});
      const title=node('span',{class:'lists-title',text:group.name});
      panel.box.append(node('summary',{},[title,group.role==='import'?node('span',{class:'lists-origin',text:'Imports'}):group.role==='output'&&!group.method?node('span',{class:'lists-origin',text:'Outputs'}):null,
        panel.done,panel.count,panel.actions]),panel.rows);
      panel.box.ontoggle=()=>{if(panel.box.open)loadPanel(panel);};
      if(group.importable){
        panel.box.ondragover=e=>{e.preventDefault();e.stopPropagation();panel.box.classList.add('lists-drop');};
        panel.box.ondragleave=()=>panel.box.classList.remove('lists-drop');
        panel.box.ondrop=e=>{e.preventDefault();e.stopPropagation();panel.box.classList.remove('lists-drop');const text=e.dataTransfer.getData('text/plain');if(e.dataTransfer.files.length)importDialog(panel,e.dataTransfer.files);else{importDialog(panel).then(()=>{const field=document.querySelector('.lists-import-dialog textarea');if(field)field.value=text;});}};
      }
      panels.set(key,panel);parent.append(panel.box);return panel;
    }
    async function loadStep(section) {
      if(section.loaded)return;
      section.loaded=true;
      try{
        const result=await get(query(section.id));
        for(const group of result.groups){
          const parent=group.parent?panels.get(section.id+':'+group.parent)?.box:section.body;
          groupPanel(section.id,group,parent||section.body);
        }
      }catch(error){section.loaded=false;say(error);}
    }
    async function show() {
      if(loaded)return;
      loaded=true;
      try{
        const result=await get(route('/lists'));readOnly=result.read_only;
        for(const step of result.steps){
          const box=node('details',{class:'lists-step','data-list-step':step.id});
          const body=node('div',{class:'lists-step-body'});
          const section={id:step.id,box,body,loaded:false};sections.set(step.id,section);
          box.append(node('summary',{},[node('strong',{text:step.name})]),body);
          box.ontoggle=()=>{if(box.open)loadStep(section);};tree.append(box);
        }
      }catch(error){loaded=false;say(error);}
    }
    async function invalidate() {
      // Keep search, selection, disclosure and scroll; refresh only visible groups.
      for(const panel of panels.values()){panel.loaded=false;panel.revision++;}
      if(!holder.hidden)await Promise.all([...panels.values()].filter(p=>p.box.open&&sections.get(p.step)?.box.open).map(p=>loadPanel(p)));
    }
    function reset(){for(const panel of panels.values())panel.revision++;loaded=false;sections.clear();panels.clear();tree.replaceChildren();say('');navigationId++;}
    return {show:async()=>{await show();await Promise.all([...panels.values()].filter(p=>p.box.open&&sections.get(p.step)?.box.open&&!p.loaded).map(p=>loadPanel(p)));},invalidate,reset};
  };
})();
