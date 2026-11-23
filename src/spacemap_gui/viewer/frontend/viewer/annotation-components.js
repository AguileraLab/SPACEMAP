/* Linked annotation component lists. Only the current 100 metadata rows render. */
'use strict';

function annotationComponentRoute(layer) { return state.reviewer ? '/components?item='+encodeURIComponent(layer.review_item) : reviewRoute()+'/components/'+encodeURIComponent(layer.token); }

function annotationComponents(layer,holder) {
  const key=layer.token, memories=state.componentLists||(state.componentLists={});
  const memory=memories[key]||(memories[key]={classes:{}});
  memory.selected=annotationSelection(layer);holder.dataset.componentSource=layer.token;
  holder.classList.add('component-browser');
  const route=annotationComponentRoute(layer);
  const reveal=new Map();holder.revealComponent=feature=>reveal.get(feature.properties.classification?.name||feature.properties.name)?.(feature.properties.id);
  const error=err=>{holder.replaceChildren(el('div',{role:'alert',text:err.message}),el('button',{class:'btn small',text:'Retry',onclick:()=>annotationComponents(layer,holder)}));};
  getJSON(componentQuery(layer,{overview:'1'})).then(overview=>{
    if(!holder.isConnected)return;holder.replaceChildren();
    applyAnnotationIdentity(layer,overview.display);
    layer.annotationLegend={items:overview.classes.map(c=>({...c,value:c.name}))};
    finishAnnotationDisplayNames(layer,layer.annotationLegend);
    const colours=new Map(overview.classes.map(i=>[i.name,i.color]));
    overview.classes.forEach(category=>{
      const memo=memory.classes[category.name]||(memory.classes[category.name]={offset:0,sort:'size_down'});
      memo.search='';
      const branch=el('details',{'data-component-class':category.name});branch.open=memo.open ?? false;
      const check=el('input',{type:'checkbox','aria-label':'Show '+category.name,'data-annotation-visibility':''});
      check.refreshVisibility=()=>{check.checked=layer.visible&&!layer.hidden.has(category.name);};check.refreshVisibility();
      check.onclick=e=>e.stopPropagation();check.onchange=()=>{
        if(check.checked){
          if(!layer.visible)layer.hidden=new Set(overview.classes.filter(c=>c.name!==category.name).map(c=>c.name));
          else layer.hidden.delete(category.name);
          setVisible(layer,true);
        }else layer.hidden.add(category.name);
        annotationDisplayChanged(layer);
      };
      const order=select([['size_down','Largest first'],['size_up','Smallest first'],['place','By place']],memo.sort,v=>{memo.sort=v;memo.offset=0;memo.reveal=null;saveView();populate();});
      order.classList.add('component-order');order.setAttribute('aria-label','Sort '+category.name+' polygons');order.onclick=event=>event.stopPropagation();
      const summary=el('summary',{class:'annotation-component-summary'},[
        el('span',{class:'component-disclosure','aria-hidden':'true'}),check,
        el('span',{class:'component-colour',style:'background:'+(colours.get(category.name)||'#bbb'),title:category.name+' colour'}),
        el('span',{class:'name',text:category.name}),order,
        el('span',{class:'muted component-count',text:category.count.toLocaleString()+(category.count===1?' polygon':' polygons'),title:'Individual polygons; expanding this list does not change their boundaries or rendering.'})]);
      branch.append(summary);
      const body=el('div');branch.append(body);let loaded=false;
      branch.ontoggle=()=>{if(!branch.isConnected)return;const changed=memo.open!==branch.open;memo.open=branch.open;if(changed)saveView();if(branch.open&&!loaded){loaded=true;populate();}};
      holder.append(branch);
      if(branch.open){loaded=true;populate();}
      reveal.set(category.name,id=>{if(branch.open){memo.reveal=id;populate();}});
      async function populate(){
        if(!body.isConnected)return;
        const rows=el('div',{class:'component-page'}),info=el('span',{class:'muted','data-selection-count':''}),foot=el('div',{class:'row'});
        const selection={name_filter:category.name,search:memo.search};
        const chooseAll=async on=>{if(on&&(!layer.visible||layer.hidden?.has(category.name)))return message('Show this annotation before selecting its polygons.',true);const found=await postJSON(route,{action:'select',...selection});if(on)selectAnnotations(layer,found.ids);else {found.ids.forEach(id=>memory.selected.delete(id));annotationOutline(layer);syncAnnotationSelection();}populate();};
        const tools=el('div',{class:'row wrap'},[
          el('button',{class:'btn tiny',text:'All',title:'Select every matching polygon, including other pages',onclick:()=>chooseAll(true).catch(error)}),
          el('button',{class:'btn tiny',text:'None',onclick:()=>chooseAll(false).catch(error)}),
          ...(!state.reviewer?[el('button',{class:'btn tiny',text:'Rename',onclick:()=>rename()})]:[]),
          el('button',{class:'btn tiny',text:state.reviewer?'Suggest review':'Flag for review',onclick:()=>state.reviewer?postJSON('/nominate',{item:layer.review_item,ids:[...memory.selected]}).then(()=>message('Sent to owner.')).catch(error):flagSelectedAnnotations()}),
          ...(!state.reviewer?[el('button',{class:'btn tiny',text:'Delete',onclick:()=>edit('delete')})]:[]),info]);
        body.replaceChildren(tools,rows,foot);
        const clean=el('details'),limit=el('input',{type:'number',min:0,value:0,'aria-label':'Minimum polygon area'}), unit=pixelSize()?'µm²':'px²';
        clean.append(el('summary',{text:'Size cleanup'}),el('div',{class:'row'},[limit,el('span',{text:unit}),el('button',{class:'btn small',text:'Remove smaller',onclick:async()=>{
          try {const found=await postJSON(route,{action:'select',name_filter:category.name,below:Number(limit.value)/(pixelSize()?pixelSize()**2:1)});
            if(!found.ids.length)return message('No smaller polygons.');
            if(confirm('Remove '+found.ids.length+' polygons from the editable copy? The linked original is kept.'))await edit('delete',found.ids);
          }catch(err){error(err);}
        }})]));if(!state.reviewer)body.append(clean);
        let data;
        try {data=await getJSON(componentQuery(layer,{name:category.name,search:memo.search,sort:memo.sort,offset:memo.offset,...(memo.reveal?{locate:memo.reveal}:{})}));}
        catch(err){error(err);return;}if(!rows.isConnected)return;
        memo.offset=data.offset;memo.reveal=null;
        summary.querySelector('.component-count').textContent=data.total.toLocaleString()+(data.total===1?' polygon':' polygons');
        info.textContent=memory.selected.size+' selected';
        data.rows.forEach(r=>{
          const tick=el('input',{type:'checkbox','aria-label':'Select '+category.name+' polygon'});tick.checked=memory.selected.has(r.id);
          tick.onclick=e=>e.stopPropagation();tick.onchange=()=>{if(!layer.visible||layer.hidden?.has(category.name)){tick.checked=false;return message('Show this annotation before selecting its polygons.',true);}selectAnnotations(layer,[r.id],{toggle:true});};
          const row=el('div',{class:'shape-row'+(memory.selected.has(r.id)?' on':''),tabindex:0,'data-component-id':r.id},[tick,
            el('span',{class:'component-colour',style:'background:'+(colours.get(category.name)||'#bbb'),'aria-hidden':'true'}),
            el('span',{class:'grow',text:category.name,title:category.name}),el('span',{class:'area',text:areaText(r.area)}),
            el('button',{class:'btn tiny',text:'Go',onclick:event=>{event.stopPropagation();focus(r,true);}})]);
          row.onclick=event=>focus(r,false,!!(event.ctrlKey||event.metaKey||event.shiftKey));row.onkeydown=e=>{if(e.key==='F2'&&!state.reviewer){e.preventDefault();rename([r.id]);}else if(e.key==='Enter')focus(r);};
          row.oncontextmenu=e=>{e.preventDefault();e.stopPropagation();if(!state.reviewer)rename([r.id]);};rows.append(row);
        });
        foot.replaceChildren(el('span',{class:'muted',text:(data.total?memo.offset+1:0)+'–'+Math.min(memo.offset+data.rows.length,data.total)+' / '+data.total}),
          el('button',{class:'btn tiny',text:'←',disabled:memo.offset?null:'',onclick:()=>{memo.offset=Math.max(0,memo.offset-100);populate();}}),
          el('button',{class:'btn tiny',text:'→',disabled:memo.offset+data.rows.length<data.total?null:'',onclick:()=>{memo.offset+=100;populate();}}));
        async function focus(r,zoom=false,additive=false){try{
          if(!layer.visible||layer.hidden?.has(category.name))return message('Show this annotation before selecting its polygons.',true);
          selectAnnotations(layer,[r.id],{replace:!additive,toggle:additive});
          if(zoom){const f=await getJSON(componentQuery(layer,{id:r.id}));if(state.layers.includes(layer))zoomTo(f.feature);}
          state.reviewLinkedSelection={layer,ids:[r.id],rename:()=>rename([...memory.selected])};
        }catch(err){error(err);}}
        function rename(ids){const name=prompt('Annotation name (an existing or new name):',category.name);if(name?.trim())edit('rename',ids,name);}
        async function edit(action,ids,name=''){
          ids=ids||Array.from(memory.selected);if(!ids.length)return message('Select polygons first.',true);
          const generation=state.initGeneration;
          message('Saving '+ids.length+' polygon'+(ids.length===1?'':'s')+'…');
          try{const found=await postJSON(route,{action,ids,name,revision:data.revision});
            if(generation!==state.initGeneration||!state.layers.includes(layer))return;
            await replaceAnnotationSource(layer,found.layer);memory.selected.clear();message(found.count+' polygons updated.');
          }catch(err){message(err.message,true);}
        }
      }
    });
  }).catch(error);
}

document.addEventListener('keydown',event=>{
  if(state.reviewer||event.key!=='F2'||event.target?.isContentEditable||/INPUT|SELECT|TEXTAREA/.test(event.target?.tagName||''))return;
  if(annotationSelectedGroups().length){event.preventDefault();renameSelectedAnnotations();}
});
