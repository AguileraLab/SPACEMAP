/* A saved annotation equation: result = union(from) − union(subtract). */
(() => {
  'use strict';
  window.createAnnotationDifference = ({node,help,get,post,route,image,flush,showPreview,showResult,readOnly,picked,jobs}) => {
    const details=node('details',{class:'studio-sec annotation-difference'});
    const body=node('div',{class:'studio-sec-body'});
    details.append(node('summary',{class:'studio-sec-h'},[
      node('span',{text:'Subtract annotations'}),help('Create a new editable layer, for example Stroma = Tissue − Tumor − Epithelia. The names are those of Layers (this image’s, then each imported file’s); a renamed polygon belongs to its new name. Several names on a side are combined. Sources stay unchanged; Undo and History apply to the result.')]),body);
    const label=node('input',{type:'text',maxlength:60,placeholder:'Stroma', 'aria-label':'Subtraction result label'});
    const status=node('p',{class:'muted',role:'status'});
    const equation=node('div',{class:'annotation-equation'});
    let layers=[],busy=false,loading=false,again=false,preview=null;
    const token=image()?.token;
    const active=()=>details.isConnected&&image()?.token===token;
    const sides={base:[],subtract:[]},pickers={};
    const key=item=>item.key;
    const imageName=()=>String(image()?.name||'This image').replace(/(\.ome)?\.(tiff?|qptiff|svs)$/i,'');
    const sourceName=item=>item.source||imageName();
    const clearPreview=()=>{preview=null;showPreview('annotation-difference-preview',null);clear.hidden=true;};
    const changed=()=>{clearPreview();status.textContent='';sync();};
    // One entry per name, as Layers lists them; artifacts say so.
    const options=()=>layers.map(item=>({...item,title:item.name+(item.role==='artifact'?' (artifact)':'')+' · '+sourceName(item)+' ('+item.count+')'}));
    const drawPicker=side=>{
      const {chips,select}=pickers[side],available=options();
      chips.replaceChildren(...sides[side].map(item=>{
        const current=available.find(option=>key(option)===key(item));
        const title=current?.title||item.title+' (unavailable)';
        return node('span',{class:'annotation-operand',title},[
          node('span',{text:item.name+(item.source?' · '+item.source:'')}),
          node('button',{class:'btn small ghost',text:'×',title:'Remove '+title,disabled:busy,onclick:()=>{
            sides[side]=sides[side].filter(other=>key(other)!==key(item));drawPicker(side);changed();}})]);
      }));
      select.replaceChildren(node('option',{value:'',text:side==='base'?'Add source…':'Add annotation…'}));
      for(const source of [...new Set(available.map(item=>item.source))]){
        const group=node('optgroup',{label:source||imageName()});
        for(const option of available.filter(item=>item.source===source)){
          if(!sides[side].some(item=>key(item)===key(option)))group.append(node('option',{
            value:key(option),text:option.name+(option.role==='artifact'?' (artifact)':'')+' ('+option.count+')'}));
        }
        if(group.children.length)select.append(group);
      }
      select.disabled=busy||loading;
    };
    for(const side of ['base','subtract']){
      const chips=node('div',{class:'annotation-operands'});
      const select=node('select',{'aria-label':side==='base'?'From annotations':'Subtract annotations'});
      select.onchange=()=>{const item=options().find(option=>key(option)===select.value);
        if(item){sides[side].push(item);drawPicker(side);changed();}};
      pickers[side]={chips,select,box:node('div',{class:'annotation-operand-picker'},[chips,select])};
    }
    equation.append(node('div',{class:'annotation-result'},[node('label',{text:'Result label'}),label]),
      node('span',{class:'annotation-operator',text:'='}),pickers.base.box,
      node('span',{class:'annotation-operator',text:'−'}),pickers.subtract.box);
    const run=async action=>{
      if(busy||loading||!active())return;
      busy=true;sync();status.textContent=action==='preview'?'Calculating…':'Creating layer…';
      try{
        await flush?.();if(!active())return;
        // Keep the selected revisions: a changed source must be reviewed, not silently substituted.
        const answer=await post(route('/annotation_difference'),{action,image_token:token,label:label.value.trim(),
          ...Object.fromEntries(Object.entries(sides).map(([side,items])=>[side,items.map(({key,revisions})=>({key,revisions}))])),
          ...(preview?{target_revision:preview.target_revision}:{})});
        if(!active())return;
        if(action==='preview'){
          preview=answer;showPreview('annotation-difference-preview',{outlines:answer.outlines,colour:'#56b4a6',kind:'example'});
          clear.hidden=false;status.textContent=answer.count+' remaining polygon'+(answer.count===1?'':'s')+' · preview';
        }else{
          clearPreview();await showResult(answer);if(!active())return;
          status.textContent=answer.workflow_warning||answer.count+' polygon'+(answer.count===1?'':'s')+' saved as '+answer.recipe.label+'.';
          await refresh();
        }
      }catch(error){if(active())status.textContent=error.message;}
      finally{busy=false;if(active()){sync();if(again){again=false;refresh();}}}
    };
    const test=node('button',{class:'btn small',text:'Preview',onclick:()=>run('preview')});
    const save=node('button',{class:'btn small primary',text:'Create layer',onclick:()=>run('save')});
    // The same equation, by name, on each image ticked in the list: a job; each image uses its own annotations of
    // those names (a minus name it lacks takes nothing away; an image without the base names is skipped).
    const some=node('button',{class:'btn small',text:'Selected',title:'Create this layer on each image ticked in the list, from its own annotations of these names',onclick:async()=>{
      const images=picked?.()||[];
      if(!images.length){status.textContent='Tick the images in the list first.';return;}
      busy=true;sync();status.textContent='Starting…';
      try{
        await flush?.();
        await post(route('/annotation_difference'),{action:'batch',image_token:token,images,label:label.value.trim(),
          ...Object.fromEntries(Object.entries(sides).map(([side,items])=>[side,items.map(({name,role})=>({name,role}))]))});
        status.textContent='Creating '+label.value.trim()+' on '+images.length+' image'+(images.length===1?'':'s')+' in the background.';
        jobs?.();
      }catch(error){status.textContent=error.message;}
      finally{busy=false;sync();}
    }});
    const refreshButton=node('button',{class:'btn small ghost',text:'Refresh choices',title:'Reload saved layers and labels after editing them.',onclick:()=>refresh(true)});
    const clear=node('button',{class:'btn small ghost',text:'Clear preview',hidden:true,onclick:clearPreview});
    function sync(){
      const valid=label.value.trim()&&sides.base.length&&sides.subtract.length&&Object.values(sides).flat().every(item=>
        options().some(option=>key(option)===key(item)));
      test.disabled=busy||loading||!valid;save.disabled=busy||loading||!valid||readOnly();some.disabled=save.disabled;
      label.disabled=busy;refreshButton.disabled=busy||loading;clear.disabled=busy;
      for(const picker of Object.values(pickers)){
        picker.select.disabled=busy||loading;
        picker.chips.querySelectorAll('button').forEach(button=>button.disabled=busy);
      }
    }
    async function refresh(explicit=false){
      if(!active()||!details.open)return;
      if(loading||busy){again=true;return;}
      loading=true;sync();
      try{
        await flush?.();const answer=await get(route('/annotation_difference'));if(!active())return;
        layers=answer.names||[];
        let changedSources=false;
        for(const side of Object.keys(sides))sides[side]=sides[side].map(item=>{
          const current=options().find(option=>key(option)===key(item));
          if(!current||JSON.stringify(current.revisions)!==JSON.stringify(item.revisions))changedSources=true;
          return current||item;
        });
        if(changedSources){clearPreview();status.textContent='Choices updated after annotation edits. Preview again.';}
        else if(explicit){clearPreview();status.textContent=layers.length?'Choices updated.':'Save annotation polygons to use them here.';}
        for(const side of Object.keys(sides))drawPicker(side);
      }catch(error){if(active())status.textContent=error.message;}
      finally{loading=false;if(active())sync();if(again){again=false;refresh();}}
    }
    label.addEventListener('input',changed);
    body.append(equation,node('div',{class:'row wrap'},[test,save,some,help('Create layer makes it on this image. Selected makes it on each ticked image from that image’s own annotations of the same names.'),refreshButton,clear]),status);
    details.addEventListener('toggle',()=>{if(details.open)refresh();else clearPreview();});
    sync();
    return {box:details,refresh};
  };
})();
