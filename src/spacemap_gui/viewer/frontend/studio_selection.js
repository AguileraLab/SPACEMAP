/* Selection controls load only when requested; the image viewer stays mounted. */
(() => {
  'use strict';
  window.createStudioSelection = ({holder,node,get,session,runCommand,showResult,setSelectionTool}) => {
    let current, options, imageId, timer, busy=false, rerun=false;
    const details=node('details',{id:'studio-select'}),form=node('div',{class:'studio-form'}),summary=node('p',{role:'status',class:'hint'});
    details.append(node('summary',{text:'Select cells by a marker rule'}),form);holder.append(details);
    const call=async(name,args,confirmed=false)=>{const result=await runCommand(name,args,{confirmed});showResult(result);if(result.status!=='done')throw new Error(result.message);return result;};
    let conditions,scope,join;
    const read=()=>({scope:scope.value,join:join.value,conditions:Array.from(conditions.children).map(row=>({marker:row.querySelector('[data-marker]').value,transform:row.querySelector('[data-transform]').value,op:row.querySelector('[data-op]').value,value:Number(row.querySelector('input').value)}))});
    const apply=async()=>{
      if(busy){rerun=true;return;}busy=true;summary.textContent='Finding matching cells… You can still move around the image.';
      try{const result=await call('select_cells',{rule:read()});update(result.spec);}
      catch(error){summary.textContent=error.message;}
      finally{busy=false;if(rerun){rerun=false;apply();}}
    };
    const live=()=>{clearTimeout(timer);if(scope.value==='image')timer=setTimeout(apply,450);};
    const select=(entries,value,label)=>{const input=node('select',{'aria-label':label});entries.forEach(item=>input.append(node('option',{value:Array.isArray(item)?item[0]:item,text:Array.isArray(item)?item[1]:item})));input.value=value;return input;};
    const add=condition=>{
      const row=node('div',{class:'studio-rule-row'});
      const marker=select(options.markers,condition?.marker || options.markers[0],'Rule marker');marker.dataset.marker='';
      const transform=select(['Raw','Arcsinh','Z1','Z2'],condition?.transform || 'Raw','Rule scale');transform.dataset.transform='';
      const op=select(['>','>=','<','<=','=='],condition?.op || '>','Rule comparison');op.dataset.op='';
      const value=node('input',{type:'number',step:'any',value:condition?.value ?? 0,'aria-label':'Rule value'});
      const remove=node('button',{class:'btn tiny',text:'Remove',onclick:()=>{if(conditions.children.length>1){row.remove();live();}}});
      row.append(marker,transform,op,value,remove);row.addEventListener('change',live);conditions.append(row);
    };
    const heatmap=value=>{
      const box=node('div',{class:'studio-heatmap'});box.append(node('p',{class:'hint',text:value.note}));
      const table=node('table');table.append(node('tr',{},[node('th',{text:'Marker'}),...value.groups.map((name,index)=>node('th',{text:name+' ('+value.counts[index].toLocaleString()+')'}))]));
      const numbers=value.values.flat().filter(v=>v!==null),low=Math.min(...numbers),high=Math.max(...numbers);
      value.markers.forEach((marker,i)=>table.append(node('tr',{},[node('th',{text:marker}),...value.values.map(values=>{
        const number=values[i],alpha=number===null?0:.15+.75*(number-low)/Math.max(1e-12,high-low);
        return node('td',{text:number===null?'—':number.toFixed(2),style:'background:rgba(65,135,225,'+alpha+')'});
      })])));box.append(table);return box;
    };
    const update=spec=>{
      if(spec?.id && imageId && spec.id!==imageId){options=null;clearTimeout(timer);form.replaceChildren();if(details.open)details.dispatchEvent(new Event('toggle'));}
      if(spec?.id)imageId=spec.id;
      current=spec?.selection;
      if(!current){summary.textContent='No cells selected yet.';form.querySelector('.studio-selection-hists')?.replaceChildren();return;}
      if(!details.open)return;
      summary.textContent=current.count.toLocaleString()+' of '+current.total.toLocaleString()+' cells selected.';
      const graphs=form.querySelector('.studio-selection-hists');if(!graphs)return;graphs.replaceChildren();
      (current.histograms || []).forEach(hist=>{
        const group=node('div',{},[node('small',{text:hist.marker+' · '+hist.transform})]);
        const chart=node('div',{class:'studio-hist',role:'img','aria-label':'Sampled histogram for '+hist.marker});const largest=Math.max(1,...hist.counts);
        hist.counts.forEach((count,i)=>chart.append(node('span',{style:'height:'+Math.max(1,100*count/largest)+'%',title:hist.edges[i].toPrecision(3)+' to '+hist.edges[i+1].toPrecision(3)+': '+count})));
        const condition=current.rule.conditions.find(item=>item.marker===hist.marker && item.transform===hist.transform);
        if(condition){const x=100*(condition.value-hist.edges[0])/(hist.edges[hist.edges.length-1]-hist.edges[0] || 1);chart.append(node('i',{class:'studio-gate',style:'left:'+Math.max(0,Math.min(100,x))+'%',title:'Cut-off: '+condition.value}));}
        group.append(chart);graphs.append(group);
      });
    };
    details.addEventListener('toggle',async()=>{
      if(!details.open || options)return;
      form.textContent='Reading marker choices…';
      try{
        options=await get('/studio/'+encodeURIComponent(session)+'/selection_options');form.replaceChildren();
        scope=select([['image','This image'],['project','All ROIs']],'image','Selection scope');
        join=select([['all','Match all conditions'],['any','Match any condition']],'all','Join conditions');join.addEventListener('change',live);
        conditions=node('div');form.append(scope,join,conditions);add();
        form.append(node('button',{class:'btn small',text:'Add condition',onclick:()=>add()}),node('button',{class:'btn primary',text:'Apply selection',onclick:apply}),summary,
          node('p',{class:'hint',text:'Rules use saved measurements. Edits update this image after a short pause. For all ROIs, press Apply. Z1 compares cells within each ROI; Z2 compares markers within each cell. Arcsinh uses cofactor 5.'}),node('div',{class:'studio-selection-hists'}),
          node('p',{class:'hint',text:'Histograms show a sample; every cell is checked by the rule.'}));
        const refine=node('div',{class:'studio-refine'});
        ['add','remove'].forEach(mode=>['click','rect','polygon'].forEach(shape=>refine.append(node('button',{class:'btn small',text:(mode==='add'?'Add':'Remove')+' by '+(shape==='rect'?'box':shape),onclick:()=>{setSelectionTool(mode,shape==='click'?'off':shape);summary.textContent='Use '+shape+' to '+mode+' cells. Drawn regions use cell centres.';}}))));
        refine.append(node('button',{class:'btn small',text:'Finish selecting',onclick:()=>{setSelectionTool(null,'off');update({selection:current});}}));form.append(refine);
        const comparison=node('div');form.append(node('button',{class:'btn',text:'Compare markers',onclick:async()=>{try{const result=await call('selection_heatmap',{});comparison.replaceChildren(heatmap(result.data.heatmap));}catch(error){summary.textContent=error.message;}}}),comparison);
        const column=node('input',{value:'phenotype_ref1',list:'studio-reference-columns','aria-label':'Selection reference column'}),columns=node('datalist',{id:'studio-reference-columns'});
        options.columns.forEach(value=>columns.append(node('option',{value})));
        const phenotype=node('input',{list:'studio-reference-phenotypes',placeholder:'Phenotype','aria-label':'Selection phenotype'}),names=node('datalist',{id:'studio-reference-phenotypes'});options.phenotypes.forEach(value=>names.append(node('option',{value})));
        form.append(node('label',{},[node('span',{text:'Reference column'}),column]),columns,node('label',{},[node('span',{text:'Phenotype'}),phenotype]),names,
          node('button',{class:'btn primary',text:'Assign selected cells',onclick:async()=>{try{const result=await call('assign_selection',{column:column.value,phenotype:phenotype.value},true);summary.textContent=result.message;}catch(error){summary.textContent=error.message;}}}));
        const ruleName=node('input',{placeholder:'Rule name','aria-label':'Saved rule name'}),saved=select(options.rules.map(rule=>rule.name),options.rules[0]?.name,'Saved rule');
        form.append(ruleName,node('button',{class:'btn',text:'Save current rule',onclick:async()=>{try{await call('save_rule',{rule_name:ruleName.value},true);saved.append(node('option',{value:ruleName.value,text:ruleName.value}));saved.value=ruleName.value;}catch(error){summary.textContent=error.message;}}}),saved,
          node('button',{class:'btn',text:'Apply saved rule',onclick:async()=>{try{const result=await call('apply_rule',{rule_name:saved.value,scope:scope.value});conditions.replaceChildren();result.data.selection.rule.conditions.forEach(add);join.value=result.data.selection.rule.join;update(result.spec);}catch(error){summary.textContent=error.message;}}}));
        update({selection:current});
      }catch(error){form.textContent=error.message;options=null;}
    });
    return {update,heatmap};
  };
})();
