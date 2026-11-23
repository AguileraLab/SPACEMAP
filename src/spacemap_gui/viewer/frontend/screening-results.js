/* Saved-result presentation only: no inference or recalculation of p/q. */
window.spacemapScreeningResults = (() => {
  const finite = value => value === null || value === undefined || value === '' || !Number.isFinite(Number(value)) ? null : Number(value);
  const direction = (endpoint, effect) => {
    const value = finite(effect);
    if(value === null || endpoint.type === 'describe') return {label:'—', short:'—', group:null};
    const sign = Math.sign(value - (endpoint.type === 'survival' ? 1 : 0));
    if(!sign) return {label:'Equal estimate', short:'=', group:null};
    if(endpoint.type === 'binary') {
      const group = String(sign > 0 ? endpoint.group_b : endpoint.group_a);
      return {group, short:group, label:(endpoint.change_col ? 'Larger change in ' : 'Higher in ') + group};
    }
    if(endpoint.type === 'survival') return {group:null, short:sign > 0 ? '↑ hazard' : '↓ hazard', label:sign > 0 ? 'Higher event hazard' : 'Lower event hazard'};
    return {group:null, short:sign > 0 ? 'Positive' : 'Negative', label:sign > 0 ? 'Positive association' : 'Negative association'};
  };
  const colour = (group, endpoint, palette={}) => palette[group] || (String(group) === String(endpoint.group_b) ? '#e79562' : '#71b8df');
  const recovery = row => ['bootstrap','subsample'].includes(row.resampling) ? finite(row.recovery) : null;
  const sortRows = (rows, key, descending, endpoint, families) => {
    const value = row => key === 'name' ? (families[row.family] || row.family) + ' · ' + row.label :
      key === 'direction' ? direction(endpoint,row.effect).label : key === 'recovery' ? recovery(row) :
      key === 'status' ? row.status : finite(row[key]);
    return rows.map((row,index)=>({row,index,value:value(row)})).sort((a,b)=>{
      if(a.value === null || a.value === undefined) return b.value === null || b.value === undefined ? a.index-b.index : 1;
      if(b.value === null || b.value === undefined) return -1;
      const delta = typeof a.value === 'string' ? a.value.localeCompare(b.value,undefined,{numeric:true}) : a.value-b.value;
      return (descending ? -delta : delta) || a.index-b.index;
    }).map(item=>item.row);
  };
  return {finite,direction,colour,recovery,sortRows};
})();
