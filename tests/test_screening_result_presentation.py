"""Numeric sorting and endpoint direction must be independent of page/display text."""
from pathlib import Path
import subprocess


def test_sort_and_direction_semantics():
    source = Path(__file__).parents[1] / 'src/spacemap_gui/viewer/frontend/screening-results.js'
    script = r'''
    global.window={};eval(require('fs').readFileSync(process.argv[1],'utf8'));
    const {sortRows,direction,recovery,colour}=window.spacemapScreeningResults;
    const assert=require('node:assert/strict'),e={type:'binary',group_a:'CR',group_b:'PR'};
    const rows=Array.from({length:211},(_,i)=>({family:'density',label:'Type '+i,q:i===0?null:(211-i)/1000,zone:i,effect:i-100,n_samples:i,status:'tested'}));
    const before=JSON.stringify(rows),asc=sortRows(rows,'q',false,e,{density:'Density'}),desc=sortRows(rows,'q',true,e,{density:'Density'});
    assert.equal(asc[0].q,.001);assert.equal(desc[0].q,.210);assert.equal(asc.at(-1).q,null);assert.equal(desc.at(-1).q,null);
    assert.equal(JSON.stringify(rows),before);
    assert.deepEqual(sortRows(rows,'zone',false,e,{}).map(r=>r.zone),Array.from({length:211},(_,i)=>i));
    assert.equal(direction(e,2).label,'Higher in PR');assert.equal(direction(e,-2).label,'Higher in CR');
    assert.equal(direction({...e,change_col:'visit'},-2).label,'Larger change in CR');
    assert.equal(direction({type:'survival'},.7).label,'Lower event hazard');assert.equal(direction({type:'survival'},1).short,'=');
    assert.equal(direction({type:'survival'},2).label,'Higher event hazard');
    assert.equal(direction({type:'continuous'},-.2).label,'Negative association');
    assert.equal(direction(e,null).label,'—');assert.equal(recovery({resampling:'bootstrap',recovery:0}),0);
    assert.equal(recovery({recovery:.9}),null);
    assert.equal(colour('CR',e,{CR:'#abc'}),'#abc');
    '''
    subprocess.run(['node', '-e', script, str(source)], check=True, capture_output=True, text=True)
