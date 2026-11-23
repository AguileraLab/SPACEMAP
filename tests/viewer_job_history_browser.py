"""Local start times, useful summaries, and persistent history-only clearing."""
from viewer_threshold_browser import main
from viewer_studio_leiden_browser import setup as prepare
from test_studio_job_history import record


def setup(root):
    from spacemap import pending_columns
    import pandas as pd

    prepare(root)
    table = root / 'cells.csv'
    frame = pd.read_csv(table)
    pending_columns.save(table, {'phenotype_history':['T'] * len(frame)},
                         anchor_column='cell_id', anchor_values=frame.cell_id)
    for key, status in [('a','done'),('b','failed'),('c','running')]:
        record(root, key, status, {'table':str(table),'selected_images':['P11_roi_2'],
                                 'output_col':'phenotype_history','markers':['M0','M1']})


DRIVER = r'''
<script>
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const until=async f=>{for(let i=0;i<1800;i++){if(await f())return;await delay(50);}throw Error('Timed out: '+f);};
let checks=Number(sessionStorage.getItem('history-checks')||0);
const assert=(v,t)=>{if(!v)throw Error(t);checks++;fetch('/result?progress='+encodeURIComponent(t));};
const card=key=>document.querySelector('[data-job-id="'+key.repeat(32)+'"]');
const button=text=>[...document.querySelectorAll('#studio-jobs button')].find(b=>b.textContent===text);
const jobSelection=key=>card(key).querySelector('input[type="checkbox"]');
setTimeout(async()=>{try{
 await until(()=>window.__state.studio&&card('c'));
 document.querySelector('[data-tab="jobs"]').click();
 if(!sessionStorage.getItem('history-reloaded')){
   await until(()=>card('a')&&card('b'));
   assert(card('a').querySelector('time').textContent.includes('3:00:00 PM')&&card('a').querySelector('time').textContent.includes('CDT'),'Start time uses the browser time zone (Chicago), including seconds');
   assert(card('a').querySelector('.studio-job-summary').textContent==='P11_roi_2 · Output: phenotype_history · 2 features','The description identifies the image, output column and feature count');
   assert(jobSelection('c').disabled&&!jobSelection('a').disabled,'Only finished jobs can be selected for clearing');
   jobSelection('a').click();assert(!button('Clear selected').disabled,'Selecting a finished card enables Clear selected');
   button('Clear selected').click();await until(()=>!card('a')&&card('b'));
   assert(card('b')&&card('c'),'Clearing one card keeps the other jobs');
   const status=await(await fetch('/spacemap-viewer/api/studio/browser-check/jobs/'+'a'.repeat(32))).json();
   assert(status.job.hidden&&status.log==='Saved predictions\n','A cleared job still has its original log');
   button('Clear finished').click();await until(()=>!card('b')&&card('c'));
   assert(button('Clear finished').disabled&&button('Clear selected').disabled,'Clearing finished history leaves the running job alone');
   sessionStorage.setItem('history-checks',String(checks));sessionStorage.setItem('history-reloaded','yes');
   await delay(500);location.reload();return;
 }
 assert(!card('a')&&!card('b')&&card('c'),'Cleared history stays cleared after page reload');
 const saved=await(await fetch('/spacemap-viewer/api/studio/browser-check/saved')).json();
 assert(saved.pending.some(c=>c.column==='phenotype_history'),'Automatically saved Parquet predictions survive both clear actions and reload');
 await until(()=>document.querySelector('#studio-jobs').textContent.includes('saved in Parquet'));
 assert(document.querySelector('#studio-jobs').textContent.includes('saved in Parquet'),'Jobs clearly reports that outputs are already saved');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks})));
}catch(error){fetch('/result?error='+encodeURIComponent(String(error)+'\n'+error.stack));}},100);
</script>
'''


if __name__ == '__main__':
    main(DRIVER, studio=True, timeout=180, setup=setup, screenshot=False)
