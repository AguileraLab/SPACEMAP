"""Disposable representative-size benchmark; never opens research data."""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spacemap import flow_gating as engine
from spacemap_gui.studio import flow_gating as studio


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=2_500_000)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='spacemap-flow-benchmark-') as folder:
        path=Path(folder)/'cells.parquet';rng=np.random.default_rng(42)
        writer=None
        for start in range(0,args.rows,100000):
            n=min(100000,args.rows-start);index=np.arange(start,start+n)
            frame=pd.DataFrame({f'M{i}':rng.lognormal(4+i/4,1,n).astype('float32') for i in range(8)})
            frame['image_name']=['image_'+str(i%44) for i in index]
            frame['seg_id']=index//44+1;frame['cell_key']=index
            frame['arm']=np.where(index%2,'A','B')
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(path,table.schema)
            writer.write_table(table)
        writer.close();del frame,table
        axes={c:{'column':c,'scale':'arcsinh','cofactor':5} for c in ['M0','M1']}
        gates=[{'id':'a','kind':'rectangle','parent':'root','x':axes['M0'],'y':axes['M1'],'bounds':[2,5,2,5]},
               {'id':'b','kind':'range','parent':'a','x':axes['M0'],'bounds':[3,4]},
               {'id':'c','kind':'boolean','parent':'a','op':'not','refs':['b']}]
        workspace={'sources':[{'table':str(path),'roi_column':'image_name','label_column':'seg_id','cell_id_column':'cell_key','selected_images':['image_'+str(i) for i in range(44)]}],'gates':gates}
        started=time.perf_counter();data,revision,results=studio.evaluate(workspace,['M'+str(i) for i in range(8)])
        cold=time.perf_counter()-started
        started=time.perf_counter();studio.evaluate(workspace,['M'+str(i) for i in range(8)]);warm=time.perf_counter()-started
        started=time.perf_counter();hist=engine.distributions(data,results,{'x':axes['M0'],'bins':100,'series':[{'population':'a'},{'population':'b'},{'population':'c'}]});hist_time=time.perf_counter()-started
        started=time.perf_counter();recipe=engine.fit_axis(data,{'column':'M0','markers':['M'+str(i) for i in range(8)],'pipeline':[{'op':'arcsinh','cofactor':5},{'op':'zscore'},{'op':'row_zscore'},{'op':'cdf'},{'op':'nlog'}]});values=engine.measurement(data,recipe);recipe_time=time.perf_counter()-started
        started=time.perf_counter();engine.fit_axis(data,recipe);engine.measurement(data,recipe);recipe_warm=time.perf_counter()-started
        print(json.dumps({'rows':args.rows,'images':44,'markers':8,'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
            'cold_projected_read_and_gates_s':round(cold,3),'cached_evaluation_s':round(warm,3),
            'exact_three_histograms_s':round(hist_time,3),'eight_marker_recipe_fit_and_apply_s':round(recipe_time,3),
            'cached_recipe_fit_and_apply_s':round(recipe_warm,3),
            'counts':{k:int(v[0].sum()) for k,v in results.items()},'finite_recipe_values':int(np.isfinite(values).sum()),
            'parquet_mb':round(path.stat().st_size/1e6,1)},indent=2))


if __name__=='__main__':main()
