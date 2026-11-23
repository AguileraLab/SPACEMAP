"""Revision-safe statistics adapter, reusable disposable cache and reports."""
from copy import deepcopy
import base64
import csv
from io import BytesIO, StringIO
import json
import time
from pathlib import Path
import uuid
import zipfile
from xml.sax.saxutils import escape

from spacemap import flow_statistics as engine
from . import flow_gating as flow, flow_cache, data_editor


def run(channel, body):
    started=time.perf_counter()
    ws=flow.load(channel)
    if body.get('revision') != ws['revision']:
        raise ValueError('The workspace changed. Run the analysis again.')
    design=deepcopy(body['design'])
    # Resolve named thresholds authoritatively; imports may carry a plot axis.
    thresholds={t['id']:t for t in ws.get('thresholds',[]) + [g for g in ws['gates'] if g.get('kind')=='threshold']}
    for f in design.get('features',[]):
        if f.get('metric') in ('proportion','count'):
            f.pop('column',None)
            f.pop('axis',None)
            if f.get('denominator')=='@parent':
                parents={g['id']:g.get('parent','root') for g in ws['gates']}
                f['denominators']={g['id']:parents.get(g.get('population',g['id']),'root') for g in design['groups']}
        if f.get('metric')=='positive':
            t=thresholds.get(f.get('threshold_id'))
            if not t:
                raise ValueError('This threshold was removed. Select an active or named threshold.')
            f['threshold']={k:deepcopy(t[k]) for k in ('id','name','x','bounds')}
    stamps=[]
    for source in ws['sources']:
        with data_editor.locked(source['table']) as editor:
            if editor.cursor:
                raise ValueError('Save the Data edits before running statistics.')
        stamps.append((source['table'],data_editor.stamp(source['table'])))
    _,_,folder=flow.context(channel)
    key=flow.digest(['flow-statistics-v3',ws['id'],ws['sources'],stamps,flow.gate_meaning(ws['gates']),design])
    cached=flow_cache.get(folder,key)
    if cached and 'statistics' in cached:
        return {**deepcopy(cached['statistics']),'revision':ws['revision'],'cached':True,'elapsed_ms':round(1000*(time.perf_counter()-started),1)}
    extra=flow.required({'gates':[design]},[design['group_by']] if design.get('group_by') else [])
    frame,revision,masks=flow.evaluate(ws,extra)
    for f in design.get('features',[]):
        axis=f.get('threshold',{}).get('x') if f['metric']=='positive' else f.get('axis')
        if axis and 'fitted' not in axis and not axis.get('pipeline_fitted'):
            reference=masks.get(axis.get('reference_population','root'))
            if reference is None:
                raise ValueError('The transformation reference population is missing.')
            fitted=flow.engine.fit_axis(frame,axis,reference[0])
            if f['metric']=='positive':f['threshold']['x']=fitted
            else:f['axis']=fitted
    result=engine.run(frame,masks,design)
    if stamps != [(s['table'],data_editor.stamp(s['table'])) for s in ws['sources']] or flow.load(channel)['revision'] != ws['revision']:
        raise ValueError('The data or workspace changed during calculation. Run the analysis again.')
    result.update(revision=ws['revision'],dataset_revision=revision,result_id=key,
                  gate_signature=flow.digest(flow.gate_meaning(ws['gates'])),source_signature=flow.digest(ws['sources']),
                  elapsed_ms=round(1000*(time.perf_counter()-started),1),cached=False)
    # Same bounded files, overwrite semantics and Jobs-tab cleanup as plots.
    flow_cache.put(folder,key,{'statistics':result})
    return result


def report_frames(result):
    groups={g['id']:g.get('label',g['id']) for g in result['design']['groups']}
    features={f['id']:f.get('label',f.get('column',f['id'])) for f in result['design']['features']}
    lookup={(s['feature'],s['group']):s for s in result['summaries']}
    def decorate(row):
        r=dict(row)
        r['feature_label']=features.get(r.get('feature'),'')
        for key in ('group','a','b'):
            if key in r:r[key+'_label']=groups.get(r[key],r[key])
        return r
    tests=[]
    for r in result['tests']:
        row=decorate(r)
        for suffix,key in (('a','a'),('b','b')):
            for k,v in lookup.get((r['feature'],r.get(key)),{}).items():
                if k in ('n_cells','denominator_cells','n_valid','mean','median','iqr','q1','q3','N'):
                    row[k+'_'+suffix]=v if k!='N' else row.get('N_'+suffix,v)
        tests.append(row)
    return {'Tests':tests,'Group summaries':[decorate(r) for r in result['summaries']],
            'Unit values':[decorate(r) for r in result['samples']],
            'Heatmap':[decorate(r) for r in result['heatmap']],
            'Methods':[{'field':k,'value':v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)} for k,v in
                       {'unit':result['unit'],'correction':result['correction'],'family':result['correction_family'],
                        'hypotheses':result['hypotheses'],'notices':result['notices'],'design':result['design'],
                        'dataset_revision':result['dataset_revision'],'display_sampling':result['sampled_display']}.items()]}


def _xlsx(sheets):
    """Small standards-compliant workbook; no optional Excel package required."""
    stream=BytesIO();ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        names=list(sheets)
        types+=''.join(f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(len(names)))+'</Types>'
        z.writestr('[Content_Types].xml',types)
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{escape(name)}" sheetId="{i+1}" r:id="rId{i+1}"/>' for i,name in enumerate(names))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i+1}.xml"/>' for i in range(len(names)))+'</Relationships>')
        for i,records in enumerate(sheets.values()):
            keys=list(dict.fromkeys(k for r in records for k in r))
            rows=[keys]+[[r.get(k) for k in keys] for r in records]
            if len(rows)>1048576:raise ValueError('Excel supports at most 1,048,576 rows per sheet. Use CSV.')
            xml=f'<worksheet xmlns="{ns}"><sheetData>'
            for row in rows:
                xml+='<row>'+''.join('<c/>' if v is None else f'<c><v>{v}</v></c>' if isinstance(v,(int,float)) and not isinstance(v,bool) else '<c t="inlineStr"><is><t xml:space="preserve">'+escape(str(v))+'</t></is></c>' for v in row)+'</row>'
            z.writestr(f'xl/worksheets/sheet{i+1}.xml',xml+'</sheetData></worksheet>')
    return stream.getvalue()


def export(channel,body):
    result=run(channel,body)
    if (body.get('expected_dataset_revision',result['dataset_revision']) != result['dataset_revision'] or
            body.get('expected_gate_signature',result['gate_signature']) != result['gate_signature']):
        raise ValueError('The measurements or gates changed. Run the analysis again before exporting.')
    sheets=report_frames(result)
    kind=body.get('format','csv')
    if kind=='xlsx':
        raw=_xlsx(sheets);mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif kind=='csv':
        # A single tidy report contains every section and complete group statistics.
        rows=[{'section':name,**row} for name,records in sheets.items() for row in records]
        keys=list(dict.fromkeys(k for row in rows for k in row));stream=StringIO()
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        raw=stream.getvalue().encode('utf-8-sig');mime='text/csv'
    else:raise ValueError('Choose CSV or Excel.')
    ws=flow.load(channel);folder=Path(ws['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    stem='flow-statistics-'+uuid.uuid4().hex[:8];target=folder/(stem+'.'+kind)
    target.write_bytes(raw)
    flow.atomic_json(folder/(stem+'.json'),{'design':result['design'],'dataset_revision':result['dataset_revision'],
                     'workspace_revision':ws['revision'],'sources':ws['sources'],'gates':ws['gates']})
    return {'file':str(target),'download':'data:'+mime+';base64,'+base64.b64encode(raw).decode()}


def validate_figure(workspace, result):
    """Do not attach stale scientific results to a freshly exported figure."""
    stamps=[]
    for source in workspace['sources']:
        with data_editor.locked(source['table']) as editor:
            if editor.cursor:
                raise ValueError('Save the Data edits and run the analysis again before exporting.')
        stamps.append((source['table'],data_editor.stamp(source['table'])))
    if (result.get('dataset_revision')!=flow.digest(stamps) or
            result.get('gate_signature')!=flow.digest(flow.gate_meaning(workspace['gates'])) or
            result.get('source_signature')!=flow.digest(workspace['sources'])):
        raise ValueError('The measurements or gates changed. Run the analysis again before exporting.')
