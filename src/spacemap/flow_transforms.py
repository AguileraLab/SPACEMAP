"""Ordered normalization recipes with frozen reference parameters.

Z1, Z2, Normal CDF and NLog use Studio's definitions. No expression evaluation:
every operation, its units, marker scope and fitted parameters are explicit.
"""
from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
import threading
import weakref
import numpy as np
import pandas as pd
from scipy.special import ndtr

OPS = ('identity','arcsinh','log10','log1p','logicle','zscore','row_zscore','cdf','nlog',
       'clip','percentile','minmax','robust_zscore','relu','sqrt','linear')
_CACHE=OrderedDict()
_FITS=OrderedDict()
_LOCK=threading.RLock()


def cache_key(frame,spec,reference=None):
    content={k:spec.get(k) for k in ('pipeline','markers')}
    key=hashlib.sha256(json.dumps(content,sort_keys=True).encode())
    if reference is not None:
        key.update(np.asarray(reference,bool).tobytes())
    return id(frame),key.hexdigest()


def remember(frame,key,value):
    with _LOCK:
        cache=_FITS if isinstance(value,dict) else _CACHE
        cache[key]=(weakref.ref(frame),value)
        # Small fitted parameters are independent of the two large matrices.
        while len(cache)>(16 if cache is _FITS else 2):
            cache.popitem(last=False)


def cached(frame,key):
    with _LOCK:
        cache=_CACHE if key in _CACHE else _FITS
        entry=cache.get(key)
        if entry is not None and entry[0]() is frame:
            cache.move_to_end(key)
            return entry[1]
    return None


def validate(spec):
    steps=spec.get('pipeline',[])
    if not isinstance(steps,list) or len(steps)>16:
        raise ValueError('Use at most 16 transformation steps.')
    if any(s.get('op') not in OPS for s in steps):
        raise ValueError('Unknown transformation step.')
    row_steps=[s for s in steps if s.get('op')=='row_zscore' and s.get('enabled',True)]
    row_markers=[]
    for step in row_steps:
        chosen=list(dict.fromkeys(step.get('markers',spec.get('markers',[]))))
        if spec['column'] not in chosen:
            raise ValueError('Include the axis marker in the explicit Z2 marker set.')
        if len(chosen)<2:
            raise ValueError('Z2 needs at least two explicitly selected markers.')
        row_markers.extend(chosen)
    markers=list(dict.fromkeys([*spec.get('markers',[]),*row_markers,spec['column']]))
    if len(markers)>100:
        raise ValueError('Use at most 100 markers in one recipe.')
    return markers


def matrix(frame,markers):
    if any(m not in frame for m in markers):
        raise ValueError('A recipe marker is missing from this dataset.')
    data=np.column_stack([pd.to_numeric(frame[m],errors='coerce').to_numpy(float) for m in markers])
    data[~np.isfinite(data)]=np.nan
    return data


def groups(frame,step):
    column=step.get('group','')
    if column:
        if column not in frame:
            raise ValueError('Normalization grouping column is missing: '+column)
        return frame[column].astype('string')
    return pd.Series(['all']*len(frame),dtype='string')


def fit_step(data,frame,step,reference,markers):
    op=step['op'];fitted={};grouping=groups(frame,step)
    for group in grouping.dropna().unique():
        keep=grouping.eq(group).fillna(False).to_numpy(bool)&reference
        params={}
        for i,marker in enumerate(markers):
            values=data[keep,i];values=values[np.isfinite(values)]
            if not len(values):
                continue
            item={'n':int(len(values))}
            if op=='zscore':
                item.update(offset=float(values.mean()),scale=float(values.std(ddof=0)))
            elif op=='robust_zscore':
                median=np.median(values)
                item.update(offset=float(median),scale=float(np.median(np.abs(values-median))*1.4826))
            elif op=='minmax':
                item.update(low=float(values.min()),high=float(values.max()))
            elif op in ('clip','percentile'):
                low,high=float(step.get('low',1)),float(step.get('high',99))
                if not 0<=low<high<=100:
                    raise ValueError('Percentiles must satisfy 0 ≤ low < high ≤ 100.')
                lo,hi=np.percentile(values,[low,high])
                item.update(low=float(lo),high=float(hi))
            elif op=='arcsinh':
                p=float(step.get('cofactor_percentile',5))
                if not 0<=p<=100:
                    raise ValueError('Dynamic cofactor percentile must be 0–100.')
                c=float(np.percentile(np.abs(values),p))
                if c<=0:
                    raise ValueError('Dynamic cofactor is zero for '+marker+'. Use a fixed cofactor or another reference percentile.')
                item['cofactor']=c
            params[marker]=item
        fitted[str(group)]=params
    return fitted


def apply_step(data,frame,step,markers):
    from .flow_gating import axis_values
    op=step['op']
    if not step.get('enabled',True) or op=='identity':
        return data
    if 'fitted' in step:
        output=np.full(data.shape,np.nan);grouping=groups(frame,step)
        for group,params in step['fitted'].items():
            keep=grouping.eq(group).fillna(False).to_numpy(bool)
            for i,marker in enumerate(markers):
                item=params.get(marker)
                if not item:
                    continue
                values=data[keep,i]
                if op=='arcsinh':
                    values=np.arcsinh(values/item['cofactor'])
                elif op in ('zscore','robust_zscore'):
                    values=(values-item['offset'])/(item['scale'] or 1.)
                elif op=='clip':
                    values=np.clip(values,item['low'],item['high'])
                elif op in ('percentile','minmax'):
                    values=(values-item['low'])/((item['high']-item['low']) or 1.)
                    if step.get('clip',op=='percentile'):
                        values=np.clip(values,0,1)
                output[keep,i]=values
        return output
    with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
        if op in ('arcsinh','log10','logicle'):
            output=np.column_stack([axis_values(data[:,i],{**step,'scale':op}) for i in range(len(markers))])
        elif op=='log1p':
            output=np.where(data>-1,np.log1p(data),np.nan)
        elif op=='sqrt':
            output=np.where(data>=0,np.sqrt(data),np.nan)
        elif op=='cdf':
            output=ndtr(data)
        elif op=='nlog':
            from spacemap.ph_thres_utils import NORMALIZATION_EPSILON
            output=-np.log(np.maximum(1-data,NORMALIZATION_EPSILON))
        elif op=='row_zscore':
            indices=[markers.index(m) for m in dict.fromkeys(step.get('markers',markers))]
            values=data[:,indices]
            valid=np.isfinite(values);n=valid.sum(axis=1,keepdims=True)
            mean=np.where(valid,values,0).sum(axis=1,keepdims=True)/np.maximum(n,1)
            sd=np.sqrt(np.where(valid,(values-mean)**2,0).sum(axis=1,keepdims=True)/np.maximum(n,1))
            scaled=(values-mean)/np.where(sd==0,1,sd);scaled[~valid]=np.nan
            output=data.copy();output[:,indices]=scaled
        elif op=='relu':
            threshold=float(step.get('threshold',0));ceiling=float(step.get('ceiling',1))
            if not np.isfinite([threshold,ceiling]).all() or ceiling<=threshold:
                raise ValueError('ReLU ceiling must exceed its threshold in the preceding step’s units.')
            output=np.clip((data-threshold)/(ceiling-threshold),0,1)
        elif op=='linear':
            multiplier=float(step.get('multiplier',1));offset=float(step.get('offset',0))
            if not np.isfinite([multiplier,offset]).all():
                raise ValueError('Linear scale and offset must be finite.')
            output=data*multiplier+offset
        else:
            raise ValueError('Fit the transformation recipe on its reference cells first.')
    output[~np.isfinite(output)]=np.nan
    return output


def fit(frame,spec,reference=None):
    spec=deepcopy(spec);markers=validate(spec);legacy_markers=spec.get('markers',markers);spec['markers']=markers
    reference=np.ones(len(frame),bool) if reference is None else np.asarray(reference,bool)
    if not reference.any():
        raise ValueError('The normalization reference has no cells.')
    for step in spec['pipeline']:
        step.pop('fitted',None)
        if step.get('op')=='row_zscore' and 'markers' not in step:
            step['markers']=list(legacy_markers)
    key=cache_key(frame,spec,reference)
    existing=cached(frame,key)
    if existing is not None:
        return {**spec, 'pipeline':deepcopy(existing['pipeline']), 'pipeline_fitted':True,
                'reference_n':int(reference.sum())}
    data=matrix(frame,markers)
    for step in spec['pipeline']:
        step.pop('fitted',None)
        if not step.get('enabled',True):
            continue
        if step['op'] in ('zscore','robust_zscore','minmax','clip','percentile') or (step['op']=='arcsinh' and step.get('dynamic')):
            step['fitted']=fit_step(data,frame,step,reference,markers)
        data=apply_step(data,frame,step,markers)
    spec.update(pipeline_fitted=True,reference_n=int(reference.sum()),markers=markers)
    remember(frame,key,{'pipeline':deepcopy(spec['pipeline'])})
    remember(frame,cache_key(frame,spec),data)
    return spec


def transform(frame,spec):
    markers=validate(spec)
    if not spec.get('pipeline_fitted'):
        raise ValueError('Fit this transformation recipe before using it.')
    key=cache_key(frame,{**spec,'markers':markers})
    data=cached(frame,key)
    if data is not None:
        return data[:,markers.index(spec['column'])]
    data=matrix(frame,markers)
    for step in spec['pipeline']:
        data=apply_step(data,frame,step,markers)
    remember(frame,key,data)
    return data[:,markers.index(spec['column'])]
