"""Exact gate-aware statistics. Display sampling never enters hypothesis tests.

Designs and results are JSON records so additional metrics, tests and figure
renderers can be added without coupling the scientific engine to the UI.
"""
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from . import flow_gating as flow

METRICS = {'mean': 'Mean intensity', 'median': 'Median intensity',
           'positive': '% positive', 'proportion': '% of denominator', 'count': 'Cell count'}
TESTS = {'auto', 'mannwhitney', 'welch', 'student', 'wilcoxon', 'paired_t', 'kruskal', 'anova'}


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def describe(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(N=0, mean=None, median=None, q1=None, q3=None, iqr=None, lower_whisker=None, upper_whisker=None)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return dict(N=len(values), mean=float(np.mean(values)), median=float(median),
                q1=float(q1), q3=float(q3), iqr=float(q3-q1),
                lower_whisker=float(values[values>=q1-1.5*(q3-q1)].min()),
                upper_whisker=float(values[values<=q3+1.5*(q3-q1)].max()))


def _test(a, b, method):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(len(a), len(b)) < 2:
        return {'p': None, 'statistic': None, 'status': 'At least two eligible units per group are required.'}
    if np.all(a == a[0]) and np.all(b == a[0]):
        return dict(p=1., statistic=0., status='Identical constant values')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if method == 'mannwhitney':
            test = stats.mannwhitneyu(a, b, alternative='two-sided', method='auto')
        elif method in ('welch', 'student'):
            test = stats.ttest_ind(a, b, equal_var=method == 'student')
        elif method == 'paired_t':
            test = stats.ttest_rel(a, b)
        elif method == 'wilcoxon':
            diff = a-b
            if not np.any(diff):
                return dict(p=1., statistic=0., status='All matched differences are zero')
            test = stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox', method='auto')
        else:
            raise ValueError('Choose a supported two-group test.')
    return dict(p=finite_number(test.pvalue), statistic=finite_number(test.statistic),
                status='ok' if np.isfinite(test.pvalue) else 'Test is undefined for these values.')


def pair(a, b, method, paired=False):
    av, bv = a['values'], b['values']
    matched = None
    if paired:
        # One observation per biological unit was already aggregated above.
        aa, bb = dict(zip(a['units'], av)), dict(zip(b['units'], bv))
        matched = sorted(aa.keys() & bb.keys())
        av, bv = np.array([aa[k] for k in matched]), np.array([bb[k] for k in matched])
    return {**_test(av, bv, method), 'test': method, 'N_a':len(av), 'N_b':len(bv),
            'matched_N':len(matched) if matched is not None else None,
            'unmatched_a':len(a['values'])-len(av), 'unmatched_b':len(b['values'])-len(bv),
            **{'tested_'+side+'_'+key:value for side,values in [('a',av),('b',bv)] for key,value in describe(values).items()},
            'delta':float(np.mean(bv)-np.mean(av)) if len(av) and len(bv) else None}


def compare(groups, method):
    """Omnibus plus pairwise tests. Dunn uses pooled, tie-corrected ranks.

    Tukey p is separately labelled as familywise-adjusted. Nominal p for each
    ANOVA pair uses its pooled residual variance; BH applies to nominal p only.
    """
    paired = method in ('wilcoxon', 'paired_t')
    if len(groups) == 2 or paired:
        use = 'mannwhitney' if method in ('auto', 'kruskal') else 'student' if method == 'anova' else method
        return [dict(a=a['id'], b=b['id'], **pair(a,b,use,paired)) for a,b in combinations(groups,2)]
    if method not in ('auto','kruskal','anova'):
        raise ValueError('For three or more independent groups choose Kruskal–Wallis / Dunn or ANOVA / Tukey.')
    arrays = [g['values'] for g in groups]
    if any(len(a) < 2 for a in arrays):
        return [dict(a='all',b='',test=method,p=None,status='At least two eligible units in every group are required.')]
    joined = np.concatenate(arrays)
    constant = np.all(joined == joined[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        omnibus = None if constant else (stats.f_oneway(*arrays) if method == 'anova' else stats.kruskal(*arrays))
    rows = [dict(a='all',b='',test='anova' if method=='anova' else 'kruskal',p=1. if constant else finite_number(omnibus.pvalue),
                 statistic=0. if constant else finite_number(omnibus.statistic),status='ok')]
    n, k = len(joined), len(groups)
    if method == 'anova':
        mse = sum(float(np.sum((a-a.mean())**2)) for a in arrays)/(n-k)
    else:
        ranks = stats.rankdata(joined)
        _, counts = np.unique(joined, return_counts=True)
        variance = n*(n+1)/12 - np.sum(counts.astype(float)**3-counts)/(12*(n-1))
        offsets = np.r_[0,np.cumsum([len(a) for a in arrays])]
        means = [ranks[offsets[i]:offsets[i+1]].mean() for i in range(k)]
    for i,j in combinations(range(k),2):
        a,b=arrays[i],arrays[j]
        row=dict(a=groups[i]['id'],b=groups[j]['id'],N_a=len(a),N_b=len(b),delta=float(b.mean()-a.mean()),status='ok')
        if method == 'anova':
            se=np.sqrt(mse*(1/len(a)+1/len(b)))
            t=abs(a.mean()-b.mean())/se if se else (0. if a.mean()==b.mean() else np.inf)
            row.update(test='anova_pair',statistic=finite_number(t),p=float(2*stats.t.sf(t,n-k)),
                       tukey_p=float(stats.studentized_range.sf(t*np.sqrt(2),k,n-k)))
        else:
            se=np.sqrt(variance*(1/len(a)+1/len(b)))
            z=abs(means[i]-means[j])/se if se else 0.
            row.update(test='dunn',statistic=float(z),p=float(2*stats.norm.sf(z)))
        rows.append(row)
    return rows


def _aggregate(frame, masks, design):
    groups, features = design['groups'], design['features']
    unit = design.get('group_by') or None
    if unit and unit not in frame:
        raise ValueError('The biological-unit column is unavailable.')
    minimum = int(design.get('min_cells',20))
    if minimum < 1:
        raise ValueError('Minimum cells must be at least one.')
    unit_values = frame[unit].astype('string') if unit else None
    if unit:
        unit_values=unit_values.str.strip().replace('',pd.NA)
    summaries, samples, arrays, notices = [], [], {}, []
    if not unit:
        notices.append('Pooled cells: N counts cells. P-values describe cell-level variation, not independent biological replication.')
    for feature in features:
        fid, metric = feature['id'],feature['metric']
        if metric not in METRICS:
            raise ValueError('Unknown quantitative metric.')
        if metric=='positive':
            threshold=feature.get('threshold')
            if not threshold:
                raise ValueError('Select a named or active threshold for % positive.')
            x=flow.measurement(frame,threshold['x'])
            values=np.where(np.isfinite(x), (x > float(threshold['bounds'][0])).astype(float)*100, np.nan)
        elif metric in ('proportion','count'):
            values=np.zeros(len(frame))
        else:
            values=flow.measurement(frame,feature.get('axis') or {'column':feature['column'],'scale':'linear'})
        arrays[fid]=[]
        for group in groups:
            gid=group['id']; population=group.get('population',gid)
            if population not in masks:
                raise ValueError('A selected population no longer exists.')
            selected=masks[population][0] & flow.filter_rule(frame,group.get('rule'))[0]
            denominator=selected
            if metric in ('proportion','count'):
                if feature.get('target_population') and feature.get('target_rule'):
                    raise ValueError('Choose a population gate or a category filter, not both.')
                if feature.get('target_population'):
                    target=feature['target_population']
                    if target not in masks:
                        raise ValueError('The target population no longer exists.')
                    selected=selected & masks[target][0]
                if feature.get('target_rule'):
                    selected=selected & flow.filter_rule(frame,feature['target_rule'])[0]
            if metric=='proportion':
                parent=feature.get('denominators',{}).get(gid,feature.get('denominator','root'))
                if parent!='@group':
                    if parent not in masks:
                        raise ValueError('The denominator population no longer exists.')
                    denominator=masks[parent][0]
                if np.any(selected & ~denominator):
                    raise ValueError('The proportion denominator must contain every cell in the target gate.')
                values=selected.astype(float)*100
            elif metric=='count':
                values=selected.astype(float)
            ids=np.flatnonzero(denominator)
            y=values[ids]
            measured=np.isfinite(y)
            if metric=='positive':
                # User-defined denominator includes all cells, not just measured cells.
                y=np.nan_to_num(y,nan=0.)
            local=[]
            missing_units=0
            if unit:
                data=pd.DataFrame({'unit':unit_values.iloc[ids].to_numpy(),'value':y,'valid':measured.astype(int),
                                   'target':selected[ids].astype(int)})
                missing_units=int(data['unit'].isna().sum())
                agg=data.groupby('unit',sort=True,dropna=True).agg(n_cells=('value','size'),n_valid=('valid','sum'),
                    n_target=('target','sum'),value=('value','sum' if metric=='count' else 'median' if metric=='median' else 'mean'))
                excluded=int((agg.n_cells<minimum).sum())
                for key,row in agg.loc[(agg.n_cells>=minimum)&agg.value.notna()].iterrows():
                    local.append(dict(feature=fid,group=gid,unit=str(key),value=float(row.value),n_cells=int(row.n_cells),
                                      n_valid=int(row.n_valid),n_target=int(row.n_target)))
                av=np.array([r['value'] for r in local],float); units=[r['unit'] for r in local]
            else:
                # Counts and proportions describe one pooled group, not millions
                # of independent observations. Keep totals but never test them.
                av=(np.array([y.sum() if metric=='count' else y.mean()]) if len(y) else np.array([])) if metric in ('count','proportion') else y[np.isfinite(y)]
                units=[];excluded=0
            arrays[fid].append(dict(id=gid,values=av,units=units))
            summary=dict(feature=fid,group=gid,label=group.get('label',gid),colour=group.get('colour','#4e79a7'),
                         n_cells=int(selected.sum()),denominator_cells=len(ids),n_valid=int(measured.sum()),
                         missing_units=missing_units,excluded_units=excluded,**describe(av))
            summaries.append(summary)
            if unit:
                samples.extend(local)
            elif metric in ('count','proportion'):
                samples.extend(dict(feature=fid,group=gid,unit='pooled group',value=float(v),n_cells=len(ids),n_target=int(selected.sum())) for v in av)
            else:
                # Display is deterministic and capped; all cells were used by tests and summaries.
                chosen=np.linspace(0,len(av)-1,min(1500,len(av)),dtype=int) if len(av) else []
                samples.extend(dict(feature=fid,group=gid,unit='cell',value=float(av[i])) for i in chosen)
            if missing_units:
                notices.append(f"{group.get('label',gid)}: {missing_units:,} cells have no {unit} ID and were excluded.")
    return summaries,samples,arrays,notices


def run(frame, masks, design):
    groups,features=design.get('groups',[]),design.get('features',[])
    if not 2<=len(groups)<=32 or not 1<=len(features)<=128:
        raise ValueError('Choose 2–32 groups and 1–128 features.')
    for records in (groups,features):
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Group and feature identifiers must be unique.')
    method=design.get('test','auto')
    if method not in TESTS:
        raise ValueError('Unknown statistical test.')
    unit=design.get('group_by')
    if not unit and method in ('wilcoxon','paired_t'):
        raise ValueError('Paired tests require a biological-unit column.')
    summaries,samples,arrays,notices=_aggregate(frame,masks,design)
    tests=[]
    for fid,groups_data in arrays.items():
        tests.extend(dict(feature=fid,family='comparison',**r) for r in compare(groups_data,method))
        if unit and method not in ('wilcoxon','paired_t'):
            for a,b in combinations(groups_data,2):
                shared=set(a['units'])&set(b['units'])
                if shared:
                    notices.append(f"{len(shared)} {unit} IDs occur in both {a['id']} and {b['id']}; independent tests assume these observations are independent. Consider a paired test.")
    heat=design.get('heatmap') or {}
    heat_values=[];delta_groups={}
    mode=heat.get('mode','absolute');matched=heat.get('matched',True)
    if mode not in ('absolute','delta','log2'):
        raise ValueError('Choose absolute, delta or log2 fold change.')
    if mode!='absolute':
        if matched and not unit:
            raise ValueError('Patient-matched deltas require a biological-unit column.')
        pseudo=float(heat.get('pseudocount',0))
        if not np.isfinite(pseudo) or pseudo<0:
            raise ValueError('Pseudocount must be a finite non-negative number.')
        for feature in features:
            fid=feature['id'];byid={g['id']:g for g in arrays[fid]};delta_groups[fid]=[]
            for h in heat.get('pairs',[]):
                if h['before'] not in byid or h['after'] not in byid or h['before']==h['after']:
                    raise ValueError('Choose distinct before and after groups for every delta column.')
                a,b=byid[h['before']],byid[h['after']]
                testmethod=('paired_t' if method in ('student','welch','anova','paired_t') else 'wilcoxon') if matched else ('welch' if method in ('student','welch','anova','paired_t') else 'mannwhitney')
                test=pair(a,b,testmethod,matched)
                av,bv=a['values'],b['values'];keys=[]
                if matched:
                    aa,bb=dict(zip(a['units'],av)),dict(zip(b['units'],bv));keys=sorted(aa.keys()&bb.keys())
                    av,bv=np.array([aa[k] for k in keys]),np.array([bb[k] for k in keys])
                if not matched and len(av) and len(bv):
                    aggregate=np.median if heat.get('summary')=='median' else np.mean
                    av,bv=np.array([aggregate(av)]),np.array([aggregate(bv)])
                differences=bv-av if len(av) and len(bv) else np.array([])
                reason=''
                if not len(av) or not len(bv):
                    reason='No eligible before/after observations.'
                elif mode=='log2':
                    if np.any(av+pseudo<=0) or np.any(bv+pseudo<=0):
                        differences=np.array([]);reason='Log2 fold change requires positive before/after values. Choose an explicit pseudocount or absolute delta.'
                    else:
                        differences=np.log2((bv+pseudo)/(av+pseudo))
                value=float((np.median if heat.get('summary')=='median' else np.mean)(differences)) if len(differences) else None
                test.update(feature=fid,family='heatmap',a=h['before'],b=h['after'],heat_column=h['id'],heatmap_value=value,heatmap_mode=mode)
                tests.append(test)
                heat_values.append(dict(feature=fid,column=h['id'],label=h.get('label',h['id']),value=value,
                    delta=value,matched_N=len(keys) if matched else None,status=reason or test['status']))
                if matched and len(differences):
                    delta_groups[fid].append(dict(id=h['id'],values=differences,units=keys))
        if heat.get('compare_deltas'):
            if not matched:
                raise ValueError('Comparing deltas across cohorts requires patient-matched individual deltas.')
            for fid,ds in delta_groups.items():
                if len(ds)>1:
                    tests.extend(dict(feature=fid,family='delta_comparison',**r) for r in compare(ds,heat.get('delta_test','auto')))
    else:
        for s in summaries:
            heat_values.append(dict(feature=s['feature'],column=s['group'],label=s['label'],value=s['median' if heat.get('summary')=='median' else 'mean'],delta=None,status='Descriptive absolute value; no before/after test.'))
    if not unit:
        proportions={f['id'] for f in features if f['metric'] in ('proportion','count')}
        for test in tests:
            if test['feature'] in proportions:
                test.update(p=None,statistic=None,status='Pooled population counts and proportions are descriptive. Choose a biological unit for inference.')
        if proportions:
            notices.append('Population counts and proportions: one pooled total per group; no p-values. Choose a biological unit to compare patients or samples.')
    correction=design.get('correction','fdr_bh')
    if correction not in ('fdr_bh','bonferroni'):
        raise ValueError('Choose Benjamini–Hochberg or Bonferroni correction.')
    valid=[r for r in tests if r.get('p') is not None]
    adjusted=multipletests([r['p'] for r in valid],method=correction)[1] if valid else []
    for r in tests:r['q']=None
    for r,q in zip(valid,adjusted):r['q']=float(q)
    for value in heat_values:
        match=next((r for r in tests if r['family']=='heatmap' and r['feature']==value['feature'] and r.get('heat_column')==value['column']),{})
        value.update(p=match.get('p'),q=match.get('q'))
    # Detect reused cells independently of unit ID spelling.
    for a,b in combinations(groups,2):
        overlap=int(np.count_nonzero(masks[a.get('population',a['id'])][0] & masks[b.get('population',b['id'])][0] &
                                    flow.filter_rule(frame,a.get('rule'))[0] & flow.filter_rule(frame,b.get('rule'))[0]))
        if overlap:
            notices.append(f"{a.get('label',a['id'])} and {b.get('label',b['id'])} share {overlap:,} cells; comparisons are not independent.")
    return dict(version=1,design=design,summaries=summaries,samples=samples,tests=tests,heatmap=heat_values,
                notices=list(dict.fromkeys(notices)),correction=correction,hypotheses=len(valid),
                correction_family='All valid omnibus, pairwise, heatmap and delta-comparison hypotheses in this run.',
                unit=unit or 'cells',sampled_display=not bool(unit))
