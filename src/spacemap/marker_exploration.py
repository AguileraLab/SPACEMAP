"""Marker exploration: one set of measurements and denominators for every view.

The caller supplies the selected cells. No labels are assigned and no missing
measurements are imputed. Plotting and exports use the same saved summaries.
"""
from contextlib import contextmanager
import math
import re
import threading
import textwrap

import numpy as np
import pandas as pd

PLOT_LOCK = threading.RLock()
KINDS = {'review': 'Marker review', 'dot': 'Marker × phenotype', 'histogram': 'Marker distributions',
         'patients': 'High expression by patient', 'composition': 'Phenotypes among high cells',
         'variation': 'Marker variation', 'coexpression': 'Marker combinations',
         'stratified': 'Patients by group', 'groups': 'Group comparison'}
APPEARANCE_DEFAULTS = dict(font_size=10, x_size=9, y_size=10, x_angle=None, y_angle=0,
                           width=100, height=100, wrap=24, label_pad=6, show_values=True)


def appearance_options(body):
    """Display settings only; these never change measurements or cutoffs."""
    if not isinstance(body, dict) or set(body) - set(APPEARANCE_DEFAULTS):
        raise ValueError('Choose valid figure appearance settings.')
    found = {**APPEARANCE_DEFAULTS, **body}
    for key, low, high in [('font_size', 6, 24), ('x_size', 6, 24), ('y_size', 6, 24),
                           ('x_angle', -90, 90), ('y_angle', -90, 90),
                           ('width', 50, 200), ('height', 50, 200), ('wrap', 0, 60), ('label_pad', 0, 30)]:
        value = found[key]
        if key == 'x_angle' and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f'{key.replace("_", " ")} must be between {low} and {high}.')
    if int(found['wrap']) != found['wrap']:
        raise ValueError('Label wrap must be a whole number.')
    if not isinstance(found['show_values'], bool):
        raise ValueError('Choose whether to show heatmap values.')
    return found


def display_name(marker):
    return re.sub(r'_corr$', '', marker)


def natural_key(value):
    return [(0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r'(\d+)', str(value))]


def summarize(frame, parent, markers, *, patient='', z=1.5, min_cells=100,
              transform='auto', cofactor=5., pair=None, stratify=None, kinds=None, report=lambda *a: None):
    """Exact finite-cell summaries; thresholds share the selected-cell reference.

    Auto transforms raw channels once with arcsinh(value/cofactor), preserving
    *_corr channels. Display subsets never recalculate reference statistics.
    Optional kinds limit expensive distribution and patient calculations.
    """
    kinds = set(KINDS if kinds is None else kinds)
    if not kinds & {'patients', 'stratified', 'groups'}:
        stratify = []
    if not kinds & {'histogram', 'patients', 'stratified', 'groups'}:
        patient = ''
    if 'coexpression' not in kinds:
        pair = []
    frame = frame.reset_index(drop=True)
    groups = frame[parent].astype('string')
    valid = groups.notna() & groups.str.strip().ne('')
    frame, groups = frame.loc[valid].reset_index(drop=True), groups.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError('No labelled cells match this selection.')
    from .marker_preprocessing import transform as transform_markers
    values = transform_markers(frame, markers, transform, cofactor)
    indices = groups.groupby(groups, sort=False).indices
    if patient:
        patients = frame[patient].astype('string').str.strip()
        patients = patients.mask(patients.eq(''))
        if patients.nunique() > 500:
            raise ValueError('The sample column has over 500 values. Choose sample categories, not cell IDs.')
    else:
        patients = pd.Series(pd.NA, index=frame.index, dtype='string')
    stratify = stratify or []
    strata, units, stratified = {}, {}, []
    unit_indices = {p: {} for p in indices}
    missing_strata = 0
    if stratify:
        annotations = frame[stratify].astype('string').apply(lambda c: c.str.strip().replace('', pd.NA))
        complete = annotations.notna().all(axis=1)
        missing_strata = int((~complete).sum())
        codes = pd.Series(pd.NA, index=frame.index, dtype='string')
        for i, (key, idx) in enumerate(annotations[complete].groupby(stratify, sort=True).groups.items()):
            key = key if isinstance(key, tuple) else (key,)
            code = f's{i+1}'
            strata[code] = ' · '.join(f'{c}={v}' for c, v in zip(stratify, key))
            codes.loc[idx] = code
        if len(strata) > 60:
            raise ValueError('These columns create over 60 groups. Choose fewer categories, not cell IDs.')
        unit_ids = codes + (':' + patients if patient else '')
        for code in strata:
            pats = sorted(patients[codes.eq(code).fillna(False)].dropna().unique(), key=natural_key) if patient else ['']
            for pat in pats:
                unit = code + (':' + pat if patient else '')
                units[unit] = dict(patient=pat, stratum=code)
        for p, idx in indices.items():
            unit_indices[p] = unit_ids.iloc[idx].groupby(unit_ids.iloc[idx], sort=False).indices
    thresholds, summaries, histograms, by_patient, pairs = [], [], [], [], []
    need_patient = 'histogram' in kinds or ('patients' in kinds and not stratify)
    patient_indices = {p: (patients.iloc[idx].groupby(patients.iloc[idx], sort=False).indices if need_patient else {})
                       for p, idx in indices.items()}
    for j, m in enumerate(markers):
        report(f'Measuring {display_name(m)} ({j+1}/{len(markers)})…', .08+.55*j/len(markers))
        arr = values[m].to_numpy()
        finite = arr[np.isfinite(arr)]
        mean = float(finite.mean()) if len(finite) else np.nan
        sd = float(finite.std()) if len(finite) else np.nan
        cut = mean + z*sd
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        if hi <= lo:
            lo, hi = lo-.5, hi+.5
        edges = np.linspace(lo, hi, 61)
        thresholds.append(dict(marker=m,mean=mean,sd=sd,cutoff=cut,n_finite=len(finite),
                               n_missing=len(frame)-len(finite)))
        for p, idx in indices.items():
            x = arr[idx]; x = x[np.isfinite(x)]
            q = np.quantile(x, [.05,.25,.5,.75,.95]) if len(x) else np.full(5,np.nan)
            high = int((x>cut).sum())
            summaries.append(dict(phenotype=str(p),marker=m,n_cells=len(idx),n_finite=len(x),
                n_high=high,pct_high=100*high/len(x) if len(x) else np.nan,
                p05=q[0],q25=q[1],median=q[2],q75=q[3],p95=q[4],
                median_z=(q[2]-mean)/sd if sd>0 else (0. if len(x) else np.nan),
                iqr=q[3]-q[1],sd=float(x.std()) if len(x) else np.nan,
                relative_iqr=(q[3]-q[1])/sd if sd>0 else (0. if len(x) else np.nan),
                relative_sd=float(x.std())/sd if sd>0 and len(x) else (0. if len(x) else np.nan)))
            balanced=[]
            for pat, local_idx in patient_indices[p].items():
                px = arr[idx[local_idx]]; px=px[np.isfinite(px)]
                phigh=int((px>cut).sum())
                by_patient.append(dict(phenotype=str(p),marker=m,patient=str(pat),n_cells=len(local_idx),
                    n_finite=len(px),n_high=phigh,pct_high=100*phigh/len(px) if len(px) else np.nan,
                    median=float(np.median(px)) if len(px) else np.nan))
                if 'histogram' in kinds and len(px)>=min_cells:
                    balanced.append(np.histogram(px,edges)[0]/len(px)*100)
            if 'histogram' in kinds:
                histograms.append(dict(phenotype=str(p),marker=m,edges=edges.tolist(),counts=np.histogram(x,edges)[0].tolist(),
                    balanced=(np.mean(balanced,axis=0) if balanced else np.full(60,np.nan)).tolist(),
                    eligible_patients=len(balanced)))
            for unit, local_idx in unit_indices[p].items():
                ux = arr[idx[local_idx]]; ux = ux[np.isfinite(ux)]
                high = int((ux > cut).sum())
                stratified.append(dict(phenotype=str(p), marker=m, unit=unit, **units[unit],
                    n_cells=len(local_idx), n_finite=len(ux), n_high=high,
                    pct_high=100*high/len(ux) if len(ux) else np.nan))
    if pair and len(pair)==2:
        a,b=pair
        for p,idx in indices.items():
            xy=values.iloc[idx][pair].dropna()
            if len(xy):
                counts,xe,ye=np.histogram2d(xy[a],xy[b],bins=60)
                rho=float(xy.corr(method='spearman').iloc[0,1]) if xy[a].nunique()>1 and xy[b].nunique()>1 else np.nan
                pairs.append(dict(phenotype=str(p),marker_x=a,marker_y=b,n_finite=len(xy),
                    xedges=xe.tolist(),yedges=ye.tolist(),counts=counts.ravel().astype('int64').tolist(),spearman=rho))
    summary=pd.DataFrame(summaries)
    high_totals=summary.groupby('marker').n_high.transform('sum')
    summary['composition_pct']=np.where(high_totals>0,100*summary.n_high/high_totals,np.nan)
    strata_table=pd.DataFrame(stratified,columns=['phenotype','marker','unit','patient','stratum','n_cells','n_finite','n_high','pct_high'])
    group_rows=[]
    group_iterator = strata_table.groupby(['phenotype','marker','stratum'], sort=False) if 'groups' in kinds else []
    for (p,m,code), group in group_iterator:
        eligible=group[group.n_finite>=min_cells]
        group_rows.append(dict(phenotype=p,marker=m,stratum=code,n_units=len(group),n_eligible=len(eligible),
                               pct_high=float(eligible.pct_high.mean()) if len(eligible) else np.nan,
                               n_finite=int(group.n_finite.sum()),n_high=int(group.n_high.sum())))
    return {'thresholds':pd.DataFrame(thresholds),'summary':summary,
            'histograms':pd.DataFrame(histograms,columns=['phenotype','marker','edges','counts','balanced','eligible_patients']),
            'patients':pd.DataFrame(by_patient,columns=['phenotype','marker','patient','n_cells','n_finite','n_high','pct_high','median']),
            'stratified':strata_table,
            'groups':pd.DataFrame(group_rows,columns=['phenotype','marker','stratum','n_units','n_eligible','pct_high','n_finite','n_high']),
            'coexpression':pd.DataFrame(pairs,columns=['phenotype','marker_x','marker_y','n_finite','xedges','yedges','counts','spearman']),
            'metadata':{'cells':len(frame),'phenotypes':[str(p) for p in indices], 'markers':markers,
                'patient_column':patient,'patients':int(patients.nunique()),
                'missing_patient_cells':int(patients.isna().sum()) if patient else 0,
                'stratify':stratify,'strata':strata,'units':units,'missing_stratum_cells':missing_strata,
                'z':z,'min_cells':min_cells,'transform':transform,'cofactor':cofactor,'pair':pair or [],
                'reference':'All finite measurements among the selected phenotypes and region filters.',
                'threshold_rule':'value > mean + z * population SD (ddof=0); one fixed cutoff per marker',
                'sampling':'None. All finite measurements are used.',
                'missing':'Missing/nonfinite measurements are excluded per marker, never replaced with zero.'}}


@contextmanager
def plot_style():
    import matplotlib
    with PLOT_LOCK, matplotlib.rc_context({'font.size':10,'font.family':'DejaVu Sans',
            'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
        yield


def draw(bundle, kind, *, phenotype=None, markers=None, variation='iqr', patients=None, appearance=None):
    if kind == 'review':
        from .marker_review import draw as draw_review
        return draw_review(bundle, phenotype, markers)

    """Create a figure from saved summaries; must be used within plot_style()."""
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib import colormaps
    style = appearance_options({} if appearance is None else appearance)
    meta=bundle['metadata']; markers=markers or meta['markers']
    if kind not in KINDS or not markers or any(m not in meta['markers'] for m in markers):
        raise ValueError('Choose a listed figure and its saved markers.')
    if phenotype is not None and phenotype not in meta['phenotypes']:
        raise ValueError('Choose a phenotype from this figure run.')
    if variation not in {'iqr','sd','relative_iqr','relative_sd'}:
        raise ValueError('Choose IQR or standard deviation for variation.')
    s=bundle['summary']; t=bundle['thresholds'].set_index('marker')
    parents=[phenotype] if phenotype else meta['phenotypes']
    def name(marker):
        # Keep raw and corrected channels distinguishable if both were selected.
        return marker if marker.endswith('_corr') and marker[:-5] in meta['markers'] else display_name(marker)
    high_note=f"High: Z > {meta['z']:g} in the saved selection. Exploratory, not a biological positive gate."
    blue='#247c9a'; red='#a73d47'
    fig=Figure(figsize=(12,7))
    if kind=='dot':
        fig.set_size_inches(max(10,.42*len(markers)+3),max(5,.4*len(parents)+2.8))
        ax=fig.subplots(); fig.subplots_adjust(left=.19,right=.83,bottom=.28,top=.83)
        for y,p in enumerate(parents):
            if y%2==0: ax.axhspan(y-.5,y+.5,color='#f1f5f7',zorder=0)
            for x,m in enumerate(markers):
                v=s[(s.phenotype==p)&(s.marker==m)].iloc[0]
                if not v.n_finite: ax.text(x,y,'×',ha='center',va='center',color='#999')
                elif not v.pct_high: ax.scatter(x,y,s=8,facecolors='none',edgecolors='#aab6bf',linewidths=.5)
                else: ax.scatter(x,y,s=v.pct_high*3.5,c=[v.median_z],cmap='RdBu_r',vmin=-2,vmax=2,edgecolors='#83919c',linewidths=.35)
        ax.set(xticks=range(len(markers)),xticklabels=[name(m) for m in markers],
            yticks=range(len(parents)),yticklabels=parents,xlim=(-.65,len(markers)-.35),ylim=(len(parents)-.5,-.65))
        ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
        for label in ax.get_xticklabels(): label.set_rotation(55);label.set_ha('right')
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-2,2),cmap='RdBu_r'),cax=fig.add_axes([.88,.42,.015,.32]),extend='both')
        cb.set_label('Median Z',fontsize=9)
        handles=[ax.scatter([],[],s=v*3.5,color='#71899a',label=f'{v}%') for v in [5,25,50,100]]
        fig.legend(handles=handles,title='% high within phenotype',loc='lower right',bbox_to_anchor=(1,.10),frameon=False,fontsize=9)
        title='Marker × phenotype'; subtitle='Dot area: % high within phenotype · Colour: median expression, standardized per marker'
        high_note+='  × = no measurements; empty ring = zero high cells.'
    elif kind=='composition':
        fig.set_size_inches(max(10,.42*len(markers)+3),7)
        ax=fig.subplots();fig.subplots_adjust(left=.08,right=.80,bottom=.27,top=.83)
        # Composition always uses all selected phenotypes; a display subset must not change its denominator.
        bottom=np.zeros(len(markers));colors=colormaps['tab20'](np.linspace(0,1,len(meta['phenotypes'])))
        for p,color in zip(meta['phenotypes'],colors):
            vals=s[s.phenotype==p].set_index('marker').reindex(markers).composition_pct.to_numpy()
            ax.bar(range(len(markers)),np.nan_to_num(vals),bottom=bottom,color=color,label=p,edgecolor='white',linewidth=.4)
            bottom+=np.nan_to_num(vals)
        for j,total in enumerate(bottom):
            if total==0: ax.text(j,3,'No high cells',rotation=90,ha='center',fontsize=8)
        ax.set(ylim=(0,101),ylabel='% of high-expressing cells',xticks=range(len(markers)),xticklabels=[name(m) for m in markers])
        for label in ax.get_xticklabels(): label.set_rotation(55);label.set_ha('right')
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=9)
        title='Phenotypes among high-expressing cells';subtitle='Each bar divides the high cells for one marker among the selected phenotypes'
    elif kind=='histogram':
        if len(parents)!=1: raise ValueError('Choose one phenotype for distributions.')
        ncols=min(3,len(markers));nrows=math.ceil(len(markers)/ncols)
        fig.set_size_inches(max(6,ncols*4.3),nrows*2.8+2.3)
        axes=fig.subplots(nrows,ncols,squeeze=False)
        height=fig.get_figheight()
        fig.subplots_adjust(left=.10 if ncols==1 else .08,right=.98,bottom=1.15/height,
                            top=1-1.12/height,hspace=.65,wspace=.32)
        h=bundle['histograms'];p=parents[0]
        for ax,m in zip(axes.flat,markers):
            row=h[(h.phenotype==p)&(h.marker==m)].iloc[0];v=s[(s.phenotype==p)&(s.marker==m)].iloc[0]
            edges=np.asarray(row.edges);counts=np.asarray(row.counts)
            if not v.n_finite:
                ax.text(.5,.5,'No measurements',ha='center',transform=ax.transAxes)
            else:
                ax.stairs(100*counts/v.n_finite,edges,fill=True,color=blue,alpha=.6)
                if row.eligible_patients: ax.stairs(np.asarray(row.balanced),edges,color='#c77b20',linewidth=1)
                ax.axvspan(v.q25,v.q75,color=blue,alpha=.13);ax.axvline(v['median'],color='#243746',lw=1.2)
                ax.axvline(t.loc[m,'cutoff'],color=red,ls='--',lw=1)
                ax.text(.98,.95,f'Median {v["median"]:.2f}\nIQR {v.q25:.2f}–{v.q75:.2f}',ha='right',va='top',transform=ax.transAxes,fontsize=8,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
            ax.set_title(f'{name(m)} · {v.pct_high:.1f}% high' if v.n_finite else name(m),loc='left',fontsize=10)
            ax.set(xlabel='Expression (analysis scale)',ylabel='% of cells / bin',xlim=(edges[0],edges[-1]),ylim=(0,None))
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[len(markers):]: ax.axis('off')
        handles=[Line2D([0],[0],color=blue,lw=7,alpha=.6,label='All measured cells'),Line2D([0],[0],color='#243746',label='Median'),
            Line2D([0],[0],color=blue,lw=7,alpha=.15,label='Middle 50%'),Line2D([0],[0],color=red,ls='--',label='High cutoff')]
        if meta['patient_column']: handles.append(Line2D([0],[0],color='#c77b20',label='Equal patient weight'))
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.38/height),
                   ncol=min(3 if ncols==1 else 5,len(handles)),frameon=False,fontsize=8)
        title=p+' · marker distributions';subtitle=f"{int(s[s.phenotype==p].n_cells.iloc[0]):,} selected cells · missing values excluded per marker"
    elif kind in ('patients','stratified','groups'):
        if kind in ('patients','stratified') and not meta['patient_column']:
            raise ValueError('Choose a patient column and generate figures first.')
        u=bundle[kind]
        key={'patients':'patient','stratified':'unit','groups':'stratum'}[kind]
        ids=patients or sorted(u[key].unique(),key=natural_key)
        if not ids or any(p not in set(u[key]) for p in ids): raise ValueError('No groups to display.')
        rows=[(p,m) for p in parents for m in markers]
        arr=np.full((len(rows),len(ids)),np.nan)
        eligible=np.zeros_like(arr)
        lookup=u.set_index(['phenotype','marker',key])
        for i,(p,m) in enumerate(rows):
            for j,pat in enumerate(ids):
                if (p,m,pat) in lookup.index:
                    v=lookup.loc[p,m,pat]
                    if kind=='groups':
                        eligible[i,j]=v.n_eligible
                        if v.n_eligible: arr[i,j]=v.pct_high
                    elif v.n_finite>=meta['min_cells']: arr[i,j]=v.pct_high
        grouped=kind!='patients'
        fig.set_size_inches(max(10,.60*len(ids)+4),max(6 if grouped else 5,(.5 if kind=='groups' else .36)*len(rows)+(4 if grouped else 2.2)))
        ax=fig.subplots();fig.subplots_adjust(left=.25,right=.92,bottom=(2.5/fig.get_figheight() if grouped else .16),top=1-1.1/fig.get_figheight())
        cmap=colormaps['YlGnBu'].with_extremes(bad='#e3e7ea')
        im=ax.imshow(arr,vmin=0,vmax=100,aspect='auto',cmap=cmap)
        for i in range(len(rows)):
            for j in range(len(ids)):
                val=arr[i,j]
                text='—' if not np.isfinite(val) else f'{val:.0f}'
                if kind=='groups' and meta['patient_column'] and np.isfinite(val): text+=f'\nn={int(eligible[i,j])}'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if val>60 else '#243746')
        def group_label(identifier):
            code=meta['units'][identifier]['stratum'] if kind=='stratified' else identifier
            parts=meta['strata'][code].split(' · ')
            return ('Patient '+meta['units'][identifier]['patient']+'\n' if kind=='stratified' else '')+'\n'.join(parts)
        labels=[group_label(identifier) for identifier in ids] if grouped else ids
        ax.set(xticks=range(len(ids)),xticklabels=labels,yticks=range(len(rows)),
               yticklabels=[p+' · '+name(m) for p,m in rows],xlabel='' if grouped else meta['patient_column'])
        for label in ax.get_xticklabels(): label.set_rotation(60);label.set_ha('right')
        fig.colorbar(im,ax=ax,pad=.02,fraction=.025,label='% high within phenotype')
        title={'patients':'High expression across patients','stratified':'High expression · patients by group','groups':'High expression · group comparison'}[kind]
        subtitle=(f"Equal patient weight · n = eligible patients per tile · Minimum {meta['min_cells']} cells per patient"
                  if kind=='groups' and meta['patient_column'] else
                  f"One fixed cutoff per marker · Grey: fewer than {meta['min_cells']} measured cells")
        if kind=='groups' and not meta['patient_column']:
            subtitle=f"Pooled cells per group · No patient column selected · Minimum {meta['min_cells']} measured cells"
    elif kind=='variation':
        if len(parents)!=1: raise ValueError('Choose one phenotype for variation.')
        p=parents[0];v=s[(s.phenotype==p)&s.marker.isin(markers)].sort_values(variation,ascending=True)
        fig.set_size_inches(10,max(5,.28*len(v)+2.2));ax=fig.subplots();fig.subplots_adjust(left=.23,right=.96,bottom=.18,top=.84)
        ax.barh([name(m) for m in v.marker],v[variation],color=blue)
        metric='Middle 50% spread (IQR)' if variation in ('iqr','relative_iqr') else 'Standard deviation'
        ax.set_xlabel(metric+(' / reference SD for that marker' if variation.startswith('relative_') else ' (analysis scale)'))
        for j,val in enumerate(v[variation]):
            if np.isfinite(val): ax.text(val,j,f' {val:.2f}',va='center',fontsize=8)
        ax.margins(x=.15);title=p+' · markers ranked by variation'
        subtitle='Highest variation at the top · '+('Relative to all selected cells' if variation.startswith('relative_') else 'Spread of the measured expression values')
        high_note='Variation describes spread, not biological importance. Noise and technical effects can also increase spread.'
    else:
        if len(parents)!=1: raise ValueError('Choose one phenotype for co-expression.')
        data=bundle['coexpression'];data=data[data.phenotype==parents[0]]
        if data.empty: raise ValueError('Choose two different co-expression markers and generate figures first.')
        v=data.iloc[0];a,b=v.marker_x,v.marker_y
        ax=fig.subplots();fig.subplots_adjust(left=.12,right=.9,bottom=.15,top=.84)
        counts=np.asarray(v.counts).reshape(len(v.xedges)-1,len(v.yedges)-1)
        im=ax.pcolormesh(v.xedges,v.yedges,np.ma.masked_less(counts.T,1),norm=LogNorm(vmin=1,vmax=max(2,counts.max())),cmap='viridis',rasterized=True)
        ax.axvline(t.loc[a,'cutoff'],ls='--',color=red);ax.axhline(t.loc[b,'cutoff'],ls='--',color=red)
        ax.set(xlabel=name(a)+' (analysis scale)',ylabel=name(b)+' (analysis scale)')
        fig.colorbar(im,ax=ax,label='Cells per bin')
        title=parents[0]+' · '+name(a)+' × '+name(b)
        subtitle=f'{v.n_finite:,} measured cells · Spearman correlation {v.spearman:.2f}'
    _apply_appearance(fig, kind, style)
    height=fig.get_figheight()
    scale=style['font_size']/10
    fig.suptitle(title,x=.04,ha='left',y=1-.15*scale/height,va='top',fontsize=16*scale,fontweight='bold')
    fig.text(.04,1-.60*scale/height,textwrap.fill(subtitle,width=max(30,int(fig.get_figwidth()*15/scale))),
             fontsize=9*scale,va='top')
    fig.text(.04,.08/height,textwrap.fill(high_note,width=max(30,int(fig.get_figwidth()*16/scale))),
             fontsize=8*scale,va='bottom')
    return fig


def _apply_appearance(fig, kind, style):
    """Fit categorical labels using their rendered size, including multiline labels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    scale = style['font_size'] / 10
    heatmap = kind in ('patients', 'stratified', 'groups')
    axes = [ax for ax in fig.axes if not hasattr(ax, '_colorbar')]
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * scale)
    wrap = int(style['wrap'])
    def wrapped(label):
        return '\n'.join(textwrap.fill(line, width=wrap, break_on_hyphens=False)
                         for line in label.splitlines()) if wrap else label
    for ax in axes:
        if heatmap or kind in ('dot', 'composition'):
            ax.set_xticks(ax.get_xticks(), [wrapped(t.get_text()) for t in ax.get_xticklabels()])
        if heatmap or kind in ('dot', 'variation'):
            ax.set_yticks(ax.get_yticks(), [wrapped(t.get_text()) for t in ax.get_yticklabels()])
        angle = style['x_angle']
        if angle is None:
            angle = 90 if kind in ('stratified', 'groups') else (55 if kind in ('dot', 'composition') else 0)
        ax.tick_params(axis='x', labelsize=style['x_size'], pad=style['label_pad'])
        ax.tick_params(axis='y', labelsize=style['y_size'], pad=style['label_pad'])
        for label in ax.get_xticklabels():
            label.set_rotation(angle)
            label.set_rotation_mode('anchor')
            label.set_ha('right' if angle > 0 else 'left' if angle < 0 else 'center')
            label.set_va('center' if abs(angle) == 90 else 'top')
        for label in ax.get_yticklabels():
            label.set_rotation(style['y_angle'])
            label.set_rotation_mode('anchor')
        if heatmap and not style['show_values']:
            for text in ax.texts:
                text.set_visible(False)
    width, height = fig.get_size_inches()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    if heatmap or kind == 'dot':
        ax = axes[0]
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        xb = [t.get_window_extent(renderer) for t in xticks]
        yb = [t.get_window_extent(renderer) for t in yticks]
        # Match spacing to the actual text extent, not the number of characters.
        plot_w = max(ax.get_window_extent(renderer).width / fig.dpi,
                     (max((b.width for b in xb), default=0) + 12) * len(xb) / fig.dpi)
        plot_h = max(ax.get_window_extent(renderer).height / fig.dpi,
                     (max((b.height for b in yb), default=0) + 8) * len(yb) / fig.dpi)
        plot_w *= style['width'] / 100
        plot_h *= style['height'] / 100
        left = max(.65, max((b.width for b in yb), default=0) / fig.dpi + .25)
        below = max((ax.bbox.y0 - b.y0 for b in xb), default=0) / fig.dpi
        bottom = max(.8*scale, below + .7*scale + (.25 if ax.get_xlabel() else 0))
        top, right = 1.25*scale, (2.2 if kind == 'dot' else 1.2)*scale
        width, height = left + plot_w + right, bottom + plot_h + top
        fig.set_size_inches(width, height)
        ax.set_position([left/width, bottom/height, plot_w/width, plot_h/height])
        for bar in (a for a in fig.axes if a not in axes):
            bar.set_axes_locator(None)
            bar.set_box_aspect(None)
            bar.set_position([(left+plot_w+.18)/width,
                              (bottom + (plot_h*.55 if kind == 'dot' else 0))/height,
                              .16/width, plot_h*(.4 if kind == 'dot' else 1)/height])
        if kind == 'dot':
            for legend in fig.legends:
                legend.set_bbox_to_anchor((1-.05/width, bottom/height))
    else:
        width *= style['width'] / 100
        height *= style['height'] / 100
        bottom = (1.15 if kind == 'histogram' else .65) * scale
        width, height = max(width, 4*scale), max(height, bottom+1.2*scale+2)
        fig.set_size_inches(width, height)
        fig.tight_layout(rect=(.015, bottom/height, 1-.2/width, 1-1.2*scale/height),
                         pad=.7, h_pad=2, w_pad=1.5)
        if kind == 'histogram':
            for legend in fig.legends:
                legend.set_bbox_to_anchor((.5, .38*scale/height))
