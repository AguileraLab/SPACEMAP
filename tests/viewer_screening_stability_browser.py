"""Ranked stability matrix, exact percentages, ring labels and linked findings."""
from viewer_threshold_browser import main
from viewer_screening_comparisons_browser import DRIVER as COMPARISONS, fixture as comparison_fixture

CHECKS = r"""

 const findings=()=>tools().querySelector('.screening-results-table');
 assert(findings().querySelectorAll('thead button').length===9,'All result columns have sortable headers');
 assert(/Higher in Case|Larger change in Case/.test(findings().textContent)&&findings().querySelector('.screening-group-dot'),'Table shows the named higher group with its colour');
 assert([...findings().querySelectorAll('tbody tr')].some(r=>/\d+%/.test(r.children[6].textContent)),'Recovery percentages are available in the sortable table');
 const sortQ=()=>findings().querySelector('[aria-label="Sort screening results by q"]');
 sortQ().click();assert(sortQ().closest('th').getAttribute('aria-sort')==='descending','First q-header click reverses the default ascending order');
 sortQ().click();assert(sortQ().closest('th').getAttribute('aria-sort')==='ascending','Second q-header click restores ascending order');
 const familyBefore=by('Screening result feature type').value,first=findings().querySelector('tbody tr');
 document.querySelector('[data-pane-collapse="right"]').click();first.click();
 await until(()=>!document.body.classList.contains('studio-hide-right'));
 assert(by('Screening result feature type').value===familyBefore,'Selecting a finding preserves the table feature filter');
 assert(!document.querySelector('#studio-dock').hidden&&document.querySelector('#studio-figures > [data-figure^="screening-"]'),'Clicking a feature restores the right panel and displays its figure');
 assert(document.querySelector('#studio-figures .screening-statistics').textContent.includes('Bootstrap recovery')&&document.querySelector('#studio-figures .screening-statistics').textContent.includes('95% bootstrap interval'),'Selected feature shows bootstrap recovery and the effect interval');
 button(tools(),'Stability across distances').click();
 const overview=()=>document.querySelector('[data-figure="screening-stability"]');
 await until(()=>overview()?.querySelector('.screening-stability-cell'));
 assert([...overview().querySelectorAll('thead th')].some(th=>th.textContent==='50–100 µm')&&![...overview().querySelectorAll('thead th')].some(th=>th.textContent==='0–100 µm'),'Stability columns show actual non-overlapping ring bounds');
 assert(overview().querySelectorAll('tbody tr').length===10,'Matrix ranks features across all five comparisons');
 assert([...overview().querySelectorAll('.screening-stability-cell strong')].some(e=>/\d+%/.test(e.textContent)),'Exact resampling percentages are rendered inside matrix cells');
 assert([...overview().querySelectorAll('.screening-stability-cell')].some(e=>e.title.includes('Higher in Case')&&e.textContent.includes('Case')),'Two-group direction names the actual higher group');
 const cell=[...overview().querySelectorAll('.screening-stability-cell')].find(e=>!e.disabled&&e.title.includes('Higher in Case'));
 assert(cell.title.includes('testable repeats')&&cell.title.includes('p=')&&cell.title.includes('q='),'Each matrix cell exposes p, q, sample counts and resampling coverage');
 const csvLink=[...overview().querySelectorAll('a')].find(a=>a.textContent==='All rows CSV');
 const ranked=await fetch(csvLink.href).then(r=>r.text());
 assert(ranked.includes('supported_zones')&&ranked.includes('median_recovery')&&ranked.includes('nearby_recovery')&&ranked.trim().split('\n').length===21,'Ranked export contains every feature-zone-comparison with exact and nearby recovery');
 const svgLink=[...overview().querySelectorAll('a')].find(a=>a.textContent==='SVG');
 const exported=await fetch(svgLink.href).then(r=>r.text());
 assert(exported.includes('<svg')&&exported.includes('Feature stability across distances'),'Current matrix page has a vector export');
 const runBefore=by('Screening saved run').value;cell.click();
 await until(()=>by('Screening result comparison')?.value===cell.dataset.screeningComparison&&document.querySelector('[data-figure="screening-proportion"] [aria-label="Screening interactive finding"]')?.selectedOptions[0].textContent.endsWith(cell.dataset.screeningZone==='0'?'0–50 µm':'50–100 µm'));
 assert(document.querySelector('[data-figure="screening-proportion"] [aria-label="Screening interactive finding"]').selectedOptions[0].textContent.startsWith(JSON.parse(cell.dataset.screeningFeature)[1]+' ·'),'Clicking a matrix cell opens its exact comparison, feature and distance');
 assert(by('Screening saved run').value===runBefore,'Opening a matrix cell keeps its saved run');
 cell.click();choose('Screening result feature type','');await delay(750);
 assert(by('Screening result feature type').value==='','A newer feature selection wins over an earlier matrix-cell response');
 const family=overview().querySelector('[aria-label="Stability feature type"]');family.value='density';family.dispatchEvent(new Event('change'));
 await until(()=>overview().textContent.includes('No features match these filters.'));
 assert(!overview().querySelector('.screening-stability-cell'),'Empty family filter does not retain previous findings');
 family.value='proportion';family.dispatchEvent(new Event('change'));
 await until(()=>overview().querySelector('.screening-stability-cell'));
 assert(overview().querySelector('.screening-stability-scroll').getBoundingClientRect().width<=overview().getBoundingClientRect().width+2,'Wide matrix scrolls within the dock');
 overview().querySelector('[aria-label="Expand stability matrix"]').click();
 assert(document.querySelector('.studio-figure-large [data-figure="screening-stability"]'),'Large view uses the existing Studio overlay');
 document.querySelector('.studio-figure-large').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
 assert(!document.querySelector('.studio-figure-large')&&overview().isConnected,'Escape restores the same interactive matrix to its pane');
 overview().querySelector('[aria-label="Expand stability matrix"]').click();
"""

DRIVER=COMPARISONS.replace("csv.trim().split('\\n').length===11", "csv.trim().split('\\n').length===21")
DRIVER=DRIVER.replace("more.open=false;by('Screening result comparison').scrollIntoView({block:'center'});",CHECKS)


def fixture(root):
    comparison_fixture(root)
    import pandas as pd
    from spacemap import dataio
    from spacemap_gui import persistence
    from spacemap_gui.studio.settings import read_config
    frame=pd.read_csv(root/'cells.csv')
    frame['distance']=[25 if i%2==0 else 75 for i in range(len(frame))]
    frame.to_csv(root/'cells.csv',index=False)
    dataio.ensure_fast_copy(root/'cells.csv',force=True)
    config=read_config(root)
    config['tab_configurations']['tab8_analysis']['studio_screening'].update(
        resampling='bootstrap',resamples=40,zone_source='distance',distance_col='distance',distance_unit='um',zone_mode='rings',widths=[50,100])
    persistence.save_project_config(str(root),config)


if __name__=='__main__':
    import os
    main(DRIVER,studio=True,setup=fixture,timeout=300,capture=os.environ.get('SPACEMAP_TEST_CAPTURE'))
