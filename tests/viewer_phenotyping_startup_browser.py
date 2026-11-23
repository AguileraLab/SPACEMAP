"""Slow summaries cannot hold the controls or overwrite unsaved folder input."""
from viewer_threshold_browser import main

def setup(root):
    import shutil
    import time
    images=root/'outputs/rois';images.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'roi.ome.tiff',images/'roi.ome.tiff')
    from spacemap_gui.studio import phenotyping_inputs
    phenotyping_inputs.save_folders(root, {'images': str(images)})
    original=phenotyping_inputs.plan
    def delayed(*args,**kwargs):
        if kwargs.get('summaries',True):
            time.sleep(8)
        return original(*args,**kwargs)
    phenotyping_inputs.plan=delayed

DRIVER=r'''<script>
const pause=ms=>new Promise(r=>setTimeout(r,ms)),until=async f=>{for(let i=0;i<1000;i++){if(f())return;await pause(40);}throw Error('Timed out '+f);};
(async()=>{try{
 await until(()=>document.querySelector('[aria-label="Phenotyping images folder"]'));
 const input=document.querySelector('[aria-label="Phenotyping images folder"]');
 if(!document.querySelector('#studio-tools').textContent.includes('Counting…'))throw Error('Controls waited for slow summaries');
 input.focus();input.value='unsaved folder draft';
 await until(()=>document.querySelector('#studio-tools').textContent.includes('40 cells in table'));
 if(input!==document.querySelector('[aria-label="Phenotyping images folder"]')||input.value!=='unsaved folder draft')throw Error('Background summary overwrote the editor');
 const choices=document.querySelector('[aria-label="Colour cells by"]');
 if(![...choices.options].some(o=>o.value==='phenotype_ml_consensus'))throw Error('Schema choices missing');
 fetch('/result?ok='+encodeURIComponent(JSON.stringify({checks:3})));
}catch(e){fetch('/result?error='+encodeURIComponent(String(e)+'\n'+e.stack));}})();
</script>'''
if __name__=='__main__':
    main(driver=DRIVER,studio=True,setup=setup,timeout=100,screenshot=False)
